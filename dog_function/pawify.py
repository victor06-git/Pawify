"""Pawify — Go2 rescue operator console.

One window: live camera feed, movement + trick controls, and a push-to-talk /
listen audio link to whoever is near the robot. Built for a first-responder
operator triaging an injured person the robot has located.

SAFETY NOTE: this is a first version of a rescue tool, not a certified
medical device. Verify every control — especially Talk/Listen audio and the
video feed's latency — against the real robot before ever relying on it in
an actual emergency. The "Listen" feature in particular uses a WebRTC audio
path this SDK doesn't wrap or document as thoroughly as video; treat it as
best-effort until you've confirmed it works on your unit.

Reuses direct_go2_move.py for the connection, movement ramping, and trick
tables — run that script's --interactive mode if you just need a terminal,
no camera/audio.
"""

import argparse
import math
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import customtkinter as ctk
import numpy as np
import sounddevice as sd
from PIL import Image as PILImage, ImageDraw as PILImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import direct_go2_move as dg2  # noqa: E402  (needs the sys.path insert above)

APP_TITLE = "🐾 Pawify — Go2 Rescue Console"
VIDEO_W, VIDEO_H = 720, 405
TALK_SAMPLE_RATE = 44100
TALK_CHANNELS = 1
MAX_TALK_SECONDS = 10  # keep push-to-talk clips short — upload is slow (chunked, ~0.1s/4KB)

# Navigate mode: the Go2 only accepts one WebRTC peer at a time (Unitree
# firmware rejects a second connection attempt outright), so Pawify's own
# teleop connection and DimOS's relocalization/navigation blueprint can
# never both hold the robot at once. Navigate mode closes Pawify's own
# connection, runs the blueprint as a subprocess for as long as the operator
# wants map/nav, then closes it and reconnects teleop.
NAV_BLUEPRINT = "unitree-go2-relocalization"
NAV_READY_TIMEOUT_S = 90.0  # blueprint spin-up (11 worker processes + WebRTC) is slow

# Quick-trick launcher (paw button menu) + trimmed trick panel both draw from
# this one safe list — dances/flips/recovery-stand are deliberately excluded
# (see README): (button label, SPORT_COMMANDS name).
SAFE_TRICKS: list[tuple[str, str]] = [
    ("Hello", "Hello"),
    ("Stretch", "Stretch"),
    ("Sit", "Sit"),
    ("Rise", "RiseSit"),
    ("Lie Down", "StandDown"),
    ("Heart 💜", "FingerHeart"),
    ("Content", "Content"),
    ("Jump", "FrontJump"),
]

_MOVE_BUTTON_KEYS = {
    "forward": "w",
    "back": "s",
    "strafe_left": "q",
    "strafe_right": "e",
    "turn_left": "a",
    "turn_right": "d",
}


def _yaw_from_quaternion(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class TalkRecorder:
    """Records the operator's mic to memory while active; saves a WAV on stop."""

    def __init__(self, samplerate: int = TALK_SAMPLE_RATE, channels: int = TALK_CHANNELS) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self._frames: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None

    def start(self) -> None:
        self._frames = []

        def callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
            self._frames.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.samplerate, channels=self.channels, dtype="int16", callback=callback
        )
        self._stream.start()

    def stop_and_save(self, path: Path) -> Path | None:
        if self._stream is None:
            return None
        self._stream.stop()
        self._stream.close()
        self._stream = None
        if not self._frames:
            return None
        import wave

        data = np.concatenate(self._frames, axis=0)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # int16
            wf.setframerate(self.samplerate)
            wf.writeframes(data.tobytes())
        return path


def _audio_frame_to_mono_int16(frame) -> np.ndarray:
    """Convert a PyAV/aiortc AudioFrame to a flat mono int16 PCM array.

    Uses the frame's own format/layout metadata (planar vs packed, channel
    count, sample format) rather than guessing from array shape — aiortc
    audio is typically s16 at 48kHz, but this handles float formats too in
    case the decoder produces flt/fltp frames.
    """
    arr = frame.to_ndarray()
    channels = len(frame.layout.channels) if frame.layout and frame.layout.channels else 1
    is_planar = bool(getattr(frame.format, "is_planar", False))

    if is_planar and arr.ndim > 1 and arr.shape[0] == channels:
        mono = arr.astype(np.float64).mean(axis=0)
    else:
        flat = arr.reshape(-1).astype(np.float64)
        mono = flat.reshape(-1, channels).mean(axis=1) if channels > 1 else flat

    format_name = getattr(frame.format, "name", "s16")
    if format_name.startswith("flt") or format_name.startswith("dbl"):
        mono = mono * 32767.0  # float formats are typically normalized to [-1.0, 1.0]

    return np.clip(mono, -32768, 32767).astype(np.int16)


class RobotListener:
    """Plays live audio received from the robot's mic through the operator's
    speakers, via the WebRTC audio track's receive callback.

    Best-effort: the frame format isn't documented by this SDK, so the first
    decode failure is logged once and the listener keeps trying rather than
    spamming or crashing. Verify this actually works on your robot/firmware
    before relying on it operationally.
    """

    def __init__(self, conn, log) -> None:
        self._conn = conn
        self._log = log
        self._queue: queue.Queue = queue.Queue(maxsize=100)
        self._active = False
        self._warned = False
        self._player_thread: threading.Thread | None = None
        self.volume = 1.0  # 0.0 - 2.0, set by the UI's volume slider

    def start(self) -> None:
        if self._active:
            return
        self._active = True
        raw = self._conn.conn  # unitree_webrtc_connect's low-level driver

        async def on_frame(frame) -> None:
            try:
                pcm = _audio_frame_to_mono_int16(frame)
                rate = getattr(frame, "sample_rate", 48000)
                self._queue.put_nowait((pcm, rate))
            except queue.Full:
                pass
            except Exception as e:
                if not self._warned:
                    self._warned = True
                    self._log(
                        f"Listen: audio decode issue ({e}) — feature may not be fully "
                        "supported on this robot/firmware, verify before relying on it."
                    )

        def register() -> None:
            raw.audio.add_track_callback(on_frame)
            raw.audio.switchAudioChannel(True)

        self._conn.loop.call_soon_threadsafe(register)
        self._log("Listen: enabled — playing robot mic audio through your speakers.")

        self._player_thread = threading.Thread(target=self._play_loop, daemon=True)
        self._player_thread.start()

    def _play_loop(self) -> None:
        stream: sd.OutputStream | None = None
        stream_rate: int | None = None
        while self._active:
            try:
                pcm, rate = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if stream is None or rate != stream_rate:
                    if stream is not None:
                        stream.stop()
                        stream.close()
                    stream = sd.OutputStream(samplerate=rate, channels=1, dtype="int16")
                    stream.start()
                    stream_rate = rate
                if self.volume != 1.0:
                    pcm = np.clip(pcm.astype(np.float32) * self.volume, -32768, 32767).astype(
                        np.int16
                    )
                stream.write(pcm)
            except Exception:
                pass
        if stream is not None:
            stream.stop()
            stream.close()

    def stop(self) -> None:
        if not self._active:
            return
        self._active = False
        try:
            self._conn.loop.call_soon_threadsafe(
                lambda: self._conn.conn.audio.switchAudioChannel(False)
            )
        except Exception:
            pass
        self._log("Listen: disabled.")


class VideoFeed:
    """Subscribes to the robot's camera stream and keeps the latest frame
    (as an OpenCV BGR ndarray) available for the UI's polling loop.

    Runs on the connection's background asyncio thread — never touches
    Tkinter directly, just stores the latest frame behind a lock.
    """

    def __init__(self, conn) -> None:
        self._conn = conn
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._image = None  # last dimos Image msg, for detectors that need it
        self._last_ts = 0.0
        self._fps = 0.0
        self._disposable = None

    def start(self) -> None:
        self._disposable = self._conn.get_video_stream().subscribe(
            on_next=self._on_frame,
            on_error=lambda e: None,
        )

    def _on_frame(self, image) -> None:
        now = time.monotonic()
        with self._lock:
            if self._last_ts:
                dt = now - self._last_ts
                if dt > 0:
                    self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
            self._last_ts = now
            self._frame = image.to_opencv()
            self._image = image

    def latest(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            return (None if self._frame is None else self._frame.copy(), self._fps)

    def latest_image(self):
        """The last dimos Image message (not just its ndarray) — needed by
        detectors that call .process_image(Image)."""
        with self._lock:
            return self._image

    def stop(self) -> None:
        if self._disposable is not None:
            try:
                self._disposable.dispose()
            except Exception:
                pass


class ObstacleRadar:
    """Nearest-obstacle distance from the robot's own lidar, in a forward
    cone roughly matching the robot's body width. This is a supplementary
    readout for the operator — the robot's onboard obstacle avoidance
    (enabled via --obstacle-avoidance) is the actual safety mechanism, this
    just surfaces the same kind of number in the UI.
    """

    FORWARD_HALF_WIDTH_M = 0.35
    MIN_HEIGHT_M = 0.05
    MAX_HEIGHT_M = 0.8

    def __init__(self, conn) -> None:
        self._conn = conn
        self._lock = threading.Lock()
        self._distance: float | None = None
        self._disposable = None

    def start(self) -> None:
        try:
            self._disposable = self._conn.lidar_stream().subscribe(
                on_next=self._on_cloud, on_error=lambda e: None
            )
        except Exception:
            self._disposable = None

    def _on_cloud(self, cloud) -> None:
        try:
            pts = cloud.points_f32()
            if pts.shape[0] == 0:
                return
            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
            mask = (
                (x > 0.05)
                & (np.abs(y) < self.FORWARD_HALF_WIDTH_M)
                & (z > self.MIN_HEIGHT_M)
                & (z < self.MAX_HEIGHT_M)
            )
            forward = x[mask]
            with self._lock:
                self._distance = float(forward.min()) if forward.size > 0 else None
        except Exception:
            pass

    def latest(self) -> float | None:
        with self._lock:
            return self._distance

    def stop(self) -> None:
        if self._disposable is not None:
            try:
                self._disposable.dispose()
            except Exception:
                pass


class NavigateBlueprint:
    """Runs the DimOS relocalization/navigation blueprint as a child process
    and reports when it's up (by polling for the blueprint's own topics —
    see PawifyApp's nav-mode subscriber) and when it exits.

    Deliberately dumb about *what* the blueprint publishes — that's the
    subscriber's job (PawifyApp reads costmap/pose off the pub/sub bus
    directly once this is running). This class only owns the OS process.
    """

    POLL_INTERVAL_S = 1.0

    def __init__(
        self,
        dimos_python: Path,
        dimos_dir: Path,
        blueprint: str,
        ready_timeout: float = NAV_READY_TIMEOUT_S,
    ) -> None:
        self._dimos_python = dimos_python
        self._dimos_dir = dimos_dir
        self._blueprint = blueprint
        self._ready_timeout = ready_timeout
        self._proc: subprocess.Popen | None = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _find_dimos_bin(self) -> Path:
        candidate = self._dimos_python.parent / "dimos"
        if candidate.exists():
            return candidate
        found = shutil.which("dimos")
        if found:
            return Path(found)
        raise FileNotFoundError(
            f"No 'dimos' CLI found next to {self._dimos_python} or on PATH — "
            "is dimos_dir's venv the one Pawify itself is running under?"
        )

    def start(self, map_file: Path, is_ready, on_ready, on_exit, log=print) -> None:
        """Non-blocking. `is_ready()` is polled (from a background watcher
        thread) until it returns True, at which point `on_ready()` fires
        exactly once, or until the process exits or `ready_timeout` elapses
        (logged but not fatal — the operator can keep waiting or give up).
        `on_exit(code)` fires exactly once when the process dies for any
        reason (clean stop() or a crash)."""
        dimos_bin = self._find_dimos_bin()
        cmd = [
            str(dimos_bin), "run", self._blueprint,
            "-o", f"relocalizationmodule.map_file={map_file}",
        ]
        log(f"Navigate: launching {' '.join(cmd)}")
        self._proc = subprocess.Popen(
            cmd,
            cwd=str(self._dimos_dir),
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        proc = self._proc

        def _stream_output() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                log(f"  [dimos] {line.rstrip()}")

        def _watch() -> None:
            became_ready = False
            deadline = time.monotonic() + self._ready_timeout
            while True:
                code = proc.poll()
                if code is not None:
                    on_exit(code)
                    return
                if not became_ready:
                    if is_ready():
                        became_ready = True
                        log("Navigate: blueprint is up.")
                        on_ready()
                    elif time.monotonic() > deadline:
                        log(
                            f"Navigate: still not seeing blueprint data after "
                            f"{self._ready_timeout:.0f}s — check the [dimos] log lines above "
                            "for errors (bad map path, robot unreachable, etc). Still waiting."
                        )
                        deadline = time.monotonic() + self._ready_timeout
                time.sleep(self.POLL_INTERVAL_S)

        threading.Thread(target=_stream_output, daemon=True).start()
        threading.Thread(target=_watch, daemon=True).start()

    def stop(self, log=print, timeout: float = 15.0) -> None:
        """Blocking. SIGTERM is DimOS's own documented clean-shutdown path
        (dimos/core/daemon.py: install_signal_handlers -> coordinator.stop()
        + registry cleanup + sys.exit(0)); SIGKILL is only a last resort."""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        log("Navigate: stopping the DimOS blueprint...")
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            log("Navigate: blueprint didn't stop cleanly, killing it.")
            proc.kill()
            proc.wait(timeout=5.0)


class NavMapView(ctk.CTkFrame):
    """Renders the running Navigate-mode blueprint's live costmap + robot
    pose inside Pawify's own window — no DimOS web UI involved, this reads
    the blueprint's pub/sub topics directly (see PawifyApp's nav-mode
    subscriber) and draws them here. A click is converted from image pixels
    to a world-frame (x, y) and handed to `on_click_goal`.

    Occupancy values follow the standard OccupancyGrid convention (see
    dimos/mapping/pointclouds/occupancy.py's cost table): 0 = free,
    100 = impassable, -1 = unknown.
    """

    UNKNOWN_RGB = (60, 60, 60)
    FREE_RGB = (25, 130, 60)
    WALL_RGB = (170, 30, 30)
    ROBOT_RGB = (0, 170, 255)
    GOAL_RGB = (241, 196, 15)

    def __init__(self, parent, *, width: int, height: int, on_click_goal, **kwargs) -> None:
        super().__init__(parent, fg_color="transparent", **kwargs)
        self._width = width
        self._height = height
        self._on_click_goal = on_click_goal

        self._image_label = ctk.CTkLabel(self, text="Enter Navigate Mode to see the map.")
        self._image_label.pack(padx=8, pady=(8, 4))
        self._image_label.bind("<Button-1>", self._on_click)

        self._pose_label = ctk.CTkLabel(
            self, text="pose: --", font=ctk.CTkFont(size=12), text_color="#888888"
        )
        self._pose_label.pack(fill="x", padx=8, pady=(0, 8))

        self._grid_info: tuple[float, float, float, int, int] | None = None
        self._base_rgb: np.ndarray | None = None  # last colorized costmap, HxWx3 uint8
        self._robot_xy: tuple[float, float] | None = None
        self._robot_yaw: float = 0.0
        self._goal_xy: tuple[float, float] | None = None
        self._scale = 1  # rendered-image px per grid cell

    def update_costmap(
        self, resolution: float, origin_x: float, origin_y: float,
        width: int, height: int, data: np.ndarray,
    ) -> None:
        """`data` is a (height, width) array, row 0 = the row at `origin_y`,
        values -1 (unknown) or 0-100 (traversal cost, 100 = impassable)."""
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        rgb[:] = self.UNKNOWN_RGB
        known = data >= 0
        t = np.clip(data[known].astype(np.float32) / 100.0, 0.0, 1.0)
        free_arr = np.array(self.FREE_RGB, dtype=np.float32)
        wall_arr = np.array(self.WALL_RGB, dtype=np.float32)
        rgb[known] = (free_arr[None, :] * (1 - t[:, None]) + wall_arr[None, :] * t[:, None]).astype(
            np.uint8
        )
        # Grid row 0 is the lowest-y (origin) row; image row 0 is the top,
        # so flip vertically to draw with "up" = +y like the physical map.
        self._base_rgb = np.flipud(rgb)
        self._grid_info = (resolution, origin_x, origin_y, width, height)
        self._scale = max(1, min(self._width // max(width, 1), self._height // max(height, 1)))
        self._redraw()

    def update_pose(self, x: float, y: float, yaw: float) -> None:
        self._robot_xy = (x, y)
        self._robot_yaw = yaw
        self._pose_label.configure(text=f"pose: x={x:.2f}m y={y:.2f}m yaw={math.degrees(yaw):.0f}°")
        self._redraw()

    def update_goal(self, x: float, y: float) -> None:
        self._goal_xy = (x, y)
        self._redraw()

    def clear(self) -> None:
        self._grid_info = None
        self._base_rgb = None
        self._robot_xy = None
        self._goal_xy = None
        self._image_label.configure(image=None, text="Enter Navigate Mode to see the map.")
        self._pose_label.configure(text="pose: --")

    def _world_to_px(self, x: float, y: float) -> tuple[int, int] | None:
        if self._grid_info is None:
            return None
        resolution, origin_x, origin_y, width, height = self._grid_info
        col = int((x - origin_x) / resolution)
        row = int((y - origin_y) / resolution)
        if not (0 <= col < width and 0 <= row < height):
            return None
        return col * self._scale, (height - 1 - row) * self._scale

    def _redraw(self) -> None:
        if self._base_rgb is None:
            return
        img = PILImage.fromarray(self._base_rgb, mode="RGB")
        if self._scale != 1:
            img = img.resize((img.width * self._scale, img.height * self._scale), PILImage.NEAREST)
        draw = PILImageDraw.Draw(img)
        if self._robot_xy is not None:
            p = self._world_to_px(*self._robot_xy)
            if p is not None:
                r = max(3, self._scale)
                draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=self.ROBOT_RGB)
                hx = p[0] + int(r * 2 * math.cos(-self._robot_yaw))
                hy = p[1] + int(r * 2 * math.sin(-self._robot_yaw))
                draw.line([p[0], p[1], hx, hy], fill=self.ROBOT_RGB, width=max(2, self._scale // 2))
        if self._goal_xy is not None:
            p = self._world_to_px(*self._goal_xy)
            if p is not None:
                r = max(4, self._scale)
                draw.line([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=self.GOAL_RGB, width=2)
                draw.line([p[0] - r, p[1] + r, p[0] + r, p[1] - r], fill=self.GOAL_RGB, width=2)
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(img.width, img.height))
        self._image_label.configure(image=ctk_img, text="")
        self._image_label.image = ctk_img  # keep a reference, avoid GC

    def _on_click(self, event) -> None:
        if self._grid_info is None:
            return
        resolution, origin_x, origin_y, width, height = self._grid_info
        col = event.x // self._scale
        row = height - 1 - (event.y // self._scale)
        if not (0 <= col < width and 0 <= row < height):
            return
        x = origin_x + (col + 0.5) * resolution
        y = origin_y + (row + 0.5) * resolution
        self._on_click_goal(x, y)


class PawifyApp(ctk.CTk):
    def __init__(self, conn, args: argparse.Namespace) -> None:
        super().__init__()
        self._log_queue: queue.Queue = queue.Queue()
        self._conn = conn
        self._args = args
        self._state = dg2.TeleopState(
            speed=args.speed, turn_speed=args.turn_speed, max_speed=args.max_speed,
            accel=args.accel, turn_accel=args.turn_accel,
        )
        self._held: dict[str, bool] = {}
        self._video = VideoFeed(conn)
        self._recorder = TalkRecorder()
        self._listener = RobotListener(conn, self._log)
        self._obstacle_radar = ObstacleRadar(conn)
        self._talking = False
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="pawify_"))

        # Navigate mode: closes the connection above, hands the robot's one
        # WebRTC slot to a DimOS blueprint subprocess, and renders that
        # blueprint's costmap/pose (read straight off its pub/sub bus — see
        # NavMapView) inside this same window. See NAV_BLUEPRINT's comment.
        self._nav_state = "idle"
        self._nav_stop_requested = False
        self._nav_got_costmap = False
        self._nav_blueprint = NavigateBlueprint(
            dimos_python=Path(sys.executable), dimos_dir=args.dimos_dir, blueprint=args.nav_blueprint,
        )
        self._ui_action_queue: queue.Queue = queue.Queue()

        ctk.set_appearance_mode("dark")
        self.title(APP_TITLE)
        self.geometry("1100x760")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._bind_keys()

        self._video.start()
        self._obstacle_radar.start()
        self._log("Pawify ready. Robot connected and standing by.")
        self.after(33, self._video_tick)
        self.after(50, self._movement_tick)
        self.after(100, self._drain_log_queue)
        self.after(100, self._drain_ui_actions)

    # ─── UI construction ────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 0))
        ctk.CTkLabel(header, text=APP_TITLE, font=ctk.CTkFont(size=20, weight="bold")).pack(
            side="left"
        )
        self._status_label = ctk.CTkLabel(
            header, text="● connected", text_color="#2ecc71", font=ctk.CTkFont(size=14)
        )
        self._status_label.pack(side="right")
        self._build_paw_button(header)

        # Video feed. The label is packed at its NATIVE image size (no
        # fill/expand stretch) — a holdover from when click coordinates
        # needed to map 1:1 to image pixels for click-to-follow; harmless
        # to keep now, just means the feed doesn't stretch to fill.
        view_frame = ctk.CTkFrame(self)
        view_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        view_toggle_row = ctk.CTkFrame(view_frame, fg_color="transparent")
        view_toggle_row.pack(fill="x", padx=8, pady=(8, 0))
        self._camera_view_btn = ctk.CTkButton(
            view_toggle_row, text="📷 Camera", width=110, command=lambda: self._show_view("camera")
        )
        self._camera_view_btn.pack(side="left", padx=(0, 6))
        self._map_view_btn = ctk.CTkButton(
            view_toggle_row, text="🗺 Map", width=110, fg_color="transparent", border_width=1,
            command=lambda: self._show_view("map"),
        )
        self._map_view_btn.pack(side="left")

        self._camera_view = ctk.CTkFrame(view_frame, fg_color="transparent")
        self._video_label = ctk.CTkLabel(self._camera_view, text="waiting for camera...")
        self._video_label.pack(padx=8, pady=(8, 4))
        self._obstacle_label = ctk.CTkLabel(
            self._camera_view, text="obstacle: --", font=ctk.CTkFont(size=13, weight="bold")
        )
        self._obstacle_label.pack(fill="x", padx=8, pady=(0, 8))

        self._map_view = NavMapView(
            view_frame, width=VIDEO_W, height=VIDEO_H, on_click_goal=self._on_map_click_goal
        )

        self._view_frame = view_frame
        self._show_view("camera")

        # Sidebar
        sidebar = ctk.CTkScrollableFrame(self, label_text="Controls")
        sidebar.grid(row=1, column=1, sticky="nsew", padx=(0, 10), pady=10)
        self._build_movement_panel(sidebar)
        self._build_trick_panel(sidebar)
        self._build_ai_panel(sidebar)
        self._build_talk_panel(sidebar)
        self._build_nav_panel(sidebar)

        # Log
        log_frame = ctk.CTkFrame(self)
        log_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        ctk.CTkLabel(log_frame, text="Log", anchor="w").pack(fill="x", padx=8, pady=(6, 0))
        self._log_box = ctk.CTkTextbox(log_frame, height=140)
        self._log_box.pack(fill="x", expand=True, padx=8, pady=8)
        self._log_box.configure(state="disabled")

    @staticmethod
    def _make_paw_icon(size: int = 30, color: str = "#f5f5f5") -> "PILImage.Image":
        """Draws a paw-print silhouette (one big pad + four toes) with PIL —
        rendered at 4x then downsampled for smooth anti-aliased edges, since
        Tkinter/customtkinter has no native SVG or CSS clip-path support."""
        scale = 4
        s = size * scale
        img = PILImage.new("RGBA", (s, s), (0, 0, 0, 0))
        draw = PILImageDraw.Draw(img)
        draw.ellipse([s * 0.22, s * 0.40, s * 0.78, s * 0.92], fill=color)
        toe_w, toe_h = s * 0.20, s * 0.26
        for cx, cy in ((0.20, 0.22), (0.40, 0.10), (0.60, 0.10), (0.80, 0.22)):
            draw.ellipse(
                [s * cx - toe_w / 2, s * cy - toe_h / 2, s * cx + toe_w / 2, s * cy + toe_h / 2],
                fill=color,
            )
        return img.resize((size, size), PILImage.LANCZOS)

    def _build_paw_button(self, parent) -> None:
        """A distinct paw-shaped button that pops up a quick-trick launcher
        menu — fast access to the same safe trick set as the Tricks panel,
        without scrolling the sidebar."""
        paw_img = self._make_paw_icon()
        self._paw_ctk_image = ctk.CTkImage(light_image=paw_img, dark_image=paw_img, size=(30, 30))
        paw_btn = ctk.CTkButton(
            parent, image=self._paw_ctk_image, text="", width=44, height=36,
            fg_color="#8e44ad", hover_color="#6c3483", command=self._show_paw_menu,
        )
        paw_btn.pack(side="right", padx=(0, 12))
        self._paw_button = paw_btn

    def _show_paw_menu(self) -> None:
        menu = ctk.CTkToplevel(self)
        menu.overrideredirect(True)
        menu.attributes("-topmost", True)
        x = self._paw_button.winfo_rootx()
        y = self._paw_button.winfo_rooty() + self._paw_button.winfo_height() + 4
        menu.geometry(f"+{x}+{y}")

        frame = ctk.CTkFrame(menu, border_width=1)
        frame.pack(fill="both", expand=True)
        ctk.CTkLabel(
            frame, text="🐾 Quick Tricks", font=ctk.CTkFont(size=12, weight="bold")
        ).pack(fill="x", padx=10, pady=(8, 4))

        def fire_and_close(name: str) -> None:
            menu.destroy()
            self._fire_trick(name)

        for label, name in SAFE_TRICKS:
            ctk.CTkButton(
                frame, text=label, width=160, anchor="w",
                command=lambda n=name: fire_and_close(n),
            ).pack(fill="x", padx=10, pady=2)

        ctk.CTkButton(
            frame, text="close", width=160, fg_color="transparent",
            border_width=1, command=menu.destroy,
        ).pack(fill="x", padx=10, pady=(2, 8))

        # Close if it loses focus (click elsewhere) rather than lingering.
        menu.bind("<FocusOut>", lambda _e: menu.destroy())
        menu.after(10, menu.focus_force)

    def _build_movement_panel(self, parent) -> None:
        frame = ctk.CTkFrame(parent)
        self._movement_frame = frame
        frame.pack(fill="x", padx=6, pady=6)
        ctk.CTkLabel(frame, text="Movement (click & hold, or WASD/QE)", anchor="w").pack(
            fill="x", padx=8, pady=(8, 4)
        )

        pad = ctk.CTkFrame(frame, fg_color="transparent")
        pad.pack(padx=8, pady=4)
        for c in range(3):
            pad.grid_columnconfigure(c, weight=1)

        self._hold_button(pad, "↖ strafe L", "strafe_left", 0, 0)
        self._hold_button(pad, "↑ forward", "forward", 0, 1)
        self._hold_button(pad, "↗ strafe R", "strafe_right", 0, 2)
        self._hold_button(pad, "↺ turn L", "turn_left", 1, 0)
        stop_btn = ctk.CTkButton(
            pad, text="STOP", fg_color="#c0392b", hover_color="#922b21",
            font=ctk.CTkFont(weight="bold"), command=self._emergency_stop,
        )
        stop_btn.grid(row=1, column=1, padx=4, pady=4, sticky="nsew")
        self._hold_button(pad, "↻ turn R", "turn_right", 1, 2)
        self._hold_button(pad, "↓ back", "back", 2, 1)

        speed_row = ctk.CTkFrame(frame, fg_color="transparent")
        speed_row.pack(fill="x", padx=8, pady=(8, 8))
        ctk.CTkLabel(speed_row, text="Speed").pack(side="left")
        self._speed_value_label = ctk.CTkLabel(speed_row, text=f"{self._state.speed:.2f} m/s")
        self._speed_value_label.pack(side="right")
        self._speed_slider = ctk.CTkSlider(
            frame, from_=0.0, to=self._state.max_speed, command=self._on_speed_change
        )
        self._speed_slider.set(self._state.speed)
        self._speed_slider.pack(fill="x", padx=8, pady=(0, 10))

        standup_btn = ctk.CTkButton(frame, text="Stand Up (full sequence)", command=self._standup)
        standup_btn.pack(fill="x", padx=8, pady=(0, 8))

    def _hold_button(self, parent, label: str, key: str, row: int, col: int) -> None:
        btn = ctk.CTkButton(parent, text=label)
        btn.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
        btn.bind("<ButtonPress-1>", lambda _e, k=key: self._held.__setitem__(k, True))
        btn.bind("<ButtonRelease-1>", lambda _e, k=key: self._held.pop(k, None))

    def _build_trick_panel(self, parent) -> None:
        frame = ctk.CTkFrame(parent)
        self._trick_frame = frame
        frame.pack(fill="x", padx=6, pady=6)
        ctk.CTkLabel(frame, text="Tricks (safe poses only — no dances/flips)", anchor="w").pack(
            fill="x", padx=8, pady=(8, 4)
        )

        grid = ctk.CTkFrame(frame, fg_color="transparent")
        grid.pack(fill="x", padx=8, pady=(0, 8))
        for c in range(2):
            grid.grid_columnconfigure(c, weight=1)

        for i, (label, name) in enumerate(SAFE_TRICKS):
            btn = ctk.CTkButton(
                grid, text=label, fg_color="#2980b9", command=lambda n=name: self._fire_trick(n)
            )
            btn.grid(row=i // 2, column=i % 2, padx=4, pady=4, sticky="nsew")

    def _build_ai_panel(self, parent) -> None:
        frame = ctk.CTkFrame(parent)
        self._ai_frame = frame
        frame.pack(fill="x", padx=6, pady=6)
        ctk.CTkLabel(frame, text="AI Command (natural language)", anchor="w").pack(
            fill="x", padx=8, pady=(8, 4)
        )
        ctk.CTkLabel(
            frame,
            text="Claude (via OpenRouter) can move/turn, do a trick, or speak — "
            "not vision-grounded, so it can't pick navigate/follow targets itself.",
            anchor="w", justify="left", font=ctk.CTkFont(size=11), text_color="#888888",
            wraplength=260,
        ).pack(fill="x", padx=8, pady=(0, 6))

        row = ctk.CTkFrame(frame, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=(0, 10))
        self._ai_entry = ctk.CTkEntry(row, placeholder_text="e.g. wave hello, then back up a bit")
        self._ai_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self._ai_entry.bind("<Return>", lambda _e: self._ai_send())
        ctk.CTkButton(row, text="Send", width=60, command=self._ai_send).pack(side="left")

    def _build_talk_panel(self, parent) -> None:
        frame = ctk.CTkFrame(parent)
        self._talk_frame = frame
        frame.pack(fill="x", padx=6, pady=6)
        ctk.CTkLabel(
            frame, text="Talk to the person near the robot", anchor="w"
        ).pack(fill="x", padx=8, pady=(8, 4))

        self._talk_btn = ctk.CTkButton(
            frame, text=f"🎙 HOLD TO TALK (max {MAX_TALK_SECONDS}s)", fg_color="#8e44ad",
            hover_color="#6c3483",
        )
        self._talk_btn.pack(fill="x", padx=8, pady=4)
        self._talk_btn.bind("<ButtonPress-1>", lambda _e: self._talk_start())
        self._talk_btn.bind("<ButtonRelease-1>", lambda _e: self._talk_stop_and_send())

        type_row = ctk.CTkFrame(frame, fg_color="transparent")
        type_row.pack(fill="x", padx=8, pady=(4, 4))
        self._say_entry = ctk.CTkEntry(type_row, placeholder_text="Type a message to speak...")
        self._say_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self._say_entry.bind("<Return>", lambda _e: self._say_typed_message())
        ctk.CTkButton(type_row, text="Send", width=60, command=self._say_typed_message).pack(
            side="left"
        )

        self._listen_switch = ctk.CTkSwitch(
            frame, text="🔊 Listen to robot's mic (best-effort)", command=self._toggle_listen
        )
        self._listen_switch.pack(fill="x", padx=8, pady=(4, 4))

        volume_row = ctk.CTkFrame(frame, fg_color="transparent")
        volume_row.pack(fill="x", padx=8, pady=(0, 10))
        ctk.CTkLabel(volume_row, text="🔉").pack(side="left")
        self._volume_slider = ctk.CTkSlider(
            volume_row, from_=0.0, to=2.0, command=self._on_volume_change
        )
        self._volume_slider.set(1.0)
        self._volume_slider.pack(side="left", fill="x", expand=True, padx=6)
        ctk.CTkLabel(volume_row, text="🔊").pack(side="left")

    def _build_nav_panel(self, parent) -> None:
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", padx=6, pady=6)
        ctk.CTkLabel(frame, text="Map / Navigate (DimOS relocalization)", anchor="w").pack(
            fill="x", padx=8, pady=(8, 4)
        )
        ctk.CTkLabel(
            frame,
            text="The Go2 only accepts one connection at a time, so this swaps the robot's "
            "link over to DimOS's mapping/navigation stack. Camera, movement, tricks, and "
            "talk here are unavailable until you exit Navigate mode.",
            anchor="w", justify="left", font=ctk.CTkFont(size=11), text_color="#888888",
            wraplength=260,
        ).pack(fill="x", padx=8, pady=(0, 6))

        self._map_file_entry = ctk.CTkEntry(frame, placeholder_text="premap name or path")
        self._map_file_entry.insert(0, self._args.map_file)
        self._map_file_entry.pack(fill="x", padx=8, pady=(0, 6))

        self._nav_button = ctk.CTkButton(
            frame, text="Enter Navigate Mode", fg_color="#16a085", hover_color="#117a65",
            command=self._toggle_navigate_mode,
        )
        self._nav_button.pack(fill="x", padx=8, pady=(0, 4))

        self._nav_status_label = ctk.CTkLabel(
            frame, text="Navigate: idle", anchor="w", font=ctk.CTkFont(size=11), text_color="#888888"
        )
        self._nav_status_label.pack(fill="x", padx=8, pady=(0, 8))

    # ─── keyboard bindings (same held-key movement, driven by real key events) ───

    def _bind_keys(self) -> None:
        for name, key in _MOVE_BUTTON_KEYS.items():
            self.bind(f"<KeyPress-{key}>", lambda _e, n=name: self._held.__setitem__(n, True))
            self.bind(f"<KeyRelease-{key}>", lambda _e, n=name: self._held.pop(n, None))
        self.bind("<space>", lambda _e: self._emergency_stop())

    # ─── movement ───────────────────────────────────────────────────

    def _on_speed_change(self, value: float) -> None:
        self._state.speed = float(value)
        self._speed_value_label.configure(text=f"{self._state.speed:.2f} m/s")

    def _movement_tick(self) -> None:
        if self._conn is not None:
            held_keys = {_MOVE_BUTTON_KEYS[n] for n in self._held if n in _MOVE_BUTTON_KEYS}
            twist = dg2._twist_from_held(held_keys, self._state)
            if twist is not None:
                try:
                    self._conn.move(twist, duration=0)
                except Exception as e:
                    self._log(f"Move failed: {e}")
        self.after(50, self._movement_tick)

    def _emergency_stop(self) -> None:
        self._held.clear()
        self._state.cur_x = self._state.cur_y = self._state.cur_yaw = 0.0
        # In Navigate mode self._conn is None (DimOS holds the robot's one
        # WebRTC connection instead) — nothing for this button to stop here;
        # exit Navigate mode or use DimOS's own stop path for that.
        if self._conn is None:
            self._log("STOP — no teleop connection right now (in Navigate mode?).")
            return
        try:
            self._conn.stop_movement()
        except Exception as e:
            self._log(f"Stop failed: {e}")
        self._log("STOP — movement zeroed.")

    def _standup(self) -> None:
        threading.Thread(
            target=lambda: dg2.prepare_locomotion(self._conn, self._args, log=self._log),
            daemon=True,
        ).start()

    # ─── tricks ─────────────────────────────────────────────────────

    def _fire_trick(self, name: str) -> None:
        # Don't fight a trick against still-held movement keys (e.g. WASD
        # held while clicking a trick button with the mouse).
        self._held.clear()
        self._state.cur_x = self._state.cur_y = self._state.cur_yaw = 0.0
        threading.Thread(
            target=lambda: dg2._fire_trick(self._conn, name, log=self._log), daemon=True
        ).start()

    # ─── talk / listen ──────────────────────────────────────────────

    def _talk_start(self) -> None:
        if self._talking:
            return
        self._talking = True
        self._talk_btn.configure(text="🔴 RECORDING... release to send")
        self._recorder.start()
        self.after(int(MAX_TALK_SECONDS * 1000), self._talk_auto_stop)

    def _talk_auto_stop(self) -> None:
        if self._talking:
            self._log(f"Talk: hit the {MAX_TALK_SECONDS}s max, sending automatically.")
            self._talk_stop_and_send()

    def _talk_stop_and_send(self) -> None:
        if not self._talking:
            return
        self._talking = False
        self._talk_btn.configure(text=f"🎙 HOLD TO TALK (max {MAX_TALK_SECONDS}s)")
        wav_path = self._tmp_dir / f"talk_{int(time.time())}.wav"
        saved = self._recorder.stop_and_save(wav_path)
        if saved is None:
            self._log("Talk: nothing recorded.")
            return
        threading.Thread(target=self._send_talk_clip, args=(saved,), daemon=True).start()

    def _send_talk_clip(self, wav_path: Path) -> None:
        self._log("Talk: sending to robot speaker (upload + playback wait — can take up to ~40s)...")
        if dg2._send_wav_to_robot(self._conn, wav_path, log=self._log):
            self._log("Talk: sent — robot should have broadcast it.")

    def _say_typed_message(self) -> None:
        text = self._say_entry.get().strip()
        if not text:
            return
        self._say_entry.delete(0, "end")
        threading.Thread(
            target=lambda: dg2._speak_text(self._conn, text, log=self._log), daemon=True
        ).start()

    def _ai_send(self) -> None:
        instruction = self._ai_entry.get().strip()
        if not instruction:
            return
        self._ai_entry.delete(0, "end")
        self._held.clear()
        api_key = getattr(self._args, "openrouter_key", None) or os.environ.get(
            "OPENROUTER_API_KEY"
        )
        threading.Thread(
            target=lambda: dg2.run_ai_command(self._conn, instruction, api_key, log=self._log),
            daemon=True,
        ).start()

    def _toggle_listen(self) -> None:
        if self._listen_switch.get():
            self._listener.start()
        else:
            self._listener.stop()

    def _on_volume_change(self, value: float) -> None:
        self._listener.volume = float(value)

    # ─── navigate mode (DimOS relocalization + navigation, rendered here) ─

    def _show_view(self, which: str) -> None:
        active_color = ("#3a7ebf", "#1f538d")
        for w in (self._camera_view, self._map_view):
            w.pack_forget()
        if which == "camera":
            self._camera_view.pack(padx=8, pady=(4, 8))
            self._camera_view_btn.configure(fg_color=active_color, border_width=0)
            self._map_view_btn.configure(fg_color="transparent", border_width=1)
        else:
            self._map_view.pack(padx=8, pady=(4, 8), fill="both", expand=True)
            self._map_view_btn.configure(fg_color=active_color, border_width=0)
            self._camera_view_btn.configure(fg_color="transparent", border_width=1)

    def _on_map_click_goal(self, x: float, y: float) -> None:
        if self._nav_state != "active":
            return
        self._log(f"Navigate: sending goal x={x:.2f}m y={y:.2f}m...")
        self._map_view.update_goal(x, y)
        threading.Thread(target=lambda: self._send_nav_goal(x, y), daemon=True).start()

    _NAV_BUTTON_STYLE = {
        "idle": ("Enter Navigate Mode", "normal", "#16a085", "#117a65"),
        "entering": ("Entering Navigate Mode...", "disabled", "#16a085", "#117a65"),
        "active": ("Exit Navigate Mode", "normal", "#c0392b", "#922b21"),
        "exiting": ("Exiting Navigate Mode...", "disabled", "#c0392b", "#922b21"),
        "reconnecting": ("Reconnecting teleop...", "disabled", "#16a085", "#117a65"),
        "reconnect_failed": ("Retry Reconnect", "normal", "#e67e22", "#ca6f1e"),
    }
    _NAV_STATUS_STYLE = {
        "idle": ("Navigate: idle", "#888888"),
        "entering": ("Navigate: closing teleop, launching DimOS...", "#f39c12"),
        "active": ("Navigate: active — click the map to send a goal", "#2ecc71"),
        "exiting": ("Navigate: stopping DimOS...", "#f39c12"),
        "reconnecting": ("Navigate: reconnecting teleop...", "#f39c12"),
        "reconnect_failed": ("Navigate: teleop reconnect failed — see log", "#e74c3c"),
    }

    def _apply_nav_state(self) -> None:
        text, state, fg, hover = self._NAV_BUTTON_STYLE[self._nav_state]
        self._nav_button.configure(text=text, state=state, fg_color=fg, hover_color=hover)
        status_text, color = self._NAV_STATUS_STYLE[self._nav_state]
        self._nav_status_label.configure(text=status_text, text_color=color)
        self._set_live_controls_enabled(self._nav_state == "idle")

    def _toggle_navigate_mode(self) -> None:
        if self._nav_state == "idle":
            self._enter_navigate_mode()
        elif self._nav_state == "active":
            self._exit_navigate_mode()
        elif self._nav_state == "reconnect_failed":
            self._begin_teleop_reconnect()

    def _enter_navigate_mode(self) -> None:
        if self._nav_state != "idle":
            return
        raw_map = self._map_file_entry.get().strip() or self._args.map_file
        try:
            map_path = dg2._resolve_map_file(raw_map, self._args.dimos_dir)
        except FileNotFoundError as e:
            self._log(f"Navigate: {e}")
            return

        self._nav_state = "entering"
        self._nav_got_costmap = False
        self._apply_nav_state()
        self._show_view("map")
        self._map_view.clear()
        self._log(f"Navigate: switching the robot's connection to DimOS (map={map_path}).")

        def work() -> None:
            self._teardown_robot_session()
            time.sleep(1.5)  # best-effort: give the robot a moment to release the connection
            try:
                self._nav_blueprint.start(
                    map_file=map_path,
                    is_ready=self._nav_data_ready,
                    on_ready=lambda: self._post_to_ui(self._nav_became_active),
                    on_exit=lambda code: self._post_to_ui(lambda: self._on_nav_exit(code)),
                    log=self._log,
                )
                self._start_nav_subscriptions()
            except Exception as e:
                self._log(f"Navigate: failed to launch DimOS ({e}).")
                self._nav_blueprint.stop(log=self._log)  # don't leak the subprocess/connection
                self._post_to_ui(self._begin_teleop_reconnect)

        threading.Thread(target=work, daemon=True).start()

    def _nav_became_active(self) -> None:
        """Called (via _post_to_ui) once _nav_data_ready() first returns
        True — i.e. real costmap/pose data is actually flowing, not just
        "the process started"."""
        if self._nav_state != "entering":
            return
        self._nav_state = "active"
        self._apply_nav_state()

    def _exit_navigate_mode(self) -> None:
        if self._nav_state != "active":
            return
        self._nav_state = "exiting"
        self._apply_nav_state()
        self._nav_stop_requested = True
        self._stop_nav_subscriptions()
        threading.Thread(target=lambda: self._nav_blueprint.stop(log=self._log), daemon=True).start()

    def _on_nav_exit(self, code) -> None:
        was_requested = self._nav_stop_requested
        self._nav_stop_requested = False
        self._stop_nav_subscriptions()
        if not was_requested:
            self._log(
                f"Navigate: DimOS blueprint exited unexpectedly (code={code}); reconnecting teleop."
            )
        self._begin_teleop_reconnect()

    def _teardown_robot_session(self) -> None:
        """Tears down everything holding the robot's one connection slot, so
        the DimOS blueprint can claim it. Mirrors _on_close()'s robot-side
        cleanup, minus destroying the window."""
        try:
            self._listener.stop()
        except Exception:
            pass
        try:
            self._obstacle_radar.stop()
        except Exception:
            pass
        try:
            self._video.stop()
        except Exception:
            pass
        if self._conn is not None:
            try:
                self._conn.stop_movement()
                self._conn.stop()
            except Exception as e:
                self._log(f"Navigate: error closing teleop connection ({e}) — continuing anyway.")
        self._conn = None

    def _begin_teleop_reconnect(self) -> None:
        self._nav_state = "reconnecting"
        self._apply_nav_state()

        def work() -> None:
            try:
                conn = dg2._connect(self._args)
                dg2.prepare_locomotion(conn, self._args, log=self._log)
            except Exception as e:
                self._post_to_ui(lambda: self._finish_teleop_reconnect(None, e))
                return
            self._post_to_ui(lambda: self._finish_teleop_reconnect(conn, None))

        threading.Thread(target=work, daemon=True).start()

    def _finish_teleop_reconnect(self, conn, error: Exception | None) -> None:
        if error is not None or conn is None:
            self._log(
                f"Navigate: failed to reconnect teleop ({error}). Fix the issue (robot "
                "reachable? no other client still connected?) and click Retry Reconnect."
            )
            self._nav_state = "reconnect_failed"
            self._apply_nav_state()
            return
        self._conn = conn
        self._listener = RobotListener(conn, self._log)
        self._video = VideoFeed(conn)
        self._obstacle_radar = ObstacleRadar(conn)
        self._video.start()
        self._obstacle_radar.start()
        self._log("Navigate: teleop reconnected.")
        self._show_view("camera")
        self._map_view.clear()
        self._nav_state = "idle"
        self._apply_nav_state()

    _LIVE_CONTROL_WIDGET_TYPES = (ctk.CTkButton, ctk.CTkEntry, ctk.CTkSlider, ctk.CTkSwitch)

    def _disableable_widgets(self, root) -> list:
        found = []
        for child in root.winfo_children():
            if isinstance(child, self._LIVE_CONTROL_WIDGET_TYPES):
                found.append(child)
            found.extend(self._disableable_widgets(child))
        return found

    def _set_live_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        widgets = [self._paw_button]
        for frame in (self._movement_frame, self._trick_frame, self._ai_frame, self._talk_frame):
            widgets.extend(self._disableable_widgets(frame))
        for w in widgets:
            try:
                w.configure(state=state)
            except Exception:
                pass

    def _post_to_ui(self, fn) -> None:
        """Thread-safe: schedule fn to run on the Tk main thread. Same
        reasoning as _log's docstring — calling .after() itself from a
        background thread is NOT safe, so this only enqueues;
        _drain_ui_actions (driven by the main thread's own .after() loop)
        actually runs it."""
        self._ui_action_queue.put(fn)

    def _drain_ui_actions(self) -> None:
        while True:
            try:
                fn = self._ui_action_queue.get_nowait()
            except queue.Empty:
                break
            try:
                fn()
            except Exception as e:
                self._log(f"Navigate: internal UI error ({e}).")
        self.after(100, self._drain_ui_actions)

    # ─── navigate-mode bus subscriptions ──────────────────────────────
    # Reads the running blueprint's costmap/pose straight off its pub/sub
    # bus (plain LCM, same as `dimos topic echo/send`) and feeds NavMapView
    # — no DimOS web UI involved. Topics/schema below are the blueprint's
    # actual autoconnect wiring, not guesses:
    #   /global_costmap (OccupancyGrid) — CostMapper's sole output, feeds
    #     ReplanningAStarPlanner same as it would with or without
    #     relocalization (relocalization changes the costmap's *content* by
    #     also feeding CostMapper a merged_map, not which topic carries it).
    #   /odom (PoseStamped) — published by GO2Connection, frame_id="world".
    #   /target (PoseStamped) — a ReplanningAStarPlanner goal input that,
    #     unlike /goal_request, isn't also written to by this blueprint's
    #     own WavefrontFrontierExplorer/PatrollingModule modules, so our
    #     clicks can't race an autonomous goal those publish.

    def _nav_data_ready(self) -> bool:
        """Polled by NavigateBlueprint's watcher thread (a BACKGROUND
        thread — must stay Tk-free): True once at least one costmap message
        has come off the bus."""
        return self._nav_got_costmap

    def _start_nav_subscriptions(self) -> None:
        from dimos.core.transport import LCMTransport
        from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
        from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid

        costmap_t = LCMTransport("/global_costmap", OccupancyGrid)
        odom_t = LCMTransport("/odom", PoseStamped)
        goal_t = LCMTransport("/target", PoseStamped)

        def on_costmap(msg) -> None:
            self._nav_got_costmap = True
            info = msg.info
            self._post_to_ui(
                lambda: self._map_view.update_costmap(
                    info.resolution, info.origin.position.x, info.origin.position.y,
                    info.width, info.height, msg.grid,
                )
            )

        def on_odom(msg) -> None:
            yaw = _yaw_from_quaternion(msg.orientation)
            self._post_to_ui(lambda: self._map_view.update_pose(msg.position.x, msg.position.y, yaw))

        self._nav_unsub_costmap = costmap_t.subscribe(on_costmap)
        self._nav_unsub_odom = odom_t.subscribe(on_odom)
        goal_t.start()
        self._nav_costmap_transport = costmap_t
        self._nav_odom_transport = odom_t
        self._nav_goal_transport = goal_t

    def _stop_nav_subscriptions(self) -> None:
        for unsub_attr in ("_nav_unsub_costmap", "_nav_unsub_odom"):
            unsub = getattr(self, unsub_attr, None)
            if unsub is not None:
                try:
                    unsub()
                except Exception:
                    pass
                setattr(self, unsub_attr, None)
        for transport_attr in ("_nav_costmap_transport", "_nav_odom_transport", "_nav_goal_transport"):
            transport = getattr(self, transport_attr, None)
            if transport is not None:
                try:
                    transport.stop()
                except Exception:
                    pass
                setattr(self, transport_attr, None)

    def _send_nav_goal(self, x: float, y: float) -> None:
        from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped

        transport = getattr(self, "_nav_goal_transport", None)
        if transport is None:
            self._log("Navigate: goal channel isn't open (not in Navigate mode?).")
            return
        goal = PoseStamped(position=(x, y, 0.0), orientation=(0.0, 0.0, 0.0, 1.0), frame_id="world")
        try:
            transport.publish(goal)
            self._log(f"Navigate: goal sent (x={x:.2f}m, y={y:.2f}m, frame=world).")
        except Exception as e:
            self._log(f"Navigate: failed to send goal ({e}).")

    # ─── video ──────────────────────────────────────────────────────

    def _video_tick(self) -> None:
        frame, fps = self._video.latest()
        if frame is not None:
            frame = cv2.resize(frame, (VIDEO_W, VIDEO_H))
            cv2.putText(
                frame, f"Pawify LIVE  {fps:4.1f} fps", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
            )
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb)
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(VIDEO_W, VIDEO_H))
            self._video_label.configure(image=ctk_img, text="")
            self._video_label.image = ctk_img  # keep a reference, avoid GC

        distance = self._obstacle_radar.latest()
        if distance is None:
            self._obstacle_label.configure(text="obstacle: --", text_color="#888888")
        else:
            if distance < 0.5:
                color = "#e74c3c"
            elif distance < 1.0:
                color = "#f39c12"
            else:
                color = "#2ecc71"
            self._obstacle_label.configure(text=f"nearest obstacle: {distance:.2f} m", text_color=color)

        self.after(33, self._video_tick)

    # ─── logging (thread-safe: marshal onto the Tk main thread) ──────

    def _log(self, message: str) -> None:
        """Thread-safe: just enqueue. Tkinter widgets may only be touched
        from the main thread, so the actual update happens in
        _drain_log_queue, which runs on the main thread's .after() loop —
        calling .after() itself from a background thread is NOT safe
        (observed hangs/segfaults), so this never does that."""
        self._log_queue.put(message)

    def _drain_log_queue(self) -> None:
        drained = False
        while True:
            try:
                message = self._log_queue.get_nowait()
            except queue.Empty:
                break
            self._log_box.configure(state="normal")
            self._log_box.insert("end", f"{message}\n")
            drained = True
        if drained:
            self._log_box.see("end")
            self._log_box.configure(state="disabled")
        self.after(100, self._drain_log_queue)

    # ─── shutdown ───────────────────────────────────────────────────

    def _on_close(self) -> None:
        self._log("Shutting down...")
        try:
            self._listener.stop()
        except Exception:
            pass
        try:
            self._obstacle_radar.stop()
        except Exception:
            pass
        try:
            self._video.stop()
        except Exception:
            pass
        if self._nav_blueprint.running:
            try:
                self._stop_nav_subscriptions()
                self._nav_blueprint.stop(log=self._log)
            except Exception:
                pass
        if self._conn is not None:
            try:
                self._conn.stop_movement()
                self._conn.stop()
            except Exception:
                pass
        self.destroy()


def build_parser() -> argparse.ArgumentParser:
    parser = dg2.build_parser()
    parser.description = "Pawify — Go2 rescue operator console (camera + controls + talk/listen)."
    parser.add_argument(
        "--map-file",
        default="recording_go2",
        help="Premap name/path for Navigate mode's DimOS relocalization blueprint "
        "(resolved against cwd, --dimos-dir's parent, and --dimos-dir; '.pc2.lcm' is "
        "appended if missing). Editable in the UI too. Default: %(default)s.",
    )
    parser.add_argument(
        "--nav-blueprint",
        default=NAV_BLUEPRINT,
        help="DimOS blueprint Navigate mode launches. Default: %(default)s.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.speed <= 0:
        raise SystemExit("--speed must be greater than zero.")

    conn = dg2._connect(args)
    try:
        dg2.prepare_locomotion(conn, args)
    except Exception:
        conn.stop()
        raise

    app = PawifyApp(conn, args)
    app.mainloop()


if __name__ == "__main__":
    main()
