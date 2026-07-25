"""send_goal — lightweight nav-goal sender for an already-running DimOS
navigation blueprint, e.g.:

    dimos run unitree-go2-relocalization -o relocalizationmodule.map_file=recording_go2

Doesn't launch or manage that blueprint (relocalization_daemon.py does that) —
this is only a client. It renders the blueprint's live costmap + robot pose
(read straight off its pub/sub bus, same topics Pawify's own Navigate mode
uses — see pawify.py's NavMapView / _send_nav_goal) and lets you either click
a spot on the map or type x/y directly to publish a goal to /target.

Safe to run alongside Pawify or on its own: it never touches the robot's
WebRTC/teleop connection, only the DimOS pub/sub bus, so it doesn't compete
for the Go2's one-peer-at-a-time slot.

Also exposes a Cancel Navigation button — publishes to /stop_movement, the
one other pub/sub hook ReplanningAStarPlanner listens on for a control
(non-goal) command (dimos/navigation/replanning_a_star/module.py:
`stop_movement.subscribe(self._on_stop_movement)` -> `cancel_goal()`).
Robot velocity itself is never sent from here: once a goal is accepted,
DimOS's own planner drives the robot (its `nav_cmd_vel` output) — this tool
only starts/cancels that, it doesn't compete with it.
"""

import argparse
import math
import queue
import threading
import time

import customtkinter as ctk
import numpy as np
from PIL import Image as PILImage, ImageDraw as PILImageDraw

APP_TITLE = "🐾 Send Goal — DimOS nav-goal sender"
MAP_W, MAP_H = 720, 540
STALE_AFTER_S = 5.0  # no costmap update in this long -> warn the operator
WALL_COST = 100  # OccupancyGrid convention: 100 = impassable


def _yaw_from_quaternion(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class MapView(ctk.CTkFrame):
    """Renders a live costmap + robot pose and turns a click into a
    world-frame (x, y) handed to `on_click_goal`. Same rendering/click math
    as pawify.py's NavMapView — kept as a separate copy here so this script
    has no dependency on Pawify's camera/audio/teleop machinery.

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

        self._image_label = ctk.CTkLabel(
            self, text="Waiting for costmap — is the navigation blueprint running?"
        )
        self._image_label.pack(padx=8, pady=(8, 4))
        self._image_label.bind("<Button-1>", self._on_click)

        self._pose_label = ctk.CTkLabel(
            self, text="pose: --", font=ctk.CTkFont(size=12), text_color="#888888"
        )
        self._pose_label.pack(fill="x", padx=8, pady=(0, 8))

        self._grid_info: tuple[float, float, float, int, int] | None = None
        self._grid_data: np.ndarray | None = None  # raw costmap, for classify()
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
        self._grid_data = data
        self._grid_info = (resolution, origin_x, origin_y, width, height)
        self._scale = max(1, min(self._width // max(width, 1), self._height // max(height, 1)))
        self._redraw()

    def classify(self, x: float, y: float) -> str:
        """Bounds/occupancy check for a candidate goal, world-frame (x, y).
        Returns "no_map" (no costmap received yet — can't validate),
        "out_of_bounds", "wall" (cost == 100, impassable), "unknown"
        (cost == -1, unmapped), or "free"."""
        if self._grid_info is None or self._grid_data is None:
            return "no_map"
        resolution, origin_x, origin_y, width, height = self._grid_info
        col = int((x - origin_x) / resolution)
        row = int((y - origin_y) / resolution)
        if not (0 <= col < width and 0 <= row < height):
            return "out_of_bounds"
        value = int(self._grid_data[row, col])
        if value < 0:
            return "unknown"
        if value >= WALL_COST:
            return "wall"
        return "free"

    def update_pose(self, x: float, y: float, yaw: float) -> None:
        self._robot_xy = (x, y)
        self._robot_yaw = yaw
        self._pose_label.configure(text=f"pose: x={x:.2f}m y={y:.2f}m yaw={math.degrees(yaw):.0f}°")
        self._redraw()

    def update_goal(self, x: float, y: float) -> None:
        self._goal_xy = (x, y)
        self._redraw()

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


class SendGoalApp(ctk.CTk):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self._args = args
        self._log_queue: queue.Queue = queue.Queue()
        self._ui_action_queue: queue.Queue = queue.Queue()
        self._got_costmap = False
        self._last_costmap_ts: float | None = None

        ctk.set_appearance_mode("dark")
        self.title(APP_TITLE)
        self.geometry("900x760")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._status_label = ctk.CTkLabel(
            self, text="Connecting to DimOS pub/sub bus...", font=ctk.CTkFont(size=13, weight="bold")
        )
        self._status_label.pack(fill="x", padx=10, pady=(10, 4))

        self._map_view = MapView(self, width=MAP_W, height=MAP_H, on_click_goal=self._send_goal)
        self._map_view.pack(padx=10, pady=(0, 4), fill="both", expand=True)

        entry_row = ctk.CTkFrame(self, fg_color="transparent")
        entry_row.pack(fill="x", padx=10, pady=(0, 4))
        ctk.CTkLabel(entry_row, text="x:").pack(side="left", padx=(0, 4))
        self._x_entry = ctk.CTkEntry(entry_row, width=90, placeholder_text="meters")
        self._x_entry.pack(side="left", padx=(0, 10))
        ctk.CTkLabel(entry_row, text="y:").pack(side="left", padx=(0, 4))
        self._y_entry = ctk.CTkEntry(entry_row, width=90, placeholder_text="meters")
        self._y_entry.pack(side="left", padx=(0, 10))
        ctk.CTkButton(entry_row, text="Send Goal", command=self._send_goal_from_entries).pack(
            side="left"
        )
        ctk.CTkButton(
            entry_row, text="⏹ Cancel Navigation", fg_color="#c0392b", hover_color="#922b21",
            command=self._cancel_navigation,
        ).pack(side="left", padx=(10, 0))

        self._log_box = ctk.CTkTextbox(self, height=140, state="disabled")
        self._log_box.pack(fill="x", padx=10, pady=(0, 10))

        self.after(100, self._drain_log_queue)
        self.after(100, self._drain_ui_actions)
        self.after(1000, self._check_staleness)
        threading.Thread(target=self._connect, daemon=True).start()

    # ─── pub/sub bus ────────────────────────────────────────────────

    def _connect(self) -> None:
        from dimos.core.transport import LCMTransport
        from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
        from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid
        from dimos.msgs.std_msgs.Bool import Bool

        costmap_t = LCMTransport(self._args.costmap_topic, OccupancyGrid)
        odom_t = LCMTransport(self._args.odom_topic, PoseStamped)
        goal_t = LCMTransport(self._args.goal_topic, PoseStamped)
        stop_t = LCMTransport(self._args.stop_topic, Bool)

        def on_costmap(msg) -> None:
            self._last_costmap_ts = time.monotonic()
            if not self._got_costmap:
                self._got_costmap = True
                self._log("Connected — costmap received. Click the map to send a goal.")
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

        costmap_t.subscribe(on_costmap)
        odom_t.subscribe(on_odom)
        goal_t.start()
        stop_t.start()
        self._goal_transport = goal_t
        self._stop_transport = stop_t
        self._log(
            f"Listening on {self._args.costmap_topic} / {self._args.odom_topic}, "
            f"publishing goals to {self._args.goal_topic}, cancel to {self._args.stop_topic}."
        )

    def _send_goal(self, x: float, y: float) -> None:
        from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped

        classification = self._map_view.classify(x, y)
        if classification == "out_of_bounds":
            self._log(f"Rejected goal x={x:.2f}m y={y:.2f}m — outside the loaded costmap.")
            return
        if classification == "wall":
            self._log(f"Rejected goal x={x:.2f}m y={y:.2f}m — that cell is marked impassable.")
            return
        if classification == "no_map":
            self._log(f"No costmap yet — sending x={x:.2f}m y={y:.2f}m without bounds checking.")
        elif classification == "unknown":
            self._log(f"x={x:.2f}m y={y:.2f}m is unmapped territory — sending anyway.")

        transport = getattr(self, "_goal_transport", None)
        if transport is None:
            self._log("Not connected to the DimOS bus yet — try again in a moment.")
            return
        self._post_to_ui(lambda: self._map_view.update_goal(x, y))
        goal = PoseStamped(
            position=(x, y, 0.0), orientation=(0.0, 0.0, 0.0, 1.0), frame_id=self._args.frame_id
        )
        try:
            transport.publish(goal)
            self._log(f"Goal sent: x={x:.2f}m y={y:.2f}m (frame={self._args.frame_id}).")
        except Exception as e:
            self._log(f"Failed to send goal ({e}).")

    def _send_goal_from_entries(self) -> None:
        try:
            x = float(self._x_entry.get().strip())
            y = float(self._y_entry.get().strip())
        except ValueError:
            self._log("Enter numeric x and y (meters) before sending.")
            return
        self._send_goal(x, y)

    def _cancel_navigation(self) -> None:
        from dimos.msgs.std_msgs.Bool import Bool

        transport = getattr(self, "_stop_transport", None)
        if transport is None:
            self._log("Not connected to the DimOS bus yet — try again in a moment.")
            return
        try:
            transport.publish(Bool(data=True))
            self._log("Cancel sent — DimOS should stop and drop the current goal.")
        except Exception as e:
            self._log(f"Failed to send cancel ({e}).")

    def _check_staleness(self) -> None:
        """LCM is connectionless multicast — there's no socket to notice
        dropping. The practical equivalent is "we used to get costmap
        frames and now we don't", so warn on that instead."""
        if self._got_costmap and self._last_costmap_ts is not None:
            age = time.monotonic() - self._last_costmap_ts
            if age > STALE_AFTER_S:
                self._status_label.configure(
                    text=f"⚠ No costmap update in {age:.0f}s — is the DimOS blueprint still running?",
                    text_color="#e74c3c",
                )
            else:
                self._status_label.configure(
                    text="Connected — click the map (or type x/y) to send a goal.",
                    text_color="#2ecc71",
                )
        self.after(1000, self._check_staleness)

    # ─── thread-safe UI plumbing ────────────────────────────────────
    # LCM subscription callbacks fire on a background thread; Tkinter
    # widgets may only be touched from the main thread, so both logging and
    # map/pose updates go through a queue drained by the main thread's own
    # .after() loop instead of calling .after() from the background thread
    # itself (that pattern is not safe — see pawify.py's _log docstring).

    def _log(self, message: str) -> None:
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

    def _post_to_ui(self, fn) -> None:
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
                self._log(f"Internal UI error ({e}).")
        self.after(100, self._drain_ui_actions)

    def _on_close(self) -> None:
        for attr in ("_goal_transport", "_stop_transport"):
            transport = getattr(self, attr, None)
            if transport is not None:
                try:
                    transport.stop()
                except Exception:
                    pass
        self.destroy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Send navigation goals to an already-running DimOS navigation blueprint "
            "(e.g. `dimos run unitree-go2-relocalization -o relocalizationmodule.map_file=...`)."
        )
    )
    parser.add_argument(
        "--costmap-topic", default="/global_costmap", help="Default: %(default)s."
    )
    parser.add_argument("--odom-topic", default="/odom", help="Default: %(default)s.")
    parser.add_argument(
        "--goal-topic",
        default="/target",
        help="ReplanningAStarPlanner's goal input. Default: %(default)s — deliberately not "
        "/goal_request, which this blueprint's own frontier-exploration/patrolling modules "
        "also write to, so a click here can't race an autonomous goal those publish.",
    )
    parser.add_argument("--frame-id", default="world", help="Default: %(default)s.")
    parser.add_argument(
        "--stop-topic",
        default="/stop_movement",
        help="Cancel-navigation control input (Bool). Default: %(default)s.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    app = SendGoalApp(args)
    app.mainloop()


if __name__ == "__main__":
    main()
