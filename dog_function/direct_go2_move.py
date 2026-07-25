import argparse
import difflib
import json
import math
import os
import queue
import select
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import termios
    import tty
except ImportError:  # pragma: no cover - interactive mode is POSIX-only
    termios = None
    tty = None


MCF_FREE_WALK_API_ID = 2045

# Sport command api_ids for MCF mode — the mode this script's connection
# actually runs in (mode="ai" in UnitreeWebRTCConnection, confirmed by
# MCF_FREE_WALK_API_ID=2045 being required to enable walking at all).
# MCF shares the rt/api/sport/request topic with "normal" mode but has a
# PARTLY DIFFERENT api_id space (e.g. BackFlip is 2043 in MCF vs 1044 in
# normal) — see unitree_webrtc_connect/constants.py:SPORT_CMD_MCF. Using the
# "normal" ids here would silently no-op under MCF.
#
# Dances and flips (Dance1/Dance2/FrontFlip/LeftFlip/BackFlip) are
# intentionally NOT in this table — this is a rescue/utility console, not a
# stunt console, and those moves risk destabilizing the robot near an
# incident scene. RightFlip/Wallow/WiggleHips/MoonWalk have no MCF
# equivalent and were never available here.
SPORT_COMMANDS: dict[str, int] = {
    "BalanceStand": 1002,
    "StandUp": 1004,
    "StandDown": 1005,
    "RecoveryStand": 1006,
    "Sit": 1009,
    "RiseSit": 1010,
    "Hello": 1016,
    "Stretch": 1017,
    "Content": 1020,
    "Scrape": 1029,
    "FrontJump": 1031,
    "FrontPounce": 1032,
    "FingerHeart": 1036,
    "Handstand": 2044,
    "FreeBound": 2046,
    "FreeJump": 2047,
    "ClassicWalk": 2049,
    "BackStand": 2050,
    "CrossStep": 2051,
}

TRICK_ALIASES: dict[str, str] = {
    "jump": "FrontJump",
    "hi": "Hello",
    "wave": "Hello",
    "greet": "Hello",
    "bow": "Hello",
    "pounce": "FrontPounce",
    "lie": "StandDown",
    "lay": "StandDown",
    "down": "StandDown",
    "recover": "RecoveryStand",
    "stand": "StandUp",
    "bound": "FreeBound",
}

# Commands that leave the robot in a posture where it needs RecoveryStand
# before it will respond well to normal walking commands again.
_DYNAMIC_COMMANDS = {
    "FrontJump",
    "FrontPounce",
    "Handstand",
    "Sit",
    "StandDown",
    "FreeBound",
    "FreeJump",
    "BackStand",
}

_MOVE_KEYS = {"w", "a", "s", "d", "q", "e"}
_HOLD_WINDOW = 0.3  # seconds a held key stays "active" between OS key-repeats
_TICK = 0.05

_HELP_TEXT = """
Go2 interactive control
  Movement (hold):   W/S forward/back   A/D turn left/right   Q/E strafe left/right
  Speed:             +/- adjust linear speed
  Quick tricks:      J jump   H hello   U stand up   Z lie down
  Stop:              SPACE (zero velocity immediately)
  Command mode:      ENTER or :  then type a command, e.g.:
                       forward 1.5 | back 0.5 | left 1 | right 1 | turn 90
                       speed 0.4 | stop | standup | list | help
                       say Hello, is anyone there?
                       ai wave hello then back up a bit
                       or any trick name, e.g. jump, Stretch, FingerHeart
  Quit:              ESC, Ctrl-C, or Q
"""


@dataclass
class TeleopState:
    speed: float
    turn_speed: float
    max_speed: float
    accel: float = 1.5  # m/s^2 slew rate for linear velocity ramping
    turn_accel: float = 3.0  # rad/s^2 slew rate for angular velocity ramping
    cur_x: float = 0.0
    cur_y: float = 0.0
    cur_yaw: float = 0.0


def _slew(current: float, target: float, max_delta: float) -> float:
    """Move `current` toward `target` by at most `max_delta` (rate limiting)."""
    if current < target:
        return min(current + max_delta, target)
    if current > target:
        return max(current - max_delta, target)
    return current


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def add_dimos_to_path(dimos_dir: Path) -> None:
    if dimos_dir.exists():
        sys.path.insert(0, str(dimos_dir))


def require_robot_step(label: str, action, log=print) -> None:
    log(f"Preparing Go2: {label}...")
    if not action():
        raise SystemExit(f"Go2 rejected preparation step: {label}")
    log(f"Preparing Go2: {label} accepted.")


def prepare_locomotion(conn, args: argparse.Namespace, log=print) -> None:
    require_robot_step("stand up", conn.standup, log=log)
    time.sleep(args.stand_settle)

    if getattr(args, "obstacle_avoidance", True):
        require_robot_step(
            "enable obstacle avoidance",
            lambda: conn.set_obstacle_avoidance(True),
            log=log,
        )
    else:
        log("Obstacle avoidance disabled by --no-obstacle-avoidance — the robot will NOT stop itself near walls/obstacles.")

    if args.control_api == "velocity":
        require_robot_step(
            f"free walk (API {MCF_FREE_WALK_API_ID})",
            lambda: conn.sport_command(MCF_FREE_WALK_API_ID),
            log=log,
        )
        time.sleep(args.free_walk_settle)
        return

    require_robot_step("balance stand", conn.balance_stand, log=log)
    time.sleep(args.mode_settle)
    require_robot_step("enable joystick", lambda: conn.switch_joystick(True), log=log)
    time.sleep(args.mode_settle)


def _connect(args: argparse.Namespace):
    add_dimos_to_path(args.dimos_dir)
    load_env_file(args.dimos_dir / ".env")

    from dimos.robot.unitree.connection import UnitreeWebRTCConnection

    ip = args.ip or os.environ.get("ROBOT_IP")
    aes_key = args.aes_key or os.environ.get("UNITREE_AES_128_KEY")
    if not ip:
        raise SystemExit("Missing robot IP. Pass --ip or set ROBOT_IP.")
    if not aes_key:
        raise SystemExit("Missing AES key. Pass --aes-key or set UNITREE_AES_128_KEY.")

    print(f"Connecting Go2 ip={ip} control_api={args.control_api}")
    return UnitreeWebRTCConnection(
        ip=ip,
        aes_128_key=aes_key,
        velocity_api=args.control_api == "velocity",
    )


def _make_twist(x: float, y: float, yaw: float):
    from dimos.msgs.geometry_msgs.Twist import Twist
    from dimos.msgs.geometry_msgs.Vector3 import Vector3

    return Twist(linear=Vector3(x, y, 0.0), angular=Vector3(0.0, 0.0, yaw))


def move_go2(args: argparse.Namespace) -> None:
    if args.speed <= 0:
        raise SystemExit("--speed must be greater than zero.")

    distance = max(abs(args.forward), abs(args.left))
    duration = args.duration
    if duration is None:
        duration = max(args.min_duration, distance / args.speed) if distance > 0 else 0.2

    x_speed = 0.0 if args.forward == 0 else args.speed * (1 if args.forward > 0 else -1)
    y_speed = 0.0 if args.left == 0 else args.speed * (1 if args.left > 0 else -1)

    conn = _connect(args)
    try:
        prepare_locomotion(conn, args)

        twist = _make_twist(x_speed, y_speed, args.yaw)
        print(
            "Sending continuous movement commands... "
            f"forward={args.forward:.2f}m left={args.left:.2f}m "
            f"speed={args.speed:.2f}m/s duration={duration:.2f}s"
        )
        if not conn.move(twist, duration=duration):
            raise SystemExit("Go2 move command failed.")
        time.sleep(0.2)
        conn.stop_movement()
        print("Go2 movement commands sent and stop command issued.")
    finally:
        conn.stop()


# ─── interactive keyboard + command control ────────────────────────────


class _KeyReader:
    """Background stdin reader for cbreak-mode single-key input."""

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._queue: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def drain(self) -> list[str]:
        chars = []
        while True:
            try:
                chars.append(self._queue.get_nowait())
            except queue.Empty:
                return chars

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._paused.is_set():
                time.sleep(0.05)
                continue
            ready, _, _ = select.select([self._fd], [], [], 0.1)
            if ready:
                chunk = os.read(self._fd, 1)
                if chunk:
                    self._queue.put(chunk.decode(errors="ignore"))


def _resolve_trick_name(raw: str) -> str | None:
    raw = raw.strip()
    if not raw:
        return None
    alias = TRICK_ALIASES.get(raw.lower())
    if alias:
        return alias
    for name in SPORT_COMMANDS:
        if name.lower() == raw.lower():
            return name
    return None


def _suggest_trick_names(raw: str) -> list[str]:
    pool = list(SPORT_COMMANDS) + list(TRICK_ALIASES)
    return difflib.get_close_matches(raw, pool, n=3, cutoff=0.5)


def _extract_response_code(response) -> int | None:
    """Best-effort pull of a status/error code out of a raw SPORT_MOD response.

    conn.sport_command() just does bool(response), which is True for any
    non-empty dict — including one carrying a firmware-side rejection. This
    digs into the actual payload so failures are visible instead of silently
    printed as "accepted".
    """
    if not isinstance(response, dict):
        return None
    data = response.get("data")
    if not isinstance(data, dict):
        return None
    header = data.get("header")
    if isinstance(header, dict):
        status = header.get("status")
        if isinstance(status, dict) and "code" in status:
            return status["code"]
    if "code" in data:
        return data["code"]
    return None


def _fire_trick(conn, name: str, log=print) -> None:
    from unitree_webrtc_connect.constants import RTC_TOPIC

    # Dynamic moves (flips especially) can be silently ignored by the
    # firmware if it's still mid-stream on velocity/Move commands — stop and
    # let the robot settle before asking for a trick.
    conn.stop_movement()
    time.sleep(0.3)

    api_id = SPORT_COMMANDS[name]
    log(f"Trick: {name} (api_id={api_id})")
    try:
        response = conn.publish_request(RTC_TOPIC["SPORT_MOD"], {"api_id": api_id})
    except Exception as e:
        log(f"  -> request failed: {e}")
        return

    code = _extract_response_code(response)
    if code is None:
        log(f"  -> sent; robot response: {response!r}")
    elif code == 0:
        log("  -> accepted (code 0).")
    else:
        log(f"  -> REJECTED by the robot (code {code}) — it did not perform {name}.")
        return

    if name in _DYNAMIC_COMMANDS:
        time.sleep(1.5)
        log("  -> auto RecoveryStand to reset posture...")
        conn.sport_command(SPORT_COMMANDS["RecoveryStand"])


def _wav_duration_seconds(wav_path) -> float:
    import wave

    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


# Verified (via dimos's own working dimos/skills/unitree/unitree_speak.py)
# that the robot's speaker pipeline expects mono 16-bit PCM WAV at 22050 Hz,
# peak-normalized. Nothing upstream of this script guarantees that shape —
# sounddevice recordings default to 44.1kHz, and pyttsx3/espeak-ng's output
# rate depends on the installed voice/system. The robot does not reject a
# malformed upload; it just plays silence or noise, which is exactly the
# "chunks uploaded, nothing heard" symptom this was chasing.
ROBOT_AUDIO_SAMPLE_RATE = 22050


def _normalize_wav_for_robot(src_path, log=print) -> tuple[Path, float]:
    """Re-encode any WAV to mono/16-bit-PCM/22050Hz + peak-normalized gain
    before it ever reaches the megaphone upload. Returns (path, duration).

    Best-effort: if decoding fails for any reason, falls back to sending
    the original file untouched (with a log line explaining why) rather
    than blocking playback outright — a format quirk should degrade to
    "maybe garbled," not "nothing happens, silently"."""
    import wave

    import numpy as np

    try:
        with wave.open(str(src_path), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            rate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())

        if sampwidth == 2:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
        elif sampwidth == 1:
            samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) * 256.0
        elif sampwidth == 4:
            samples = np.frombuffer(raw, dtype=np.int32).astype(np.float64) / 65536.0
        else:
            raise ValueError(f"unsupported WAV sample width: {sampwidth} bytes")

        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)

        if samples.size and rate != ROBOT_AUDIO_SAMPLE_RATE:
            old_len = samples.size
            new_len = max(1, int(old_len * ROBOT_AUDIO_SAMPLE_RATE / rate))
            samples = np.interp(np.linspace(0, old_len - 1, new_len), np.arange(old_len), samples)
            rate = ROBOT_AUDIO_SAMPLE_RATE

        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak > 0:
            samples = samples * (32767.0 / peak)
        pcm16 = np.clip(samples, -32768, 32767).astype(np.int16)

        out_path = Path(tempfile.gettempdir()) / f"go2_audio_norm_{int(time.time() * 1000)}.wav"
        with wave.open(str(out_path), "wb") as wf_out:
            wf_out.setnchannels(1)
            wf_out.setsampwidth(2)
            wf_out.setframerate(rate)
            wf_out.writeframes(pcm16.tobytes())

        return out_path, (pcm16.size / float(rate) if rate else 0.0)
    except Exception as e:
        log(f"  -> audio normalization failed ({e}); sending the original file as-is.")
        try:
            return Path(src_path), _wav_duration_seconds(src_path)
        except Exception:
            return Path(src_path), 3.0


def _audiohub_request(hub, api_id: int, parameter: dict | None = None):
    """Issue one rt/api/audiohub/request call directly against the data
    channel, bypassing WebRTCAudioHub's enter_megaphone()/exit_megaphone()
    (which discard the response) so the robot's actual accept/reject can be
    checked instead of just assuming success from "no exception"."""
    return hub.data_channel.pub_sub.publish_request_new(
        "rt/api/audiohub/request",
        {"api_id": api_id, "parameter": json.dumps(parameter or {})},
    )


def _send_wav_via_megaphone(conn, wav_path, log=print) -> bool:
    """Broadcast a WAV file through the robot's speaker (megaphone mode).
    Shared by voice push-to-talk and typed say() — both just need to get a
    WAV onto the robot; only how the WAV was produced differs.

    Full audio channel lifecycle, each step logged and checked against the
    robot's real response rather than assumed:
      1. normalize the WAV to the format the robot's speaker actually wants
         (see _normalize_wav_for_robot — the most likely fix for silent
         playback).
      2. enter megaphone mode (checked — a rejection here is now visible).
      3. upload the WAV in chunks.
      4. wait out the clip's actual playback duration — exiting megaphone
         mode immediately after upload cuts the robot's speaker off right
         as playback would start (chunks upload fine, nothing is heard).
      5. exit megaphone mode — in a `finally`, so a failure in steps 2-3
         can't leave the robot stuck in megaphone mode, which would
         silently break every subsequent send too.
    Matches the timing dimos's own dimos/skills/unitree/unitree_speak.py
    already uses successfully."""
    import asyncio

    from unitree_webrtc_connect.constants import AUDIO_API
    from unitree_webrtc_connect.webrtc_audiohub import WebRTCAudioHub

    norm_path, duration = _normalize_wav_for_robot(wav_path, log=log)
    log(f"  -> audio ready: {duration:.1f}s, {ROBOT_AUDIO_SAMPLE_RATE}Hz mono 16-bit PCM.")

    try:
        hub = WebRTCAudioHub(conn.conn)
    except Exception as e:
        log(f"  -> could not open the robot's audio data channel: {e}")
        if norm_path != Path(wav_path):
            norm_path.unlink(missing_ok=True)
        return False

    entered = False

    async def _send() -> bool:
        nonlocal entered
        resp = await _audiohub_request(hub, AUDIO_API["ENTER_MEGAPHONE"])
        code = _extract_response_code(resp)
        if code not in (None, 0):
            log(f"  -> robot REJECTED entering megaphone mode (code {code}).")
            return False
        entered = True
        await asyncio.sleep(0.2)  # let the robot's audio pipeline actually switch modes

        await hub.upload_megaphone(str(norm_path))
        await asyncio.sleep(duration + 1.0)  # let it actually finish playing before exiting
        return True

    async def _exit() -> None:
        resp = await _audiohub_request(hub, AUDIO_API["EXIT_MEGAPHONE"])
        code = _extract_response_code(resp)
        if code not in (None, 0):
            log(f"  -> robot reported an issue exiting megaphone mode (code {code}).")

    ok = False
    try:
        ok = asyncio.run_coroutine_threadsafe(_send(), conn.loop).result(timeout=60 + duration)
    except Exception as e:
        log(f"  -> failed to send audio: {e}")
    finally:
        if entered:
            try:
                asyncio.run_coroutine_threadsafe(_exit(), conn.loop).result(timeout=10)
            except Exception as e:
                log(f"  -> failed to cleanly exit megaphone mode ({e}) — a later send may need a retry.")
        if norm_path != Path(wav_path):
            norm_path.unlink(missing_ok=True)

    return ok


def _speak_text(conn, text: str, log=print) -> None:
    """Synthesize `text` offline (pyttsx3 — no network/API key needed, since
    the robot's own hotspot likely has no internet route) and broadcast it
    through the robot's speaker via megaphone mode."""
    text = text.strip()
    if not text:
        return
    log(f'Say: "{text}"')
    try:
        import pyttsx3
    except ImportError:
        log("  -> pyttsx3 not installed. Run: pip install pyttsx3")
        return

    tmp_path = Path(tempfile.gettempdir()) / f"go2_say_{int(time.time() * 1000)}.wav"
    try:
        engine = pyttsx3.init()
        engine.save_to_file(text, str(tmp_path))
        engine.runAndWait()
    except Exception as e:
        log(
            f"  -> speech synthesis failed ({e}). On Linux, pyttsx3 needs the "
            "espeak-ng package: sudo pacman -S espeak-ng"
        )
        return

    if not tmp_path.exists() or tmp_path.stat().st_size == 0:
        log("  -> speech synthesis produced no audio (espeak-ng likely missing).")
        return

    try:
        log("  -> sending to robot speaker (upload + playback wait — can take a while)...")
        if _send_wav_via_megaphone(conn, tmp_path, log=log):
            log("  -> sent.")
    finally:
        tmp_path.unlink(missing_ok=True)


# ─── natural-language control via OpenRouter (tool-calling) ────────────
#
# Tools are deliberately limited to things a language model can reason
# about from text alone: movement amounts, trick names, speech, stop. It is
# NOT given vision or map data, so it can't meaningfully pick navigate-to
# coordinates or choose which detected person to follow — those need actual
# perception and stay under direct operator control (map/camera clicks in
# Pawify). Using an LLM to do real-time obstacle/collision detection from
# camera frames would be slow (network round-trip per frame), unreliable,
# and worse than the lidar-based obstacle avoidance already wired in —
# deliberately not done here.

AI_MODEL = "anthropic/claude-sonnet-5"
AI_SERVER_URL = "https://ai.hackclub.com/proxy/v1"
AI_MAX_STEPS = 4
AI_MAX_MOVE_M = 2.0
AI_MAX_TURN_DEG = 180.0

_AI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": "Move the robot forward/backward and/or strafe left/right, a short cautious pulse.",
            "parameters": {
                "type": "object",
                "properties": {
                    "forward_m": {"type": "number", "description": "Meters forward; negative = backward."},
                    "left_m": {"type": "number", "description": "Meters left; negative = right."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "turn",
            "description": "Turn the robot in place, in degrees.",
            "parameters": {
                "type": "object",
                "properties": {
                    "degrees": {
                        "type": "number",
                        "description": "Positive = turn left, negative = turn right.",
                    },
                },
                "required": ["degrees"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trick",
            "description": (
                "Perform a named robot trick/sport command. Available names: "
                + ", ".join(sorted(SPORT_COMMANDS))
            ),
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "say",
            "description": "Speak text out loud through the robot's speaker.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop",
            "description": "Immediately stop all robot movement.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

_ai_state_singleton: TeleopState | None = None


def _ai_teleop_state() -> TeleopState:
    """A separate, deliberately cautious speed profile for AI-triggered
    movement — independent of whatever speed the human operator has dialed
    in for manual control."""
    global _ai_state_singleton
    if _ai_state_singleton is None:
        _ai_state_singleton = TeleopState(speed=0.25, turn_speed=0.6, max_speed=0.5)
    return _ai_state_singleton


def _execute_ai_tool_call(conn, name: str, args: dict, log=print) -> str:
    try:
        if name == "move":
            forward = max(-AI_MAX_MOVE_M, min(AI_MAX_MOVE_M, float(args.get("forward_m", 0.0))))
            left = max(-AI_MAX_MOVE_M, min(AI_MAX_MOVE_M, float(args.get("left_m", 0.0))))
            _pulse_move(conn, _ai_teleop_state(), forward, left, 0.0)
            return f"moved forward={forward:.2f}m left={left:.2f}m"
        if name == "turn":
            degrees = max(-AI_MAX_TURN_DEG, min(AI_MAX_TURN_DEG, float(args.get("degrees", 0.0))))
            state = _ai_teleop_state()
            duration = max(0.2, abs(math.radians(degrees)) / state.turn_speed)
            yaw = state.turn_speed if degrees > 0 else -state.turn_speed
            _pulse_move(conn, state, 0.0, 0.0, yaw, duration=duration)
            return f"turned {degrees:.0f} degrees"
        if name == "trick":
            trick_name = _resolve_trick_name(str(args.get("name", "")))
            if trick_name is None:
                return f"error: unknown trick {args.get('name')!r}. Available: {sorted(SPORT_COMMANDS)}"
            _fire_trick(conn, trick_name, log=log)
            return f"performed {trick_name}"
        if name == "say":
            _speak_text(conn, str(args.get("text", "")), log=log)
            return "spoke the message"
        if name == "stop":
            conn.stop_movement()
            return "stopped"
        return f"error: unknown tool {name!r}"
    except Exception as e:
        return f"error: {e}"


def run_ai_command(conn, instruction: str, api_key: str | None, log=print) -> None:
    """Send a natural-language instruction to Claude (via OpenRouter,
    tool-calling) and execute whatever robot actions it decides to take."""
    instruction = instruction.strip()
    if not instruction:
        return
    if not api_key:
        log(
            "AI: no OpenRouter API key configured (set OPENROUTER_API_KEY in "
            "dimos/.env or pass --openrouter-key)."
        )
        return

    try:
        from openrouter import OpenRouter
    except ImportError:
        log("AI: openrouter package not installed. Run: pip install openrouter")
        return

    log(f'AI: "{instruction}"')
    client = OpenRouter(api_key=api_key, server_url=AI_SERVER_URL)
    messages = [
        {
            "role": "system",
            "content": (
                "You control a Unitree Go2 robot dog through the provided tools. "
                "Carry out the user's instruction using small, cautious movements. "
                "After acting, reply with a brief one-sentence summary of what you did."
            ),
        },
        {"role": "user", "content": instruction},
    ]

    for _ in range(AI_MAX_STEPS):
        try:
            response = client.chat.send(model=AI_MODEL, messages=messages, tools=_AI_TOOLS)
        except Exception as e:
            log(f"AI: request failed ({e}).")
            return

        choice = response.choices[0]
        msg = choice.message
        tool_calls = getattr(msg, "tool_calls", None) or []

        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": c.type,
                        "function": {"name": c.function.name, "arguments": c.function.arguments},
                    }
                    for c in tool_calls
                ],
            }
        )

        if not tool_calls:
            if msg.content:
                log(f"AI: {msg.content}")
            return

        for call in tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = _execute_ai_tool_call(conn, call.function.name, args, log=log)
            log(f"AI: {call.function.name}({args}) -> {result}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    log("AI: reached step limit without a final answer.")


_RAMP_TIME = 0.4  # seconds spent ramping up to / down from target velocity


def _pulse_move(conn, state: TeleopState, x_m: float, y_m: float, yaw_rate: float, duration: float | None = None) -> None:
    """Move with a trapezoidal velocity profile (ramp up, cruise, ramp down)
    instead of snapping straight to full speed and hard-stopping — the abrupt
    step function is what made chained commands feel disjointed."""
    distance = max(abs(x_m), abs(y_m))
    if duration is None:
        duration = max(0.2, distance / state.speed) if distance > 0 else 0.2
    target_x = 0.0 if x_m == 0 else state.speed * (1 if x_m > 0 else -1)
    target_y = 0.0 if y_m == 0 else state.speed * (1 if y_m > 0 else -1)
    print(f"Moving: x={target_x:+.2f} y={target_y:+.2f} yaw={yaw_rate:+.2f} duration={duration:.2f}s")

    max_lin_delta = state.accel * _TICK
    max_ang_delta = state.turn_accel * _TICK
    cur_x = cur_y = cur_yaw = 0.0
    start = time.monotonic()
    while True:
        remaining = duration - (time.monotonic() - start)
        if remaining <= 0:
            break
        ramping_down = remaining <= _RAMP_TIME
        goal_x = 0.0 if ramping_down else target_x
        goal_y = 0.0 if ramping_down else target_y
        goal_yaw = 0.0 if ramping_down else yaw_rate
        cur_x = _slew(cur_x, goal_x, max_lin_delta)
        cur_y = _slew(cur_y, goal_y, max_lin_delta)
        cur_yaw = _slew(cur_yaw, goal_yaw, max_ang_delta)
        conn.move(_make_twist(cur_x, cur_y, cur_yaw), duration=0)
        time.sleep(_TICK)
    conn.stop_movement()


def _dispatch_command(conn, state: TeleopState, args: argparse.Namespace, line: str) -> bool:
    """Handle one typed command. Returns True if the user asked to quit."""
    parts = line.split()
    cmd = parts[0].lower()
    rest = parts[1:]

    if cmd in ("quit", "exit"):
        return True
    if cmd == "help":
        print(_HELP_TEXT)
        return False
    if cmd == "list":
        print("Tricks:", ", ".join(sorted(SPORT_COMMANDS)))
        print("Aliases:", ", ".join(sorted(TRICK_ALIASES)))
        return False
    if cmd == "stop":
        conn.stop_movement()
        return False
    if cmd == "standup":
        print("Standing up...")
        prepare_locomotion(conn, args)
        return False
    if cmd == "say":
        text = line[len(parts[0]) :].strip()
        if not text:
            print("Usage: say <message>")
            return False
        _speak_text(conn, text)
        return False
    if cmd == "ai":
        instruction = line[len(parts[0]) :].strip()
        if not instruction:
            print("Usage: ai <instruction>  (e.g. ai wave hello then take a step back)")
            return False
        api_key = getattr(args, "openrouter_key", None) or os.environ.get("OPENROUTER_API_KEY")
        run_ai_command(conn, instruction, api_key)
        return False
    if cmd == "speed":
        if not rest:
            print(f"Current speed: {state.speed:.2f} m/s. Usage: speed <m/s>")
            return False
        try:
            state.speed = max(0.05, min(float(rest[0]), state.max_speed))
        except ValueError:
            print(f"Not a number: {rest[0]!r}")
            return False
        print(f"speed -> {state.speed:.2f} m/s")
        return False
    if cmd in ("forward", "back", "backward", "left", "right"):
        if not rest:
            print(f"Usage: {cmd} <meters>")
            return False
        try:
            distance = abs(float(rest[0]))
        except ValueError:
            print(f"Not a number: {rest[0]!r}")
            return False
        x = distance if cmd == "forward" else -distance if cmd in ("back", "backward") else 0.0
        y = distance if cmd == "left" else -distance if cmd == "right" else 0.0
        _pulse_move(conn, state, x, y, 0.0)
        return False
    if cmd == "turn":
        if not rest:
            print("Usage: turn <degrees>  (positive = left, negative = right)")
            return False
        try:
            degrees = float(rest[0])
        except ValueError:
            print(f"Not a number: {rest[0]!r}")
            return False
        duration = max(0.2, abs(math.radians(degrees)) / state.turn_speed)
        yaw = state.turn_speed if degrees > 0 else -state.turn_speed
        _pulse_move(conn, state, 0.0, 0.0, yaw, duration=duration)
        return False

    name = _resolve_trick_name(line)
    if name is None:
        suggestions = _suggest_trick_names(cmd)
        hint = f" Did you mean: {suggestions}?" if suggestions else " Type 'list' for tricks or 'help'."
        print(f"Unknown command {cmd!r}.{hint}")
        return False
    _fire_trick(conn, name)
    return False


def _enter_command_mode(conn, reader: _KeyReader, fd: int, old_settings, state: TeleopState, args: argparse.Namespace) -> bool:
    reader.pause()
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    try:
        line = input("\ncmd> ").strip()
    except (EOFError, KeyboardInterrupt):
        line = ""
    finally:
        tty.setcbreak(fd)
        reader.resume()
    if not line:
        return False
    return _dispatch_command(conn, state, args, line)


def _handle_key(ch: str, conn, reader: _KeyReader, fd: int, old_settings, state: TeleopState, held: dict, now: float, args: argparse.Namespace) -> bool:
    if ch in ("\x03", "Q"):  # Ctrl-C, capital Q
        return True
    if ch == "\x1b":  # ESC
        return True
    if ch == " ":
        held.clear()
        state.cur_x = state.cur_y = state.cur_yaw = 0.0
        conn.stop_movement()
        return False
    if ch in ("\r", "\n", ":"):
        return _enter_command_mode(conn, reader, fd, old_settings, state, args)
    if ch in ("?", "/"):
        print(_HELP_TEXT)
        return False

    lower = ch.lower()
    if lower in _MOVE_KEYS:
        held[lower] = now
        return False
    if ch in ("+", "="):
        state.speed = min(state.speed + 0.05, state.max_speed)
        print(f"speed -> {state.speed:.2f} m/s")
        return False
    if ch in ("-", "_"):
        state.speed = max(state.speed - 0.05, 0.05)
        print(f"speed -> {state.speed:.2f} m/s")
        return False
    if lower == "j":
        _fire_trick(conn, "FrontJump")
        return False
    if lower == "h":
        _fire_trick(conn, "Hello")
        return False
    if lower == "u":
        print("Standing up...")
        prepare_locomotion(conn, args)
        return False
    if lower == "z":
        _fire_trick(conn, "StandDown")
        return False
    return False


def _twist_from_held(held: dict, state: TeleopState):
    """Ramp state.cur_{x,y,yaw} toward whatever WASD/QE currently target,
    rate-limited by state.accel/turn_accel, so starting/stopping/changing
    direction is a smooth slew instead of an instant step."""
    target_x = (state.speed if "w" in held else 0.0) - (state.speed if "s" in held else 0.0)
    target_y = (state.speed if "q" in held else 0.0) - (state.speed if "e" in held else 0.0)
    target_yaw = (state.turn_speed if "a" in held else 0.0) - (state.turn_speed if "d" in held else 0.0)

    max_lin_delta = state.accel * _TICK
    max_ang_delta = state.turn_accel * _TICK
    state.cur_x = _slew(state.cur_x, target_x, max_lin_delta)
    state.cur_y = _slew(state.cur_y, target_y, max_lin_delta)
    state.cur_yaw = _slew(state.cur_yaw, target_yaw, max_ang_delta)

    if state.cur_x == 0.0 and state.cur_y == 0.0 and state.cur_yaw == 0.0:
        return None
    return _make_twist(state.cur_x, state.cur_y, state.cur_yaw)


def _run_interactive_loop(conn, args: argparse.Namespace) -> None:
    if termios is None or tty is None:
        raise SystemExit("--interactive requires a POSIX terminal (termios/tty); not supported here.")
    if not sys.stdin.isatty():
        raise SystemExit("--interactive requires an interactive terminal (stdin is not a TTY).")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    state = TeleopState(
        speed=args.speed,
        turn_speed=args.turn_speed,
        max_speed=args.max_speed,
        accel=args.accel,
        turn_accel=args.turn_accel,
    )
    held: dict[str, float] = {}
    reader = _KeyReader(fd)

    print(_HELP_TEXT)
    try:
        tty.setcbreak(fd)
        reader.start()
        while True:
            now = time.monotonic()
            quit_requested = False
            for ch in reader.drain():
                if _handle_key(ch, conn, reader, fd, old_settings, state, held, now, args):
                    quit_requested = True
                    break
            if quit_requested:
                break

            for key in list(held):
                if now - held[key] > _HOLD_WINDOW:
                    del held[key]

            twist = _twist_from_held(held, state)
            if twist is not None:
                conn.move(twist, duration=0)

            time.sleep(_TICK)
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        conn.stop_movement()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print("\nExited interactive control.")


def interactive_go2(args: argparse.Namespace) -> None:
    if args.speed <= 0:
        raise SystemExit("--speed must be greater than zero.")

    conn = _connect(args)
    try:
        prepare_locomotion(conn, args)
        _run_interactive_loop(conn, args)
    finally:
        conn.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Direct Go2 movement, and live keyboard/command control, over Unitree WebRTC."
    )
    parser.add_argument(
        "--dimos-dir", type=Path, default=Path(__file__).resolve().parent / "dimos"
    )
    parser.add_argument("--ip", default=None)
    parser.add_argument("--aes-key", default=None)
    parser.add_argument(
        "--openrouter-key",
        default=None,
        help="OpenRouter API key for the 'ai <instruction>' command. Falls back to "
        "OPENROUTER_API_KEY in dimos/.env or the environment.",
    )
    parser.add_argument("--forward", type=float, default=0.3, help="Approximate forward meters.")
    parser.add_argument("--left", type=float, default=0.0, help="Approximate left meters.")
    parser.add_argument("--yaw", type=float, default=0.0, help="Yaw speed in rad/s.")
    parser.add_argument("--speed", type=float, default=0.25, help="Linear speed in m/s.")
    parser.add_argument("--duration", type=float, default=None, help="Override movement duration.")
    parser.add_argument("--min-duration", type=float, default=0.2)
    parser.add_argument(
        "--control-api",
        choices=("velocity", "joystick"),
        default="velocity",
        help="Use direct walking velocity commands or virtual joystick commands.",
    )
    parser.add_argument("--stand-settle", type=float, default=3.0)
    parser.add_argument("--free-walk-settle", type=float, default=2.0)
    parser.add_argument("--mode-settle", type=float, default=0.5)
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Live keyboard (WASD/QE) + typed-command control, with trick commands (jump, flip, ...).",
    )
    parser.add_argument(
        "--turn-speed", type=float, default=0.8, help="Yaw rate (rad/s) for A/D turning in --interactive mode."
    )
    parser.add_argument(
        "--max-speed", type=float, default=3.0, help="Upper bound for +/- speed adjustment in --interactive mode."
    )
    parser.add_argument(
        "--obstacle-avoidance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the robot's onboard obstacle avoidance (default: on) so it stops itself "
        "near walls/obstacles instead of only relying on the operator.",
    )
    parser.add_argument(
        "--accel",
        type=float,
        default=1.5,
        help="Linear acceleration (m/s^2) used to ramp velocity smoothly in --interactive mode.",
    )
    parser.add_argument(
        "--turn-accel",
        type=float,
        default=3.0,
        help="Angular acceleration (rad/s^2) used to ramp turning smoothly in --interactive mode.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.interactive:
        interactive_go2(args)
    else:
        move_go2(args)


if __name__ == "__main__":
    main()
