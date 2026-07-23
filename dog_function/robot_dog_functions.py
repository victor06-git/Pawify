"""Convenience functions for a phone app that controls a DimOS robot dog.

The functions are intentionally small and import DimOS lazily, so this file can
live in a separate app repo and still fail clearly when DimOS is not installed.

Typical usage:

    from pawify.robot_dog_functions import emergency_stop, come_find_me

    emergency_stop()
    come_find_me()

To expose HTTP endpoints for a phone app:

    uvicorn pawify.robot_dog_functions:create_app --factory --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
import subprocess
import threading
import time
from typing import Any


DEFAULT_CMD_VEL_TOPIC = "/cmd_vel"
DEFAULT_CAMERA_TOPIC = "/color_image"
DEFAULT_FIND_ME_PROMPT = (
    "Follow the person who pressed the emergency button and stop at a safe distance."
)
MJPEG_BOUNDARY = "frame"


@dataclass(frozen=True)
class VelocityCommand:
    """Ground robot velocity command in DimOS/ROS Twist terms."""

    forward: float = 0.0
    left: float = 0.0
    yaw: float = 0.0


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp a number to an inclusive range."""
    return max(min_value, min(max_value, float(value)))


def make_twist(
    forward: float = 0.0,
    left: float = 0.0,
    yaw: float = 0.0,
    *,
    max_linear: float = 0.8,
    max_angular: float = 1.2,
) -> Any:
    """Build a bounded DimOS Twist message.

    Args:
        forward: Forward velocity in m/s.
        left: Left/strafe velocity in m/s.
        yaw: Yaw velocity in rad/s.
        max_linear: Absolute limit for x/y velocity.
        max_angular: Absolute limit for yaw velocity.
    """
    from dimos.msgs.geometry_msgs.Twist import Twist
    from dimos.msgs.geometry_msgs.Vector3 import Vector3

    return Twist(
        linear=Vector3(
            x=clamp(forward, -max_linear, max_linear),
            y=clamp(left, -max_linear, max_linear),
            z=0.0,
        ),
        angular=Vector3(
            x=0.0,
            y=0.0,
            z=clamp(yaw, -max_angular, max_angular),
        ),
    )


def make_stop_twist() -> Any:
    """Build a zero-velocity DimOS Twist message."""
    from dimos.msgs.geometry_msgs.Twist import Twist

    return Twist.zero()


def make_cmd_vel_publisher(topic: str = DEFAULT_CMD_VEL_TOPIC) -> Callable[[Any], None]:
    """Create a function that publishes Twist messages to a DimOS LCM topic."""
    from dimos.core.transport import LCMTransport
    from dimos.msgs.geometry_msgs.Twist import Twist

    transport = LCMTransport(topic, Twist)
    transport.start()

    def publish(twist: Any) -> None:
        transport.broadcast(None, twist)

    return publish


def publish_cmd_vel(
    forward: float = 0.0,
    left: float = 0.0,
    yaw: float = 0.0,
    *,
    topic: str = DEFAULT_CMD_VEL_TOPIC,
    max_linear: float = 0.8,
    max_angular: float = 1.2,
) -> Any:
    """Publish one bounded velocity command to the robot."""
    twist = make_twist(
        forward,
        left,
        yaw,
        max_linear=max_linear,
        max_angular=max_angular,
    )
    make_cmd_vel_publisher(topic)(twist)
    return twist


def emergency_stop(
    *,
    topic: str = DEFAULT_CMD_VEL_TOPIC,
    repeat: int = 5,
    interval_s: float = 0.05,
    publisher: Callable[[Any], None] | None = None,
) -> None:
    """Stop the robot by publishing repeated zero velocity commands.

    Repeating the stop command makes the call robust against brief LCM packet
    loss and trips downstream command-timeout/deadman behavior quickly.
    """
    publish = publisher or make_cmd_vel_publisher(topic)
    stop_twist = make_stop_twist()
    for index in range(max(1, int(repeat))):
        publish(stop_twist)
        if index < repeat - 1 and interval_s > 0:
            time.sleep(interval_s)


def send_agent_command(prompt: str, *, timeout_s: float = 10.0) -> str:
    """Send a natural-language command to a running DimOS agentic stack.

    Requires a stack such as `dimos run unitree-go2-agentic --daemon`.
    """
    result = subprocess.run(
        ["dimos", "agent-send", prompt],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return (result.stdout or result.stderr).strip()


def come_find_me(
    prompt: str = DEFAULT_FIND_ME_PROMPT,
    *,
    stop_first: bool = True,
    timeout_s: float = 10.0,
) -> str:
    """Ask the robot to find/follow the user through the running DimOS agent.

    This is the fastest integration path for the phone emergency button because
    DimOS already ships an agentic Go2 stack with person-follow skills.
    """
    if stop_first:
        emergency_stop()
    return send_agent_command(prompt, timeout_s=timeout_s)


def encode_camera_jpeg(
    image: Any,
    *,
    quality: int = 75,
    max_width: int | None = 960,
    max_height: int | None = 540,
) -> bytes:
    """Encode a DimOS Image as JPEG bytes for browser/mobile display."""
    import cv2
    import numpy as np

    bgr_image = image.to_bgr().to_opencv()
    height, width = bgr_image.shape[:2]

    scale = 1.0
    if max_width is not None and width > max_width:
        scale = min(scale, max_width / width)
    if max_height is not None and height > max_height:
        scale = min(scale, max_height / height)

    if scale < 1.0:
        resized = (max(1, round(width * scale)), max(1, round(height * scale)))
        bgr_image = cv2.resize(bgr_image, resized, interpolation=cv2.INTER_AREA)

    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(clamp(quality, 1, 100))]
    ok, encoded = cv2.imencode(".jpg", np.ascontiguousarray(bgr_image), params)
    if not ok:
        raise ValueError("Failed to encode camera image as JPEG")
    return encoded.tobytes()


def format_mjpeg_frame(jpeg_bytes: bytes, *, boundary: str = MJPEG_BOUNDARY) -> bytes:
    """Wrap one JPEG as a multipart MJPEG frame."""
    header = (
        f"--{boundary}\r\n"
        "Content-Type: image/jpeg\r\n"
        f"Content-Length: {len(jpeg_bytes)}\r\n\r\n"
    )
    return header.encode("ascii") + jpeg_bytes + b"\r\n"


class LatestCameraFrame:
    """Thread-safe holder for the latest camera JPEG."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._timestamp_s: float | None = None

    def update(self, image: Any) -> None:
        jpeg = encode_camera_jpeg(image)
        with self._lock:
            self._jpeg = jpeg
            self._timestamp_s = time.time()

    def get(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    @property
    def timestamp_s(self) -> float | None:
        with self._lock:
            return self._timestamp_s


def subscribe_camera(
    frame_store: LatestCameraFrame,
    *,
    topic: str = DEFAULT_CAMERA_TOPIC,
) -> Callable[[], None]:
    """Subscribe to a DimOS camera topic and update `frame_store`."""
    from dimos.core.transport import LCMTransport
    from dimos.msgs.sensor_msgs.Image import Image

    transport = LCMTransport(topic, Image)
    unsubscribe = transport.subscribe(frame_store.update)

    def stop() -> None:
        if unsubscribe:
            unsubscribe()
        transport.stop()

    return stop


def mjpeg_stream(
    frame_store: LatestCameraFrame,
    *,
    fps: float = 12.0,
    boundary: str = MJPEG_BOUNDARY,
) -> Iterator[bytes]:
    """Yield an endless MJPEG byte stream from the latest camera frame."""
    period = 1.0 / max(1.0, fps)
    while True:
        jpeg = frame_store.get()
        if jpeg is not None:
            yield format_mjpeg_frame(jpeg, boundary=boundary)
        time.sleep(period)


def create_app(
    *,
    cmd_vel_topic: str = DEFAULT_CMD_VEL_TOPIC,
    camera_topic: str = DEFAULT_CAMERA_TOPIC,
) -> Any:
    """Create a FastAPI app for a mobile phone UI.

    Routes:
        GET  /health
        GET  /camera.mjpeg
        POST /cmd_vel        JSON: {"forward": 0.2, "left": 0, "yaw": 0}
        POST /emergency
        POST /come-find-me   JSON optional: {"prompt": "..."}
    """
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    class CmdVelRequest(BaseModel):
        forward: float = 0.0
        left: float = 0.0
        yaw: float = 0.0

    class FindMeRequest(BaseModel):
        prompt: str = DEFAULT_FIND_ME_PROMPT

    app = FastAPI(title="Pawify Robot Dog Controls")
    camera = LatestCameraFrame()
    publish = make_cmd_vel_publisher(cmd_vel_topic)
    stop_camera_subscription: Callable[[], None] | None = None

    @app.on_event("startup")
    def _startup() -> None:
        nonlocal stop_camera_subscription
        stop_camera_subscription = subscribe_camera(camera, topic=camera_topic)

    @app.on_event("shutdown")
    def _shutdown() -> None:
        if stop_camera_subscription is not None:
            stop_camera_subscription()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "camera_topic": camera_topic,
            "cmd_vel_topic": cmd_vel_topic,
            "has_camera_frame": camera.get() is not None,
            "camera_timestamp_s": camera.timestamp_s,
        }

    @app.get("/camera.mjpeg")
    def camera_mjpeg() -> StreamingResponse:
        return StreamingResponse(
            mjpeg_stream(camera),
            media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
        )

    @app.post("/cmd_vel")
    def cmd_vel(request: CmdVelRequest) -> dict[str, Any]:
        twist = make_twist(request.forward, request.left, request.yaw)
        publish(twist)
        return {"ok": True, "command": request.model_dump()}

    @app.post("/emergency")
    def emergency() -> dict[str, Any]:
        emergency_stop(publisher=publish)
        return {"ok": True, "stopped": True}

    @app.post("/come-find-me")
    def find_me(request: FindMeRequest) -> dict[str, Any]:
        emergency_stop(publisher=publish)
        output = send_agent_command(request.prompt)
        return {"ok": True, "prompt": request.prompt, "output": output}

    return app


__all__ = [
    "DEFAULT_CAMERA_TOPIC",
    "DEFAULT_CMD_VEL_TOPIC",
    "DEFAULT_FIND_ME_PROMPT",
    "LatestCameraFrame",
    "VelocityCommand",
    "clamp",
    "come_find_me",
    "create_app",
    "emergency_stop",
    "encode_camera_jpeg",
    "format_mjpeg_frame",
    "make_cmd_vel_publisher",
    "make_stop_twist",
    "make_twist",
    "mjpeg_stream",
    "publish_cmd_vel",
    "send_agent_command",
    "subscribe_camera",
]
