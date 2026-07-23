import argparse
import os
import sys
import time
from pathlib import Path


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


def require_robot_step(label: str, action) -> None:
    print(f"Preparing Go2: {label}...")
    if not action():
        raise SystemExit(f"Go2 rejected preparation step: {label}")
    print(f"Preparing Go2: {label} accepted.")


def prepare_locomotion(conn, args: argparse.Namespace) -> None:
    require_robot_step("stand up", conn.standup)
    time.sleep(args.stand_settle)

    if args.control_api == "velocity":
        require_robot_step("free walk", conn.free_walk)
        time.sleep(args.free_walk_settle)
        return

    require_robot_step("balance stand", conn.balance_stand)
    time.sleep(args.mode_settle)
    require_robot_step("enable joystick", lambda: conn.switch_joystick(True))
    time.sleep(args.mode_settle)


def move_go2(args: argparse.Namespace) -> None:
    add_dimos_to_path(args.dimos_dir)
    load_env_file(args.dimos_dir / ".env")

    from dimos.msgs.geometry_msgs.Twist import Twist
    from dimos.msgs.geometry_msgs.Vector3 import Vector3
    from dimos.robot.unitree.connection import UnitreeWebRTCConnection

    ip = args.ip or os.environ.get("ROBOT_IP")
    aes_key = args.aes_key or os.environ.get("UNITREE_AES_128_KEY")
    if not ip:
        raise SystemExit("Missing robot IP. Pass --ip or set ROBOT_IP.")
    if not aes_key:
        raise SystemExit("Missing AES key. Pass --aes-key or set UNITREE_AES_128_KEY.")
    if args.speed <= 0:
        raise SystemExit("--speed must be greater than zero.")

    distance = max(abs(args.forward), abs(args.left))
    duration = args.duration
    if duration is None:
        duration = max(args.min_duration, distance / args.speed) if distance > 0 else 0.2

    x_speed = 0.0 if args.forward == 0 else args.speed * (1 if args.forward > 0 else -1)
    y_speed = 0.0 if args.left == 0 else args.speed * (1 if args.left > 0 else -1)
    yaw_speed = args.yaw

    print(
        "Connecting Go2 "
        f"ip={ip} forward={args.forward:.2f}m left={args.left:.2f}m "
        f"speed={args.speed:.2f}m/s duration={duration:.2f}s "
        f"control_api={args.control_api}"
    )
    conn = UnitreeWebRTCConnection(
        ip=ip,
        aes_128_key=aes_key,
        velocity_api=args.control_api == "velocity",
    )
    try:
        prepare_locomotion(conn, args)

        twist = Twist(linear=Vector3(x_speed, y_speed, 0.0), angular=Vector3(0.0, 0.0, yaw_speed))
        print("Sending continuous movement commands...")
        if not conn.move(twist, duration=duration):
            raise SystemExit("Go2 move command failed.")
        time.sleep(0.2)
        conn.stop_movement()
        print("Go2 movement commands sent and stop command issued.")
    finally:
        conn.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Direct short Go2 movement over Unitree WebRTC.")
    parser.add_argument("--dimos-dir", type=Path, default=Path("/Users/landyhuang/Documents/dimos"))
    parser.add_argument("--ip", default=None)
    parser.add_argument("--aes-key", default=None)
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
    return parser


def main() -> None:
    move_go2(build_parser().parse_args())


if __name__ == "__main__":
    main()
