import argparse
import asyncio
import shlex
import subprocess
import sys
import time
from pathlib import Path

import collect_gesture_set
import collect_imu
import gesture_detector
import ring_sound as sdk


SCRIPT_DIR = Path(__file__).resolve().parent


def classify(path: Path, model: Path) -> dict[str, object]:
    result = gesture_detector.classify_file(path, model)
    gesture_detector.print_classification(result)
    return result


def trigger_dimos_relative_move(args: argparse.Namespace) -> None:
    command = [
        *shlex.split(args.dimos_cmd),
        "mcp",
        "call",
        "relative_move",
        "--json-args",
        (
            "{"
            f'"forward": {args.dimos_forward}, '
            f'"left": {args.dimos_left}, '
            f'"degrees": {args.dimos_degrees}'
            "}"
        ),
    ]
    print("+", " ".join(shlex.quote(part) for part in command))
    completed = subprocess.run(
        command,
        cwd=args.dimos_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.stderr.strip():
        print(completed.stderr.strip())
    if completed.returncode != 0:
        print(f"DimOS command failed with exit code {completed.returncode}")


def trigger_go2_direct_move(args: argparse.Namespace) -> None:
    command = [
        args.python_cmd,
        str(Path(__file__).with_name("direct_go2_move.py")),
        "--dimos-dir",
        str(args.dimos_dir),
        "--forward",
        str(args.go2_forward),
        "--left",
        str(args.go2_left),
        "--speed",
        str(args.go2_speed),
    ]
    if args.go2_duration is not None:
        command.extend(["--duration", str(args.go2_duration)])
    if args.go2_ip:
        command.extend(["--ip", args.go2_ip])
    if args.go2_aes_key:
        command.extend(["--aes-key", args.go2_aes_key])

    print("+", " ".join(shlex.quote(part) for part in command))
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.stderr.strip():
        print(completed.stderr.strip())
    if completed.returncode != 0:
        print(f"Go2 direct command failed with exit code {completed.returncode}")


def maybe_trigger_dimos(
    result: dict[str, object],
    args: argparse.Namespace,
    last_sos_at: float | None,
) -> float | None:
    if result["decision"] != "sos_shake_hand":
        return last_sos_at
    if not args.dimos_on_sos and not args.go2_direct_on_sos:
        print(
            "SOS detected. Robot trigger is disabled; pass --go2-direct-on-sos "
            "or --dimos-on-sos to move the robot."
        )
        return last_sos_at

    now = time.monotonic()
    if last_sos_at is not None and now - last_sos_at < args.sos_cooldown:
        remaining = args.sos_cooldown - (now - last_sos_at)
        print(f"SOS detected, skipping DimOS trigger during cooldown ({remaining:.1f}s left).")
        return last_sos_at

    if args.go2_direct_on_sos:
        print("SOS detected. Triggering direct Go2 move.")
        trigger_go2_direct_move(args)
    if args.dimos_on_sos:
        print("SOS detected. Triggering DimOS relative_move.")
        trigger_dimos_relative_move(args)
    return now


def latest_recording(raw_dir: Path) -> Path | None:
    files = sorted(raw_dir.glob("test_*.csv"), key=lambda path: path.stat().st_mtime)
    return files[-1] if files else None


async def live_demo(args: argparse.Namespace) -> None:
    ring = await collect_gesture_set.connect_first_gesture_ring(args)
    last_sos_at: float | None = None
    try:
        print()
        print("Live demo connected. Commands: Enter=test, q=quit")
        while True:
            command = input("demo> ").strip().lower()
            if command in {"q", "quit", "exit"}:
                break

            output = args.output_dir / f"demo_{int(time.time() * 1000)}.csv"
            await sdk.start_sensor_report(ring, timeout_s=args.timeout)
            try:
                rows = await collect_imu.collect_samples_to_csv(
                    ring,
                    output,
                    seconds=args.seconds,
                    batches=None,
                    timeout=args.timeout,
                )
            finally:
                if ring.is_connected:
                    try:
                        await sdk.stop_sensor_report(ring, timeout_s=args.timeout)
                    except sdk.RingSoundError as exc:
                        print(f"skip stop_sensor_report: {exc}")

            print(f"saved {rows} samples to {output}")
            result = classify(output, args.model)
            last_sos_at = maybe_trigger_dimos(result, args, last_sos_at)
    finally:
        await ring.disconnect()


def fallback_demo(args: argparse.Namespace) -> None:
    sample = args.fallback_input or latest_recording(args.output_dir)
    if sample is None:
        raise SystemExit("No fallback CSV found. Pass --fallback-input path/to/file.csv.")
    print(f"Fallback demo using {sample}")
    result = classify(sample, args.model)
    maybe_trigger_dimos(result, args, None)


async def main_async(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.fallback:
        fallback_demo(args)
        return

    try:
        await live_demo(args)
    except Exception as exc:
        print(f"Live demo failed: {type(exc).__name__}: {exc}")
        if args.auto_fallback:
            fallback_demo(args)
        else:
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple live/fallback SOS gesture demo.")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--model", type=Path, default=SCRIPT_DIR / "gesture_model.json")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "gesture_raw")
    parser.add_argument("--scan-timeout", type=float, default=30.0)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--phone-mac")
    parser.add_argument("--address", default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--scan-retries", type=int, default=5)
    parser.add_argument("--connect-retries", type=int, default=1)
    parser.add_argument("--start-retries", type=int, default=1)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--mode-timeout", type=float, default=60.0)
    parser.add_argument("--min-rssi", type=int, default=-70)
    parser.add_argument("--fallback", action="store_true", help="Classify a prerecorded CSV instead of connecting.")
    parser.add_argument("--auto-fallback", action="store_true", help="Use fallback if live BLE fails.")
    parser.add_argument("--fallback-input", type=Path)
    parser.add_argument("--dimos-on-sos", action="store_true", help="Call DimOS relative_move when SOS is detected.")
    parser.add_argument("--dimos-dir", type=Path, default=Path("/Users/landyhuang/Documents/dimos"))
    parser.add_argument("--dimos-cmd", default="uv run dimos", help="Command prefix used to run DimOS CLI.")
    parser.add_argument("--dimos-forward", type=float, default=2.0)
    parser.add_argument("--dimos-left", type=float, default=0.0)
    parser.add_argument("--dimos-degrees", type=float, default=0.0)
    parser.add_argument("--go2-direct-on-sos", action="store_true", help="Directly command Go2 over WebRTC when SOS is detected.")
    parser.add_argument("--python-cmd", default=sys.executable)
    parser.add_argument("--go2-ip", default=None, help="Defaults to ROBOT_IP from dimos/.env.")
    parser.add_argument("--go2-aes-key", default=None, help="Defaults to UNITREE_AES_128_KEY from dimos/.env.")
    parser.add_argument("--go2-forward", type=float, default=0.3)
    parser.add_argument("--go2-left", type=float, default=0.0)
    parser.add_argument("--go2-speed", type=float, default=0.25)
    parser.add_argument("--go2-duration", type=float, default=None)
    parser.add_argument("--sos-cooldown", type=float, default=8.0)
    parser.set_defaults(wait_single_press=False)
    return parser


def main() -> None:
    asyncio.run(main_async(build_parser().parse_args()))


if __name__ == "__main__":
    main()
