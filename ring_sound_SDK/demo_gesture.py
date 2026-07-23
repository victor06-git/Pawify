import argparse
import asyncio
import subprocess
import sys
import time
from pathlib import Path

import collect_gesture_set
import collect_imu
import ring_sound as sdk


def classify(path: Path, model: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "gesture_detector.py",
            "classify",
            "--input",
            str(path),
            "--model",
            str(model),
        ],
        check=True,
    )


def latest_recording(raw_dir: Path) -> Path | None:
    files = sorted(raw_dir.glob("test_*.csv"), key=lambda path: path.stat().st_mtime)
    return files[-1] if files else None


async def live_demo(args: argparse.Namespace) -> None:
    ring = await collect_gesture_set.connect_first_gesture_ring(args)
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
            classify(output, args.model)
    finally:
        await ring.disconnect()


def fallback_demo(args: argparse.Namespace) -> None:
    sample = args.fallback_input or latest_recording(args.output_dir)
    if sample is None:
        raise SystemExit("No fallback CSV found. Pass --fallback-input path/to/file.csv.")
    print(f"Fallback demo using {sample}")
    classify(sample, args.model)


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
    parser.add_argument("--model", type=Path, default=Path("gesture_model.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("gesture_raw"))
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
    parser.set_defaults(wait_single_press=False)
    return parser


def main() -> None:
    asyncio.run(main_async(build_parser().parse_args()))


if __name__ == "__main__":
    main()
