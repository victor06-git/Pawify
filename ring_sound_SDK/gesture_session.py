import argparse
import asyncio
import subprocess
import sys
import time
from pathlib import Path

import collect_gesture_set
import collect_imu
import ring_sound as sdk


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


async def stop_sensor_if_connected(ring: sdk.RingSoundClient, timeout: float) -> None:
    if not ring.is_connected:
        return
    try:
        await sdk.stop_sensor_report(ring, timeout_s=timeout)
    except sdk.RingSoundError as exc:
        print(f"skip stop_sensor_report: {exc}")


async def collect_label(
    ring: sdk.RingSoundClient,
    *,
    label: str,
    count: int,
    seconds: float,
    timeout: float,
    output_dir: Path,
    dataset: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, count + 1):
        print()
        print(f"{label} sample {index}/{count}")
        input("Press Enter, then perform the gesture during the collection window...")

        output = output_dir / f"{label}_{index:02d}_{int(time.time() * 1000)}.csv"
        await sdk.start_sensor_report(ring, timeout_s=timeout)
        try:
            rows = await collect_imu.collect_samples_to_csv(
                ring,
                output,
                seconds=seconds,
                batches=None,
                timeout=timeout,
            )
        finally:
            await stop_sensor_if_connected(ring, timeout)
        print(f"saved {rows} samples to {output}")
        run(
            [
                sys.executable,
                "gesture_detector.py",
                "add-template",
                "--gesture",
                label,
                "--input",
                str(output),
                "--dataset",
                str(dataset),
            ]
        )


def train(dataset: Path, model: Path) -> None:
    run(
        [
            sys.executable,
            "gesture_detector.py",
            "train",
            "--dataset",
            str(dataset),
            "--model",
            str(model),
        ]
    )


async def test_once(
    ring: sdk.RingSoundClient,
    *,
    seconds: float,
    timeout: float,
    output_dir: Path,
    model: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"test_{int(time.time() * 1000)}.csv"
    input("Press Enter, then perform the gesture during the test window...")
    await sdk.start_sensor_report(ring, timeout_s=timeout)
    try:
        rows = await collect_imu.collect_samples_to_csv(
            ring,
            output,
            seconds=seconds,
            batches=None,
            timeout=timeout,
        )
    finally:
        await stop_sensor_if_connected(ring, timeout)
    print(f"saved {rows} samples to {output}")
    run(
        [
            sys.executable,
            "gesture_detector.py",
            "classify",
            "--input",
            str(output),
            "--model",
            str(model),
        ]
    )


async def main_async(args: argparse.Namespace) -> None:
    print("Finding one ring that accepts IMU start...")
    ring = await collect_gesture_set.connect_first_gesture_ring(args)
    sensor_started = False
    await sdk.stop_sensor_report(ring, timeout_s=args.timeout)

    try:
        print()
        print("Connected. Commands:")
        print("  collect <label> [count] [seconds]")
        print("  test [seconds]")
        print("  train")
        print("  quit")
        print()
        print("Examples:")
        print("  collect sos_shake_hand 10 3")
        print("  collect idle 5 3")

        while True:
            raw = input("gesture-session> ").strip()
            if not raw:
                continue
            parts = raw.split()
            command = parts[0].lower()

            if command in {"quit", "exit", "q"}:
                break
            if command == "train":
                train(args.dataset, args.model)
                continue
            if command == "test":
                seconds = float(parts[1]) if len(parts) >= 2 else args.seconds
                try:
                    await test_once(
                        ring,
                        seconds=seconds,
                        timeout=args.timeout,
                        output_dir=args.output_dir,
                        model=args.model,
                    )
                except sdk.TransportError as exc:
                    print(f"BLE disconnected: {exc}")
                    print("Restart gesture_session.py to reconnect.")
                    break
                continue
            if command != "collect":
                print("unknown command")
                continue
            if len(parts) < 2:
                print("usage: collect <label> [count] [seconds]")
                continue

            label = parts[1]
            count = int(parts[2]) if len(parts) >= 3 else args.count
            seconds = float(parts[3]) if len(parts) >= 4 else args.seconds
            try:
                await collect_label(
                    ring,
                    label=label,
                    count=count,
                    seconds=seconds,
                    timeout=args.timeout,
                    output_dir=args.output_dir,
                    dataset=args.dataset,
                )
                train(args.dataset, args.model)
            except sdk.TransportError as exc:
                print(f"BLE disconnected: {exc}")
                print("Restart gesture_session.py to reconnect.")
                break
    finally:
        try:
            if sensor_started:
                await stop_sensor_if_connected(ring, args.timeout)
        except sdk.RingSoundError as exc:
            print(f"skip stop_sensor_report: {exc}")
        finally:
            await ring.disconnect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive one-connection gesture data collection session.")
    parser.add_argument("--count", type=int, default=5, help="Default sample count per collect command.")
    parser.add_argument("--seconds", type=float, default=3.0, help="Default seconds per sample.")
    parser.add_argument("--dataset", type=Path, default=Path("gesture_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("gesture_raw"))
    parser.add_argument("--model", type=Path, default=Path("gesture_model.json"))
    parser.add_argument("--phone-mac", help="BLE MAC shown by the phone app.")
    parser.add_argument("--scan-timeout", type=float, default=15.0)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--address", default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--scan-retries", type=int, default=5)
    parser.add_argument("--connect-retries", type=int, default=1)
    parser.add_argument("--start-retries", type=int, default=1)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--mode-timeout", type=float, default=60.0)
    parser.add_argument("--min-rssi", type=int, default=-70)
    parser.set_defaults(wait_single_press=False)
    return parser


def main() -> None:
    asyncio.run(main_async(build_parser().parse_args()))


if __name__ == "__main__":
    main()
