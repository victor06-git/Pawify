import argparse
import asyncio
import subprocess
import sys
import time
from pathlib import Path

import collect_imu
import ring_sound as sdk


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


async def connect_first_gesture_ring(args: argparse.Namespace) -> sdk.RingSoundClient:
    devices = []
    min_rssi = getattr(args, "min_rssi", None)
    scan_retries = max(1, getattr(args, "scan_retries", 1))
    retry_delay = getattr(args, "retry_delay", 2.0)
    for attempt in range(1, scan_retries + 1):
        print(f"Scanning for ring ({attempt}/{scan_retries})...")
        devices = await collect_imu.scan_ring_devices(args.scan_timeout, args.phone_mac)
        if min_rssi is not None:
            strong_devices = [
                device for device in devices if device[3] is None or device[3] >= min_rssi
            ]
            if strong_devices:
                devices = strong_devices
        if devices:
            break
        if attempt < scan_retries:
            await asyncio.sleep(retry_delay)

    devices.sort(key=lambda item: item[3] if item[3] is not None else -999, reverse=True)
    if not devices:
        if min_rssi is None:
            raise SystemExit("No device named 'ring' found.")
        raise SystemExit(
            f"No device named 'ring' found with RSSI >= {min_rssi}. "
            "Move the ring closer to the Mac and try again."
        )

    print("Auto-finding ring that accepts IMU start:")
    last_error: Exception | None = None
    for index, (device, address, name, rssi, _phone_mac_match) in enumerate(devices):
        print(f"  [{index}] probing {address} {name or '(no name)'} rssi={rssi}")
        probe_args = argparse.Namespace(**vars(args))
        probe_args.address = None
        probe_args.index = 0
        probe_args.scan_retries = 1

        transport = sdk.NusClient(address=address, scan_timeout_s=args.scan_timeout)

        async def connect_from_scan() -> None:
            try:
                from bleak import BleakClient
            except ImportError as exc:
                raise sdk.TransportError("Install bleak to use BLE transport") from exc

            transport._client = BleakClient(
                device,
                disconnected_callback=transport._handle_disconnect,
            )
            try:
                await transport._client.connect()
                await transport._client.start_notify(transport.tx_uuid, transport._handle_notify)
                transport._notify_started = True
            except Exception:
                transport._client = None
                raise

        transport.connect = connect_from_scan
        ring = sdk.RingSoundClient(transport=transport)
        try:
            await ring.connect()
            start = await sdk.start_sensor_report(ring, timeout_s=args.timeout)
            print(
                "  OK using gesture-mode ring: "
                f"{address} {start.sample_rate_hz}Hz accel=+/-{start.accel_range_g}g "
                f"gyro=+/-{start.gyro_range_dps}dps"
            )
            return ring
        except Exception as exc:
            last_error = exc
            print(f"  not usable: {type(exc).__name__}: {exc}")
            await ring.disconnect()

    raise SystemExit(f"No scanned ring accepted IMU start. Last error: {last_error}")


async def main_async(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ring: sdk.RingSoundClient | None = None
    sensor_started = False

    try:
        try:
            ring, _transport = await collect_imu.connect_ring(args)
            await collect_imu.start_sensor_with_mode_retry(ring, args)
        except (sdk.TransportError, sdk.DeviceError) as exc:
            if not args.auto_find_gesture_ring:
                raise
            if ring is not None:
                await ring.disconnect()
            print(f"Selected ring failed: {type(exc).__name__}: {exc}")
            ring = await connect_first_gesture_ring(args)
        sensor_started = True

        for index in range(1, args.count + 1):
            print()
            print(f"Sample {index}/{args.count}: get ready to perform {args.gesture}.")
            input("Press Enter, then perform the gesture during the collection window...")

            output = args.output_dir / f"{args.gesture}_{index:02d}_{int(time.time() * 1000)}.csv"
            rows = await collect_imu.collect_samples_to_csv(
                ring,
                output,
                seconds=args.seconds,
                batches=None,
                timeout=args.timeout,
            )
            print(f"saved {rows} samples to {output}")

            run(
                [
                    sys.executable,
                    "gesture_detector.py",
                    "add-template",
                    "--gesture",
                    args.gesture,
                    "--input",
                    str(output),
                    "--dataset",
                    str(args.dataset),
                ]
            )
    finally:
        if ring is not None:
            try:
                if sensor_started and ring.is_connected:
                    await sdk.stop_sensor_report(ring)
            except sdk.RingSoundError as exc:
                print(f"skip stop_sensor_report: {exc}")
            finally:
                await ring.disconnect()

    run(
        [
            sys.executable,
            "gesture_detector.py",
            "train",
            "--dataset",
            str(args.dataset),
            "--model",
            "gesture_model.json",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect repeated labeled gesture samples over one BLE connection.")
    parser.add_argument("--gesture", required=True, help="Gesture label, e.g. sos_shake_hand.")
    parser.add_argument("--count", type=int, default=10, help="Number of samples to collect.")
    parser.add_argument("--seconds", type=float, default=3.0, help="Seconds per sample.")
    parser.add_argument("--dataset", type=Path, default=Path("gesture_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("gesture_raw"))
    parser.add_argument("--address", help="Ring CoreBluetooth UUID/address from a successful scan.")
    parser.add_argument("--phone-mac", help="BLE MAC shown by the phone app.")
    parser.add_argument("--scan-timeout", type=float, default=15.0)
    parser.add_argument("--scan-retries", type=int, default=5)
    parser.add_argument("--min-rssi", type=int, default=-70)
    parser.add_argument("--connect-retries", type=int, default=5)
    parser.add_argument("--start-retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--mode-timeout", type=float, default=60.0)
    parser.add_argument("--timeout", type=float, default=5.0, help="Timeout for each IMU batch.")
    parser.add_argument("--index", type=int, default=None, help="Device index from the scan list.")
    parser.add_argument(
        "--no-auto-find-gesture-ring",
        dest="auto_find_gesture_ring",
        action="store_false",
        help="Disable fallback probing of scanned ring devices that are already in gesture mode.",
    )
    parser.add_argument(
        "--wait-single-press",
        dest="wait_single_press",
        action="store_true",
        help="Wait for one button press before starting IMU. Use this only if the ring is not already in gesture mode.",
    )
    parser.add_argument(
        "--setup-gesture-mode",
        dest="wait_single_press",
        action="store_true",
        help="Alias for --wait-single-press; use once when the ring returns device busy.",
    )
    parser.set_defaults(wait_single_press=False)
    parser.set_defaults(auto_find_gesture_ring=True)
    return parser


def main() -> None:
    asyncio.run(main_async(build_parser().parse_args()))


if __name__ == "__main__":
    main()
