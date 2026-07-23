import argparse
import asyncio
import csv
import time
from pathlib import Path

import ring_sound as sdk

RingDevice = tuple[object, str, str | None, int | None, bool]


def address_matches(found: str, expected: str | None) -> bool:
    if expected is None:
        return True
    return found.strip().lower() == expected.strip().lower()


def mac_bytes(value: str | None) -> bytes | None:
    if not value:
        return None
    compact = value.replace(":", "").replace("-", "").strip()
    if len(compact) != 12:
        raise SystemExit(f"Invalid phone MAC: {value!r}")
    try:
        return bytes.fromhex(compact)
    except ValueError as exc:
        raise SystemExit(f"Invalid phone MAC: {value!r}") from exc


def adv_contains_phone_mac(device: object, adv: object, phone_mac: str | None) -> bool:
    target = mac_bytes(phone_mac)
    if target is None:
        return False

    chunks = [
        str(getattr(device, "address", "")).encode("utf-8", errors="ignore"),
        str(getattr(device, "name", "")).encode("utf-8", errors="ignore"),
        str(getattr(adv, "local_name", "")).encode("utf-8", errors="ignore"),
    ]
    chunks.extend(str(uuid).encode("utf-8", errors="ignore") for uuid in getattr(adv, "service_uuids", []))

    for company_id, payload in getattr(adv, "manufacturer_data", {}).items():
        chunks.append(int(company_id).to_bytes(2, "little", signed=False))
        chunks.append(bytes(payload))
    for uuid, payload in getattr(adv, "service_data", {}).items():
        chunks.append(str(uuid).encode("utf-8", errors="ignore"))
        chunks.append(bytes(payload))

    haystack = b"".join(chunks).lower()
    text_target = phone_mac.lower().encode("utf-8")
    return target in haystack or target[::-1] in haystack or text_target in haystack


async def scan_ring_devices(scan_timeout: float, phone_mac: str | None = None) -> list[RingDevice]:
    try:
        from bleak import BleakScanner
    except ImportError as exc:
        raise SystemExit("Install bleak first: python -m pip install bleak") from exc

    found = await BleakScanner.discover(timeout=scan_timeout, return_adv=True)
    devices: list[RingDevice] = []
    for device, adv in found.values():
        uuids = [str(uuid).upper() for uuid in getattr(adv, "service_uuids", [])]
        name = getattr(device, "name", None) or getattr(adv, "local_name", None)
        is_named_ring = (name or "").strip().lower() == "ring"
        if not is_named_ring:
            continue
        address = str(getattr(device, "address", ""))
        rssi = getattr(adv, "rssi", None)
        phone_mac_match = adv_contains_phone_mac(device, adv, phone_mac)
        devices.append((device, address, name, rssi, phone_mac_match))
    return devices


def print_devices(devices: list[RingDevice]) -> None:
    print("NUS devices:")
    for idx, (_device, address, name, rssi, phone_mac_match) in enumerate(devices):
        label = name or "(no name)"
        rssi_text = "" if rssi is None else f" rssi={rssi}"
        phone_mac_text = " phone-mac-match" if phone_mac_match else ""
        print(f"  [{idx}] {address} {label}{rssi_text}{phone_mac_text}")


async def choose_device(
    address: str | None,
    scan_timeout: float,
    index: int | None,
    scan_retries: int,
    retry_delay: float,
    phone_mac: str | None,
) -> tuple[object | None, str]:
    devices: list[RingDevice] = []
    for attempt in range(1, scan_retries + 1):
        print(f"Scanning for ring ({attempt}/{scan_retries})...")
        devices = await scan_ring_devices(scan_timeout, phone_mac)
        if address:
            matched = [device for device in devices if address_matches(device[1], address)]
            if matched:
                devices = matched
                break
            if devices:
                print("Requested address was not seen, but these devices named 'ring' were found:")
                print_devices(devices)
        elif phone_mac:
            matched = [device for device in devices if device[4]]
            if matched:
                devices = matched
                break
            if devices:
                print("Ring-like devices were found, but none exposed that phone MAC:")
                print_devices(devices)
        elif devices:
            break
        if attempt < scan_retries:
            await asyncio.sleep(retry_delay)

    if not devices and address:
        print(
            f"Requested address {address} was not seen during scanning; "
            "trying direct CoreBluetooth connect."
        )
        return None, address

    if not devices:
        raise SystemExit(
            "No device named 'ring' found during scanning. Keep the ring nearby/awake, "
            "make sure it is not connected to another app, then re-run."
        )

    if address:
        print(
            f"Requested address {address} was not found in the scan; "
            "trying direct CoreBluetooth connect."
        )
        return None, address

    if phone_mac:
        matched = [device for device in devices if device[4]]
        if matched:
            devices = matched
        else:
            named_ring_devices = [
                device
                for device in devices
                if (device[2] or "").strip().lower() == "ring"
            ]
            if not named_ring_devices:
                raise SystemExit(
                    f"No scanned ring exposed phone MAC {phone_mac}. On macOS, Bleak usually "
                    "shows a CoreBluetooth UUID instead of the real BLE MAC, so this phone "
                    "address may not be usable from Python."
                )
            print(
                f"No scanned ring exposed phone MAC {phone_mac}; "
                "falling back to the strongest device named 'ring'."
            )
            devices = named_ring_devices

    devices.sort(key=lambda item: item[3] if item[3] is not None else -999, reverse=True)
    print_devices(devices)
    if index is None:
        index = 0
        if len(devices) > 1:
            print("Using device index 0 by default.")
    elif index == -1 and len(devices) > 1:
        raw_index = input("Select device index: ").strip()
        try:
            index = int(raw_index)
        except ValueError as exc:
            raise SystemExit(f"Invalid device index: {raw_index!r}") from exc

    if index < 0 or index >= len(devices):
        raise SystemExit(f"Device index out of range: {index}")

    device, selected_address, _name, _rssi, _phone_mac_match = devices[index]
    return device, selected_address


async def connect_ring(args: argparse.Namespace) -> tuple[sdk.RingSoundClient, sdk.NusClient]:
    device, address = await choose_device(
        args.address,
        args.scan_timeout,
        args.index,
        args.scan_retries,
        args.retry_delay,
        args.phone_mac,
    )

    transport = sdk.NusClient(address=address, scan_timeout_s=args.scan_timeout)
    if device is None:
        ring = sdk.RingSoundClient(transport=transport)
        await ring.connect()
        return ring, transport

    transport.address = address
    transport_device = device

    async def connect_from_current_scan() -> None:
        try:
            from bleak import BleakClient
        except ImportError as exc:
            raise sdk.TransportError("Install bleak to use BLE transport") from exc

        last_exc: Exception | None = None
        for attempt in range(1, args.connect_retries + 1):
            print(f"Connecting to ring {address} ({attempt}/{args.connect_retries})...")
            transport._client = BleakClient(
                transport_device,
                disconnected_callback=transport._handle_disconnect,
            )
            try:
                await transport._client.connect()
                await transport._client.start_notify(transport.tx_uuid, transport._handle_notify)
                transport._notify_started = True
                return
            except Exception as exc:
                last_exc = exc
                transport._client = None
                if attempt >= args.connect_retries:
                    break
                await asyncio.sleep(args.retry_delay)

        if last_exc is not None:
            transport._client = None
            raise sdk.TransportError(
                f"BLE connect failed for address={address!r} from current scan: "
                f"{type(last_exc).__name__}: {last_exc}"
            ) from last_exc
        raise sdk.TransportError(f"BLE connect failed for address={address!r}")

    transport.connect = connect_from_current_scan
    ring = sdk.RingSoundClient(transport=transport)
    await ring.connect()
    return ring, transport


async def wait_for_single_press(
    ring: sdk.RingSoundClient,
    prompt: str,
    mode_timeout: float,
) -> None:
    print(prompt)
    try:
        event = await sdk.wait_sensor_key_single_press_event(
            ring,
            timeout_s=mode_timeout,
        )
        print(f"single press event timestamp_ms={event.timestamp_ms}")
    except sdk.TimeoutError:
        print(
            "single press event timed out; trying to start IMU anyway "
            "in case the ring is already in gesture mode."
        )


async def start_sensor_with_mode_retry(
    ring: sdk.RingSoundClient,
    args: argparse.Namespace,
) -> sdk.SensorStartInfo:
    if args.wait_single_press:
        await wait_for_single_press(
            ring,
            "Press the ring button once to switch into gesture mode...",
            args.mode_timeout,
        )

    for start_attempt in range(1, args.start_retries + 1):
        try:
            start = await sdk.start_sensor_report(ring)
            print(
                "sensor started: "
                f"{start.sample_rate_hz}Hz, accel=+/-{start.accel_range_g}g, "
                f"gyro=+/-{start.gyro_range_dps}dps"
            )
            return start
        except sdk.DeviceError as exc:
            if exc.error_code != int(sdk.ErrorCode.DEVICE_BUSY):
                raise
            if not args.wait_single_press:
                raise sdk.DeviceError(
                    exc.error_code,
                    "device busy; selected ring is not in gesture mode or is the wrong ring",
                ) from exc
            if start_attempt >= args.start_retries:
                raise
            await wait_for_single_press(
                ring,
                "Ring is busy, likely still in recording mode. "
                "Press the ring button once again to toggle modes...",
                args.mode_timeout,
            )
            await asyncio.sleep(0.5)

    raise sdk.DeviceError(int(sdk.ErrorCode.DEVICE_BUSY), "device busy")


async def collect_samples_to_csv(
    ring: sdk.RingSoundClient,
    output_path: Path,
    *,
    seconds: float,
    batches: int | None,
    timeout: float,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    drain_queue = getattr(ring, "_drain_queue", None)
    if callable(drain_queue):
        drain_queue(int(sdk.SensorCommand.DATA_FRAME))

    deadline = None if seconds <= 0 else time.monotonic() + seconds
    batches_left = batches
    rows_written = 0

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "sequence",
                "timestamp_ms",
                "accel_x",
                "accel_y",
                "accel_z",
                "gyro_x",
                "gyro_y",
                "gyro_z",
            ]
        )

        while True:
            if deadline is not None and time.monotonic() >= deadline:
                break
            if batches_left is not None and batches_left <= 0:
                break

            batch = await sdk.wait_sensor_data(ring, timeout_s=timeout)
            for index, sample in enumerate(batch.samples):
                writer.writerow(
                    [
                        batch.sequence_start + index,
                        sample.timestamp_ms,
                        sample.accel_x,
                        sample.accel_y,
                        sample.accel_z,
                        sample.gyro_x,
                        sample.gyro_y,
                        sample.gyro_z,
                    ]
                )
                rows_written += 1

            if batches_left is not None:
                batches_left -= 1

    return rows_written


async def collect(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    ring: sdk.RingSoundClient | None = None
    try:
        ring, _transport = await connect_ring(args)
        await start_sensor_with_mode_retry(ring, args)
        try:
            rows_written = await collect_samples_to_csv(
                ring,
                output_path,
                seconds=args.seconds,
                batches=args.batches,
                timeout=args.timeout,
            )
        finally:
            await sdk.stop_sensor_report(ring)
    except sdk.TransportError as exc:
        raise SystemExit(
            f"BLE connection failed: {exc}\n"
            "Run without --address to use the strongest device named 'ring'."
        ) from exc
    except sdk.DeviceError as exc:
        raise SystemExit(
            f"Device rejected the IMU command: {exc}\n"
            "Make sure the ring is in gesture mode. If it just powered on, run without "
            "`--no-wait-single-press` and press the ring button once when prompted."
        ) from exc
    finally:
        if ring is not None:
            await ring.disconnect()

    print(f"saved {rows_written} samples to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect Ring Sound IMU data to CSV.")
    parser.add_argument("--address", help="Ring BLE MAC address. If omitted, scan first.")
    parser.add_argument(
        "--phone-mac",
        help="BLE MAC shown by the phone app, for example FE:77:CE:08:AB:ED.",
    )
    parser.add_argument(
        "--index",
        type=int,
        help="Device index from the scan list. Defaults to 0; use -1 to choose interactively.",
    )
    parser.add_argument("--output", default="imu_0605.csv", help="CSV output path.")
    parser.add_argument("--seconds", type=float, default=10.0, help="Collect duration. Use 0 with --batches.")
    parser.add_argument("--batches", type=int, default=None, help="Collect this many 0x0605 batches.")
    parser.add_argument("--timeout", type=float, default=5.0, help="Timeout for each IMU batch.")
    parser.add_argument("--scan-timeout", type=float, default=5.0, help="BLE scan timeout.")
    parser.add_argument("--scan-retries", type=int, default=3, help="Number of scan attempts.")
    parser.add_argument("--connect-retries", type=int, default=3, help="Number of connect attempts.")
    parser.add_argument("--start-retries", type=int, default=3, help="Number of IMU start attempts.")
    parser.add_argument("--retry-delay", type=float, default=2.0, help="Seconds between scan attempts.")
    parser.add_argument("--mode-timeout", type=float, default=60.0, help="Timeout waiting for single press.")
    parser.add_argument(
        "--wait-single-press",
        dest="wait_single_press",
        action="store_true",
        help="Wait for one button press before starting IMU. Use this only if the ring is not already in gesture mode.",
    )
    parser.add_argument(
        "--no-wait-single-press",
        dest="wait_single_press",
        action="store_false",
        help="Deprecated compatibility flag; this is now the default.",
    )
    parser.set_defaults(wait_single_press=False)
    return parser


if __name__ == "__main__":
    asyncio.run(collect(build_parser().parse_args()))
