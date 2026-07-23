import argparse
import asyncio

import ring_sound as sdk


async def scan_with_service(timeout_s: float) -> list[tuple[str, str | None, int | None, list[str]]]:
    try:
        from bleak import BleakScanner
    except ImportError as exc:
        raise SystemExit("Install bleak first: python -m pip install bleak") from exc

    found = await BleakScanner.discover(timeout=timeout_s, return_adv=True)
    rows: list[tuple[str, str | None, int | None, list[str]]] = []

    for device, adv in found.values():
        uuids = [str(uuid).upper() for uuid in getattr(adv, "service_uuids", [])]
        if sdk.NUS_SERVICE_UUID.upper() not in uuids:
            continue
        rows.append(
            (
                str(getattr(device, "address", "")),
                getattr(device, "name", None) or getattr(adv, "local_name", None),
                getattr(adv, "rssi", None),
                uuids,
            )
        )

    return rows


async def probe(address: str, timeout_s: float) -> bool:
    try:
        async with sdk.RingSoundClient(address=address, command_timeout_s=timeout_s) as ring:
            info = await sdk.get_system_info(ring, timeout_s=timeout_s)
    except Exception as exc:
        print(f"  connect/read failed: {type(exc).__name__}: {exc}")
        return False

    print("  OK - Ring Sound device")
    print(f"  firmware={info.firmware_version} battery={info.battery_percent}% model={info.model}")
    print(f"  address={address}")
    return True


async def main(args: argparse.Namespace) -> None:
    devices = await scan_with_service(args.scan_timeout)
    if not devices:
        raise SystemExit(
            "No NUS devices found. Keep the ring nearby/awake, then scan again."
        )

    print("NUS devices:")
    for index, (address, name, rssi, _uuids) in enumerate(devices):
        label = name or "(no name)"
        rssi_text = "" if rssi is None else f" rssi={rssi}"
        print(f"  [{index}] {address} {label}{rssi_text}")

    if args.index is None and not args.probe_all:
        print("\nRun collect_imu.py with one of these addresses, or re-run with --probe-all.")
        return

    selected = range(len(devices)) if args.probe_all else [args.index]
    for index in selected:
        if index is None or index < 0 or index >= len(devices):
            raise SystemExit(f"Device index out of range: {index}")
        address = devices[index][0]
        print(f"\nProbing [{index}] {address}")
        if await probe(address, args.command_timeout) and not args.probe_all:
            break


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find the Ring Sound BLE address.")
    parser.add_argument("--index", type=int, help="Probe one scanned device index.")
    parser.add_argument("--probe-all", action="store_true", help="Try every NUS device.")
    parser.add_argument("--scan-timeout", type=float, default=8.0)
    parser.add_argument("--command-timeout", type=float, default=5.0)
    return parser


if __name__ == "__main__":
    asyncio.run(main(build_parser().parse_args()))
