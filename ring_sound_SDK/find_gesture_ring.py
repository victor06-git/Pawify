import argparse
import asyncio

import collect_imu
import ring_sound as sdk


async def connect_scanned_device(device: object, address: str, timeout_s: float) -> sdk.RingSoundClient:
    try:
        from bleak import BleakClient
    except ImportError as exc:
        raise SystemExit("Install bleak first: python -m pip install bleak") from exc

    transport = sdk.NusClient(address=address, scan_timeout_s=timeout_s)

    async def connect_from_scan() -> None:
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
    ring = sdk.RingSoundClient(transport=transport, command_timeout_s=timeout_s)
    await ring.connect()
    return ring


async def probe(args: argparse.Namespace) -> None:
    devices = await collect_imu.scan_ring_devices(args.scan_timeout, args.phone_mac)
    devices.sort(key=lambda item: item[3] if item[3] is not None else -999, reverse=True)
    if not devices:
        raise SystemExit("No device named 'ring' found.")

    print("Probing devices named 'ring' for IMU start:")
    for index, (device, address, name, rssi, _phone_mac_match) in enumerate(devices):
        print(f"\n[{index}] {address} {name or '(no name)'} rssi={rssi}")
        ring: sdk.RingSoundClient | None = None
        try:
            ring = await connect_scanned_device(device, address, args.command_timeout)
            info = await sdk.get_system_info(ring, timeout_s=args.command_timeout)
            print(f"  system: firmware={info.firmware_version} battery={info.battery_percent}% model={info.model}")
            start = await sdk.start_sensor_report(ring, timeout_s=args.command_timeout)
            print(
                "  OK gesture-mode ring: "
                f"{start.sample_rate_hz}Hz accel=+/-{start.accel_range_g}g "
                f"gyro=+/-{start.gyro_range_dps}dps"
            )
            print(f"  use address: {address}")
            await sdk.stop_sensor_report(ring, timeout_s=args.command_timeout)
            return
        except sdk.DeviceError as exc:
            print(f"  not usable for IMU: {exc}")
        except Exception as exc:
            print(f"  connect/probe failed: {type(exc).__name__}: {exc}")
        finally:
            if ring is not None:
                await ring.disconnect()

    raise SystemExit("No scanned ring accepted start_sensor_report().")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find the ring that is already in gesture mode.")
    parser.add_argument("--scan-timeout", type=float, default=15.0)
    parser.add_argument("--command-timeout", type=float, default=5.0)
    parser.add_argument("--phone-mac", help="BLE MAC shown by the phone app, if available.")
    return parser


if __name__ == "__main__":
    asyncio.run(probe(build_parser().parse_args()))
