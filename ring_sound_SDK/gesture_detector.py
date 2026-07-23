import argparse
import csv
import json
import math
import re
import shutil
import time
from pathlib import Path


FEATURES = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]


def read_imu_csv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = []
        for row in csv.DictReader(file):
            parsed = {"timestamp_ms": float(row["timestamp_ms"])}
            for feature in FEATURES:
                parsed[feature] = float(row[feature])
            rows.append(parsed)
    if len(rows) < 2:
        raise SystemExit(f"Need at least 2 rows in {path}")
    return rows


def write_imu_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["timestamp_ms", *FEATURES])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: int(row[key]) for key in ["timestamp_ms", *FEATURES]})


def sanitize_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
    if not cleaned:
        raise SystemExit("Gesture label is empty")
    return cleaned


def active_window(rows: list[dict[str, float]], margin_ms: float = 300.0) -> list[dict[str, float]]:
    gyro_scores = [
        math.sqrt(row["gyro_x"] ** 2 + row["gyro_y"] ** 2 + row["gyro_z"] ** 2)
        for row in rows
    ]
    sorted_scores = sorted(gyro_scores)
    median = sorted_scores[len(sorted_scores) // 2]
    peak = max(gyro_scores)
    if peak <= median:
        return rows

    threshold = median + (peak - median) * 0.25
    active = [index for index, score in enumerate(gyro_scores) if score >= threshold]
    if not active:
        return rows

    start_ts = rows[max(0, min(active))]["timestamp_ms"] - margin_ms
    end_ts = rows[min(len(rows) - 1, max(active))]["timestamp_ms"] + margin_ms
    clipped = [row for row in rows if start_ts <= row["timestamp_ms"] <= end_ts]
    return clipped if len(clipped) >= 2 else rows


def resample(rows: list[dict[str, float]], points: int) -> list[list[float]]:
    if points < 2:
        raise SystemExit("--points must be at least 2")

    times = [row["timestamp_ms"] for row in rows]
    start = times[0]
    end = times[-1]
    if end <= start:
        raise SystemExit("Input timestamps are not increasing")

    output: list[list[float]] = []
    cursor = 0
    for point in range(points):
        target = start + (end - start) * point / (points - 1)
        while cursor < len(rows) - 2 and times[cursor + 1] < target:
            cursor += 1

        left = rows[cursor]
        right = rows[min(cursor + 1, len(rows) - 1)]
        left_t = left["timestamp_ms"]
        right_t = right["timestamp_ms"]
        ratio = 0.0 if right_t == left_t else (target - left_t) / (right_t - left_t)
        output.append(
            [
                left[feature] + (right[feature] - left[feature]) * ratio
                for feature in FEATURES
            ]
        )
    return output


def normalize(sequence: list[list[float]]) -> list[list[float]]:
    columns = list(zip(*sequence))
    means = [sum(column) / len(column) for column in columns]
    stds = []
    for mean, column in zip(means, columns):
        variance = sum((value - mean) ** 2 for value in column) / len(column)
        stds.append(math.sqrt(variance) or 1.0)

    return [
        [(value - means[index]) / stds[index] for index, value in enumerate(row)]
        for row in sequence
    ]


def prepare_sequence(path: Path, points: int, auto_segment: bool) -> list[list[float]]:
    rows = read_imu_csv(path)
    if auto_segment:
        rows = active_window(rows)
    return normalize(resample(rows, points))


def motion_stats(path: Path) -> dict[str, float]:
    rows = read_imu_csv(path)
    gyro_mag = [
        math.sqrt(row["gyro_x"] ** 2 + row["gyro_y"] ** 2 + row["gyro_z"] ** 2)
        for row in rows
    ]
    accel_mag = [
        math.sqrt(row["accel_x"] ** 2 + row["accel_y"] ** 2 + row["accel_z"] ** 2)
        for row in rows
    ]
    duration_s = (rows[-1]["timestamp_ms"] - rows[0]["timestamp_ms"]) / 1000.0
    gyro_rms = math.sqrt(sum(value * value for value in gyro_mag) / len(gyro_mag))
    accel_rms = math.sqrt(sum(value * value for value in accel_mag) / len(accel_mag))
    return {
        "rows": float(len(rows)),
        "duration_s": duration_s,
        "gyro_rms": gyro_rms,
        "gyro_peak": max(gyro_mag),
        "accel_rms": accel_rms,
        "accel_peak": max(accel_mag),
    }


def distance(left: list[list[float]], right: list[list[float]]) -> float:
    if len(left) != len(right):
        raise ValueError("Sequences must have the same length")

    total = 0.0
    count = 0
    for left_row, right_row in zip(left, right):
        for left_value, right_value in zip(left_row, right_row):
            total += (left_value - right_value) ** 2
            count += 1
    return math.sqrt(total / max(1, count))


def cmd_add_template(args: argparse.Namespace) -> None:
    label = sanitize_label(args.gesture)
    target_dir = args.dataset / label
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{int(time.time() * 1000)}_{args.input.name}"
    shutil.copy2(args.input, target)
    print(f"added template: {target}")


def cmd_train(args: argparse.Namespace) -> None:
    templates = []
    for gesture_dir in sorted(args.dataset.iterdir()):
        if not gesture_dir.is_dir():
            continue
        for csv_path in sorted(gesture_dir.glob("*.csv")):
            templates.append(
                {
                    "gesture": gesture_dir.name,
                    "source": str(csv_path),
                    "sequence": prepare_sequence(csv_path, args.points, not args.no_auto_segment),
                }
            )

    if not templates:
        raise SystemExit(f"No templates found under {args.dataset}")

    model = {
        "points": args.points,
        "features": FEATURES,
        "auto_segment": not args.no_auto_segment,
        "templates": templates,
    }
    args.model.write_text(json.dumps(model), encoding="utf-8")
    gestures = sorted({template["gesture"] for template in templates})
    print(f"trained {len(templates)} templates for gestures: {', '.join(gestures)}")
    print(f"model: {args.model}")


def cmd_classify(args: argparse.Namespace) -> None:
    model = json.loads(args.model.read_text(encoding="utf-8"))
    stats = motion_stats(args.input)
    sequence = prepare_sequence(
        args.input,
        int(model["points"]),
        bool(model.get("auto_segment", True)) and not args.no_auto_segment,
    )

    scored = []
    for template in model["templates"]:
        scored.append(
            (
                distance(sequence, template["sequence"]),
                template["gesture"],
                template["source"],
            )
        )
    scored.sort(key=lambda item: item[0])
    best_distance, best_gesture, best_source = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else best_distance
    confidence = 1.0 - best_distance / runner_up if runner_up > 0 else 1.0
    vote_window = scored[: max(1, args.vote_k)]
    vote_counts: dict[str, int] = {}
    for _dist, gesture, _source in vote_window:
        vote_counts[gesture] = vote_counts.get(gesture, 0) + 1

    gestures = {template["gesture"] for template in model["templates"]}
    decision = "unknown"
    if (
        "idle" in gestures
        and stats["gyro_rms"] <= args.idle_gyro_rms
        and stats["gyro_peak"] <= args.idle_gyro_peak
    ):
        decision = "idle"
    elif (
        vote_counts.get("sos_shake_hand", 0) > 0
        and (
            stats["gyro_rms"] < args.sos_min_gyro_rms
            or stats["gyro_peak"] < args.sos_min_gyro_peak
        )
    ):
        decision = "unknown"
    elif (
        vote_counts.get("sos_shake_hand", 0) >= args.sos_min_votes
        and best_distance <= args.max_distance
    ):
        decision = "sos_shake_hand"
    elif (
        best_gesture != "sos_shake_hand"
        and best_distance <= args.max_distance
        and confidence >= args.non_sos_min_confidence
    ):
        if args.specific_non_sos:
            decision = best_gesture
        else:
            decision = "non_sos_motion"
    elif best_distance <= args.max_distance and confidence >= args.min_confidence:
        decision = best_gesture
    else:
        decision = "unknown"

    print(f"decision: {decision}")
    print(f"nearest_gesture: {best_gesture}")
    print(f"distance: {best_distance:.4f}")
    print(f"confidence: {confidence:.2f}")
    print(
        "votes: "
        + ", ".join(
            f"{gesture}={count}" for gesture, count in sorted(vote_counts.items())
        )
    )
    print(
        "motion: "
        f"rows={int(stats['rows'])} duration={stats['duration_s']:.2f}s "
        f"gyro_rms={stats['gyro_rms']:.1f} gyro_peak={stats['gyro_peak']:.1f}"
    )
    print(f"nearest_template: {best_source}")
    print("top matches:")
    for dist, gesture, source in scored[: args.top]:
        print(f"  {gesture:20s} {dist:.4f} {source}")


def cmd_segment(args: argparse.Namespace) -> None:
    rows = active_window(read_imu_csv(args.input), margin_ms=args.margin_ms)
    write_imu_csv(args.output, rows)
    print(f"wrote active segment with {len(rows)} rows to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Template-based gesture detector for Ring Sound IMU CSV.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_template = subparsers.add_parser("add-template", help="Add one labeled CSV to a dataset.")
    add_template.add_argument("--gesture", required=True, help="Gesture label, e.g. rotate_front.")
    add_template.add_argument("--input", type=Path, required=True, help="IMU CSV file.")
    add_template.add_argument("--dataset", type=Path, default=Path("gesture_data"))
    add_template.set_defaults(func=cmd_add_template)

    train = subparsers.add_parser("train", help="Build a JSON template model.")
    train.add_argument("--dataset", type=Path, default=Path("gesture_data"))
    train.add_argument("--model", type=Path, default=Path("gesture_model.json"))
    train.add_argument("--points", type=int, default=64)
    train.add_argument("--no-auto-segment", action="store_true")
    train.set_defaults(func=cmd_train)

    classify = subparsers.add_parser("classify", help="Classify one IMU CSV with a trained model.")
    classify.add_argument("--input", type=Path, required=True)
    classify.add_argument("--model", type=Path, default=Path("gesture_model.json"))
    classify.add_argument("--top", type=int, default=5)
    classify.add_argument("--no-auto-segment", action="store_true")
    classify.add_argument("--min-confidence", type=float, default=0.08)
    classify.add_argument("--non-sos-min-confidence", type=float, default=0.015)
    classify.add_argument("--max-distance", type=float, default=1.45)
    classify.add_argument("--vote-k", type=int, default=3)
    classify.add_argument("--sos-min-votes", type=int, default=2)
    classify.add_argument("--idle-gyro-rms", type=float, default=1800.0)
    classify.add_argument("--idle-gyro-peak", type=float, default=7000.0)
    classify.add_argument("--sos-min-gyro-rms", type=float, default=5000.0)
    classify.add_argument("--sos-min-gyro-peak", type=float, default=10000.0)
    classify.add_argument("--specific-non-sos", action="store_true")
    classify.set_defaults(func=cmd_classify)

    segment = subparsers.add_parser("segment", help="Extract the active gesture window from one CSV.")
    segment.add_argument("--input", type=Path, required=True)
    segment.add_argument("--output", type=Path, required=True)
    segment.add_argument("--margin-ms", type=float, default=300.0)
    segment.set_defaults(func=cmd_segment)

    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    parsed_args.func(parsed_args)
