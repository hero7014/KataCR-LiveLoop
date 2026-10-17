import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_TIMING_KEYS = [
    "policy",
    "step",
    "env.visual_fusion",
    "env.visual.yolo",
    "env.visual.yolo.model0",
    "env.visual.yolo.model1",
    "env.visual.card_cls",
    "env.visual.ocr_time",
    "env.visual.ocr_elixir",
    "env.state_update",
    "env.state.text_info",
    "env.sar.get_reward",
    "env.sar.reward_get_detail.tower_hp_ocr",
]


def find_latest_frame_log():
    paths = list((ROOT / "logs" / "video_eval").glob("*/*/*frames.jsonl"))
    if not paths:
        raise FileNotFoundError("No *_frames.jsonl found under logs/video_eval")
    return max(paths, key=lambda p: p.stat().st_mtime)


def load_rows(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def nested_get(value, dotted_key, default=None):
    cur = value
    for key in dotted_key.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def timing_value(row, key):
    if key.startswith("env."):
        env = nested_get(row, "timings.env", {}) or {}
        return env.get(key[len("env.") :])
    return nested_get(row, "timings." + key)


def as_ms(value):
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value * 1000.0
    return None


def percentile(sorted_values, ratio):
    if not sorted_values:
        return None
    idx = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * ratio))))
    return sorted_values[idx]


def stats_ms(rows, key):
    values = [as_ms(timing_value(row, key)) for row in rows]
    values = [v for v in values if v is not None]
    if not values:
        return None
    values.sort()
    return {
        "key": key,
        "count": len(values),
        "avg_ms": sum(values) / len(values),
        "p50_ms": percentile(values, 0.50),
        "p90_ms": percentile(values, 0.90),
        "p95_ms": percentile(values, 0.95),
        "max_ms": values[-1],
    }


def compact_action(action):
    if not isinstance(action, dict):
        return ""
    card = action.get("card_name")
    card_id = action.get("card_id")
    xy = action.get("xy")
    if card is not None:
        return f"{card}:{card_id}@{xy}"
    return f"{card_id}@{xy}"


def row_summary(row):
    env = nested_get(row, "timings.env", {}) or {}
    return {
        "episode": row.get("episode"),
        "step_index": row.get("step_index"),
        "visual_count": row.get("visual_count"),
        "source_frame": row.get("source_frame"),
        "time": nested_get(row, "state.time"),
        "elixir": nested_get(row, "state.elixir"),
        "cards": " ".join(map(str, nested_get(row, "state.cards", []) or [])),
        "policy_card": nested_get(row, "policy_action_raw.card_name"),
        "policy_action": compact_action(row.get("policy_action")),
        "target_action": compact_action(row.get("target_action")),
        "reward": row.get("reward"),
        "total_reward": row.get("total_reward"),
        "done": row.get("done"),
        "policy_ms": as_ms(nested_get(row, "timings.policy")),
        "step_ms": as_ms(nested_get(row, "timings.step")),
        "visual_fusion_ms": as_ms(env.get("visual_fusion")),
        "yolo_ms": as_ms(env.get("visual.yolo")),
        "state_update_ms": as_ms(env.get("state_update")),
        "state_text_info_ms": as_ms(env.get("state.text_info")),
        "reward_get_ms": as_ms(env.get("sar.get_reward")),
        "tower_hp_ocr_ms": as_ms(env.get("sar.reward_get_detail.tower_hp_ocr")),
    }


def write_slow_csv(rows, path, top_n):
    summaries = [row_summary(row) for row in rows]
    summaries.sort(key=lambda x: x.get("step_ms") or -1, reverse=True)
    selected = summaries[:top_n]
    fieldnames = [
        "episode",
        "step_index",
        "visual_count",
        "source_frame",
        "time",
        "elixir",
        "cards",
        "policy_card",
        "policy_action",
        "target_action",
        "reward",
        "total_reward",
        "done",
        "policy_ms",
        "step_ms",
        "visual_fusion_ms",
        "yolo_ms",
        "state_update_ms",
        "state_text_info_ms",
        "reward_get_ms",
        "tower_hp_ocr_ms",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    return selected


def write_summary_csv(stats_rows, path):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["key", "count", "avg_ms", "p50_ms", "p90_ms", "p95_ms", "max_ms"],
        )
        writer.writeheader()
        writer.writerows(stats_rows)


def fmt_ms(value):
    return "n/a" if value is None else f"{value:.1f}ms"


def print_stats(stats_rows):
    print("\nTiming summary")
    print("-" * 88)
    print(f"{'key':36} {'avg':>9} {'p50':>9} {'p90':>9} {'p95':>9} {'max':>9} {'n':>5}")
    for s in stats_rows:
        print(
            f"{s['key'][:36]:36} "
            f"{fmt_ms(s['avg_ms']):>9} "
            f"{fmt_ms(s['p50_ms']):>9} "
            f"{fmt_ms(s['p90_ms']):>9} "
            f"{fmt_ms(s['p95_ms']):>9} "
            f"{fmt_ms(s['max_ms']):>9} "
            f"{s['count']:>5}"
        )


def print_counter(title, counter, n=10):
    print(f"\n{title}")
    print("-" * 88)
    for key, count in counter.most_common(n):
        print(f"{str(key):36} {count}")


def main():
    parser = argparse.ArgumentParser(description="Analyze KataCR per-decision frame JSONL logs.")
    parser.add_argument("path", nargs="?", default=None, help="Path to *_frames.jsonl. Defaults to latest log.")
    parser.add_argument("--top-n", type=int, default=30, help="Number of slowest rows to export and print.")
    parser.add_argument("--out-dir", type=str, default=None, help="Directory for CSV outputs. Defaults to log directory.")
    parser.add_argument(
        "--timing-key",
        action="append",
        default=None,
        help="Extra dotted timing key, e.g. env.state.text_info. Can be repeated.",
    )
    args = parser.parse_args()

    path = Path(args.path) if args.path else find_latest_frame_log()
    rows = load_rows(path)
    out_dir = Path(args.out_dir) if args.out_dir else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = list(DEFAULT_TIMING_KEYS)
    if args.timing_key:
        keys.extend(args.timing_key)
    stats_rows = [stats_ms(rows, key) for key in keys]
    stats_rows = [s for s in stats_rows if s is not None]

    summary_csv = out_dir / f"{path.stem}_analysis_summary.csv"
    slow_csv = out_dir / f"{path.stem}_slowest_steps.csv"
    write_summary_csv(stats_rows, summary_csv)
    slow_rows = write_slow_csv(rows, slow_csv, args.top_n)

    policy_cards = Counter(nested_get(row, "policy_action_raw.card_name") for row in rows)
    policy_cards.pop(None, None)
    raw_policy_ids = Counter(nested_get(row, "policy_action_raw.card_id") for row in rows)
    target_ids = Counter(nested_get(row, "target_action.card_id") for row in rows)
    action_changed = sum(
        1
        for row in rows
        if nested_get(row, "policy_action_raw.card_id") != nested_get(row, "policy_action.card_id")
    )
    target_nonzero = sum(1 for row in rows if nested_get(row, "target_action.card_id", 0))
    policy_nonzero = sum(1 for row in rows if nested_get(row, "policy_action.card_id", 0))
    time_zero = sum(1 for row in rows if nested_get(row, "state.time") == 0)
    elixir_missing = sum(1 for row in rows if nested_get(row, "state.elixir") is None)

    print(f"Frame log: {path}")
    print(f"Rows: {len(rows)}")
    print(f"Source frames: {rows[0].get('source_frame')} -> {rows[-1].get('source_frame')} / {rows[-1].get('total_frame')}")
    print(f"Final score: {rows[-1].get('total_reward')}  done={rows[-1].get('done')}")
    print(f"Policy nonzero actions: {policy_nonzero}/{len(rows)}")
    print(f"Target nonzero actions: {target_nonzero}/{len(rows)}")
    print(f"Policy actions changed by guardrails: {action_changed}/{len(rows)}")
    print(f"Rows with state.time == 0: {time_zero}/{len(rows)}")
    print(f"Rows with missing elixir: {elixir_missing}/{len(rows)}")
    print_stats(stats_rows)
    print_counter("Top policy cards", policy_cards)
    print_counter("Top raw policy card ids", raw_policy_ids)
    print_counter("Top target card ids", target_ids)

    print("\nSlowest steps")
    print("-" * 88)
    for row in slow_rows[: min(10, len(slow_rows))]:
        print(
            f"step={row['step_index']} frame={row['source_frame']} "
            f"step={fmt_ms(row['step_ms'])} text={fmt_ms(row['state_text_info_ms'])} "
            f"yolo={fmt_ms(row['yolo_ms'])} reward={fmt_ms(row['reward_get_ms'])} "
            f"policy={row['policy_card']} target={row['target_action']}"
        )

    print(f"\nWrote: {summary_csv}")
    print(f"Wrote: {slow_csv}")


if __name__ == "__main__":
    main()
