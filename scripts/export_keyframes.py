import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEARCH_DIRS = [
    ROOT / "logs" / "video_eval",
]
if os.environ.get("KATACR_VIDEO_DIR"):
    DEFAULT_SEARCH_DIRS.insert(0, Path(os.environ["KATACR_VIDEO_DIR"]))


def find_latest_frame_log():
    paths = list((ROOT / "logs" / "video_eval").glob("*/*/*frames.jsonl"))
    if not paths:
        raise FileNotFoundError("No *_frames.jsonl found under logs/video_eval")
    return max(paths, key=lambda p: p.stat().st_mtime)


def load_rows(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
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


def env_timing(row, key):
    env = nested_get(row, "timings.env", {}) or {}
    value = env.get(key)
    if isinstance(value, (int, float)):
        return value * 1000.0
    return None


def timing_ms(row, key):
    value = nested_get(row, "timings." + key)
    if isinstance(value, (int, float)):
        return value * 1000.0
    return None


def fmt_ms(value):
    return "n/a" if value is None else f"{value:.1f}ms"


def find_video_for_log(frame_log):
    stem = Path(frame_log).parent.name
    candidates = []
    for base in DEFAULT_SEARCH_DIRS:
        if not base.exists():
            continue
        for name in (stem + ".mp4", stem + ".*.mp4"):
            candidates.extend(base.glob(name))
        if base.name == "video_eval":
            candidates.extend(base.glob(f"*/*/{stem}.mp4"))
            candidates.extend(base.glob(f"*/*/{stem}.*.mp4"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not infer source video for log directory '{stem}'. Pass --video /path/to/video.mp4"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def row_score(row, metric):
    if metric == "step":
        return timing_ms(row, "step") or -1
    if metric == "state_text_info":
        return env_timing(row, "state.text_info") or -1
    if metric == "reward":
        return env_timing(row, "sar.get_reward") or -1
    if metric == "yolo":
        return env_timing(row, "visual.yolo") or -1
    raise ValueError(f"Unknown metric: {metric}")


def select_rows(rows, metric, top_n, include_target_actions, include_guardrails):
    selected = []
    seen = set()

    def add(row, reason):
        key = row.get("step_index")
        if key in seen:
            return
        seen.add(key)
        selected.append((row, reason))

    for row in sorted(rows, key=lambda r: row_score(r, metric), reverse=True)[:top_n]:
        add(row, f"top_{metric}")

    if include_target_actions:
        for row in rows:
            if nested_get(row, "target_action.card_id", 0):
                add(row, "target_action")

    if include_guardrails:
        for row in rows:
            if nested_get(row, "policy_action_raw.card_id") != nested_get(row, "policy_action.card_id"):
                add(row, "guardrail_changed")

    selected.sort(key=lambda item: item[0].get("source_frame") or item[0].get("step_index") or 0)
    return selected


def seek_frame(cap, source_frame, video_interval):
    # ImageAndVideoLoader increments source_frame after grabbing video_interval frames.
    raw_frame = max(0, int(source_frame) * int(video_interval) - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, raw_frame)
    ok, frame = cap.read()
    if ok:
        return frame, raw_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, raw_frame - int(video_interval)))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Failed to read video frame near source_frame={source_frame}")
    return frame, max(0, raw_frame - int(video_interval))


def text_lines(row, reason, raw_frame):
    raw = row.get("policy_action_raw", {}) or {}
    policy = row.get("policy_action", {}) or {}
    target = row.get("target_action", {}) or {}
    state = row.get("state", {}) or {}
    prev = row.get("prev_state", {}) or {}
    lines = [
        f"reason: {reason}",
        f"step: {row.get('step_index')}  source_frame: {row.get('source_frame')}  raw_frame: {raw_frame}",
        f"time: {state.get('time')}  elixir: {state.get('elixir')}  prev_elixir: {prev.get('elixir')}",
        f"cards: {state.get('cards')}",
        f"policy_raw: {raw.get('card_name')} id={raw.get('card_id')} xy={raw.get('xy')}",
        f"policy_used: id={policy.get('card_id')} xy={policy.get('xy')}",
        f"target: id={target.get('card_id')} xy={target.get('xy')} offset={target.get('offset')}",
        f"reward: {row.get('reward')} total: {row.get('total_reward')} done={row.get('done')}",
        f"policy: {fmt_ms(timing_ms(row, 'policy'))}  step: {fmt_ms(timing_ms(row, 'step'))}",
        f"visual: {fmt_ms(env_timing(row, 'visual_fusion'))}  yolo: {fmt_ms(env_timing(row, 'visual.yolo'))}",
        f"card_cls: {fmt_ms(env_timing(row, 'visual.card_cls'))}  elixir_ocr: {fmt_ms(env_timing(row, 'visual.ocr_elixir'))}",
        f"time_ocr: {fmt_ms(env_timing(row, 'visual.ocr_time'))}  reward_get: {fmt_ms(env_timing(row, 'sar.get_reward'))}",
        f"state_update: {fmt_ms(env_timing(row, 'state_update'))}",
        f"state_text_info: {fmt_ms(env_timing(row, 'state.text_info'))}",
    ]
    return lines


def draw_panel(frame, row, reason, raw_frame, panel_width=560):
    h, w = frame.shape[:2]
    panel = np.full((h, panel_width, 3), 245, dtype=np.uint8)
    y = 32
    for i, line in enumerate(text_lines(row, reason, raw_frame)):
        color = (20, 20, 20)
        if i == 0:
            color = (0, 80, 180)
        if "state_text_info" in line or "step:" in line:
            color = (0, 0, 180)
        cv2.putText(panel, line[:74], (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
        y += 27
    return np.concatenate([frame, panel], axis=1)


def safe_name(row, reason):
    return f"step_{int(row.get('step_index', 0)):04d}_frame_{int(row.get('source_frame', 0)):05d}_{reason}.jpg"


def write_manifest(path, records):
    fieldnames = [
        "image",
        "reason",
        "step_index",
        "source_frame",
        "raw_frame",
        "policy_card",
        "policy_card_id",
        "policy_used_id",
        "target_card_id",
        "step_ms",
        "state_text_info_ms",
        "yolo_ms",
        "reward_get_ms",
        "total_reward",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def make_contact_sheet(image_paths, out_path, thumb_width=360, cols=2):
    images = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        scale = thumb_width / img.shape[1]
        thumb = cv2.resize(img, (thumb_width, max(1, int(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        images.append(thumb)
    if not images:
        return None
    rows = []
    for i in range(0, len(images), cols):
        row_imgs = images[i : i + cols]
        max_h = max(img.shape[0] for img in row_imgs)
        padded = []
        for img in row_imgs:
            if img.shape[0] < max_h:
                pad = np.full((max_h - img.shape[0], img.shape[1], 3), 255, dtype=np.uint8)
                img = np.concatenate([img, pad], axis=0)
            padded.append(img)
        while len(padded) < cols:
            padded.append(np.full((max_h, thumb_width, 3), 255, dtype=np.uint8))
        rows.append(np.concatenate(padded, axis=1))
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(str(out_path), sheet)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Export annotated keyframe screenshots from KataCR frame logs.")
    parser.add_argument("frame_log", nargs="?", default=None, help="Path to *_frames.jsonl. Defaults to latest.")
    parser.add_argument("--video", default=None, help="Source video path. Defaults to inferred by log directory name.")
    parser.add_argument("--metric", default="step", choices=["step", "state_text_info", "reward", "yolo"])
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--video-interval", type=int, default=3)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--include-target-actions", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--include-guardrails", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--contact-sheet", default=True, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    frame_log = Path(args.frame_log) if args.frame_log else find_latest_frame_log()
    rows = load_rows(frame_log)
    video_path = Path(args.video) if args.video else find_video_for_log(frame_log)
    out_dir = Path(args.out_dir) if args.out_dir else frame_log.parent / "keyframes"
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = select_rows(rows, args.metric, args.top_n, args.include_target_actions, args.include_guardrails)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    image_paths = []
    manifest_rows = []
    try:
        for row, reason in selected:
            frame, raw_frame = seek_frame(cap, row.get("source_frame", 0), args.video_interval)
            annotated = draw_panel(frame, row, reason, raw_frame)
            out_path = out_dir / safe_name(row, reason)
            cv2.imwrite(str(out_path), annotated)
            image_paths.append(out_path)
            manifest_rows.append(
                {
                    "image": str(out_path),
                    "reason": reason,
                    "step_index": row.get("step_index"),
                    "source_frame": row.get("source_frame"),
                    "raw_frame": raw_frame,
                    "policy_card": nested_get(row, "policy_action_raw.card_name"),
                    "policy_card_id": nested_get(row, "policy_action_raw.card_id"),
                    "policy_used_id": nested_get(row, "policy_action.card_id"),
                    "target_card_id": nested_get(row, "target_action.card_id"),
                    "step_ms": timing_ms(row, "step"),
                    "state_text_info_ms": env_timing(row, "state.text_info"),
                    "yolo_ms": env_timing(row, "visual.yolo"),
                    "reward_get_ms": env_timing(row, "sar.get_reward"),
                    "total_reward": row.get("total_reward"),
                }
            )
    finally:
        cap.release()

    manifest_path = out_dir / "manifest.csv"
    write_manifest(manifest_path, manifest_rows)
    sheet_path = None
    if args.contact_sheet:
        sheet_path = make_contact_sheet(image_paths, out_dir / "contact_sheet.jpg")

    print(f"Frame log: {frame_log}")
    print(f"Video: {video_path}")
    print(f"Exported images: {len(image_paths)}")
    print(f"Output dir: {out_dir}")
    print(f"Manifest: {manifest_path}")
    if sheet_path:
        print(f"Contact sheet: {sheet_path}")


if __name__ == "__main__":
    main()
