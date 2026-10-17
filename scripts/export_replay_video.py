import argparse
from pathlib import Path

import cv2

from export_keyframes import (
    draw_panel,
    find_latest_frame_log,
    find_video_for_log,
    load_rows,
    nested_get,
    seek_frame,
)


def ensure_even_size(img):
    h, w = img.shape[:2]
    return img[: h - (h % 2), : w - (w % 2)]


def resize_to_width(img, max_width):
    if not max_width or img.shape[1] <= max_width:
        return img
    scale = max_width / img.shape[1]
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    width -= width % 2
    height -= height % 2
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def frame_reason(row):
    if nested_get(row, "target_action.card_id", 0):
        return "target_action"
    if nested_get(row, "policy_action_raw.card_id") != nested_get(row, "policy_action.card_id"):
        return "guardrail_changed"
    return "decision"


def draw_progress_bar(img, index, total, color=(0, 180, 0)):
    h, w = img.shape[:2]
    bar_h = max(8, h // 160)
    cv2.rectangle(img, (0, h - bar_h), (w, h), (35, 35, 35), -1)
    end_x = int(w * (index + 1) / max(1, total))
    cv2.rectangle(img, (0, h - bar_h), (end_x, h), color, -1)
    return img


def iter_selected_rows(rows, every, max_frames):
    selected = rows[:: max(1, every)]
    if max_frames:
        selected = selected[:max_frames]
    return selected


def main():
    parser = argparse.ArgumentParser(description="Export a lightweight OpenCV replay video from KataCR frame logs.")
    parser.add_argument("frame_log", nargs="?", default=None, help="Path to *_frames.jsonl. Defaults to latest.")
    parser.add_argument("--video", default=None, help="Source video path. Defaults to inferred by log directory name.")
    parser.add_argument("--video-interval", type=int, default=3)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--every", type=int, default=1, help="Use every N decision rows.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-width", type=int, default=1280, help="Resize replay frames to this max width. 0 disables.")
    parser.add_argument("--out", default=None, help="Output mp4 path. Defaults to replay/replay_decisions.mp4")
    args = parser.parse_args()

    frame_log = Path(args.frame_log) if args.frame_log else find_latest_frame_log()
    rows = load_rows(frame_log)
    rows = iter_selected_rows(rows, args.every, args.max_frames)
    if not rows:
        raise ValueError("No rows selected for replay")

    video_path = Path(args.video) if args.video else find_video_for_log(frame_log)
    out_path = Path(args.out) if args.out else frame_log.parent / "replay" / "replay_decisions.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    writer = None
    written = 0
    try:
        for i, row in enumerate(rows):
            frame, raw_frame = seek_frame(cap, row.get("source_frame", 0), args.video_interval)
            annotated = draw_panel(frame, row, frame_reason(row), raw_frame)
            annotated = draw_progress_bar(annotated, i, len(rows))
            annotated = resize_to_width(annotated, args.max_width)
            annotated = ensure_even_size(annotated)
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    str(out_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    args.fps,
                    (w, h),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer: {out_path}")
            writer.write(annotated)
            written += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    print(f"Frame log: {frame_log}")
    print(f"Video: {video_path}")
    print(f"Rows selected: {len(rows)}")
    print(f"Frames written: {written}")
    print(f"FPS: {args.fps}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
