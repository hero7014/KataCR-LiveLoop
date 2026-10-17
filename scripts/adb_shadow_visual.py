import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.adb_probe import connect, normalize_portrait_frame, screencap
from scripts.adb_ui_flow import detect_page_state, execute_action, resolve_action


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def summarize_arena(arena):
    data = arena.get_data()
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    rows = []
    for row in np.asarray(data):
        values = row.tolist()
        rows.append({
            "xyxy": [float(x) for x in values[:4]],
            "conf": float(values[4]) if len(values) > 4 else None,
            "cls": int(values[5]) if len(values) > 5 else None,
            "bel": int(values[6]) if len(values) > 6 else None,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run read-only realtime VisualFusion from MuMu ADB screenshots.")
    parser.add_argument("--adb", default=os.environ.get("KATACR_ADB", "adb"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=16384)
    parser.add_argument("--serial", default="127.0.0.1:16384")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--target-ratio", type=float, default=2.16)
    parser.add_argument("--out-dir", default=str(ROOT / "logs" / "adb_shadow" / time.strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--ocr-time-interval", type=int, default=3)
    parser.add_argument("--ocr-elixir-interval", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--skip-connect", action="store_true")
    parser.add_argument("--ui-flow", choices=["off", "suggest", "act"], default="off")
    parser.add_argument("--confirm-execute", action="store_true", help="Allow guarded UI flow actions to send ADB taps.")
    parser.add_argument("--force-ui-action", action="store_true", help="Bypass UI page-state guard for recovery.")
    parser.add_argument("--ui-action-cooldown", type=float, default=2.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "frames.jsonl"

    if not args.skip_connect:
        code, out, err, elapsed = connect(args.adb, args.host, args.port)
        print(f"connect {args.host}:{args.port}: code={code} ({elapsed*1000:.1f}ms)", flush=True)
        if out.strip():
            print(out.strip(), flush=True)
        if err.strip():
            print(err.strip(), flush=True)

    visual = None
    if not args.capture_only:
        print("Importing VisualFusion...", flush=True)
        from katacr.policy.visualization.visual_fusion import VisualFusion

        print("Loading VisualFusion...", flush=True)
        visual = VisualFusion(
            ocr_gpu=True,
            time_ocr_interval=args.ocr_time_interval,
            elixir_ocr_interval=args.ocr_elixir_interval,
        )
        print("VisualFusion ready.", flush=True)

    last_ui_action_at = 0.0

    with jsonl_path.open("w", encoding="utf-8") as f:
        for idx in range(args.frames):
            frame_start = time.perf_counter()
            raw, nbytes, capture_dt = screencap(args.adb, serial=args.serial)
            img, norm_info = normalize_portrait_frame(raw, target_ratio=args.target_ratio)

            row = {
                "frame": idx,
                "time_wall": time.time(),
                "raw_shape": raw.shape,
                "shape": img.shape,
                "bytes": nbytes,
                "capture_ms": capture_dt * 1000,
                "normalize": norm_info,
            }

            if args.ui_flow != "off":
                state = detect_page_state(raw)
                decision = resolve_action(
                    "next",
                    state,
                    confirmed=args.ui_flow == "act" and args.confirm_execute,
                    force=args.force_ui_action,
                )
                row["ui_state"] = state.value
                row["ui_decision"] = json_safe(decision.__dict__)
                if decision.allowed:
                    now = time.perf_counter()
                    if now - last_ui_action_at >= args.ui_action_cooldown:
                        execute_action(decision.action, args.adb, args.serial)
                        row["ui_executed"] = True
                        last_ui_action_at = now
                    else:
                        row["ui_executed"] = False
                        row["ui_skip_reason"] = "cooldown"

            if visual is not None:
                t = time.perf_counter()
                info = visual.process(img, pil=False)
                visual_dt = time.perf_counter() - t
                arena_rows = summarize_arena(info["arena"])
                row.update({
                    "visual_ms": visual_dt * 1000,
                    "visual_detail_ms": {k: v * 1000 for k, v in getattr(visual, "dt", {}).items()},
                    "battle_time": info.get("time"),
                    "elixir": info.get("elixir"),
                    "cards": info.get("cards"),
                    "arena_count": len(arena_rows),
                    "arena": arena_rows,
                })

            should_save = args.save_every > 0 and idx % args.save_every == 0
            if should_save:
                raw_path = frames_dir / f"{idx:04d}_raw.png"
                frame_path = frames_dir / f"{idx:04d}_norm.png"
                cv2.imwrite(str(raw_path), raw)
                cv2.imwrite(str(frame_path), img)
                row["raw_path"] = raw_path
                row["frame_path"] = frame_path
                if visual is not None and not args.no_render:
                    render_path = frames_dir / f"{idx:04d}_render.png"
                    render = visual.render(img, pil=False, verbose=False)
                    cv2.imwrite(str(render_path), render)
                    row["render_path"] = render_path

            row["total_ms"] = (time.perf_counter() - frame_start) * 1000
            f.write(json.dumps(json_safe(row), ensure_ascii=False, separators=(",", ":")) + "\n")
            f.flush()
            print(
                f"frame={idx} capture={row['capture_ms']:.1f}ms total={row['total_ms']:.1f}ms "
                f"visual={row.get('visual_ms')} boxes={row.get('arena_count')} "
                f"ui={row.get('ui_state')} action={row.get('ui_decision', {}).get('action')}",
                flush=True,
            )
            if idx + 1 < args.frames and args.interval > 0:
                time.sleep(args.interval)

    print(f"saved: {jsonl_path}", flush=True)
    print(f"frames: {frames_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
