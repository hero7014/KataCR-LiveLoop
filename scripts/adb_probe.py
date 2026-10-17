import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_TARGET_RATIO = 2.16


def default_windows_host():
    resolv = Path("/etc/resolv.conf")
    if resolv.exists():
        for line in resolv.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) == 2 and parts[0] == "nameserver":
                return parts[1]
    return "127.0.0.1"


def run_cmd(args, timeout=15, input_bytes=None, check=False):
    start = time.perf_counter()
    proc = subprocess.run(
        args,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    elapsed = time.perf_counter() - start
    out = proc.stdout.decode("utf-8", errors="replace")
    err = proc.stderr.decode("utf-8", errors="replace")
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(args)}\n{out}\n{err}")
    return proc.returncode, out, err, elapsed


def adb_cmd(adb, *args):
    return [adb, *args]


def adb_available(adb):
    try:
        code, out, err, elapsed = run_cmd(adb_cmd(adb, "version"), timeout=10)
    except FileNotFoundError:
        return False, "", "adb executable not found", 0.0
    return code == 0, out, err, elapsed


def connect(adb, host, port):
    target = f"{host}:{port}"
    return run_cmd(adb_cmd(adb, "connect", target), timeout=20)


def is_missing_device_error(error):
    error = error.lower()
    return "device" in error and ("not found" in error or "offline" in error)


def screencap(adb, serial=None):
    args = [adb]
    if serial:
        args.extend(["-s", serial])
    args.extend(["exec-out", "screencap", "-p"])

    def capture_once():
        start = time.perf_counter()
        proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        return proc, time.perf_counter() - start

    proc, elapsed = capture_once()
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")
        if serial and is_missing_device_error(err):
            run_cmd(adb_cmd(adb, "connect", serial), timeout=20)
            proc, retry_elapsed = capture_once()
            elapsed += retry_elapsed
            err = proc.stderr.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(f"screencap failed ({proc.returncode}): {err}")
    candidates = [
        proc.stdout,
        proc.stdout.replace(b"\r\r\n", b"\n"),
        proc.stdout.replace(b"\r\n", b"\n"),
        proc.stdout.replace(b"\r", b""),
    ]
    for data in candidates:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            return img, len(data), elapsed
    raise RuntimeError(f"Failed to decode screencap PNG, bytes={len(proc.stdout)}")


def normalize_portrait_frame(img, target_ratio=DEFAULT_TARGET_RATIO):
    h, w = img.shape[:2]
    if h < w:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w = img.shape[:2]
    ratio = h / w
    if ratio < target_ratio:
        target_w = int(h / target_ratio)
        if 0 < target_w < w:
            x0 = (w - target_w) // 2
            return img[:, x0:x0 + target_w].copy(), {
                "op": "center_crop_width",
                "source_shape": (h, w),
                "target_shape": (h, target_w),
                "crop_x0": x0,
                "crop_x1": x0 + target_w,
                "target_ratio": target_ratio,
            }
    if ratio > target_ratio:
        target_h = int(round(w * target_ratio))
        if 0 < target_h < h:
            y0 = (h - target_h) // 2
            return img[y0:y0 + target_h, :].copy(), {
                "op": "center_crop_height",
                "source_shape": (h, w),
                "target_shape": (target_h, w),
                "crop_y0": y0,
                "crop_y1": y0 + target_h,
                "target_ratio": target_ratio,
            }
    return img, {"op": "none", "source_shape": (h, w), "target_ratio": target_ratio}


def probe_roi(img):
    from katacr.build_dataset.utils.split_part import process_part, ratio2name

    ratio_name = ratio2name(img)
    results = {"ratio_name": ratio_name}
    for part in (1, 2, 3):
        try:
            crop, params = process_part(img, part, verbose=True)
            results[f"part{part}_shape"] = tuple(int(x) for x in crop.shape)
            results[f"part{part}_params"] = params
        except Exception as e:
            results[f"part{part}_error"] = str(e)
    return results


def main():
    parser = argparse.ArgumentParser(description="Probe MuMu/Android ADB connectivity from WSL.")
    parser.add_argument("--adb", default="adb", help="adb executable path. Can be Windows adb.exe under /mnt/c/...")
    parser.add_argument("--host", default=None, help="ADB host. Defaults to WSL nameserver Windows host IP.")
    parser.add_argument("--port", type=int, default=7555)
    parser.add_argument("--serial", default=None, help="ADB device serial. Defaults to host:port after connect.")
    parser.add_argument("--out-dir", default=str(ROOT / "logs" / "adb_probe"))
    parser.add_argument("--skip-connect", action="store_true")
    parser.add_argument("--normalize", action="store_true", help="Center-crop MuMu portrait frames to a KataCR-supported ratio.")
    parser.add_argument("--target-ratio", type=float, default=DEFAULT_TARGET_RATIO)
    args = parser.parse_args()

    host = args.host or default_windows_host()
    serial = args.serial or f"{host}:{args.port}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"adb: {args.adb}")
    ok, out, err, elapsed = adb_available(args.adb)
    print(f"adb_available: {ok} ({elapsed*1000:.1f}ms)")
    if out.strip():
        print(out.strip())
    if err.strip():
        print(err.strip())
    if not ok:
        print("Hint: pass --adb /mnt/c/path/to/adb.exe, or install android-tools-adb in WSL.")
        return 2

    if not args.skip_connect:
        code, out, err, elapsed = connect(args.adb, host, args.port)
        print(f"connect {host}:{args.port}: code={code} ({elapsed*1000:.1f}ms)")
        if out.strip():
            print(out.strip())
        if err.strip():
            print(err.strip())

    code, out, err, elapsed = run_cmd(adb_cmd(args.adb, "devices"), timeout=10)
    print(f"devices: code={code} ({elapsed*1000:.1f}ms)")
    print(out.strip())
    if err.strip():
        print(err.strip())

    size_cmd = [args.adb, "-s", serial, "shell", "wm", "size"]
    code, out, err, elapsed = run_cmd(size_cmd, timeout=10)
    print(f"wm size: code={code} ({elapsed*1000:.1f}ms)")
    if out.strip():
        print(out.strip())
    if err.strip():
        print(err.strip())

    img, nbytes, elapsed = screencap(args.adb, serial=serial)
    original_img = img
    norm_info = None
    if args.normalize:
        img, norm_info = normalize_portrait_frame(img, args.target_ratio)
    frame_path = out_dir / f"frame_{time.strftime('%Y%m%d_%H%M%S')}.png"
    latest_path = out_dir / "frame_latest.png"
    raw_latest_path = out_dir / "frame_latest_raw.png"
    if args.normalize:
        cv2.imwrite(str(raw_latest_path), original_img)
    cv2.imwrite(str(frame_path), img)
    cv2.imwrite(str(latest_path), img)
    h, w = img.shape[:2]
    print(f"screencap: {elapsed*1000:.1f}ms bytes={nbytes} shape={img.shape} ratio={h/w:.4f}")
    if norm_info is not None:
        print(f"normalize: {norm_info}")
        print(f"saved raw: {raw_latest_path}")
    print(f"saved: {frame_path}")
    print(f"saved: {latest_path}")

    roi = probe_roi(img)
    for key, value in roi.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
