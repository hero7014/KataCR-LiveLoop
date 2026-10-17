import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import cv2

from scripts.adb_probe import connect, screencap


DEFAULT_ADB = os.environ.get("KATACR_ADB", "adb")
DEFAULT_SERIAL = "127.0.0.1:16384"

# Coordinates are Android screencap coordinates for the current 720x1280 MuMu setup.
COORDS = {
    "battle": (360, 1000),
    "confirm": (365, 1120),
    "reward_chest": (135, 1015),
    "screen_center": (360, 640),
}

ROIS = {
    "battle_button": (235, 895, 500, 1080),
    "confirm_button": (260, 1095, 470, 1235),
    "battle_backdrop": (35, 100, 685, 930),
}


class PageState(str, Enum):
    HOME = "home"
    POST_MATCH = "post_match"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class UiDecision:
    requested: str
    state: str
    action: str
    allowed: bool
    reason: str


def _roi(img, box):
    x1, y1, x2, y2 = box
    return img[y1:y2, x1:x2]


def _color_ratio(region, predicate):
    if region.size == 0:
        return 0.0
    mask = predicate(region)
    return float(mask.mean())


def detect_page_signals(img):
    battle = _roi(img, ROIS["battle_button"])
    confirm = _roi(img, ROIS["confirm_button"])
    backdrop = _roi(img, ROIS["battle_backdrop"])

    battle_yellow = _color_ratio(
        battle,
        lambda px: (px[:, :, 2] > 190) & (px[:, :, 1] > 110) & (px[:, :, 0] < 90),
    )
    confirm_blue = _color_ratio(
        confirm,
        lambda px: (px[:, :, 0] > 170) & (px[:, :, 1] > 90) & (px[:, :, 2] < 130),
    )
    backdrop_dim = _color_ratio(
        backdrop,
        lambda px: (px[:, :, 0] < 130) & (px[:, :, 1] < 130) & (px[:, :, 2] < 130),
    )
    return {
        "battle_yellow": battle_yellow,
        "confirm_blue": confirm_blue,
        "backdrop_dim": backdrop_dim,
    }


def detect_page_state(img):
    signals = detect_page_signals(img)

    if signals["confirm_blue"] > 0.18 and signals["backdrop_dim"] > 0.20:
        return PageState.POST_MATCH
    if signals["battle_yellow"] > 0.16:
        return PageState.HOME
    return PageState.UNKNOWN


def resolve_action(requested, state, confirmed=False, force=False):
    state = PageState(state)
    requested = requested.replace("_", "-")

    if requested == "next":
        if state == PageState.HOME:
            action = "battle"
            allowed = confirmed
            reason = "home detected; battle is the next suggested action"
        elif state == PageState.POST_MATCH:
            action = "post-match"
            allowed = confirmed
            reason = "post-match screen detected; confirm and rewards are suggested"
        else:
            action = "none"
            allowed = False
            reason = "page state is unknown; no click will be sent"
        if force and action != "none":
            allowed = True
            reason = f"{reason}; force enabled"
        return UiDecision(requested, state.value, action, allowed, reason)

    expected = {
        "battle": PageState.HOME,
        "confirm": PageState.POST_MATCH,
        "post-match": PageState.POST_MATCH,
        "post-match-and-battle": PageState.POST_MATCH,
    }
    if force:
        return UiDecision(requested, state.value, requested, True, "force enabled")

    if requested in expected:
        ok = state == expected[requested] and confirmed
        reason = (
            f"{expected[requested].value} detected; {requested} is allowed"
            if ok else
            f"{requested} requires {expected[requested].value}, got {state.value}"
        )
        if state == expected[requested] and not confirmed:
            reason = f"{requested} is suggested, but confirmation was not provided"
        return UiDecision(requested, state.value, requested, ok, reason)

    return UiDecision(requested, state.value, requested, confirmed, "explicit action has no page guard")


def adb_base(adb, serial):
    return [adb, "-s", serial]


def run(args, timeout=10):
    import subprocess

    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    out = proc.stdout.decode("utf-8", errors="replace").strip()
    err = proc.stderr.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(args)}\n{out}\n{err}")
    return out


def tap(adb, serial, x, y, label=None):
    name = f" {label}" if label else ""
    print(f"tap{name}: ({x}, {y})", flush=True)
    run([*adb_base(adb, serial), "shell", "input", "tap", str(int(x)), str(int(y))])


def capture(adb, serial, out_dir):
    img, nbytes, elapsed = screencap(adb, serial=serial)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ui_{time.strftime('%Y%m%d_%H%M%S')}.png"
    latest = out_dir / "ui_latest.png"
    cv2.imwrite(str(path), img)
    cv2.imwrite(str(latest), img)
    print(f"capture: {elapsed*1000:.1f}ms bytes={nbytes} shape={img.shape} saved={path}", flush=True)
    return path, img


def print_decision(decision):
    print(json.dumps(asdict(decision), ensure_ascii=False), flush=True)


def open_chest(adb, serial, taps=6, interval=0.35, last_wait=1.2):
    # Tap a visible chest/reward area, then keep tapping the center to advance reward pages.
    tap(adb, serial, *COORDS["reward_chest"], label="reward_chest")
    time.sleep(interval)
    for i in range(max(0, taps - 1)):
        wait = last_wait if i == taps - 2 else interval
        tap(adb, serial, *COORDS["screen_center"], label=f"chest_advance_{i+2}/{taps}")
        time.sleep(wait)


def post_match_to_home(adb, serial, chest_taps=6):
    tap(adb, serial, *COORDS["confirm"], label="confirm")
    time.sleep(1.5)
    open_chest(adb, serial, taps=chest_taps)


def execute_action(action, adb, serial, chest_taps=6):
    if action == "battle":
        tap(adb, serial, *COORDS["battle"], label="battle")
    elif action == "confirm":
        tap(adb, serial, *COORDS["confirm"], label="confirm")
    elif action == "open-chest":
        open_chest(adb, serial, taps=chest_taps)
    elif action == "post-match":
        post_match_to_home(adb, serial, chest_taps=chest_taps)
    elif action == "post-match-and-battle":
        post_match_to_home(adb, serial, chest_taps=chest_taps)
        time.sleep(1.5)
        tap(adb, serial, *COORDS["battle"], label="battle")
    elif action == "none":
        return
    else:
        raise ValueError(f"Unknown action: {action}")


def main():
    parser = argparse.ArgumentParser(description="Small, explicit MuMu ADB UI actions for Clash Royale.")
    parser.add_argument("action", choices=[
        "capture",
        "suggest-next",
        "act-next",
        "battle",
        "confirm",
        "open-chest",
        "post-match",
        "post-match-and-battle",
    ])
    parser.add_argument("--adb", default=DEFAULT_ADB)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=16384)
    parser.add_argument("--serial", default=DEFAULT_SERIAL)
    parser.add_argument("--skip-connect", action="store_true")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "logs" / "adb_ui_flow"))
    parser.add_argument("--chest-taps", type=int, default=6)
    parser.add_argument("--confirm-execute", action="store_true", help="Actually send taps for guarded actions.")
    parser.add_argument("--force", action="store_true", help="Bypass page-state guard for explicit manual recovery.")
    args = parser.parse_args()

    if not args.skip_connect:
        code, out, err, elapsed = connect(args.adb, args.host, args.port)
        print(f"connect {args.host}:{args.port}: code={code} ({elapsed*1000:.1f}ms)", flush=True)
        if out.strip():
            print(out.strip(), flush=True)
        if err.strip():
            print(err.strip(), flush=True)

    if args.action == "capture":
        path, img = capture(args.adb, args.serial, Path(args.out_dir))
        print(f"state={detect_page_state(img).value}", flush=True)
    elif args.action in {"suggest-next", "act-next"}:
        _, img = capture(args.adb, args.serial, Path(args.out_dir))
        state = detect_page_state(img)
        decision = resolve_action("next", state, confirmed=args.confirm_execute, force=args.force)
        print_decision(decision)
        if decision.allowed:
            execute_action(decision.action, args.adb, args.serial, chest_taps=args.chest_taps)
    else:
        _, img = capture(args.adb, args.serial, Path(args.out_dir))
        state = detect_page_state(img)
        confirmed = args.confirm_execute or args.action == "open-chest"
        decision = resolve_action(args.action, state, confirmed=confirmed, force=args.force)
        print_decision(decision)
        if decision.allowed:
            execute_action(decision.action, args.adb, args.serial, chest_taps=args.chest_taps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
