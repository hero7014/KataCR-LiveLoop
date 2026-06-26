from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "logs" / "video_eval"


def sanitize_name(name: str) -> str:
    name = name.rstrip(" .")
    return name or "video"


def main() -> int:
    if not ROOT.exists():
      print(f"Missing directory: {ROOT}")
      return 1

    changed = 0
    for run_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        for child in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            cleaned = sanitize_name(child.name)
            if cleaned == child.name:
                continue
            target = run_dir / cleaned
            if target.exists():
                print(f"Skip existing target: {target}")
                continue
            shutil.copytree(child, target)
            changed += 1
            print(f"Copied:\n  from {child}\n  to   {target}")

    if changed == 0:
        print("No legacy result directories needed sanitizing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
