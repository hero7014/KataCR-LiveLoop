from __future__ import annotations

import importlib
from pathlib import Path


MODULES = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("cv2", "opencv-python"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("ultralytics", "ultralytics"),
    ("jax", "jax"),
    ("jaxlib", "jaxlib"),
    ("flax", "flax"),
    ("optax", "optax"),
    ("orbax.checkpoint", "orbax-checkpoint"),
    ("paddleocr", "paddleocr"),
]


def main() -> int:
    failed = False
    print("== Import check ==")
    for module_name, label in MODULES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "NO_VERSION")
            print(f"OK   {label:<18} {version}")
        except Exception as exc:
            failed = True
            print(f"FAIL {label:<18} {type(exc).__name__}: {exc}")

    root = Path(__file__).resolve().parents[1]
    print("\n== Asset check ==")
    assets = [
        root / "runs" / "detector1_v0.7.13.pt",
        root / "runs" / "detector2_v0.7.13.pt",
        root / "logs" / "CardClassification-checkpoints" / "010",
        root / "logs" / "Policy" / "StARformer_no_delay_2L_v0.8_golem_ai_cnn_blocks__nbc128__ep20__step50__0__20240520_205252" / "ckpt" / "001",
    ]
    for path in assets:
        state = "OK" if path.exists() else "MISS"
        print(f"{state:<4} {path}")
        if not path.exists():
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
