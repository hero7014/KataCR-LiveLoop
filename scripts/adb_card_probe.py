"""Capture and inspect the exact card crops used by the live MuMu pipeline."""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))


def normalize_live_frame(img, target_ratio=2.16):
  """Match SARDaemon's resize_width ADB normalization mode."""
  h, _ = img.shape[:2]
  target_w = int(h / target_ratio)
  if target_w <= 0:
    raise ValueError(f"invalid target ratio {target_ratio}")
  return cv2.resize(img, (target_w, h), interpolation=cv2.INTER_AREA)


def extract_card_crops(img):
  from katacr.build_dataset.constant import part3_bbox_params
  from katacr.build_dataset.utils.split_part import extract_bbox, process_part

  part3 = process_part(img, 3)
  crops = [extract_bbox(part3, *params) for params in part3_bbox_params]
  return part3, crops


def make_contact_sheet(crops, labels, tile_size=(150, 210)):
  width, height = tile_size
  tiles = []
  for index, (crop, label) in enumerate(zip(crops, labels)):
    tile = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (width - 1, height - 1), (0, 255, 255), 2)
    cv2.putText(
      tile, f"{index}: {label}", (6, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
      0.45, (0, 255, 255), 1, cv2.LINE_AA)
    tiles.append(tile)
  return np.concatenate(tiles, axis=1)


def predict_cards(part3):
  from katacr.classification.predict import CardClassifier

  classifier = CardClassifier()
  return list(classifier.process_part3(part3))


def capture_frame(adb, serial):
  from scripts.adb_probe import screencap

  img, nbytes, elapsed = screencap(adb, serial=serial)
  return img, {"capture_bytes": nbytes, "capture_ms": round(elapsed * 1000, 1)}


def save_probe(img, output_dir, predictions, metadata):
  part3, crops = extract_card_crops(img)
  output_dir.mkdir(parents=True, exist_ok=True)
  cv2.imwrite(str(output_dir / "frame.png"), img)
  cv2.imwrite(str(output_dir / "part3.png"), part3)
  for index, crop in enumerate(crops):
    cv2.imwrite(str(output_dir / f"card_{index}.png"), crop)
  cv2.imwrite(str(output_dir / "contact_sheet.png"), make_contact_sheet(crops, predictions))
  manifest = {
    "frame_shape": list(img.shape),
    "part3_shape": list(part3.shape),
    "predictions": predictions,
    **metadata,
  }
  (output_dir / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  return manifest


def main():
  parser = argparse.ArgumentParser(description="Export the live MuMu hand crops used by KataCR card classification.")
  parser.add_argument("--adb", default="adb")
  parser.add_argument("--serial", default="127.0.0.1:16384")
  parser.add_argument("--image", help="Use an existing BGR image instead of ADB screencap.")
  parser.add_argument("--target-ratio", type=float, default=2.16)
  parser.add_argument("--out-dir", default=str(ROOT / "logs" / "card_probe"))
  args = parser.parse_args()

  metadata = {"target_ratio": args.target_ratio}
  if args.image:
    raw = cv2.imread(args.image)
    if raw is None:
      raise FileNotFoundError(args.image)
    metadata["source"] = str(Path(args.image).resolve())
  else:
    raw, capture_metadata = capture_frame(args.adb, args.serial)
    metadata.update(capture_metadata)
    metadata["source"] = "adb"

  frame = normalize_live_frame(raw, args.target_ratio)
  part3, _ = extract_card_crops(frame)
  predictions = predict_cards(part3)
  output_dir = Path(args.out_dir) / time.strftime("%Y%m%d_%H%M%S")
  manifest = save_probe(frame, output_dir, predictions, metadata)
  print(f"saved: {output_dir}")
  print(f"predictions: {manifest['predictions']}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
