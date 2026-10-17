import argparse
import sys
import time
from pathlib import Path

import cv2
import paddle

sys.path.append(str(Path(__file__).parents[1]))

from katacr.build_dataset.utils.split_part import process_part
from katacr.ocr_text.paddle_ocr import OCR


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--video",
    required=True,
    help="Path to a local test video.",
  )
  args = parser.parse_args()

  print("paddle", paddle.__version__)
  print("compiled_cuda", paddle.device.is_compiled_with_cuda())
  ocr = OCR(use_gpu=True, lang="en")
  print("device", paddle.device.get_device())

  cap = cv2.VideoCapture(args.video)
  ok, frame = cap.read()
  cap.release()
  if not ok:
    raise RuntimeError(f"failed to read video frame: {args.video}")

  part1 = process_part(frame, 1)
  part3 = process_part(frame, 3)
  for i in range(3):
    t = time.perf_counter()
    battle_time = ocr.process_part1(part1, pil=False)
    dt_time = time.perf_counter() - t
    t = time.perf_counter()
    elixir = ocr.process_part3_elixir(part3, pil=False)
    dt_elixir = time.perf_counter() - t
    print(
      f"run={i} time={battle_time} time_ocr={dt_time*1e3:.1f}ms "
      f"elixir={elixir} elixir_ocr={dt_elixir*1e3:.1f}ms"
    )


if __name__ == "__main__":
  main()
