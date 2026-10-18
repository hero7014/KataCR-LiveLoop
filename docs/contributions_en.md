# Contribution Boundary

[中文](contributions.md) | [English](contributions_en.md)

KataCR LiveLoop is a personal engineering adaptation of [wty-yy/KataCR](https://github.com/wty-yy/KataCR). It keeps the upstream research code and policy capabilities while adding a reproducible runtime path for WSL2, NVIDIA GPUs, video evaluation, and the MuMu emulator.

## Upstream foundation

The following capabilities are inherited from the upstream project:

- KataCR's research goal, perception-to-policy abstraction, and core layout.
- YOLOv8 detection, PaddleOCR, card classification, and the original visual features.
- StARformer, DT, their training workflows, datasets, and model-weight documentation.
- The original data generation, replay processing, visualization, and training scripts.

## Work in this adaptation

- WSL2, Ubuntu 24.04, Python 3.11, and NVIDIA GPU deployment paths.
- Verified JAX, YOLO, Paddle, and PaddleOCR GPU paths with CPU fallback.
- Diagnostics for CUDA compilation tools, JAX/Orbax checkpoint restoration, and optional weights.
- Parameterized offline evaluation for inputs, outputs, saving, display, deterministic runs, and verbose logs.
- ROI, OCR, card/elixir, state-fusion, and stage-level timing diagnostics.
- Keyframe sampling, lightweight replay export, result sanitization, and frame-log analysis.
- MuMu/WSL2 ADB probing, screenshot capture, page-state recognition, coordinate calibration, and action throttling.
- Windows launch helpers, WSL runners, and GPU/ADB checks.

## Verification scope

“Verified” means exercised in the author's WSL2, NVIDIA GPU, and MuMu environment. It is not a guarantee for every device, driver, or game version. MuMu automation is currently a trainer or low-risk experiment entry point; ladder play and unattended long-running operation are outside the current scope.

ADB action execution is screen input, not a claim that the upstream policy model was retrained. Model quality should be evaluated with recognition accuracy, action validity, latency, and multi-match results rather than a single win or loss.

## Attribution and license

Upstream code and materials remain subject to the upstream project and dataset/model licenses. The runtime adaptation, diagnostic scripts, launch helpers, and supporting documentation added here are maintained by `hero7014`; individual files and history remain the authoritative record.
