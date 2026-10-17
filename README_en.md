# KataCR LiveLoop

An engineering adaptation of [wty-yy/KataCR](https://github.com/wty-yy/KataCR), organized for a reproducible **WSL2 + NVIDIA GPU + MuMu emulator** workflow.

> This is a Fork/adaptation of the upstream KataCR project. The original policy models, datasets, research architecture, and core algorithm design belong to the upstream author. This repository focuses on runtime integration, performance work, diagnostics, and the MuMu live loop developed in this adaptation.

## Pipeline

```text
Video or MuMu screenshot
    -> ROI cropping
    -> YOLO detection + PaddleOCR
    -> card, elixir, and arena state fusion
    -> JAX/Orbax policy inference
    -> action-to-screen coordinate mapping
    -> ADB input to MuMu
    -> logs, keyframes, and replay analysis
```

The project keeps two entry points:

- **Offline evaluation**: inspect perception, state construction, policy output, and action logs from a recorded match.
- **MuMu live experiments**: capture the emulator through ADB, identify the page state, and execute actions in trainer or other low-risk modes.

## Main work in this adaptation

- GPU runtime paths for JAX, YOLO, Paddle, and PaddleOCR under WSL2, with CPU fallback.
- Parameterized offline evaluation for video paths, output directories, saving, display, deterministic runs, and verbose logs.
- Deployment fixes for CUDA compilation tools, JAX/Orbax checkpoint restoration, and missing auxiliary weights.
- ROI, card, OCR, elixir, and arena-state diagnostics, including keyframe and lightweight replay export.
- MuMu ADB capture, page-state recognition, trainer flow handling, and action coordinate mapping.
- Emulator aspect-ratio, window-offset, action-throttling, and recognition-stability diagnostics.
- Windows and WSL launch helpers for reproducible local testing.

See [`docs/contributions.md`](docs/contributions.md) for the authorship boundary.

## Status and scope

The video-to-policy path and the basic MuMu ADB action loop have been exercised locally. Recognition stability, action timing, coordinate calibration, and the practical quality of the inherited policy remain active engineering topics.

The current scope is offline evaluation and trainer/low-risk experiments. Ladder play and unattended long-running operation are outside the current promise.

## Environment

- MuMu emulator on Windows with ADB debugging enabled.
- WSL2 with Ubuntu 24.04.
- Python 3.11 virtual environment.
- NVIDIA GPU with framework CUDA versions matched to the installed driver.

Do not commit personal absolute paths, model weights, videos, logs, or account information. Pass paths through environment variables or command-line arguments.

## Documentation

- [Public Windows launch guide](docs/windows_launch_public_zh-CN.md)
- [Contribution ledger](docs/contributions.md)
- [Public release checklist](docs/publication.md)
- [Upstream PR map](docs/upstream_pr_map.md)

## Upstream and attribution

- Upstream project: [wty-yy/KataCR](https://github.com/wty-yy/KataCR)
- Upstream detection dataset: [Clash-Royale-Detection-Dataset](https://github.com/wty-yy/Clash-Royale-Detection-Dataset)
- Upstream replay dataset: [Clash-Royale-Replay-Dataset](https://github.com/wty-yy/Clash-Royale-Replay-Dataset)

This repository follows the upstream MIT License. Use upstream code, models, datasets, and demonstration materials according to their licenses and author notices. The additional engineering code and documentation in this adaptation are maintained by `hero7014`; see [`docs/contributions.md`](docs/contributions.md) for the boundary.
