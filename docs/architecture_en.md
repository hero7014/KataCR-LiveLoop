# KataCR LiveLoop Architecture

[中文](architecture.md) | [English](architecture_en.md)

## Runtime flow

```mermaid
flowchart LR
    A[Match video<br/>or MuMu screenshot] --> B[Input normalization<br/>and ROI cropping]
    B --> C[YOLOv8<br/>unit and tower detection]
    B --> D[PaddleOCR<br/>text and health bars]
    B --> E[ResNet classifier<br/>cards and elixir]
    C --> F[Visual Fusion<br/>perception merge]
    D --> F
    E --> F
    F --> G[SARBuilder<br/>state / action / reward]
    G --> H[JAX policy<br/>DT / StARformer]
    H --> I[ActionBuilder<br/>filter and coordinate mapping]
    I --> J{Execution mode}
    J --> K[Offline evaluation<br/>logs / keyframes / replay]
    J --> L[MuMu ADB<br/>screenshots and taps]
```

## Module layers

```mermaid
flowchart TB
    UI[Entry points and scripts<br/>run_offline_eval.sh / launch_katacr_mumu_dt.bat]
    UI --> ENV[Runtime environments<br/>VideoEnv / InteractEnv]
    ENV --> PERCEPTION[Perception<br/>YOLOv8 / OCR / classifiers]
    PERCEPTION --> FUSION[Fusion<br/>VisualFusion / SARBuilder]
    FUSION --> POLICY[Policy<br/>DT / StARformer / JAX]
    POLICY --> ACT[Execution<br/>ActionBuilder / ADB]
    ACT --> OBS[Observability<br/>JSONL / keyframes / replay]
```

The project has two practical paths:

1. Recorded video enters the perception and policy stack, producing frame-level actions and analysis artifacts.
2. MuMu provides screenshots over ADB; the same perception and policy stack produces filtered screen actions for low-risk experiments.

The inherited policy model remains separate from the runtime work added in this adaptation. See [`docs/contributions_en.md`](contributions_en.md) for the attribution boundary.
