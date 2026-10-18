# KataCR LiveLoop Architecture

[中文](architecture.md) | [English](architecture_en.md)

## 中文

### 运行链路

```mermaid
flowchart LR
    A[对局视频<br/>或 MuMu 截屏] --> B[输入归一化<br/>ROI 裁剪]
    B --> C[YOLOv8<br/>单位与防御塔检测]
    B --> D[PaddleOCR<br/>文字与血量识别]
    B --> E[ResNet 分类器<br/>手牌与圣水]
    C --> F[Visual Fusion<br/>视觉结果融合]
    D --> F
    E --> F
    F --> G[SARBuilder<br/>state / action / reward]
    G --> H[JAX 策略模型<br/>DT / StARformer]
    H --> I[ActionBuilder<br/>动作过滤与坐标映射]
    I --> J{运行模式}
    J --> K[离线评估<br/>日志 / 关键帧 / 回放]
    J --> L[MuMu ADB<br/>截屏与点击]
```

### 代码分层

```mermaid
flowchart TB
    UI[入口与脚本<br/>run_offline_eval.sh / launch_katacr_mumu_dt.bat]
    UI --> ENV[运行环境<br/>VideoEnv / InteractEnv]
    ENV --> PERCEPTION[感知层<br/>YOLOv8 / OCR / 分类器]
    PERCEPTION --> FUSION[融合层<br/>VisualFusion / SARBuilder]
    FUSION --> POLICY[策略层<br/>DT / StARformer / JAX]
    POLICY --> ACT[执行层<br/>ActionBuilder / ADB]
    ACT --> OBS[观测与诊断<br/>JSONL / keyframes / replay]
```

## English

### Runtime flow

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

The same modules support two entry points: recorded-video evaluation and a MuMu ADB experiment loop. The policy is inherited from KataCR; this repository concentrates on runtime integration, diagnostics, and reproducibility.
