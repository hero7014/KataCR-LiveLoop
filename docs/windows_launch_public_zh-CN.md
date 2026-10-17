# 公开运行说明

本文档只保留可公开复现的入口，不包含个人机器的绝对路径、账号信息或本地运行镜像位置。

## 环境

1. Windows 侧启动 MuMu，并在模拟器设置中开启 ADB 调试。
2. WSL2 中创建 Python 3.11 虚拟环境并安装项目依赖。
3. 使用 `nvidia-smi`、JAX、Paddle 和 PyTorch 的检查脚本确认 GPU 可见。
4. 通过环境变量提供项目目录、模型目录、视频路径和 ADB 可执行文件路径。

示例：

```bash
export KATACR_SOURCE_ROOT=/path/to/KataCR
export KATACR_RUNTIME_ROOT=/path/to/runtime
export KATACR_ADB=/path/to/adb
```

## 离线评估

先使用视频入口验证感知和策略输出，再查看生成的日志、关键帧和回放。视频、模型和日志建议放在仓库外，并通过命令行参数传入。

```bash
./run_offline_eval.sh /path/to/match.mp4
```

## MuMu 实验

MuMu 实时入口依赖 ADB 截屏和点击。首次运行只建议使用训练营或其他低风险模式，并先执行 ADB 探测脚本确认目标设备、画面尺寸和坐标比例。

实时执行前应确认：

- 模拟器窗口尺寸和项目配置一致；
- 页面状态已经识别为可进入对局的状态；
- 动作坐标经过当前分辨率校准；
- 日志目录位于仓库外，且不会被提交。

## 不提交的内容

不要提交个人绝对路径、账号信息、视频、模型权重、截图、运行日志、CUDA 安装包或设备专用配置。公开仓库只保留通用代码、参数说明和可复现的最小示例。
