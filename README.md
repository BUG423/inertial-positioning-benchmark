<div align="center">

# 🧭 Inertial Positioning Benchmark

### 面向惯性定位研究的统一数据接口、采集工具与可复现实验平台

[![CI](https://github.com/BUG423/inertial-positioning-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/BUG423/inertial-positioning-benchmark/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Android](https://img.shields.io/badge/Android-API%2026%2B-3DDC84?logo=android&logoColor=white)
![Status](https://img.shields.io/badge/status-active-2ea44f)

**🇨🇳 中文** · [🇬🇧 English](README_EN.md)

[快速开始](#-快速开始) · [目录结构](#-目录结构) · [研究内容](#-研究内容) · [路线图](#-路线图)

</div>

---

## 🌟 项目简介

不同惯性里程计项目往往采用不同的数据格式、坐标系、窗口定义和评测协议，导致方法难以公平比较、实验难以完整复现。本仓库将以下内容统一组织在一套边界清晰的研究基础设施中：

- 📦 **标准数据格式 IPB v1**：统一 200 Hz 时间轴、重力对齐坐标系、单位、HDF5 存储、划分与转换报告；
- 🏋️ **训练与评测框架**：Ultralytics 风格的 `ipb` 命令行与 `NIO` Python API，统一任务视图、训练循环与轨迹级指标；
- 📱 **数据采集工具**：使用 Android 设备采集 IMU 与 ARCore VIO 参考轨迹；
- 🧪 **研究内容**：保存方法提案、实验研究和相关图表；
- 🧠 **科研工作流**：仅保留 PaperFlow 中面向惯性里程计的 IO 技能；
- 📚 **环境复现文档**：归档 FAST-LIVO 在 Ubuntu 20.04 + ROS Noetic 下的依赖、编译与测试记录；
- ✅ **自动化检查**：验证 Python 核心、Android 工程、转换工具和文档链接。

目前，IPB v1 数据层、训练/评测框架、RoNIN 参考基线与 Android 工具已经可用；各公开数据集转换器、更多基线与排行榜仍在建设中。
设计约束见 [设计宪法](docs/DESIGN.md)。

### 数据处理流程

```text
公开数据集 / 设备采集
        ↓  ipb convert（解析 → 200 Hz 重采样 → 掩码 → 重力等校验 → HDF5 + 划分 + 清单）
IPB v1 序列（Sequence）
        ↓  任务视图（坐标系、窗口、目标、增强）
ipb train / ipb val（模型 → 窗口速度 → 统一轨迹重建）
        ↓
轨迹级与窗口级指标 → ipb benchmark / ipb report（表格、统计检验、论文图）
```

> [!IMPORTANT]
> `research/methods/` 中的 PostDiffIO 与 ModeMoEIO 当前属于**方法提案**，不代表已经获得可复现的 benchmark 成绩。只有代码、配置、数据划分和结果均可审计的方法，才会进入正式基线。

## 🧩 当前组件

| 组件 | 状态 | 说明 |
| --- | --- | --- |
| IPB v1 数据层 | ✅ 可用 | `Sequence` 读写与校验、200 Hz 重采样、统一转换流水线、分组划分与泄漏检查 |
| 训练与评测框架 | ✅ 可用 | `ipb` CLI / `NIO` API、配置系统、Trainer / Validator / Predictor、指标与报表 |
| RoNIN ResNet 基线 | ✅ 可用 | 依据论文与官方仓库独立实现，参数量与官方逐一对齐 |
| 公开数据集转换器 | 🚧 建设中 | RoNIN、RIDI、OxIOD、TLIO、IDOL、RNIN、IMUNet、PedLocData |
| Inertial Positioning Lab | ✅ 可用 | Android 采集、`.iplab` 归档、数据转换与端侧推理 |
| Pedestrian Coordinate Frames | 📊 研究记录 | Global frame 与 body frame 的实验分析 |
| PostDiffIO | 💡 方法提案 | 条件扩散后验细化与不确定性建模 |
| ModeMoEIO | 💡 方法提案 | 面向不同运动状态的混合专家路由 |
| PaperFlow IO | 🛠️ 研究工具 | IO 选题、文献复核、可行性评审与实验规划 |
| FAST-LIVO Setup | 📚 环境文档 | ROS Noetic 下的 FAST-LIVO 历史复现与依赖说明 |
| 更多基线与排行榜 | 🚧 规划中 | TLIO、RNIN、IMUNet 等基线与版本化结果 |

## 🗂️ 目录结构

```text
inertial-positioning-benchmark/
├── src/inertial_benchmark/       # IPB 核心：cfg / data / nn / models / engine / metrics / utils / cli
├── tests/                        # 核心、集成与文档一致性测试
├── docs/                         # 数据规范、调研、环境复现与项目约定
│   └── setup/fast-livo.md         # FAST-LIVO 历史环境复现指南
├── tools/
│   ├── inertial-positioning-lab/ # Android 采集与端侧评测工具
│   └── paperflow-io/             # 仅面向 IO 的科研工作流
├── research/
│   ├── methods/                  # 尚未纳入正式基线的方法提案
│   └── studies/                  # 可追溯的实验研究与资源
├── .github/workflows/            # 全仓唯一的 CI 与发布流程
└── pyproject.toml                # Python 项目与依赖配置
```

| 目录 | 内容入口 |
| --- | --- |
| `docs/` | [文档索引](docs/README.md) · [设计宪法](docs/DESIGN.md) · [指标](docs/METRICS.md) · [命令行](docs/CLI.md) · [v0.1 数据格式](docs/FORMAT.md) · [数据集](docs/DATASETS.md) · [论文](docs/PAPERS.md) · [FAST-LIVO 配置](docs/setup/fast-livo.md) |
| `tools/` | [工具索引](tools/README.md) · [Android 工具](tools/inertial-positioning-lab/) · [PaperFlow IO](tools/paperflow-io/) |
| `research/` | [研究索引](research/README.md) · [方法提案](research/methods/) · [实验研究](research/studies/) |

目录职责、命名方式和新组件准入条件见[仓库目录规范](docs/REPOSITORY_LAYOUT.md)。

## 🚀 快速开始

### 1. 获取并安装

```bash
git clone https://github.com/BUG423/inertial-positioning-benchmark.git
cd inertial-positioning-benchmark
python -m pip install -e ".[train,test]"   # 只用数据层时：pip install -e .
```

项目要求 Python 3.9 或更高版本；核心数据层只依赖 numpy / h5py / scipy / pyyaml，训练与报表需要 `train` 可选依赖（PyTorch、pandas、matplotlib）。

### 2. 运行测试

```bash
pytest -q                 # 全部测试（含约 20 秒的端到端冒烟测试）
pytest -q -m "not slow"   # 跳过端到端冒烟测试
```

测试覆盖数据格式、重采样、转换流水线、任务视图、模型、指标解析解、训练引擎、命令行、Android 工具集成与文档链接，全部使用合成数据。

### 3. 转换并检查数据集

```bash
export IPB_DATASETS=~/datasets/ipb
ipb convert dataset=ronin source=/path/to/raw/ronin workers=8
ipb check data=ronin full=true
```

转换结果包含 `sequences/*.h5`、`splits/*.txt`、`dataset.json` 与 `conversion_report.json`，格式见 [设计宪法](docs/DESIGN.md) 第 2 节。

### 4. 训练、评测与推理

```bash
ipb train model=ronin_resnet18 data=ronin epochs=40 device=0
ipb val   model=runs/train/exp/weights/best.pt data=ronin split=test_unseen
ipb predict model=runs/train/exp/weights/best.pt source=path/to/sequence.h5
```

等价的 Python API：

```python
from inertial_benchmark import NIO, load_sequence

model = NIO("ronin_resnet18")
model.train(data="ronin", epochs=40, device=0)
result = model.val(data="ronin", split="test")      # RunResult：逐序列与聚合指标
traj = model.predict("path/to/sequence.h5")          # Trajectory：统一时间轴上的轨迹

seq = load_sequence("path/to/sequence.h5")           # 数据层不依赖 PyTorch；也可读取 v0.1 文件
```

### 5. 基准矩阵与报告

```bash
ipb benchmark cfg=main                    # 模型 × 数据集 × 种子（cfg/benchmarks/main.yaml），已完成项自动跳过
ipb report runs=runs/benchmark/main out=reports/main
```

报告包含均值/中位数、按 `group_id` 的 bootstrap 置信区间、多种子统计、配对 Wilcoxon 检验、CSV/Markdown/LaTeX 表格和论文图。
命令与配置键详见 [命令行文档](docs/CLI.md)，指标定义详见 [指标文档](docs/METRICS.md)。

### 6. 转换 Android 采集数据

```bash
python tools/inertial-positioning-lab/tools/convert_dataset.py \
  recording.iplab \
  --output datasets/android
```

生成的 v0.1 文件可直接由 `load_sequence()` 读取（内存中重采样为 v1）。格式详情请参阅 [Inertial Positioning Lab](tools/inertial-positioning-lab/)。

### 7. 构建 Android 应用

```bash
cd tools/inertial-positioning-lab
./gradlew testDebugUnitTest lintDebug assembleDebug
```

Debug APK 位于 `app/build/outputs/apk/debug/app-debug.apk`。

## 🔬 研究内容

- 💡 [PostDiffIO](research/methods/postdiffio/)：条件扩散速度残差细化与不确定性建模提案；
- 💡 [ModeMoEIO](research/methods/moe-io/)：运动模式感知混合专家提案；
- 📊 [Pedestrian Coordinate Frames](research/studies/pedestrian-coordinate-frames/)：global-frame 与 body-frame 表征研究；
- 🧠 [PaperFlow IO](tools/paperflow-io/)：仅包含 IO 选题、文献复核、可行性评审与实验设计技能；
- 📚 [FAST-LIVO 环境配置](docs/setup/fast-livo.md)：Ubuntu 20.04 与 ROS Noetic 环境下的历史复现说明。

## 🛣️ 路线图

- [x] 定义统一序列结构、坐标系与单位约定
- [x] 实现 HDF5 I/O、窗口数据集和核心测试
- [x] 集成 Android 数据采集与端侧评测工具
- [x] 整合研究方法、坐标系研究和 PaperFlow IO
- [x] 归档并整合 FAST-LIVO 的 ROS 1 环境复现指南
- [x] 制定 IPB v1 设计宪法：200 Hz 统一格式、任务视图、推理协议与指标体系
- [x] 实现统一转换流水线、分组划分与泄漏检查
- [x] 实现训练/评测框架（`ipb` CLI、`NIO` API、基准矩阵与报表）
- [x] 集成 RoNIN ResNet 参考基线（参数量与官方实现对齐）
- [x] 固化轨迹级与窗口级评测指标（[METRICS.md](docs/METRICS.md)）
- [ ] 接入八个核心公开数据集的转换器并发布划分与指纹
- [ ] 集成 TLIO、RNIN、IMUNet 等更多可复现基线
- [ ] 在真实数据上完成基准矩阵并发布版本化结果与排行榜

## 🤝 贡献与许可

本仓库只维护 `main`。提交内容前请阅读[贡献规范](CONTRIBUTING.md)，并为数据集、模型和第三方资源记录来源、许可证与再分发限制。

仓库级统一开源许可证尚未确定。源码公开不自动授予再分发或商业使用权；带独立许可证的组件继续遵循其目录中的条款。

---

<div align="center">

如果这个项目对你的研究有帮助，欢迎 ⭐ Star 并关注后续 benchmark 更新。

</div>
