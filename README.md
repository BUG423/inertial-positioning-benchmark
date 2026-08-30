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

- 📦 **标准数据接口**：统一时间戳、坐标系、单位、HDF5 存储与窗口化；
- 📱 **数据采集工具**：使用 Android 设备采集 IMU 与 ARCore VIO 参考轨迹；
- 🧪 **研究内容**：保存方法提案、实验研究和相关图表；
- 🧠 **科研工作流**：仅保留 PaperFlow 中面向惯性里程计的 IO 技能；
- ✅ **自动化检查**：验证 Python 核心、Android 工程、转换工具和文档链接。

目前，统一数据层与 Android 工具已经可用；公开数据集适配器、标准基线、固定评测协议和排行榜仍在建设中。

### 数据处理流程

```text
设备采集 / 公开数据集
        ↓
CanonicalSequence（统一时间戳、坐标系与单位）
        ↓
HDF5 持久化 + WindowDataset 窗口化
        ↓
模型训练 / 方法评测 / 端侧部署
```

> [!IMPORTANT]
> `research/methods/` 中的 PostDiffIO 与 ModeMoEIO 当前属于**方法提案**，不代表已经获得可复现的 benchmark 成绩。只有代码、配置、数据划分和结果均可审计的方法，才会进入正式基线。

## 🧩 当前组件

| 组件 | 状态 | 说明 |
| --- | --- | --- |
| Benchmark Core | ✅ 可用 | `CanonicalSequence`、HDF5 I/O、窗口数据集与测试 |
| Inertial Positioning Lab | ✅ 可用 | Android 采集、`.iplab` 归档、数据转换与端侧推理 |
| Pedestrian Coordinate Frames | 📊 研究记录 | Global frame 与 body frame 的实验分析 |
| PostDiffIO | 💡 方法提案 | 条件扩散后验细化与不确定性建模 |
| ModeMoEIO | 💡 方法提案 | 面向不同运动状态的混合专家路由 |
| PaperFlow IO | 🛠️ 研究工具 | IO 选题、文献复核、可行性评审与实验规划 |\n| FAST-LIVO Setup | 📚 环境文档 | ROS Noetic 下的 FAST-LIVO 历史复现与依赖说明 |
| Public Baselines & Leaderboard | 🚧 规划中 | 数据适配器、固定划分、统一指标与排行榜 |

## 🗂️ 目录结构

```text
inertial-positioning-benchmark/
├── src/inertial_benchmark/       # Python 核心数据接口
├── tests/                        # 核心、集成与文档一致性测试
├── docs/                         # 数据规范、数据集调研、环境配置与项目约定
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
| `docs/` | [文档索引](docs/README.md) · [数据格式](docs/FORMAT.md) · [数据集](docs/DATASETS.md) · [论文](docs/PAPERS.md) · [FAST-LIVO 配置](docs/setup/fast-livo.md) |
| `tools/` | [工具索引](tools/README.md) · [Android 工具](tools/inertial-positioning-lab/) · [PaperFlow IO](tools/paperflow-io/) |
| `research/` | [研究索引](research/README.md) · [方法提案](research/methods/) · [实验研究](research/studies/) |

目录职责、命名方式和新组件准入条件见[仓库目录规范](docs/REPOSITORY_LAYOUT.md)。

## 🚀 快速开始

### 1. 获取并安装

```bash
git clone https://github.com/BUG423/inertial-positioning-benchmark.git
cd inertial-positioning-benchmark
python -m pip install -e ".[test]"
```

项目要求 Python 3.9 或更高版本。

### 2. 运行测试

```bash
pytest -q
```

该命令检查 Python 核心接口、Android 工具集成状态以及仓库内的相对文档链接。

### 3. 读取统一格式数据

```python
from inertial_benchmark import CanonicalSequence, WindowDataset

sequence = CanonicalSequence.from_hdf5("sequence.h5")
dataset = WindowDataset(
    sequence,
    window_size=200,
    stride=10,
    target="velocity",
)
sample = dataset[0]
```

核心数据层不依赖 PyTorch，可以由不同训练框架继续封装。

### 4. 转换 Android 采集数据

```bash
python tools/inertial-positioning-lab/tools/convert_dataset.py \
  recording.iplab \
  --output datasets/android
```

格式详情请参阅 [Inertial Positioning Lab](tools/inertial-positioning-lab/)。

### 5. 构建 Android 应用

```bash
cd tools/inertial-positioning-lab
./gradlew testDebugUnitTest lintDebug assembleDebug
```

Debug APK 位于 `app/build/outputs/apk/debug/app-debug.apk`。

## 🔬 研究内容

- 💡 [PostDiffIO](research/methods/postdiffio/)：条件扩散速度残差细化与不确定性建模提案；
- 💡 [ModeMoEIO](research/methods/moe-io/)：运动模式感知混合专家提案；
- 📊 [Pedestrian Coordinate Frames](research/studies/pedestrian-coordinate-frames/)：global-frame 与 body-frame 表征研究；
- 🧠 [PaperFlow IO](tools/paperflow-io/)：仅包含 IO 选题、文献复核、可行性评审与实验设计技能；\n- 📚 [FAST-LIVO 环境配置](docs/setup/fast-livo.md)：Ubuntu 20.04 与 ROS Noetic 环境下的历史复现说明。

## 🛣️ 路线图

- [x] 定义统一序列结构、坐标系与单位约定
- [x] 实现 HDF5 I/O、窗口数据集和核心测试
- [x] 集成 Android 数据采集与端侧评测工具
- [x] 整合研究方法、坐标系研究和 PaperFlow IO
- [ ] 接入首个可自动转换的公开数据集适配器
- [ ] 集成至少一个端到端可复现基线
- [ ] 固化数据划分与统一评测指标
- [ ] 发布版本化 benchmark 结果和排行榜

## 🤝 贡献与许可

本仓库只维护 `main`。提交内容前请阅读[贡献规范](CONTRIBUTING.md)，并为数据集、模型和第三方资源记录来源、许可证与再分发限制。

仓库级统一开源许可证尚未确定。源码公开不自动授予再分发或商业使用权；带独立许可证的组件继续遵循其目录中的条款。

---

<div align="center">

如果这个项目对你的研究有帮助，欢迎 ⭐ Star 并关注后续 benchmark 更新。

</div>
