<div align="center">

# Inertial Positioning Benchmark

**面向惯性定位研究的统一数据接口、采集工具与可复现实验平台**

[中文](README.md) · [English](README_EN.md) · [数据规范](docs/FORMAT.md) · [目录说明](docs/REPOSITORY_LAYOUT.md)

</div>

## 为什么建立这个仓库

惯性里程计研究常使用不同的数据格式、坐标系、窗口定义和评测协议，使方法之间难以公平比较。本仓库把可执行 benchmark、Android 采集工具、研究提案、实验研究和 IO 专用科研工作流放入一套边界明确的结构中，目标是形成可检查、可扩展、可复现的研究基础设施。

当前已经具备统一序列表示、HDF5 读写、滑动窗口数据集和 Android 采集链路；公开数据集适配器、标准基线与正式排行榜仍在建设中。

## 当前能力

| 模块 | 状态 | 提供内容 |
| --- | --- | --- |
| Benchmark 核心 | **可用** | `CanonicalSequence`、HDF5 I/O、窗口化与测试 |
| Android 工具 | **可用** | IMU + ARCore VIO 采集、`.iplab` 导出、HDF5 转换与端侧推理 |
| 坐标系研究 | **研究记录** | 行人 IO 中 global frame 与 body frame 的实验分析 |
| PostDiffIO / ModeMoEIO | **方法提案** | 方法定义与实验计划；尚无可复现结果 |
| PaperFlow IO | **研究工具** | 仅保留惯性里程计选题与实验设计技能 |
| 公共基线与排行榜 | **规划中** | 数据集适配器、统一划分、指标和可复现基线 |

> “方法提案”不等同于 benchmark 结果。只有实现、配置、数据划分和结果均可审计的方法，才能进入正式基线与排行榜。

## 数据流

```text
设备采集 / 公开数据集
        ↓
CanonicalSequence（统一时间戳、坐标系和单位）
        ↓
HDF5 持久化与 WindowDataset 窗口化
        ↓
基线训练 / 方法评测 / 端侧部署
```

## 目录布局

```text
inertial-positioning-benchmark/
├── src/inertial_benchmark/       # 稳定、可测试的 Python 核心接口
├── tests/                        # 核心、集成与文档一致性测试
├── docs/                         # 数据规范、调研与仓库约定
├── tools/
│   ├── inertial-positioning-lab/ # Android 采集和端侧评测应用
│   └── paperflow-io/             # 仅面向惯性里程计的科研工作流
└── research/
    ├── methods/                  # 尚未成为正式基线的方法提案
    └── studies/                  # 可追溯的实验研究与分析资源
```

完整的目录职责、准入规则和新增组件方式见[仓库目录规范](docs/REPOSITORY_LAYOUT.md)。

## 快速开始

```bash
git clone https://github.com/BUG423/inertial-positioning-benchmark.git
cd inertial-positioning-benchmark
python -m pip install -e ".[test]"
pytest -q
```

```python
from inertial_benchmark import CanonicalSequence, WindowDataset

sequence = CanonicalSequence.from_hdf5("sequence.h5")
dataset = WindowDataset(sequence, window_size=200, stride=10, target="velocity")
sample = dataset[0]
```

核心数据层不依赖 PyTorch，可由任意训练框架封装。转换 Android 采集数据：

```bash
python tools/inertial-positioning-lab/tools/convert_dataset.py \
  recording.iplab --output datasets/android
```

## 组件导航

| 入口 | 用途 |
| --- | --- |
| [数据格式](docs/FORMAT.md) | 序列结构、坐标系、单位和预处理规则 |
| [数据集目录](docs/DATASETS.md) | 数据规模、真值、许可和接入状态 |
| [论文调研](docs/PAPERS.md) | 代表性方法、数据与开放状态 |
| [工具说明](docs/TOOLS.md) | 采集、转换和端侧评测边界 |
| [研究内容](research/README.md) | 方法提案与实验研究索引 |
| [开发工具](tools/README.md) | Android 工具与 PaperFlow IO 索引 |
| [贡献规范](CONTRIBUTING.md) | 目录约定、证据要求与检查命令 |

## 路线图

- [x] 定义统一序列结构、坐标系与单位约定
- [x] 实现 HDF5 I/O、窗口数据集和核心测试
- [x] 集成 Android 数据采集与端侧评测工具
- [x] 整合 PostDiffIO、ModeMoEIO、坐标系研究和 PaperFlow IO
- [ ] 接入首个可自动下载或转换的公开数据集适配器
- [ ] 集成至少一个端到端可复现基线
- [ ] 固化训练/验证/测试划分与评测指标
- [ ] 发布版本化 benchmark 结果和排行榜

## 贡献与许可

本仓库只在 `main` 上维护。贡献前请阅读[贡献规范](CONTRIBUTING.md)，并为数据、模型和第三方资源记录来源、许可及再分发限制。

仓库级统一开源许可证尚未确定。源码公开不自动授予再分发或商业使用权；带独立许可证的组件继续遵循其目录内的条款。
