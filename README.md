<div align="center">

# 惯性定位基准

统一、可复现、可扩展的惯性定位研究平台，覆盖数据规范、采集工具、评测基础设施、方法提案与 IO 科研工作流。

[中文](README.md) | [English](README_EN.md)

</div>

## 项目定位

惯性定位研究长期面临数据格式、坐标系、预处理流程、评测协议和基线实现不统一的问题。本仓库将基准代码、研究记录和惯性定位专用工具整合到同一套结构中，形成从数据采集到方法研究的完整入口。

```text
数据采集与公开数据集
    -> 统一格式与坐标系
    -> 数据加载与窗口化
    -> 基线与研究方法
    -> 标准化评测
    -> 论文选题、实验与复现
```

## 仓库结构

| 类别 | 目录 | 内容 |
| --- | --- | --- |
| Benchmark 核心 | `src/`、`tests/` | CanonicalSequence、HDF5 读写、窗口数据集与测试 |
| 规范与调研 | `docs/` | 数据格式、数据集、论文时间表和工具说明 |
| 数据采集工具 | `tools/inertial-positioning-lab/` | Android IMU + ARCore VIO 采集、导出、转换与端侧评测 |
| 研究方法 | `research/methods/` | PostDiffIO 与 MoE-IO 方法提案 |
| 实验研究 | `research/studies/` | 行人 IO 坐标系实验与图表 |
| IO 科研工作流 | `tools/research-workflows/inertial/` | 惯性定位专用选题、文献检索、实验设计与可行性评审技能 |

## Benchmark 核心

- [统一数据规范](docs/FORMAT.md)：序列结构、坐标系、单位、预处理规则与首批适配器；
- [数据集调研](docs/DATASETS.md)：数据规模、真值来源、许可、下载与接入状态；
- [论文与时间表](docs/PAPERS.md)：代表性方法、实验数据和开放状态；
- [工具与数据流](docs/TOOLS.md)：采集、归档、转换和评测工具的边界。

### Inertial Positioning Lab

[Android 工具](tools/inertial-positioning-lab)可采集手机 IMU 与 ARCore VIO 参考轨迹，导出无损 `.iplab` 归档，转换为 canonical HDF5，并在真机运行 LiteRT/TFLite 模型。

> 界面轨迹会抑制静止漂移与跟踪跳点，但归档保留未经显示滤波的 ARCore VIO 样本。VIO 是参考轨迹，不等同于 Vicon/RTK 级独立真值。

## 已整合的研究内容

### 方法提案

- [PostDiffIO](research/methods/postdiffio)：以条件扩散模型对确定性 IO 骨干的速度残差进行后验细化，并建模预测不确定性；
- [MoE-IO](research/methods/moe-io)：利用混合专家与运动感知路由处理不同运动状态下的惯性里程计预测。

这两个目录目前属于研究提案与实验设计，不代表已经形成可复现的正式 benchmark 基线。代码、配置和完整结果发布后再纳入标准排行榜。

### 实验研究

- [行人 IO 坐标系研究](research/studies/pedestrian-coordinate-frames)：比较全局坐标系与设备坐标系表征，并保留原始实验图表和中英文说明。

### 惯性定位科研工作流

[IO Research Workflows](tools/research-workflows/inertial)仅保留原 PaperFlow 的 `IO` 分支：

- `paper-ideation-inertial/`：惯性定位选题和创新性、影响力、可行性评审；
- `paper-experiment-inertial/`：文献复核、实验设计、代码可行性检查与实验计划生成。

通用论文技能没有纳入本仓库，避免 benchmark 的职责范围失控。

## 快速开始

```bash
pip install -e ".[test]"
pytest

python tools/inertial-positioning-lab/tools/convert_dataset.py \
  recording.iplab -o datasets/android
```

```python
from inertial_benchmark import CanonicalSequence, WindowDataset

sequence = CanonicalSequence.from_hdf5("sequence.h5")
dataset = WindowDataset(sequence, window_size=200, stride=10, target="velocity")
sample = dataset[0]
```

核心数据层不依赖 PyTorch，可由不同训练框架封装。

## 路线图

- [x] 设计统一数据格式与坐标系约定（草案 v0.1）
- [x] 实现通用数据接口与核心测试
- [x] 提供 Android 数据采集与端侧评测工具
- [x] 整合研究方法、坐标系实验和 IO 科研工作流
- [ ] 接入首个公开数据集适配器
- [ ] 集成可复现的代表性基线
- [ ] 固化评测指标、数据划分和排行榜协议
- [ ] 发布可复现的 benchmark 结果

## 内容成熟度

| 状态 | 含义 |
| --- | --- |
| Core | 已进入统一接口并由测试覆盖 |
| Tool | 可独立构建或运行的配套工具 |
| Study | 已保存实验过程、分析和资源 |
| Proposal | 方法构想或实验设计，尚未纳入正式排行榜 |
| Planned | 已规划但尚未实现 |

## 参与贡献

欢迎贡献数据集适配器、基线实现、坐标系验证、评测协议和复现报告。涉及数据集与第三方方法时，请同时说明来源、许可和可再分发范围。

## 许可证

仓库级统一开源许可证仍在确定中；源码公开不自动授予再分发或商业使用权。整合内容和第三方资源继续遵循各自目录中声明的许可证与使用条款。
