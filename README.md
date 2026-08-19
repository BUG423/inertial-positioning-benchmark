<div align="center">

# 惯性定位基准

一个统一、可复现、可扩展的惯性定位研究基准。

[中文](README.md) | [English](README_EN.md)

</div>

## 项目动机

目前，惯性定位研究所使用的数据集、预处理流程、数据格式、评测协议和基线实现各不相同。这些差异使算法之间难以进行公平比较，也给复现和拓展已有工作带来了不必要的负担。

本项目希望将具有代表性的数据集与方法组织在一套统一流程下，为惯性定位社区提供一个开放、规范且易用的研究基础设施。

## 项目目标

- 为惯性定位数据集定义统一的数据格式；
- 为支持的数据集提供可复现的预处理流程；
- 提供简单、一致的数据加载接口；
- 集成具有代表性的开源基线方法；
- 建立标准化的评测协议与指标；
- 降低复现、比较和拓展现有研究的门槛。

## 规划中的工作流程

```text
原始数据集
    -> 数据集专用预处理
    -> 统一数据格式
    -> 通用数据加载器
    -> 基线方法
    -> 标准化评测
```

真实设备数据还可以通过仓库内置的 Android 工具进入同一流程：

```text
Android IMU + ARCore VIO
    -> Inertial Positioning Lab
    -> .iplab 无损归档
    -> canonical HDF5
    -> 通用数据加载器与标准化评测
```

## 路线图

- [x] 调研公开惯性定位数据集与开源方法
- [x] 设计统一数据格式（草案 v0.1）
- [x] 统一预处理流程与坐标系约定（草案 v0.1）
- [x] 实现通用数据集接口（核心层 v0.1）
- [x] 提供 Android 端数据采集与真机模型评测工具
- [ ] 接入首个数据集
- [ ] 集成具有代表性的基线方法
- [ ] 定义评测指标与基准协议
- [ ] 提供可复现的基准实验结果

## 数据接口

- [统一数据规范](docs/FORMAT.md)：规范序列、坐标系、单位、预处理规则与六个首批适配器。

## Benchmark 工具

- [Inertial Positioning Lab](tools/inertial-positioning-lab)：采集手机 IMU 与 ARCore VIO 参考轨迹、导出 `.iplab`、转换 canonical HDF5，并在 Android 真机上运行 LiteRT/TFLite 模型；
- [工具与数据流说明](docs/TOOLS.md)：工具定位、构建安装、格式边界和维护方式；
- [GitHub Releases](https://github.com/BUG423/inertial-positioning-benchmark/releases)：下载自动构建的 Android 预览安装包。

该工具的界面轨迹会抑制静止漂移与跟踪跳点，但归档始终保留未经显示滤波的 ARCore VIO 样本。VIO 是参考轨迹，不等同于 Vicon/RTK 级独立真值。

## 调研

- [论文与时间表](docs/PAPERS.md)：按年份追踪论文、方法、实验数据和开放状态。
- [数据集](docs/DATASETS.md)：记录数据规模、真值、许可、下载与接入状态。

两份文档持续更新；论文信息只进入论文表，数据细节只进入数据集表，避免重复。

## 快速使用

当前代码假设数据已转换为[统一数据规范](docs/FORMAT.md)中的 HDF5 格式：

```bash
pip install -e ".[test]"
pytest

# 将 Android 工具导出的 .iplab 转为 benchmark canonical HDF5
python tools/inertial-positioning-lab/tools/convert_dataset.py recording.iplab -o datasets/android
```

```python
from inertial_benchmark import CanonicalSequence, WindowDataset

sequence = CanonicalSequence.from_hdf5("sequence.h5")
dataset = WindowDataset(sequence, window_size=200, stride=10, target="velocity")

sample = dataset[0]
print(sample.features.shape)  # (6, 200)
print(sample.target.shape)    # (3,)
```

核心层不依赖 PyTorch；后续可在训练框架中直接封装 `WindowDataset`。

## 项目状态

本项目目前处于早期阶段。数据格式、支持的任务、数据集、基线方法和评测协议将逐步完善，并以开放方式记录设计与讨论过程。

## 参与贡献

我们欢迎任何形式的贡献与讨论，也特别期待数据集作者、算法作者以及从事惯性导航与定位研究的同行参与。

你可以通过以下方式帮助建设本项目：

- 推荐值得纳入的数据集、方法、任务或评测协议；
- 协助验证预处理流程和坐标系约定；
- 贡献数据集适配器或基线实现；
- 报告复现过程中遇到的问题；
- 对 benchmark 的整体设计提出建议。

随着项目结构逐渐成熟，我们会补充完整的贡献指南。现阶段，欢迎通过 Issue 介绍你的建议、方案或合作意向。

## 许可证

本项目的统一开源许可证仍在确定中；公开源码不自动授予再分发或商业使用权。所接入的数据集、基线方法和工具仍遵循各自目录中声明的许可证与使用条款。
