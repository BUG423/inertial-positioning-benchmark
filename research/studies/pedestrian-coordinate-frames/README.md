# 🚶 Pedestrian Inertial Odometry 中 Global Frame 与 Body Frame 表征的实验与分析

**Language:** 简体中文 | [English](README_EN.md)

> 🧭 本仓库用于整理一组关于**行人惯性里程计（Pedestrian Inertial Odometry, Pedestrian IO）中坐标系表征选择**的实验观察、理论分析与可视化结果。

---

## 📌 项目起源

本工作最初来自一个实际投稿过程中的问题 🤔：

在 TLIO/RoNIN 类行人惯性定位任务中，将 IMU 测量旋转到重力对齐的全局坐标系后再训练模型，通常能够获得更好的定位精度 🎯；但相关无人机惯性里程计工作（如 AirIO）指出，在载体/设备坐标系中直接学习，并保留重力项，反而可能带来更好的效果 🛩️。

**两者结论看似矛盾！** 💥

### 🎯 本仓库目标

本仓库的目标不是提出一个完整新算法，而是把这一现象作为一个独立研究问题进行说明：

> 🧩 **为什么在无人机 IO 中有效的 body/device-frame 学习策略，不能直接迁移到行人 IO？**
>
> 🧩 **为什么在行人 IO 中，global/HACF 表征往往更稳定、更容易学习？**

---

## 📑 目录

- [🔭 问题背景](#问题背景)
- [📡 数据采集设置与核心差异](#数据采集设置与核心差异)
- [🧭 坐标系定义](#坐标系定义)
- [⚙️ 实验流程](#实验流程)
- [📊 实验结果](#实验结果)
- [🔬 理论解释](#理论解释)
- [🛩️ 与 AirIO 类无人机 IO 结论的差异](#与-airio-类无人机-io-结论的差异)
- [✅ 主要结论](#主要结论)
- [📖 建议引用方式](#建议引用方式)
- [📝 备注](#备注)

---

## 🔭 问题背景

行人惯性里程计旨在仅使用消费级 IMU 的加速度和角速度测量估计行人的运动轨迹 🚶‍♂️。近年来，RoNIN 等学习式方法通常采用一个预处理步骤：

> 📐 利用 AHRS 或数据集提供的姿态四元数，将原始 IMU 测量从设备坐标系旋转到一个统一的全局坐标系，或更准确地说，旋转到 **heading-agnostic coordinate frame（HACF）**。

该坐标系的关键性质是：**z 轴与重力方向对齐** 🧲，而水平航向可以不唯一。

### 🤯 一个有趣的发现

在一次实验中，我们发现：

> 💡 当对 TLIO 数据进行处理时，使用去除/处理重力后的 global/HACF 表征进行训练与推理，定位误差会出现一定幅度的下降 📉。

该现象在投稿中被描述后，审稿人指出 AirIO 等无人机 IO 工作在 body/device frame 下学习，并保留重力项，取得了更好的结果 😮，因此质疑我们关于 global frame 更适合行人 IO 的结论。

这促使我们重新进行了一组对照实验 ⚗️：在相同网络结构、相同训练/测试划分和相同评价指标下，分别构造 **global frame 学习流程** 与 **body frame 学习流程**，并比较二者在行人 IO 数据集上的表现。

---

## 📡 数据采集设置与核心差异

RoNIN 类行人 IO 数据集的采集方式与无人机 IO 存在本质差异 ⚡。如下图所示，真值轨迹来自固定在人体前胸的设备或外部跟踪系统，而 IMU 测量往往来自另一部以任意方式携带的手机 📱。

![RoNIN-style data collection setup](assets/collect_data.png)

### 🏷️ 术语定义

为了便于描述，本仓库采用如下命名：

| 缩写 | 含义 | 说明 |
|:---:|:---|:---|
| 🎒 **CAR / carrier** | 行人主体载体 | 可近似理解为胸前固定设备或行人身体主体 |
| 📱 **DEV / device** | IMU 测量设备 | 例如手持手机、口袋手机或任意姿态携带的手机 |
| 🌍 **global frame / HACF** | 全局坐标系 | z 轴与重力方向对齐 |
| 📐 **body frame / device frame** | 设备坐标系 | IMU 设备自身的瞬时坐标系 |

### ✈️ vs 🚶 核心差异

在无人机 IO 中，IMU 通常刚性固定在无人机机体上 ✈️，DEV 与 CAR 之间的相对位姿基本固定；

而在行人 IO 中 🚶，手机与人体主体之间存在显著的**非刚性、随机、时变**相对运动 🌀：

- 💪 手臂摆动
- 🤳 手持姿态变化
- 👖 口袋晃动

这些都会引入与行人主体运动无关的**局部扰动** ⚠️。

> 🔑 **这正是两类任务结论不同的核心原因！**

---

## 🧭 坐标系定义

忽略测量噪声时，DEV 在 global frame 下的加速度可以写成：

$$
a_{\mathrm{DEV}}^{g}=a_{\mathrm{CAR}}^{g}+a_{\mathrm{LOC}}^{g}
$$

其中：

| 符号 | 含义 |
|:---:|:---|
| 📏 $a_{\mathrm{DEV}}^{g}$ | 设备 IMU 测量对应的 global-frame 加速度 |
| 🚶 $a_{\mathrm{CAR}}^{g}$ | 行人主体/载体在 global frame 下的加速度 |
| 🔀 $a_{\mathrm{LOC}}^{g}$ | 设备相对人体主体的局部随机运动引起的扰动项 |

在设备自身的 body/device frame 下，观测量为：

$$
a_{\mathrm{DEV}}^{d}=R_{g\to d}a_{\mathrm{DEV}}^{g}
=R_{g\to d}a_{\mathrm{CAR}}^{g}+R_{g\to d}a_{\mathrm{LOC}}^{g}
$$

其中 $R_{g\to d}$ 是从 global frame 到 device frame 的旋转矩阵 🔄。

### 🔄 实际数据处理

实际数据处理中，通常由 AHRS 或数据集姿态估计得到 $\hat{R}_{d\to g}$，并将原始测量旋转到 global/HACF：

$$
\hat{a}_{\mathrm{DEV}}^{g}=\hat{R}_{d\to g}a_{\mathrm{DEV}}^{d}, \qquad
\hat{R}_{d\to g}R_{g\to d}\approx I
$$

如果姿态估计完全准确 ✅，则 global-frame 表征可以把设备朝向变化显式消除；但实际中 $\hat{R}_{d\to g}$ 存在误差 ⚠️，因此 global-frame 学习也并非没有代价。

> 🎯 关键问题：对于行人 IO，这种代价是否小于 body-frame 中由随机姿态与局部扰动共同引入的学习难度？
>
> 实验结果表明，**答案通常是肯定的** ✅！

---

## ⚙️ 实验流程

我们对同一类网络模型在两种坐标系下进行了重新训练与推理 🔬。

### 1️⃣ Global/HACF frame 设置

在 global/HACF 设置下：

1️⃣ 使用数据集提供的四元数或姿态估计，将原始 IMU 测量从 device frame 旋转到 global/HACF。

2️⃣ 网络输入为 global/HACF 下的加速度与角速度。

3️⃣ 标签保持在 global/HACF 或全局坐标系中，主要学习二维水平速度或位移。

4️⃣ 模型输出积分后得到行人轨迹 🚶‍♂️。

> ✨ 这一流程与 RoNIN 类行人 IO 方法的常见预处理方式一致。

### 2️⃣ Body/device frame 设置

在 body/device-frame 设置下：

1️⃣ 原始 IMU 测量保持在设备自身坐标系下 📱，不进行 global/HACF 旋转。

2️⃣ 将位置/速度标签通过姿态四元数旋转到设备坐标系下。

3️⃣ 网络需要直接从 device-frame IMU 学习 device-frame 速度/位移。

4️⃣ 推理后再将预测结果旋转回 global frame 进行轨迹重建与误差评估 🔄。

> 💡 该流程更接近 AirIO 等工作中"直接在载体坐标系学习"的思想。

### 3️⃣ 评价指标

实验采用惯性里程计中常见的两个指标 📏：

| 指标 | 全称 | 说明 |
|:---:|:---|:---|
| 📊 **ATE** | Absolute Trajectory Error | 衡量整体轨迹与真值轨迹之间的全局位置误差 |
| 📈 **RTE** | Relative Trajectory Error | 衡量局部窗口内的相对轨迹误差，反映短时漂移与局部一致性 |

---

## 📊 实验结果

### 🏆 Global vs Body Frame 对比

下图展示了 RoNIN ResNet 在多个数据集上的 global-frame 与 body-frame 对比结果 📈。整体上，global-frame 表征在大多数数据集和指标上表现更稳定，误差更低 🎯。

![Global vs Body Frame Comparison — RoNIN ResNet](assets/RoNIN-resnet18_global_vs_body.png)

### 🔍 观察发现

从图中可以观察到：

- 📉 **IMUNet 数据集**：body-frame 表征的误差显著增大，说明模型难以从强姿态耦合的原始设备坐标系中学习稳定运动模式。

  > ⚠️ 需要指出的是，在 IMUNet 数据集上，将 IMU 数据转换到全局坐标系后，重力并不总是分布在 z 轴，这可能是导致偏差的重要原因。而其他数据集，IMU 数据旋转后重力都集中在 z 轴。

- 🏅 **RoNIN、RNIN 等数据集**：global-frame 也表现出更低的 ATE/RTE。

- ⚖️ 少数数据集或指标上 body-frame 可能接近 global-frame，但整体稳定性较差 ⚠️。

### 🗺️ 单条轨迹可视化

下图进一步给出单条轨迹的可视化结果 🖼️：

- 🔵 **蓝色虚线**：body-frame 推理结果
- 🟠 **橙色曲线**：global-frame 推理结果
- 🟢 **绿色曲线**：ground truth

![Trajectory comparison under body and global frames](assets/12_RoNIN-resnet18_comparison.png)

### 📝 轨迹分析

该轨迹结果表明：

- ✅ global-frame 轨迹整体更贴近真值轨迹
- ❌ body-frame 结果在转弯和闭环结构处更容易产生形状畸变 🌀
- ⚠️ 虽然 body-frame 也能学习到部分运动趋势，但其误差更容易随姿态变化和局部扰动放大 📈

---

## 🔬 理论解释

### 1️⃣ Global frame 中主体运动与局部扰动更容易分离

在 global/HACF 表征下：

$$
a_{\mathrm{DEV}}^{g}=a_{\mathrm{CAR}}^{g}+a_{\mathrm{LOC}}^{g}
$$

该表达具有较直接的物理含义 💡：

- 🚶 **行人主体运动**：低频、连续、受地面约束的二维水平运动
- 🔀 **局部扰动**：来自手臂摆动、手机晃动等高频或局部运动

> 🧩 二者在 global/HACF 下具有更清晰的统计差异，因此更容易被神经网络区分和抑制 ✅！

### 2️⃣ Body frame 中姿态变化会耦合所有运动分量

在 body/device frame 下：

$$
a_{\mathrm{DEV}}^{d}=R_{g\to d}a_{\mathrm{CAR}}^{g}+R_{g\to d}a_{\mathrm{LOC}}^{g}
$$

此时，行人主体运动和设备局部扰动都被瞬时设备姿态 $R_{g\to d}$ 调制 🔄。

对于手持或口袋手机 📱，$R_{g\to d}$ 会快速、随机变化 🌀，因此同样的行人运动可能在 device frame 中呈现出完全不同的分量分布。

> ⚠️ 这会导致表示不连续、速度标签不稳定、训练目标复杂化。

### 3️⃣ 输出目标维度与运动约束不同

在 global/HACF 下 🌍：

- 行人运动主要发生在水平面内
- 模型通常只需学习水平速度或二维位移趋势
- 即使 z 轴方向存在小幅上下运动 🚶‍♂️，其对最终平面定位的贡献也相对有限

在 body/device frame 下 📐：

设备姿态不断变化 🔄，原本平滑的二维水平运动会被旋转到设备三轴中。模型必须同时学习：

1️⃣ 行人主体真实运动 🚶‍♂️

2️⃣ 设备自身姿态变化 📱🔄

3️⃣ 设备相对人体的局部运动 🔀

4️⃣ 从 device-frame 输出恢复到 global-frame 轨迹所需的姿态关系 🔄🗺️

> 💪 因此，body-frame 学习任务显著更难！

---

## 🛩️ 与 AirIO 类无人机 IO 结论的差异

AirIO 类无人机 IO 工作中，body-frame 学习可能有效 ✅，原因在于无人机满足近似刚体假设 🤖：

$$
a_{\mathrm{LOC}}^{g}\approx 0
$$

也就是说 📝，IMU 与无人机机体中心之间的相对位姿基本固定，IMU 观测可以较好地代表载体本身运动。在这种情况下，保留 body-frame 表征和重力项可能提供更直接的动力学信息 🎯，并避免输入阶段姿态估计误差对数据的污染 🛡️。

### ⚠️ 但是，行人 IO 不满足该假设！

手机并非刚性固定在人体质心处 📱❌，而是相对人体不断运动 🚶‍♂️🔄。

因此：

| 场景 | 结论 |
|:---:|:---|
| ✈️ **无人机 IO** | device frame ≈ carrier frame |
| 🚶 **行人 IO** | device frame ≠ carrier frame（由手、口袋或身体局部运动驱动的随机坐标系 🌀） |

> 🔑 **这意味着 AirIO 的结论不能直接外推到行人 IO！**
>
> 坐标系选择不是一个绝对问题，而是与具体任务中的载体-传感器耦合关系密切相关 🔗。

---

## ✅ 主要结论

本仓库支持如下结论 🎯：

1️⃣ 🌍 **在行人 IO 中，global/HACF frame 通常比 body/device frame 更适合学习式定位模型。**

2️⃣ 📐 原因不是 global frame 在数学上包含更多信息，而是它提供了更稳定、更符合行人运动约束的表征。

3️⃣ ✈️ body frame 在无人机 IO 中有效，主要依赖于刚体安装假设；该假设在手持/口袋手机行人 IO 中通常不成立 ❌。

4️⃣ 🔀 行人 IO 中的主要困难不是简单的坐标旋转，而是 DEV 与 CAR 之间时变、非刚性、不可观测的局部运动。

5️⃣ 🧩 global/HACF 表征能够在一定程度上解耦人体主体运动与设备局部扰动，因此更有利于模型训练和轨迹重建 ✅！

---

## 📖 建议引用方式

如果你在论文、报告或项目中引用本仓库 📝，可以使用如下描述：

```bibtex
@misc{pedestrian_io_frame_analysis,
  title  = {Global Frame vs Body Frame Representation in Pedestrian Inertial Odometry},
  author = {BUG423},
  year   = {2026},
  url    = {https://github.com/BUG423/inertial-positioning-benchmark/tree/main/research/studies/pedestrian-coordinate-frames},
  note   = {Experimental study}
}
```

---

## 📝 备注

本仓库记录的是一个独立的实验现象与分析视角 🔭。它可以作为后续方法设计的动机 💡，也可以作为解释行人 IO 与无人机 IO 坐标系选择差异的补充材料 📚。

> ⚠️ 当前结论主要基于 RoNIN 风格的数据组织与实验流程；若设备固定方式、真值定义或姿态估计方式发生改变，global-frame 与 body-frame 的相对优劣仍需要重新实验验证 🔬。

---

<p align="center">
  <b>🎉 感谢阅读！如果有任何问题或建议，欢迎提 Issue 或 PR！</b>
</p>
