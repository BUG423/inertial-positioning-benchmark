# 惯性定位数据集

> 版本：v2（论文反向追踪版）  
> 最近核验：2026-07-19  
> 状态：持续更新

## 1. 调研方法

本版采用“最新论文反向追踪”：

1. 从近年代表性论文与综述的实验章节抽取全部数据集；
2. 回溯数据集原始论文、官网、GitHub 和长期存储地址；
3. 核验是否真实公开、公开比例、文件内容、划分和许可；
4. 区分方法名、论文名与数据集名；
5. 建立“论文 → 数据集 → 下载入口 → 可接入状态”的证据链。

公开状态分为：

- **公开可下载**：当前存在无需作者审批的下载入口；
- **受协议约束公开**：可下载，但需要接受非商业或特定用途协议；
- **未完整公开**：论文使用了数据，但公开比例不足、需要申请或尚未发布。

## 2. 名称辨析

| 名称 | 类型 | 独立数据集 | 说明 |
|---|---|---:|---|
| RoNIN | 方法、论文、数据集 | 是 | 官方仅公开约 50%，但仍是最常用行人基准之一 |
| RIDI | 方法、论文、数据集 | 是 | Robust IMU Double Integration |
| OxIOD | 数据集、论文 | 是 | Oxford Inertial Odometry Dataset |
| TLIO | 方法、论文、数据集 | 是 | EqNIO 官方仓库现提供可下载的 TLIO golden 数据 |
| IDOL | 方法、论文、数据集 | 是 | 20+ 小时数据公开在 Zenodo |
| IMUNet | 方法、论文 | 是，另有 IMUNet_dataset | 论文还使用 RoNIN、RIDI、OxIOD |
| CTIN | 方法、论文 | 自有数据未完整公开 | 论文使用 RIDI、OxIOD、RoNIN、IDOL 和自有数据 |
| iMoT | 方法、论文 | 否 | AAAI 2025，实验使用 RIDI、RoNIN、OxIOD、IDOL |
| EqNIO | 方法、论文 | 否 | ICLR 2025；官方仓库提供 TLIO 数据下载 |
| AirIO | 方法、论文 | 有 Pegasus 仿真集 | 使用 EuRoC、Blackbird、Pegasus |
| RNIN-VIO | 方法、论文 | 是，另有自采 SenseINS 数据 | 使用 IDOL 20 小时与约 7 小时自采多手机数据 |
| X-IONet | 方法、论文 | 有自采 Go2 数据 | 使用 RoNIN、GrandTour、Go2 |

## 3. 论文到数据集的溯源矩阵

| 论文 | 年份/出处 | 实验数据集 |
|---|---|---|
| [Deep Learning for Inertial Positioning: A Survey](https://arxiv.org/abs/2303.03757) | IEEE T-ITS 2024 | 行人、车辆、无人机、机器人等领域数据 |
| [IMUNet](https://ieeexplore.ieee.org/document/10480886/) | IEEE TIM 2024 | RoNIN、RIDI、OxIOD、IMUNet_dataset |
| [CTIN](https://arxiv.org/abs/2112.02143) | AAAI 2022 | RIDI、OxIOD、RoNIN、IDOL、自有数据 |
| [iMoT](https://ojs.aaai.org/index.php/AAAI/article/view/32664) | AAAI 2025 | RIDI、RoNIN、OxIOD、IDOL |
| [EqNIO](https://openreview.net/forum?id=C8jXEugWkq) | ICLR 2025 | TLIO、Aria、RoNIN、RIDI、OxIOD |
| [Neural Inertial Odometry from Lie Events](https://arxiv.org/abs/2505.09780) | RSS 2025 | TLIO、Aria、RoNIN、RIDI、OxIOD |
| [AirIO](https://arxiv.org/abs/2501.15659) | 2025 | EuRoC、Blackbird、Pegasus |
| [X-IONet](https://arxiv.org/abs/2511.08277) | 2026 | RoNIN、GrandTour、Go2 |
| [RNIN-VIO](https://zju3dv.github.io/rnin-vio/) | ISMAR 2021 | IDOL 约 20 小时、自采 SenseINS 约 7 小时 |
| [RoNIN](https://arxiv.org/abs/1905.12853) | 2019/2020 | RoNIN、RIDI、OxIOD |
| [TLIO](https://arxiv.org/abs/2007.01867) | IEEE RA-L 2020 | TLIO 自建数据 |

论文证据得到的主干集合是：

> **行人/人体：RIDI、OxIOD、RoNIN、IDOL、TLIO、RNIN-VIO/SenseINS、IMUNet_dataset**  
> **跨平台：Aria、EuRoC、Blackbird、GrandTour、Pegasus、Go2**

## 4. 核心行人和人体携带数据集

### 4.1 RoNIN — P0

- 超过 40 小时、100 名受试者、自然人体运动；
- 200 Hz IMU 和 3D 轨迹真值；
- 包含 seen/unseen 测试；
- 官方说明因安全原因仅公开约 50% 数据；
- EqNIO 与 RSS 2025 Lie Events 均只使用公开部分。

入口：[论文](https://arxiv.org/abs/1905.12853)、[代码](https://github.com/Sachini/ronin)、[数据页](https://ronin.cs.sfu.ca/)。

接入要求：明确标记 public subset，不能声称复现原论文全量训练。

### 4.2 RIDI — P0

- 约 150 分钟，200 Hz；
- 手持、口袋、包内等携带方式；
- 真值来自 Google Tango 视觉—惯性系统；
- 被 RoNIN、CTIN、iMoT、EqNIO、Lie Events、IMUNet 等反复使用。

入口：[项目页](https://yanhangpublic.github.io/ridi/)、[论文](https://openaccess.thecvf.com/content_ECCV_2018/papers/Hang_Yan_RIDI_Robust_IMU_ECCV_2018_paper.pdf)、[代码](https://github.com/higerra/ridi_imu)。

风险：真值属于 VIO 参考，不能与独立 Vicon 真值等价描述。

### 4.3 OxIOD — P0

- 158 条序列、总距离超过 42 km；
- 4 类手机、5 名用户；
- 手持、口袋、手提包、推车；
- 停止、慢走、正常行走、跑步；
- 多数序列为光学动捕真值，长距离序列参考来源不同；
- 被 RoNIN、CTIN、iMoT、EqNIO、Lie Events、IMUNet 使用。

入口：[官方页面](https://deepio.cs.ox.ac.uk/)、[论文](https://arxiv.org/abs/1809.07491)。

风险：必须按序列记录真值来源，不能假设所有序列同精度。

### 4.4 IDOL — P0

- 20+ 小时室内行人手机 IMU；
- 15 名用户、3 栋建筑；
- 同时评测设备方向与位置；
- 完整数据公开在 Zenodo；
- CTIN 与 iMoT 均将其作为核心实验数据。

入口：[论文](https://arxiv.org/abs/2102.04024)、[仓库](https://github.com/KlabCMU/IDOL)、[Zenodo](https://zenodo.org/records/4484093)。

上一版漏掉 IDOL 是明确缺项。

### 4.5 TLIO Dataset — P0

EqNIO 论文附录和官方仓库说明：

- 400 条序列，总计约 60 小时；
- 原始 IMU 1 kHz，发布版本含 200 Hz 处理数据；
- Bosch BMI055 安装在与相机刚性连接的头戴设备上；
- 超过 5 名受试者；
- 包含行走、整理厨房、上下楼梯等活动；
- 参考状态含位置、方向、速度、IMU bias 和 noise；
- 80%/10%/10% train/val/test；
- 发布数据包含校准、原始测试 IMU、处理后的 IMU/VIO ground truth 和 split；
- attitude filter data 未包含。

入口：[论文](https://arxiv.org/abs/2007.01867)、[项目页](https://cathias.github.io/TLIO/)、[EqNIO 数据说明](https://github.com/RoyinaJayanth/EqNIO)。

关键修正：TLIO 原始代码仓库没有直接给出完整下载，不等于数据无法获得；EqNIO 后续已经提供 TLIO golden v1.5 下载。许可与原始权属仍需核验。

### 4.6 IMUNet_dataset — P1

IMUNet 是方法，但官方仓库还发布了独立数据：

- Android 手机 IMU；
- Google ARCore API 提供轨迹参考；
- 结合 RIDI 的 Lenovo Tango 真值采集思路；
- 提供数据下载、预处理代码与 Android 采集应用；
- IMUNet 论文还使用 RoNIN、RIDI、OxIOD。

入口：[论文](https://arxiv.org/abs/2208.00068)、[仓库与数据](https://github.com/BehnamZeinali/IMUNet)。

待核验：规模、设备、采样率、ARCore/Tango 真值质量和许可证。

### 4.7 RNIN-VIO / SenseINS — P0/P1

RNIN-VIO 是 ISMAR 2021 的视觉—惯性融合方法，但其核心 RNIN 网络可仅用 IMU 输出 3D 相对位移与协方差；官方同时公开了配套惯性数据和代码。

数据组成需要严格区分：

- 论文训练数据合计约 27 小时：约 20 小时来自公开 IDOL，约 7 小时为作者自采；
- 官方实际额外发布的是约 7 小时自采 SenseINS 部分，而不是重新发布 IDOL；
- 自采数据来自 Huawei、Xiaomi、OPPO 等多款手机；
- 共 5 名采集者；
- 包含行走、跑步、静止、上下楼梯、随机晃动等运动；
- 图像为 30 Hz，IMU 为 200 Hz；
- 多数参考位置由 BVIO 生成，并在 IMU 频率上输出重力对齐位置；
- 部分序列在 Vicon 场地采集，具有高精度 Vicon 轨迹；
- 已提供 train/val/test 目录和 SenseINS.csv 字段说明；
- 官方仓库代码采用 Apache-2.0，但仍需单独核验数据文件本身的授权范围。

入口：[项目页](https://zju3dv.github.io/rnin-vio/)、[官方仓库与下载](https://github.com/zju3dv/rnin-vio)、[论文](https://zju3dv.github.io/rnin-vio/)。

接入结论：**自采 SenseINS 数据列为 P0/P1 之间的高优先级候选。** 它补足了多手机、上下楼、跑步和 3D 人体运动，但 BVIO 真值与 Vicon 真值必须分别标记，不能混成统一精度。

## 5. 最新论文引出的跨平台数据集

### Aria Everyday Activities — P1

EqNIO 与 Lie Events 用它测试 TLIO 类模型。包含 143 条日常活动序列、多名佩戴者和 5 个地点，并提供 Project Aria 眼镜的高频全局轨迹。下载需接受专用许可。

入口：[官方页面](https://www.projectaria.com/datasets/aea)。

### Blackbird — P1

AirIO 使用的高动态 UAV 数据集：

- 168 次飞行、17 种轨迹、5 个环境；
- 超过 10 小时、最高 7 m/s；
- 100 Hz IMU、约 190 Hz 电机转速；
- 360 Hz 毫米级动捕真值。

入口：[论文](https://arxiv.org/abs/1810.01987)、[数据页](http://blackbird-dataset.mit.edu/)。

### EuRoC MAV — P1

AirIO 已将 EuRoC 用于纯惯性学习实验，而不只是 VIO。它有 11 条主要飞行序列、同步 IMU/双目和精确真值；应放入 UAV track。

入口：[官方页面](https://projects.asl.ethz.ch/datasets/euroc-mav/)。

### Pegasus — P2

AirIO 的仿真 UAV 数据，共 7 条轨迹，4 条训练、3 条测试。是否随代码完整公开仍需核验。

### GrandTour — P1

X-IONet 使用的四足机器人数据：

- ANYmal-D、49+ 环境/任务；
- 多相机、LiDAR、IMU、本体感知；
- RTK-GNSS 与 Leica 全站仪真值；
- 官网、GitHub、Hugging Face 均有入口；
- 可提取 IMU-only 子集，但不能忽略其多模态原始定位。

入口：[官网](https://grand-tour.leggedrobotics.com/)、[GitHub](https://github.com/leggedrobotics/grand_tour_dataset)、[论文](https://arxiv.org/abs/2602.18164)。

### Go2 — P2

X-IONet 自采 Unitree Go2 数据。论文有结果，但完整数据的稳定公开入口仍需核验。

## 6. 独立扩展轨道

| 数据集 | 平台 | 状态与用途 |
|---|---|---|
| [UTIAS Foot-Mounted INS](https://starslab.ca/foot-mounted-inertial-navigation-dataset/) | 足绑 | 公开；ZUPT/足绑 INS |
| [pyShoe](https://github.com/utiasSTARS/pyshoe) | 足绑 | 公开；39 次走/跑试验 |
| [IPIN 2019 Track 4](https://zenodo.org/records/3937220) | 足绑 | 公开；竞赛式评测 |
| [IO-VNBD](https://doi.org/10.1016/j.dib.2021.106885) | 车辆 | 下载与许可待深查 |
| [AI-IMU-DR](https://github.com/mbrossar/ai-imu-dr) | 车辆 | 公开代码与数据流程 |
| [Multiple and Gyro-Free](https://pmc.ncbi.nlm.nih.gov/articles/PMC11450167/) | 车/机器人/转台 | 公开；MIMU/GFINS |
| [Wheel-Mounted](https://www.nature.com/articles/s41597-025-06224-w) | 车辆/机器人 | 公开；轮载 IMU |
| [ADVIO](https://github.com/AaltoVision/ADVIO) | 手持 | 公开；多模态辅助 |
| [TUM-VI](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset) | 手持 | 公开；VIO/格式兼容 |
| [UZH-FPV](https://fpv.ifi.uzh.ch/) | UAV | 非商业使用限制 |

## 7. 推荐的 benchmark tracks

### Track A：手机/人体携带式 2D IO

RoNIN、RIDI、OxIOD、IDOL、RNIN-VIO/SenseINS、IMUNet_dataset。

兼容 RoNIN、CTIN、iMoT、IMUNet 类速度/位移回归，报告 in-domain 和 cross-dataset 泛化。

### Track B：头戴式 3D IO

TLIO、Aria Everyday Activities。

兼容 TLIO、EqNIO、Lie Events 类 3D displacement + uncertainty + EKF。

### Track C：高动态 UAV IO

Blackbird、EuRoC、Pegasus（确认公开后）。

### Track D：足绑 INS

UTIAS、pyShoe、IPIN 2019 Track 4。

### Track E：腿式机器人 IO

GrandTour、Go2（若公开）。

不同 track 可以共享基础 IMU schema，但不能强行共享输出定义、训练协议和排行榜。

## 8. 接入状态

状态流：待发现 → 已确认来源 → 已确认下载 → 已下载 → 已解析 → 已转换 → 已验证 → 已接入。

只有完成“已验证”的数据集才能进入正式 benchmark。每次更新同时记录最后核验日期、下载状态、许可变化和已知问题。

## 9. 下一阶段实物核验

- [ ] RoNIN：公开序列数、体积、字段和 seen/unseen split；
- [ ] RIDI：目录、坐标系、Tango 真值和许可；
- [ ] OxIOD：逐序列设备、携带方式和真值来源；
- [ ] IDOL：Zenodo 文件、方向/位置标签、建筑/人员划分；
- [ ] TLIO：下载 golden v1.5，核验列描述、raw/resampled IMU、calibration、split 和 CC BY-NC 条款；
- [ ] RNIN-VIO/SenseINS：下载约 7 小时自采数据，区分 BVIO 与 Vicon 真值并核验字段和数据许可；
- [ ] IMUNet_dataset：规模、设备、真值、采样率和许可；
- [ ] 为每个数据集记录下载方式、SHA256、许可和不可再分发要求；
- [ ] 根据六个核心数据集反推 unified schema；
- [ ] 建立论文—数据集引用数据库，以后新增论文时自动暴露遗漏项。

## 10. 当前结论

上一版识别 RIDI、OxIOD、RoNIN 为核心没有错，但明显不完整：漏掉了 IDOL、TLIO 后续公开版本和 IMUNet_dataset，也没有从最新方法论文建立证据链。

当前核心范围应至少包括：

> **RoNIN、RIDI、OxIOD、IDOL、TLIO、RNIN-VIO/SenseINS、IMUNet_dataset**

并将 **Aria、Blackbird、EuRoC、GrandTour** 纳入跨设备、UAV 与腿式机器人扩展。TLIO 类 3D 头戴式任务和 RoNIN 类 2D 手机任务应属于同一项目下的不同 track，而不是混在一个排行榜中。
