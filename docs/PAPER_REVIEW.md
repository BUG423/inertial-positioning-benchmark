# 惯性定位论文 Review

> 最近更新：2026-07-19  
> 维护规则：新增论文必须同时登记实验数据集、代码/模型状态、任务定义和对 benchmark 的影响。

## 状态标记

- **已核验**：已检查论文原文和官方项目/仓库。
- **部分核验**：已检查论文，但代码、数据或许可仍待确认。
- **待复现**：已有公开实现，尚未在本项目环境复现。

## 按时间排序的核心论文

| 时间 | 论文 | 主要贡献 | 实验数据集 | 开放状态 | 状态 |
|---|---|---|---|---|---|
| 2018 | [IONet](https://ojs.aaai.org/index.php/AAAI/article/view/11217) | 从 IMU 窗口回归极坐标位移，开学习式 IO 先河 | 自采数据 | 代码/数据需继续核验 | 部分核验 |
| 2018 | [RIDI](https://openaccess.thecvf.com/content_ECCV_2018/papers/Hang_Yan_RIDI_Robust_IMU_ECCV_2018_paper.pdf) | 学习速度并修正双积分；处理任意手机放置 | RIDI | 数据、代码公开 | 已核验 |
| 2018/2019 | [OxIOD](https://arxiv.org/abs/1809.07491) | 发布多设备、多携带方式的大规模手机 IO 数据 | OxIOD | 数据公开 | 已核验 |
| 2019/2020 | [RoNIN](https://arxiv.org/abs/1905.12853) | ResNet/LSTM/TCN 速度回归；seen/unseen benchmark | RoNIN、RIDI、OxIOD | 代码公开，数据约 50% 公开 | 已核验 |
| 2020 | [TLIO](https://arxiv.org/abs/2007.01867) | 回归 3D 位移与不确定性并紧耦合 EKF | TLIO | 数据后由 EqNIO 仓库提供 | 已核验 |
| 2021 | [IDOL](https://arxiv.org/abs/2102.04024) | 联合学习手机方向与位置 | IDOL | Zenodo 数据公开 | 已核验 |
| 2021 | [RNIN-VIO](https://zju3dv.github.io/rnin-vio/) | RNIN 预测 3D 位移/协方差并增强 VIO | IDOL、SenseINS | 代码和约 7 小时自采数据公开 | 已核验 |
| 2022 | [CTIN](https://arxiv.org/abs/2112.02143) | Contextual Transformer、多任务速度与轨迹估计 | RIDI、OxIOD、RoNIN、IDOL、自有数据 | 代码公开；自有数据受限 | 已核验 |
| 2022 | [RIO](https://openaccess.thecvf.com/content/CVPR2022/html/Cao_RIO_Rotation-Equivariance_Supervised_Learning_of_Robust_Inertial_Odometry_CVPR_2022_paper.html) | 旋转等变监督学习与稳健 IO | 待逐项抽取 | 代码/数据待核验 | 部分核验 |
| 2023 | [RIOT](https://www.mdpi.com/1424-8220/23/6/3217) | 递归 Transformer 与姿态不变 IO | 待逐项抽取 | 待核验 | 部分核验 |
| 2024 | [IMUNet](https://ieeexplore.ieee.org/document/10480886/) | 面向边缘设备的高效 1D CNN 架构 | RoNIN、RIDI、OxIOD、IMUNet_dataset | 代码和自建数据公开 | 已核验 |
| 2024 | [Deep Learning for Inertial Positioning: A Survey](https://arxiv.org/abs/2303.03757) | 系统总结行人、车辆、无人机与机器人惯性定位 | 多领域 | 论文公开 | 已核验 |
| 2025 | [iMoT](https://ojs.aaai.org/index.php/AAAI/article/view/32664) | Motion Transformer，建模运动/旋转跨模态关系 | RIDI、RoNIN、OxIOD、IDOL | 代码状态待核验 | 已核验论文 |
| 2025 | [EqNIO](https://openreview.net/forum?id=C8jXEugWkq) | 子等变 canonicalization，适配 TLIO/RoNIN | TLIO、Aria、RoNIN、RIDI、OxIOD | 代码与 TLIO 数据入口公开 | 已核验 |
| 2025 | [AirIO](https://arxiv.org/abs/2501.15659) | 面向高动态 UAV 的 body-frame IO | EuRoC、Blackbird、Pegasus | 公开性逐项核验中 | 部分核验 |
| 2025 | [Neural IO from Lie Events](https://arxiv.org/abs/2505.09780) | 用 Lie Events 表示替代固定频率 IMU 序列 | TLIO、Aria、RoNIN、RIDI、OxIOD | 论文公开 | 已核验论文 |
| 2026 | [X-IONet](https://arxiv.org/abs/2511.08277) | 行人与四足机器人跨平台 IO | RoNIN、GrandTour、Go2 | GrandTour 公开；Go2 待核验 | 部分核验 |

## Review 模板

每篇新增论文使用以下字段：

- Bibliography：题目、作者、出处、年份、DOI/arXiv；
- Task：2D/3D，velocity/displacement/pose/bias/uncertainty；
- Input：IMU 类型、频率、窗口、姿态或其他辅助输入；
- Method：模型、滤波器、坐标表示与损失；
- Datasets：训练、验证、测试和跨域设置；
- Metrics：ATE、RTE、drift、AYE、推理速度等；
- Reproducibility：代码、权重、配置、许可证；
- Benchmark impact：需要新增的接口、指标或 track；
- Verification：最后核验日期与未解决问题。

## 待办

- [ ] 补齐 IONet、RIO、RIOT 的完整数据集与开源状态；
- [ ] 增加 A2DIO、NILoc、DIDO、DIVE、Legolas 等跨平台工作；
- [ ] 对每篇方法记录官方 split 与指标定义；
- [ ] 建立可机读的 papers.yaml。
