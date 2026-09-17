# TLIO golden 数据集卡片

> 转换器：[`converters/tlio.py`](../../src/inertial_benchmark/data/converters/tlio.py)（`tlio@1.0.0`）｜共享自检工具：[`_rig_utils.py`](../../src/inertial_benchmark/data/converters/_rig_utils.py)
> 数据版本：`golden-new-format-cc-by-nc-with-imus-v1.5.zip`｜最近核验：2026-09-17｜格式规范：[DESIGN.md 第 2 节](../DESIGN.md)

## 1. 概览

| 项目 | 内容 |
|---|---|
| 来源 | TLIO 官方仓库 <https://github.com/CathIAS/TLIO> README 的 Google Drive 链接；EqNIO 仓库 <https://github.com/RoyinaJayanth/EqNIO> 提供同一文件 |
| 许可 | 数据：CC BY-NC 4.0（见发布文件名 `cc-by-nc`）；官方代码：BSD（Meta） |
| 平台 | 头戴设备（VR 头显）上的 Bosch BMI055 IMU（TLIO 论文 III-B 节），`placement=head` |
| 规模 | 354 条序列，31.60 h；单条 82–725 s（中位 295 s） |
| 采样 | `imu0_resampled.npy` 严格 200 Hz（间隔 5000 µs） |
| 活动 | 行走、静立、整理厨房、打台球、上下楼梯等（论文）；因此大量序列以原地活动为主 |
| 真值 | 头显上的视觉惯性滤波器（MSCKF 类）输出，**是参考轨迹而非独立真值** |

## 2. 原始文件与字段

```text
tlio_golden/
├── <seq_id>/imu0_resampled.npy               (N,17) float64
├── <seq_id>/imu0_resampled_description.json  列名、行数、频率、起止时间
├── <seq_id>/calibration.json                 离线 IMU 标定
├── <seq_id>/imu_samples_0.csv                约 1 kHz 原始 IMU（仅 212/354 条）
└── train_list.txt / val_list.txt / test_list.txt / all_ids.txt / spline_metrics.csv
```

| 列（描述文件原文） | 宽度 | 单位 / 含义 | IPB v1 去向 |
|---|---:|---|---|
| `ts_us` | 1 | 设备时钟微秒（非 Unix） | `imu_time = pose_time = ts_us·1e-6` |
| `gyr_compensated_rotated_in_World` | 3 | rad/s，**世界系**，已补偿 | `gyroscope = R_WDᵀ·gyr` |
| `acc_compensated_rotated_in_World` | 3 | m/s²，**世界系比力**（含重力），已补偿 | `accelerometer = R_WDᵀ·acc` |
| `qxyzw_World_Device` | 4 | 四元数 xyzw，Device→World | `orientation`（重排为 wxyz） |
| `pos_World_Device` | 3 | m，世界系 | `position` |
| `vel_World` | 3 | m/s，世界系 | `velocity` |

`calibration.json` 含加速度计/陀螺的常值零偏 `Bias.Offset`、整流矩阵 `RectificationMatrix`、
`T_Device_Imu` 与 `TimeOffsetSec_Device_*`（0.0013 s）。`imu_samples_0.csv` 列为
`timestamp[ns], temperature, w_RS_S_xyz, a_RS_S_xyz`（IMU 系，未标定）。

## 3. 约定核实与转换

| 问题 | 结论 | 证据 |
|---|---|---|
| IMU 坐标系 | npy 中的 gyr/acc 在**世界系** | 描述文件列名；`acc` 全段均值 ≈ [0, 0, 9.81]；官方 `sequences_dataset.py` 直接把这两列作为“重力对齐系”网络输入 |
| 如何回到机体系 | 用同一行 `q_World_Device` 逆旋转 | 逆旋转结果与 `imu_samples_0.csv` 经 `calibration.json` 标定后的数据一致（陀螺 RMS 差约 0.01 rad/s，加速度约 0.04–0.09 m/s²，主要来自 1 kHz→200 Hz 重采样），且**不需要** `T_Device_Imu`，即 “Device” 系就是原始 IMU 系 |
| 是否已补偿零偏 | 是，且补偿量来自 **VIO 在线估计** | 与离线标定的差值是缓慢变化的常量级偏移（加速度 0.02–0.09 m/s²）；最佳对齐需要 −1.3 ms 时移，与 `TimeOffsetSec` 一致 |
| 四元数 | xyzw、Device→World | 列名；逆旋转后重力检查通过 |
| 世界系 | 重力对齐、z 轴向上 | 世界系比力均值 ≈ [0, 0, +9.81] |
| 位置单位 | 米 | 位置二阶差分与世界系比力的比例因子 k = 0.990–1.005 |

转换规则（写入每条序列的 `notes`）：

1. `ts_us → s`；四元数 `xyzw → wxyz` 并归一化；
2. `ω_B = R_WDᵀ ω_W`、`f_B = R_WDᵀ f_W`，`body_frame="tlio_headset_imu"`；
3. 不重新应用 `calibration.json`（数据已被补偿），在 `attrs['imu_calibration']` 中声明“源数据已用 VIO 估计补偿”；
4. 非有限行（发布数据中未出现）标记为无效；`velocity` 直接取 `vel_World`；
5. 不使用 `imu_samples_0.csv`。它能提供**未经参考系统补偿**的 IMU，但只覆盖 212 条（test 36、val 21、train 155），
   留待后续版本作为可选视图。

## 4. 真值性质与精度

- 参考轨迹来自头显上的视觉惯性滤波（论文称其以 1000 Hz 输出位姿，训练监督与测试真值都用它），
  没有外部动捕对照；官方未公布其误差。
- `spline_metrics.csv` 提供每条序列的样条拟合残差（角速度/加速度误差均值与最大值），本转换器未使用，
  可作为参考质量的旁证。
- **信息泄漏提示**：IMU 已用参考滤波器的零偏估计补偿，并以参考姿态旋转过；我们把它逆旋转回机体系，
  但补偿无法撤销。基于本数据的“纯惯性”结果因此略显乐观，这与官方 TLIO 训练设定一致。

## 5. 官方划分与分组键

| 划分 | 序列数 | 接收 | 拒收 | 接收时长 (h) |
|---|---:|---:|---:|---:|
| `train` | 283 | 283 | 0 | 25.30 |
| `val` | 35 | 35 | 0 | 3.13 |
| `test` | 36 | 36 | 0 | 3.18 |
| 合计 | 354 | 354 | 0 | 31.60 |

- 列表互不重叠，三者并集等于全部 354 个目录（论文：随机 80/10/10 划分）。
- `list_sequences()` 枚举所有含 `imu0_resampled.npy` 的子目录；发布版中没有官方列表之外的序列（若将来出现，也会被枚举并照常转换）。
- 附加根属性：`imu_calibration`；`start_time_unix` 为 NaN（时间戳是设备时钟，不是 Unix 时间）。
- 数据未提供受试者或会话信息。`calibration.json` 的离线标定共有 9 种取值，对应 9 台物理头显；转换器以其哈希
  `headset_<sha1 前 8 位>` 作为 `device_id`，并用作 `group_id`。
- 官方随机划分**并非设备不相交**：9 台设备都在 train 中出现，val 覆盖其中 6 台，test 覆盖 5 台。
  因此 `ipb check` 会报告 `group_id` 跨划分重叠，这是数据本身的性质。

## 6. 物理自检（全量 354 条）

方法（`_rig_utils.physical_checks`）：参考姿态经 SLERP 插值到 IMU 时刻；(1) 世界系比力均值与 [0, 0, 9.80665]
的距离；(2) 1 s 间隔的水平速度；(3) 不重叠 10 s 窗口内陀螺积分与参考相对姿态的角度差；(4) 位置二阶中心差分
（h = 0.25 s，2 s 去趋势）与同核平滑的世界系比力减重力之间的比例 k 与残差比；(5) 0.1 s 平滑角速度的互相关时延。
硬阈值：重力误差 < 0.5 m/s²，10 s 陀螺误差中位数 < 5°（若去掉一个 |b| ≤ 0.1 rad/s 的常值零偏后能满足，则只记警告），
水平速度中位数 < 4.5 m/s、p95 < 8 m/s，时间重叠 ≥ 90%。

| 指标 | min | p5 | 中位数 | p95 | max |
|---|---:|---:|---:|---:|---:|
| 重力均值误差 (m/s²) | 0.001 | 0.002 | 0.006 | 0.015 | 0.020 |
| 重力方向倾角 (°) | 0.001 | 0.005 | 0.023 | 0.074 | 0.114 |
| 水平速度中位数 (m/s) | 0.001 | 0.006 | 0.130 | 1.194 | 3.752 |
| 水平速度 p95 (m/s) | 0.122 | 0.369 | 1.132 | 1.449 | 5.166 |
| 10 s 陀螺误差中位数 (°) | 0.127 | 0.152 | 0.201 | 0.255 | 0.701 |
| 去常值零偏后 (°) | 0.115 | 0.145 | 0.191 | 0.234 | 0.281 |
| 加速度一致性比例 k | 0.990 | 0.995 | 0.998 | 1.000 | 1.005 |
| 加速度残差比 | 0.018 | 0.023 | 0.037 | 0.055 | 0.078 |
| 角速度互相关时延 (s) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 时间重叠比例 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

说明：重力与陀螺指标几乎为零，是因为 IMU 本身就是用同一姿态旋转、同一滤波器补偿得到的，这些检查在这里
主要验证列含义与四元数约定；加速度一致性（k≈1）则独立证实了位置单位与时间同步。

警告（不拒收）：

- 157 条水平速度中位数 < 0.1 m/s（原地活动为主，符合论文描述的活动类型）；
- 1 条快速运动：`253159831907410`，159 s 内走完约 480 m 的闭环（中位 3.75 m/s）。VIO 速度的导数与世界系比力
  吻合（残差 0.01 m/s²），起终点重合，属于真实的跑步量级运动，因此保留。

## 7. 被拒序列清单

无（0/354）。

## 8. 已知问题

1. IMU 已被参考滤波器补偿（见第 4 节），不能代表未标定的原始传感器。
2. 没有受试者标签，分组只能到设备级；官方划分跨设备随机。
3. `imu_samples_0.csv` 时间戳列头写作 `[ns]`，数值与 npy 的 `ts_us` 相差 1000 倍，即确为纳秒；本版本不使用该文件。
4. 大量低速序列会让以“行人速度”为目标的指标（如方向误差，只统计 ‖v‖ > 0.2 m/s）样本偏少。

## 9. 复现

```python
from inertial_benchmark.data.converters import tlio
splits = tlio.official_splits("/path/to/TLIO")          # 接受 tlio_golden 或其父目录
for raw in tlio.iter_raw_sequences("/path/to/TLIO"):     # RawSequence，notes 中含 physical_check JSON
    ...
```

真实数据抽样测试：`PYTHONPATH=src python -m pytest tests/data/test_raw_tlio.py`（环境变量 `IPB_RAW_TLIO` 可覆盖路径）。
