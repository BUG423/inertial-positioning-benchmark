# RoNIN 数据集卡片

> 转换器：[`converters/ronin.py`](../../src/inertial_benchmark/data/converters/ronin.py)，`ronin@1.0`；
> 共享工具与物理自检：[`converters/_phone_utils.py`](../../src/inertial_benchmark/data/converters/_phone_utils.py)。
> 统计基于本机 `/workspace/webCodex/datasets/imu_odometry/raw/_staging/RoNIN`（152 条），2026-09-17。

## 1. 概览

| 项 | 内容 |
|---|---|
| 论文 | Herath, Yan, Furukawa. *RoNIN: Robust Neural Inertial Navigation in the Wild*, ICRA 2020 |
| 数据 | FRDR <https://doi.org/10.20383/102.0543>（4 个 zip：train_dataset_1/2、seen/unseen_subjects_test_set） |
| 官方代码 | <https://github.com/Sachini/ronin>，本卡片依据提交 `805b7f0`（`source/data_glob_speed.py`、`source/data_utils.py`、`lists/`） |
| 许可 | RoNIN Data License：仅限非商业科研，禁止再分发（见数据包 `LICENSE.txt`）；转换器只读原始数据，不复制 |
| 公开比例 | 原数据 327 条，出于安全原因只公开约 50%；本地 152 条，官方列表 153 条中仅缺 `a007_3` |
| 采集方式 | 两台设备：胸前固定的 Tango 手机提供 VIO 位姿；另一台 Android 手机（asus3/4/5/6/7、samsung1）以自然方式携带，记录 IMU |
| 输入 `source` | `.../_staging/RoNIN`（其下 `seen/ train1/ train2/ unseen/`）或直接包含序列目录的文件夹；zip 需先解压 |

## 2. 原始文件与字段

每条序列一个目录 `<受试者>_<k>/`，含 `data.hdf5` 与 `info.json`。转换器使用的字段：

| 路径 | 形状 | 单位/含义 | 用途 |
|---|---|---|---|
| `synced/time` | (N,) | 手机系统时间（开机时钟，秒），200 Hz 等间隔 | IMU 与位姿共同时间轴 |
| `synced/gyro_uncalib` | (N,3) | rad/s，Android `TYPE_GYROSCOPE_UNCALIBRATED` | 陀螺（减官方初始零偏） |
| `synced/acce` | (N,3) | m/s²，Android `TYPE_ACCELEROMETER`，含重力的比力 | 加速度计（官方标定） |
| `synced/game_rv` | (N,4) | **wxyz**，Android `TYPE_GAME_ROTATION_VECTOR` | 设备姿态；参考姿态的来源之一 |
| `pose/tango_pos` | (N,3) | m，Tango 起始系（重力对齐、z 向上），与 `synced/time` 逐行对应 | 参考位置 |
| `pose/tango_ori` | (N,4) | wxyz，**胸前 Tango 设备**的姿态 | 仅第 0 帧用于起点对齐 |
| `pose/ekf_ori` | (N,4) | wxyz，手机姿态的传感器融合结果（README：测试时不应使用） | 参考姿态的来源之一 |
| `info.json` | — | `imu_init_gyro_bias`、`imu_acce_bias`、`imu_acce_scale`、`start_frame`、`start_calibration`、`grv_ori_error`/`ekf_ori_error`/`gyro_integration_error`（末端对齐误差，度）、`device`、`date` | 标定与元信息 |

未使用：`raw/*`（各传感器原始时间戳）、`synced/gyro`（Android 在线去零偏版本）、`synced/{linacce,grav,magnet,rv}`、`pose/tango_adf_pose`、WiFi/GPS。

## 3. 约定与转换

| 量 | 转换 | 依据 |
|---|---|---|
| 时间 | `synced/time` 原样（秒，开机时钟）；`start_time_unix = NaN` | README |
| 截取 | 丢弃 `start_frame` 之前的预校准/同步段 | 官方 `GlobSpeedSequence` |
| 陀螺 | `gyro_uncalib − imu_init_gyro_bias` | 官方代码 |
| 加速度 | `imu_acce_scale ⊙ (acce − imu_acce_bias)`，比力，含重力 | 官方代码 |
| 参考姿态 `pose/orientation` | `q_wb(t) = tango_ori[0] ⊗ start_calibration ⊗ src[0]* ⊗ src(t)` | 官方代码 |
| 姿态源 `src` | 官方训练规则：`grv_ori_error < 20°` 用 game_rv，否则取 `[gyro_integration, game_rv, ekf]` 中末端误差最小者 | 官方 `select_orientation_source`（`max_ori_error=20`） |
| 设备姿态 `imu/orientation` | 原始 game_rv（与参考世界系差一个常值、近似纯偏航的旋转） | — |
| 位置 | `pose/tango_pos` 原样 | — |
| 机体系 | Android 传感器坐标系（x 右、y 上、z 出屏） | — |

要点：

- **`pose/tango_ori` 不是手机姿态。** 它属于胸前的 Tango，是另一个刚体。物理自检把它当成手机姿态时，重力均值变成 `[0.15, 0.76, −2.6]` 之类，10 s 窗口误差 40–100°（见 `tests/data/test_raw_ronin.py` 的负对照）。官方做法是在起点（手机与 Tango 预先对齐放置的校准段，`start_calibration` 为该时刻手机→Tango 的旋转）把手机姿态源对齐到 Tango 世界系，之后只使用手机自身的姿态源。
- 对齐旋转 `tango_ori[0] ⊗ start_calibration ⊗ game_rv[0]*` 应当近似为纯偏航。全量统计中其倾斜角的中位数为 0.32°，P95 为 0.87°，最大 1.55°，说明 Tango 世界系与 game_rv 世界系都是重力对齐的。
- **参考姿态的性质：它是“起点对齐后的设备姿态”，不是独立真值。** 129 条使用 game_rv，22 条（`grv_ori_error ≥ 20°`）使用 EKF，1 条选中陀螺积分但被拒收（见第 8 节）。官方给出的所选来源末端对齐误差：中位 5.7°，P95 16.4°，最大 19.6°。其偏航误差会随时间累积，这是本数据集参考姿态的固有精度。
- 使用 EKF 的 22 条：`a002_1 a003_2 a004_2 a007_2 a010_1 a010_3 a011_2 a012_3 a018_2 a021_1 a025_1 a025_3 a034_1 a035_1 a037_3 a046_1 a046_3 a049_1 a051_1 a053_1 a057_1 a058_1`。其中 6 条在 test（`a011_2`，以及 unseen 中的 `a049_1 a051_1 a053_1 a057_1 a058_1`）。官方**测试**代码只用 game_rv；在 IPB 中，该协议对应任务视图的 `orientation=device`（原始 game_rv，并在首个有效样本处用参考姿态补偿常值偏航）。
- 数据中实际存在的逐序列附加属性：`ronin_date`、`ronin_orientation_error_deg`（所选来源的官方末端误差）、`ronin_grv_error_deg`。

## 4. 真值性质与精度

- 位置：胸前 Tango 的 VIO（官方用 area-learning 位姿修正过），**不是手机位置**。手持、口袋、包内的手机相对胸部有数十厘米量级的相对运动，所以位置真值只在轨迹尺度上代表“行人”的运动。
- 姿态：见上，属于对齐后的设备姿态，精度由官方末端误差表征（最大 19.6°）。
- 航向一致性检查（手机加速度 vs Tango 位置加速度）的偏差中位数为 5.3°，P95 为 17.9°，最大 26.1°，明显高于单设备数据集（RIDI 中位 1.1°）。原因有两个：其一，两者来自不同刚体，高频加速度不共享；其二，参考姿态存在偏航漂移。`|δψ| > 15°` 的 13 条为 `a000_8 a004_3 a010_2 a011_2 a016_3 a031_1 a035_3 a045_1 a045_3 a046_2 a047_2 a047_3 a056_3`，其中大多数的官方末端误差在 8–18° 之间。
- 放置方式：`placement = mixed`。论文描述为受试者自然地携带、使用手机（手持、口袋、包等，序列内可能变化），数据没有逐序列标注。

## 5. 元信息映射

| 属性 | 取值 |
|---|---|
| `subject_id` / `group_id` | 目录名 `_` 之前的部分（如 `a000`），共 57 个受试者 |
| `device_id` | `info.json` 的 `device`：samsung1 68、asus4 51、asus3 14、asus7 10、asus5 8、asus6 1 |
| `placement` | `mixed` |
| `position_source` / `orientation_source` | `tango_vio_body_mounted` / `<src>_aligned_to_tango_start` |
| `device_orientation_source` / `body_frame` | `android_game_rotation_vector` / `android_device` |

## 6. 官方划分

官方仓库的 `lists/` 已嵌入转换器（不依赖外部文件），`official_splits()` 会过滤为本地存在的序列：

| 划分 | 官方条数 | 本地存在 | 其中接收 |
|---|---:|---:|---:|
| train | 73 | 72（缺 `a007_3`） | 71 |
| val | 16 | 16 | 16 |
| test_seen | 32 | 32 | 32 |
| test_unseen | 32 | 32 | 32 |
| test（= seen + unseen） | 64 | 64 | 64 |

本地 152 条全部在官方列表中。注意：3 条 train 序列打包在 seen 测试压缩包中（按列表而不是目录归属）。`test_seen` 的受试者与 train/val 重叠，这是官方设计（seen subjects），因此 `group_id` 重叠检查对 RoNIN 会报告预期内的重叠。

## 7. 物理自检（全量）

方法（`physics_check`，阈值定义在 `_phone_utils.py` 顶部）：

1. **重力**：参考姿态把比力旋到世界系，全段均值与 `[0, 0, 9.80665]` 的偏差须 < 0.5 m/s²；
2. **速度**：参考位置在 1 s 网格上差分，“运动秒”（>0.1 m/s）的水平速度中位数须在 0.3–2.0 m/s，静止比例须 ≤ 0.8；
3. **陀螺一致性**：每 10 s 窗口（步长 5 s）内，陀螺积分的相对旋转与参考姿态相对旋转之差的中位数须 < 5°。分别评估不去零偏、常值零偏、60 s 分段零偏三种模型，取最优者；常值零偏估计须 < 0.05 rad/s；
4. **航向一致性**：世界系线加速度（IMU）与位置二阶差分的水平分量，在 0.2 s / 1 s 三角核下做二维对齐，取相关系数高的尺度；相关系数 ≥ 0.3 时须满足 |δψ| ≤ 30°（只拦截坐标系级别的错误）；
5. 诊断量：陀螺与参考姿态差分角速度的最佳常值旋转（“残余旋转”）和时间偏移（角速度模长互相关）。

**全部接收序列**（151 条）

| 指标 | 最小 | P5 | 中位数 | P95 | 最大 |
|---|---:|---:|---:|---:|---:|
| 有效时长 (s) | 226.4 | 305.1 | 570.2 | 821.1 | 1128.7 |
| 路径长度 (m) | 124.8 | 216.3 | 382.1 | 693.6 | 868.1 |
| IMU / 位姿采样率 (Hz) | 200.0 | 200.0 | 200.0 | 200.0 | 200.0 |
| 重力误差 (m/s²) | 0.004 | 0.019 | 0.068 | 0.173 | 0.250 |
| 重力水平分量 (m/s²) | 0.003 | 0.017 | 0.068 | 0.173 | 0.250 |
| 运动秒水平速度中位数 (m/s) | 0.54 | 0.64 | 0.96 | 1.26 | 1.37 |
| 静止比例 | 0.02 | 0.05 | 0.14 | 0.40 | 0.69 |
| 10 s 窗口姿态误差中位数 (°，判定用) | 0.10 | 0.49 | 1.22 | 1.76 | 3.84 |
| 同上，不去零偏 (°) | 0.48 | 0.71 | 1.28 | 1.85 | 3.84 |
| 10 s 窗口姿态误差 P90 (°) | 0.26 | 0.93 | 2.53 | 3.52 | 4.58 |
| 常值陀螺零偏估计 (rad/s) | 0.0002 | 0.0005 | 0.0019 | 0.0086 | 0.0138 |
| 陀螺/参考姿态残余旋转 (°) | 0.00 | 0.03 | 0.40 | 0.95 | 1.97 |
| 陀螺/参考姿态时间偏移 (s) | −0.0037 | −0.0032 | −0.0021 | 0.0023 | 0.0024 |
| 航向偏差 \|δψ\| (°) | 0.05 | 0.65 | 5.31 | 17.91 | 26.10 |
| 航向对齐相关系数 | 0.31 | 0.43 | 0.60 | 0.84 | 0.90 |

合计有效时长 23.11 h，路径 62.0 km。判定所用零偏模型：不去零偏 112 条，常值 30 条，60 s 分段 9 条。

按设备：

| device_id | 条数 | 时长 (h) | 重力误差 中位/最大 | 窗口误差° 中位/最大 | 残余旋转° 最大 | 航向偏差° 中位/最大 | 速度中位 (m/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| asus3 | 14 | 1.96 | 0.12 / 0.18 | 1.43 / 1.90 | 0.78 | 6.8 / 23.7 | 0.97 |
| asus4 | 51 | 8.32 | 0.06 / 0.14 | 1.12 / 1.76 | 1.97 | 4.1 / 25.1 | 1.01 |
| asus5 | 8 | 1.09 | 0.04 / 0.09 | 1.07 / 1.25 | 0.73 | 4.0 / 8.5 | 1.04 |
| asus6 | 1 | 0.08 | 0.05 / 0.05 | 1.05 / 1.05 | 0.12 | 10.6 / 10.6 | 1.12 |
| asus7 | 10 | 1.54 | 0.04 / 0.13 | 0.98 / 1.64 | 1.09 | 8.0 / 14.7 | 1.10 |
| samsung1 | 67 | 10.13 | 0.08 / 0.25 | 1.29 / 3.84 | 1.34 | 5.7 / 26.1 | 0.92 |

解读：重力误差很小，说明参考姿态重力对齐且 z 向上，加速度标定与符号正确；残余旋转 <2°、时间偏移 <4 ms，说明陀螺与参考姿态属于同一刚体、同一时钟（这是必然的，因为参考姿态来自同一手机的 game_rv/EKF）。长时间静止的序列（如 `a006_2`，坐下后仍在操作手机，Tango 位置冻结约 270 s）按“运动秒”判定后可以通过。

## 8. 被拒序列

| sequence_id | 官方划分 | 原因 |
|---|---|---|
| `a011_3` | train | game_rv（21.4°）与 EKF（23.1°）的末端误差都超过 20°，官方规则因此选择陀螺积分（2.99°）；但纯陀螺积分没有重力校正，倾角会漂移，重力均值为 `[−0.12, 0.70, 9.78]`（倾斜 4.1°），不满足重力对齐。数据中没有同时满足官方末端误差与重力对齐的姿态源，因此拒收。 |

## 9. 已知问题与注意事项

- 参考姿态存在偏航漂移（见第 3、4 节）。评测航向相关指标时，应把 RoNIN 的参考姿态视为“对齐后的设备姿态”。
- 位置是胸前设备的位置，手机与胸部之间的相对运动不在真值中。
- `test_seen` 与 train/val 共享受试者（官方设计）。
- `imu_end_gyro_bias` 未使用（与官方一致，只减初始零偏）；陀螺积分源使用零阶保持指数积分，与官方的一阶欧拉积分相比差异可忽略。
- 统一流水线重采样时：本数据已经是 200 Hz 等间隔，IMU 与位姿共用同一时间戳。

## 10. 复现

```bash
PYTHONPATH=src python -m pytest tests/converters/test_ronin.py tests/data/test_raw_ronin.py
```

```python
from inertial_benchmark.data.converters import ronin, _phone_utils as pu
src = "/workspace/webCodex/datasets/imu_odometry/raw/_staging/RoNIN"
for raw in ronin.iter_raw_sequences(src, only=["a000_1"]):
    print(raw.rejected, pu.parse_physics_note(raw.notes))
```


## 11. 转换结果（IPB v1）

输出：`/workspace/webCodex/datasets/ipb/ronin`｜转换器 `ronin@1.0`｜转换时间 2026-09-17｜
指纹 `f29e5c58e71935cc…`

接收 151 / 拒收 1（`a011_3`，第 8 节），共 23.11 h、61.94 km、57 名受试者；原始数据已是 200 Hz 等间隔，
重采样为恒等线性插值，全部序列 `valid` 比例 100%。

| 划分 | 序列 | 时长 (h) | 距离 (km) | 受试者 |
|---|---:|---:|---:|---:|
| `train` | 71 | 10.83 | 28.17 | 40 |
| `val` | 16 | 2.27 | 6.04 | 12 |
| `test_seen` | 32 | 4.93 | 13.30 | 31 |
| `test_unseen` | 32 | 5.07 | 14.43 | 15 |
| `test`（= seen ∪ unseen） | 64 | 10.01 | 27.73 | 46 |

`ipb check data=ronin full=true hash=true`：**通过**（0 错误，5 警告）。警告全部是官方 seen-subject 设定
造成的受试者重叠：`test_seen/train` 31 组、`test_seen/val` 7 组、`train/val` 10 组（官方 val 列表本身与 train
共享受试者）。`test_unseen` 与 train/val 的受试者不相交（否则会报错）。无重复内容、无未分配序列。

### 一致性抽检（5 条随机序列）

> 抽检方法：用固定种子（`numpy.random.default_rng(0)`）从 `dataset.json` 的序列中随机抽 5 条，
> 比较**转换器解析结果**（原生时钟、未重采样，已含该转换器声明的单位/坐标系/外参/时钟修正）与
> **200 Hz 输出**：路径长度按 METRICS 的 1 s 分辨率折线计算（原始侧取每个刻度最近的原生样本，
> 不插值、不跨缺口），窗口取 200 Hz 结果实际覆盖的时间区间；重力对齐加速度均值为
> `mean(R(q) · f)`（原始侧用 SLERP 把参考姿态插到 IMU 时刻）；位置 RMS 是把 200 Hz 结果插回
> 原生位姿时刻后的水平误差。


| 序列 | 时长 (s) | 路径长度 原始 → 200 Hz (m) | Δ | 平均速度 (m/s) | 重力对齐加速度均值 (m/s²) | Δ\|acc\| | 位置 RMS (cm) |
|---|---:|---|---:|---:|---|---:|---:|
| `a026_2` | 597.00 | 571.66 → 571.66 | +0.000% | 0.958 | [−0.028, 0.005, 9.812] | 0.0000 | 0.00 |
| `a033_1` | 457.37 | 339.72 → 339.72 | +0.000% | 0.743 | [0.177, 0.099, 9.816] | 0.0000 | 0.00 |
| `a046_1` | 420.72 | 342.34 → 342.34 | +0.000% | 0.814 | [0.146, 0.067, 9.817] | 0.0000 | 0.00 |
| `a046_3` | 530.33 | 408.11 → 408.11 | −0.000% | 0.770 | [0.149, 0.104, 9.819] | 0.0000 | 0.00 |
| `a057_2` | 677.20 | 276.14 → 276.14 | +0.000% | 0.408 | [−0.081, 0.020, 9.816] | 0.0000 | 0.00 |

路径长度、时长、样本数与原始数据完全一致（截取 `start_frame` 之后的差异只体现在起点）；
加速度均值的水平分量最大 0.18 m/s²（约 1.0°），来自参考姿态的偏航/倾角误差（第 4 节），不是转换误差。
