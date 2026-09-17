# IDOL 数据集卡片

> 转换器：[`converters/idol.py`](../../src/inertial_benchmark/data/converters/idol.py)（`idol@1.0.0`）｜共享自检工具：[`_rig_utils.py`](../../src/inertial_benchmark/data/converters/_rig_utils.py)
> 数据版本：Zenodo 4484093（2021-01-31，`building{1,2,3}.zip`）｜最近核验：2026-09-17｜格式规范：[DESIGN.md 第 2 节](../DESIGN.md)

## 1. 概览

| 项目 | 内容 |
|---|---|
| 来源 | Zenodo <https://zenodo.org/record/4484093>；论文 IDOL（AAAI 2021）；仓库 <https://github.com/KlabCMU/IDOL>（只有 README，无代码） |
| 许可 | CC BY 4.0 |
| 平台 | iPhone 8 刚性固定在 Kaarta Stencil（LiDAR + 相机 + Xsens IMU 的 SLAM 装置）上，手持行走，`placement=handheld` |
| 规模 | 130 条轨迹，19.90 h，3 栋楼，15 名受试者（ID 全局一致） |
| 采样 | 100 Hz，Unix 秒时间戳；42 条含 > 0.05 s 的缺口（最大 15.85 s） |
| 真值 | Stencil 的 LiDAR-视觉-惯性 SLAM；论文报告在 Vicon 下姿态 RMS < 1.5°、位置 RMS < 10 cm，并对轨迹做了低通滤波 |
| 可选依赖 | 读取 `.feather` 需要 `pyarrow`（本机验证版本 24.0.0）：`pip install pyarrow`。`pyproject.toml` 未改动，建议维护者增加 `idol = ["pyarrow"]` 可选依赖组 |

## 2. 原始文件与字段

```text
building{1,2,3}/{train,known,unknown}/<k>.feather + metadata.json   # train 只在 building1
```

| 列（按名称访问；列顺序在文件间不同） | 单位 / 含义 | IPB v1 去向 |
|---|---|---|
| `timestamp` | Unix 秒 | `imu_time = pose_time`，`start_time_unix` |
| `orient[WXYZ]` | Stencil 机体 body→world，wxyz | 调平 + 换到 iPhone 机体后作为 `orientation` |
| `processedPos[XYZ]` | 米（文档未写，由步速与平面拟合核实），已平滑 | 调平后作为 `position` |
| `iphoneAcc[XYZ]` | G，iOS 符号（静止平放读数为 −1 G） | `accelerometer = −9.80665 · iphoneAcc` |
| `iphoneGyro[XYZ]` | rad/s，iPhone 机体系 | `gyroscope`（不变） |
| `iphoneOrient[WXYZ]` | CoreMotion 姿态，body→world，z 轴向上，偏航参考不同 | `device_orientation` |
| `stencilAcc*` / `stencilGyro*` | Xsens，Stencil 机体系（与真值同系） | 只用于调平与诊断 |
| `iphoneMag*` | 磁力计 | 未使用 |
| `index`（及 `building3/known/12` 多出的 `level_0`） | pandas 索引残留 | 忽略，并在 notes 中记录 |

`metadata.json`：`subjectID` 与 `calibration`（`none` 93 条、`start` 30 条、`end` 7 条，指开头/结尾的三轴绕转标定动作）。

## 3. 约定核实与转换

| 问题 | 结论 | 证据 |
|---|---|---|
| iPhone 加速度单位与符号 | G，且为 **−比力** | 映射到 Stencil 系、再用 Stencil 姿态旋到世界系后的均值 ≈ [0, 0, −1.02] G；同样的数据用 CoreMotion 姿态旋转得到 ≈ [0, 0, −0.98] G。取反并乘 g 后，重力检查全部通过 |
| iPhone 陀螺符号 | 与右手系一致，不变号 | 与 Stencil 陀螺逐轴相关：x −0.94、y −0.93、z +0.95，与 README 的 x→−x、y→−y、z→z 轴映射吻合 |
| `orient` 的性质 | **Stencil 机体**姿态（不是手机），body→world | README；Stencil 陀螺与 `orient` 差分角速度之间的对齐旋转只有 0.36°（中位），即两者同系；Stencil 加速度经 `orient` 正向旋转后均值 ≈ [0, 0, 9.8]，逆向旋转的残差明显更大 |
| iPhone↔Stencil 外参 | `R_stencil_iphone` = 旋转向量 [−0.01853, 0.02801, 3.12916] rad，与 README 先验 Rz(180°) 相差 1.42° | 陀螺与真值角速度做 Wahba 对齐，130 条中 121 条的估计彼此相差 < 3°，取其稳健平均；三栋楼各自的平均值与全局值相差 0.2–0.7° |
| 世界系 | **相对重力倾斜 0.31–8.41°**（中位 1.67°） | 见下文“调平” |
| 位置单位 | 米 | 水平速度中位数 0.60–1.29 m/s |

转换规则（逐条写入 `notes`）：

1. `f = −9.80665 · iphoneAcc`；陀螺不变；`body_frame="ios_device"`，**IMU 保留在 iPhone 机体系**。
2. **调平真值世界系**：用 Stencil 自带 IMU（与真值同系，且独立于待检的 iPhone IMU）的平均比力方向 `u = mean(R_stencil·f_stencil)`，
   以最小旋转把 `u` 转到 `e_z`，同时作用于姿态与位置，偏航不变。独立验证：对位置点云做地面平面拟合，两种倾角估计的差值中位 0.48°、
   p90 1.5°；调平后单条轨迹的高度范围中位数从 1.5 m 降到 1.1 m（最大从 14.1 m 降到 6.2 m）。倾角 > 2° 的有 52 条，> 4° 的有 19 条，
   最大为 `building3_known_21`（8.41°）、`building3_known_15`（7.25°）、`building2_known_5`（7.08°）。
   不调平时有 40 条的 iPhone 重力误差超过 0.5 m/s²（最大 1.47），Stencil 自带 IMU 也有 35 条超过，说明问题出在真值世界系而不是手机。
3. **换到 iPhone 机体**：`q_ref = q_stencil_leveled ⊗ q(R_stencil_iphone)`。
4. **时延修正**（两个独立证据同时成立）：0.1 s 平滑后的角速度互相关给出 |τ| ≥ 0.03 s 且峰值 ≥ 0.7，并且 IMU 平移 τ 后，1 s 窗口
   去偏陀螺误差中位数至少下降 20%。满足时令 `pose_time ← pose_time − τ`。由于 `processedPos` 被平滑过，这里不能像 RNIN 那样用
   加速度一致性确认，因此改用陀螺确认。共修正 20 条：

   | 序列 | pose_time 平移 (s) | 1 s 去偏误差 (°) | 序列 | pose_time 平移 (s) | 1 s 去偏误差 (°) |
   |---|---:|---|---|---:|---|
   | `building2_known_8` | −0.190 | 3.89 → 1.27 | `building3_known_1` | −0.060 | 1.78 → 1.01 |
   | `building3_known_15` | −0.170 | 4.53 → 2.70 | `building2_known_3` | −0.050 | 1.29 → 0.57 |
   | `building2_unknown_12` | −0.110 | 2.87 → 0.82 | `building3_known_4` | −0.050 | 1.73 → 0.84 |
   | `building1_train_39` | −0.080 | 1.72 → 1.25 | `building2_known_10` | +0.050 | 1.38 → 0.87 |
   | `building1_unknown_13` | −0.080 | 2.25 → 0.91 | `building2_unknown_7` | −0.040 | 1.10 → 0.83 |
   | `building2_known_1` | −0.080 | 2.15 → 0.68 | `building3_known_3` | −0.040 | 1.34 → 0.89 |
   | `building2_known_6` | +0.080 | 1.99 → 1.39 | `building3_known_5` | −0.040 | 1.33 → 0.68 |
   | `building2_unknown_13` | −0.070 | 2.29 → 0.91 | `building2_unknown_3` | −0.030 | 0.55 → 0.36 |
   | `building3_known_16` | −0.070 | 1.94 → 0.72 | `building3_known_0` | −0.030 | 1.19 → 0.89 |
   | `building3_known_13` | −0.030 | 1.07 → 0.62 | `building3_known_17` | −0.030 | 1.02 → 0.66 |

   另有 7 条估计出 0.04–0.07 s 的时延但未通过确认，仅记警告：`building1_train_11`、`building1_unknown_12`、`building2_unknown_11`、
   `building2_unknown_14`、`building3_known_11`、`building3_known_22`、`building3_unknown_3`。`building3_known_15` 在修正前因 10 s 去偏误差
   5.68° 被拒，修正后通过。
5. 时间缺口原样保留，不插值：42 条含缺口，`building3_known_14` 最长 15.85 s，`building3_known_12` 为 4.66 s。
6. README 中 building2/3 的 xy 平面旋转偏移（1.8510 / 0.2822 rad，用于还原到各楼原始坐标）只影响偏航，本转换器不应用。
7. 开头/结尾的同步晃动和标定绕转段不裁剪（与论文评测一致），`attrs['calibration_motion']` 记录标定位置。

## 4. 真值性质与精度

- Stencil SLAM 的精度指标来自小型 Vicon 场地；在整栋楼尺度上，我们观察到**世界系倾斜最高 8.4°**（已调平），
  调平后仍有随时间变化的 ±2–3° 残余倾斜（例如 `building2_known_5` 调平前逐分钟的倾角在 4.5–10° 之间波动，整体调平量为 7.1°）。
- `processedPos` 经过低通平滑：位置二阶差分在 0.5–4 Hz 频段几乎没有能量（加速度一致性比例 k 中位 0.003），因此逐序列的加速度一致性诊断
  对 IDOL 无信息量，转换器关闭了该项警告。由位置差分得到的 1–2 s 窗口速度目标仍然可用，但会比真实速度更平滑。
- iPhone 与 Stencil 的杠杆臂小于手机尺寸（README），未做补偿。

## 5. 官方划分与分组键

| 划分 | 序列数 | 接收 | 拒收 | 接收时长 (h) | 受试者 ID |
|---|---:|---:|---:|---:|---|
| `train`（building1/train） | 43 | 43 | 0 | 5.80 | 0, 1, 3, 4, 5, 6, 7, 11 |
| `test_known`（三栋 known） | 51 | 51 | 0 | 8.17 | 0, 1, 3, 4, 5, 6, 7, 11 |
| `test_unknown`（三栋 unknown） | 36 | 36 | 0 | 5.93 | 2, 8, 9, 10, 12, 13, 14 |
| `test` = known ∪ unknown | 87 | 87 | 0 | 14.10 | |
| `test_known_building1` | 15 | 15 | 0 | 2.36 | |
| `test_unknown_building1` | 14 | 14 | 0 | 2.15 | |

各子集原始时长：building1 known 2.36 h / unknown 2.15 h / train 5.80 h；building2 known 2.13 h / unknown 2.89 h；building3 known 3.68 h / unknown 0.89 h。

- 依据 README 与论文：known 与 train 来自同一受试者池，unknown 受试者与之不相交；论文 Table 2 在 building1 train 上训练、在 building1
  known（2.4 h）/unknown（2.2 h）上测试，对应 `test_known_building1` / `test_unknown_building1`。
- 论文 Table 1/3 写“各楼栋分别训练、分别测试”，但发布数据中 building2/3 **没有训练子集**，切分细节也未公开。因此按本卡片的映射，
  building2/3 在 `test_known` / `test_unknown` 中属于“未见楼栋”测试，这与论文的 building2/3 设定不同。
- 官方没有 val，因此不返回 `val`，由统一流水线从 `train` 中按 `group_id` 抽取。
- `list_sequences()` 枚举 `building*/{train,known,unknown}/*.feather`，`sequence_id = building<b>_<subset>_<k>`；
  发布版的 130 条全部属于这三个子集，因而都在官方划分中。
- 附加根属性：`building`、`official_subset`、`calibration_motion`、`imu_calibration`；`start_time_unix` 为 IMU 第一个样本的 Unix 秒。
- `group_id = subject_id = subjectNN`。`test_known` 与 `train` 的受试者重叠是设计使然（known 的含义就是“见过的受试者”），`ipb check`
  会报告这一重叠。

## 6. 物理自检（全量 130 条）

方法与阈值见 [TLIO 卡片第 6 节](tlio.md#6-物理自检全量-354-条)；IDOL 关闭加速度一致性警告（原因见第 4 节）。

| 指标 | min | p5 | 中位数 | p95 | max |
|---|---:|---:|---:|---:|---:|
| 重力均值误差 (m/s²) | 0.188 | 0.214 | 0.227 | 0.245 | 0.257 |
| 重力方向倾角 (°) | 0.002 | 0.005 | 0.021 | 0.060 | 0.137 |
| 水平速度中位数 (m/s) | 0.602 | 0.754 | 0.972 | 1.162 | 1.287 |
| 水平速度 p95 (m/s) | 0.722 | 0.923 | 1.135 | 1.301 | 1.565 |
| 10 s 陀螺误差中位数 (°) | 7.474 | 7.820 | 8.580 | 9.284 | 9.673 |
| 去常值零偏后 (°) | 0.571 | 0.727 | 1.119 | 2.577 | 4.083 |
| 参考隐含陀螺零偏范数 (rad/s) | 0.0147 | 0.0152 | 0.0162 | 0.0168 | 0.0184 |
| 加速度一致性比例 k | 0.001 | 0.001 | 0.003 | 0.683 | 0.835 |
| 角速度互相关时延 (s)（修正后） | −0.060 | −0.020 | 0.000 | 0.030 | 0.070 |
| 角速度互相关峰值 | 0.807 | 0.905 | 0.985 | 0.996 | 0.998 |
| 时间重叠比例 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

解读：

- 重力误差几乎全在 z 分量（倾角中位 0.02°），来自 iPhone 加速度计约 +1.9% 的比例误差（比力模长中位 9.99 m/s²）；
- iPhone 8 原始陀螺有稳定的零偏，约 [−0.011, −0.005, −0.011] rad/s，所有序列几乎相同，导致 10 s 误差为 7.5–9.7°；去掉常值零偏后为
  0.6–4.1°。130 条都因此记警告而非拒收，**零偏未从存储的 IMU 中去除**。

## 7. 被拒序列清单

无（0/130）。

## 8. 已知问题

1. 真值世界系倾斜（已调平）以及调平后仍然存在的时变残余倾斜。
2. `processedPos` 被平滑；逐序列残余时延 0.03–0.19 s（20 条已修正，7 条未确认）。
3. iPhone IMU 未标定（陀螺零偏约 0.016 rad/s，加速度比例误差约 2%）。
4. 列顺序不固定、`building3/known/12` 多一列 `level_0`，转换器按列名读取。
5. 官方划分对 building2/3 的训练数据描述与发布内容不一致（见第 5 节）。

## 9. 复现

```python
from inertial_benchmark.data.converters import idol
splits = idol.official_splits("/path/to/IDOL")     # 目录下含 building1/2/3
for raw in idol.iter_raw_sequences("/path/to/IDOL"):
    ...
```

真实数据抽样测试：`PYTHONPATH=src python -m pytest tests/data/test_raw_idol.py`（环境变量 `IPB_RAW_IDOL` 可覆盖路径；缺少 pyarrow 时自动跳过）。
