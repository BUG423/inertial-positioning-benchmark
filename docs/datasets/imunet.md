# IMUNet 数据集卡片

> 转换器：[`converters/imunet.py`](../../src/inertial_benchmark/data/converters/imunet.py)，`imunet@1.0`；
> 共享工具与物理自检：[`converters/_phone_utils.py`](../../src/inertial_benchmark/data/converters/_phone_utils.py)。
> 统计基于本机 `/workspace/webCodex/datasets/imu_odometry/raw/IMUNet/IMUNet_dataset`（126 条），2026-09-17。

## 1. 概览

| 项 | 内容 |
|---|---|
| 论文 | Zeinali, Zanddizari, Chang. *IMUNet: Efficient Regression Architecture for Inertial IMU Navigation and Positioning*, IEEE TIM 2024 |
| 数据 | `IMUNet_dataset`（官方 README 中的 Google Drive 链接） |
| 官方代码 | <https://github.com/BehnamZeinali/IMUNet>（提交 `c57f14d`，`Datasets/proposed/read_data_s10.py`、`read_data_tango.py`）；采集 App <https://github.com/BehnamZeinali/IMUNet_Android>（提交 `ef6439f`） |
| 许可 | 未给出数据许可条款；按学术使用并引用论文处理，不再分发 |
| 设备 | Tango 手机（RIDI 方法，49 条）；Samsung Galaxy S10（40 条）、S21（26 条）、小米（11 条），后三者以 ARCore 提供参考位姿 |
| 受试者 | 4 人（目录中编号为 1、2、4、5），室内/室外两种环境 |
| 输入 `source` | `IMUNet_dataset` 目录或其父目录 |

## 2. 原始文件与字段

目录名为 `<Indoor|Outdoor>_Subject_<k>_<设备>_<n>`，其中 8 个室外目录拼写为 `Subjetc`。每个目录只有 `processed/data.csv`（外加 `data_plain.txt`）；只有 `Outdoor_Subject_1_Tango_6` 额外保留了原始 txt。`data.csv` 的列与 RIDI 相同（见 [RIDI 卡片](ridi.md#2-原始文件与字段)），但设备不同，含义也不同：

| 列 | Tango 设备 | ARCore 设备（S10/S21/小米） |
|---|---|---|
| `time` | Tango 位姿时间戳（ns），约 200 Hz | ARCore `Frame.getTimestamp()`（ns，约 30 Hz），官方用三次样条把**时间戳**加密到约 200 Hz |
| `gyro_*` / `acce_*` | Android IMU，线性插值到 `time` | 同左（插值时允许外推） |
| `pos_*` | Tango `START_OF_SERVICE`（z 向上） | ARCore 世界系位置，官方已做 `(x, y, z) → (x, −z, y)`，即转成 **z 向上** |
| `ori_*` | Tango `START_OF_SERVICE → DEVICE`，wxyz | ARCore `camera.getPose()` 的旋转（xyzw 已换成 wxyz），**仍是 y 向上世界系、相机坐标系**，官方没有转换 |
| `rv_*` | Android game rotation vector，wxyz | 同左 |

## 3. 约定与转换（逐设备由物理自检确定）

| 设备 | 参考姿态 `pose/orientation` | 参考位置 |
|---|---|---|
| Tango | `ori` 原样 | `pos` 原样 |
| S10 / S21 / 小米 | `R_x(+90°) ⊗ ori ⊗ R_z(+90°)` | `pos` 原样（已是 z 向上） |

- `R_x(+90°)` 把 ARCore 的 y 向上世界系转成 z 向上，与官方脚本对位置所做的变换完全相同，因此位置与姿态处在同一世界系（航向一致性检查也证实了这一点，见第 7 节）。
- `R_z(+90°)` 是相机坐标系（`Camera.getPose()`：+X 为图像读出方向的右侧，−Z 为镜头朝向）到 Android 传感器坐标系的常值旋转，符合后置摄像头传感器横向安装的情况。它由陀螺与姿态差分角速度的手眼对齐估计得到：仅做 y→z 变换时，三种 ARCore 设备的估计值都是 `q_sb ≈ [0.707, 0, 0, 0.707]`，残差约 0.04–0.07 rad/s。应用后，逐序列的残余旋转 ≤ 1.5°。
- **四元数不是共轭关系。** 把 `ori` 当作共轭（world→body）时，手眼残差会大一个数量级（0.25–0.46 rad/s），重力均值落在 **−x 轴**（例如 `[−9.73, −0.33, 1.43]`）。这正是旧流水线 77/126 条“重力在 −x”问题的成因。另外，只做 y→z 变换而不补相机外参，或者直接使用原始 `ori`，同样都会失败。以上负对照都写在 `tests/data/test_raw_imunet.py` 与 `tests/converters/test_imunet.py` 中。
- 陀螺、加速度原样使用（rad/s、m/s² 比力），数据集没有公布标定参数。
- 时间：`time / 1e9` 秒（开机时钟），IMU 与位姿共用；ARCore 帧时间戳与传感器时间戳实测处于同一时钟（时间偏移 < 1 ms）。

## 4. 真值性质与精度

- 两类参考都是“同设备 VIO”：Tango VIO（约 200 Hz）或 ARCore VIO（原始约 30 Hz，已由官方插值到 200 Hz），**不是**动捕真值。ARCore 的位置是相机光心的位置，与 IMU 之间的杆臂被忽略。
- Tango 设备的加速度计读数偏低约 2%（世界系重力 z 分量约 9.55–9.62）；S10/S21 偏高约 1%。
- 放置方式：整段序列中手机都保持竖屏直立、后摄朝前（屏幕与水平面夹角中位数约 83°），这与 VIO 需要相机视野的要求一致，但无法区分“手持”和“胸前固定”，因此 `placement = unknown`。

## 5. 元信息映射

| 属性 | 取值 |
|---|---|
| `subject_id` / `group_id` | `subject<k>`（1/2/4/5） |
| `device_id` | `tango_phone`、`samsung_galaxy_s10`、`samsung_galaxy_s21`、`xiaomi` |
| `placement` | `unknown` |
| `position_source` / `orientation_source` | Tango：`tango_vio_same_device`；ARCore：`arcore_vio_same_device` / `arcore_camera_pose_to_android_body` |
| `device_orientation_source` / `body_frame` | `android_game_rotation_vector` / `android_device` |
| 附加属性 | `imunet_environment`（indoor/outdoor） |

## 6. 官方划分

以**数据包自带**的 `list_train.txt` / `list_test.txt` 为准（90/36，覆盖全部 126 个目录）。GitHub 仓库 `Datasets/proposed/` 下的同名列表使用旧命名（`behnam_*`，36/13 条），与发布数据不对应，所以不使用。

| 划分 | 官方条数 | 本地存在 | 其中接收 |
|---|---:|---:|---:|
| train | 90 | 90 | 81 |
| test | 36 | 36 | 32 |

- 无官方 val；train 与 test 共享全部 4 名受试者（官方按序列划分）。
- **跨划分泄漏**：`Outdoor_Subjetc_1_S10_13`（train）与 `Outdoor_Subjetc_1_S10_16`（test）的 `data.csv` 逐字节相同（md5 `8e2e9f09…`）。转换器保留 test 中的副本，拒收 train 中的副本。重复表 `KNOWN_DUPLICATES` 的完整性由真实数据测试扫描全部文件来保证。

## 7. 物理自检（全量）

检查方法与阈值见 [RoNIN 卡片第 7 节](ronin.md#7-物理自检全量)。IMUNet 另有两项转换器内检查：固定约定应用后，手眼残余旋转须 ≤ 5°；原始加速度计相对 Android 融合输出 `grav + linacce` 的比例偏差须 ≤ 5%。后者不依赖参考姿态。

**全部接收序列**（113 条）

| 指标 | 最小 | P5 | 中位数 | P95 | 最大 |
|---|---:|---:|---:|---:|---:|
| 有效时长 (s) | 47.5 | 71.6 | 265.3 | 428.8 | 466.0 |
| 路径长度 (m) | 50.7 | 78.5 | 296.0 | 445.5 | 524.2 |
| IMU / 位姿采样率 (Hz) | 198.6 | 198.6 | 200.0 | 200.4 | 204.1 |
| 重力误差 (m/s²) | 0.088 | 0.099 | 0.174 | 0.265 | 0.351 |
| 重力水平分量 (m/s²) | 0.004 | 0.008 | 0.035 | 0.130 | 0.320 |
| 运动秒水平速度中位数 (m/s) | 0.84 | 0.96 | 1.12 | 1.30 | 1.52 |
| 静止比例 | 0.00 | 0.00 | 0.00 | 0.03 | 0.10 |
| 10 s 窗口姿态误差中位数 (°，判定用) | 0.40 | 0.51 | 0.85 | 1.42 | 2.14 |
| 同上，不去零偏 (°) | 0.42 | 0.63 | 1.11 | 4.12 | 5.76 |
| 10 s 窗口姿态误差 P90 (°) | 0.76 | 1.22 | 1.86 | 3.27 | 6.66 |
| 常值陀螺零偏估计 (rad/s) | 0.0002 | 0.0004 | 0.0012 | 0.0065 | 0.0103 |
| 陀螺/参考姿态残余旋转 (°) | 0.12 | 0.14 | 0.50 | 1.17 | 1.51 |
| 陀螺/参考姿态时间偏移 (s) | −0.0003 | −0.0003 | 0.0000 | 0.0002 | 0.0009 |
| 航向偏差 \|δψ\| (°) | 0.02 | 0.12 | 2.41 | 9.26 | 22.58 |
| 航向对齐相关系数 | 0.08 | 0.24 | 0.65 | 0.87 | 0.93 |

合计有效时长 7.91 h，路径 30.8 km。判定所用零偏模型：常值 71 条，60 s 分段 24 条，不去零偏 18 条。

按设备：

| device_id | 条数 | 时长 (h) | 重力误差 中位/最大 | 窗口误差° 中位/最大 | 残余旋转° 最大 | 航向偏差° 中位/最大 | 速度中位 (m/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| samsung_galaxy_s10 | 38 | 2.39 | 0.16 / 0.35 | 1.01 / 2.14 | 1.51 | 3.4 / 22.6 | 1.14 |
| samsung_galaxy_s21 | 26 | 1.99 | 0.10 / 0.15 | 0.75 / 1.14 | 0.82 | 2.8 / 18.7 | 1.07 |
| tango_phone | 49 | 3.53 | 0.22 / 0.29 | 0.84 / 1.41 | 0.30 | 1.3 / 19.6 | 1.13 |

说明：

- S10 的陀螺残余零偏较大：部分序列 x 轴约 0.009 rad/s（0.5°/s），不去零偏时 10 s 窗口误差最高 5.8°，按常值零偏补偿后降到 2.1° 以下。这是传感器质量问题，不是约定问题。
- |δψ| > 15° 的 4 条（`Indoor_Subject_4_S10_1`、`Indoor_Subject_4_S10_3`、`Outdoor_Subject_1_Tango_5`、`Outdoor_Subject_5_S21_4`）相关系数只有 0.15–0.53，位置二阶差分噪声大（VIO 重定位跳变），航向估计本身不可靠。在相关系数高的序列上，位置与姿态的世界系是一致的。

## 8. 被拒序列（13 条）

| sequence_id | 官方划分 | 原因 |
|---|---|---|
| `Indoor_Subject_1_S10_1` | test | 世界系重力均值 `[−0.81, −0.15, 9.93]`（倾斜 4.8°），而且 Android 重力传感器旋到世界系后给出相同的偏差 `[−0.79, −0.17, 9.77]`，并在整段序列中保持不变，说明这次 ARCore 会话的世界系本身倾斜了约 4.7°（数据问题） |
| `Outdoor_Subject_1_Xiaomi_1` | train | 加速度计比例误差：原始 `TYPE_ACCELEROMETER` = 0.906 × Android 自身的 `grav + linacce` |
| `Outdoor_Subject_1_Xiaomi_3` | train | 同上，0.921 × |
| `Outdoor_Subject_1_Xiaomi_4` | train | 同上，0.908 × |
| `Outdoor_Subject_1_Xiaomi_8` | train | 同上，0.903 × |
| `Outdoor_Subject_1_Xiaomi_10` | test | 同上，0.902 × |
| `Outdoor_Subject_1_Xiaomi_12` | train | 同上，0.897 × |
| `Outdoor_Subject_1_Xiaomi_16` | test | 同上，0.913 × |
| `Outdoor_Subject_2_Xiaomi_1` | train | 同上，0.910 × |
| `Outdoor_Subject_2_Xiaomi_2` | test | 同上，0.911 × |
| `Outdoor_Subject_2_Xiaomi_3` | train | 同上，0.911 × |
| `Outdoor_Subject_2_Xiaomi_4` | train | 同上，0.917 × |
| `Outdoor_Subjetc_1_S10_13` | train | 与 test 中的 `Outdoor_Subjetc_1_S10_16` 逐字节相同 |

小米设备的原始加速度计读数约为真实比力的 0.90–0.92 倍：世界系重力 z 分量为 8.73–8.96，而 Android 融合重力是 9.77–9.79，其姿态与陀螺检查都正常（窗口误差 < 1.3°）。这是设备层面的比例误差，数据集没有标定可以修正，因此整类拒收。**如果项目决定接受“按设备统一比例因子（约 1.10）校正”这种非官方修正，这 11 条可以恢复**，但需要在转换报告中明确标注。

拒收后 test 从 36 条变为 32 条（去掉 `S10_1` 与 3 条小米），train 从 90 条变为 81 条。

## 9. 已知问题与注意事项

- ARCore 设备的参考位姿原始约为 30 Hz，官方插值后才是 200 Hz，高频姿态信息来自插值。
- IMU 被官方插值到位姿时间戳（允许外推），原始 IMU 时间戳已经丢失。
- 目录名中有拼写错误 `Subjetc`，转换器保留原名作为 `sequence_id`。
- 放置方式未知（见第 4 节）。

## 10. 复现

```bash
PYTHONPATH=src python -m pytest tests/converters/test_imunet.py tests/data/test_raw_imunet.py
```
