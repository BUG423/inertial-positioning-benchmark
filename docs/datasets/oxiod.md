# OxIOD 数据集卡片

> 转换器：[`converters/oxiod.py`](../../src/inertial_benchmark/data/converters/oxiod.py)，`oxiod@1.1`；
> 共享工具与物理自检：[`converters/_phone_utils.py`](../../src/inertial_benchmark/data/converters/_phone_utils.py)。
> 统计基于本机 `/workspace/webCodex/datasets/imu_odometry/raw/_staging/OxIOD`（`Oxford Inertial Odometry Dataset/`，128 条 raw 序列），2026-09-17。

## 1. 概览

| 项 | 内容 |
|---|---|
| 论文 | Chen, Zhao, Lu, Wang, Markham, Trigoni. *OxIOD: The Dataset for Deep Inertial Odometry*, 2018（arXiv:1809.07491） |
| 数据 | <http://deepio.cs.ox.ac.uk/>（`Oxford Inertial Odometry Dataset_2.0.zip`） |
| 官方代码 | 无公开的数据读取代码；约定依据数据包内 `ReadMe.txt`、各场景 `Train.txt`/`Test.txt`/`*.xlsx`，并由物理自检确认 |
| 许可 | 牛津大学发布，供学术/非商业研究使用并引用论文；不再分发 |
| 采集 | iPhone（CoreMotion，约 100 Hz）+ Vicon 动捕（100 Hz；running / slow walking 为 200 Hz）；多设备子集含 iPhone 5、iPhone 6、Nexus 5 |
| 转换范围 | 6 个放置/运动场景（handheld 24、pocket 11、handbag 8、trolley 13、slow walking 8、running 7）+ multi users 31 + multi devices 26，共 128 条，全部使用 `raw/` |
| 不转换 | 各场景的 `syn/`（时间戳已被导出成 `1.50E+11` 这类科学计数法，信息丢失）；`large scale/`（Tango 真值、IMU 与真值分目录，floor1 11 条、floor4 18 条）；`test/`（无 raw/syn 区分） |
| 输入 `source` | `Oxford Inertial Odometry Dataset` 目录或其父目录 |

## 2. 原始文件与字段

`<场景>/<会话>/raw/`（会话为 `dataN`、`userN` 或设备名）下成对出现 `imu<k>.csv` 和 `vi<k>.csv`（trolley/data2 中名为 `hand<k>.csv`）。文件都没有表头。

**`imu<k>.csv`（iPhone，16 列）**

| 列 | 内容 | 单位 |
|---|---|---|
| 0 | 时间 | unix 秒（文件中只保留 2 位小数，分辨率 0.01 s） |
| 1–3 | attitude roll, pitch, yaw | rad（CoreMotion，参考系 z 竖直、偏航任意） |
| 4–6 | rotation rate x/y/z | rad/s（CoreMotion 已去零偏） |
| 7–9 | gravity x/y/z | G（iOS 约定：屏幕朝上静止时为 (0, 0, −1)） |
| 10–12 | user acceleration x/y/z | G（iOS 约定） |
| 13–15 | magnetic field | μT（未使用） |

**`vi<k>.csv` / `hand<k>.csv`（Vicon，9 列）**：时间（unix 纳秒）、帧号、位置 x/y/z（m）、四元数 **x, y, z, w**。

**`multi devices/nexus 5/raw/imu<k>.csv`（Android 长表）**：毫秒时间、传感器类型码（1 = 加速度计 m/s²，4 = 陀螺 rad/s，2 = 磁力计）、x、y、z。各类型有各自的时间戳，约 99.3 Hz。

## 3. 约定与转换

| 量 | 转换 | 依据 |
|---|---|---|
| IMU 行顺序 | 按时间稳定排序后去掉重复时间戳。数据中偶有相邻两行颠倒（`…37, .39, .38, .40`），iPhone 6 约 2%；排序后 iPhone 6 恰好是严格的 10 ms 等间隔，陀螺信号也更平滑，说明是行写出顺序错了，而不是时间抖动 | 实测 |
| 比力 | `−(gravity + user_acc) × 9.80665` | 物理自检：取 + 号时世界系重力均值为 −9.8，取 − 号时为 +9.8 |
| 陀螺 | rotation rate 原样 | — |
| 设备姿态 `imu/orientation` | `R_z(yaw) ⊗ R_x(pitch) ⊗ R_y(roll)`（body→world，z 向上、偏航任意） | 穷举 96 种欧拉约定：只有该约定与 iOS gravity 方向误差为 0°，且与陀螺的手眼残余 < 2° |
| Vicon 四元数 | xyzw → wxyz，归一化、符号连续化；非有限、范数偏离 1 超过 0.1 或位置全为 0 的行标为无效 | ReadMe |
| Vicon 世界系 | 原样（重力对齐、z 向上，由重力检查确认） | 自检 |
| 刚体外参 `q_sb` | 逐序列估计：陀螺角速度与 Vicon 差分角速度在 20 Hz 箱形平均网格上做 Wahba 最小二乘，并剔除一轮残差大于中位数 3 倍的格点；参考姿态 `q_wb = q_vicon ⊗ q_sb` | 手眼标定 |
| 时钟 | 线性模型 `vicon_time = imu_time + offset + skew·(imu_time − t_ref)`：先用角速度模长互相关（±5 s）粗估，再在外参下用三轴互相关（±0.3 s）细化；每 60 s 一段重复细化并线性拟合，累计漂移 > 5 ms 时才启用 skew；`pose_time` 由逆映射得到，统一到手机时钟 | 实测 |
| Vicon 野值 | 相邻帧隐含角速度 > 600°/s 或速度 > 10 m/s 的帧，连同前后 0.1 s 标为 `pose_valid = False`（标记识别造成的 180° 单帧翻转、短时抖动、位置跳变）。全量共屏蔽 53,222 个样本（0.9%），单条序列有效比例 ≥ 0.95 | 实测 |
| Nexus 5 | 取加速度计时间戳为 IMU 时钟（毫秒 → 秒），陀螺按时间线性插值到这些时刻；Android 加速度计本身即为比力；没有设备姿态 | 数据格式 |
| 起始时间 | `start_time_unix` = 首个 IMU 时间戳（手机时钟） | — |
| 机体系 | iPhone：`ios_device`（x 右、y 上、z 出屏，与 Android 轴向相同）；Nexus 5：`android_device` | — |

外参与时钟的估计值和残差逐条写入 `notes`，并以附加属性 `oxiod_extrinsic_wxyz`、`oxiod_clock_offset_s`、`oxiod_clock_skew_ppm` 保存。

### 外参与时钟（按会话汇总，接收序列）

| 会话 | 条数 | placement（来源） | 时钟偏移 (s) | 速率差 (ppm) | 外参旋转角 (°) | 手眼残差中位数最大值 (rad/s) | 位姿有效比例最小值 |
|---|---:|---|---|---|---|---:|---:|
| handbag_data1 | 4 | bag（场景目录） | 0.001 … 0.010 | 0 | 174.2 … 174.3 | 0.076 | 0.995 |
| handbag_data2 | 3 | bag（场景目录） | −0.066 … −0.063 | −33 … 0 | 174.2 … 174.7 | 0.076 | 0.985 |
| handheld_data1 | 7 | handheld（场景目录） | −0.007 … 0.056 | 0 | 168.6 … 169.6 | 0.048 | 0.982 |
| handheld_data2 | 3 | handheld（场景目录） | −0.071 … −0.064 | 0 | 169.7 … 170.1 | 0.053 | 0.992 |
| handheld_data3 | 5 | handheld（场景目录） | −0.008 … 0.003 | 0 | 176.3 … 177.2 | 0.050 | 0.986 |
| handheld_data4 | 5 | handheld（场景目录） | −0.015 … 0.003 | 0 | 172.0 … 172.4 | 0.055 | 0.988 |
| handheld_data5 | 4 | handheld（场景目录） | −0.010 … 0.007 | 0 | 172.2 … 172.7 | 0.058 | 0.992 |
| multi_devices_iphone5 | 9 | handheld / unknown（重力方向） | −0.087 … 0.271 | −483 … 0 | 174.1 … 174.9 | 0.261 | 0.959 |
| multi_devices_iphone6 | 9 | handheld / unknown（重力方向） | −0.178 … −0.027 | −70 … 54 | 174.3 … 175.0 | 0.105 | 0.961 |
| multi_devices_nexus5 | 8 | handheld（重力方向） | 0.209 … 0.290 | 0 … 10 | 172.2 … 173.6 | 0.043 | 0.999 |
| multi_users_user2 | 6 | handheld / unknown（重力方向） | −0.009 … 0.044 | 0 | 172.5 … 175.8 | 0.080 | 0.991 |
| multi_users_user3 | 6 | handheld / pocket / bag（Readme） | −0.094 … 0.153 | −494 … 0 | 174.1 … 174.8 | 0.101 | 0.985 |
| multi_users_user4 | 9 | handheld / unknown（重力方向） | −0.026 … 0.019 | 0 | 173.7 … 175.2 | 0.111 | 0.987 |
| multi_users_user5 | 9 | handheld / pocket / bag（Readme） | −0.051 … 0.008 | 0 | 171.9 … 175.8 | 0.097 | 0.950 |
| pocket_data1 | 5 | pocket（场景目录） | −0.005 … −0.004 | 0 | 174.9 … 175.2 | 0.090 | 0.996 |
| pocket_data2 | 6 | pocket（场景目录） | −0.031 … 0.015 | 0 | 175.5 … 175.7 | 0.092 | 0.995 |
| running_data1 | 7 | handheld（重力方向） | −0.030 … **1.695** | 0 | 170.7 … 171.5 | 0.071 | 0.994 |
| slow_walking_data1 | 8 | handheld（重力方向） | −0.033 … **1.654** | 0 | 171.1 … 171.5 | 0.040 | 0.974 |
| trolley_data1 | 7 | trolley（场景目录） | −0.035 … −0.012 | −12 … 0 | 172.7 … 175.7 | 0.042 | 0.998 |
| trolley_data2 | 6 | trolley（场景目录） | 0.021 … 0.041 | 0 | 170.3 … 178.9 | 0.048 | 0.995 |

观察：

- 外参约为绕机体 z 轴 170–180°，外加几度倾斜，且**同一会话内高度一致**（例如 handheld_data1 为 168.6–169.6°，data3 为 176.3–177.2°），与“每次采集重新粘贴标记点”的情形吻合。
- running 第 3–7 条和 slow walking 第 5 条的时钟偏移约为 1.65–1.70 s（2018-06-11 采集，同日的 slow walking 第 6–8 条又恢复到约 0），应该是手机时钟被重新同步所致。running 第 3–7 条的粗同步相关系数为 0.86–0.93；slow walking 第 5 条只有 0.43（慢走时角速度变化小），但三轴细化相关系数为 0.63，且陀螺与航向检查都通过（10 s 窗口误差 0.7°），估计可信。
- 明显的时钟速率差（约 −480 ppm，全序列累计 57–147 ms）出现在 `multi_users_user3_seq1/seq2` 和 `multi_devices_iphone5_seq1`，线性拟合残差约 1 ms。若不建模速率差，前两条的手眼残差为 0.375/0.315 rad/s，会被拒收；建模后降为 0.079/0.065。
- trolley 几乎只有竖直轴转动，仅凭角速度估计外参时，绕该轴的分量理论上可观性较弱。航向一致性检查（第 7 节）显示 trolley 的 |δψ| ≤ 5.8°，说明估计结果可用。

## 4. 真值性质与精度

- Vicon 光学动捕（此类系统的典型精度为毫米级；数据包未给出具体数值）。位置是 Vicon 刚体原点的位置，与手机 IMU 之间的杆臂未知且被忽略（在 0.2 s 尺度上会带来可见的航向对齐偏差，见第 7 节）。
- 参考姿态 = Vicon 姿态 ⊗ 估计外参。外参的估计精度可以从手眼残差（中位 0.058 rad/s）和航向一致性（|δψ| 中位 2.2°，最大 8.5°）来判断。
- 缺测或野值处 `pose_valid = False`，统一流水线不应跨这些区间插值。

## 5. 元信息映射

| 属性 | 取值 |
|---|---|
| `subject_id` | 6 个放置场景：`user1`（依据：multi users 目录为 user2–5，推断主采集者为 user1）；multi users：`user2`…`user5`；multi devices：`unknown` |
| `group_id` | **会话**：`<场景>_<会话>`，如 `handheld_data1`、`multi_users_user3`、`multi_devices_iphone5`（共 20 组）。主场景的所有序列都来自同一受试者，按受试者分组就无法切出 val，所以改用采集会话（同日、同一次标记点粘贴） |
| `device_id` | 主场景与 multi users：`iphone7plus`（依据 OxIOD 论文的描述，数据中未记录）；multi devices：`iphone5`、`iphone6`、`nexus5` |
| `placement` | 见下表，来源记录在 `oxiod_placement_source` 中 |
| `position_source` / `orientation_source` | `vicon` / `vicon_with_estimated_body_extrinsic` |
| `device_orientation_source` | iPhone：`ios_coremotion_attitude`；Nexus 5：`none` |
| 附加属性 | `oxiod_scene`、`oxiod_session`、`oxiod_placement_source`、`oxiod_clock_offset_s`、`oxiod_clock_skew_ppm`、`oxiod_extrinsic_wxyz` |

放置方式的来源：

| 来源 | 适用 | 规则 |
|---|---|---|
| `scene_folder` | handheld / pocket / handbag / trolley | 场景目录名（handbag → `bag`） |
| `gravity_direction` | slow walking、running、multi users 的 user2/user4、multi devices | 无文档。只有机体系重力方向显示“屏幕朝上”（加速度 z > 0.5 g 的时间占比 ≈ 100%）时才标 `handheld`；其余序列手机直立（口袋和包无法可靠区分），标 `unknown` |
| `readme` | multi users 的 user3、user5 | `syn/Readme.txt` 原文：user3 “1-2 handheld, 3-5 pocket, 6-7 handbag”；user5 “1-3 handheld, 4-6 pocket, 7-11 handbag”；已核对与重力方向证据一致（手持序列屏幕朝上比例 ≥ 0.99，其余 ≤ 0.07） |

重力方向证据（机体系平均比力方向 / 屏幕朝上比例）：主场景中，handheld 约为 `[0.0, 0.4, 0.9]` / 1.00，pocket 约为 `[0.22, 0.97, −0.05]` / 0.02，handbag 约为 `[0.11, 0.99, −0.10]` / 0.05，trolley 为 `[0.00, 0.06, 1.00]` / 1.00。slow walking、running 和 nexus5 全部为 `[0.0, 0.3–0.4, 0.9–0.96]` / 1.00（屏幕朝上）。multi users 的 user2/4/5 与 multi devices 的 iPhone 5/6 均为第 1–3 条屏幕朝上，其余直立。

## 6. 官方划分

来自 6 个场景目录中的 `Train.txt` / `Test.txt`（条目为 `dataN` 或 `dataN/imuK.csv`）：

| 划分 | 条数 | 其中接收 |
|---|---:|---:|
| train | 62 | 61（`handbag_data2_seq3` 被拒） |
| test | 9 | 9 |

test = `handheld_data5_seq1–4`、`handbag_data2_seq4`、`pocket_data2_seq6`、`running_data1_seq7`、`slow_walking_data1_seq8`、`trolley_data2_seq6`。

- 无官方 val。主场景的 train 按会话划组后，可由统一流水线抽取 val（handheld 的 test 恰好是整个会话 data5，其余场景的 test 与 train 同属一个会话）。
- **multi users（31 条，接收 30 条）与 multi devices（26 条）不在任何官方列表中**。`official_splits()` 不返回它们，
  `extra_splits()`（`oxiod@1.1` 起）把它们声明为附加测试子集，统一流水线原样写出，**绝不并入 train**（DESIGN 2.4）：
  - `test_unseen_subject`（multi users，30 条）：受试者 user2–user5 从未出现在官方 train/test 中（主场景都是 user1），
    每人包含手持、口袋、包等放置方式；
  - `test_unseen_device`（multi devices，26 条）：iPhone 5、iPhone 6、Nexus 5 三种设备（受试者未知），
    与主场景的 iPhone 7 Plus 不同；其中 Nexus 5 是 Android 格式、没有设备姿态；
  - 它们的 `group_id`（`multi_users_userN`、`multi_devices_<设备>`）与官方 train/val/test 的会话不重叠。

## 7. 物理自检（全量）

检查方法与阈值见 [RoNIN 卡片第 7 节](ronin.md#7-物理自检全量)。注意：参考姿态已经应用了估计出的外参与时钟，所以下表中的“残余旋转”和“时间偏移”基本为 0，只是自洽性检验；外参与时钟的实际估计值见第 3 节。

**全部接收序列**（126 条）

| 指标 | 最小 | P5 | 中位数 | P95 | 最大 |
|---|---:|---:|---:|---:|---:|
| 有效时长 (s) | 135.8 | 176.7 | 354.3 | 622.6 | 691.1 |
| 路径长度 (m) | 98.6 | 126.9 | 290.0 | 519.0 | 736.6 |
| IMU 采样率 (Hz) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 位姿采样率 (Hz) | 100.0 | 100.0 | 100.0 | 200.0 | 200.0 |
| 重力误差 (m/s²) | 0.066 | 0.071 | 0.123 | 0.173 | 0.201 |
| 重力水平分量 (m/s²) | 0.019 | 0.023 | 0.118 | 0.134 | 0.142 |
| 运动秒水平速度中位数 (m/s) | 0.56 | 0.60 | 0.83 | 1.11 | 1.30 |
| 静止比例 | 0.01 | 0.02 | 0.06 | 0.15 | 0.40 |
| 10 s 窗口姿态误差中位数 (°，判定用) | 0.63 | 0.72 | 1.12 | 2.63 | 3.69 |
| 同上，不去零偏 (°) | 0.63 | 0.73 | 2.59 | 10.81 | 16.51 |
| 10 s 窗口姿态误差 P90 (°) | 1.16 | 1.29 | 2.20 | 4.39 | 8.37 |
| 常值陀螺零偏估计 (rad/s) | 0.0007 | 0.0014 | 0.0060 | 0.0183 | 0.0292 |
| 陀螺/参考姿态残余旋转 (°) | 0.00 | 0.00 | 0.00 | 0.26 | 1.59 |
| 陀螺/参考姿态时间偏移 (s) | −0.0034 | −0.0016 | 0.0001 | 0.0103 | 0.0519 |
| 航向偏差 \|δψ\| (°) | 0.12 | 0.56 | 2.23 | 7.18 | 8.45 |
| 航向对齐相关系数 | 0.81 | 0.89 | 0.95 | 0.99 | 0.99 |
| 位姿有效比例 | 0.950 | 0.970 | 0.992 | 1.000 | 1.000 |

合计有效时长 13.60 h，路径 37.8 km。判定所用零偏模型：60 s 分段 86 条，常值 21 条，不去零偏 19 条。

按会话：

| group_id | 条数 | 时长 (h) | 重力误差 中位/最大 | 窗口误差° 中位/最大 | 航向偏差° 中位/最大 | 速度中位 (m/s) |
|---|---:|---:|---:|---:|---:|---:|
| handbag_data1 | 4 | 0.65 | 0.12 / 0.13 | 1.35 / 1.49 | 1.7 / 4.4 | 0.83 |
| handbag_data2 | 3 | 0.46 | 0.12 / 0.12 | 1.19 / 1.53 | 1.3 / 2.1 | 0.92 |
| handheld_data1 | 7 | 0.50 | 0.13 / 0.13 | 1.02 / 1.49 | 2.9 / 3.5 | 0.79 |
| handheld_data2 | 3 | 0.26 | 0.13 / 0.13 | 0.94 / 1.00 | 4.2 / 4.4 | 0.87 |
| handheld_data3 | 5 | 0.62 | 0.13 / 0.13 | 0.99 / 1.28 | 6.6 / 7.5 | 0.86 |
| handheld_data4 | 5 | 0.56 | 0.13 / 0.13 | 1.01 / 1.39 | 4.8 / 6.1 | 0.81 |
| handheld_data5 | 4 | 0.51 | 0.13 / 0.13 | 0.93 / 1.08 | 4.1 / 4.4 | 0.81 |
| multi_devices_iphone5 | 9 | 0.45 | 0.15 / 0.18 | 2.66 / 3.69 | 2.8 / 4.5 | 0.83 |
| multi_devices_iphone6 | 9 | 0.47 | 0.18 / 0.20 | 1.02 / 1.54 | 3.1 / 4.8 | 0.91 |
| multi_devices_nexus5 | 8 | 1.12 | 0.07 / 0.08 | 1.76 / 1.88 | 0.7 / 1.3 | 0.69 |
| multi_users_user2 | 6 | 0.61 | 0.12 / 0.13 | 1.13 / 1.89 | 1.4 / 2.1 | 0.87 |
| multi_users_user3 | 6 | 0.61 | 0.12 / 0.14 | 1.32 / 1.67 | 1.9 / 7.7 | 0.84 |
| multi_users_user4 | 9 | 0.91 | 0.12 / 0.13 | 1.31 / 1.97 | 0.8 / 1.7 | 0.94 |
| multi_users_user5 | 9 | 0.83 | 0.12 / 0.14 | 1.36 / 1.66 | 3.4 / 6.7 | 0.89 |
| pocket_data1 | 5 | 0.59 | 0.11 / 0.12 | 1.91 / 2.53 | 7.2 / 8.5 | 0.83 |
| pocket_data2 | 6 | 1.03 | 0.12 / 0.13 | 2.26 / 2.87 | 6.9 / 7.8 | 0.75 |
| running_data1 | 7 | 1.04 | 0.07 / 0.08 | 1.00 / 1.04 | 2.3 / 2.7 | 1.19 |
| slow_walking_data1 | 8 | 1.18 | 0.07 / 0.08 | 0.69 / 0.75 | 2.2 / 3.4 | 0.58 |
| trolley_data1 | 7 | 0.88 | 0.13 / 0.13 | 0.82 / 0.93 | 1.0 / 1.6 | 0.64 |
| trolley_data2 | 6 | 0.32 | 0.14 / 0.14 | 0.95 / 1.00 | 3.6 / 5.8 | 0.74 |

解读：

- 重力误差 ≤ 0.20，比力符号、Vicon 世界系 z 向上、外参三者一致；世界系重力 z 分量为 9.65–9.89，说明 CoreMotion 的 “G” 与 9.80665 m/s² 的差异在 1.6% 以内。
- CoreMotion 的 rotation rate 虽然已经去过零偏，但仍有缓慢漂移的残余零偏（常值估计中位 0.006 rad/s）。所以不去零偏时，10 s 窗口误差的 P95 达 10.8°，采用 60 s 分段零偏模型后降到 2.6°（这是传感器质量问题，不是约定问题；`pocket_data2_seq1` 在常值零偏模型下为 7.8°，分段模型下通过）。
- running 序列的“运动秒速度中位数”只有约 1.2 m/s（动捕场地内跑动、频繁转弯），在 0.3–2.0 m/s 的行人范围内。
- pocket 会话的航向偏差在 0.2 s 尺度下约 8°，在 1 s 尺度下 ≤ 3°，主要来自腿部摆动时的杆臂效应；自检取两个尺度中相关系数较高者。

## 8. 被拒序列

| sequence_id | 官方划分 | 原因 |
|---|---|---|
| `handbag_data2_seq3` | train | `vi3.csv` 的时间戳被导出成科学计数法（`1.49684E+18`，分辨率只剩 10⁴ s），`imu3.csv` 的时间戳也被取整到整秒，亚秒信息丢失，无法同步 |
| `multi_users_user5_seq5` | — | 同上（`vi5.csv` 为 `1.49753E+18`，`imu5.csv` 取整到整秒） |

## 9. 已知问题与注意事项

- 外参与时钟是估计值，不是官方标定。估计值和残差都记录在 `notes` 与附加属性中；如需复核，可按会话比较外参的一致性（第 3 节）。
- 位置是 Vicon 刚体原点的位置，与 IMU 之间的杆臂未知。
- iPhone IMU 时间戳分辨率为 0.01 s（与 100 Hz 采样同量级），统一流水线应按“先均匀化再重采样”的路径处理（100 Hz → 200 Hz）。
- running / slow walking 的 Vicon 为 200 Hz，其余为 100 Hz。
- `subject_id = user1` 与 `device_id = iphone7plus` 是依据目录命名和论文描述得出的推断，数据文件中没有记录。
- multi users / multi devices 没有官方划分；large scale 与 test 目录暂未转换。

## 10. 复现

```bash
PYTHONPATH=src python -m pytest tests/converters/test_oxiod.py tests/data/test_raw_oxiod.py
```


## 11. 转换结果（IPB v1）

输出：`/workspace/webCodex/datasets/ipb/oxiod`｜转换器 `oxiod@1.1`｜转换时间 2026-09-17｜
指纹 `559c7c3de0078876…`

接收 126 / 拒收 2（第 8 节的时间戳损坏），共 13.60 h、36.90 km、20 个采集会话。官方无 val，流水线从 `train`
按会话抽出 7 条（`slow_walking_data1`，占 train 时长 14.5%）；multi users / multi devices 按第 6 节写成附加测试子集，
**没有并入 train**。

| 划分 | 序列 | 时长 (h) | 距离 (km) | 会话 |
|---|---:|---:|---:|---:|
| `train` | 54 | 6.47 | 18.17 | 11 |
| `val` | 7 | 1.10 | 2.13 | 1 |
| `test` | 9 | 1.03 | 2.89 | 6 |
| `test_unseen_subject`（multi users） | 30 | 2.96 | 8.56 | 4 |
| `test_unseen_device`（multi devices） | 26 | 2.04 | 5.16 | 3 |

`ipb check data=oxiod full=true hash=true`：**通过**（0 错误，3 警告）：`test/train` 4 组、`test/val` 1 组
会话重叠（官方 test 与 train 同属一个会话），以及 `pocket_data2_seq2` 的“重力检查勉强通过”（倾角 2.57°）。
附加子集的会话与 train/val 不相交。无重复内容、无未分配序列。

Vicon 野值掩码使 `valid` 比例最低为 93.0%（`multi_users_user5_seq7`）；5 条序列的末尾恰好是野值区加一段
短于 1 s 的有效碎片，按 DESIGN 2.3 第 6 条被裁掉：`multi_devices_iphone5_seq9`（2.54 s）、
`multi_users_user5_seq8`（2.44 s）、`multi_users_user4_seq7`（1.29 s）、`multi_users_user5_seq3`（0.87 s）、
`multi_users_user3_seq4`（0.87 s）。

### 一致性抽检（5 条随机序列）

> 抽检方法：用固定种子（`numpy.random.default_rng(0)`）从 `dataset.json` 的序列中随机抽 5 条，
> 比较**转换器解析结果**（原生时钟、未重采样，已含该转换器声明的单位/坐标系/外参/时钟修正）与
> **200 Hz 输出**：路径长度按 METRICS 的 1 s 分辨率折线计算（原始侧取每个刻度最近的原生样本，
> 不插值、不跨缺口），窗口取 200 Hz 结果实际覆盖的时间区间；重力对齐加速度均值为
> `mean(R(q) · f)`（原始侧用 SLERP 把参考姿态插到 IMU 时刻）；位置 RMS 是把 200 Hz 结果插回
> 原生位姿时刻后的水平误差。


| 序列 | 时长 (s) | 路径长度 原始 → 200 Hz (m) | Δ | 平均速度 (m/s) | 重力对齐加速度均值 (m/s²) | Δ\|acc\| | 位置 RMS (cm) |
|---|---:|---|---:|---:|---|---:|---:|
| `multi_devices_iphone5_seq6` | 178.89 | 127.25 → 127.26 | +0.005% | 0.711 | [−0.097, 0.008, 9.683] | 0.0038 | 0.01 |
| `multi_devices_nexus5_seq7` | 186.36 | 128.18 → 128.18 | −0.000% | 0.688 | [0.004, −0.024, 9.870] | 0.0001 | 0.00 |
| `pocket_data2_seq6` | 637.90 | 466.53 → 466.53 | −0.000% | 0.731 | [−0.119, −0.005, 9.818] | 0.0004 | 0.01 |
| `trolley_data1_seq2` | 309.00 | 167.16 → 167.17 | +0.007% | 0.541 | [−0.129, 0.005, 9.773] | 0.0000 | 0.01 |
| `trolley_data2_seq4` | 163.13 | 111.45 → 111.45 | +0.000% | 0.683 | [−0.130, −0.019, 9.780] | 0.0001 | 0.01 |

100 Hz → 200 Hz 的多相上采样后路径长度差异 ≤ 0.007%，时长与样本数完全一致（Vicon 时钟经逐序列的
偏移/速率差映射到手机时钟后，网格取两者重叠区）；加速度均值的水平分量最大 0.13 m/s²（约 0.8°），
与第 7 节的外参残差一致。
