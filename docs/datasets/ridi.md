# RIDI 数据集卡片

> 转换器：[`converters/ridi.py`](../../src/inertial_benchmark/data/converters/ridi.py)，`ridi@1.0`；
> 共享工具与物理自检：[`converters/_phone_utils.py`](../../src/inertial_benchmark/data/converters/_phone_utils.py)。
> 统计基于本机 `/workspace/webCodex/datasets/imu_odometry/raw/_staging/RIDI/data_publish_v2`（94 条），2026-09-17。

## 1. 概览

| 项 | 内容 |
|---|---|
| 论文 | Yan, Shan, Furukawa. *RIDI: Robust IMU Double Integration*, ECCV 2018 |
| 数据 | `ridi_data_publish_v2.zip`（官方仓库 README 中的 Dropbox 链接） |
| 官方代码 | <https://github.com/higerra/ridi_imu>（提交 `ce5d772`，`python/gen_dataset.py`）；采集 App <https://github.com/higerra/TangoIMURecorder>（提交 `ca8657b`） |
| 许可 | 数据包与仓库均未给出数据许可条款；按学术使用并引用论文处理，不再分发 |
| 采集方式 | **单台 Tango 手机**同时记录 Android IMU 与 Tango VIO 位姿；4 种放置方式：handheld、leg（腿部口袋）、bag、body |
| 输入 `source` | `data_publish_v2` 目录或其父目录 |

## 2. 原始文件与字段

每条序列一个目录，官方预处理结果在 `processed/data.csv`（pandas `to_csv` 写出，首列是空名索引）；目录中还有原始 `acce.txt`、`gyro.txt`、`pose.txt`、`orientation.txt` 等。转换器使用 `data.csv`：

| 列 | 单位/含义 |
|---|---|
| `time` | 纳秒，Tango 位姿时间戳（Android 开机时钟），约 202.3 Hz |
| `gyro_x/y/z` | rad/s，`TYPE_GYROSCOPE`（Android 已在线补偿零偏），已线性插值到位姿时间 |
| `acce_x/y/z` | m/s²，`TYPE_ACCELEROMETER`，含重力的比力，已插值 |
| `linacce_*`、`grav_*`、`magnet_*` | Android 线加速度、重力、磁场（未使用） |
| `pos_x/y/z` | m，Tango `START_OF_SERVICE`（重力对齐、z 向上）中的设备位置 |
| `ori_w/x/y/z` | Tango `START_OF_SERVICE → DEVICE` 的姿态；`pose.txt` 为 xyzw，官方已换成 **wxyz** |
| `rv_w/x/y/z` | `TYPE_GAME_ROTATION_VECTOR`（App 的 `ROTATION_SENSOR`）；`orientation.txt` 为 xyzw，官方已换成 wxyz 并 SLERP 到位姿时间 |

官方 `gen_dataset.py` 会去掉首尾各 800 个位姿（约 4 s），并剔除重复的位姿时间戳。

## 3. 约定与转换

| 量 | 转换 |
|---|---|
| 时间 | `time / 1e9` 秒；IMU 与位姿共用该时间轴；再剔除重复或倒序的时间戳（本地数据中没有）；`start_time_unix = NaN` |
| 陀螺 / 加速度 | 原样（rad/s、m/s² 比力）；数据集没有公布标定参数 |
| 参考姿态 | `ori` 原样（wxyz、body→world），归一化并做符号连续化 |
| 参考位置 | `pos` 原样 |
| 设备姿态 | `rv` 原样（与参考世界系差一个常值偏航） |
| 机体系 | Android 传感器系；Tango 的 `DEVICE` 系与之一致 |

**IMU 与位姿是否为同一刚体？** 是的。四种放置方式下，陀螺与 `ori` 差分角速度之间的最佳常值旋转都小于 0.6°，时间偏移小于 0.2 ms（见第 7 节），所以 `ori` 可以直接作为 IMU 机体系的参考姿态，不需要像 RoNIN 那样做起点对齐。官方 RoNIN 仓库中的 `data_ridi.py` 用 `ori[0] ⊗ rv[0]* ⊗ rv(t)`（对齐后的 game_rv）作为姿态；IPB 选用独立的 Tango VIO 姿态作为参考，game_rv 则作为设备姿态保存。

## 4. 真值性质与精度

- 位置与姿态都来自同一台 Tango 手机的 VIO（未做 area learning），属于“同设备 VIO”，精度约为厘米级、低漂移，**不是**动捕真值。
- 加速度计存在约 1–3% 的比例误差：世界系重力均值的 z 分量为 9.87–10.10（中位 9.95），比标准重力大 0.07–0.30 m/s²。数据集没有提供标定，转换器不做修正，只在自检中记录。
- 航向一致性（IMU 加速度 vs VIO 位置加速度）：|δψ| 中位 1.1°，最大 11.4°，相关系数 ≥ 0.69。

## 5. 元信息映射

| 属性 | 取值 |
|---|---|
| `subject_id` / `group_id` | 目录名第一段（如 `dan`、`hang`），共 11 人 |
| `device_id` | `tango_phone`（数据中未记录型号） |
| `placement` | 优先取官方列表第二列，否则从目录名中去掉数字后的词元解析；映射为 handheld→handheld、leg→**pocket**、bag→bag、body→body。原始名称保存在 `ridi_placement` 中 |
| `position_source` / `orientation_source` | `tango_vio_same_device` |
| `device_orientation_source` / `body_frame` | `android_game_rotation_vector` / `android_device` |

各放置方式的条数：body 29、handheld 27、bag 23、leg（pocket）15。

## 6. 官方划分

| 划分 | 文件 | 官方条数 | 本地存在 | 其中接收 |
|---|---|---:|---:|---:|
| train | `list_train_publish_v2.txt` | 49 | 49 | 49 |
| test | `list_test_publish_v2.txt` | 25 | 23（缺 `hang_leg_new3`、`huayi_handheld_test1`） | 23 |

- 无官方 val。
- 22 条本地序列不在任何官方列表中，`list_sequences()` 会枚举到它们，统一流水线会把它们归入 train 候选或单独标注：
  - 11 条属于官方列表中已有的受试者：`dan_handheld2 hang_bag_normal2 hang_bag_side1 hang_bag_stop1 hang_handheld_speed2 hao_bag2 hao_body2 hao_handheld2 hao_leg2 huayi_bag2 huayi_handheld2`。其中 `dan`、`hang`、`huayi` 也出现在 test 中；由于官方划分本来就按序列而非受试者划分，把它们并入 train 不会造成序列级泄漏，但会增加受试者重叠；
  - 11 条属于**不在 train/test 中出现**的受试者：`ruixuan`（8 条）和 `shali`（3 条）。它们对应官方的 cross-subject 列表，更适合作为“未见受试者”的附加测试集，不建议并入 train。
- 数据包还带有 `list_crosssubject.txt`（ruixuan/shali 共 8 条，本地有 7 条）和 `list_submission.txt`（8 条，与 test 部分重叠）。它们与 test 不是子集关系，`official_splits()` 不返回，如有需要可自行使用。
- train 与 test 共享受试者（官方按序列划分）。

## 7. 物理自检（全量）

检查方法与阈值见 [RoNIN 卡片第 7 节](ronin.md#7-物理自检全量)。

**全部接收序列**（94 条）

| 指标 | 最小 | P5 | 中位数 | P95 | 最大 |
|---|---:|---:|---:|---:|---:|
| 有效时长 (s) | 44.1 | 50.4 | 105.4 | 164.3 | 201.7 |
| 路径长度 (m) | 43.5 | 63.5 | 111.1 | 167.9 | 192.5 |
| IMU / 位姿采样率 (Hz) | 202.3 | 202.3 | 202.3 | 202.3 | 202.3 |
| 重力误差 (m/s²) | 0.070 | 0.108 | 0.150 | 0.244 | 0.297 |
| 重力水平分量 (m/s²) | 0.002 | 0.009 | 0.023 | 0.077 | 0.138 |
| 运动秒水平速度中位数 (m/s) | 0.92 | 1.00 | 1.17 | 1.33 | 1.53 |
| 静止比例 | 0.00 | 0.01 | 0.06 | 0.13 | 0.36 |
| 10 s 窗口姿态误差中位数 (°，判定用) | 0.59 | 0.80 | 1.11 | 1.96 | 3.79 |
| 同上，不去零偏 (°) | 0.60 | 0.90 | 1.53 | 2.86 | 3.89 |
| 10 s 窗口姿态误差 P90 (°) | 1.16 | 1.39 | 2.42 | 4.33 | 7.67 |
| 常值陀螺零偏估计 (rad/s) | 0.0002 | 0.0004 | 0.0013 | 0.0045 | 0.0072 |
| 陀螺/参考姿态残余旋转 (°) | 0.08 | 0.13 | 0.30 | 0.52 | 0.59 |
| 陀螺/参考姿态时间偏移 (s) | −0.0001 | −0.0001 | 0.0000 | 0.0001 | 0.0002 |
| 航向偏差 \|δψ\| (°) | 0.01 | 0.07 | 1.06 | 7.88 | 11.38 |
| 航向对齐相关系数 | 0.69 | 0.77 | 0.92 | 0.97 | 0.97 |

合计有效时长 2.71 h，路径 10.3 km。判定所用零偏模型：60 s 分段 42 条，常值 30 条，不去零偏 22 条。

按放置方式：

| placement | 条数 | 时长 (h) | 重力误差 中位/最大 | 窗口误差° 中位/最大 | 残余旋转° 最大 | 航向偏差° 中位/最大 | 速度中位 (m/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| bag | 23 | 0.70 | 0.14 / 0.22 | 1.12 / 1.66 | 0.56 | 1.6 / 7.3 | 1.21 |
| body | 29 | 0.69 | 0.20 / 0.30 | 1.14 / 3.79 | 0.44 | 0.7 / 8.8 | 1.15 |
| handheld | 27 | 0.86 | 0.13 / 0.15 | 0.94 / 1.28 | 0.52 | 1.7 / 11.4 | 1.15 |
| pocket | 15 | 0.47 | 0.18 / 0.20 | 1.77 / 1.99 | 0.59 | 0.5 / 2.8 | 1.19 |

## 8. 被拒序列

无。

## 9. 已知问题与注意事项

- IMU 已被官方线性插值到位姿时间（约 202 Hz、等间隔性良好），因此原始 IMU 时间戳信息已经丢失。若需要原生 IMU 时钟，可改读目录中的 `acce.txt` / `gyro.txt`（加速度计与陀螺的时间戳不同，需要再插值）；v1 选择与官方处理保持一致。
- 加速度计有约 1–3% 的比例误差，且没有标定可用。
- 采样率 202.3 Hz 与 200 Hz 相差约 1.1%，在统一流水线“±3% 直接插值”的规则范围内。

## 10. 复现

```bash
PYTHONPATH=src python -m pytest tests/converters/test_ridi.py tests/data/test_raw_ridi.py
```
