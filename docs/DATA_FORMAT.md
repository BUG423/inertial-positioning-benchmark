# 数据归档与规范序列

Inertial Lab 在手机端导出 `.iplab` ZIP 归档。它是可靠传输用的无损中间层；`tools/convert_dataset.py` 将其转换为 `inertial-positioning-benchmark` 草案 v0.1 的 HDF5 规范序列。

参考轨迹源没有从原版 IMUNet 更换为其他系统：新版仍使用 ARCore。原版分别将 `Camera.getPose()`、`Frame.getAndroidSensorPose()` 和 `Camera.getDisplayOrientedPose()` 写入 `pose.txt`、`android_sensor_pose.txt` 和 `display_oriented_pose.txt`，实时轨迹使用 Android sensor pose。新版规范序列选择 Android sensor pose 作为与手机 IMU 机体对齐的主参考轨迹。

## `.iplab` 结构

```text
sequence.iplab
├── manifest.json
├── data/
│   └── sequence.csv
└── README.txt
```

CSV 每行是一个以陀螺仪时钟对齐的样本。列顺序固定：

手机端目标采样率固定为 200 Hz。应用根据加速度计和陀螺仪的 `minDelay` 计算共同硬件上限；若不足 200 Hz，`manifest.json.sample_rate_hz` 保存实际请求频率，设置页会要求训练前显式重采样，评测器不会静默修改时序。

| 组 | 字段 | 单位/约定 |
|---|---|---|
| 时间 | `timestamp` | 相对秒，严格递增 |
| IMU | `gyro_x/y/z` | rad/s，Android 机体系 |
| IMU | `accel_x/y/z` | m/s²，机体系比力，不去重力 |
| 姿态 | `orientation_w/x/y/z` | `body_to_world_wxyz` 单位四元数；启用 VIO 参考时来自 ARCore Android sensor pose |
| 位置 | `position_x/y/z` | 米，相对第一个有效 ARCore Anchor 的局部 `X 右 / Y 前 / Z 上` |
| 速度 | `velocity_x/y/z` | m/s，由相邻 ARCore 位姿差分得到 |
| 掩码 | `valid_imu/orientation/position` | `0` 或 `1` |
| 来源 | 纬度、经度、高度、水平精度 | 为外部地理参考保留；ARCore 手机采集时为空 |

`manifest.json` 包含规范要求的全部根属性：schema、数据集、序列、世界系、时间类型、姿态与加速度语义、位置/姿态来源、subject、device 和 license。ARCore 采集声明：

```json
{
  "world_frame": "arcore_local_x_right_y_forward_z_up",
  "position_source": "ARCore visual-inertial odometry",
  "orientation_source": "ARCore Android sensor pose"
}
```

ARCore 的 Android sensor pose 与 Android IMU 使用相同机体系。应用用初始位置建立 Anchor，并将 ARCore 的 `X 右 / Y 上 / Z 后` 映射为数据集的 `X 右 / Y 前 / Z 上`。只有 `TrackingState.TRACKING`、位姿距当前 IMU 样本不超过 250 ms，且已有相邻帧可计算速度时，`valid_position` 才为 `1`；ARCore 姿态不要求速度已经就绪。

采集器会等待 ARCore 连续稳定跟踪且原点在 4 cm 半径内保持约 1.2 秒后建立原点。Anchor 暂停期间 IMU 仍持续写入，参考掩码置为无效；原点长期不可跟踪时会在保持轨迹连续的前提下自动重建，ARCore 会话异常时也会自动重连。相机只由 ARCore 在内部用于 VIO，应用不读取、不展示也不写入相机帧。

采集页的显示轨迹会短时低通，并在 IMU 静止时锁定慢漂移；跟踪恢复时会连续重接，单帧不合理跳变不会绘制。上述处理只作用于界面轨迹和界面距离，`.iplab` 中的位姿、速度与有效掩码仍是未经显示滤波的 ARCore 样本，保证实验数据可审计。

Android 导入器与 Python 转换器执行相同的关键完整性检查：归档只能包含上述文件（`README.txt` 可选），解压后不得超过 512 MiB，清单必须声明格式、受支持的世界系、采样率和带时区的开始时间；所有必填数值必须有限，有效掩码只能是 `0`/`1`，时间戳必须非负且严格递增，四元数范数容差为 `1e-3`。清单中非零的样本数和时长也必须与 CSV 一致。

沿用原项目的数据角色，该轨迹是 ARCore 手机端 VIO 参考。ARCore 本身会融合相机与设备 IMU，也可能在重定位时调整世界模型；因此它不是与待评测 IMU 完全独立的 Vicon/RTK 级真值。严谨实验应在报告中保留上述来源字段。

## 转成规范 HDF5

```bash
python -m pip install -r requirements-dev.txt
python tools/convert_dataset.py walk.iplab -o datasets/my_walks
```

转换器会先校验整批输入，再写入任何 HDF5，避免批次后部的坏归档留下前半批输出。输出：

```text
datasets/my_walks/
├── sequences/walk-01.h5
├── splits/all.txt
└── conversion_report.json
```

HDF5 路径严格映射为 `timestamp`、`imu/gyroscope`、`imu/accelerometer`、`pose/orientation`、`pose/position`、`pose/velocity` 与三类 `valid/*`。转换器会拒绝倒序/重复时间戳、NaN/Inf、异常四元数、缺失属性和重复目标文件。

`all.txt` 只是未分配集合；在发布训练/验证/测试划分前，应按受试者或场景分组，避免数据泄漏。
