# Inertial Positioning Lab

一款重新设计的 Android 惯性定位实验工具：采集可训练的规范数据，导入受约束的 TFLite 模型，并在真实设备上统一评测精度、延迟、功耗和资源占用。

本项目沿用原版 IMUNet 软件以 ARCore SLAM/VIO 采集手机端参考轨迹的方案，并将旧版 Java + TensorFlow nightly 多 Activity 应用重写为 Kotlin、Jetpack Compose、Material 3、ARCore 和稳定版 LiteRT 架构。数据和模型逻辑不依赖界面。

## 在 Benchmark 中的角色

Inertial Positioning Lab 是 [`inertial-positioning-benchmark`](https://github.com/BUG423/inertial-positioning-benchmark) 的 Android 端采集与真机评测工具，不是另一套相互独立的数据规范。它负责产生带来源信息的 `.iplab` 无损归档、在手机上执行 LiteRT 模型，并将采集结果转换为 benchmark 的 canonical HDF5 v0.1：

```text
Android IMU + ARCore VIO
    -> .iplab 无损归档
    -> tools/convert_dataset.py
    -> benchmark canonical HDF5
    -> 训练、离线评测与跨模型比较
```

在 benchmark 单仓库中，本工具位于 `tools/inertial-positioning-lab/`；本仓库仍可独立构建和开发。两种入口使用相同的数据与模型契约。

## 安装

可从 benchmark 的 [Inertial Positioning Lab 1.1.0 Release](https://github.com/BUG423/inertial-positioning-benchmark/releases/tag/inertial-lab-v1.1.0) 下载 Android 预览安装包。该 APK 使用 Android debug 签名，适合真机功能验证、数据采集与实验，不作为应用商店生产签名版本。

## 能力

- 规范目标固定为 200 Hz；若加速度计与陀螺仪的共同硬件上限不足，自动使用可达的最高频率并提示后续重采样；
- 可选 ARCore 手机端 VIO 参考轨迹，保存 Android sensor pose 的局部位置、姿态与差分速度；
- 采集页在轨迹画布内实时显示加速度/角速度 XYZ，以及 `X 右 / Y 前 / Z 上` 局部俯视轨迹、起点、当前位置、轨迹长度和跟踪状态；界面轨迹会抑制静止漂移与跟踪跳点，归档仍保存未经显示滤波的原始 VIO 位姿；
- 采集实时安全落盘；默认保存在应用内，也可指定持久化目录并在停止后自动复制 `.iplab`；
- Android 与 Python 两端一致校验有限数值、二进制有效掩码、单位四元数、清单字段和归档结构；
- `.iplab` 一键转换为 [`inertial-positioning-benchmark`](https://github.com/BUG423/inertial-positioning-benchmark) 的 canonical HDF5 v0.1；
- 一次导入多个满足 [`.iplmodel` 固定契约](docs/MODEL_FORMAT.md)的 LiteRT/TFLite 模型，可切换、多选和批量评测；
- 数据包支持单选、多选与批量导入；离线模式按“模型 × 序列”运行全部组合；
- 实时联测模式一边保存 IMU/ARCore，一边运行所选模型，叠加预测轨迹与 VIO 参考并显示端点误差和延迟；
- 真机离线报告速度 RMSE、ATE、漂移、延迟分位数、吞吐、CPU、内存、热状态与可用的电量计数据；
- ARCore 作为可选能力；不支持 ARCore 的设备仍可记录纯 IMU，不申请旧式外部存储或定位权限。

## 使用流程

1. 设置页集中显示 200 Hz/硬件回退、VIO 参考与保存目录；采集页只保留实时轨迹、画布内 IMU 数据和底部开始按钮。
2. 在总览页导出 `.iplab`，或在模型页批量导入已有数据包。训练侧按[数据格式说明](docs/DATA_FORMAT.md)转换为 HDF5。
3. 用 `tools/pack_model.py` 将模型和清单打成 `.iplmodel`，在“模型测试”页单选或多选导入。
4. 选择“实时联测”进行边采边测，或选择“数据包测试”运行多模型、多序列批量评测。

## Android 构建

要求 JDK 17、Android SDK 36。Gradle Wrapper 会下载固定的 Gradle 8.13：

```bash
./gradlew test lint assembleDebug
```

Debug APK 位于 `app/build/outputs/apk/debug/app-debug.apk`。

预览安装包使用 debug 签名，仅用于测试与采集；正式分发版本应使用项目维护者持有的 release keystore 签名。Android 8.0 及以上系统安装侧载 APK 时，需要为实际打开 APK 的文件管理器或浏览器单独授予“安装未知应用”权限。

## 数据工具

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
pytest -q
python tools/convert_dataset.py recording.iplab -o dataset
```

详细契约见 [数据归档规范](docs/DATA_FORMAT.md)与[模型规范](docs/MODEL_FORMAT.md)。

## 重要边界

- 与原版 IMUNet 软件一致，本项目沿用 ARCore 输出作为手机端 VIO 参考；它融合相机和 IMU，与待评测 IMU 并非完全独立，不等同于 Vicon/RTK 级独立真值。
- ARCore 可能在跟踪恢复或地图修正时调整世界坐标；应用用原点 Anchor 降低坐标修正对局部轨迹的影响，并通过有效掩码排除暂停跟踪的样本。
- 中国大陆版 Android 通常需要自行安装 `Google Play Services for AR`（包名 `com.google.ar.core`，应用商店也可能显示“Google AR 服务”或“ARCore”）；硬件支持不代表该运行时已经安装。采集中 ARCore 暂停不会停止 IMU 落盘，应用会等待跟踪恢复、必要时连续重建原点或自动重连会话。
- Android 的能量、电荷和电流计数器取决于设备实现；报告不会伪造缺失数值。
- 采样率不同会显著改变模型输入分布，因此应用拒绝静默重采样。
- ARCore 的联网兼容性结果只作为安装流程提示：`UNKNOWN_ERROR`、`UNKNOWN_TIMED_OUT` 甚至 `UNSUPPORTED_DEVICE_NOT_CAPABLE` 都不会直接阻止采集。应用会实际创建本机 `Session` 做最终校验，以兼容中国 ROM、离线查询或侧载 AR 服务导致的错误判定；只有本地运行时也失败时才停止，并区分未安装、版本过旧和真实不兼容。
- 数据拆分仍需按 subject/scene 分组，避免训练与测试泄漏。

## 隐私

所有数据都先保存在应用私有目录；用户配置自定义目录后，应用在停止采集时自动复制 `.iplab`，即使自定义目录临时不可用也不会丢失原始采集。启用 VIO 参考时相机会由 ARCore 在内部持续使用，应用不读取、不展示也不导出相机图像，数据包只保存 IMU 与原始 ARCore 位姿。
