# Benchmark 工具 / Benchmark Tools

[中文](#中文) | [English](#english)

---

## 中文

### Inertial Positioning Lab

[`tools/inertial-positioning-lab`](../tools/inertial-positioning-lab) 是本 benchmark 的 Android 端数据采集与真机评测工具。它将移动设备上的数据入口、canonical 数据规范和端侧模型验证连接为同一条可复现流程：

```text
手机 IMU + ARCore Android sensor pose
    -> .iplab（无损传输归档）
    -> convert_dataset.py
    -> canonical HDF5 v0.1
    -> CanonicalSequence / WindowDataset
```

主要能力：

- 以 200 Hz 为规范目标采集加速度计和陀螺仪，硬件不足时明确记录实际采样率；
- 可选记录 ARCore VIO 局部位置、姿态、差分速度和逐样本有效掩码；
- 批量导入 `.iplab` 数据包与 `.iplmodel` LiteRT/TFLite 模型；
- 在 Android 真机上执行实时联测或离线“模型 × 序列”批量测试；
- 输出精度、延迟、吞吐、CPU、内存、热状态与可用能量计数据。

### 数据边界

ARCore 是融合相机与设备 IMU 的 VIO 参考源，并非独立的 Vicon/RTK 真值。采集界面会对显示轨迹执行短时平滑、静止锁定、跳点拒绝和跟踪恢复连续化；这些处理不会写回 `.iplab`。归档内保存的是原始 ARCore 位姿、来源字段和有效掩码，便于审计和重新处理。

相机只由 ARCore 内部用于 VIO，工具不展示或保存相机图像。中国大陆版 Android 设备可能需要从可信应用商店另行安装并启用 Google Play Services for AR（`com.google.ar.core`）。

### 转换到 Benchmark

从仓库根目录运行：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r tools/inertial-positioning-lab/requirements-dev.txt
python tools/inertial-positioning-lab/tools/convert_dataset.py recording.iplab -o datasets/android
```

转换结果位于 `datasets/android/sequences/`，可以直接交给 `CanonicalSequence.from_hdf5()`。转换器会校验归档路径、文件大小、清单、有限数值、时间单调性、有效掩码和四元数。

### 构建与安装

要求 JDK 17 和 Android SDK 36：

```bash
cd tools/inertial-positioning-lab
./gradlew testDebugUnitTest lintDebug assembleDebug
```

本地 APK 位于 `app/build/outputs/apk/debug/app-debug.apk`。带 `inertial-lab-v*` 的仓库标签会触发 GitHub Actions，创建 GitHub Release 并附加可安装的 debug 预览 APK。debug 签名只用于测试；生产分发应使用维护者控制的 release keystore。

### 源码同步

Android 工具同时在 [`BUG423/inertial-positioning-lab`](https://github.com/BUG423/inertial-positioning-lab) 独立开发。本仓库使用 Git subtree 保存完整源码，普通用户无需初始化 submodule。维护者同步新版时运行：

```bash
git subtree pull \
  --prefix tools/inertial-positioning-lab \
  git@github.com:BUG423/inertial-positioning-lab.git <branch-or-tag> \
  --squash
```

---

## English

### Inertial Positioning Lab

[`tools/inertial-positioning-lab`](../tools/inertial-positioning-lab) is the Android capture and on-device evaluation tool for this benchmark. It connects real-device collection, the canonical data contract, and edge-model validation in one reproducible path:

```text
Phone IMU + ARCore Android sensor pose
    -> .iplab lossless transport archive
    -> convert_dataset.py
    -> canonical HDF5 v0.1
    -> CanonicalSequence / WindowDataset
```

It targets 200 Hz IMU capture, records the actual rate when hardware cannot reach that target, optionally stores ARCore VIO pose and validity masks, imports `.iplab` datasets and `.iplmodel` LiteRT/TFLite packages, and runs live or offline model evaluation on Android.

ARCore is a camera-and-IMU-fused VIO reference, not independent Vicon/RTK ground truth. Display stabilization never modifies archived samples. The camera is used internally by ARCore and is neither displayed nor persisted by the tool.

Convert an export from the repository root with:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r tools/inertial-positioning-lab/requirements-dev.txt
python tools/inertial-positioning-lab/tools/convert_dataset.py recording.iplab -o datasets/android
```

Build the Android app with JDK 17 and Android SDK 36:

```bash
cd tools/inertial-positioning-lab
./gradlew testDebugUnitTest lintDebug assembleDebug
```

The APK is written to `app/build/outputs/apk/debug/app-debug.apk`. Tags matching `inertial-lab-v*` trigger the release workflow and publish an installable debug preview APK. Production distribution should use a maintainer-controlled release signing key.

The standalone source lives at [`BUG423/inertial-positioning-lab`](https://github.com/BUG423/inertial-positioning-lab). This repository carries a full Git subtree, so regular users do not need to initialize a submodule.
