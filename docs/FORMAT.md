# 统一数据规范 / Unified Data Specification

> 状态：草案 v0.1  
> 适用范围：RoNIN、RIDI、OxIOD、IMUNet_dataset、TLIO、RNIN-VIO/SenseINS  
> 设计原则：规范数据事实，模型窗口与训练目标由实验配置生成。

[中文](#中文) | [English](#english)

---

## 中文

### 1. 两层接口

本项目将数据接口分为两层，避免把某个模型的输入形式固化为数据格式：

1. **规范序列（canonical sequence）**：一条完整轨迹，保存 IMU、姿态、位置和来源信息；
2. **任务视图（task view）**：从规范序列生成滑窗、速度或位移目标以及训练增强。

数据集适配器只负责“原始格式 → 规范序列”。模型代码只依赖任务视图，不读取数据集专用字段。

### 2. 文件与目录

推荐每条轨迹保存为一个 HDF5 文件：

```text
<dataset>/
├── sequences/
│   ├── <sequence_id>.h5
│   └── ...
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

划分文件每行一个 `sequence_id`。优先保留数据集官方划分；若无官方划分，必须记录生成方法、随机种子和分组单位，且同一受试者或同一采集场景不得无意跨集合泄漏。

### 3. HDF5 数据契约

所有数值数组沿第 0 维按时间排列。规范序列默认将姿态和位置对齐到 IMU 时间戳，同时保留来源说明。

| 路径 | 形状 | 类型 | 单位/顺序 | 必需 | 含义 |
|---|---:|---|---|---:|---|
| `timestamp` | `(N,)` | float64 | s | 是 | 严格递增，相对或 Unix 时间均可，类型写入属性 |
| `imu/gyroscope` | `(N,3)` | float32/64 | rad/s，xyz | 是 | 机体系角速度 |
| `imu/accelerometer` | `(N,3)` | float32/64 | m/s²，xyz | 是 | 机体系比力；不得静默去除重力 |
| `pose/orientation` | `(N,4)` | float32/64 | wxyz | 是 | 机体系到世界系旋转，单位四元数 |
| `pose/position` | `(N,3)` | float32/64 | m，xyz | 是 | 世界系位置 |
| `pose/velocity` | `(N,3)` | float32/64 | m/s，xyz | 否 | 来源可靠时保存；否则由位置派生 |
| `valid/imu` | `(N,)` | bool | — | 否 | IMU 样本有效性 |
| `valid/orientation` | `(N,)` | bool | — | 否 | 姿态有效性 |
| `valid/position` | `(N,)` | bool | — | 否 | 位置真值有效性 |

根属性至少包含：

| 属性 | 示例 | 说明 |
|---|---|---|
| `schema_version` | `0.1` | 本规范版本 |
| `dataset` | `RoNIN` | 规范数据集名称 |
| `sequence_id` | — | 数据集内稳定标识 |
| `world_frame` | `gravity_aligned_local` | 世界坐标系定义 |
| `timestamp_type` | `relative` | `relative` 或 `unix` |
| `orientation_convention` | `body_to_world_wxyz` | 禁止省略旋转方向与元素顺序 |
| `accelerometer_type` | `specific_force` | 明确加速度语义 |
| `position_source` | `Vicon` / `VIO` / `Tango` | 真值或参考轨迹来源 |
| `orientation_source` | — | 姿态来源 |
| `subject_id` | — | 可匿名；用于防止数据泄漏 |
| `device_id` | — | 设备或型号；未知时写 `unknown` |
| `source_license` | — | 原数据许可或条款标识 |

允许增加数据集特有属性，但通用加载器不得依赖它们。

### 4. 预处理规则

适配器必须按以下顺序处理，并生成可复现日志：

1. 读取原始时间、IMU、姿态、位置及校准量；
2. 删除重复或倒序时间戳，标记缺失和异常样本；
3. 应用数据集公开定义的零偏、比例因子和轴映射；
4. 将单位统一为秒、rad/s、m/s²、米；
5. 将四元数统一为 `wxyz` 和 `body_to_world`；
6. 在需要时将姿态、位置插值到 IMU 时间戳；
7. 验证四元数范数、时间单调性、数组长度和有限数值；
8. 写入规范文件、来源属性和转换报告。

原则：

- 规范文件尽量保留原始采样率；重采样属于任务配置。
- 规范加速度保存机体系比力。去重力、旋转到世界系等属于派生操作。
- 四元数插值应使用符号连续处理后的球面插值；不得默认逐元素线性插值。
- 由位置求速度时使用真实时间差，并记录差分或平滑方法。
- Vicon、VIO、Tango、ARCore 等来源必须分别标注，不统称为同精度 ground truth。
- 不重新分发原数据；转换脚本要求用户自行按原许可获取数据。

### 5. 六个首批适配器

| 适配器 | 主要输入 | 必须处理的差异 |
|---|---|---|
| RoNIN | HDF5 + JSON 元数据 | IMU 校准、有效起始段、姿态源和初始坐标对齐 |
| RIDI | CSV/Pickle | 纳秒时间、Tango/rotation-vector 姿态对齐、参考位置来源 |
| OxIOD | CSV | 已处理加速度语义、设备/携带方式、逐序列真值来源 |
| IMUNet | CSV/Pickle | 字段映射、手机传感器单位、ARCore/Tango 参考来源 |
| TLIO | processed IMU/pose | 原始四元数顺序和旋转方向、全局/机体系语义、官方划分 |
| RNIN-VIO | SenseINS CSV | 非均匀时间、IMU bias、Vicon 与 BVIO/VIO 来源区分、可靠区间 |

表中只描述公开实现所需的通用转换职责；具体字段映射将以各数据集公开文档为准独立实现。

### 6. 任务视图

默认惯性速度回归视图可产生：

- `features: (6, T)`：`[gyro_xyz, accelerometer_xyz]`；
- `target_velocity: (3,)`：窗口端点位移除以真实时间间隔；
- `sequence_id`、`start_index`、`end_index`；
- 可选姿态、起终点位置和有效掩码，用于评测而非默认网络输入。

`T`、步长、采样率、坐标系、是否去重力和 2D/3D 输出均属于实验配置。为了复现已有工作，可提供 200 帧、100 Hz 或 200 Hz 等预设，但它们不是规范格式的硬约束。

### 7. 最低验证要求

每条转换轨迹必须通过：

- 时间戳严格递增且全部有限；
- 所有必需数组长度均为 `N`；
- IMU 和位置没有未声明的 NaN/Inf；
- 有效姿态四元数归一化误差小于配置阈值；
- 静止片段和运动片段量级检查；
- 从规范数据重建的轨迹与转换前参考轨迹在容差内一致；
- 相同输入和配置生成一致的结构、划分及校验摘要。

---

## English

The benchmark uses two layers:

1. a **canonical sequence**, which stores one complete trajectory and its provenance;
2. a **task view**, which derives windows, targets, augmentation, and model-specific coordinates.

Dataset adapters convert source data into the canonical HDF5 contract above. Model code consumes task views and must not depend on dataset-specific fields.

The canonical representation uses seconds, rad/s, m/s², metres, `wxyz` quaternions, and an explicit `body_to_world` convention. IMU data remain in the body frame and accelerometer samples retain their declared specific-force semantics. Resampling, gravity removal, global-frame rotation, window length, stride, and 2D/3D targets are experiment settings rather than storage requirements.

Official dataset splits take priority. Derived splits must record their method and random seed and must prevent subject or scene leakage. Reference sources such as Vicon, VIO, Tango, and ARCore remain explicitly distinguished.

The first adapter set covers RoNIN, RIDI, OxIOD, IMUNet_dataset, TLIO, and RNIN-VIO/SenseINS. Conversion code will be independently implemented from public dataset documentation. This public repository does not redistribute source datasets or private reference material.
