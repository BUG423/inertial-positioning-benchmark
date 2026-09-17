# IPB v1 设计宪法 / Design Charter

> 状态：v1 开发基线（2026-09-17）。所有实现者（人或 Agent）必须遵守本文件；
> 若实现中发现本文件有错，先改本文件并说明理由，再改代码。

## 0. 第一性原理

一个惯性定位 benchmark 只需要回答一个问题：**给定同一段 IMU 数据和同一个评测协议，
算法 A 是否比算法 B 更好地恢复了真实运动？** 由此推出四条不可妥协的约束：

1. **数据事实只有一份**：单位、坐标系、时间基、真值来源在存储层一次性确定，任何模型都不得各自解释。
2. **协议与模型分离**：窗口、步长、坐标变换、目标定义属于“任务视图”，是配置，不是数据格式，也不是模型内部的隐式假设。
3. **评测只看轨迹与物理量**：所有模型输出都被转换为同一时间轴上的轨迹，再用同一套指标比较；指标定义写进代码、文档与结果文件。
4. **可复现即可审计**：每个结果都能追溯到数据版本、划分、配置、代码提交和随机种子；不偷看 test。

## 1. 总体架构（参考 Ultralytics 的组织方式）

```text
src/inertial_benchmark/
├── __init__.py            # 公共 API：NIO、load_sequence、Sequence、__version__
├── cfg/                   # 配置系统（YAML + CLI key=value 覆盖）
│   ├── __init__.py        # get_cfg / 合并 / 校验未知键
│   ├── default.yaml       # 全部超参数的唯一默认来源
│   ├── datasets/*.yaml    # 每个数据集一个：路径、划分、元信息
│   └── models/*.yaml      # 每个模型一个：结构参数、输入规格、损失、训练配方
├── data/
│   ├── format.py          # IPB v1 序列格式：Sequence 数据类、读写、校验（兼容读取 v0.1）
│   ├── resample.py        # 统一采样率（抗混叠、SLERP、缺口掩码）
│   ├── converters/        # 原始数据 → IPB v1：ronin, ridi, oxiod, tlio, idol, rnin, imunet, pedlocdata, iplab
│   ├── splits.py          # 官方划分优先；分组感知的 val 生成与泄漏检查
│   ├── views.py           # 任务视图：坐标变换、目标构造（无 torch 依赖）
│   ├── augment.py         # 数据增强（随机偏航、时间抖动、偏置/噪声注入…）
│   ├── dataset.py         # torch Dataset / 缓存 / 采样器
│   └── build.py           # build_dataset / build_dataloader
├── nn/
│   ├── modules/           # 通用积木：1D ResNet、TCN、LSTM、Transformer 块等
│   ├── heads.py           # 速度头、高斯头（协方差）、极坐标头（速度大小+方向）
│   ├── losses.py          # 通用损失：MSE、Gaussian NLL、TLIO 式调度等
│   └── registry.py        # @register_model / build_model(cfg)
├── models/                # 每个算法一个子包：model.py（结构）+ 可选 loss / trainer 钩子
│   └── ronin/ tlio/ rnin/ imunet/ ctin/ imot/ eqnio/ ...
├── engine/
│   ├── model.py           # NIO 门面：.train() .val() .predict() .benchmark() .info()
│   ├── trainer.py         # 训练循环、回调、AMP、断点、早停、fitness
│   ├── validator.py       # 窗口级损失 + 轨迹级指标
│   ├── predictor.py       # 序列 → 窗口 → 速度 → 轨迹
│   └── results.py         # Trajectory / SequenceResult / RunResult
├── metrics/               # ATE/RTE/漂移/长度/方向/速度/效率 指标
├── utils/                 # 日志、回调、torch 工具、绘图、报表、统计检验、环境检查
└── cli.py                 # 命令行入口 `ipb`
```

Python API：

```python
from inertial_benchmark import NIO
model = NIO("ronin_resnet18")                 # 模型名 / YAML / checkpoint 路径
model.train(data="ronin", epochs=40, device=0)
metrics = model.val(data="ronin", split="test")
traj = model.predict("path/to/sequence.h5")   # -> Trajectory
```

CLI（`key=value` 风格，与 Python 参数一一对应）：

```bash
ipb convert dataset=ronin source=/path/raw output=/path/ipb
ipb check   data=ronin                         # 校验数据集
ipb train   model=ronin_resnet18 data=ronin epochs=40 device=0
ipb val     model=runs/train/exp/weights/best.pt data=ronin split=test
ipb predict model=... source=seq.h5
ipb benchmark cfg=benchmarks/main.yaml         # 模型 × 数据集 × 种子 矩阵
ipb report  runs=runs/benchmark/main out=reports/main
```

运行输出目录：

```text
runs/<mode>/<name>/
├── args.yaml               # 完整解析后的配置
├── env.json                # git commit、包版本、GPU、数据集指纹
├── weights/{best,last}.pt  # 含模型结构配置、输入规格、训练状态
├── results.csv             # 每轮训练/验证记录
├── metrics.json            # 聚合指标
├── sequences.csv           # 逐序列指标
├── predictions/<seq>.npz   # 轨迹与窗口级输出
└── plots/*.png
```

## 2. 数据格式 IPB v1.0

在 `docs/FORMAT.md`（v0.1）的基础上做三项确定性升级：**统一采样率**、**显式区分参考姿态与设备姿态**、
**数据集级清单**。

### 2.1 目录

```text
<root>/<dataset>/
├── dataset.json           # 数据集级元信息：版本、采样率、来源、许可、转换器版本、划分方法、统计、文件校验和
├── sequences/<sequence_id>.h5
├── splits/{train,val,test}.txt       # 可附加 test_seen.txt / test_unseen.txt 等官方子集
└── conversion_report.json # 每条序列的接收/拒绝原因与警告
```

默认数据根目录：环境变量 `IPB_DATASETS`，缺省为 `~/datasets/ipb`。本机为 `/workspace/webCodex/datasets/ipb`。

### 2.2 HDF5 契约

| 路径 | 形状 | 类型 | 单位/约定 | 必需 |
|---|---|---|---|---|
| `timestamp` | (N,) | f8 | s，从 0 开始，**严格均匀**，`dt = 1/sample_rate_hz` | 是 |
| `imu/gyroscope` | (N,3) | f4 | rad/s，机体系 | 是 |
| `imu/accelerometer` | (N,3) | f4 | m/s²，机体系**比力（含重力）** | 是 |
| `imu/orientation` | (N,4) | f4 | 设备自身估计姿态（如 Android game rotation vector），`body_to_world_wxyz`；其世界系与参考世界系可差一个常值偏航 | 否 |
| `pose/orientation` | (N,4) | f4 | 参考姿态，`body_to_world_wxyz`，单位四元数，符号连续 | 是 |
| `pose/position` | (N,3) | f8 | m，参考世界系 | 是 |
| `pose/velocity` | (N,3) | f4 | m/s；仅当来源提供时保存 | 否 |
| `valid/imu` | (N,) | bool | IMU 样本有效 | 是 |
| `valid/pose` | (N,) | bool | 参考位姿有效（缺口、跟踪丢失处为 False） | 是 |

参考世界系：**重力对齐、z 轴向上**（静止时机体系加速度计经 `pose/orientation` 旋转后 ≈ `[0, 0, +9.81]`）。
这是校验项，转换器必须保证。

根属性（全部必需，未知写 `unknown`）：

`schema_version="1.0"`, `dataset`, `sequence_id`, `sample_rate_hz`, `source_sample_rate_hz`（实测中位数）,
`resampling`（方法描述）, `world_frame="gravity_aligned_z_up"`, `timestamp_type="relative"`,
`start_time_unix`（可为 NaN）, `orientation_convention="body_to_world_wxyz"`,
`accelerometer_type="specific_force"`, `body_frame`（例如 `android_device`）,
`position_source`, `orientation_source`, `device_orientation_source`（无则 `none`）,
`subject_id`, `device_id`, `placement`（handheld/pocket/bag/trolley/head/body/mixed/unknown）,
`group_id`（划分时的防泄漏分组键）, `source_license`, `source_files`, `converter`（`名称@版本`）。

### 2.3 统一采样率：200 Hz

决策依据：

- 8 个核心数据集里 5 个原生约 200 Hz（RoNIN、RIDI、TLIO、IMUNet、PedLocData），RNIN 为 250 Hz，只有 OxIOD、IDOL 为 100 Hz；
- 主流公开算法（RoNIN、TLIO、RNIN、IMUNet 等）都按 200 Hz 设计，其结构（例如 RoNIN ResNet 的全连接层输入长度）依赖“窗口样本数”；统一 200 Hz 让它们可以**不改结构**地忠实复现；
- 行人运动的有效频带远低于 50 Hz，100 Hz 源上采样到 200 Hz 不引入伪信息，只需要记录 `source_sample_rate_hz` 以便分析。

重采样规则（`data/resample.py`）：

1. 先剔除重复/倒序时间戳，记录数量；
2. 源采样间隔超过 `gap_threshold`（默认 0.05 s）的区段不做跨缺口插值，对应网格点 `valid=False`；
3. `|f_src/200 − 1| ≤ 3%`：直接在统一网格上插值（IMU 线性、位置线性、四元数 SLERP）；
4. 否则：先在源频率上均匀化，再用有理数多相滤波（`scipy.signal.resample_poly`，抗混叠/抗镜像）变换到 200 Hz；位置用线性插值，四元数用 SLERP；
5. 输出前做四元数符号连续化与归一化。

### 2.4 划分

- **官方划分优先**（RoNIN seen/unseen、TLIO lists、RIDI publish lists、IMUNet lists、RNIN 目录、IDOL known/unknown、OxIOD 场景 Train/Test、PedLocData train/valid/test）。
- 官方没有 val 时，从 train 中按 `group_id`（受试者/会话/楼栋）分组抽取约 10%，固定种子 `0`，方法写入 `dataset.json`。
- `ipb check` 必须报告 train/val/test 之间的 `group_id` 与 `sequence_id` 重叠。

### 2.5 v0.1 兼容

`load_sequence()` 可读取 v0.1 文件（单时钟、可能非均匀），在内存中按 2.3 规则重采样为 v1 对象；写出时永远是 v1。

## 3. 任务视图（协议层）

`data/views.py` 从 `Sequence` 生成模型输入与目标，全部由配置决定：

| 配置项 | 取值 | 默认 |
|---|---|---|
| `window` | 窗口样本数（200 Hz 下） | 200 |
| `stride` | 训练滑窗步长（样本） | 10 |
| `eval_stride` | 推理步长（样本） | 10 |
| `frame` | `gravity_world`（用姿态旋转到重力对齐世界系）/ `body` / `gravity_yaw_local`（按窗口末端偏航去除后的重力对齐系） | `gravity_world` |
| `orientation` | `reference` / `device`（输入旋转使用哪个姿态） | `reference` |
| `remove_gravity` | bool | false |
| `target` | `avg_velocity`（窗口首末位移/时长）/ `displacement` / `velocity_at_end` | `avg_velocity` |
| `dims` | 2（水平）/ 3 | 2 |
| `augment` | 列表，例如 `[random_yaw, time_shift]` | 模型配置决定 |

模型输入通道顺序**固定为 `[gyro_xyz, acc_xyz]`**，形状 `(B, 6, T)`。模型若需要其他顺序或额外通道，
必须在自己的模型代码内部完成，并有单元测试。

## 4. 模型接口

```python
class BaseModel(nn.Module):
    input_spec:  InputSpec   # window, frame, channels, dims, target, rate=200
    def forward(self, imu: Tensor) -> dict:      # {"vel": (B,D), 可选 "logstd"/"cov": (B,D[,D]), 可选 "aux": ...}
    def loss(self, out: dict, batch: dict, epoch: int) -> tuple[Tensor, dict]   # 默认 MSE；可覆盖
```

- 通过 `@register_model("name")` 注册，`cfg/models/<name>.yaml` 给出结构参数与训练配方。
- 每个公开算法必须在模型文档字符串与 `docs/algorithms/<name>.md` 中注明：论文、官方仓库、许可、提交号、
  与官方实现的差异（例如 2D 输出、单窗口输入）及理由。
- 忠实性底线：网络结构、参数量、专用损失、训练阶段切换与官方一致；参数量必须有单元测试锁定。
- 每个模型提供两套配方：`recipe: official`（论文/官方仓库超参数）与 `recipe: unified`（benchmark 统一预算）。

## 5. 推理与轨迹重建

1. 以 `eval_stride` 滑窗，得到每个窗口的速度 `v_k`，时间戳赋给**窗口中心**；
2. 在序列首尾半个窗口内保持首/末速度（常值外推），使所有模型覆盖**同一时间轴**；
3. 对分段线性速度做梯形积分得到逐帧轨迹，起点锚定参考轨迹首个有效位置；不做任何对齐；
4. 同时保存窗口级 `vel_pred / vel_target / (logstd)`，以及“oracle 轨迹”（用窗口目标本身积分），用于分离协议效应与模型效应。

旧实现的已知错误（时间戳赋给窗口起点造成半窗错位、末尾一个窗口时长不被覆盖）在此明确禁止。

## 6. 评测体系

所有指标默认在水平面（2D）计算，逐序列计算后再聚合。定义写入 `docs/METRICS.md`。

| 指标 | 定义 |
|---|---|
| ATE | `sqrt(mean_t ‖p̂_t − p_t‖²)`，不对齐（另提供 `ATE_aligned`：刚体对齐后） |
| RTE | RoNIN 定义：Δ=60 s，`sqrt(mean_t ‖(p̂_{t+Δ}−p̂_t) − (p_{t+Δ}−p_t)‖²)`；序列短于 60 s 时取全长并按 60/时长线性缩放 |
| T-RTE@τ | 同上，τ ∈ {1 s, 10 s}（可配） |
| D-RTE@d | 参考轨迹每走过 d 米（默认 10 m）的相对位移误差 RMSE |
| Drift / PDE | `‖p̂_N − p_N‖ / L_ref × 100%` |
| PLR | 路径长度比 `L(p̂)/L(p)`；L 在固定时间分辨率 1 s 上计算 |
| PLR_dense / PLR_oracle / Oracle ratio | 参考长度分别取 200 Hz 稠密轨迹、oracle 轨迹；`PLR_dense = OracleRatio × PLR_oracle` |
| Speed bias | `mean_k(‖v̂_k‖ − ‖v_k‖)`，以及比值形式 |
| Direction error | 窗口速度夹角的均值/中位数（只统计 `‖v_k‖ > 0.2 m/s`） |
| Along/Cross-track | 窗口速度误差在目标方向上的平行/垂直分量 |
| Vel RMSE | 窗口速度 RMSE |
| 效率 | 参数量、FLOPs、单窗口延迟（CPU / GPU） |

统计与报表：均值、中位数、标准差、按 `group_id` 的 bootstrap 95% 置信区间、多种子均值±标准差、
配对 Wilcoxon 检验；输出 CSV / Markdown / LaTeX 表格；图包括轨迹叠加、误差 CDF、误差随时间曲线、
箱线图、长度比散点、参数量–精度帕累托图。

**诚实协议**：训练中只允许看 val；test 只在最终评测时运行一次；模型选择 fitness 默认为 val ATE。

## 7. 开发纪律

1. **洁净室**：新代码从零编写。实现者**不得打开** `/workspace/webCodex/Begin` 与 `/workspace/webCodex/open-inertial-benchmark` 下的源码
   （`*.py`、`*.sh`），不得复制其实现；公开算法依据论文与官方仓库独立实现，保持我们自己的代码风格与模块组织。
2. **测试先行**：每个模块都有确定性单元测试（合成数据）；真实数据测试放在 `tests/data/`，数据缺失时自动跳过。
3. **依赖**：核心数据层只依赖 numpy/h5py/scipy/pyyaml；torch 为 `train` 可选依赖。
4. **提交**：小步提交，信息用英文祈使句；不提交数据、权重、运行输出。
5. **不静默修复数据**：任何修正（单位、轴、四元数约定、时间戳）都要写入转换报告。
