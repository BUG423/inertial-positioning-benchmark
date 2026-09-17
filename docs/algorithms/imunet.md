# IMUNet（`imunet`）

> 规格卡版本：v1（2026-09-17）· fidelity：`official-code` · 夹具：`tests/fixtures/algorithms/imunet.json`

> **许可风险（必读）**：官方仓库 **没有 LICENSE 文件**。按默认著作权规则视为“保留所有权利”，
> 不得复制、分发其代码或权重。本卡只描述结构与行为，供研究性重实现；IPB 中的实现必须从本卡独立编写，
> 不得粘贴官方代码。若要分发预训练权重或派生代码，需先取得作者书面许可。

## 1. 元信息

| 项 | 值 |
|---|---|
| 论文 | *IMUNet: Efficient Regression Architecture for Inertial IMU Navigation and Positioning*；Behnam Zeinali, Hadi Zanddizari, J. Morris Chang；IEEE Transactions on Instrumentation and Measurement（TIM），vol. 73，2024，DOI [10.1109/TIM.2024.3381717](https://ieeexplore.ieee.org/document/10480886)；预印本 arXiv:[2208.00068](https://arxiv.org/abs/2208.00068)（仅 v1，2022-07-29，题为 *…for IMU Navigation and Positioning*，当时投稿 IEEE TMC） |
| 官方仓库 | https://github.com/BehnamZeinali/IMUNet @ `c57f14d0f4f5bbb8e29d836dabacc8718ca825e1`（本地：`/workspace/webCodex/third_party/IMUNet`） |
| 许可 | **无 LICENSE（all rights reserved）**；代码自述由 RoNIN（GPL-3.0）修改而来，许可链不清 |
| 框架 | PyTorch（`RONIN_torch/`，论文数值来自此实现）；另有 Keras 版（`RONIN_keras/`，不完整，见 §10） |
| fidelity | official-code |
| 参考文件 | `RONIN_torch/IMUNet.py`（结构），`RONIN_torch/main.py`（`get_model` 第 36–83 行、`train` 第 168–311 行、`recon_traj_with_preds` 第 314–327 行、`test_sequence` 第 330–543 行、参数 第 577–606 行），`RONIN_torch/utils.py`（数据集类、`RandomHoriRotate` 第 30–44 行、`StridedSequenceDataset` 第 514–560 行），`RONIN_torch/metric.py`，`Datasets/*` |

同一仓库还提供 4 个一维移动端骨干，分别见 `imunet_mobilenet.md`、`imunet_mobilenetv2.md`、`imunet_mnasnet.md`、
`imunet_efficientnetb0.md`；它们与本卡共用 §2、§3、§5 的数据流水线与训练脚本，只有网络不同。
RoNIN-ResNet18 基线另见 `ronin_resnet18.md`。

## 2. 任务（输入）

IMUNet 沿用 RoNIN-ResNet 的“窗口 → 世界系水平平均速度”任务，只替换网络。

| 项 | 官方 | IPB 映射 |
|---|---|---|
| 采样率 | 200 Hz（RoNIN、RIDI 原生；自采数据集在 `Datasets/proposed/read_data_*.py` 中重采样到 200 Hz）；OxIOD 被**原样按 100 Hz** 使用（窗口因此为 2 s） | 200 Hz |
| 窗口 | `window_size=200` 样本（200 Hz 下 1 s） | `window=200`（**必须**，见 §4 的 1200 维噪声层） |
| 训练步长 | `step_size=10`；另加随机平移 `random_shift=step_size//2=5`：`frame_id += randrange(-5, 5)`（整数 ∈ [−5, 4]），再夹到 `[window_size, N_target−1]` | `stride=10` + `time_shift` 增强（±5 样本） |
| 推理步长 | `step_size=10` | `eval_stride=10` |
| 坐标系 | 用设备姿态把机体系 IMU 旋到参考（Tango/ARCore）重力对齐世界系，z 轴向上 | `frame=gravity_world` |
| 姿态来源 | RoNIN：train/val 用 `select_orientation_source(grv_only=False, max_ori_error=20)`（game RV 的对齐误差 < 20° 时用 game RV，否则取 gyro 积分 / game RV / EKF 中误差最小者）；test 强制 game RV。RIDI、自采数据集：始终 game RV。所有设备姿态都在序列首帧用参考姿态做**整旋转**对齐（`init_rotor = q_ref[0]·q_dev[0]^{-1}`） | `orientation=device`（IPB 仅补偿常值偏航，见 §6） |
| 去重力 | 否（加速度计比力，含重力）；OxIOD 分支例外，见 §10 | `remove_gravity=false` |
| 通道顺序 | `features = concat([glob_gyro_xyz, glob_acce_xyz])`，张量 `(B, 6, T)` | 与 `[gyro_xyz, acc_xyz]` **完全一致**（恒等置换） |
| 额外输入 | 无 | 无 |
| 输入归一化 | 无（`--feature_sigma/--target_sigma` 参数未被使用） | 无 |
| RoNIN 标定 | 读入时 `gyro = gyro_uncalib − imu_init_gyro_bias`，`acce = imu_acce_scale·(acce − imu_acce_bias)`（`utils.py` 第 108–109 行） | 由 IPB 转换器负责，模型不处理 |

## 3. 输出

- 形状 `(B, 2)`：参考世界系**水平速度** `(v_x, v_y)`，单位 m/s，无协方差/不确定度。
- 训练目标（`interval = window_size = 200`）：窗口起点为 `j` 时，
  `target = (p[j+200] − p[j])[:2] / (t[j+200] − t[j])`；输入窗口是 `features[j : j+200]`。
  即目标跨 200 个采样间隔、终点比窗口最后一个输入样本晚 1 个样本。
- 官方轨迹重建（`main.py:recon_traj_with_preds`）：
  1. 取测试窗口起点索引 `ind`（步长 10），`dts = mean(t[ind[1:]] − t[ind[:-1]])`（≈ 0.05 s）；
  2. `pos[0] = p_gt[0, :2]`，`pos[1:-1] = p_gt[0,:2] + cumsum(v̂_k · dts)`，`pos[-1] = pos[-2]`；
  3. 时间戳 `[t0 − 1e−6, t[ind], t_end + 1e−6]`，线性插值到全部帧。
  第 k 个窗口的位移被记在**窗口起点**时刻（比 IPB 的“窗口中心”规则早半个窗口），末窗之后常值保持。
- IPB 前向返回 `{"vel": (B, 2)}`。

## 4. 网络结构

输入 `(B, 6, 200)`。所有 BN 为 `BatchNorm1d(eps=1e-5, momentum=0.1, affine=True)`（即 PyTorch 默认值）。
没有任何 dropout（构造参数 `dropout_rate=0.5` 未被使用）。

### 4.1 两种残差块

**DSR（`DSConv_Regular`，恒等捷径，要求 C_in = C_out 且 stride = 1）**，参数 `(C, k=3, p=1)`：

```text
y = ELU(BN1(DW(x)))           DW  = Conv1d(C, C, k=3, stride=1, pad=1, groups=C, bias=False)
y = ELU(BN2(PW(y)))           PW  = Conv1d(C, C, k=1, bias=False)
out = ELU(y + x)
```

注意主路径第二个卷积之后**已经有 ELU**，相加后**再**过一次 ELU（与 ResNet 的“BN 后相加再激活”不同）。

**DSP（`DSConv`，投影捷径）**，参数 `(C_in, C_out, stride s, k=3, p=1)`：

```text
y = ELU(BN1(DW(x)))           DW  = Conv1d(C_in, C_in, k=3, stride=s, pad=1, groups=C_in, bias=False)
y = ELU(BN2(PW(y)))           PW  = Conv1d(C_in, C_out, k=1, bias=False)
sc = BN_d(Conv1d(C_in, C_out, k=1, stride=s, pad=0, bias=False)(x))
out = ELU(y + sc)
```

长度：`L_out = ceil(L_in / s)`（两条路径一致）。ELU 的 `alpha=1`。

### 4.2 逐层表（T = 200）

| # | 层（官方名） | 参数（核/步长/填充/通道） | 归一化 | 激活 | dropout | 输出形状 | 参数量 |
|---|---|---|---|---|---|---|---|
| 0 | 输入 | — | — | — | — | (B, 6, 200) | — |
| 1 | `input_block.0` Conv1d | 6→64, k7, s2, p3, bias=False | — | — | — | (B, 64, 100) | 2,688 |
| 2 | `input_block.1/2` | — | BN(64) | **ReLU**（不是 ELU） | — | (B, 64, 100) | 128 |
| 3 | `input_block.3` MaxPool1d | k3, s2, p1 | — | — | — | (B, 64, 50) | 0 |
| 4 | `conv_1_1` DSR | C=64 | BN×2 | ELU | — | (B, 64, 50) | 4,544 |
| 5 | `conv_1_2` DSR | C=64 | BN×2 | ELU | — | (B, 64, 50) | 4,544 |
| 6 | `conv_3_1` DSP | 64→64, s1（有 1×1 投影） | BN×3 | ELU | — | (B, 64, 50) | 8,768 |
| 7 | `conv_3_2` DSR | C=64 | BN×2 | ELU | — | (B, 64, 50) | 4,544 |
| 8 | `conv_4_1` DSP | 64→128, s2 | BN×3 | ELU | — | (B, 128, 25) | 17,216 |
| 9 | `conv_4_2` DSR | C=128 | BN×2 | ELU | — | (B, 128, 25) | 17,280 |
| 10 | `conv_5_1` DSP | 128→256, s2 | BN×3 | ELU | — | (B, 256, 13) | 67,200 |
| 11 | `conv_5_2` DSR | C=256 | BN×2 | ELU | — | (B, 256, 13) | 67,328 |
| 12 | `conv_6_1` DSP | 256→512, s2 | BN×3 | ELU | — | (B, 512, 7) | 265,472 |
| 13 | `conv_6_2` DSR | C=512 | BN×2 | ELU | — | (B, 512, 7) | 265,728 |
| 14 | `conv_7_1` DSP | 512→1024, s2 | BN×3 | ELU | — | (B, 1024, 4) | 1,055,232 |
| 15 | `conv_7_2` DSR | C=1024 | BN×2 | ELU | — | (B, 1024, 4) | 1,055,744 |
| 16 | `output_block.0` Conv1d | 1024→400, k2, s1, p0, **bias=True** | — | — | — | (B, 400, 3) | 819,600 |
| 17 | `output_block.1` | — | BN(400) | **无** | — | (B, 400, 3) | 800 |
| 18 | flatten | 通道优先展平（索引 `c·3 + t`） | — | — | — | (B, 1200) | 0 |
| 19 | `noise`（`CustomLayer`） | `z = f − W ⊙ u + b`，`u = flatten(输入 x)`（索引 `c·200 + t`），`W, b ∈ R^{1×1200}` | — | — | — | (B, 1200) | 2,400 |
| 20 | ELU | — | — | ELU | — | (B, 1200) | 0 |
| 21 | `fc.0` Linear | 1200→2, bias | — | — | — | (B, 2) | 2,402 |

- **参数总量：3,661,618**（全部可训练）；官方配置 = benchmark 配置（无需改结构）。98 个参数张量，93 个 BN 缓冲区（31 个 BN × 3）。
- 计算量（本规格工程师测量，非论文值）：卷积+全连接权重乘加 **18.44 M MACs**/窗口（`torch.utils.flop_counter` 结果 ÷ 2，不含 BN/激活/偏置）。
- **参数注册顺序**（夹具 `param_shapes` 按此顺序）：`noise.W, noise.b` 最先（在 `__init__` 中第一个创建），然后
  `input_block.*`、`conv_1_1` … `conv_7_2`（块内顺序：`depth_wise, bn_1, point_wise, bn_2[, downsample_.0, downsample_.1]`）、
  `output_block.0.{weight,bias}`、`output_block.1.{weight,bias}`、`fc.0.{weight,bias}`。
- 权重初始化：全部 PyTorch 默认（Conv/Linear：`kaiming_uniform_(a=√5)` + 默认偏置；BN：γ=1、β=0）；
  **`noise.W ~ N(0, 1)`（`torch.randn`），`noise.b = 0`**。
- 注意事项：
  - “噪声层”把**原始输入**逐元素（按展平索引）接到展平特征上：特征的第 `i` 个元素与原始输入的第 `i` 个元素配对，
    两者没有语义对应关系（特征是 `(400 通道, 3 时刻)`，输入是 `(6 通道, 200 时刻)`）。复现时必须保持这种“按索引配对”，
    并使用通道优先展平（等价 `x.reshape(B, -1)`，输入需 contiguous）。
  - 该层把结构锁死在 `C·T = 6·200 = 1200` 且 `output_block` 输出长度为 3；T=100 或 400 时官方模型直接报形状错误（夹具 `other_window_output_shapes` 已记录）。
  - `IMUNet.conv_block` 是未使用的方法（EEG 代码残留），不产生参数；构造参数 `num_classes/input_size/sampling_rate/num_T/num_S/hidden/dropout_rate` 全部未使用，输出维度硬编码为 2。

## 5. 损失与训练配方（official）

| 项 | 值 | 来源 |
|---|---|---|
| 损失 | `MSELoss`（对 batch 与 2 个分量取均值） | `main.py` 第 206、259 行 |
| 优化器 | Adam，`lr=1e-4`，betas/eps 默认，**无权重衰减** | 第 207、589 行 |
| 学习率调度 | `ReduceLROnPlateau(mode=min, factor=0.1, patience=10, eps=1e-12)`，每个 epoch 用验证集平均 MSE `step` 一次 | 第 208、284 行 |
| batch | 训练 128；验证 512（shuffle） | 第 173、181、590 行 |
| epoch | 300（论文 §V-B 原文：“We have used the same parameters as [14]. However, each model has been trained for 300 epochs.”，[14] 即 RoNIN） | 第 591 行；论文 §V-B |
| 梯度裁剪 | 无 | — |
| 增强 | `RandomHoriRotate(2π)`：每个样本 `θ ~ U[0, 2π)`，把陀螺 xy、加计 xy 与目标 xy 同时绕 z 轴旋转 θ（仅训练）；±5 样本随机平移（见 §2） | `utils.py` 第 30–44、545–547 行；`main.py` 第 115–118 行 |
| 阶段切换 | 无 | — |
| 选模/早停 | 无早停；验证 MSE 最低的 epoch 存为 `checkpoint_best.pt`，测试用它 | 第 288–295、651 行 |
| 训练前 | 先在训练集上以 `train()` 模式跑一遍前向（不反传）记录初始损失——这会更新 BN 滑动统计 | 第 229 行 |
| 随机种子 | 未设置 | — |
| 数据划分 | RoNIN：`Datasets/ronin/list_{train,val,test_seen,test_unseen}.txt`（69/15/31/31 条，公开的一半数据）；RIDI：`list_train_publish_v2`（49）/ 验证=测试=`list_test_publish_v2`（23）；自采：`list_train`（36）/ 验证=测试=`list_test`（13）；OxIOD：`train/` 与 `validation/` 目录（验证=测试） | 第 622–689 行 |

## 6. benchmark 适配（unified 配方）

| 改动 | 类型 | 理由 |
|---|---|---|
| 输入 `(B,6,200)`，通道 `[gyro, acc]`，`frame=gravity_world`，`dims=2`，`target=avg_velocity` | 不改变结构 | 与官方完全一致 |
| 目标定义由 `(p[j+200]−p[j])/(t[j+200]−t[j])` 改为 IPB 的 `(p[s+199]−p[s])/(199·dt)` | 不改变结构 | IPB 协议统一；两者差 1 个样本（0.5%），对速度影响可忽略 |
| 设备姿态对齐：官方首帧**整旋转**对齐，IPB 只补偿常值偏航 | 不改变结构 | IPB 设计宪法 §3；两者在重力对齐良好的设备姿态上等价 |
| 推理时间戳由窗口起点改为窗口中心，积分用梯形法 | 不改变结构（只影响评测） | IPB 设计宪法 §5，修正官方半窗错位 |
| OxIOD 使用 IPB 统一的 200 Hz、含重力比力、四元数约定 | 不改变结构 | 官方 OxIOD 分支有约定错误（§10），不复现该错误 |
| `window` 必须保持 200、输入通道必须保持 6 | —（约束） | 噪声层维度 1200 = 6×200；若 unified 预算要求其他窗口，IMUNet **不参与**或须另立“改变结构”的变体并单独命名 |
| unified：epoch/batch/调度按 benchmark 统一预算；选模改用 val ATE，且 val 与 test 严格分离 | 不改变结构 | 诚实协议；官方 RIDI/自采/OxIOD 用测试集选模 |
| `ReduceLROnPlateau` 不传 `verbose` | 不改变结构 | 本机 torch 2.11 的该类已无 `verbose` 参数 |

结论：official 配方下 **无需任何结构改动**，夹具即官方配置。

## 7. 官方报告数值

来源：arXiv v1（2022）Table I / Table III。已发表的 TIM 2024 版本数值 **未核实**（可能不同）。
指标定义（`metric.py`）：`ATE = sqrt(mean_{t,轴} (p̂−p)²)`（**按坐标轴平均**，2D 下等于 IPB ATE / √2）；
`RTE` = 1 分钟（`pred_per_min = 12000` 帧）相对位移的同样按轴 RMSE，序列不足 1 分钟时用全长漂移 × `12000/N`。
与 IPB 比较时须乘以 √2，且官方轨迹有半窗时间错位。

| 数据集 | 指标 | ResNet18 | MobileNet | MobileNetV2 | MnasNet | EfficientNetB0 | **IMUNet** |
|---|---|---|---|---|---|---|---|
| 自采（seen） | ATE / RTE (m) | 2.73 / 3.03 | 2.98 / 3.42 | 3.03 / 3.55 | 2.75 / 3.19 | 2.67 / 3.48 | **2.59 / 2.97** |
| RoNIN seen | ATE / RTE | 3.63 / 2.76 | 4.08 / 2.83 | 3.83 / 2.85 | 3.78 / 2.75 | 3.66 / 2.79 | **3.52 / 2.71** |
| RoNIN unseen | ATE / RTE | 5.65 / 4.57 | 6.16 / 4.75 | 6.17 / 4.69 | 5.19 / 4.54 | 5.68 / 4.60 | **5.68 / 4.49** |
| OxIOD（seen） | ATE / RTE | 3.14 / 2.66 | 3.20 / 2.68 | 3.38 / 2.89 | 3.08 / 2.64 | 3.22 / 2.69 | **2.88 / 2.58** |
| RIDI（seen） | ATE / RTE | 1.56 / 1.92 | 1.73 / 2.09 | 1.55 / 1.97 | 1.71 / 2.10 | 1.67 / 2.05 | **1.56 / 1.83** |
| PX4（unseen，无人机，非行人） | ATE | 92.46 | 64.94 | 65.86 | 56.70 | 65.42 | **71.66** |

Table III（Galaxy S10 上的 **Keras→TFLite** 模型，非 PyTorch 模型）：

| 指标 | ResNet18 | MobileNet | MobileNetV2 | MnasNet | EfficientNetB0 | IMUNet |
|---|---|---|---|---|---|---|
| TFLite 大小 (MB) | 4.5 | 3.5 | 2.7 | 3.1 | 3.8 | 1.4 |
| 单窗口延迟 (µs) | 1044 | 907 | 645 | 654 | 967 | 387 |

参数量与 FLOPs 在论文中只以散点图（Fig. 1/3）给出，无表格数值，**未核实**。
论文 Table II 的 IMUNet 结构（4 个 MRBlock 至 512 通道 + Conv1D 128 + Dense）与公开代码（5 组块至 1024 通道 + 400 通道卷积 + 噪声层）**不一致**；
IPB 以公开 PyTorch 代码为准（它是 Table I 数值的来源）。TFLite 1.4 MB 暗示 Keras 版 IMUNet 可能远小于 3.66 M 参数，但 Keras 版 IMUNet 未公开，无法核实。

## 8. 忠实性测试建议

- [ ] `total_params == trainable_params == 3,661,618`（夹具）
- [ ] `param_shapes` 98 项逐项一致（注意 `noise.W/noise.b` 排在最前）；BN 缓冲区 93 个
- [ ] 输入 `(4, 6, 200)` → 输出 `(4, 2)`；逐层形状与 §4.2 一致（50 → 50 → 25 → 13 → 7 → 4 → 3）
- [ ] 输入 `(1, 6, 100)` 或 `(1, 6, 400)` 必须抛出形状错误（或在构造时显式校验 `window == 200`）
- [ ] 展平顺序：`eval()` 下把 `output_block.1` 的 γ、β 置 0（使 f=0），`noise.b=0`，`noise.W` 设为全 1，`fc.0.weight` 设为只选第 `i` 个元素的 one-hot、偏置 0，则输出等于 `ELU(−x[:, c, t])`，其中 `i = c·200 + t`
- [ ] `noise.W` 初始化为标准正态（大样本均值≈0、方差≈1），`noise.b` 为零
- [ ] 模型中不含 Dropout；`input_block` 激活为 ReLU，其余块为 ELU
- [ ] `RandomHoriRotate`：对同一样本，旋转后的输入与目标满足 `R(θ)` 关系，z 分量不变
- [ ] 模型不具备旋转等变性（不作为性质测试；仅记录）

## 9. 预训练权重

README 提供 PyTorch 预训练模型的 Google Drive 链接：
https://drive.google.com/file/d/1NGwBhvh-KjVg0CpMeFtYI72G4J0LOPWE/view?usp=sharing（未下载，内容与对应数据集未核实）。
无许可，**不得随 IPB 分发**；如用于对照，只在本地加载并记录来源。
自采数据集：https://drive.google.com/file/d/1A49YZ1G8vkEJPIb51n-MFITropK4pIhu/view?usp=drive_link（对应 IPB `imunet` 数据集）。

## 10. 官方实现的坑与未决问题

1. **PyTorch `main.py` 不能直接运行**：第 23 行 `from CNN_LSTM import *` 引用仓库中不存在的模块；`model_resnet1d.py` 导入 `pthflops`；
   `ReduceLROnPlateau(..., verbose=True)` 在本机 torch 2.11 上抛 `TypeError`（该参数已被移除）。`--arch` 的 choices 不含 `MobileNetV2`，尽管 `get_model` 支持。
2. **论文结构表与代码不一致**（§7）；论文称全网 ELU，代码 stem 用 ReLU。
3. **噪声层**把原始输入按索引与特征相减（§4），窗口/通道数被锁死在 6×200。
4. **测试集选模**：RIDI 与自采数据集的 `val_list` 就是测试列表，OxIOD 的验证目录也是测试目录（`main.py` 第 629–643、671–689 行），报告数值带有测试集信息泄漏。
5. **OxIOD 分支约定错误**（`utils.py` 第 382–417 行，`Datasets/oxiod/readOXIOD.py`）：CSV 中四元数按 `(x, y, z, w)` 存储，却直接交给期望 `(w, x, y, z)` 的
   `quaternion.from_float_array`；向量四元数构造成 `[g_x, g_y, g_z, 0]`（实部放在第 0 位），旋转结果不是正确的世界系向量；
   加速度取 OxIOD CSV 第 10–12 列（按 OxIOD 官方列定义为去重力的 user acceleration，单位 g），与其他数据集的比力不一致；
   OxIOD 数据按 100 Hz 使用而窗口仍为 200 样本；`OXIODSequence.target_dim=3` 与实际 2 维目标不符。
6. **时间对齐**：推理速度记在窗口起点（半窗错位），RTE 以“12000 帧 = 1 分钟”计，对 100 Hz 的 OxIOD 实际是 2 分钟。
7. **训练窗口**：随机平移后 `frame_id` 被夹到 ≥ 200，每条序列前 200 个起点永远不参与训练（继承自 RoNIN）；验证/测试不受影响。
8. **Keras 版不完整**：`RONIN_keras/IMUNet.py` 只有文档字符串，没有模型；`main.py` 引用未定义的 `MobileNetV2_1D_Arch`、`MobileNetV1_1D`；
   `get_model` 用 `input_shape=(6, 200)`（channels-last 下是 6 个时刻、200 个通道），与数据生成器输出的 `(B, 200, 6)` 矛盾。Keras 训练使用 Adam +
   SGDR 余弦重启（`LR_Restart(lr, lr·1e-6, 10)`，每 10 个 epoch 重启一次）、按 batch 取均值的逐分量 MSE、`save_best_only` 以验证损失选模，
   与 PyTorch 配方不同。论文数值来自 PyTorch 版，**IPB 以 PyTorch 版为准**。
9. 论文 Table III 的延迟/大小是 Keras 模型的 TFLite 版本，不对应本卡的 PyTorch 结构。
