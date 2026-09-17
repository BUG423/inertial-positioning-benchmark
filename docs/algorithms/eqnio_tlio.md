# EqNIO（TLIO 骨干）（`eqnio_tlio`）

> 规格卡版本：v1（2026-09-17）· fidelity：`official-code` · 夹具：`tests/fixtures/algorithms/eqnio_tlio.json`
>
> 姊妹卡：`eqnio_ronin`（O(2) 等变积木与输入语义的完整定义见该卡 §2.1、§4.1，本卡不再重复）。TLIO 骨干的逐层细节见 `tlio` 卡 §4。

## 1. 元信息

| 项 | 值 |
|---|---|
| 论文 | *EqNIO: Subequivariant Neural Inertial Odometry*；Jayanth\*, Xu\*, Wang, Chatzipantazis, Daniilidis, Gehrig；ICLR 2025；<https://arxiv.org/abs/2408.06321> |
| 官方仓库 | <https://github.com/RoyinaJayanth/EqNIO> @ `97b7a60a5e5cc8f0132bba343c83de114211f827`（本地：`/workspace/webCodex/third_party/EqNIO/TLIO-master`） |
| 许可 | **仓库无 LICENSE**，默认保留全部权利。`TLIO-master` 由 TLIO（BSD-3，Meta）修改而来，部分文件保留了 Meta 的 BSD 头（`src/dataloader/LICENSE`），EqNIO 新增的文件没有声明许可。风险：只能独立实现，权重不得再分发 |
| 框架 | PyTorch 2.2.2（`environment.yml`）；另需 einops、fvcore（仅用于统计 FLOPs）、numba |
| fidelity | official-code（在 CPU 上实例化官方类；已加载官方权重并做等变性核对） |
| 参考文件 | `src/main_net.py`（参数）、`src/network/model_factory.py:46-50,82-86`（结构配置）、`src/network/model_eqvncnn_frame_v2_o2.py`（O(2)，`Eq_Motion_Model_fullCov` L242-352）、`src/network/model_eq_VNCnn_frame_TLIO_v2_2vectors.py`（SO(2)，`Eq_Motion_Model_fullCov` L227-330）、`src/network/model_resnet.py`（骨干）、`src/network/train.py`、`src/network/losses.py`、`src/network/test.py`、`src/dataloader/sequences_dataset.py`、`src/dataloader/tlio_data.py`、`src/tracker/meas_source_torchscript.py`（EKF 量测）、`src/analysis/NN_output_metrics.py` |

默认变体为 **O(2)**：README 训练命令使用 `--arch eq_o2_frame_fullCov_2vec_2deep`，它也是 `main_net.py:41` 的默认值，并且在论文表 1 中表现最好。SO(2) 变体为 `eq_vn_cnn_wo_t_tlio_frame_fullCov_frameop_2vec`，其配置由发布权重的 `parameters.json` 确认。名称中的 “fullCov” 实际是指：在规范帧中用对角协方差，旋回世界系后 xy 块成为满阵。

## 2. 任务（输入）

| 项 | 官方 | IPB 映射 |
|---|---|---|
| 采样率 | 200 Hz（`--imu_freq 200`；TLIO 数据集的 npy 已重采样到 200 Hz） | 200 Hz |
| 窗口 | 1 s = 200 样本（`window_time=1`，`past_time=future_time=0`）；`net_config.in_dim = 200//32+1 = 7`（`train.py:392-400`） | `window=200` |
| 训练步长 | `decimator=10`，即起点每 10 行取一个；训练集再加 `randint(0,10)` 的随机偏移（`memmapped_sequences_dataset.py:102`，`sequences_dataset.py:532-537`） | `stride=10`，`augment=[time_shift]`（取值 [0,9]） |
| 推理步长 | 10 行，即 20 Hz（`sample_freq=20`；`test.py:616` 写死 `decimator=10`） | `eval_stride=10` |
| 坐标系 | npy 中的 IMU 已用参考姿态旋到世界系（`sequences_dataset.py:388` 注释）。之后 `express_in_local_gravity_aligned=True`：取窗口**第一个样本**的偏航 `ψ0`（`compute_euler_from_matrix(R_W_0,"xyz",extrinsic=True)[0,2]`），把输入和目标都左乘 `R_z(ψ0)ᵀ`（`sequences_dataset.py:375-397`；训练用 `tlio_data.py:45`，测试用 `test.py:617`） | 官方相当于“按窗口**起点**偏航去除”的 `gravity_yaw_local`，而 IPB 该取值按窗口末端去偏航。由于 EqNIO 对偏航严格等变，推荐直接使用 `frame=gravity_world`（见 §6） |
| 姿态来源 | TLIO 数据集的 VIO 真值姿态 | `orientation=reference` |
| 去重力 | 否（`g_compensate=False`） | `remove_gravity=false` |
| 通道顺序 | npy 列为 `ts, gyr(3), acc(3), q(xyzw), p(3), v(3)`，`imu0 = data_chunk[:,1:7]`，即 `[gyro, acc]`（`sequences_dataset.py:330`） | 与 IPB 一致，无需置换 |
| 额外输入 | 无 | 无 |
| 输入归一化 | 只把非有限值置 0（`normalize_feats`） | IPB 的有效掩码 |

**O(2) 输入语义**：与 `eqnio_ronin` §2.1 完全相同，只是计算分成两段。数据集端的 `SequencesDataset.preprocess_o2`（`sequences_dataset.py:217-255`，numpy float64）生成 9 通道 `feat_o2 = [a(3), v1(3), v2(3)]`。训练脚本端的 `preprocess_o2`（`train.py:85-91`）再按以下方式切分：`a=[0:2]`、`a_z=2`、`v1=[3:5]`、`v1_z=5`、`v2=[6:8]`、`v2_z=8`；标量顺序为 `[a_z, v1_z, v2_z, ‖a‖, ‖v1‖, ‖v2‖, a·v1, v1·v2, a·v2]`，原始标量为 `[a_z, v1_z, v2_z]`。这一步有 `.float()`，所以结果固定为 float32。经核对，两段拆分与 RoNIN 版逐通道一致，**没有**通道错配。
**SO(2) 输入语义**：`preprocess_wo_t_tlio_frame`（`train.py:68-77`）与 RoNIN 版 `preprocess_eq_frame` 相同：V 的列为 `[a_xy, ω_xy]`，S 为 `[ω_z, a_z, ‖a_xy‖, ‖ω_xy‖, a·ω]`，O 为 `[ω_z, a_z]`。

## 3. 输出

- 官方 forward 返回 `(frame, disp_c, logstd_c)`：`frame=Fᵀ ∈ R^{B×2×2}`；`disp_c ∈ R^{B×3}` 是**规范帧**中的 1 s 位移（m），规范帧只旋转 xy、不动 z；`logstd_c ∈ R^{B×3}` 是规范帧中对角标准差的对数。代码**没有**在 forward 内把结果旋回世界系（L345-350 的相关代码已被注释）。
- 回到世界系（即 IPB 所用的局部重力系）：`d_w = [F·d_c,xy, d_c,z]`，`Σ_w = [[F·diag(σx², σy²)·Fᵀ, 0], [0, σz²]]`，其中 `σ = exp(logstd_c)`。这与 `test.py:68-70` 的 `einsum('tji,tj->ti', frame, pred)`、EKF 量测 `meas_source_torchscript.py:214-233` 以及论文的 `Σ = FΣ'Fᵀ` 一致。EKF 路径在取指数之前还把 logstd 截断到不小于 −4（`meas_source_torchscript.py:220`）。
- 目标：`targ = (p[199] − p[0])`，在该窗口的局部重力系下表示（`sequences_dataset.py:176-186,386`；`train.py:260`）。训练时再把目标**旋进规范帧**：`targ_c = [Fᵀ·targ_xy, targ_z]`（`train.py:265-266`），损失在规范帧中计算。
- 官方纯网络轨迹（`test.py:62-111`）：先把预测旋回局部重力系，再除以 `window_time=1.0`（不是 0.995 s）得到速度，乘以 `dts`（窗口末端时间戳的平均间隔，0.05 s）后累加，起点为 `pos_gt[0]`。**积分前没有乘 `R_world_gla`**，得到的是逐窗口局部系位移的累加，并不是物理轨迹（见 §10）。`NN_output_metrics.py` 用同样的方式积分目标再做比较；它的 ATE 取 `mean‖·‖`（不是 RMSE），RTE 取 Δ=200 步（20 Hz 下为 10 s），切片起点有 1 的偏移。
- 论文还给出经 EKF（TLIO 滤波器，`main_filter.py`）后的结果，EKF 不在 IPB v1 范围内，其规格见 `tlio` 卡。

## 4. 网络结构

### 4.1 O(2) 规范帧网络（hidden 64，depth 2，kernel (32,1)）

与 `eqnio_ronin` §4.1–4.2 的定义**逐层、逐参数完全相同**（两份代码文件的积木实现一致，参数量都是 595,584）。输入输出形状为：`V (B,200,2,3)`、`S (B,200,9)` → … → `U (B,2,2)`。之后用 Gram–Schmidt 得到 `F`，返回 `frame=Fᵀ`，`det = ±1`（L324-331）。

### 4.2 规范化与 TLIO 骨干（L332-352）

| # | 步骤 | 输出形状 |
|---|---|---|
| 1 | `V_c = Fᵀ V`；`a_c = [V_c[...,0], O0]`，`w1 = [V_c[...,1], O1]`，`w2 = [V_c[...,2], O2]`；`ω_c = (w1×w2)/max(‖w1‖, 1e-8)` | (B,200,3)×2 |
| 2 | `X_c = cat(a_c, ω_c)`，permute 后得到骨干输入，通道为 `[a'_x, a'_y, a_z, s·ω'_x, s·ω'_y, s·ω_z]`，`s = det F`（实测误差 1.8e-15）。**加计在前**，与 TLIO 原版的 `[gyro, acc]` 相反 | (B,6,200) |
| 3 | `tlio = ResNet1D(BasicBlock1D, 6, 3, [2,2,2,2], inter_dim=7, cov_output_dim=3)`：`input_block` 为 Conv k7 s2 p3（6→64，无偏置）+ BN + ReLU + MaxPool k3 s2 p1，输出 (64,50)；残差组输出依次为 (64,50)、(128,25)、(256,13)、(512,7)；两个 `FcBlock`，每个为 prep Conv k1 512→128 + BN（BN 后**没有** ReLU）→ flatten 896 → fc1 512 + ReLU + Dropout 0.5 → fc2 512 + ReLU + Dropout 0.5 → fc3 3 | `disp_c (B,3)`，`logstd_c (B,3)` |
| 4 | 直接返回 `(frame, disp_c, logstd_c)` | — |

- 参数总量 **6,020,230** = 规范帧网络 595,584 + TLIO 骨干 5,424,646。官方配置与 benchmark 配置相同，由夹具锁定，与论文给出的 6,020,230 一致。
- 骨干与原版 TLIO 的 `model_resnet.py` 只差一个 `cov_output_dim` 参数（此处取 3），结构逐层相同，初始化也是 TLIO 的 `_initialize`（Conv 用 kaiming_normal(fan_out)，BN 为 1/0，Linear 为 N(0,0.01)/0）。规范帧网络使用 PyTorch 默认初始化。

### 4.3 SO(2) 变体（`eq_vn_cnn_wo_t_tlio_frame_fullCov_frameop_2vec`）

- 积木与 `eqnio_ronin` §4.4 相同：输入拼接 `[V, JV]`，hidden 为 128，但 **depth=2**（RoNIN 版为 1），kernel 为 (32,1)，`dim_in=2`，`dim_out=2`，`scalar_dim_in=5`，同样用双向量 Gram–Schmidt。
- 规范帧网络参数 **3,460,224**：两层卷积块各 1,638,912。模型总量 **8,884,870**，与论文一致。骨干输入通道为 `[a'_x, ω'_x, a'_y, ω'_y, ω_z, a_z]`。
- 夹具 `variants.so2` 记录了它的 `param_shapes` 与 `param_names`。

## 5. 损失与训练配方（official）

| 项 | 值 | 来源 |
|---|---|---|
| 损失 | 第 1–9 个 epoch：`(pred−targ)²`，逐轴计算（N×3），`logstd` 被 detach，协方差头**不接收梯度**（已实测：梯度为 None）。第 10 个 epoch 起：`loss_distribution_diag = e²/(2σ²) + log σ`，σ 在 `log σ ≥ log(1e-3)` 处截断，在规范帧中逐轴计算。三轴之和等于世界系满协方差 NLL `½eᵀΣ_w⁻¹e + ½log det Σ_w`（实测误差 4e-16）。最终对 batch 和 3 个轴取均值 | `losses.py:6,20-44,120-137`；`train.py:265-280` |
| 路由 | 名称含 `o2_frame_fullCov` 或 `_frameop` 时 → 对角 NLL；名称含 `_fullCov` 但不含前两者时 → 满协方差 logdet NLL（不属于发布的两个模型） | `losses.py:125-128` |
| 优化器 | Adam，lr 1e-4，无权重衰减 | `train.py:543` |
| 调度 | 构造了 `ReduceLROnPlateau(0.1, patience 10)`，但**从未调用 `step()`**，学习率实际恒为 1e-4 | `train.py:544-546`（全文件无 `scheduler.step`） |
| batch | 1024（README 与 `parameters.json`）；发布权重内保存的 args 为 512（SO(2)）和 256（O(2)），说明发生过续训 | `main_net.py:38` |
| epoch | `--epochs 50`；循环为 `range(start_epoch+1, epochs)`，即**第 1–49 个 epoch 共 49 个**，其中 MSE 阶段为第 1–9 个 epoch（9 个），NLL 阶段为第 10–49 个 epoch。论文称“先 MSE 训练 10 个 epoch” | `train.py:597` |
| 梯度裁剪 | `clip_grad_norm_(…, 0.1)` | `train.py:290` |
| **梯度清零** | 对 EqNIO 的所有 frame 架构，`do_train` **从不调用 `optimizer.zero_grad()`**：只有 `_3scalars` 分支和普通 `else` 分支会清零。因此每步梯度为 `g_t = clip(g_{t−1} + ∇L_t)`。已实测：3 个 batch 中 O(2)/SO(2) 的清零调用次数为 0，`resnet` 为 3 | `train.py:224,246-250,256` |
| 增强（O(2)） | 数据加载器层面，`'o2' in arch` 时强制开启，**train、val 都生效**：先绕随机水平轴旋转 `θ~U[0°,5°)`（重力扰动，只作用于 IMU，不改目标），再加常值偏置：陀螺 `U[−0.05,0.05]`，加计 `U[−0.2,0.2]`；之后才计算 `feat_o2`。训练变换 `TransformAddNoiseBias` 和 `TransformPerturbGravity` 只作用于 `imu0`，对 O(2) 的实际输入 `feat_o2` **不起作用**。不做偏航增强 | `train.py:470-493`；`sequences_dataset.py:426-463`；`tlio_data.py:51-65,215-230` |
| 增强（SO(2)） | 数据加载器层面不增强。训练变换作用于 `imu0`：偏置扰动（范围同上）+ 重力扰动 5°；验证集不增强 | 同上 |
| 选模 | 每个 epoch 保存一次；验证集 `losses` 均值下降时保存 `checkpoint_best.pt`。注意 epoch 10 前后比较的量纲不同（前为 MSE，后为 NLL）。发布的 O(2) 权重来自第 24 个 epoch，SO(2) 来自第 22 个 | `train.py:623-639` |
| 数据 | TLIO golden 数据集：`train_list.txt`、`val_list.txt`、`test_list.txt`；另有 Aria Everyday（论文中用于跨数据集评测） | README |

## 6. benchmark 适配（unified 配方）

| 改动 | 类型 | 理由 |
|---|---|---|
| `frame=gravity_world`（不做逐窗口去偏航） | 不改变结构 | 模型对 O(2) 严格等变（旋转误差实测 ≤ 1e-13），所以 `f(R_z x) = R_z f(x)`，与官方的“起点去偏航”在数值上等价。这样也回避了 IPB 的 `gravity_yaw_local` 按末端去偏航与官方不一致的问题 |
| 模型包装：`vel = d_w / ((T−1)·dt)`（`target=avg_velocity`，分母 0.995 s），或直接用 `target=displacement`；`cov = Σ_w / ((T−1)dt)²` | 不改变结构 | 官方目标 `p[199]−p[0]` 与 IPB 的 `displacement` 定义完全相同；官方积分时除以 1.0 s，属于协议层差异 |
| 保留 3 维输出（`dims=3`），2D 指标只取 xy | 不改变结构 | z 轴和对角 NLL 是官方损失的一部分；改为 2 维会改变 `fc3` 的形状 |
| 损失在世界系按满协方差 NLL 计算，结果除以 3；或按官方做法在规范帧中逐轴计算 | 不改变结构 | 两者在数学上恒等（§5 已实测）。必须保留 MSE→NLL 在第 10 个 epoch 的切换，以及 MSE 阶段对 logstd 的 detach |
| 梯度清零：official 配方提供开关 `grad_accumulation_bug=true` 以复现官方行为；unified 配方每步清零 | 修复官方缺陷（配方层） | 发布权重是用不清零的代码训练的；unified 配方需要行为可解释 |
| 学习率恒为 1e-4（official）；unified 按 IPB 预算 | 配方层 | 官方调度器从未生效 |
| 增强：official 配方对 O(2) 在 train 和 val 上都做重力扰动与偏置扰动；unified 配方只在 train 上做 | 协议层 | IPB 的诚实协议要求验证集确定 |
| forward 返回 `{"vel": (B,3), "cov": (B,3,3), "aux": {"frame", "disp_c", "logstd_c"}}` | 不改变结构 | 便于做等变性测试 |
| EKF 不在 v1 范围内 | — | IPB v1 只评测网络积分得到的轨迹 |

## 7. 官方报告数值

论文表 1（*Trajectory errors without (labelled with \*) and with EKF*）。MSE\* 的单位为 10⁻² m²；ATE、RTE 在论文正文和表头都写作 **mm**，但按每窗位移 MSE ≈ 3×10⁻² m² 推算，轨迹误差应在米级，**疑为 m 的笔误（未核实）**；AYE 的单位为度。带 \* 的列是纯网络结果，口径见 §3，并不是物理轨迹。

| 模型 | TLIO MSE\* | ATE | ATE\* | RTE | RTE\* | AYE | Aria MSE\* | ATE | ATE\* | RTE | RTE\* | AYE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TLIO | 3.333 | 1.722 | 3.079 | 0.521 | 0.542 | 2.366 | 15.248 | 1.969 | 4.560 | 0.834 | 0.977 | 2.309 |
| + rot. aug. | 3.242 | 1.812 | 3.722 | 0.500 | 0.551 | 2.376 | 5.322 | 1.285 | 2.103 | 0.464 | 0.521 | 2.073 |
| + SO(2) Eq. Frame | 3.194 | 1.480 | 2.401 | 0.490 | 0.501 | 2.428 | 2.457 | 1.178 | 1.864 | 0.449 | 0.484 | 2.084 |
| + O(2) Eq. Frame | **2.982** | **1.433** | **2.382** | **0.458** | **0.479** | 2.389 | **2.304** | **1.118** | **1.850** | **0.416** | **0.465** | 2.059 |

论文表 3 的消融（TLIO 数据集）中，`+S` 表示协方差只有 2 个标量（对应 `*_2scalars_*` 架构），`+P` 表示 Pearson 满协方差。O(2)+S 为 3.061/1.484/2.474/0.462/0.481/2.390，O(2)+P 为 2.990/1.827/2.316/0.578/0.478/2.534；另有 `+rot.aug+Non Eq. Frame` 等行。数值均取自 arXiv v3 HTML。论文给出的参数量为 SO(2) 8,884,870、O(2) 6,020,230、TLIO 5,424,646，与实例化结果一致。

## 8. 忠实性测试建议

以下实测值取自夹具 `equivariance_evidence`，输入与 `eqnio_ronin` 相同，均为 float64。

- [ ] 参数量 6,020,230（规范帧网络 595,584）；`param_shapes` 共 101 个（规范帧网络 23 个 + 骨干 78 个），按注册顺序一致；buffer 共 75 个。SO(2) 为 8,884,870（规范帧网络 3,460,224）。规范帧网络的参数形状应与 `eqnio_ronin` 夹具中非 `ronin.` 前缀的部分逐项相同。
- [ ] 输出：`frame (B,2,2)`，`disp (B,3)`，`logstd (B,3)`。
- [ ] **旋转**（`a_xy`、`ω_xy` 左乘 `R(θ)`）：`d_w' = M d_w`，`Σ_w' = M Σ_w Mᵀ`（`M=diag(R,1)`）；`d_c`、`logstd_c`、骨干输入都不变；`frame' = frame·Rᵀ`。误差 ≤ 1e-9（O(2) 随机初始化实测 ≤ 1e-12；SO(2) 随机初始化 ≤ 3.3e-10；官方权重 ≤ 2e-14）。
- [ ] **反射**（O(2)）：陀螺按赝矢量变换（`ω_xy ← −Qω_xy`，`ω_z ← −ω_z`）。结论同上，误差 ≤ 1e-9（实测 ≤ 1.5e-12）。
- [ ] 负向测试：反射时把陀螺当普通矢量，`d_w` 的误差必须明显大于 0（实测：随机初始化 0.52，官方权重 0.09）。SO(2) 在正确反射下也不等变（实测 0.61 / 0.75）。
- [ ] 骨干输入通道公式（§4.2 第 2 步），误差 ≤ 1e-12。
- [ ] 损失：`get_loss(epoch=9)` 等于逐轴平方误差，`epoch=10` 等于对角 NLL；MSE 阶段协方差头的梯度为 None；规范帧对角 NLL 的三轴之和等于世界系满协方差 NLL。
- [ ] 训练器：official 配方开启梯度累积开关时，第 2 步的梯度应等于 `clip(clip(g1) + g2)`。
- [ ] 可选：加载官方权重，按 `pretrained_golden`（解析输入，`disp_world`、`cov_world` 等）比对，容差 1e-4。

## 9. 预训练权重

README 提供两个 Google Drive 下载（zip）：“TLIO + SO(2) Eq. Frame”，id 为 `1kae7J8VEj2DxmSa--1COpwPzcXhsp5zk`（`TLIO_so2/checkpoint_best.pt`，第 22 个 epoch，附 `parameters.json`）；“TLIO + O(2) Eq. Frame”，id 为 `1N2kmaHSeNX0UXca8V1Iu5oRQsCvg6oOt`（`TLIO_o2/checkpoint_best.pt`，第 24 个 epoch）。两者都可以用 `weights_only=True` 读取并严格加载，sha256 记录在夹具中。仓库无许可，权重只作本地对照。

## 10. 官方实现的坑与未决问题

1. **梯度从不清零**（`train.py:246-250`）：所有 EqNIO frame 架构都会累积梯度，每步再把累积量裁剪到 0.1，行为类似一种无衰减的动量。
2. **学习率调度器从未 step**（原版 TLIO 也是如此），学习率恒为 1e-4。
3. **epoch 数少一个**：`range(start+1, epochs)`，实际为 49 个 epoch，其中 MSE 阶段 9 个。选模时 MSE 与 NLL 两种损失混在一起比较。
4. **O(2) 在验证集上也做随机增强**：`dataloader_bias_gravity_aug` 对所有 split 生效，因此验证损失是随机的。而训练变换对 O(2) 无效，因为它只作用于 `imu0`，网络实际读取的是 `feat_o2`。
5. **测试积分没有旋回世界系**：`express_in_local_gravity_aligned=True`，但 `pose_integrate` 没有乘 `R_world_gla`，所以论文中带 \* 的 ATE/RTE 是局部系位移累加的比较，不是物理轨迹误差；ATE 取 `mean‖·‖`；单位写作 mm，疑为笔误。
6. **与 TLIO 原版的坐标系不同**：原版这一版代码测试时使用全局重力系（`express_in_t0_yaw_normalized_frame=False`），EqNIO 改成了按窗口起点去偏航。
7. **论文正文的超参数与代码不一致**：论文写 SO(2) 为“1 conv block、16×1 核”，代码为 depth 2、kernel (32,1)，而论文的参数量与代码吻合。
8. 骨干输入通道被重排（O(2) 加计在前、SO(2) 交错），不能直接载入 TLIO 原版骨干的权重。
9. 名称 “fullCov” 容易误导：发布的两个模型都是规范帧中的对角协方差。
10. `pose_integrate` 对 `_2scalars` 系列架构会把 frame 再旋转一次，因为这些架构已在 forward 内部旋回过。发布的两个模型不受影响。
11. **前 10 个 epoch 的损失与 TLIO 原版不同**：原版 TLIO（`TLIO/src/network/losses.py:79-91`）在 epoch < 10 时用的是“logstd 被 detach 的对角 NLL”，EqNIO 的 `get_loss` 则是真正的 MSE。因此与 `tlio` 卡对比时，两者的阶段一并不等价。
12. 许可风险：仓库无 LICENSE。
