# EqNIO（RoNIN 骨干）（`eqnio_ronin`）

> 规格卡版本：v1（2026-09-17）· fidelity：`official-code` · 夹具：`tests/fixtures/algorithms/eqnio_ronin.json`
>
> 姊妹卡：`eqnio_tlio`（同一等变规范帧网络接 TLIO 骨干）。RoNIN 骨干逐层细节见 `ronin_resnet18` 卡 §4（代码逐行一致，参数量相同）。

## 1. 元信息

| 项 | 值 |
|---|---|
| 论文 | *EqNIO: Subequivariant Neural Inertial Odometry*；Royina Karegoudra Jayanth\*, Yinshuang Xu\*, Ziyun Wang, Evangelos Chatzipantazis, Kostas Daniilidis, Daniel Gehrig；ICLR 2025；<https://arxiv.org/abs/2408.06321>（v3），<https://openreview.net/forum?id=C8jXEugWkq> |
| 官方仓库 | <https://github.com/RoyinaJayanth/EqNIO> @ `97b7a60a5e5cc8f0132bba343c83de114211f827`（本地：`/workspace/webCodex/third_party/EqNIO`，子目录 `RONIN/`） |
| 许可 | **仓库没有 LICENSE 文件**，按默认规则保留全部权利。另外，`RONIN/source/` 由 GPL-3.0 的 RoNIN 代码修改而来，却没有声明许可。风险：不得复制代码；预训练权重只能在本地做对照，不得再分发。IPB 必须依据论文和本卡独立实现 |
| 框架 | PyTorch（`environment.yml` 固定 pytorch 2.2.2、python 3.12、numpy 1.26.4；另需 einops、numpy-quaternion、numba、tensorboardX） |
| fidelity | official-code（在 CPU 上实例化官方类，并加载官方预训练权重核对） |
| 参考文件 | `RONIN/source/ronin_resnet.py`（`get_model`、预处理、训练与测试）、`RONIN/source/model_resnet1d_eq_frame_o2.py`（O(2) 规范帧网络）、`RONIN/source/model_resnet1d_eq_frame_2vec.py`（SO(2) 规范帧网络，对应发布的预训练模型）、`RONIN/source/model_resnet1d_eq_frame.py`（单向量 SO(2) 变体，未发布权重）、`RONIN/source/model_resnet1d.py`（RoNIN 骨干）、`RONIN/source/data_glob_speed.py`、`RONIN/source/data_utils.py`、`RONIN/source/metric.py` |

默认变体采用 **O(2) Eq. Frame**：它是 README 训练命令里的 `--arch resnet18_eq_frame_o2`，也是论文表 2 中最好的一行。SO(2) 变体的结构差异在 §4.4 说明，夹具 `variants.so2` 锁定其参数形状。

## 2. 任务（输入）

| 项 | 官方 | IPB 映射 |
|---|---|---|
| 采样率 | 200 Hz（RoNIN 原生） | 200 Hz |
| 窗口 | 200 样本，即 1 s（`--window_size 200`）；骨干 FC 输入长度为 `200//32+1=7`，由模型内部写死（`model_resnet1d_eq_frame_o2.py:152`） | `window=200` |
| 训练步长 | `--step_size 10`（预训练 `config.json` 取 10；argparse 默认 5，`ronin_resnet.py:503`）。每个样本起点再加 `random.randrange(-5, 5)`，并截断到 `[200, N-1]`（`data_glob_speed.py:172-174`） | `stride=10`，`augment=[time_shift]`（±5） |
| 推理步长 | `step_size`，取 10 | `eval_stride=10` |
| 坐标系 | RoNIN 的重力对齐、偏航任意的世界系（HACF）。EqNIO 在网络内部为每个窗口再求一个 O(2) 规范帧 | `frame=gravity_world` |
| 姿态来源 | 训练和验证：`grv_only=False`、`max_ori_error=20`。game RV 的对齐误差小于 20° 时用 game RV，否则取误差最小的姿态源（`data_utils.py:106-150`）。测试：`grv_only=True`，只用 game RV（`ronin_resnet.py:163`） | official 配方的测试用 `orientation=device`；unified 配方按 IPB 默认 |
| 去重力 | 否（加计是比力，旋到世界系后 z≈+9.8） | `remove_gravity=false` |
| 通道顺序 | 特征为 `[glob_gyro_xyz, glob_acce_xyz]`（`data_glob_speed.py:70`），预处理按 `gyro=feat[...,:3]`、`accel=feat[...,-3:]` 读取（`ronin_resnet.py:83-84`） | 与 IPB 的 `[gyro_xyz, acc_xyz]` 完全一致，**无需置换**。但骨干网络的输入通道顺序已被重排，见 §2.1 |
| 额外输入 | 无。向量和标量特征都由 6 维 IMU 在模型内部构造 | 无 |
| 输入归一化 | 无（陀螺已去掉初始偏置，加计已做 scale/bias 标定，由转换器负责） | 无 |
| 增强 | 等变架构**不使用** `RandomHoriRotate`（`ronin_resnet.py:152-154`），只保留 `random_shift` | 不需要 `random_yaw` |

### 2.1 输入通道语义（逐步，O(2) 变体；实现必须与此完全一致）

记 IPB 输入 `x ∈ R^{B×6×T}`，`ω_t = x[:,0:3,t]`（世界系角速度，rad/s，**赝矢量**），`a_t = x[:,3:6,t]`（世界系比力，m/s²，普通矢量）。官方 `preprocess_eq_o2_frame`（`ronin_resnet.py:80-104`）对每个样本做以下计算：

1. `w̃ = (−ω_y, ω_x, 0)`，即绕 z 轴转 90° 后把 z 分量置零（代码中的 `gyro @ R.T`，再令 `[...,-1]=0`）。
2. 若 `ω_xy ≠ (0,0)`（**严格浮点相等**判断）：`u1 = ω × w̃`，`u2 = ω × u1`。
   否则：`u1 = e_y × ω`（`e_y=(0,1,0)`），`u2 = ω × u1`。
3. `v1 = u1·‖ω‖ / max(‖u1‖, 1e-7)`，`v2 = u2·‖ω‖ / max(‖u2‖, 1e-7)`。可以推出 `v1 × v2 = ‖ω‖·ω`，`v2 = −‖ω‖·w̃/‖w̃‖`。
4. 向量特征 `V ∈ R^{B×T×2×3}`：`V[...,i,j]` 是第 j 个向量的第 i 个分量（i∈{x,y}），列顺序为 `[a_xy, v1_xy, v2_xy]`。
5. 标量特征 `S ∈ R^{B×T×9}`：`[a_z, v1_z, v2_z, ‖a_xy‖, ‖v1_xy‖, ‖v2_xy‖, a_xy·v1_xy, v1_xy·v2_xy, a_xy·v2_xy]`。
6. 原始标量 `O ∈ R^{B×T×3}`：`[a_z, v1_z, v2_z]`。

注意：`v1`、`v2` 由 `torch.zeros` 创建，使用**默认 dtype（float32）**，所以即使输入是 float64，预处理结果也会变成 float32。

**与论文的差异**：论文写的是 `F(ω) = (‖ω‖ w1/‖w1‖, ‖ω‖ w2/‖w2‖)`，其中 `w1=[−ω_y, ω_x, 0]`，`w2=ω×w1`。代码实际用的是 `(v1, v2) = (‖ω‖·(ω×w1)/‖·‖, −‖ω‖·w1/‖w1‖)`，也就是论文中两个向量的顺序对调、并且其中一个取反。两种写法在等变性上等价，都满足“在反射下是普通矢量，且叉积能还原 ω”。但**加载官方权重时必须采用代码的顺序和符号**。

**反射下的变换规律**：设 `M = diag(Q, 1)`，`Q ∈ O(2)`。IPB 输入按物理规律变换：`a' = M a`，`ω' = det(Q)·M ω`（赝矢量）。此时 `v1' = M v1`，`v2' = M v2`，两者都按普通矢量变换，于是 S 与 O 在 O(2) 下不变，V 按 `Q` 变换。如果误把陀螺当普通矢量来变换（`ω' = Mω`），`v2` 会多出一个符号，等变性随之失效；§8 的负向测试就用来验证这一点。

## 3. 输出

- `vel ∈ R^{B×2}`：重力对齐世界系中的水平平均速度（m/s），不含协方差。辅助输出 `frame ∈ R^{B×2×2}` 等于 `Fᵀ`：每行是一个规范基向量，`det = ±1`。
- 官方目标与 RoNIN 相同：`v_s = (p[s+200] − p[s]) / (t[s+200] − t[s])`，只取 xy。输入窗口为 `features[s:s+200]`（`data_glob_speed.py:60-61,176-177`，其中 `interval=window_size`，见 `data_glob_speed.py:154`）。
- 官方轨迹重建（`ronin_resnet.py:343-356`）与 RoNIN 相同：速度乘以 `dts`（步长平均时间，约 0.05 s）后累加，时间戳落在**窗口起点**，再插值到所有帧。起点为 `gt_pos[0]`。
- 官方指标（`metric.py`，与 RoNIN 相同）：`ATE = sqrt(mean((est−gt)**2))` 对 N×2 个**坐标分量**求均值，相当于欧氏 RMSE 除以 √2；RTE 取 Δ=12000 帧（1 min），序列不足 1 min 时取 Δ=N−1，再乘以 `12000/N`。

## 4. 网络结构

整体流程：`IPB x → 预处理(§2.1) → 规范帧网络 → Gram–Schmidt 得到 F → 规范化 → RoNIN-ResNet18(6→2) → v = F·v_c`。

### 4.1 等变积木（O(2)，`model_resnet1d_eq_frame_o2.py`）

向量张量的形状为 `(…, 2, C)`，倒数第二维是空间 x/y，最后一维是通道；标量张量为 `(…, C)`。

| 积木 | 定义 | 参数 |
|---|---|---|
| `VNLinear(Ci→Co)`（L11-22） | `V ↦ V·Wᵀ`，即无偏置的 `nn.Linear` 作用在通道维，x/y 两个分量共享权重 | `W: (Co, Ci)` |
| 标量线性 | 无偏置 `nn.Linear` | `(Co, Ci)` |
| `NonLinearity(Cv, Cs → Cv, Cs')`（L24-47） | `n = ‖V‖_{空间维}`，形状 `(…,Cv)`；`z = LN(ReLU(Linear_nobias([n ‖ s])))`，`z ∈ R^{Cv+Cs'}`；`V' = z[:Cv] ⊙ V / max(‖V‖, 1e-6)`；`s' = z[−Cs':]`。当 `Cs'=0` 时只返回 V' | Linear `(Cv+Cs', Cv+Cs)`；LN 的 γ `(Cv+Cs',)` |
| `LayerNorm(D)`（L49-56） | `F.layer_norm(x, (D,), γ, β)`，eps=1e-5；γ 初始化为 1，**β 是固定为 0 的 buffer**，不参与训练 | γ `(D,)` |
| `VNLayerNorm(C)`（L58-67） | `n=‖V‖`；`V' = V / max(n, 1e-6) ⊙ LN(n)` | LN 的 γ `(C,)` |
| `Convolutional`（L80-98） | 向量分支：`Conv2d(Cv→Cv', kernel=(32,1), stride=1, padding='same', padding_mode='replicate', bias=False)` 作用在 `(B, Cv, T, 2)` 上，只在时间维卷积，x/y 共享权重。标量分支：`Conv2d(Cs→Cs', (32,1), 同上)` 作用在 `(B, Cs, T, 1)` 上。偶数核的 `'same'` 填充为**左 15、右 16**（已实测），复制边界值 | `(Cv', Cv, 32, 1)`；`(Cs', Cs, 32, 1)` |
| `MeanPooling`（L69-78） | 在时间维（dim=1）上对 V 和 s 求均值 | — |

### 4.2 规范帧网络（O(2)，hidden=64，depth=2；`get_model` 见 `ronin_resnet.py:40-44`）

输入 `V (B,200,2,3)`、`S (B,200,9)`、`O (B,200,3)`，下表以 B=1 列出：

| # | 层（注册名） | 参数 | 激活/归一化 | 输出形状 | 参数量 |
|---|---|---|---|---|---|
| 1 | `vnlinear_layer0` | VNLinear 3→64 | — | V (1,200,2,64) | 192 |
| 2 | `slinear_layer0` | Linear 9→64，无偏置 | — | S (1,200,64) | 576 |
| 3 | `nonlinearity0` | NonLinearity(64,64→64,64)：Linear 128→128 | ReLU → LN(128) | V (1,200,2,64)，S (1,200,64) | 16,384+128 |
| 4 | `layers.0.0` | Convolutional 64→64 与 64→64，k=(32,1) | — | 同上 | 131,072+131,072 |
| 5 | `layers.0.1` | NonLinearity(64,64→64,64) | ReLU → LN(128) | 同上 | 16,512 |
| 6 | `layers.0.2` / `layers.0.3` | VNLayerNorm(64)，作用于 V；LayerNorm(64)，作用于 S | — | 同上 | 64 + 64 |
| 7 | `layers.1.*` | 与 4–6 相同 | — | 同上 | 278,784 |
| 8 | `pooling_layer1` | 时间均值 | — | V (1,2,64)，S (1,64) | — |
| 9 | `vnlinear_layer1` / `slinear_layer1` | VNLinear 64→64；Linear 64→64，无偏置 | — | V (1,2,64)，S (1,64) | 4,096+4,096 |
| 10 | `nonlinearity1` | NonLinearity(64,64→64,0)：Linear 128→64 | ReLU → LN(64) | V (1,2,64)（不再输出标量） | 8,192+64 |
| 11 | `vnlinear_layer2` | VNLinear 64→64 | — | V (1,2,64) | 4,096 |
| 12 | `vector_ln1` | VNLayerNorm(64) | — | V (1,2,64) | 64 |
| 13 | `vnoutput_layer` | VNLinear 64→2 | — | U (1,2,2)，两列分别为 u1、u2 | 128 |

规范帧网络合计 **595,584** 个参数。其中 depth=1 的块共 278,784 个参数（131,072×2 + 16,512 + 128）。

**Gram–Schmidt**（L184-191）：`e1 = u1 / max(‖u1‖, 1e-8)`；`ũ2 = u2 − (u2·e1)·e1`；`e2 = ũ2 / max(‖ũ2‖, 1e-8)`；`F = [e1 e2]`（按列排）。代码返回的是 `frame = Fᵀ`。`det F` 可以是 +1 或 −1（实测两种符号都会出现），这正是 O(2) 规范化所需要的。

### 4.3 规范化与骨干（L192-205）

| # | 步骤 | 输出形状 |
|---|---|---|
| 14 | `V_c = Fᵀ·V`：对每个时刻、每一列分别左乘 | (1,200,2,3) |
| 15 | `a_c = [V_c[...,0], O[...,0]]`，`w1 = [V_c[...,1], O[...,1]]`，`w2 = [V_c[...,2], O[...,2]]` | 各 (1,200,3) |
| 16 | `ω_c = (w1 × w2) / max(‖w1‖, 1e-8)`。实测 `ω_c = det(F)·[Fᵀω_xy, ω_z]`，误差 3e-15 | (1,200,3) |
| 17 | `X_c = cat(a_c, ω_c)`，再 permute 成 (B,6,T)。**通道顺序为 `[a'_x, a'_y, a_z, s·ω'_x, s·ω'_y, s·ω_z]`，其中 `s=det F`，加计在前**，与 RoNIN 原版的陀螺在前不同 | (1,6,200) |
| 18 | `ronin`：`ResNet1D(6, 2, BasicBlock1D, [2,2,2,2], base_plane=64, kernel_size=3, FCOutputModule(fc_dim=512, in_dim=7, dropout=0.5, trans_planes=128))`。逐层结构与 `ronin_resnet18` 卡 §4 相同：Conv k7 s2 → (64,100)，MaxPool → (64,50)，四个残差组 → (64,50)/(128,25)/(256,13)/(512,7)，transition → (128,7)，flatten 896 → FC 512 → 512 → 2 | (1,2)，记为 `v_c` |
| 19 | `vel = F·v_c`（L205） | (1,2) |

- 参数总量 **5,230,466** = 规范帧网络 595,584 + 骨干 4,634,882。官方配置与 benchmark 配置相同，由夹具锁定，全部可训练。buffer 包括骨干 BN 的统计量，以及每个 LN 的 β（值恒为 0）。
- 权重初始化：骨干沿用 RoNIN 的 `_initialize`，即 Conv 用 kaiming_normal(fan_out)，BN 权重为 1、偏置为 0，Linear 权重 ~N(0,0.01)、偏置为 0。`_initialize` 只遍历骨干自身的模块，因此规范帧网络的 Linear/Conv2d 使用 **PyTorch 默认初始化**（kaiming_uniform(a=√5)），LN 的 γ 为 1。
- 训练与推理都走同一个 forward；Dropout 只在骨干 FC 中起作用。

### 4.4 SO(2) 变体（`resnet18_eq_frame_2vec`；它是发布的“RONIN + 50% data + SO(2)”权重所用的结构，依据该权重的 `config.json`）

- 预处理 `preprocess_eq_frame`（`ronin_resnet.py:71-78`）：`V ∈ R^{B×T×2×2}`，列为 `[a_xy, ω_xy]`，陀螺按普通矢量处理；`S ∈ R^{B×T×5} = [ω_z, a_z, ‖a_xy‖, ‖ω_xy‖, a_xy·ω_xy]`；`O = [ω_z, a_z]`。
- VNLinear 和 Convolutional 的输入先拼接 `[V, J·V]`，其中 `J = [[0,−1],[1,0]]`，`J·V` 的 x 分量为 −y、y 分量为 x（`orthogonal_input`，L12-13）。因此通道数翻倍，只对 SO(2) 等变。
- 超参数：hidden=128，depth=1，kernel=(32,1)，`dim_in=2`，`dim_out=2`，`scalar_dim_in=5`。规范帧网络的参数为：VNLinear0 (128,4)，Linear (128,5)，NL (256,256)+γ256；conv 向量分支 (128,256,32,1)、标量分支 (128,128,32,1)，NL (256,256)，VNLN 128，LN 128；VNLinear1 (128,256)，Linear (128,128)，NL1 (128,256)+γ128，VNLinear2 (128,256)，VNLN 128；输出 (2,256)。规范帧网络共 **1,821,312** 个参数，模型总量 **6,456,194**。
- 同样用 Gram–Schmidt 构造 F，所以 `det F` 可能为 −1，但网络只对旋转等变。
- 骨干输入通道：`V_c` 按行主序 reshape 后接上 `O`，得到 `[a'_x, ω'_x, a'_y, ω'_y, ω_z, a_z]`（实测误差为 0）。
- 仓库中另有 `resnet18_eq_frame`（`model_resnet1d_eq_frame.py`）：只输出 1 个向量 u，令 `F=[û, Jû]`，`det F` 恒为 +1，共 6,455,938 个参数。该变体没有发布权重，IPB 不采用。

## 5. 损失与训练配方（official）

| 项 | 值 | 来源 |
|---|---|---|
| 损失 | `nn.MSELoss()`，比较**旋回世界系后**的 2D 速度与目标 | `ronin_resnet.py:222,280` |
| 优化器 | Adam，默认 betas，无权重衰减 | `ronin_resnet.py:223` |
| 学习率/调度 | 1e-4（argparse 默认值，O(2) 权重的 `config.json` 也是 1e-4）；`ReduceLROnPlateau(factor=0.1, patience=10, eps=1e-12)`，每个 epoch 按验证集 MSE 调用一次 `step` | `ronin_resnet.py:224,306,506` |
| batch | 训练 128（`config.json`）；验证 DataLoader 为 512，shuffle=True | `ronin_resnet.py:198,507` |
| epoch | 预训练权重的 `config.json` 为 120。**但发布的代码循环写死为 `range(start_epoch, 1)`，实际只训练 1 个 epoch**（见 §10） | `ronin_resnet.py:263` |
| 权重衰减/梯度裁剪 | 无 / 无 | — |
| 增强 | 只有 `random_shift`，范围 ±`step_size//2`；等变架构不做随机水平旋转 | `ronin_resnet.py:152-154` |
| 阶段切换 | 无。SO(2) 权重的 `config.json` 为 `lr=1e-6`，并设 `continue_from=.../checkpoint_45.pt`，说明它是先正常训练、再以 1e-6 续训的两段式流程，论文没有交代 | 预训练 `config.json` |
| 选模 | 验证集平均 MSE 下降时保存 `checkpoint_<epoch>.pt`。发布的 O(2) 权重为 `checkpoint_38.pt`，SO(2) 为 `checkpoint_111.pt` | `ronin_resnet.py:310-317` |
| 数据 | 只用公开的 50% RoNIN 数据。`lists/list_train.txt` 有 71 条，比 RoNIN 官方列表少 `a007_3`；val、test 列表与 RoNIN 相同 | README；`lists/` |
| 随机种子 | 未设置 | — |

## 6. benchmark 适配（unified 配方）

| 改动 | 类型 | 理由 |
|---|---|---|
| `frame=gravity_world`，`window=200`，`dims=2`，`target=avg_velocity` | 不改变结构 | 与官方一致。IPB 的 `avg_velocity` 分母取 `(T−1)·dt`，官方取 200 个采样间隔（`p[s+200]−p[s]`），两者窗口跨度相差 0.5%，这是协议层差异 |
| 输入通道不置换；预处理（§2.1）放在模型 forward 内部实现 | 不改变结构 | IPB 通道顺序与官方特征一致；向量/标量的构造属于模型的一部分 |
| forward 返回 `{"vel": (B,2), "aux": {"frame": (B,2,2)}}` | 不改变结构 | 保留 frame，便于做等变性测试 |
| 训练 120 epoch，按验证集选模（IPB 的 fitness 为 val ATE） | 修复官方缺陷 | 发布的循环只跑 1 个 epoch，属于调试遗留；120 来自官方权重的 `config.json` |
| 不加 `random_yaw` 增强（加上也不影响结果） | 不改变结构 | 模型对 O(2) 严格等变，随机偏航只会带来浮点层面的差异 |
| 预处理使用与输入相同的 dtype | 数值细节 | 官方固定为 float32；在 float32 训练下两者等价 |
| 轨迹重建用窗口中心时间戳，指标用欧氏 RMSE | 协议层 | IPB §5、§6 的规定；官方重建方式和 ATE（除以 √2）仅用于复现对照 |
| 姿态：official 配方用 `orientation=device`；unified 按 IPB 默认 | 协议层 | 等变性只能消除常值偏航偏差，横滚/俯仰误差仍会影响结果 |

## 7. 官方报告数值

论文表 2（*Trajectory errors … with the RONIN architecture*），单位 m，均为纯网络结果，不经 EKF。ATE/RTE 采用 RoNIN 的定义（见 §3，按分量均值计算）。RONIN-U/S 为 unseen/seen 测试集，RIDI-T/C、OxIOD 为跨数据集评测；**仓库里没有 OxIOD 数据加载器**（只有 `data_ridi.py`）。

| 模型 | RONIN-U ATE | RONIN-U RTE | RONIN-S ATE | RONIN-S RTE | RIDI-T ATE | RIDI-T RTE | RIDI-C ATE | RIDI-C RTE | OxIOD ATE | OxIOD RTE |
|---|---|---|---|---|---|---|---|---|---|---|
| RoNIN（100% 数据，引自原论文） | 5.14 | 4.37 | 3.54 | 2.67 | 1.63 | 1.91 | 1.67 | 1.62 | 3.46 | 4.39 |
| + SO(2) Eq. Frame（50% 数据） | 5.18 | 4.35 | 3.67 | 2.72 | 0.86 | 1.59 | 0.63 | 1.39 | 1.22 | 2.39 |
| + O(2) Eq. Frame（50% 数据） | **4.42** | **3.95** | **3.32** | **2.66** | 0.82 | 1.52 | 0.70 | 1.41 | 1.28 | 2.10 |

表中还有带 † 的 50%-data 各行（+J、+TTT、+J+TTT），其 RONIN-U 的 ATE/RTE 依次为 5.57/4.38、5.02/4.23、5.05/4.14、5.07/4.17，RONIN-S 与 RIDI-C 两列为空。这些行应是引自 RIO 论文的对照结果（† 的含义未核实），见 `rio` 卡。表中 NDI 行的数值为数百米。以上数字取自 arXiv v3 HTML。论文没有给出 RoNIN 实验的 epoch、学习率、batch，本卡的对应数值来自权重的 `config.json`。

## 8. 忠实性测试建议

以下实测值均来自夹具 `equivariance_evidence`。输入为 8 条随机窗口，陀螺 σ=0.8，加计 σ=2，并在加计 z 上加 9.81；旋转角取 {0.7, −2.3} rad，反射轴取 {0.4, −1.1} rad。

- [ ] 参数量 5,230,466：规范帧网络 595,584，骨干 4,634,882。`param_shapes` 共 92 个（规范帧网络 23 个 + 骨干 69 个），按注册顺序逐项一致；buffer 共 72 个，其中 9 个是 LN 的 β，其余为 BN 统计量。SO(2) 变体为 6,456,194（规范帧网络 1,821,312），对应夹具 `variants.so2`。
- [ ] 输出形状：`vel (B,2)`，`frame (B,2,2)`，且 `frame·frameᵀ = I`、`det ∈ {±1}`。
- [ ] **旋转等变**：令 `x' = act(x, R(θ))`，其中 `a_xy ← R a_xy`、`ω_xy ← R ω_xy`，z 分量不变。要求 `vel(x') = R·vel(x)`、`frame(x') = frame(x)·Rᵀ`，且骨干输入 `X_c` 不变。float64 下误差 ≤ 1e-9（随机初始化实测 ≤ 7e-13，官方权重实测 ≤ 4e-14）；float32 下 ≤ 1e-4。
- [ ] **反射等变（仅 O(2)）**：`Q = R(φ)·diag(1,−1)·R(φ)ᵀ`，`a_xy ← Q a_xy`，`ω_xy ← −Q ω_xy`，`ω_z ← −ω_z`（赝矢量规律 `ω' = det(Q)·diag(Q,1)·ω`）。要求 `vel' = Q·vel`，且 `X_c` 不变。float64 下误差 ≤ 1e-9（实测 ≤ 1e-12）。
- [ ] **负向测试**：反射时把陀螺当普通矢量（`ω_xy ← Q ω_xy`，`ω_z` 不变），误差必须明显大于 0（实测：随机初始化 0.94，官方权重 0.10）。SO(2) 变体在正确的反射下同样**不应**等变（实测 0.55）。
- [ ] 骨干输入通道：`X_c == [Fᵀa_xy, a_z, s·Fᵀω_xy, s·ω_z]`（O(2)，`s=det F`，实测 3e-15）；SO(2) 为 `[a'_x, ω'_x, a'_y, ω'_y, ω_z, a_z]`。
- [ ] `ω ≡ 0` 或 `ω_xy == 0` 的退化窗口必须输出有限值。
- [ ] 可选（本地有官方权重时）：用 `pretrained_golden` 中的解析输入（公式写在 `input_formula`，float64、严格加载），逐元素比对 `vel` 与 `frame`，容差 1e-4；并核对 sha256。
- [ ] 训练配方：损失为 MSE，不做随机水平旋转，epoch 数为 120。

## 9. 预训练权重

README 提供 Google Drive 下载，均为 zip 包：O(2) 的 id 为 `1tM92YZCj_j8jcDSjnwsII0ckivUNVhp4`（`Ronin_o2/checkpoint_38.pt` + `config.json`）；SO(2) 的 id 为 `1A2k2Bv-zRgWIt25xsF5HHtuR3OGfxXyr`（`Ronin_so2/checkpoint_111.pt`）。解压后可以用 `torch.load(weights_only=True)` 读取，并严格加载进官方类（已验证）。sha256 记录在夹具中。仓库没有许可，权重只能在本地做对照，不得入库或再分发。

## 10. 官方实现的坑与未决问题

1. **训练循环只跑 1 个 epoch**：`ronin_resnet.py:263` 写成 `for epoch in tqdm(range(start_epoch, 1)):#args.epochs`，`--epochs` 参数被忽略。发布的权重（checkpoint_38/111，config 中 epochs=120）显然不是用这份代码训练出来的。
2. **规范帧向量的构造与论文不一致**（§2.1）：代码中 `v1 = ω×w1`，`v2 = ω×v1 ∝ −w1`。复现时以代码为准。
3. **论文正文的超参数与代码不一致**：论文写 SO(2) 为“1 conv block、16×1 核”，但论文给出的参数量（8,884,870）只与代码中 TLIO 版的 depth=2、kernel=(32,1) 吻合。RoNIN 版代码用的是 kernel=(32,1)（O(2) 为 depth 2，SO(2) 为 depth 1）。
4. **骨干输入通道被重排**：O(2) 为加计在前，SO(2) 为交错排列，所以不能直接载入 RoNIN 原版骨干的权重。
5. SO(2) 发布版使用双向量 Gram–Schmidt，F 可能是反射，但网络只对旋转等变。
6. 预处理的退化分支用浮点严格相等判断 `ω_xy == 0`，外加若干 1e-6/1e-7/1e-8 截断，因此在范数接近 0 时等变性只是近似成立。
7. 预处理张量固定为 float32（`torch.zeros` 使用默认 dtype）。
8. 仓库只用 50% 公开数据训练，训练列表还少了 `a007_3`；表 2 中的 RoNIN 基线是 100% 数据，对比口径不一致。
9. 继承自 RoNIN 的评测口径：轨迹时间戳落在窗口起点，ATE/RTE 按分量均值计算（除以 √2）。IPB 应使用自己的评测协议。
10. SO(2) 权重采用两段式训练（lr 1e-6 续训），论文未说明。代码没有设置随机种子。
11. 许可风险：仓库无 LICENSE，RoNIN 部分还派生自 GPL-3 代码。
