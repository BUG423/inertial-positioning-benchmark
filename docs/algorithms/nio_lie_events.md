# NIO from Lie Events（`nio_lie_events_ronin` / `nio_lie_events_tlio`）

> 规格卡版本：v1（2026-09-17）· fidelity：`official-code` · 夹具：`tests/fixtures/algorithms/nio_lie_events_ronin.json`、`tests/fixtures/algorithms/nio_lie_events_tlio.json`
>
> **为什么两个注册名共用一张卡**：本方法的全部贡献在于输入表示，即 SE(3) 李群事件及其事件堆叠（§2.2）。两个变体的这部分逐行相同，差别只在骨干（RoNIN-ResNet18 或 TLIO-ResNet）、阈值和训练配方。拆成两张卡会把 §2.2 重复一遍，容易出现两份描述不一致。骨干的逐层细节分别见 `ronin_resnet18` 卡 §4 与 `tlio` 卡 §4；这里的骨干只在第一层输入通道由 6 改为 12。

## 1. 元信息

| 项 | 值 |
|---|---|
| 论文 | *Neural Inertial Odometry from Lie Events*；Royina Karegoudra Jayanth, Yinshuang Xu, Evangelos Chatzipantazis, Kostas Daniilidis, Daniel Gehrig；RSS 2025；<https://www.roboticsproceedings.org/rss21/p143.pdf>，arXiv <https://arxiv.org/abs/2505.09780> |
| 官方仓库 | <https://github.com/RoyinaJayanth/NIO_Lie_Events> @ `1e87470f6479c198875c5f1a32b392a46eb5af74`（本地：`/workspace/webCodex/third_party/NIO_Lie_Events`，子目录为 `General/`、`RONIN/source_ev/`、`TLIO/src/`） |
| 许可 | **仓库无 LICENSE**，默认保留全部权利。RONIN 部分派生自 GPL-3 的 RoNIN 代码，TLIO 部分派生自 BSD-3 的 TLIO 代码，均未重新声明许可。风险：只能独立实现，权重不得再分发 |
| 框架 | PyTorch 2.2.2（`environment.yml`）+ numba（事件生成用 `@jit(nopython=True)`）+ numpy-quaternion + scipy |
| fidelity | official-code（已实例化两个骨干；已用官方函数在合成轨迹上运行测试期事件生成，并加载官方权重得到 golden 输出） |
| 参考文件 | `General/Event_generation.py`（事件生成参考实现）；`RONIN/source_ev/ronin_resnet.py`（测试期事件生成 `preprocess_event_gen` L558-616、训练循环、测试循环）；`RONIN/source_ev/data_glob_speed.py`（训练期事件缓存 L493-627、样本 L697-775）；`RONIN/source_ev/data_glob_speed_test.py`（测试集）；`TLIO/src/dataloader/sequences_dataset.py`（TLIO 事件路径 L1124-1238、局部重力系 L1339-1371）；`TLIO/src/dataloader/data_transform.py`；`TLIO/src/network/{model_factory,model_resnet,losses,train,test}.py`；`TLIO/src/main_net.py` |

## 2. 任务（输入）

| 项 | RoNIN 变体（`nio_lie_events_ronin`） | TLIO 变体（`nio_lie_events_tlio`） | IPB 映射 |
|---|---|---|---|
| 采样率 | 200 Hz | 200 Hz（`imu_freq=interp_freq=imu_channel_freq=200`，此时不做子采样） | 200 Hz |
| 窗口 | 200 样本（1 s） | 1 s = 200 样本 | `window=200` |
| 训练步长 | 事件特征按**每 10 样本一个窗口**预先计算并缓存（`data_glob_speed.py:542,549`，写死为 10，`--step_size` 不生效），每个窗口作为一个样本，没有随机偏移（L739-741 已被注释） | `decimator=10`，训练集另加 `randint(0,10)` 偏移 | `stride=10` |
| 推理步长 | 10，从第 200 个样本开始（`data_glob_speed_test.py:139`），batch=1，**按时间顺序**推理 | 10（20 Hz） | `eval_stride=10` |
| 坐标系 | 重力对齐世界系（RoNIN HACF） | 事件与 12 个通道先在世界系中生成，再整体左乘 `R_z(ψ0)ᵀ`，其中 ψ0 是**窗口起点**的偏航（`sequences_dataset.py:1340-1365`） | RoNIN 变体：`frame=gravity_world`。TLIO 变体：`gravity_yaw_local`（官方用起点偏航，IPB 用末端偏航，见 §6） |
| 姿态来源 | 训练与测试都用 `grv_only=False`、`max_ori_error=20`：game RV 误差小于 20° 时用它，否则取误差最小的源（可能是 Tango/EKF 姿态，`ronin_resnet.py:666`） | VIO 真值姿态 | 事件生成需要**逐样本姿态**，见 §2.3 |
| 去重力 | 否 | 否 | `remove_gravity=false` |
| 通道 | 原始 IMU 为 `[gyro, acc]`（世界系）。网络输入为 12 通道事件堆叠：`[E_ω(3), E_a(3), E_ρ(3), E_φ(3)]`（§2.2） | 同左 | 由 IPB 的 `[gyro, acc]` 在模型内部生成 |
| 额外输入 | **每个样本的姿态 `R_t`**、**窗口起点速度 `v0`**、窗口起点位置 `p0`（只作参考点，不影响结果）、样本时间戳 | 同左；测试时的 `v0` 来源见 §3 | 需要扩展 IPB 接口（§6） |
| 阈值 θ | 0.1（README，权重 `config.json`） | 0.01（README，权重 args） | 模型超参数 |

### 2.1 符号约定

- se(3) 切向量采用 Sophus 顺序 `ξ = [ρ(3), φ(3)]`。`Exp(ξ) = (R = exp(φ), t = J_l(φ)·ρ)`；`Log(T) = (φ = log R, ρ = J_l(φ)⁻¹·t)`。`‖ξ‖` 直接取 6 维欧氏范数，**平移（m）与旋转（rad）之间不加权**。
- `exp(φ)` 用 Rodrigues 公式；`‖φ‖ < 1e-10` 时取 `I + [φ]×`。`log(R)` 中：`x = clip((tr R − 1)/2, −1, 1)`，`θ = arccos x`；**若 `|θ − π| < 1e-3`，直接返回 0 向量**；否则 `φ = vee(R − Rᵀ)·0.5 / sinc(θ)`，其中 `|θ| < 1e-3` 时 `sinc(θ)` 取 1。
- `J_l(φ)`：`θ < 1e-5` 时为 `I + ½Φ`；否则为 `I + (1−cosθ)/θ²·Φ + (θ−sinθ)/θ³·Φ²`，其中 `Φ = [φ]×`。
- `J_l⁻¹(φ)`：`θ < 1e-5` 时为 `I − ½Φ + Φ²/12`；否则为 `I − ½Φ + (1 − ½θ·cos(θ/2)/sin(θ/2))/θ²·Φ²`。
- 以上定义见 `General/Event_generation.py:8-123,190-352`，RONIN 与 TLIO 两处代码的拷贝与之逐行一致。

### 2.2 SE(3) 李群事件与事件堆叠（逐步算法）

输入为一个窗口：时间戳 `t_0..t_{N−1}`（N=200），世界系 IMU `ω^w_i`、`a^w_i`（即 IPB 输入），逐样本姿态 `R_i`，起点位置 `p0`、起点速度 `v0`，以及阈值 θ。

1. **机体系量测**：`ω^b_i = R_iᵀ ω^w_i`，`a^b_i = R_iᵀ a^w_i`。RoNIN 测试期用四元数共轭实现（`ronin_resnet.py:562-567`）；RoNIN 训练期直接使用已标定的机体系原始量测（`data_glob_speed.py:554-555`）；TLIO 用 `einsum('tji,tj->ti', R, ·)`。
2. **预积分**（`_preintegrate_measurement`）。初始值 `R_k = R_0`，`p_k = p0`，`v_k = v0`，`T_k = T_ref = (R_0, p0)`。对 i = 1..N−1，令 `Δ = t_i − t_{i−1}`，并使用**第 i 个样本**的量测（右端矩形法）：
   `R' = R_k·exp(ω^b_i Δ)`；`a_w = R_k a^b_i`；`v' = v_k + a_w Δ + g Δ`；`p' = p_k + v_k Δ + ½ a_w Δ² + ½ g Δ²`，其中 `g = (0,0,−9.81)`。积分用的是**积分出来的** `R_k`，不是外部给定的 `R_i`。
3. **测地线水平穿越**（`_geodesic_events_se3_vectorized`）。记 `T_0 = T_k`，`T_1 = (R', p')`：
   - `d01 = Log(T_0⁻¹T_1)`，`n01 = d01/‖d01‖`；
   - `w = Log(T_ref⁻¹T_1)`，`u = w/‖w‖`（6 维单位“极性”），`n = ⌊‖w‖/θ⌋`；
   - 对 j = 1..n：`T_ref,j = T_ref·Exp(j·θ·u)`；`β_j = Log(T_0⁻¹T_ref,j)·n01 / ‖d01‖`；事件时间 `τ_j = t_{i−1} + β_j·(t_i − t_{i−1})`；
   - 事件量测：`ω^w(τ_j) = ω^w_{i−1} + (ω^w_i − ω^w_{i−1})·β_j`，`a^w(τ_j)` 同理，均为线性插值，`β` 不做截断；
   - 事件极性：`p_j = [R_ref,(j−1)·u_ρ, R_ref,(j−1)·u_φ]`，即**同一个** `u` 分成两段、分别用**上一个**参考位姿的旋转转到世界系（j=1 时用 `T_ref` 自身的旋转）；
   - 更新 `T_ref ← T_ref·Exp(n·θ·u)`，然后 `R_k, p_k, v_k, T_k ← R', p', v', T_1`。
   - 每个事件是一个 13 维向量 `[τ, ω^w(3), a^w(3), p(6)]`。
4. **首尾伪事件**（RoNIN，`ronin_resnet.py:574-602`）。新建两条事件：
   - 时间戳分别为 `t_0` 与最后一个 `t_k`；
   - 量测分别取**第一个与最后一个样本的世界系 IMU**；
   - 极性分别取：按时间排序后的前两个事件中的第一个事件的极性，以及排序后最后两个事件中的最后一个事件的极性；
   - 若没有任何事件，两条伪事件的极性都取 `R_0` 旋转后的 `Log(T_0⁻¹ T_final)/‖·‖`。

   TLIO 变体（`sequences_dataset.py:1186-1213`）的做法相同，**但伪事件的量测保持为 0**。
5. 把全部事件（含伪事件）按 τ 排序（`argsort`，非稳定排序）。TLIO 变体还会丢弃 `τ < t_0` 的事件。记事件总数为 M。
6. **按序号分桶**（`generate_event_stack`，B=200）：第 j 个事件（从 0 起）落入桶 `b_j = int(linspace(0, B−1, M)[j])`，即 `⌊j·(B−1)/(M−1)⌋`。分桶**只看事件的序号，不看时间戳**，与论文式 (26)(27) 一致。
   - `E_{ω,a}[b]` = 该桶内所有事件 `[ω^w, a^w]` 的均值，空桶为 0。**代码中的通道顺序为陀螺在前**；论文写作 `[â‖ω̂]`，即加计在前，**以代码为准**。
   - `Ē_p[b]` = 该桶内极性的均值；`E_p[b] = Ē_p[b] / (‖Ē_p[b]‖ + 1e-4)`，即 6 维联合归一化，空桶为 0。
7. 输出 `X = cat(E_{ω,a}, E_p)`，转为 `(12, 200)` 的 float32 张量，通道顺序为 `[ω_x, ω_y, ω_z, a_x, a_y, a_z, ρ̂_x, ρ̂_y, ρ̂_z, φ̂_x, φ̂_y, φ̂_z]`（世界系）。
8. 仅 TLIO：把 `X` 的四个三元组分别左乘 `R_z(ψ0)ᵀ`（`sequences_dataset.py:1360-1365`）。

**数值事实**（夹具 `event_generation_evidence`；合成轨迹的定义见夹具 `synthetic_trajectory`，量级约为 1.2 m/s、0.5 rad/s）：θ=0.1 时 1 s 内只有 14 个真实事件，200 个桶中 **184 个为空**，输入非常稀疏；θ=0.01 时有 140 个事件，58 个桶为空。

### 2.3 事件表示的性质（可测试）

- **对全局偏航等变**：把世界系 IMU、姿态、`v0` 一起左乘 `R_z(ψ)`，事件堆叠满足 `X' = diag(R_z, R_z, R_z, R_z)·X`（实测误差 5.4e-8，float64）。原因是事件时间、极性方向和 `‖Log‖` 都只依赖相对位姿，而极性和量测已被转到世界系。严格来说，这对任意 `R ∈ SO(3)` 都成立，但重力 g 固定在 z 轴上，所以实际上只对绕 z 轴的旋转成立。
- **对 `p0` 不变**：`p0` 只是参考点。
- **对采样率近似不变**（论文定理 1：在连续时间极限下，事件时间随时间重参数化协变，极性不变）。离散实现只能近似满足：同一轨迹降采样到 100 Hz 后，事件数由 14 变为 13，极性通道最大差 0.92。因为按序号分桶，事件数只要差一个，桶的对齐就会整体错位。
- **不是**等变网络：骨干是普通 ResNet，因此输出不具备 O(2) 等变性。TLIO 变体靠偏航增强近似获得不变性。

## 3. 输出

| 变体 | 输出 | 物理量 | 不确定度 | 官方轨迹 |
|---|---|---|---|---|
| RoNIN | `(B,2)` | 世界系水平平均速度 `(p[s+200]−p[s])/Δt`（`glob_v[ind]`，`data_glob_speed.py:519-520,613,624`） | 无 | `recon_traj_with_preds`（`ronin_resnet.py:863-877`）：时间轴与参考轨迹都从第 200 个样本开始，速度乘以 `dts` 后累加，时间戳落在窗口起点，再插值。ATE/RTE 口径与 RoNIN 相同（按分量均值） |
| TLIO | `(B,3)` 位移 + `(B,3)` 对数标准差 | 窗口起点局部重力系中的 `p[199]−p[0]` | 对角 | 与 TLIO 相同（`test.py:pose_integrate`，除以 `window_time`，累加），EKF 路径另行处理 |

**起点速度 `v0` 的来源**（这一点决定了评测是否公平）：
- RoNIN **训练**：`v0 = glob_v[ind−200]`，即**前一个 1 s 窗口**的真值平均速度，再对每行加 `U(−0.5, 0.5)` 噪声，并令 `v0_z = 0`（`data_glob_speed.py:529-534,553`）。
- RoNIN **测试**：第 1 个窗口 `v0_xy` 取真值 `targets[frame_id−200]`；之后的窗口 `v0_xy` 取**上一个窗口的网络预测**（窗口起点相差 10 个样本），`v0_z` 始终为 0（`ronin_resnet.py:625-633`）。这与论文 §IV-C 的“use the last velocity prediction”一致，是一个有状态的递归推理过程。
- TLIO **训练**：真值 `vel[0]` 加逐行 `U(−0.5,0.5)` 噪声（`add_vel_perturb`，`sequences_dataset.py:1132-1136`）。
- TLIO **纯网络测试**（`test.py`）：直接用**真值 `vel[0]`**，不加噪声，属于特权信息。论文称测试期用 EKF 状态，但该路径只存在于 EKF 中。

## 4. 网络结构

两个变体的骨干都来自官方工厂函数，结构与原版一致，只是第一层卷积的输入通道由 6 变为 12，多出 `6·64·7 = 2,688` 个参数。

| 变体 | 构造 | 逐层 | 参数量（官方 = benchmark） |
|---|---|---|---|
| RoNIN | `ResNet1D(12, 2, BasicBlock1D, [2,2,2,2], base_plane=64, kernel_size=3, FCOutputModule(fc_dim=512, in_dim=7, dropout=0.5, trans_planes=128))`（`ronin_resnet.py:31-32,541-544`；`model_resnet1d.py` 与 RoNIN 原版相同） | Conv k7 s2 p3 12→64 + BN + ReLU → (64,100)；MaxPool k3 s2 p1 → (64,50)；四个残差组 → (64,50)/(128,25)/(256,13)/(512,7)；transition Conv k1 512→128 + BN → (128,7)；flatten 896 → FC 512 + ReLU + Dropout 0.5 → FC 512 + ReLU + Dropout 0.5 → FC 2 | **4,637,570**（69 个参数张量，63 个 BN buffer） |
| TLIO | `model_factory.get_model('resnet', {'in_dim': 7}, input_dim=12, output_dim=3)` → `ResNet1D(BasicBlock1D, 12, 3, [2,2,2,2], 7)`，`cov_output_dim` 取默认值 3 | Conv k7 s2 p3 12→64 + BN + ReLU + MaxPool → (64,50)；四个残差组 → (512,7)；两个 FcBlock：prep Conv k1 512→128 + BN → flatten 896 → fc1 512 + ReLU + Dropout 0.5 → fc2 512 + ReLU + Dropout 0.5 → fc3 3 | **5,427,334**（78 个参数张量，66 个 BN buffer） |

两个权重文件都已用 `weights_only=True` 严格加载，参数量与上表一致。初始化与原版相同：Conv 用 kaiming_normal(fan_out)，BN 为 1/0，Linear 为 N(0,0.01)/0。

## 5. 损失与训练配方（official）

| 项 | RoNIN 变体 | TLIO 变体 |
|---|---|---|
| 损失 | `nn.MSELoss()`（2D） | 第 1–9 个 epoch：MSE（logstd 被 detach）；第 10 个 epoch 起：对角 NLL，`log σ ≥ log 1e-3`（`losses.py:120-137`，与 TLIO 相同，另加一个 `_only_mse` 开关） |
| 优化器 / 学习率 | Adam 1e-4；`ReduceLROnPlateau(0.1, patience 10, eps 1e-12)`，按验证集 MSE 调用 `step`（`ronin_resnet.py:742-743,819`） | Adam 1e-4；调度器已构造但**从未 step**，学习率恒定 |
| batch | 128（验证 512） | 1024 |
| epoch | 120（README 与 config；权重为 `checkpoint_last.pt`，第 99 个 epoch，由续训得到） | `--epochs 50`，循环为 `range(start+1, 50)`，共 49 个；最佳权重来自第 43 个 epoch。论文称“先训练 10 个 epoch，再训练 40 个 epoch” |
| 梯度 | 每步 `zero_grad`；不裁剪 | 每步 `zero_grad`（`train.py:110`）；`clip_grad_norm_ 0.1` |
| 随机种子 | `seed_everything(42)`，cudnn 设为 deterministic | 未设置 |
| 增强 | ① `v0` 噪声 `U(−0.5,0.5)`，逐行、仅训练集，**在缓存时固定**，不会每个 epoch 重新采样。② 极性噪声：缓存阶段对每个窗口的所有事件加同一个 6 维 `U(−0.5,0.5)` 向量（L599-601，**不分 split**，验证集的缓存同样带噪）；`__getitem__` 中训练集再对非零项加一次 `U(−0.5,0.5)`，然后重新按桶归一化（L748-763）。③ 不做水平旋转，也不做时间偏移 | 按 README 与权重 args：① 数据加载器在事件生成前加偏置：陀螺 `(N(0,1)−0.5)·0.1`，加计 `(N(0,1)−0.5)·0.4`，**用的是高斯分布并带负均值**（`sequences_dataset.py:1013-1026`）。② `TransformAddNoiseBias` 对事件堆叠的通道 0–5 加均匀偏置（±0.05 / ±0.2）。③ `TransformPerturbGravity` 做 5° 倾斜，同时旋转全部 4 个三元组（`data_transform.py:133-138`）。④ `TransformInYawPlane` 做随机偏航，旋转 4 个三元组和目标（L353-365）。⑤ `v0` 噪声 ±0.5。⑥ 极性噪声 ±0.5（L1233-1235，**不分 split**）。当 `perturb_gravity=False` 时，改为在数据加载器中、事件生成前做重力扰动（L1027-1059） |
| 选模 | 验证集 MSE 下降时保存；另每 20 个 epoch 保存一次，结束时保存 `checkpoint_last.pt`；**README 测试用的是 last** | `checkpoint_best.pt`，依据验证集损失均值（MSE 与 NLL 混合比较） |
| 数据 | 50% 公开 RoNIN 数据；`data_split_percentage` 默认 1.0；训练列表比 RoNIN 原版少 `a007_3` | TLIO golden 数据集；Aria 用于跨数据集评测 |

## 6. benchmark 适配（unified 配方）

| 改动 | 类型 | 理由 |
|---|---|---|
| 事件生成（§2.2）放在模型包装内部，输入为 IPB `(B,6,200)` 世界系 IMU，输出为 `(B,12,200)` 张量，再送入骨干 | 不改变结构 | 这是方法本身；骨干参数量由夹具锁定 |
| **接口扩展**：forward 额外接收 `orientation (B,T,4)`（与视图旋转所用的姿态相同）、`v0 (B,3)` 与时间戳 | 改变接口 | 预积分需要机体系量测和起点速度，仅凭 IPB 的 6 通道张量无法重建。`p0` 不影响结果，可以取 0 |
| `v0` 协议：训练取前一个窗口的真值平均速度加 `U(±0.5)`、`v_z=0`，并在**每个 epoch 重新采样**（官方为缓存固定）；推理时第 1 个窗口取真值 `v0`（与 IPB 用真值锚定起点位置属于同一类初始条件），之后取上一个窗口的预测 | 改变推理协议（有状态预测器） | 与官方 RoNIN 测试一致。IPB 的 predictor 需要支持按时间顺序、batch=1（或按序列分组）的递归推理。**禁止**在每个窗口都使用真值 `v0`（这是官方 TLIO 纯网络测试的特权做法） |
| RoNIN 变体：`frame=gravity_world`，`dims=2`，`target=avg_velocity` | 不改变结构 | 与官方一致 |
| TLIO 变体：`dims=3`，`target=displacement` 或 `avg_velocity`（分母为 0.995 s）；偏航参考点使用窗口**起点**（需要为视图增加 `yaw_ref=start` 选项），或接受 IPB 的末端偏航 | 协议层 | 骨干不等变，所以参考点的选择会影响结果；官方训练带偏航增强，可以缓解这一差异 |
| TLIO 伪事件量测：official 配方保持为 0；unified 配方可以与 RoNIN 统一为首/末样本的值 | 配方层 | 两份官方实现不一致 |
| 修正 `generate_event_stack` 调用处的签名错误（§10-1），语义按函数体实现 | 修复官方缺陷 | 发布的代码按原样运行会抛出 TypeError |
| 偏置增强使用均匀分布 `U(±r)` | 修复官方缺陷 | 官方误用了 `randn − 0.5` |
| 极性噪声只加在训练集上 | 协议层 | 验证集必须确定 |

## 7. 官方报告数值

以下数值均从 arXiv v1 HTML 抽取；表 II、表 III 的完整列结构**未完全核实**。论文把 ATE\* 的单位写作 mm，与 EqNIO 的写法相同，疑为 m。

- 表 II（TLIO 数据集与 Aria Everyday）：

  | 模型 | Rate Aug. | TLIO MSE\* | TLIO ATE\* | Aria-Right ATE\* | Aria-Left ATE\* | TLIO ATE（+EKF） |
  |---|---|---|---|---|---|---|
  | TLIO | ✗ | 0.013 | 1.660 | 1.240 | 1.314 | 1.410 |
  | + interp. | ✓ | 0.013 | 1.519 | 1.195 | 1.326 | 1.410 |
  | + splat. | ✓ | 0.014 | 1.498 | 1.216 | 1.292 | 1.440 |
  | + events（ours） | ✗ | 0.015 | 1.445 | 1.099 | 1.064 | 1.282 |

- 表 III（RoNIN 架构，单位 m，均为纯网络结果）：RoNIN + events（50% 数据）在 RoNIN-U 上 ATE\* 为 5.35、RTE\* 为 4.63；RIDI-T 上 ATE\* 为 1.03；OxIOD 上 ATE\* 为 1.52。对照组 RoNIN-ResNet（100% 数据）分别为 5.14、1.63、3.46。其余单元未核实。
- 表 IV（TLIO 消融，列为 MSE\*/ATE\*/ATE+EKF）：无事件 0.013/1.660/1.410；有事件无极性 0.014/1.515/1.724；S³×R³ 流形 0.015/1.421/1.622；SE(3) 无噪声 0.005/0.912/6.080；SE(3) 带噪声 0.015/1.445/1.282。
- 表 I：事件时间戳与规范时间戳之间的 Chamfer 距离（%）。
- 图 4：TLIO 在不同 IMU 频率下的敏感性，只有曲线，具体数值未核实。
- 超参数（论文 §IV）：TLIO 为 lr 1e-4、Adam、batch 1024、先 10 个 epoch MSE 再 40 个 epoch，θ=0.01；RoNIN 为 batch 128、lr 1e-4、最多 120 个 epoch，θ=0.1。

## 8. 忠实性测试建议

- [ ] 参数量：RoNIN 变体 4,637,570，TLIO 变体 5,427,334；`param_shapes` 逐项一致；第一层卷积权重形状为 `(64, 12, 7)`。
- [ ] 输出形状：RoNIN `(B,2)`；TLIO `(B,3)`+`(B,3)`。
- [ ] SE(3) 工具函数：`Exp(Log(T)) = T`（θ 远离 π 时）；`|θ−π| < 1e-3` 时 `log` 返回 0，这是官方行为，需要单测覆盖；`J_l·J_l⁻¹ = I`。
- [ ] 事件生成 golden：用夹具 `synthetic_trajectory` 给出的合成轨迹，θ=0.1 时应得到 14 个事件、184 个空桶，θ=0.01 时应得到 140 个事件、58 个空桶；`channel_sums`、`bin0`、`bin199`、`nonempty_bins_sparse` 逐值比对（容差 1e-6）；`final_preintegrated_pose_T` 也要比对。
- [ ] 分桶：M 个事件对应的桶号为 `int(linspace(0,199,M))`；空桶输出 0；每个非空桶的极性范数落在 `(0, 1)` 内，且接近 1。
- [ ] 偏航等变：同时旋转 IMU、姿态和 `v0` 后，`X' = diag(R_z×4)·X`，float64 下误差 ≤ 1e-6（实测 5.4e-8）。
- [ ] `p0` 平移不变：改变 `p0` 后，事件堆叠不变。
- [ ] 静止窗口：当 `T_1 == T_ref`（`‖w‖=0`）时，官方实现会产生 NaN（§10-5）。IPB 实现必须给出有限输出，并用单测锁定这一处修正。
- [ ] 推理协议：第 k 个窗口的 `v0_xy` 等于第 k−1 个窗口的预测，`v0_z = 0`。
- [ ] TLIO 损失：epoch 9 为 MSE，epoch 10 为对角 NLL（夹具 `loss_evidence` 实测误差为 0）。
- [ ] 可选：加载官方权重，把 golden 事件堆叠送入网络，比对 `pretrained_golden` 中的 `vel` / `disp` / `logstd`，容差 1e-4。

## 9. 预训练权重

README 给出以下 Google Drive 下载（zip）：
- TLIO：SE(3)+极性 `1nEW9bckdB9OekSdmxMuxHSaNUlUTMg2B`（`tlio_ev_se3p/checkpoint_best.pt`，第 43 个 epoch，附 `parameters.json`，已下载核对）；SE(3) `1H8dTUSSm_CFn0z8FT9tWo0Q_3gS5Z0gB`；SO(3)+R(3) `1FSwg8FbjLQqRcV7QIYym98RChQqyrpit`；R(3) `1kVy7CfUpc8WIRJ3I8r7feq3lHyHvZwbV`。后三个未下载。
- RoNIN：SE(3)+极性 `1eP3H9Ch7J6lKcLrQ6sPEGCMwSN1E3iA-`（`ronin_ev_se3p/checkpoint_last.pt`，第 99 个 epoch，`config.json` 中 θ=0.1，已核对）。

两个权重的 sha256 记录在夹具中。权重中保存的 args 含有大量发布代码里不存在的开关（如 `incl_first_event`、`ind_polarity_noise`、`scale_invariant_input`），说明训练所用代码与发布代码不是同一版本。仓库无许可，权重只作本地对照。

## 10. 官方实现的坑与未决问题

1. **训练期事件堆叠的调用签名错误，代码无法直接运行**：
   - `data_glob_speed.py:605` 调用 `generate_event_stack(events, ts[...], 200, se3=True, start_idx=1)`，报 `TypeError: got multiple values for argument 'se3'`；
   - `sequences_dataset.py:1224/1236/1256/1304` 调用 `self.generate_event_stack(events, ts, window_size=...)`，报 `TypeError: ... 'window_size'`。

   两处均已实测。这些调用残留了旧签名（带 `ts` 参数），而函数体只按序号分桶、不使用时间戳。本卡按函数体的语义描述。
2. **TLIO 纯网络测试使用真值 `v0`**：论文说测试期用 EKF 状态，但带 \* 的指标实际走的是 `test.py`，用的是真值。RoNIN 变体则是递归使用上一个预测。
3. **RoNIN 训练与测试的 `v0` 语义不一致**：训练取前 1 s 的真值平均速度（加噪声），测试取上一个窗口的预测，两个窗口只差 10 个样本、在时间上重叠。另外 `v0[ind−200:ind]` 在 `ind < 200` 时会发生负索引回绕，得到空切片后报错；能否正常运行取决于序列的 `start_frame` 是否不小于 200（未核实）。
4. **RoNIN 训练缓存的目标可能错位**：每个窗口的特征固定为 200 行，而 `glob_v[ind:ind+200]` 在序列末尾会短于 200 行，所以拼接后序列最后约 20 个窗口的目标与特征错位。
5. **静止或零位移时产生 NaN**：`‖w‖ = 0` 时 `u = w/0`，而 `Exp(0·NaN)` 会把 `T_ref` 污染成 NaN，之后的事件全部失效。`|θ−π| < 1e-3` 时 `log` 返回 0，也是不连续的。
6. **TLIO 首尾伪事件的量测为 0**（RoNIN 用的是实测值），会把首桶和末桶的 IMU 均值拉向 0。
7. **TLIO 数据加载器的偏置增强用了 `randn − 0.5`**：分布是高斯而非均匀，而且均值为负（陀螺 −0.05 rad/s、加计 −0.2 m/s²）。启用 `do_bias_shift` 时还会与变换层的均匀偏置叠加。
8. **极性噪声不分 split**，验证集因此带噪；RoNIN 的训练缓存中，`v0` 噪声和第一次极性噪声都是固定的。
9. **论文与代码的通道顺序相反**：论文写 `E_{a,ω}` 为 `[â‖ω̂]`，代码为 `[ω, a]`。
10. **按序号分桶对事件数非常敏感**：采样率变化时往往只差一两个事件，但桶的对齐会整体平移，理论上的采样率不变性在离散实现中不成立（实测极性差 0.92）。
11. **θ=0.1 时 RoNIN 的输入极度稀疏**：一个窗口约 16 个非空桶，其余 184 个为 0。
12. **`main_net.py` 的默认值不适合训练**（`epochs=3`、`mode=test`）。README 训练命令里的 `-- polarity_input` 多了一个空格，argparse 会报错。RoNIN 的 README 测试命令用的是 `checkpoint_last`。
13. TLIO 的学习率调度器从未 step；epoch 实际为 49 个；选模时 MSE 与 NLL 混合比较（与 `eqnio_tlio`、`tlio` 相同）。前 10 个 epoch 用真正的 MSE，而原版 TLIO 用的是 logstd 被 detach 的 NLL。
14. RoNIN 测试集的姿态源是 `grv_only=False`，可能用到 Tango/EKF 姿态，与 RoNIN 原版测试只用 game RV 的做法不同。
15. 许可风险：仓库无 LICENSE。
