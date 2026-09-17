# LLIO / LLIO-Net（`llio`）

> 规格卡版本：v1（2026-09-17）· fidelity：`official-code`（仅网络结构；训练与数据管线为论文/TLIO 推断）· 夹具：`tests/fixtures/algorithms/llio.json`

## 1. 元信息

| 项 | 值 |
|---|---|
| 论文 | *LLIO: Lightweight Learned Inertial Odometer*；Yan Wang, Jian Kuang, Xiaoji Niu, Jingnan Liu；IEEE Internet of Things Journal, vol. 10, no. 3, pp. 2508–2518（2022 年在线发表，2023 年卷期）；DOI [10.1109/JIOT.2022.3214087](https://doi.org/10.1109/JIOT.2022.3214087)；预印本 TechRxiv [10.36227/techrxiv.16408383.v1](https://doi.org/10.36227/techrxiv.16408383.v1)（2021-08-24） |
| 官方仓库 | <https://github.com/i2Nav-WHU/LightweightLearnedInertialOdometer> @ `e225ab8f3080c35c92481673b6af1d68692941f6`（本地：`/workspace/webCodex/third_party/LightweightLearnedInertialOdometer`） |
| 许可 | GPL-3.0（README 与 `LICENSE`）。IPB 的实现是按本卡重写的洁净室实现，不复制代码；如果日后要直接引入官方代码，整个衍生模块都要按 GPL-3.0 分发 |
| 框架 | PyTorch + einops（`Rearrange`/`Reduce` 层） |
| fidelity | official-code（网络）。仓库**只有两个模型文件**，没有数据加载、训练、EKF、评测代码，也没有权重 |
| 参考文件 | `model_twolayer.py`（`TwoLayerModel`、`ResMLPExtractor`、`PoolingMLPReg`、`SimpleMLPReg`）；`model_MLP.py`（`Affine`、`PreAffinePostLayerScale`、`ResMLP`、`MLPExtractor`、`MLPReg`、`MLPCombineNet`）；`README.md`（官方配置） |

**哪个模型是 LLIO**：README 明确写着 “The LLIO contained in the model_twolayer.py”，并给出配置（`model_twolayer.py:244-262` 与 README 相同）：`input_len=100, input_channel=6, patch_len=25, feature_dim=512, out_dim=3, GELU, extractor=ResMLP(layer_num=6, expansion=2, dropout=0.2), reg=MeanMLP(layer_num=3)`。注释说明这对应论文图 3 的 Feature Convert + ResMLP Module + Regression。因此 IPB 只注册一个名称 `llio`，即 `TwoLayerModel` + README 配置。`model_MLP.py::MLPCombineNet` 是仓库中的另一种纯 MLP 网络，README 没有把它称为 LLIO，而且默认参数下无法实例化（见 §10），所以不注册，只在 §10 附录中记录。

**论文全文可得性**：IEEE 版本需要订阅；TechRxiv、figshare 与 cloudfront 上的预印本 PDF 从本机访问时都返回 403 或连接被重置，Semantic Scholar 只能取到摘要。因此本卡中所有“论文”字段只来自摘要，其余一律标注“未核实”。摘要写道：TLIO 计算量大，不适合移动设备；作者设计了 LLIO-Net，并“By replacing the network in TLIO with the LLIO-Net”构成完整系统，精度相近，“inference efficiency … up to 12 times improved than that of TLIO”。据此，LLIO 的数据视图、损失和 EKF **继承 TLIO**（README 也致谢 TLIO），细节见 [`tlio.md`](tlio.md)。

## 2. 任务（输入）

| 项 | 官方 | IPB 映射 |
|---|---|---|
| 采样率 | README 注释：“1 seconds of imu output at 100Hz, we got [6 x 100] matrix”（`model_twolayer.py:266-269`）。**未核实**论文是否真用 100 Hz（TLIO 原生 200 Hz） | 200 Hz 输入，模型入口先做无参数 ×2 抽取（见 §6） |
| 窗口 | `input_len=100` 个样本（按 README 注释即 1 s） | `window=200`（1 s @200 Hz）→ 抽取后 100 |
| 训练步长 | 仓库无训练代码；TLIO 默认 `decimator=10`（200 Hz 下每 10 个样本一个窗口）。**未核实** | `stride=10` |
| 推理步长 | 继承 TLIO：网络 20 Hz（`main_net.py --sample_freq 20`）。**未核实** | `eval_stride=10`（200 Hz 下即 20 Hz） |
| 坐标系 | 继承 TLIO：局部重力对齐系；训练数据为世界系加 U[−π,π] 随机偏航（见 `tlio.md` §2） | `frame=gravity_yaw_local`（与 `tlio` 卡一致） |
| 姿态来源 | 继承 TLIO：训练用参考（VIO）姿态，EKF 中用滤波器姿态 | `orientation=reference` |
| 去重力 | 否（TLIO 输入是比力） | `remove_gravity=false` |
| 通道顺序 | 张量 `(B, 6, 100)`；仓库没有数据管线，按 TLIO 约定为 `[gyro_xyz, acc_xyz]`（TLIO `sequences_dataset.py:185` “gyro0, accelerometer0”） | 与 IPB `[gyro, acc]` 相同，不需要置换 |
| 额外输入 | 无 | 无 |
| 输入归一化 | 网络内无；数据侧继承 TLIO（当前 TLIO 实现中无缩放，见 `tlio.md`） | 无 |

## 3. 输出

- 官方：`forward` 返回二元组 `(y, y_cov)`，形状均为 `(B, out_dim)`，README 配置下 `out_dim=3`。`y` 是 3D 位移（TLIO 语义：窗口首尾位置差，单位 m，局部重力对齐系）。`y_cov` 由独立的线性层 `out2_linear` 输出，仓库**没有**对它施加任何变换或约束。按 TLIO 约定，它应解释为对角 `log σ`（TLIO `losses.py:25-42`：`pred_logstd` 在损失中被下限截断为 `log(1e-3)`，NLL = `(pred−targ)²/(2·exp(2·logstd)) + logstd`）。该语义在 LLIO 仓库内**无法核实**。
- 官方轨迹：继承 TLIO，把网络的位移与协方差作为 EKF 的量测（20 Hz），和 IMU 积分紧耦合，得到 3D 位姿。本仓库没有这部分代码。
- IPB：与 `tlio` 卡相同，保留 3 维输出（`dims=3`）。损失在位移尺度上计算；对外输出 `{"vel": (B,3), "logstd": (B,3)}`，`vel = disp / ((T−1)·dt) = disp / 0.995 s`，`logstd_vel = logstd − log(0.995)`；水平指标只用 xy。轨迹按 DESIGN §5 用窗口速度积分得到，不使用 EKF。

## 4. 网络结构

记号：`B` 为 batch，`L=4` 为 patch 数，`D=512` 为特征维；`A(·)` 是逐特征仿射 `x·g + b`，其中 `g,b` 形状为 `(1,1,D)`，初值分别为 1 和 0（`model_MLP.py:15-22`）；`s` 是 LayerScale 参数，形状 `(1,1,D)`，初值 0.1（`model_MLP.py:25-41`：`depth≤18` 时 `init_eps=0.1`，这里 `depth` 为层号 0…5）。
`PreAffinePostLayerScale(fn)(x) = A_out( x + s ⊙ fn(A_in(x)) )`：输入先做仿射，残差相加**之后**再做一次仿射（Post-Affine）。

逐层表（IPB 基准配置；输入 `(1,6,200)`；括号内为参数量）：

| # | 层 | 参数（核/步长/填充/通道） | 归一化 | 激活 | dropout | 输出形状 |
|---|---|---|---|---|---|---|
| 0 | 抽取（IPB 适配，无参数） | `avg_pool1d(kernel=2, stride=2)`，即相邻两样本求平均 | — | — | — | (1,6,100) |
| 1 | Feature Convert: Rearrange | `b c (l w) -> b l (w c)`，`w=patch_len=25`，**patch 内按时间优先展平**：第 l 个 token 的第 `τ·6+c` 维取 `x[:, c, 25l+τ]`（τ∈[0,25)，c∈[0,6)） | — | — | — | (1,4,150) |
| 2 | Feature Convert: Linear (77,312) | 150→512，带 bias，作用在最后一维 | — | — | — | (1,4,512) |
| 3.i.a | ResMLP 第 i 层，token-mixing 子层 (2,576)，i=0…5 | `PreAffinePostLayerScale(Conv1d(4,4,k=1,bias=False))`；Conv1d 把 **patch 轴（dim 1）当作通道**、特征轴当作长度，即对每个特征维做同一个 4×4 线性 patch 混合：`y[:,p,d]=Σ_q W[p,q]·x[:,q,d]` | 前/后仿射 A | — | — | (1,4,512) |
| 3.i.b | ResMLP 第 i 层，channel-mixing 子层 (1,051,136) | `PreAffinePostLayerScale(Seq(Linear(512→1024, bias=False), GELU, Dropout(p=0.2, inplace=True), Linear(1024→512, bias=False)))` | 前/后仿射 A | GELU | 0.2（隐层） | (1,4,512) |
| 4 | 末端 Affine (1,024) | `A(x)`，`g,b` 形状 (1,1,512) | — | — | — | (1,4,512) |
| 5 | Regression: Reduce | `b l f -> b f`，对 patch 轴求均值（`MeanMLP`；`MaxMLP` 为取最大值） | — | — | — | (1,512) |
| 6–8 | 3× [Linear(512→512, bias) → GELU → Dropout] (各 262,656) | 3 个块共用同一个 Dropout 实例 | — | GELU | **0.5**（`PoolingMLPReg` 默认值，配置不传入，见 §10） | (1,512) |
| 9 | `out_linear` (1,539) | Linear 512→3 | — | 无 | — | `disp` (1,3) |
| 10 | `out2_linear` (1,539) | Linear 512→3，与 9 并联，输入同为第 8 步输出 | — | 无 | — | `logstd` (1,3) |

每个 ResMLP 层含 3.i.a 与 3.i.b 两个子层，共 1,053,712 个参数，6 层合计 6,322,272。

- 参数总量：官方 README 配置（输入 6×100，`out_dim=3`）为 **7,191,654**；IPB 基准配置（前置无参数 ×2 抽取，结构不变）同为 **7,191,654**，由夹具锁定。不采用的候选配置：`out_dim=2` 为 7,190,628；窗口 200、`patch_len=50`（4 个 patch）为 7,267,428；窗口 200、`patch_len=25`（8 个 patch）为 7,190,916。
- 参数注册顺序（夹具 `param_names`）：`extractor.net.1.{weight,bias}` → 对每层 i：`net.{2+i}.0.{scale, affine.g, affine.b, affine_out.g, affine_out.b, fn.weight}`，然后 `net.{2+i}.1.{scale, affine.g, affine.b, affine_out.g, affine_out.b, fn.0.weight, fn.3.weight}` → `extractor.net.8.{g,b}` → `reg.net.{1,2,3}.0.{weight,bias}` → `reg.out_linear.{weight,bias}` → `reg.out2_linear.{weight,bias}`，共 92 个张量，没有 buffer。Conv1d 权重形状为 `(4,4,1)`。仿射与 scale 参数保持 `(1,1,512)` 形状，这样才能与夹具逐项比对。
- 权重初始化：`TwoLayerModel._initialize`（`model_twolayer.py:219-229`）**从未被调用**，所以实际使用 PyTorch 默认初始化（Linear/Conv1d 为 `kaiming_uniform_(a=√5)`，bias 为均匀分布）；仿射 g=1、b=0；scale=0.1。
- 激活模块：一个 `nn.GELU()` 实例在全网共享（它也作为 `TwoLayerModel.active_function` 注册）。GELU 无状态，共享不影响结果。
- 计算量特征：token 数只有 4，所以 FLOPs 约为参数量 × 4 量级。这里的“轻量”指计算量低，参数量（约 7.19 M）并不小，比 TLIO ResNet 还多。论文中的 FLOPs 与参数量数值**未核实**。

## 5. 损失与训练配方（official）

仓库没有训练代码。下表“来源”一栏写“TLIO 推断”的条目，依据是摘要中“用 LLIO-Net 替换 TLIO 的网络”，具体数值取 TLIO 官方实现，见 [`tlio.md`](tlio.md) §5。**全部未经 LLIO 论文核实**。

| 项 | 值 | 来源 |
|---|---|---|
| 损失 | 对角高斯 NLL（位移 `disp` 与 `logstd`），`logstd` 下限 `log(1e-3)`；第一阶段按 TLIO 官方代码实际行为为“`logstd` detach 的 NLL”（论文写作 MSE，见 `tlio` 卡 §5、§10） | TLIO 推断 |
| 优化器 | Adam | TLIO 推断 |
| 学习率 / 调度 | 同 TLIO | TLIO 推断 |
| batch | 同 TLIO | TLIO 推断 |
| epoch | 同 TLIO | TLIO 推断 |
| 权重衰减 | 同 TLIO | TLIO 推断 |
| 梯度裁剪 | 同 TLIO | TLIO 推断 |
| 增强 | 同 TLIO（随机偏航、偏置扰动、重力方向扰动等） | TLIO 推断 |
| 阶段切换 | 同 TLIO：epoch（从 1 计）< 10 为第一阶段，之后完整 NLL | TLIO 推断 |
| 选模/早停 | 同 TLIO（val 损失最优） | TLIO 推断 |
| 网络 dropout | 特征提取隐层 0.2（README 配置）；回归 MLP 0.5（代码默认值） | 官方代码 |

## 6. benchmark 适配（unified 配方）

| 改动 | 类型 | 理由 |
|---|---|---|
| 输入 200 Hz、`window=200`，模型入口先做 `avg_pool1d(k=2,s=2)` 抽到 100 个样本，再送入官方 `input_len=100, patch_len=25` 网络 | 不改变结构（无参数前处理） | 官方网络按“1 s、100 个样本”设计（README 注释）。先抽取，就能保持 1 s 时长、4 个 0.25 s patch、全部参数形状与官方完全一致；参数量测试可以直接锁定官方结构。两点均值抽取与 IPB 重采样器的多相滤波不同，这一差异写入模型 docstring。备选方案（`patch_len=50` 或 8 个 patch）会改变 Feature Convert 或 token-mixing 的形状，不采用 |
| 保留 `out_dim=3`，视图 `dims=3`，水平指标只用 xy | 不改变结构 | 与 `tlio` 卡一致：LLIO 是 TLIO 网络的替代品，3D 位移与不确定度是官方损失的一部分；改为 2 维会改变两个输出头（参数量 7,190,628），不采用 |
| 目标：`target=displacement`，损失在位移尺度上计算；对外 `vel = disp / 0.995 s`，`logstd_vel = logstd − log(0.995)` | 不改变结构 | 与 `tlio` 卡相同的换算，NLL 形式不变 |
| 视图：`frame=gravity_yaw_local, orientation=reference, remove_gravity=false, stride=10, eval_stride=10` | 不改变结构 | 与 `tlio` 卡保持一致（LLIO 复用 TLIO 管线） |
| 损失与阶段切换：与 `tlio` 卡 §6 相同（epoch < 10 时 `logstd.detach()`，之后完整 NLL） | 不改变结构 | 官方没有单独给出，按“替换 TLIO 网络”的系统定义继承 |
| 轨迹：窗口速度积分（DESIGN §5），不使用 EKF | 不改变结构（评测协议） | IPB 统一评测协议；EKF 版本属于系统级复现，不在 v1 范围内 |
| 选模：val ATE | 不改变结构 | IPB 诚实协议 |

## 7. 官方报告数值

| 数据集 | 指标（定义） | 数值 | 来源（表号） |
|---|---|---|---|
| — | 推理效率（相对 TLIO） | “up to 12 times improved” | 摘要 |
| — | ATE / RTE / 漂移 / 参数量 / FLOPs / 移动端耗时 | 未核实（全文不可得） | — |

## 8. 忠实性测试建议

- [ ] `total_params == 7,191,654`（基准配置 = 官方配置；抽取层无参数）；`out_dim=2` 的变体应为 7,190,628（仅作对照，不注册）
- [ ] `param_shapes` 与夹具 92 项逐项一致（注册顺序见 §4）
- [ ] 输入 `(B,6,200)` → 模块输出 `disp (B,3)`、`logstd (B,3)`，包装层 `vel (B,3)`；输入长度不是 200 时应报错
- [ ] 展平顺序：构造 `x[:,c,t] = 1000·c + t`，经抽取前的 Rearrange 后，第 0 个 token 的前 8 维应为 `[0,1000,…,5000,1,1001]`，第 1 个 token 的首元素为 25（官方实测结果）
- [ ] token-mixing 等价性：把 token-mixing 的 Conv1d(4,4,1) 权重换成单位阵，并令 scale=0、仿射为恒等时，子层应为恒等映射
- [ ] patch 置换性质：对输入 patch 做置换并同样置换 Conv1d 权重的行和列，回归输出不变（因为回归前是均值池化）
- [ ] `eval()` 下前向是确定性的；`train()` 下 dropout 生效（p=0.2 与 p=0.5 两处）
- [ ] 初始化：仿射 g=1、b=0，scale=0.1（常数，可以直接断言）

## 9. 预训练权重

官方没有发布任何权重。

## 10. 官方实现的坑与未决问题

1. **回归 MLP 的 dropout 不受配置控制**：`PoolingMLPReg` 的 dropout 默认为 0.5（`model_twolayer.py:105`），`TwoLayerModel` 构造它时没有传入（`model_twolayer.py:204-215`）。README 中的 `dropout: 0.2` 只作用于特征提取器。
2. **默认配置无法实例化**：`TwoLayerModel(None)` 的默认字典缺少 `extractor.dropout`，在 `model_twolayer.py:194` 抛出 `KeyError: 'dropout'`。
3. **`reg.name="MLP"` 分支忽略 `out_dim` 和 dropout**：`SimpleMLPReg` 构造时不传 `out_dim`（`model_twolayer.py:200-202`），输出固定为 3 维；实测 `out_dim=2` 时仍输出 `(B,3)`，参数量为 19,001,958。
4. **`_initialize` 从未被调用**（`model_twolayer.py:219`），论文或代码中若有“特定初始化”的描述都不生效。
5. **Post-affine 位置**：仿射作用在“残差相加之后”（`model_MLP.py:41`），与原始 ResMLP 论文的写法（只有 pre-affine，输出不再仿射）不同；移植时必须照此实现。
6. **token-mixing 用的是 Conv1d(patch_num, patch_num, 1)**，它依赖 `(B, L, D)` 的轴语义，窗口或 patch 数一变，参数形状就跟着变；`input_len` 必须能被 `patch_len` 整除（`int()` 截断会导致 Rearrange 报错）。
7. **numpy ≥ 2 无法直接导入**：`model_MLP.py:1-3` 导入了已删除的 `numpy.lib.arraypad`/`numpy.lib.arraysetops`，而且从未使用；导出夹具时用 `sys.modules` 桩绕过。
8. **`model_MLP.py::MLPCombineNet` 无法实例化**：`MLPExtractor.__initialization` 对一维 bias 调用 `xavier_normal`（`model_MLP.py:111`），抛出 `ValueError: Fan in and fan out can not be computed…`。绕过初始化后，该网络有 812,886 个参数。结构为：输入 `(B,6,100)` 转置后切成 5 段、每段 20 个样本，时间优先展平为 `(B,5,120)` → 4×ResMLP(120, expansion 4，带 bias，ReLU) → Linear 120→60 / 60→30（ReLU、Dropout 0.5）→ Linear 30→20 → 拉平为 `(B,100)` → `MLPReg`：4×ResMLP(100) → [Linear→ReLU→BN→Dropout 0.5]×3（80/50/20）→ Linear 20→6，拆成 `(3,3)`。README 未把它称作 LLIO，IPB 不注册。
9. **数据管线、训练、EKF 全部缺失**：采样率（README 注释写 100 Hz，而 TLIO 原生是 200 Hz）、窗口时长、损失与切换 epoch、数据集与划分都无法从仓库核实。论文全文不可得是本卡的主要缺口，拿到 PDF 后应补齐 §5 与 §7。
