# RIO：旋转等变自监督惯性里程计（`rio`）

> 规格卡版本：v1（2026-09-17）· fidelity：`paper-only` · 夹具：无（参数量由本卡给出的解析/实例化计数锁定，见第 8 节）

## 1. 元信息

| 项 | 值 |
|---|---|
| 论文 | *RIO: Rotation-Equivariance Supervised Learning of Robust Inertial Odometry*；Xiya Cao, Caifa Zhou, Dandan Zeng, Yongliang Wang（华为 2012 实验室 Riemann lab）；CVPR 2022, pp. 6614–6623；arXiv:2111.11676（v1，2021-11-23）；补充材料：CVF open access `Cao_RIO_Rotation-Equivariance_Supervised_CVPR_2022_supplemental.pdf` |
| 官方仓库 | **GitHub 上未找到**。CVPR 版正文写 “We release our code and dataset at this website”，链接指向华为云 AI Gallery 数据集页（`developer.huaweicloud.com/develop/aigallery/dataset/detail?id=9eab7d68-…`），该页面为需要 JS/登录的单页应用，本次**未能核实**其中是否含代码。因此按论文写重实现规格。 |
| 许可 | 无官方代码可引用；论文本身为 CVF open access。本规格只描述方法，不涉及代码许可。骨干沿用 RoNIN ResNet（官方代码为 GPL-3.0，见 `ronin_resnet18.md`）；我们是**按规格独立实现**，不复制代码。 |
| 框架 | 论文未说明（从 “Update θ using Adam”、GN 等描述看应为 PyTorch） |
| fidelity | `paper-only` |
| 参考 | arXiv v1 第 3 节（式 (1)–(5)、Algorithm 1）、第 4 节（表 1、表 2）；CVPR 补充材料 Algorithm 1（Adaptive TTT）与表 1（IPS 数据集）；骨干结构：`docs/algorithms/ronin_resnet18.md`，官方 `third_party/ronin/source/model_resnet1d.py`、`source/ronin_resnet.py:19-35` |

RIO 不是新网络，而是**训练与推理协议**：
(a) 联合训练（J-ResNet）：RoNIN 速度 MSE 加上“水平旋转等变”自监督辅助损失；
(b) 自适应测试时训练（A-TTT）：推理时只用辅助损失在线更新模型，用深度集成（3 个模型）的方差决定“何时更新 / 保持 / 恢复初始参数”。
骨干为 RoNIN ResNet-18，把全部 BatchNorm 换成 GroupNorm（为了小批量 TTT）。

## 2. 任务（输入）

| 项 | 官方（论文） | IPB 映射 |
|---|---|---|
| 采样率 | 200 Hz | 200 Hz |
| 窗口 | 200 帧 = 1 s（“we take every 200 continuous frames as input”） | `window=200` |
| 训练步长 | 未单独说明；“Other implementations are exactly the same as they claimed in [RoNIN]” → RoNIN 官方 `step_size=10`，训练时随机偏移 `step_size // 2` | `stride=10` |
| 推理步长 | 10 帧（“sampled every 10 frames at 20 Hz”） | `eval_stride=10` |
| 坐标系 | RoNIN 的 HACF（重力对齐、偏航任意的世界系） | `frame=gravity_world` |
| 姿态来源 | 与 RoNIN 相同：训练用参考姿态（RoNIN 训练时用 Tango 姿态），测试 RoNIN 用设备姿态；论文未单列 | 训练/评测都跟随视图 `orientation`（主表默认 `reference`，另报 `device`） |
| 去重力 | 否（RoNIN 输入为世界系比力，含重力） | `remove_gravity=false` |
| 通道顺序 | RoNIN 管线 `[gyro_xyz, acc_xyz]`（世界系） | 与 IPB 一致，无需置换 |
| 额外输入 | 无（A-TTT 需要同一序列中**连续**的窗口流，见第 5.3 节） | 推理器需按时间顺序成批送入窗口 |
| 输入归一化 | 无（与 RoNIN 相同） | 无 |

## 3. 输出

- 每个窗口输出 D 维**平均速度** `v̂`（m/s），RoNIN 为水平 2 维（D=2）。论文 Algorithm 1 第 7 行写 “Σ_{x,y,z}”，但 R-ResNet（RoNIN 发布模型）与 B/J-ResNet 同构，且 RoNIN 官方 `_output_channel=2`，因此按 **D=2** 实现，求和改为对 D 个分量求和（见第 10 节）。
- 目标：论文写 “ground truth velocity is calculated as the displacement … divided by the time length”，而同段公式写 `v_i^gt = P_i − P_{i−199}`（漏除时长）。以文字为准，即 IPB `target=avg_velocity`。
- 不输出协方差。深度集成的方差 `σ²` 只用于 TTT 决策，不作为输出不确定度（可以另存为诊断量 `aux.ens_var`）。
- 轨迹：与 RoNIN 相同，逐窗速度积分（IPB §5 统一重建）。

## 4. 网络结构

骨干 = `ronin_resnet18`（逐层表见 `docs/algorithms/ronin_resnet18.md`），唯一的结构改动：**全部 21 个 `BatchNorm1d` 换成 `GroupNorm`**（论文 “We replace Batch Normalization (BN) with Group Normalization (GN)”）。

| # | 层（T=200，D=2） | 参数 | 归一化 | 激活 | dropout | 输出形状 |
|---|---|---|---|---|---|---|
| 0 | 输入 | — | — | — | — | (B, 6, 200) |
| 1 | Conv1d 6→64, k7, s2, p3, 无 bias | 2,816 | **GN(32, 64)** | ReLU | — | (B, 64, 100) |
| 2 | MaxPool1d k3, s2, p1 | 0 | — | — | — | (B, 64, 50) |
| 3 | 残差组 1：2×BasicBlock1D(64, k3, s1) | 49,664 | GN(32,64) ×4 | ReLU | — | (B, 64, 50) |
| 4 | 残差组 2：BasicBlock1D(64→128, s2, 下采样 1×1 conv+GN) + BasicBlock1D(128) | 181,504 | GN(32,128) ×5 | ReLU | — | (B, 128, 25) |
| 5 | 残差组 3：同上 128→256 | 723,456 | GN(32,256) ×5 | ReLU | — | (B, 256, 13) |
| 6 | 残差组 4：同上 256→512 | 2,888,704 | GN(32,512) ×5 | ReLU | — | (B, 512, 7) |
| 7 | 过渡：Conv1d 512→128, k1, 无 bias | 65,792 | GN(32,128) | — | — | (B, 128, 7) |
| 8 | 展平 | 0 | — | — | — | (B, 896) |
| 9 | Linear 896→512 | 459,264 | — | ReLU | 0.5 | (B, 512) |
| 10 | Linear 512→512 | 262,656 | — | ReLU | 0.5 | (B, 512) |
| 11 | Linear 512→2 | 1,026 | — | — | — | (B, 2) |

（每行参数含该行归一化层的仿射参数；GN 与 BN 的可学习参数量都是 `2C`，所以替换后**参数量不变**。）

- 参数总量（单模型，D=2，T=200）：**4,634,882**。本分叉用官方 RoNIN `ResNet1D` 实例化并把 BN 替换成 `GroupNorm(32, C)` 后实测（脚本在 scratchpad，未入库）；数值应与 `tests/fixtures/algorithms/ronin_resnet18.json` 的 `total_params` 完全相同（以夹具为准）。D=3 变体为 4,635,395。
- 缓冲区：GN 没有 running statistics，因此 `buffers()` 为空（BN 版有 `running_mean/var/num_batches_tracked`）。
- GN 组数：**论文未给出**。本规格取 `num_groups=32`（Wu & He 2018 的默认值；所有通道数 64/128/256/512 都能整除）。这是可配置项 `gn_groups`，改变它不改变参数量。
- 初始化：沿用 RoNIN（Conv：`kaiming_normal_(mode='fan_out', nonlinearity='relu')`；Linear：`N(0, 0.01)`、bias 0；归一化层 weight=1、bias=0）。
- 深度集成：A-TTT 需要 M=3 个**独立训练**的 J-ResNet（不同随机初始化与数据打乱，论文 “randomization-based approach”），推理时总参数量 3×4,634,882 = 13,904,646（外加一份被更新的在线副本，见 5.3）。

## 5. 损失与训练配方（official = 论文）

### 5.1 旋转算子与辅助损失

- 水平旋转 `Rot(·|φ)`：绕世界 z 轴旋转 φ。对输入窗口，**加计与陀螺的三个分量都乘 `R_z(φ)`**（z 分量不变；陀螺是赝矢量，但在真旋转下与普通向量同样变换）；对 2 维速度乘 `R(φ) = [[cos φ, −sin φ], [sin φ, cos φ]]`。
- 差异度量（式 (2)）：负余弦相似度 `D(a, b) = −⟨a, b⟩ / (‖a‖₂ ‖b‖₂)`，取值 [−1, 1]，完全一致时为 −1。实现时分母加 `eps=1e-8`。
- 门控：`‖v_i‖₂ ≤ 0.5 m/s` 时该样本辅助损失置 0（Algorithm 1 第 9–13 行；式 (5)）。Algorithm 1 中被比较的是**模型预测**的范数（第 8 行把 `v_i` 改写为旋转后的预测，旋转不改变范数），测试时也只能用预测。本规格：门控用 `‖v̂_i‖`（`detach`），可选 `ssl_gate_on=target`（仅训练期可用）。
- 自监督目标（式 (3)(4)）：`L_ssl(X) = (1/K) Σ_j D( F(Rot(X|φ_j)), Rot(F(X)|φ_j) )`。
- 梯度：论文引用 SimSiam 的负余弦，但**没有提到 stop-gradient**。本规格默认两支都回传梯度（`ssl_stop_grad=false`），作为可配项保留。

### 5.2 联合训练（J-ResNet，Algorithm 1）

每个训练样本 `(X_i, v_i^gt)`：
1. 采样 `φ_i ~ U(0, 2π]`（每样本独立，K=1），构造共轭输入 `X_i^φ = Rot(X_i|φ_i)`；
2. `v̂_i = F(X_i)`，`v̂_i^c = F(X_i^φ)`（同一模型、同一 batch 前向，可把两者拼成 2B 的 batch）；
3. `l_v,i = Σ_d (v̂_{i,d} − v^gt_{i,d})²`；
4. `l_ssl,i = D(Rot(v̂_i|φ_i), v̂_i^c)`，若 `‖v̂_i‖ ≤ 0.5` 则为 0；
5. `L = Σ_i l_v,i + Σ_i l_ssl,i`，Adam 更新。

| 项 | 值 | 来源 |
|---|---|---|
| 损失 | `L = Σ_i l_v,i + Σ_i l_ssl,i`（两项权重 1:1）；本规格实现为 `mean_i(l_v,i) + mean_i(l_ssl,i)`，与“求和”只差一个全局常数 B（Adam 对此近似不敏感）。**注意** RoNIN 的 `nn.MSELoss()` 对 B·D 取平均，若照搬会使速度项相对辅助项缩小 D 倍——不要照搬 | 论文 Alg. 1 第 15 行 |
| 优化器 | Adam | Alg. 1 第 16 行 |
| 学习率 / 调度 | 未给出；“其他实现与 RoNIN 完全相同” → RoNIN 官方 `lr=1e-4`、`ReduceLROnPlateau(factor=0.1, patience=10, eps=1e-12)`（`ronin_resnet.py:136-137`） | 推断，未核实 |
| batch | 未给出；RoNIN 官方 128（每个样本再加 1 个共轭，前向实际 256） | 推断，未核实 |
| epoch | 未给出（RoNIN 官方脚本默认 `epochs=10000`，靠人工/调度停止） | 未核实 |
| 权重衰减 | 未给出（RoNIN 官方无） | 推断 |
| 梯度裁剪 | 未提及 | — |
| 增强 | “consistent … data augmentation strategy for all models”（CVPR 版）→ RoNIN 的随机水平旋转 `RandomHoriRotate(2π)`（输入与目标同转）+ 随机窗口偏移；RIO 的共轭旋转在其之后另加 | 推断 |
| 阶段切换 | 无（两项损失从第 0 个 epoch 起同时生效） | Alg. 1 |
| 选模/早停 | 未给出（RoNIN：按 val loss 保存最好模型） | 推断 |
| 训练数据 | RoNIN 公开的一半数据（train 列表）；R-ResNet 为 RoNIN 发布的全量 BN 模型，仅作参考 | 第 4 节 |

### 5.3 自适应测试时训练（A-TTT，补充材料 Algorithm 1）

状态：在线模型 `F(·|θ)`，初值 `θ₀ = θ*`（预训练 J-ResNet）；冻结的集成 `{F(·|θ_m)}_{m=1..3}`；一个 Adam 优化器。

对**同一序列**按时间顺序，每累计 128 个窗口（窗口 200 帧、步长 10 帧）组成批 `X_t`：
1. 集成前向（eval 模式）：`v_t^m = F(X_t|θ_m)`，`v̄_t = (1/M) Σ_m v_t^m`，逐样本方差 `σ²_{t,i} = (1/M) Σ_m ‖v^m_{t,i} − v̄_{t,i}‖²`（论文写 `(p_θm − p*)²`，对向量的含义不明；本规格取平方范数，即协方差迹，见第 10 节）；
2. 若 `min_i σ²_{t,i} < 1e-4`：**恢复** `θ_t ← θ*`（并重置 Adam 状态）；
3. 否则若 `mean_i σ²_{t,i} < 0.04`：**保持** `θ_t ← θ_{t−1}`；
4. 否则：从 `θ_{t−1}` 出发做 `N_TTT = 5` 次 Adam 更新，每次的损失为
   `L = Σ_{i=1..k} Σ_j L(Rot(v_j|φ_i), v_j^{c_i})`，其中 `k=4`，`φ ∈ {72°, 144°, 216°, 288°}`，`v = F(X_t|θ)`，`v^{c_i} = F(Rot(X_t|φ_i)|θ)`，`L` 同式 (5)（`‖v_j‖ ≤ 0.5` 时为 0）；
5. 输出 `v̂_t = F(X_t|θ_t)`（eval 模式；**只用在线模型的输出，不用集成均值**）。

| 项 | 值 | 来源 |
|---|---|---|
| 批大小 | 128 个窗口（≈ 6.4 s 新数据） | 正文 3.3 |
| 每批更新次数 | 5（图 7：1→5 明显变好，>5 无益，15 略差） | 正文 3.3、第 5 节 |
| 共轭角 | 4 个固定角 72°/144°/216°/288° | 正文 3.3 |
| 阈值 | 保持：`mean σ² < 0.04`；恢复：`min σ² < 1e-4` | 正文 3.3、补充 Alg. 1 |
| 集成规模 | M=3 | 正文 3.3 |
| TTT 学习率 | **未给出**；本规格默认 `1e-4`（与训练初始学习率一致），标记未核实 | — |
| 模式 | 论文未说明；本规格：梯度步用 `train()`（dropout 生效），集成与最终预测用 `eval()` | 假设 |
| 序列边界 | 每条新序列都从 `θ*` 与新 Adam 状态开始；不跨序列携带状态 | 假设（IPB 诚实协议必需） |
| 末尾不足 128 的批 | 论文未说明；本规格按同样规则处理不足批 | 假设 |
| 朴素 TTT（N-TTT，对照） | 每批都更新、从不恢复 | 第 4.4 节 |

## 6. benchmark 适配（unified 配方）

| 改动 | 类型 | 理由 |
|---|---|---|
| 输出维度 D=2，速度项对 D 个分量求和 | 不改变结构 | 与 RoNIN/IPB `dims=2` 一致；论文 “x,y,z” 求和与 2D 头矛盾 |
| BN→GN（`gn_groups=32`） | 改变结构（相对 `ronin_resnet18`；相对 RIO 论文不变） | 论文明确要求；参数量不变，缓冲区消失 |
| 目标取 IPB `avg_velocity` | 不改变结构 | 论文文字定义即平均速度 |
| 推理器提供两种模式：`ttt=off`（标准滑窗推理）与 `ttt=adaptive`（A-TTT）；`ttt=naive` 仅作消融 | 不改变结构 | 论文表 1 同时报告两者；主表应分别列出，并在结果文件中标记“使用了测试时自适应（无标签）” |
| A-TTT 严格因果、逐序列复位 | 不改变结构 | 避免跨序列/跨划分信息泄漏；TTT 只使用测试 IMU，不使用任何测试标签 |
| TTT 超参（阈值、次数、角度）固定为论文值，禁止在 test 上调 | 不改变结构 | IPB 诚实协议；如需调，只能在 val 上调 |
| 集成的 3 个模型用 3 个训练种子（与 IPB 多种子协议共用），在线模型取种子 0 的那个 | 不改变结构 | 论文未说明在线模型是哪一个；复用种子避免额外训练 |
| unified 配方下 epoch/调度改用 IPB 统一预算 | 不改变结构 | 论文未给出 |

## 7. 官方报告数值

指标定义（论文 4.1）：ATE = 整条轨迹位置 RMSE；RTE = RoNIN 式时间 RTE（1 min）；D-drift = `|L̂ − L| / L`（轨迹长度相对误差，**不是** IPB 的终点漂移）。OxIOD、RIDI、IPS 的整条估计轨迹先做 **Umeyama 对齐**再评测；RoNIN 不对齐。训练只用 RoNIN 公开训练集，其余三个数据集为跨数据集测试。与 IPB 数值**不可直接比较**（对齐方式、划分、长度定义均不同）。

表 1（arXiv v1 与 CVPR 版一致）：

| 数据集 | 指标 | R-ResNet | B-ResNet | J-ResNet | B-ResNet-TTT | J-ResNet-TTT |
|---|---|---|---|---|---|---|
| RoNIN | ATE (m) | 5.14 | 5.57 | 5.02 | 5.05 | 5.07 |
| RoNIN | RTE (m) | 4.37 | 4.38 | 4.23 | 4.14 | 4.17 |
| RoNIN | D-drift | 11.54% | 9.79% | 9.59% | 8.49% | 9.10% |
| OxIOD | ATE (m) | 3.46 | 3.52 | 3.59 | 2.92 | 2.96 |
| OxIOD | RTE (m) | 4.39 | 4.42 | 4.43 | 3.67 | 3.74 |
| OxIOD | D-drift | 20.67% | 19.68% | 17.43% | 15.50% | 15.98% |
| RIDI | ATE (m) | 1.33 | 1.19 | 1.13 | 1.04 | 1.03 |
| RIDI | RTE (m) | 2.01 | 1.75 | 1.65 | 1.53 | 1.51 |
| RIDI | D-drift | 10.50% | 7.99% | 7.61% | 6.89% | 6.93% |
| IPS（自采，155 条） | ATE (m) | 1.60 | 1.84 | 1.67 | 1.55 | 1.55 |
| IPS | RTE (m) | 1.52 | 1.68 | 1.65 | 1.46 | 1.47 |
| IPS | D-drift | 8.38% | 7.66% | 7.96% | 5.93% | 6.75% |

R-ResNet = RoNIN 发布的 BN 模型（全量数据训练）；B-ResNet = 用公开一半数据重训的 GN 基线；J-ResNet = 联合训练；“-TTT” = A-TTT 推理。论文称 R-ResNet 的 RoNIN 结果与 RoNIN 论文 unseen 集一致。

表 2（B-ResNet 上 A-TTT 与 N-TTT 对比；A-TTT 列与表 1 的 B-ResNet-TTT 相同）：

| 数据集 | 指标 | A-TTT | N-TTT |
|---|---|---|---|
| RoNIN | ATE / RTE / D-drift | 5.05 / 4.14 / 8.49% | 4.94 / 4.27 / 9.43% |
| OxIOD | ATE / RTE / D-drift | 2.92 / 3.67 / 15.50% | 3.50 / 4.39 / 19.55% |
| RIDI | ATE / RTE / D-drift | 1.04 / 1.53 / 6.89% | 1.11 / 1.64 / 7.56% |
| IPS | ATE / RTE / D-drift | 1.55 / 1.46 / 5.93% | 1.63 / 1.54 / 6.73% |

其他：图 6 显示用 30% 数据训练的 J-ResNet-TTT 在 IPS 上与 100% B-ResNet 相当（只有图，未给数表）。

## 8. 忠实性测试建议

- [ ] 参数量：单模型（D=2，T=200）== 4,634,882，且等于 `ronin_resnet18` 夹具的 `total_params`（解析/实例化计数，非官方 RIO 代码）
- [ ] 结构：模型中 `BatchNorm*` 个数为 0，`GroupNorm` 个数为 21；`list(model.buffers()) == []`
- [ ] `param_shapes` 与 `ronin_resnet18` 夹具逐项一致（BN 仿射参数与 GN 仿射参数形状相同）
- [ ] 输出形状 `(B, 2)`；batch 大小 1 时训练模式前向不报错（GN 不依赖 batch 统计——BN 版在 B=1 训练模式下会报错，可作为区分测试）
- [ ] 旋转算子：`Rot(Rot(X|a)|b) == Rot(X|a+b)`；`Rot(X|2π) == X`；z 通道（索引 2、5）不变；对 `(B,6,T)` 同时旋转 0–2 与 3–5 通道
- [ ] 负余弦：`D(v, v) == −1`，`D(v, −v) == 1`，`D(v, R(π/2)v) == 0`
- [ ] 辅助损失对**精确等变**的模型为最小值：构造 `F_eq(X) = R(ψ(X)) · u(X)`（ψ 为输入的某个等变方向，如窗口内水平加计均值方向，u 为不变量），则对任意 φ，`mean L_ssl == −1`（门控样本除外）；对恒定输出模型（不随输入旋转）`L_ssl` 随 φ 变化，期望值 ≈ 0（φ 均匀时）
- [ ] 门控：`‖v̂‖ ≤ 0.5` 的样本辅助损失与梯度均为 0
- [ ] 联合损失：`φ=0` 时 `L_ssl == −1`（门控外所有样本；须在 `eval()` 下检查，否则两次前向的 dropout 掩码不同）；`L_total` 对速度项与辅助项的权重为 1:1
- [ ] A-TTT 状态机（用 mock 集成方差）：`min σ² < 1e-4` → 参数逐元素等于 `θ*`；`mean σ² < 0.04` → 参数与上一批相同；否则恰好 5 次优化器 step；共轭角恰为 4 个固定值
- [ ] A-TTT 因果性：第 t 批的预测只依赖 `X_1..X_t`（打乱后续批不改变前面的输出）；换序列时状态复位
- [ ] 集成方差：三个相同模型时 `σ² ≡ 0` → 必走“恢复”分支

## 9. 预训练权重

无公开权重（GitHub 无仓库；华为云页面未核实）。R-ResNet 可用 RoNIN 官方发布的 ResNet 权重（BN 版，不是 RIO 模型）。

## 10. 官方实现的坑与未决问题

1. **没有公开代码**：所有实现细节来自论文，GN 组数、TTT 学习率、训练 epoch、stop-gradient 均未给出，本卡给出的默认值都是假设，已在表中标注。
2. **速度求和维度矛盾**：Algorithm 1 写 `Σ_{x,y,z}`，但骨干与 RoNIN 一样是 2 维输出。按 2 维实现。
3. **目标公式漏除时长**：`v_i^gt = P_i − P_{i−199}` 与文字 “divided by the time length” 矛盾，以文字为准。
4. **门控依据**：伪代码用预测速度的范数（测试时也只能如此），正文措辞像是真值速度。默认用预测（`detach`）。
5. **方差的定义**：`(p_θm − p*)²` 对向量未说明是逐分量还是平方范数；阈值 0.04 / 1e-4 的含义会差 D 倍。本规格取平方范数（迹），并在结果中同时记录逐分量均值以便核对。
6. **“在线模型”是谁**：补充材料 Algorithm 1 分别初始化 `F(·|θ*)` 与集成 `{F(·|θ_i)}`，没说 `F` 是否是集成成员；本规格取种子 0 的成员的一个可更新副本。
7. **损失缩放**：若照搬 RoNIN 的 `nn.MSELoss()`（对 B·D 取均值）而辅助项按样本取均值，两项权重不再是 1:1。
8. RoNIN 评测不对齐、其余数据集做 Umeyama 对齐，论文表 1 跨数据集数值不能与 IPB 不对齐 ATE 直接比较。
9. TTT 使用测试数据（无标签）更新参数，属于转导式推理：结果表必须显式标注，并与标准推理分列。
