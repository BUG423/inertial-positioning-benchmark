# CTIN（`ctin`）

> 规格卡版本：v1（2026-09-17）· fidelity：`paper-only` · 夹具：无（官方仓库不含代码；参数量由本卡参考规格推导，见 §8）

标注约定：**[P]** 表示论文明确写出的内容，**[A]** 表示本卡为了能实现而做的假设（§10 汇总）。

## 1. 元信息

| 项 | 值 |
|---|---|
| 论文 | *CTIN: Robust Contextual Transformer Network for Inertial Navigation*；Bingbing Rao, Ehsan Kazemi, Yifan Ding, Devu M Shila, Frank M Tucker, Liqiang Wang；AAAI 2022（Vol. 36, No. 5, pp. 5413–5421）；<https://arxiv.org/abs/2112.02143>（本卡依据 arXiv v2，2021-12-20，含附录） |
| 官方仓库 | <https://github.com/bingrao/ctin> @ `1441c726811ac87283903471562023fd429e3a08`。**只有 `LICENSE` 和 `README.md`，没有任何源码**。README 写明代码和数据归 Unknot.id 所有，要等合作方批准后才发布。本地未克隆 |
| 许可 | 仓库 LICENSE 为 MIT（Copyright 2021 Bing），但 README 徽章写的是 Apache-2.0，两者不一致。由于没有代码，许可对本复现没有实际约束 |
| 框架 | PyTorch 1.7.1，Adam，RTX 2080Ti [P] |
| fidelity | `paper-only` |
| 参考位置 | §3.1（System Overall）、§3.2（Attention）、§3.3（Eq. 6–8）、§4、附录 A.2（数据准备与旋转）、A.3（超参数）、Table 1–3、Fig. 1 |

## 2. 任务（输入）

| 项 | 官方 | IPB 映射 |
|---|---|---|
| 采样率 | 各数据集原生频率：RIDI、RoNIN、CTIN 为 200 Hz；OxIOD、IDOL 为 100 Hz（Table 1）[P] | 200 Hz |
| 窗口 | N = m = 200 样本（A.2）[P]，即 200 Hz 下 1 s、100 Hz 下 2 s | `window=200` |
| 训练步长 | OxIOD 20、RIDI 50、其余 10（A.2）[P]；另外对滑窗起点加随机偏移 [P] | `stride=10`，加 `time_shift` 增强 |
| 推理步长 | 未说明 | `eval_stride=10` |
| 坐标系 | 导航系：z 轴与重力反向（即 z 向上），水平轴取决于初始姿态（§2.2、A.2）[P]。A.2 原文：“IMU samples in each window are rotated … using device orientations **at beginning of the window**” | `frame=gravity_world`。IPB 按每个样本自己的姿态旋转，与“窗口起点姿态”说法不同，见 §10 [A] |
| 姿态来源 | “Rotation Matrix Selector” 按数据集选择（A.2）[P]：RIDI 训练/验证/测试都用设备估计；OxIOD 训练/验证用 Vicon 真值、测试用设备欧拉角；RoNIN 沿用 RoNIN 规则（测试用设备；训练时若序列末端对齐误差 < 20° 用设备，否则用真值）；IDOL、CTIN 训练/验证/测试**都用真值** | 由 benchmark 全局 `orientation` 决定，不逐数据集切换 |
| 去重力 | 未说明（Eq. 3d 描述的是 SINS，并非网络输入）| `remove_gravity=false` [A] |
| 通道顺序 | “an IMU sample is the concatenation of data from gyroscope and accelerometer”，即 `[gyro, acc]`（§3.1）[P] | 与 IPB 一致，无需置换 |
| 额外输入 | 无 | 无 |
| 输入归一化 | 未说明 | 不归一化 [A] |

## 3. 输出

- **官方**：序列到序列。窗口内每个时刻 `t=1..m` 输出 2D 速度 `vel_{1:m}` 和对应的协方差 `cov_{1:m}`（§3.1，Fig. 1）[P]。
  协方差只建模对角线，“parametrized by two coefficients of a velocity”（§3.3）[P]。本卡规定协方差头输出 `logσ ∈ R²`，`Σ_t = diag(exp(2·logσ_t))` [A]。
- **官方轨迹**：“Position can be obtained by the integration of velocity”（§3.1）[P]。推理时各窗口的序列如何拼接没有说明。
- **IPB 适配**：
  - `vel = mean_t vel_t`，形状 (B, 2)；
  - `logstd = ½·log(mean_t exp(2·logσ_t))`，形状 (B, 2) [A]。这是保守的“误差完全相关”近似，理由见 §6。
  - 另外在 `aux` 中保留 `vel_seq (B,200,2)` 和 `logstd_seq (B,200,2)`。

## 4. 网络结构

论文只给出模块级描述，**没有**给出通道数、隐层维度、FFN 宽度等。下表是本卡的**参考规格**（d=64，w=64，heads=8，ff=512，groups=4，reduction=2），每行标注证据等级。输入为 `(B, 6, 200)`。

| # | 层 | 参数（核/步长/填充/通道） | 归一化 | 激活 | dropout | 输出形状 |
|---|---|---|---|---|---|---|
| **空间嵌入** [P：1D CNN → BN → linear] | | | | | | |
| S1 | Conv1d | 6→64，k=3，s=1，p=1，bias [A: k 与通道] | – | – | – | (B, 64, 200) |
| S2 | BatchNorm1d | 64 | BN [P] | – | – | (B, 64, 200) |
| S3 | Linear（逐时刻） | 64→64 [A: 1 层] | – | – | – | (B, 64, 200) |
| **空间编码器，Nx=1**（A.3）[P]：改造的 ResNet-18 “bottleneck”，把空间卷积换成局部自注意力，并在最后一个 1×1 卷积前插入全局自注意力 [P] | | | | | | |
| E1 | Conv1d | 64→64，k=1，无 bias | BN [P] | ReLU [P] | – | (B, 64, 200) |
| E2a | 局部注意力·Key（C1） | Conv1d 64→64，k=3，p=1，groups=4 [P: 3×3 group conv；A: groups]，无 bias | BN [A] | ReLU [A] | – | (B, 64, 200) |
| E2b | 局部注意力·Value | Conv1d 64→64，k=1 [P]，无 bias | BN [A] | – | – | (B, 64, 200) |
| E2c | 局部注意力·权重 γ(Q, C1) | 拼接 `[X, C1]`（Q=X [P]）得 128 通道 → Conv1d 128→32，k=1，无 bias → BN → ReLU → Conv1d 32→64，k=1，带 bias；再沿时间维做 softmax [P: 拼接 + 1×1 conv + ReLU；A: 两层结构与 softmax] | BN [A] | ReLU [P] | – | (B, 64, 200) |
| E2d | 局部注意力·全局上下文 C2 | `C2 = γ ⊙ V × m` [P: γ×V；A: 逐元素相乘，乘 m 保持尺度] | – | – | – | (B, 64, 200) |
| E2e | 局部注意力·融合 | `u = mean_t(C1 + C2)` → Linear 64→32（无 bias）→ BN1d → ReLU → Linear 32→128 → 变形为 (B,2,64) → 沿分支维 softmax → `Y = a0⊙C1 + a1⊙C2` [P: “fused by an attention mechanism between C1 and C2”；A: SK/CoT 式 split-attention] | BN [A] | ReLU | – | (B, 64, 200) |
| E3 | BN + ReLU | 64 | BN [P] | ReLU [P] | – | (B, 64, 200) |
| E4 | 全局自注意力 | Q/K/V 各为 Conv1d 64→64，k=1，带 bias [P: 三个独立 1×1 conv]；8 头 [P: §2.3 h=8]；缩放点积 + softmax [P]；各头拼接，无输出投影 [A] | – | – | – | (B, 64, 200) |
| E5 | Conv1d | 64→64，k=1，无 bias [P: 最后的 1×1 conv]；**不降采样** [A] | BN [P] | – | – | (B, 64, 200) |
| E6 | 残差相加 + ReLU | shortcut 为恒等映射 [P: Add & ReLU] | – | ReLU | 0.5 [P: 编码器 dropout 0.5；A: 位置] | z：(B, 64, 200) → 转置为 (B, 200, 64) |
| **时间嵌入** [P：单层双向 LSTM + 可训练位置编码] | | | | | | |
| T1 | BiLSTM | input=6，hidden=32/方向 [A]，1 层 [P] | – | – | – | (B, 200, 64) |
| T2 | 可训练位置编码 | `Embedding(200, 64)`，按 t=0..199 索引后相加 [P: trainable；A: 查表形式] | – | – | – | (B, 200, 64) |
| **时间解码器，Nx=4**（A.3）[P]：沿用 vanilla Transformer decoder（post-norm Add & Norm）[P] | | | | | | |
| D1 | 带因果掩码的多头自注意力 | d=64，8 头 [A: 头数]，上三角掩码 [P] | LayerNorm（post）[P] | – | 0.05 [P] | (B, 200, 64) |
| D2 | 多头交叉注意力 | q 来自 D1，k=v=z [P] | LayerNorm [P] | – | 0.05 | (B, 200, 64) |
| D3 | FFN | 64→512→64 [A: 512] | LayerNorm [P] | ReLU [A] | 0.05 | (B, 200, 64) |
| **输出头** [P：两个 MLP 分支，“a simple linear network followed by a layer normalization”] | | | | | | |
| H1 | 速度头 | Linear 64→64 → LayerNorm(64) → Linear 64→2 [A: 末层投影，见 §10] | LN [P] | – | – | `vel_seq` (B, 200, 2) |
| H2 | 协方差头 | 结构同 H1，输出 `logσ` [A] | LN | – | – | `logstd_seq` (B, 200, 2) |
| L | 多任务权重 | 2 个标量 `u_v = ln δv`，`u_c = ln δc`，初值 0 [P: δv、δc 为观测噪声；A: 可学习对数参数化] | – | – | – | – |

参数分解（参考规格，由 PyTorch 原型推导）：

| 模块 | 参数量 |
|---|---|
| S1 + S2 + S3 | 1,216 + 128 + 4,160 = 5,504 |
| 编码器块 | 41,088。其中 E1 4,224；局部注意力 20,032（key 3,200 / value 4,224 / 权重 6,272 / 融合 6,336）；E3 128；E4 12,480；E5 4,224 |
| T1 BiLSTM | 10,240 |
| T2 位置编码 | 12,800 |
| 解码器 4 层 | 399,104（每层 99,776：自注意力 16,640，交叉注意力 16,640，FFN 33,280 + 32,832，3 个 LN 共 384） |
| H1 + H2 | 4,418 × 2 = 8,836 |
| **网络合计** | **477,572**；加上 2 个损失权重标量为 477,574 |

- 论文 Table 3 报告的官方参数量为 **0.5571 × 10⁶**。参考规格比它少 14.3%。论文信息不足以唯一复原这一数字。例如只把 FFN 宽度改为 640，合计为 543,620（−2.4%），但没有任何证据支持这个取值，所以**仍以 ff=512 为推荐默认值**，并在结果表中注明“参数量由规格推导”。
- 权重初始化：未说明，采用 PyTorch 默认 [A]。

## 5. 损失与训练配方（official）

| 项 | 值 | 来源 |
|---|---|---|
| 总损失 | `L = Lv/(2δv²) + Lc/(2δc²) + log(δv·δc)`，按 Kendall 同方差不确定度加权多任务 | Eq. 7 [P] |
| 速度损失 IVL `Lv` | `Lv = Lpv + Lev`。`Lpv`：把预测速度积分成位置，与同一段的真值位移差求 L2；`Lev`：`v̂` 与 `v` 之间的“累计误差” | §3.3 [P]。精确公式见下方 [A] |
| 协方差损失 CNL `Lc` | `½‖y_v − f(x)‖²_Σ + ½ln|Σ|`，Σ 为对角阵 | Eq. 8 [P] |
| 优化器 | Adam | §4、A.3 [P] |
| 学习率 / 调度 | 初始 5e-4；调度未说明 | A.3 [P] |
| batch | 未给出 | – |
| epoch | 未给出（依赖早停） | – |
| 权重衰减 | 1e-6 | A.3 [P] |
| 梯度裁剪 | 未给出 | – |
| dropout | 空间编码器 0.5；时间解码器 0.05 | A.3 [P] |
| 增强 | (1) 水平面随机偏航旋转（“coordinate frame augmentation agnostic to the heading”）；(2) 加性偏置扰动：加计 U[−0.2, 0.2] m/s²，陀螺 U[−0.05, 0.05] rad/s（原文 “for each sample”）；(3) 滑窗随机偏移 | A.2 [P] |
| 阶段切换 | 论文**未提及**。iMoT 论文附录称其复现 CTIN 时先只用速度损失热身、再联合协方差损失，这是第三方做法，不是 CTIN 原文 | iMoT arXiv:2412.12190 附录 |
| 选模/早停 | 早停 patience = 30，依据验证集表现 | §4、A.3 [P] |
| 数据划分 | 训练/验证/测试 = 8:1:1；测试分 seen 与 unseen 受试者（CTIN 数据集只有一个测试集）| §4.1 [P] |

**损失的精确形式（[A]，推荐实现）**。窗口内 `t=1..m`，`dt = 1/200`，`v_t` 为逐帧真值水平速度，`p_t` 为真值位置：

```text
Lpv = mean_t ‖ dt·Σ_{τ≤t} v̂_τ − (p_t − p_1) ‖²          # 积分位置误差（相对窗口起点）
Lev = mean_t ‖ v̂_t − v_t ‖²                               # 逐帧速度误差
Lv  = Lpv + Lev
Lc  = mean_t Σ_k [ ½ (v_{t,k} − v̂_{t,k})² · exp(−2·logσ_{t,k}) + logσ_{t,k} ]
L   = ½·exp(−2u_v)·Lv + ½·exp(−2u_c)·Lc + u_v + u_c
```

**IPB `recipe: official`**：Adam，lr 5e-4，wd 1e-6，无调度 [A]，batch 128 [A]，最多 300 epoch [A]，val 早停 patience 30，dropout 0.5/0.05，增强为 `[random_yaw, bias_noise(acc 0.2, gyro 0.05), time_shift]`，从第 0 epoch 起联合训练（论文未提阶段切换）。
**IPB `recipe: unified`**：benchmark 统一预算，其余同上。

## 6. benchmark 适配（unified 配方）

| 改动 | 类型 | 理由 |
|---|---|---|
| 通道顺序 | 不改变 | 论文即 `[gyro, acc]` |
| `window=200` @200 Hz | 不改变结构 | 论文窗口就是 200 样本。原本 100 Hz 的数据集（OxIOD、IDOL）在官方设置下窗口为 2 s，IPB 下为 1 s（统一重采样到 200 Hz），这是协议差异，结果中需注明 |
| 逐样本旋转到 `gravity_world` | 预处理差异 | 论文附录说用“窗口起点姿态”旋转，但数据加载又说遵循 RoNIN 协议（RoNIN 是逐样本旋转）。IPB 没有“窗口起点整体旋转”这种视图，采用逐样本旋转 |
| 窗口级 `vel = mean_t vel_t` | 不改变结构（后处理） | IPB 每个窗口需要一个速度。当逐帧速度准确时，窗口平均速度 ≈ `(p_end − p_start)/时长`，与 `avg_velocity` 目标一致 |
| 窗口级 `logstd = ½·log mean_t σ_t²` | 不改变结构（后处理） | 窗口内误差高度相关，保守地取“完全相关”近似；不用 `σ²/m` |
| 需要逐帧速度序列目标（视图扩展 `target_seq`，形状 (B,200,2)，有 `pose/velocity` 时直接取，否则用中心差分） | **协议扩展** | IVL/CNL 都定义在逐帧速度上；RoNIN-LSTM/TCN 也有同样需求 |
| 若视图只提供窗口级目标：`Lv = ‖vel − v̄‖²`，`Lc` 用窗口级 `logstd`，加权形式不变 | **改变损失**（降级模式） | 只在未实现 `target_seq` 时作为兜底，结果必须注明 |
| 逐数据集的姿态选择器改为全局 `orientation` | 协议统一 | 官方在 IDOL/CTIN 上测试时用真值姿态，与 RoNIN/RIDI 的设备姿态不可比；IPB 统一口径 |
| 逐数据集步长（20/50/10）改为统一 `stride=10` | 协议统一 | 统一训练采样密度 |

## 7. 官方报告数值

指标定义（§4.2）：ATE 为整条轨迹位置 RMSE；T-RTE 为 60 s 时间窗的相对位移 RMSE；D-RTE 为每走 1 m 的相对位移 RMSE；PDE = 末端误差 / 轨迹长度。单位均为 m。以下为 arXiv v2 Table 2 中 CTIN 一列：

| 数据集 | 测试集 | ATE | T-RTE | D-RTE |
|---|---|---|---|---|
| RIDI | seen / unseen | 1.39 / 1.86 | 1.99 / 2.49 | 0.11 / 0.11 |
| OxIOD | seen / unseen | 2.32 / 3.34 | 0.62 / 1.33 | 0.07 / 0.13 |
| RoNIN | seen / unseen | 4.62 / 5.61 | 2.81 / 4.48 | 0.18 / 0.25 |
| IDOL | seen / unseen | 2.90 / 3.69 | 1.35 / 1.65 | 0.13 / 0.15 |
| CTIN | seen | 1.28 | 1.29 | 0.08 |

效率（Table 3，CTIN 数据集）：参数量 0.5571×10⁶；GFLOPs/s 7.27；单序列平均 GPU 时间 65.96 ms（RTX 2080Ti）。
同表 RoNIN 基线：R-LSTM 0.2058M，R-TCN 2.0321M，R-ResNet 4.6349M。

第三方复现（iMoT arXiv:2412.12190 Table 2 中的 CTIN 列），ATE seen/unseen：RIDI 1.69/2.15，RoNIN 5.54/6.89，OxIOD 6.71/2.34，IDOL 3.15/3.70。其中 RoNIN unseen 为 6.89，明显差于原文的 5.61，说明复现差异显著。

## 8. 忠实性测试建议

- [ ] **参数量（参考规格推导，非官方）**：网络 477,572（加 2 个损失标量为 477,574）。同时断言各子模块计数与 §4 的分解表一致。注释里写明官方 Table 3 为 557,100，差距来自论文未给出的宽度。
- [ ] 形状：输入 `(B,6,200)` → `vel_seq (B,200,2)`、`logstd_seq (B,200,2)`、`vel (B,2)`、`logstd (B,2)`。
- [ ] 窗口聚合：`vel == vel_seq.mean(1)`；`logstd_seq` 为常数 c 时，`logstd == c`。
- [ ] **因果性说明**：解码器自注意力虽有上三角掩码，但 tgt 来自双向 LSTM，memory 来自非因果编码器，所以输出**不是因果的**。测试应断言：只扰动窗口最后一个样本，`t=0` 的输出也会改变（参考原型中变化约 3e-4）。不要把 CTIN 当作因果模型来测。
- [ ] 掩码：解码器自注意力掩码为严格上三角的 `-inf`。
- [ ] 损失：`v̂ ≡ v` 时 `Lv = 0`；`logσ` 取常数 a 时，`Lc` 对 a 的导数在 `a = ½·ln mean(err²)` 处为 0。
- [ ] 多任务权重：`u_v`、`u_c` 作为参数能收到梯度；其初值使总损失等于 `½(Lv + Lc)`。
- [ ] 增强：偏置扰动的取值范围分别为 ±0.2 m/s² 和 ±0.05 rad/s；随机偏航同时旋转加计和陀螺的 xy 分量，也同时旋转目标速度。

## 9. 预训练权重

无。官方仓库没有发布代码和权重；CTIN 数据集也未公开（Table 1 写的是 “will be released soon”）。

## 10. 官方实现的坑与未决问题

1. **没有代码**。官方仓库只有 README 和 LICENSE，所以所有宽度参数都是假设，参数量无法对齐 0.5571M。
2. **旋转方式自相矛盾**：A.2 写的是“按窗口起点姿态旋转”，同时又说“数据加载遵循 RoNIN 协议”（逐样本旋转）。本卡采用逐样本旋转。
3. **“linear + LayerNorm” 头不能按字面实现**：如果先 Linear 64→2 再对 2 维输出做 LayerNorm，输出只可能是 `±γ + β`，速度尺度信息被抹掉。本卡改为 Linear→LN(64)→Linear(→2)。
4. **ResNet-18 “bottleneck”**：ResNet-18 本身用的是 BasicBlock，论文所说的 bottleneck 实际是 CoTNet 风格的 1×1 → 注意力 → 1×1 结构；“spatial downsampling preserved” 与逐帧 seq2seq 输出存在张力。本卡取步长 1、不降采样。
5. **非因果**：见 §8。
6. **姿态口径不一致**：IDOL/CTIN 在测试时用了真值姿态，因此这两个数据集上的数字不能与 RoNIN/RIDI 上用设备姿态的数字直接比较。
7. **阶段切换**：CTIN 原文没有 MSE→NLL 切换，只有 iMoT 复现时加了热身。IPB official 配方不加热身。
8. **协方差参数化未说明**（方差、标准差还是对数），本卡取 `logσ`。
9. 假设清单与推荐默认值：

| 假设 | 推荐默认值 |
|---|---|
| d_model / 编码器宽度 w | 64 / 64 |
| 头数 | 8（论文 §2.3 以 h=8 为例） |
| 解码器 FFN 宽度 | 512，激活 ReLU |
| 空间嵌入 | Conv1d(k=3) → BN → Linear |
| BiLSTM hidden | 32/方向（拼接后等于 d） |
| 位置编码 | `Embedding(200, d)` |
| 局部注意力 | groups=4、reduction=2、沿时间维 softmax、乘 m、split-attention 融合 |
| 全局注意力 | 无输出投影 |
| 编码器 dropout 位置 | 残差 ReLU 之后 |
| 协方差头 | 输出 logσ |
| Lpv / Lev | 见 §5 公式 |
| batch / 最大 epoch / 调度 | 128 / 300 / 无 |
| 旋转 | 逐样本（IPB `gravity_world`） |
| 窗口聚合 | 均值；logstd 取“完全相关”近似 |
