# iMoT（`imot`）

> 规格卡版本：v1（2026-09-17）· fidelity：`paper-only` · 夹具：无（官方仓库不含代码；参数量由本卡参考规格推导，见 §8）

标注约定：**[P]** 表示论文明确写出，**[A]** 表示本卡为可实现而补的假设（汇总见 §10）。

## 1. 元信息

| 项 | 值 |
|---|---|
| 论文 | *iMoT: Inertial Motion Transformer for Inertial Navigation*；arXiv v1 作者为 Son Minh Nguyen, Linh Duy Tran, Duc Viet Le, Paul J.M Havinga；AAAI 2025（Vol. 39, No. 6, pp. 6209–6217，<https://ojs.aaai.org/index.php/AAAI/article/view/32664>；OJS 的 APA 引文只列出 Nguyen、Le、Havinga 三位作者，与 arXiv 不一致，**未核实**原因）；<https://arxiv.org/abs/2412.12190>（本卡依据 arXiv v1，2024-12-13，含附录） |
| 官方仓库 | <https://github.com/Minh-Son-Nguyen/iMoT> @ `7f275702bd9b0dbdb60bb4d5dae802f8e9ab9dac`，论文首页也标了这个地址。**只有 `LICENSE` 和 `README.md`**，README 写着 “will be made publicly available soon”。仓库标题为 “…for Indoor Navigation”，与论文标题不同。本地未克隆 |
| 许可 | 仓库 LICENSE 为 MIT（Copyright 2024 Minh-Son-Nguyen）。由于没有代码，对本复现没有实际约束 |
| 框架 | PyTorch 2.4.0，H100 80 GB [P] |
| fidelity | `paper-only` |
| 参考位置 | “Proposed Method”一节：Encoder（Eq. 1–3，Fig. 2–3）、Decoder（Eq. 4–10）；“Implementation Details”；Table 1–3；附录（基线设置、计算量） |

## 2. 任务（输入）

| 项 | 官方 | IPB 映射 |
|---|---|---|
| 采样率 | 各数据集原生频率：200 Hz（RIDI、RoNIN）或 100 Hz（OxIOD、IDOL）[P] | 200 Hz |
| 窗口 | 1 s：`A ∈ R^{2D×T}`，D=3，T 为 1 s 内的样本数 [P]。token 维度在 100 Hz 数据上为 100，在 200 Hz 数据上为 200 [P] | `window=200`，T=200 |
| 训练步长 | 未给出 | `stride=10` [A] |
| 推理步长 | 未给出 | `eval_stride=10` |
| 坐标系 | 未说明。基线按 RoNIN 原实现运行，速度是 2D（x、y）[P] | `frame=gravity_world` [A] |
| 姿态来源 | 未说明 | benchmark 全局 `orientation` [A] |
| 去重力 | 未说明 | `remove_gravity=false` [A] |
| 通道顺序 | `A = {A_a, A_g}`：先加速度后角速度，各 3 轴 [P] | 模型内部置换：`x[:, [3,4,5,0,1,2], :]` |
| token 化 | **变量 token**：每个通道的整条长度为 T 的序列作为一个 token（iTransformer 式），而不是每个时刻一个 token [P] | 模型内部实现 |
| 额外输入 | 无 | 无 |
| 输入归一化 | 未说明 | 不归一化 [A] |

## 3. 输出

- **官方**：每个窗口输出一个 2D “瞬时速度段” `v_m ∈ R^{1×2}` [P]。它是 P 个查询运动粒子 `v̂ ∈ R^{P×2}` 经动态打分（DSM）后的加权结果（Eq. 10）[P]。不输出协方差 [P]。
- **目标定义**：论文没有写明“velocity segment”具体对应窗口内哪个速度量 [A]。IPB 取 `avg_velocity`。
- **官方轨迹**：“Trajectory reconstruction is performed by doing integration of predicted velocity segments”（附录）[P]；时间戳与首尾处理未说明。
- **IPB**：`{"vel": v_m (B,2), "aux": {"particles": (B,P,2)}}`。

## 4. 网络结构

参考规格（T=200，D=3，P=128，编码器 N=2、解码器 M=2，头数 8，FFN 宽度 4T，dropout 0.1）。token 布局固定为 18 个：`[acc_raw(3), acc_trend(3), acc_seasonal(3), gyro_raw(3), gyro_trend(3), gyro_seasonal(3)]`。

### 4.1 编码器（N=2 [P]，逐层结构相同）

| # | 层 | 参数 | 归一化 | 激活 | dropout | 输出形状 |
|---|---|---|---|---|---|---|
| 0 | 通道置换与 token 化 | `[gyro, acc] → [acc, gyro]`；每个通道一个 T 维 token [P] | – | – | – | (B, 6, 200) |
| 1 | PSD（Progressive Series Decoupler）[P]，放在每个编码层最前面 | 取当前 6 个 raw 槽（第 1 层为输入 A；第 j>1 层为上一层输出中的 raw 槽 [A]），计算 `A_t = MA_{k2}(MA_{k1}(pad(A)))`，其中 k1=9、k2=3 [P]，为居中滑动平均，stride=1，replicate padding [A]，保持长度 [P]；`A_s = A − A_t` [P]。然后重组为 18 个 token（`Ã ∈ R^{6D×T}` [P]），trend/seasonal 槽被新值替换 [A]。**无参数** | – | – | – | (B, 18, 200) |
| 2 | APE（Adaptive Positional Encoding）[P] | 基础编码 `E_A ∈ R^{D×T}` 为正弦编码 [P]：位置为轴序号 0..2，特征维为 T [A]，沿 token 维平铺 3 份得到 (9, T) [A]。`Ẽ = [MLP_a(Ã_a) ⊙ E, MLP_g(Ã_g) ⊙ E]` [P]；MLP 逐 token 为 Linear(T,T)→ReLU→Linear(T,T)，acc 与 gyro 各一个 [A] | – | ReLU [A] | – | Ẽ：(B, 18, 200) |
| 3 | 多头自注意力 | `q = k = Ã + Ẽ`，`v = Ã`（Eq. 3）[P]；`nn.MultiheadAttention(200, 8)` [A: 头数] | – | – | 0.1 [A] | (B, 18, 200) |
| 4 | 残差 1：`LN(X + Drop(Attn) + ASC₁(X))` | ASC 放在**每个**残差连接上 [P]；post-LN [A] | LayerNorm(200) [A] | – | 0.1 | (B, 18, 200) |
| 4a | ASC-① 跨通道卷积 | 按时间维拼接 acc 与 gyro 的对应槽，得到 `Ã_T ∈ R^{3D×2T}`，即 (B, 9, 400) [P]；沿 9 个 token 的方向做 “1×3 conv” [P]：`Conv1d(400→400, k=3, p=1)`（400 维当作通道，token 维当作长度）[A] | – | – | – | (B, 400, 9) |
| 4b | ASC-② 通道门控 | 沿 token 维做 GAP → `Conv1d(400→400, k=1)` → sigmoid，得到 `W ∈ R^{2T}` [P]；与 ① 的结果逐元素相乘 [A] | – | sigmoid [P] | – | (B, 400, 9) |
| 4c | ASC-③ 投影回 token 空间 | reshape 为 (B, 9, 400)，拆成 acc 与 gyro 两半后拼回 (B, 18, 200) [A]，再做 “1×1 conv + GELU” [P]：逐 token `Linear(200→200)` 后接 GELU | – | GELU [P] | – | (B, 18, 200) |
| 5 | FFN | Linear 200→800 → GELU → Dropout → Linear 800→200 [A] | – | GELU [A] | 0.1 | (B, 18, 200) |
| 6 | 残差 2：`LN(X + Drop(FFN) + ASC₂(X))` | 同第 4 行 | LayerNorm(200) | – | 0.1 | 编码输出 X：(B, 18, 200)，同时输出本层的 Ẽ |

### 4.2 解码器（M=2 [P]）

| # | 层 | 参数 | 归一化 | 激活 | dropout | 输出形状 |
|---|---|---|---|---|---|---|
| 7 | 查询运动粒子 `v̂⁰` | 可学习参数，形状 (P, 2) [P: learnable]；初值为 U[−1, 1] m/s [A] | – | – | – | (B, 128, 2) |
| 8 | 内容特征 `C⁰` | 全零 [P]，不是参数 | – | – | – | (B, 128, 200) |
| 9 | 粒子位置编码 `E_v̂ʲ = MLP_pos(PE(v̂^{j−1}))` | PE 为正弦编码（Eq. 4）[P]：每个轴 T/2 维，DAB-DETR 式，温度 10000，尺度 2π [A]；`MLP_pos` 为 Linear(200,200)→ReLU→Linear(200,200)，**各层共享** [A] | – | ReLU | – | (B, 128, 200) |
| 10 | 自注意力（Eq. 5） | `q = k = C^{j−1} + E_v̂ʲ`，`v = C^{j−1}` [P]；`MHA(200, 8)`；`C_saʲ = LN(C + Drop(SA))` [A] | LN [A] | – | 0.1 | (B, 128, 200) |
| 11 | 位置缩放 | `qpos = MLP_s(C^{j−1}) ⊙ E_v̂ʲ` [P]；`MLP_s` 为 Linear(200,200)→ReLU→Linear(200,200)，每层独立 [A] | – | ReLU | – | (B, 128, 200) |
| 12 | 交叉注意力 ×2（acc、gyro 各一个，Eq. 6） | `query = [C_saʲ, qpos]`（400 维），`key = [Ã_a, Ẽ_a]`（400 维，取编码器最后一层的 9 个 acc token 及其 Ẽ），`value = Ã_a`（200 维）[P]；gyro 分支同理 [P]。投影：Wq 400→400，Wk 400→400，Wv 200→200，Wo 200→200，8 头（q/k 每头 50 维，v 每头 25 维）[A] | – | softmax | 0.1（注意力权重）[A] | C_a、C_g：(B, 128, 200)；注意力图 (B, 8, 128, 9) |
| 13 | 融合 | `Cʲ = LN(C_saʲ + Drop(MLP_c([C_a, C_g])))` [P: MLP 融合；A: 残差与 LN]；`MLP_c` 为 Linear(400,200)→ReLU→Linear(200,200) | LN [A] | ReLU | 0.1 | (B, 128, 200) |
| 14 | FFN | Linear 200→800 → ReLU → Dropout → Linear 800→200，残差后接 LN [A]（论文未提到解码器 FFN） | LN | ReLU | 0.1 | (B, 128, 200) |
| 15 | 粒子细化（Eq. 7） | `Δv̂ʲ = MLP_Δ(Cʲ)`，`v̂ʲ = v̂^{j−1} + Δv̂ʲ` [P]；`MLP_Δ` **各层共享** [P]，为 Linear(200,200)→ReLU→Linear(200,2) [A]；层间不 detach [A] | – | ReLU | – | (B, 128, 2) |
| 16 | DSM（Eq. 10） | `S^d = MLP_dsm(v̂ᵀ)`，其中 `v̂ᵀ ∈ R^{2×P}` [P]：x、y 两个方向分别输入 P 维，经 Linear(128,128)→ReLU→Linear(128,128)，再沿 P 维 softmax [A: 归一化方式]；`v_m = Σ_p S^d[:,:,p] · v̂[:,p,:]` [P: `v_m = S^d · v̂_p`] | – | softmax [A] | – | `vel`：(B, 2) |

参数分解（参考规格，由 PyTorch 原型推导）：

| 模块 | 参数量 |
|---|---|
| 每个编码层 | 2,005,400。其中 APE 两个 MLP 共 160,800；MHA 160,800；ASC 两个共 1,362,000，每个 681,000（conv3 480,400 + gate 160,400 + proj 40,200）；FFN 321,000；2 个 LN 共 800 |
| 编码器（2 层） | 4,010,800 |
| 粒子 `v̂⁰` | 256 |
| `MLP_pos`（共享） | 80,400 |
| `MLP_Δ`（共享） | 40,602 |
| 每个解码层 | 1,486,200。其中 SA 160,800；MLP_s 80,400；两个交叉注意力 802,400；MLP_c 120,400；FFN 321,000；3 个 LN 共 1,200 |
| 解码器（2 层） | 2,972,400 |
| DSM | 33,024 |
| **合计** | **7,137,482** |

- 论文 Table 3 报告：200 Hz 下参数量为 **14.49M**，GFLOPs/s 为 7.79。参考规格只有它的约 49%。把 FFN 宽度改为 PyTorch 默认的 2048，合计也只有 9,139,274。论文未给出的宽度或模块（例如 MLP 深度、ASC 的具体卷积形式）必然大得多，**无法由论文唯一确定**。
- T=100（100 Hz 数据）时，8 头无法整除 100，必须改用例如 4 头；此时合计为 1,815,382。IPB 统一为 200 Hz，不涉及这一情况。
- 权重初始化：未说明，采用 PyTorch 默认 [A]。

## 5. 损失与训练配方（official）

| 项 | 值 | 来源 |
|---|---|---|
| 损失（最终模型） | 只用 `J_vel = (1/B)·Σ_b ‖v_GT − v_m‖²`，其中 `v_m` 由 DSM 给出 | Eq. 8（J_vel 部分）、Eq. 10 及其后文字 [P]：“iMoT can be efficiently optimized using only the velocity loss J_vel” |
| 不带 DSM 的变体（消融用） | `S_p = softmax_p(−d(v̂_p, v_GT))`，`v_m = Σ_p (1 − S_p)^γ · v̂_p`，再加熵损失 `J_ent = (1/BP)·Σ −S log S + ε`，ε=1e-10；测试时用粒子的平均池化。γ 的数值未给出 | Eq. 8–9 [P] |
| 优化器 | Adam | Implementation Details [P] |
| 学习率 / 调度 | 1e-4；调度未给出 | [P] |
| batch | 128 | [P] |
| epoch | 未给出 | – |
| 权重衰减 | 未给出 | – |
| 梯度裁剪 | 未给出 | – |
| 增强 | 未给出 | – |
| 阶段切换 | 无（最终模型只用 J_vel） | [P] |
| 选模/早停 | 未给出 | – |
| 结构超参 | k1=9，k2=3；P=128（Fig. 4 显示 128 最优）；N=2，M=2；token 维度 = 1 s 的样本数 | [P] |
| 数据集 | RIDI、RoNIN、OxIOD、IDOL；RIDI 与 OxIOD 按放置方式预先配置 | Dataset 一节 [P] |

**IPB `recipe: official`**：Adam，lr 1e-4，batch 128，无调度 [A]，最多 100 epoch [A]，按 val 选模，不做增强 [A]，只用 J_vel。
**IPB `recipe: unified`**：benchmark 统一预算，其余同上。

## 6. benchmark 适配（unified 配方）

| 改动 | 类型 | 理由 |
|---|---|---|
| 模型内部置换为 `[acc, gyro]` | 不改变结构 | 论文 token 顺序为 `{A_a, A_g}`；APE 与交叉注意力都区分这两种模态 |
| T = 200 | 不改变结构 | 与官方 200 Hz 配置相同。对 OxIOD、IDOL，官方用 T=100（1 s），IPB 重采样后用 T=200（仍为 1 s），token 维度随之翻倍，参数量也随之变化（这是论文自身按采样率设定的规则） |
| `frame=gravity_world`，`orientation` 取全局设置 | 协议假设 | 论文未说明输入系，而其 RoNIN 系基线使用重力对齐世界系 |
| 目标 `avg_velocity`，`dims=2` | 协议假设 | 论文未定义 velocity segment 的精确含义 |
| 训练与推理步长均为 10 | 协议统一 | 论文未给出 |
| 输出 `vel = v_m` | 不改变结构 | 窗口级 2D 速度，可直接接入 IPB |

## 7. 官方报告数值

指标定义（论文与附录）：ATE 为整条轨迹 RMSE；T-RTE 为 1 min 时间窗的相对误差 RMSE；D-RTE 为每 1 m 距离的相对误差 RMSE；PDE 为末端漂移 / 总路程。单位均为 m。

Table 2（iMoT 一列）：

| 数据集 | seen：ATE / T-RTE / D-RTE | unseen：ATE / T-RTE / D-RTE |
|---|---|---|
| RIDI | 1.68 / 1.91 / 0.21 | 1.49 / 1.33 / 0.20 |
| RoNIN | 3.78 / 2.68 / 0.26 | 5.31 / 4.39 / 0.36 |
| OxIOD | 1.86 / 0.94 / 0.21 | 0.90 / 1.32 / 0.22 |
| IDOL | 2.22 / 1.86 / 0.24 | 3.00 / 2.85 / 0.28 |

同表中作者复现的基线（RoNIN unseen ATE）：RoLSTM 6.90，RoTCN 7.19，RoResnet18 5.95，CTIN 6.89，TLIO 6.77。

Table 1（RoNIN unseen 消融）：完整模型 (xv) 为 ATE 5.31 / T-RTE 4.39 / D-RTE 0.36；去掉所有模块的基线 (i) 为 6.48 / 5.53 / 0.41；只加粒子、不加 DSM 的 (ii) 为 7.10 / 4.61 / 0.42。

Table 3（附录，200 Hz）：iMoT 14.49M 参数，7.79 GFLOPs/s。同表 CTIN 0.56M / 7.27；RoLSTM 0.21M / 7.17；RoTCN 2.03M / 33.17；RoResnet18 4.64M / 9.16。

## 8. 忠实性测试建议

- [ ] **参数量（参考规格推导，非官方）**：合计 7,137,482；各子模块计数与 §4 分解表一致。注释中写明官方为 14.49M，差距来源见 §4。
- [ ] 形状：输入 `(B,6,200)` → `vel (B,2)`，`aux.particles (B,128,2)`；编码器 token 为 (B,18,200)；交叉注意力图为 (B,8,128,9)。
- [ ] 通道置换：模型内部 raw 槽 0:3 等于 IPB 输入的 3:6（acc），raw 槽 9:12 等于输入的 0:3（gyro）。
- [ ] PSD 性质（无参数，可精确测试）：
  - 常值序列：trend 等于该常值，seasonal ≡ 0；
  - 输出长度 = T；
  - 线性斜坡序列在远离边界处 seasonal ≈ 0（居中对称滑动平均）；
  - `A_t + A_s == A`。
- [ ] APE：把 `MLP_a` 的输出置为全 1 时，`Ẽ_a` 等于平铺后的正弦编码。
- [ ] DSM：S 沿 P 维求和为 1；`v_m` 的每个分量落在该轴所有粒子取值的 [min, max] 之内（凸组合）。
- [ ] 粒子细化：`MLP_Δ` 与 `MLP_pos` 在各层之间是同一对象（参数只计一次）。
- [ ] 损失：`v_m == v_GT` 时 `J_vel = 0`。

## 9. 预训练权重

无（官方仓库尚未发布代码或权重）。

## 10. 官方实现的坑与未决问题

1. **没有代码**，官方仓库只是占位。14.49M 参数量无法由论文复原，参考规格只有约 7.14M。
2. **维度记号不自洽**：`E_A ∈ R^{D×T}` 与 `MLP(Ã_a) ∈ R^{3D×T}` 逐元素相乘，需要平铺或广播；Eq. 6 虽然写作 “SelfAttn”，实际是交叉注意力；“Progressive” PSD 在第 j>1 层作用于哪些 token 也没有说明。
3. **token 数随 PSD 增长的问题**：若每层都对全部 18 个 token 再分解，token 数会变成 54；本卡只对 raw 槽重新分解。
4. **头数与 T=100 不兼容**：T=100 时不能用 8 头。
5. **DSM 归一化未说明**：本卡取 softmax，使 `v_m` 为凸组合。若不归一化，`v_m` 的尺度将没有约束。
6. **“permutation-invariant to token orders” 的说法**：ASC 沿 token 维做 k=3 卷积，模型对 token 顺序并不置换不变，所以不设这一测试。
7. **输入坐标系、目标定义、步长均未说明**，与 RoNIN 系比较时要注意协议差异。另外，作者复现的 CTIN 在 RoNIN unseen 上为 6.89，而 CTIN 原文为 5.61。
8. 假设清单与推荐默认值：

| 假设 | 推荐默认值 |
|---|---|
| 输入系 / 目标 | `gravity_world` / `avg_velocity` |
| 步长 | 10 |
| 头数 | 8 |
| dropout | 0.1 |
| FFN | 4T；编码器用 GELU，解码器用 ReLU |
| 归一化 | post-LN |
| PSD | 每层只对 raw 槽重新分解；replicate padding |
| APE | 正弦编码的位置取轴序号并平铺 3 份；acc、gyro 各一个两层 MLP |
| ASC | Conv1d(2T,2T,3) 沿 token 维；门控 Conv1d(2T,2T,1)；逐 token Linear(T,T) + GELU |
| 粒子 | 初值 U[−1,1]；PE 为 DAB-DETR 式（温度 1e4，尺度 2π）；`MLP_pos` 与 `MLP_Δ` 各层共享；`MLP_s` 每层独立；层间不 detach |
| 交叉注意力 | q/k 为 2T 维，v 为 T 维（Conditional-DETR 式） |
| 解码器 | 带 FFN |
| DSM | 两层 MLP(P,P) + softmax |
| 训练 | 最多 100 epoch；无调度；无增强 |
