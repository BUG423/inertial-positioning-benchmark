# IONet（`ionet`）

> 规格卡版本：v1（2026-09-17）· fidelity：`paper-only` · 夹具：无（无官方代码；参数量由本卡规格推导，见 §8）

本卡把论文原文明确写出的内容标为 **[P]**，本卡为了可实现而补的假设标为 **[A]**（§10 汇总了全部假设和推荐默认值）。
没有标注的内容是 IPB 协议本身的约定。

## 1. 元信息

| 项 | 值 |
|---|---|
| 论文 | *IONet: Learning to Cure the Curse of Drift in Inertial Odometry*；Changhao Chen, Xiaoxuan Lu, Andrew Markham, Niki Trigoni；AAAI 2018；<https://arxiv.org/abs/1802.02209>（本卡依据 arXiv v1） |
| 相关后续工作 | (1) OxIOD 数据集论文中的 “DeepIO”：Chen et al., *OxIOD: The Dataset for Deep Inertial Odometry*, arXiv:1809.07491, 2018，第 V 节，按 IONet 复现；(2) 期刊扩展版：Chen, Lu, Wahlström, Markham, Trigoni, *Deep Neural Network Based Inertial Odometry Using Low-Cost Inertial Measurement Units*, IEEE TMC 20(4):1351–1364, 2021。扩展版摘要说明会额外预测不确定度，正文细节**未核实**（未取得全文） |
| 官方仓库 | **无**。RoNIN 论文（arXiv:1905.12853 §5.1）写明 “IONet: We use our local implementation, as the code is not publicly available” |
| 许可 | 无官方代码。本卡只描述论文里的方法，不含任何第三方代码 |
| 框架 | 论文用 TensorFlow，在 NVIDIA TITAN X 上训练 [P]；OxIOD 的 DeepIO 用 PyTorch [P] |
| fidelity | `paper-only` |
| 参考位置 | AAAI18 §“Tracking Down A Cure”（Eq. 7–17）、§“Deep Neural Network Framework”（Eq. 18–20）、§“Training Details”；OxIOD 论文 §V-A/B（Eq. 1–4） |

## 2. 任务（输入）

| 项 | 官方 | IPB 映射 |
|---|---|---|
| 采样率 | 100 Hz。论文没有直接写频率，是从 “window length of 200 frames (2 s)” 推出来的 [P]；OxIOD 数据本身是 100 Hz | 200 Hz |
| 窗口 | 200 样本 = 2 s [P]（论文脚注：试过 50/100/200/400，选 200） | `window=400`（仍是 2 s）；模型内部先做 2× 平均池化，降到 200 步（无参数，见 §6） |
| 训练步长 | 10 样本 = 0.1 s [P] | `stride=20`（同为 0.1 s） |
| 推理步长 | 10 样本，输出 10 Hz 轨迹 [P] | `eval_stride=10`（IPB 默认值；默认适配下窗口彼此独立，不影响语义）。可选的“忠实协议模式”要求 `eval_stride=20`（§6.3） |
| 坐标系 | **机体系原始读数**：网络输入是原始 `(â, ŵ)`，初始速度和机体系重力作为潜变量由网络隐式估计（Eq. 14–16）[P] | 默认适配用 `frame=gravity_world`，这会**改变语义**（§6）；忠实协议模式用 `frame=body` |
| 姿态来源 | 不用任何姿态 [P] | 默认适配下与 benchmark 中其他 `gravity_world` 模型保持一致（默认 `orientation=reference`；主表若统一用 `device`，这里也用 `device`） |
| 去重力 | 否（重力 `g0^b` 是网络要隐式处理的潜变量）[P] | `remove_gravity=false` |
| 通道顺序 | `(a, w)`，即 `[acc_xyz, gyro_xyz]`（Eq. 18 “(a, w)200∗6”）[P]；轴向约定未说明 | 模型内部置换：`x[:, [3,4,5,0,1,2], :]` |
| 额外输入 | 无 | 无 |
| 输入归一化 | 未说明 | 不归一化 [A] |

## 3. 输出

- **官方**：每个窗口输出一个极坐标增量 `(Δl, Δψ)`，形状 `1×2`（Eq. 18）[P]。
  - `Δl`：窗口内的**水平**位移模长，单位 m。论文假设窗口内 z 向平均位移为 0（Eq. 13 前的讨论）[P]。
  - `Δψ`：窗口内的航向变化，单位 rad [P]。
  - 不输出协方差（AAAI 版）[P]。TMC 扩展版声称会预测不确定度，形式**未核实**。
- **官方轨迹**：`x_n = x_0 + Δl·cos(ψ_0 + Δψ)`，`y_n = y_0 + Δl·sin(ψ_0 + Δψ)`（Eq. 17；OxIOD Eq. 4）[P]。这需要序列起点位置 `(x_0, y_0)` 和初始航向 `ψ_0`。
  论文**没有写清**步长 10、窗口 200 的重叠窗口之间如何串联。常见的第三方解释（例如 <https://github.com/jpsml/6-DOF-Inertial-Odometry> 的 `dataset.py`，非官方）如下：
  - `Δl` 取窗口**中心**一个步长区间 `[c−r/2, c+r/2]` 内的位移模长；
  - `Δψ` 取该区间位移方向相对前一区间 `[c−3r/2, c−r/2]` 位移方向的变化；
  - 推理时先累加航向 `ψ_k = ψ_{k−1} + Δψ_k`，再累加位置，得到 10 Hz 轨迹；`ψ_0` 取自真值。
  RoNIN 在评测 IONet 前用 ICP 对齐前 5 s 轨迹（“IONet is ambiguous in rotation”），与这种解释一致。
- **OxIOD 的 DeepIO 变体**：输出 `(v̄, ψ̇) = (Δl/n, Δψ/n)`（OxIOD Eq. 1–2）[P]，即平均速度和航向角速度。
- **IPB 默认适配**：`{"vel": (B,2)}`，即 `vel = s·[cos ψ, sin ψ]`，其中 `(s, ψ)` 由同一个 2 维全连接头输出（§6）。

## 4. 网络结构

参考规格按 PyTorch 语义给出。输入为 `(B, 6, 400)`，经 IPB 适配后 LSTM 展开 200 步。

| # | 层 | 参数（核/步长/填充/通道） | 归一化 | 激活 | dropout | 输出形状 |
|---|---|---|---|---|---|---|
| 0 | 通道置换 | `[gyro, acc] → [acc, gyro]` | – | – | – | (B, 6, 400) |
| 1 | AvgPool1d（IPB 适配，无参数） | k=2, s=2, p=0 | – | – | – | (B, 6, 200) |
| 2 | 转置 | `(B, C, T) → (B, T, C)` | – | – | – | (B, 200, 6) |
| 3 | BiLSTM-1 [P] | input=6，hidden=96/方向 [P]，1 层，双向 [P]，`batch_first` | 无 | LSTM 内部 sigmoid/tanh | 0.25，作用在层输出上 [P: “Dropout in each LSTM layer … 25%”；位置 A] | (B, 200, 192) |
| 4 | BiLSTM-2 [P] | input=192，hidden=96/方向，1 层，双向 | 无 | 同上 | – | 序列 (B, 200, 192)；取 `h_n`：前向末状态与反向末状态拼接 [A] → (B, 192) |
| 5 | Dropout | p=0.25 [P] | – | – | 0.25 | (B, 192) |
| 6 | Linear [P: “outputs one polar vector”] | 192 → 2，带 bias | – | 无 | – | (B, 2) |
| 7 | IPB 头（无参数） | `s = out[:,0]`，`ψ = out[:,1]`；`vel = s·[cos ψ, sin ψ]` | – | – | – | `vel`: (B, 2) |

- **参数总量（由规格推导，非官方数字）**：
  - `hidden=96`（AAAI 版）：LSTM-1 79,872 + LSTM-2 222,720 + FC 386 = **302,978**。这是 PyTorch 的计数，每个门有 `bias_ih` 和 `bias_hh` 两个偏置向量。TensorFlow/Keras 每门只有一个偏置，计数为 **301,442**。
  - `hidden=128`（OxIOD DeepIO 版）：PyTorch 计数为 **535,042**（139,264 + 395,264 + 514），TF 计数为 532,994。
  - 参数按注册顺序的形状（PyTorch，`hidden=96`）：`(384,6) (384,96) (384,) (384,)`，然后反向同形状各一组；`(384,192) (384,96) (384,) (384,)`，然后反向同形状各一组；`(2,192) (2,)`。
    也可以写成 `nn.LSTM(6, 96, num_layers=2, bidirectional=True, dropout=0.25)`，形状序列和总量完全相同；但这种写法的 dropout 只作用在层间，所以还要在 `h_n` 之后补一个 Dropout，才能与第 5 行一致。
- IPB 默认适配**参数量与官方相同**：第 1 行和第 7 行都没有参数，FC 仍是 192→2。
- 权重初始化：论文未说明，采用框架默认值 [A]。
- 注意：论文说 “The second LSTM outputs one polar vector”，OxIOD 说 “a fully-connected layer … map the last output of LSTM”。本卡按 TF/Keras 中 `Bidirectional(LSTM(return_sequences=False))` 的语义取 `h_n` 拼接。**不要**写成 PyTorch 的 `out[:, -1, :]`：那里的反向分量只看过最后一个样本（见 §10）。

## 5. 损失与训练配方（official）

| 项 | 值 | 来源 |
|---|---|---|
| 损失 | `ℓ = Σ ‖Δl̃ − Δl‖² + κ‖Δψ̃ − Δψ‖²`（Eq. 20），κ 用来平衡两项，**数值未给出** | AAAI18 §Deep Neural Network Framework [P] |
| 优化器 | Adam | §Training [P] |
| 学习率 / 调度 | 0.0015；调度未说明 | §Training [P] |
| batch | 未给出 | – |
| epoch | 原文 “training converges typically after 100 iterations”；Fig. 8 横轴为 0–100 epoch，所以按约 100 epoch 理解 | §Training、Fig. 8 [P] |
| 权重衰减 | 未给出 | – |
| 梯度裁剪 | 未给出 | – |
| 增强 | 无；靠采集运动丰富的数据加 dropout 防过拟合 | §Training [P] |
| 阶段切换 | 无 | – |
| 选模/早停 | 看验证损失（Fig. 8），具体策略未给出 | [P]/未给出 |
| 训练数据组织 | **每种放置方式单独训练一个模型**，论文称效果优于联合训练 | §Testing [P] |
| 数据 | 自采 Vicon 数据，iPhone 7 Plus，手持/口袋/手包/手推车各约 2 h，均由 User 1 采集 | §Training Details [P] |
| DeepIO（OxIOD）差异 | 双层 BiLSTM，h=128，最后输出接 FC；Adam lr=1e-4；MSE；PyTorch；取 handheld 24、pocket 11、handbag 8、trolley 13 条序列训练 | OxIOD §V-A/B [P] |

**IPB `recipe: official`**：Adam，lr=1.5e-3，无调度，100 epoch，dropout 0.25，batch=128 [A]，κ=1 [A]，无增强，按 IPB 规则（val）选模。
**IPB `recipe: unified`**：benchmark 统一预算，损失与头部形式同 §6.2。

## 6. benchmark 适配（unified 配方）

### 6.1 改动一览

| 改动 | 类型 | 理由 |
|---|---|---|
| 模型内部把通道置换为 `[acc, gyro]` | 不改变结构 | 论文顺序是 `(a, w)`，第一层 LSTM 的权重列与通道顺序绑定 |
| `window=400` + 模型内 2× AvgPool1d | 不改变结构（无参数） | 保持官方的 2 s 时长和 200 个 LSTM 步。LSTM 参数量与序列长度无关，不降采样、直接展开 400 步也不改变参数量，但会改变时间尺度，只作为消融 |
| `stride=20` | 不改变结构 | 官方每 0.1 s 取一个训练样本 |
| 输入系从 `body` 改为 `gravity_world` | **改变语义** | 官方 `Δψ` 是相对航向变化，必须带状态累加，还需要初始航向 `ψ_0`。IPB 协议要求窗口独立、每个窗口输出世界系速度，而绝对航向只有在重力对齐世界系输入下才可观测 |
| 输出语义从 `(Δl, Δψ)` 改为 `(s, ψ)`，`s` 为窗口平均水平速度的模长（m/s），`ψ` 为其在世界系的方向角；`vel = s·[cos ψ, sin ψ]` | **改变语义**（FC 仍为 192→2，参数量不变） | 保留“模长 + 角度”的极坐标头，同时让输出能直接接到 IPB 的 `avg_velocity` 目标 |
| 损失改为 `‖s̃ − s‖² + κ·m·wrap(ψ̃ − ψ)²`，其中 `m = 1[‖v̄‖ > 0.1 m/s]`，`wrap(x) = atan2(sin x, cos x)` | **改变损失** | 近静止时方向没有定义，不加掩码会引入噪声梯度；`wrap` 避免 ±π 跳变 |
| `dims=2`，`target=avg_velocity` | 不改变结构 | 官方本来就是 2D 水平运动 |

### 6.2 目标构造（默认适配）

对窗口 `[s, s+400)`，IPB 的 `avg_velocity` 给出 `v̄ = (p[s+399] − p[s])_xy / (399·dt)`。由此：`s̃ = ‖v̄‖`，`ψ̃ = atan2(v̄_y, v̄_x)`。
推理时 `vel = s·[cos ψ, sin ψ]`。之后的轨迹重建完全按 DESIGN §5 进行，不需要任何状态。

### 6.3 可选：忠实协议模式（协议扩展，只用于复现研究，不进主表）

- 视图：`frame=body`，`window=400`，`stride=eval_stride=r=20`，输出 `(Δl, Δψ)`。
- 目标（按 §3 的第三方解释）：中心 `c = s+200`，`a = c − r/2 = s+190`，`b = c + r/2 = s+210`，`a' = a − r = s+170`。
  - `Δl = ‖(p[b] − p[a])_xy‖`
  - `Δψ = wrap(atan2(p[b]−p[a]) − atan2(p[a]−p[a']))`
- 后处理是**有状态**的，需要一个 predictor 钩子：`ψ_k = ψ_{k−1} + Δψ̂_k`，`ψ_0` 取首个窗口的参考航向（与 RoNIN 的初始对齐同类）；再令 `v_k = Δl̂_k / (r·dt) · [cos ψ_k, sin ψ_k]`，时间戳记在窗口中心，交给 DESIGN §5 积分器。
- 这个模式用到了参考航向，也违背了“窗口独立”，所以只作为附表报告。

## 7. 官方报告数值

| 数据集 | 指标（定义） | 数值 | 来源 |
|---|---|---|---|
| 自采 Vicon 房间（多用户，手持/口袋/手包） | 位置误差 CDF；最大误差（90% 测试时间内） | 约 2 m，比 PDR 改进 30–40% | AAAI18 Fig. 4、Fig. 10a 及正文 [P] |
| 多设备（iPhone 5/6/7） | 位置误差 CDF | 只有图，无表格数值（**未核实**具体数） | AAAI18 Fig. 5、Fig. 10b |
| 大场景 Floor A/B（Tango 伪真值） | 50 m / 100 m / 终点绝对误差 | 只有柱状图（**未核实**具体数） | AAAI18 Fig. 11 |
| 手推车（Vicon） | 误差 CDF | 只有图 | AAAI18 Fig. 12 |
| RIDI（第三方复现，RoNIN 论文） | ATE / RTE（m），RTE 窗口 1 min；前 5 s 用 ICP 对齐 | seen 11.46 / 14.22；unseen 12.50 / 13.38 | RoNIN arXiv:1905.12853 Table 1 |
| OxIOD（同上） | ATE / RTE | seen 1.79 / 1.97；unseen 2.63 / 2.63 | 同上 |
| RoNIN（同上） | ATE / RTE | seen 31.07 / 24.61；unseen 32.03 / 26.93 | 同上 |

注：RoNIN 表中的 IONet 是 RoNIN 作者的本地复现，并用 ICP 对齐了旋转，与 IPB “不做任何对齐”的协议不可直接比较。

## 8. 忠实性测试建议

- [ ] **参数量（由规格推导）**：`hidden=96` 时为 302,978（PyTorch 双偏置）。在测试里用 `hidden=128` 再断言 535,042，用来锁定公式。
- [ ] `param_shapes` 按注册顺序与 §4 列表一致（共 18 个张量）。
- [ ] 形状：输入 `(B,6,400)` → `vel (B,2)`；内部 LSTM 输入 `(B,200,6)`。输入 `(B,6,T)` 中 T 为奇数时应报错，或在文档里写明截断规则。
- [ ] 通道置换：交换 IPB 输入中的 gyro 与 acc 块后，模型内部看到的序列应与手工置换结果逐元素相等。
- [ ] 降采样：常值输入经 AvgPool 后仍为同一常值。
- [ ] 极坐标头：`‖vel‖ == |s|`，`atan2(vel) == wrap(ψ)`（s>0 时）。
- [ ] 损失：`ψ̃ − ψ = 2π − ε` 时，角度项约等于 `κ·ε²`，不是 `(2π)²`；`‖v̄‖ < 0.1` 时角度项为 0。
- [ ] `h_n` 取法：对随机输入，读出特征的前 96 维应等于第 2 层 `out[:, -1, :96]`，后 96 维应等于 `out[:, 0, 96:]`（确认没有误用 `out[:, -1]`）。
- [ ] 忠实协议模式：用合成的匀速圆周运动构造 `(Δl, Δψ)`，经有状态后处理重建的轨迹应与真值一致，误差小于 1e-6 m。

## 9. 预训练权重

无（没有官方代码，也没有官方权重）。

## 10. 官方实现的坑与未决问题

1. **没有官方代码**。所有已发表的 IONet 数字都来自不同的复现，协议也各不相同（RoNIN 表格用了 ICP 旋转对齐）。
2. **重叠窗口如何串联没有写清**（窗口 200、步长 10）。本卡给出默认适配（绝对航向）和忠实协议模式（中心步长区间解释）两种方案。
3. **输入是机体系，不用姿态**：论文的核心主张是网络隐式估计初始速度与机体系重力。IPB 的默认适配改为重力对齐世界系输入，这是有意的语义改变，结果表里必须注明 “IONet (gravity_world adaptation)”。
4. **`out[:, -1, :]` 陷阱**：在 PyTorch 中，双向 LSTM 最后一个时间步的反向分量只看过 1 个样本。应取 `h_n` 拼接（TF `Bidirectional(return_sequences=False)` 的语义）。
5. **偏置计数差异**：TF 与 PyTorch 的 LSTM 相差 `4h`/方向/层，`hidden=96` 时总差 1,536。IPB 以 PyTorch 计数 302,978 为准。
6. **κ 未给出**；AAAI 版和 OxIOD 版的 hidden 分别为 96 和 128，学习率分别为 1.5e-3 和 1e-4，两版不一致。
7. 假设清单（[A] 项的推荐默认值）：

| 假设 | 推荐默认值 |
|---|---|
| dropout 位置 | 每个 BiLSTM 层的输出上（第 2 层作用于 `h_n` 拼接） |
| 读出方式 | `h_n` 前向与反向拼接 |
| κ | 1.0；在 val 上扫 {0.1, 1, 10} |
| batch | 128 |
| 输入归一化 | 无 |
| 权重初始化 | 框架默认 |
| 默认适配的角度掩码阈值 | 0.1 m/s |
| 200 Hz 的处理 | window=400，模型内 AvgPool(2,2) |
| 正文“100 iterations” | 按 100 epoch 理解（Fig. 8 横轴） |
