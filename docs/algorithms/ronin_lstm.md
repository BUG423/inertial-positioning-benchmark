# RoNIN-LSTM（`ronin_lstm`）

> 规格卡版本：v1（2026-09-17）· fidelity：`official-code` · 夹具：`tests/fixtures/algorithms/ronin_lstm.json`

## 1. 元信息

| 项 | 值 |
|---|---|
| 论文 | *RoNIN: Robust Neural Inertial Navigation in the Wild: Benchmark, Evaluations, & New Methods*；Herath, Yan, Furukawa；ICRA 2020；<https://arxiv.org/abs/1905.12853> |
| 官方仓库 | <https://github.com/Sachini/ronin> @ `805b7f0f28bb164ce89ada9ac05a9470dbe3d715`（本地：`/workspace/webCodex/third_party/ronin`） |
| 许可 | GPL-3.0（独立实现，不得复制代码） |
| 框架 | PyTorch |
| fidelity | official-code |
| 参考文件 | `source/model_temporal.py`（`BilinearLSTMSeqNetwork`、`LSTMSeqNetwork`）、`source/ronin_lstm_tcn.py`（损失/训练/测试）、`source/data_glob_speed.py`（`SequenceToSequenceDataset`）、`source/transformations.py`（`RandomHoriRotateSeq`）、`config/temporal_model_defaults.json` |

**“双向”核实结论**：官方 `--type lstm_bi` 中的 `bi` 指 **Bilinear**，不是 bidirectional。两个 LSTM 类都用 `torch.nn.LSTM(..., batch_first=True, dropout=dropout)`，未设 `bidirectional`，即**单向**（夹具导出时 `lstm.bidirectional == False`）。论文 §4.2 原文为 “a stacked **unidirectional** LSTM while enriching its input feature by concatenating the output of a bilinear layer … three layers each with 100 units”。因此 IPB 的 `ronin_lstm` = 官方 `--type lstm_bi`（`BilinearLSTMSeqNetwork`）。官方另有不带 bilinear 的 `--type lstm`（`LSTMSeqNetwork`，205,832 参数），作为变体记录在夹具 `plain_lstm_variant` 中，不作为主模型。

## 2. 任务（输入）

| 项 | 官方 | IPB 映射 |
|---|---|---|
| 采样率 | 200 Hz | 200 Hz |
| 窗口 | 训练/验证：400 帧 = 2 s 的 seq2seq 片段（`window_size: 400`）。测试：**整条序列一次前向**（batch=1） | `window=400`；推理见 §6 |
| 训练步长 | 100 帧（`step_size: 100`），加 `randrange(-50, 50)` 随机偏移（论文：每 k∈[50,150] 帧取一段） | `stride=100` + `time_shift(±50)` |
| 推理步长 | 不适用（整序列逐帧输出） | `eval_stride=10` |
| 坐标系 | HACF（重力对齐世界系），与 ResNet 相同的 `GlobSpeedSequence` | `frame=gravity_world` |
| 姿态来源 | 测试 game RV（`grv_only=True`）；训练/验证：game RV 对齐误差 < 20° 时用 game RV，否则选误差最小的源（同 `ronin_resnet18` 卡 §2） | 测试 `orientation=device`；训练同 `ronin_resnet18` 卡 §6 |
| 去重力 | 否 | `remove_gravity=false` |
| 通道顺序 | `[glob_gyro_xyz, glob_acce_xyz]`，张量为 **batch-first `(B, T, 6)`** | 与 IPB 通道顺序相同；包装层需 `(B,6,T) → (B,T,6)` 转置 |
| 额外输入 | 无 | 无 |
| 输入归一化 | 无；特征序列去掉最后一帧（`features[:-1]`）以与逐帧速度对齐 | — |
| 样本过滤 | 训练/验证样本若 `[j−450, j+50)` 内任一帧真值速度模长 > 3.0 m/s（`max_velocity_norm`）则丢弃 | 需视图支持 `max_speed_filter=3.0`（可选） |

样本定义（`SequenceToSequenceDataset`，`data_glob_speed.py:188-252`）：候选终点 `j ∈ {450, 550, …}`，`j ← j + randrange(−50,50)` 并钳制到 `[400, len−1]`；输入 `features[j−400 : j]`，目标 `targets[j−400 : j]`，其中逐帧目标 `v_k = (p[k+1] − p[k]) / (t[k+1] − t[k])`（`interval=1`，只取 xy）。

## 3. 输出

- 模块输出：逐帧 `(B, T, 2)`，含义为世界系水平速度（m/s）；无不确定度。
- 官方轨迹（`ronin_lstm_tcn.py:315-330,386-396`）：整条序列 `(1, N−1, 6)` 一次前向（隐藏状态从 0 开始，贯穿全序列），得到逐帧速度 `v̂_k`；`pos = cumsum(v̂ · dt)` 且把 `pos[0]` 覆盖为 `p0`，即 `p̂_k = p0 + Σ_{i=1..k} v̂_i · dt`（`dt` 为帧间隔均值 ≈ 5 ms），与 `gt_pos[:N−1]` 逐帧比较。
- 官方 ATE/RTE：与 `ronin_resnet18` 相同，按坐标分量求均值（欧氏值的 1/√2），RTE Δ=12000 帧。

## 4. 网络结构

`BilinearLSTMSeqNetwork(input_size=6, out_size=2, batch_size=B, device, lstm_size=100, lstm_layers=3, dropout=0)`（`model_temporal.py:53-97`，配置 `layers: 3, layer_size: 100`）。

逐层表（输入 `(B, 400, 6)`，形状省略 batch 维）：

| # | 层 | 参数 | 归一化 | 激活 | dropout | 输出形状 |
|---|---|---|---|---|---|---|
| 0 | 输入 x | — | — | — | — | (400, 6) |
| 1 | Bilinear | `nn.Bilinear(6, 6, 24)`，两个输入都是 x：`m_k = xᵀ A_k x + b_k` | — | — | — | (400, 24) |
| 2 | concat | `[x, m]`（x 在前） | — | — | — | (400, 30) |
| 3 | LSTM | `nn.LSTM(30, 100, num_layers=3, batch_first=True, dropout=0)`，单向；`(h0, c0)` 全零 | — | LSTM 内部 tanh/sigmoid | 层间 dropout=0（`--dropout` 额外参数可改） | (400, 100) |
| 4 | concat | `[input_mix(30), lstm_out(100)]`（input_mix 在前） | — | — | — | (400, 130) |
| 5 | Linear1 | 130→10（= out_size×5） | — | **无** | 无 | (400, 10) |
| 6 | Linear2 | 10→2 | — | — | — | (400, 2) |

- 参数量：**216,620**（官方配置 = benchmark 配置，夹具锁定）。分段：Bilinear 888（24×6×6 + 24）；LSTM 214,400（第 0 层 4·100·(30+100)+800 = 52,800；第 1、2 层各 80,800）；Linear1 1,310；Linear2 22。
- 参数注册顺序：`bilinear.weight (24,6,6)`、`bilinear.bias`、每层 LSTM 的 `weight_ih_l{i}, weight_hh_l{i}, bias_ih_l{i}, bias_hh_l{i}`（门顺序 i,f,g,o，PyTorch 约定）、`linear1.*`、`linear2.*`。
- 变体 `--type lstm`（`LSTMSeqNetwork`）：LSTM(6,100,3) → Linear(100,10) → Linear(10,2)，无 bilinear、无拼接，205,832 参数。
- 初始化：全部 PyTorch 默认（Bilinear/Linear 为 `U(−1/√fan_in, 1/√fan_in)` 系列，LSTM 为 `U(−1/√100, 1/√100)`），无自定义初始化。
- 隐藏状态：`init_weights()` 每次前向都新建形状为 `(3, batch_size, 100)` 的零张量，其中 `batch_size` 在**构造时固定**（训练 72，测试 1）。

## 5. 损失与训练配方（official）

| 项 | 值 | 来源 |
|---|---|---|
| 损失 | `GlobalPosLoss(mode='full')`：`P = cumsum(targ[:, 1:], dim=1)`，`P̂ = cumsum(pred[:, 1:], dim=1)`，`loss = mean((P̂ − P)²)`，对 batch、T−1 个前缀、2 个分量求均值。**不乘 dt**，所以“位置”单位是 m/s·帧（= 200 × 米）。论文称为 latent velocity loss | `ronin_lstm_tcn.py:35-60,131-139` |
| 优化器 | Adam，lr 3e-4，无权重衰减 | `ronin_lstm_tcn.py:193`；config `lr: 0.0003` |
| 学习率调度 | `ReduceLROnPlateau('min', patience=10, factor=0.75, eps=1e-12)`，**仅当传入 `--use_scheduler`** 时按验证损失调用（README 示例命令带该参数） | `ronin_lstm_tcn.py:194-196,285-286` |
| batch | 72，`drop_last=True`（训练与验证都丢最后不足一批，因隐藏状态 batch 固定） | config；`ronin_lstm_tcn.py:157-165` |
| epoch | 1000（config）；论文：约 300 epoch 收敛、40 小时 | config；论文 §5 |
| 权重衰减 / 梯度裁剪 | 无 / 无 | — |
| 增强 | `RandomHoriRotateSeq([0,3,6], [0,2])`：每个片段取 `a~U[0,2π)`，用四元数 `[cos a, 0, 0, sin a]`（绕 z 旋转 2a，分布仍均匀）同时旋转陀螺、加计三维向量与 2D 目标；随机时间偏移 ±50 | `transformations.py:84-114`，`ronin_lstm_tcn.py:79-82` |
| dropout | 代码：线性层无 dropout，LSTM 层间 dropout 默认 0。论文写 LSTM/TCN 线性层 keep 0.8——**代码未实现**，以代码为准 | `model_temporal.py:78-80`，论文 §5 |
| 平滑 | config 中 `feature_sigma: 0.001, target_sigma: 0.0`，但数据集读取键名写成 `'feature_sigma,'`（多一个逗号），**平滑永远不生效** | `data_glob_speed.py:208-213` |
| 阶段切换 | 无 | — |
| 选模/早停 | 验证损失（同一 `GlobalPosLoss`）创新低时保存；否则每 20 epoch（`save_interval`）保存一次；训练损失为 NaN 时停止；无早停 | `ronin_lstm_tcn.py:275-302` |
| 设备 | config 默认 `"device": "cpu"`，不覆盖时在 CPU 上训练 | config |

## 6. benchmark 适配（unified 配方）

| 改动 | 类型 | 理由 |
|---|---|---|
| 包装层：`imu (B,6,T)` → 转置为 `(B,T,6)` → 官方模块 → 逐帧 `y (B,T,2)` | 不改变结构 | IPB 统一输入为通道优先 |
| 隐藏状态按实际 batch 置零（等价于 `nn.LSTM` 的 `hx=None`） | 不改变结构 | 官方把 batch 固定在构造时，数值上与零初始状态相同 |
| **窗口级速度取法**：`vel = mean_{k=0..T−2} y_k`（去掉最后一帧），`aux = {"frame_vel": y}` | 不改变结构（只加无参池化） | `y_k` 估计 `(p[k+1]−p[k])/dt`，其在 `k=0..T−2` 上的均值恰为 IPB `avg_velocity = (p[s+T−1]−p[s])/((T−1)dt)`；也与 latent velocity loss 的“逐帧向量求和 = 位移”语义一致 |
| `window=400`、`stride=100`、`time_shift(±50)`，推理 `eval_stride=10`，速度时间戳为窗口中心（IPB §5） | 不改变结构 | 400 帧与官方训练展开长度一致，训练与推理的上下文长度相同 |
| official 配方损失：官方 `GlobalPosLoss(full)`，需要视图提供窗口内逐帧目标 `aux_target=frame_velocity`，形状 `(B,T,2)`，`v_k = (p[s+k+1]−p[s+k])/dt`，`k=0..T−1`（需要窗口后一个样本，窗口不得以序列最后一个样本结尾） | 不改变结构（需要视图扩展） | DESIGN §4 要求保留专用损失；官方损失用的是逐帧目标 |
| unified 配方损失：同样使用 `GlobalPosLoss(full)`；只有在视图不提供逐帧目标时才退化为 `MSE(vel, avg_velocity)`，且须在结果中标注 | 不改变结构 | 保持专用损失；退化形式只约束窗口端点 |
| **推理方式差异**：IPB 默认对每个 400 帧窗口从零状态独立推理；官方是整序列流式推理。建议实现可选钩子 `forward_sequence(imu_full) → (1,N,2)`，由预测器在整序列输出上对每个 IPB 窗口按上式取均值，作为 `ronin_lstm@stream` 单独报告 | 不改变结构 | 窗口模式与训练条件一致、协议统一；流式模式与官方测试一致，用于忠实性核对 |
| 不加论文提到、代码未实现的线性层 dropout；不复刻平滑键名错误（即不做平滑） | 不改变结构 | 以官方代码实际行为为准 |
| `max_velocity_norm=3.0` 样本过滤：official 配方保留，unified 可关闭 | 不改变结构 | 只影响训练样本选择 |

## 7. 官方报告数值

论文 Table 1，RoNIN LSTM 列（官方代码口径，见 `ronin_resnet18` 卡 §3）：

| 数据集 | 测试集 | ATE (m) | RTE (m) |
|---|---|---|---|
| RIDI | seen | 2.00 | 2.64 |
| RIDI | unseen | 2.08 | 2.10 |
| OxIOD | seen | 2.02 | 2.33 |
| OxIOD | unseen | 7.12 | 5.42 |
| RoNIN | seen | 4.18 | 2.63 |
| RoNIN | unseen | 5.32 | 3.58 |

论文用完整 RoNIN 数据集，公开版约 50%。

## 8. 忠实性测试建议

- [ ] 参数量 216,620；`param_shapes` 与夹具逐项一致（18 个张量）；`plain_lstm_variant` 若实现则为 205,832。
- [ ] LSTM 为单向：`bidirectional == False`，每层 `weight_hh` 形状 `(400, 100)`。
- [ ] 模块输入 `(4, 400, 6)` → 逐帧输出 `(4, 400, 2)`；包装层输入 `(4, 6, 400)` → `vel (4, 2)`，`aux["frame_vel"] (4, 400, 2)`。
- [ ] 池化性质：若把模块替换为恒等于常数 c 的输出，则 `vel == c`；若逐帧输出为 `(p[k+1]−p[k])/dt`，则 `vel` 与 IPB `avg_velocity` 相等（合成轨迹，容差 1e-6）。
- [ ] 因果性：修改 `t ≥ t0` 的输入不改变 `t < t0` 的逐帧输出。
- [ ] batch 无关性：同一样本在 batch=1 与 batch=7 中输出一致（验证去掉了固定 batch 的隐藏状态）。
- [ ] 损失：`GlobalPosLoss(full)` 对 `pred == targ` 为 0；对 `pred = targ + δ`（常数 δ）损失为 `δ²·mean_j (j²)`，`j=1..T−1`（验证未乘 dt、从第 1 帧开始累加）。
- [ ] 流式钩子：`forward_sequence` 在长度 800 的输入上，前 400 帧输出与窗口模式第一个窗口的输出一致。

## 9. 预训练权重

与 `ronin_resnet18` 相同：README 指向 FRDR DOI 10.20383/102.0543，2026-09-17 记录页无文件；且预训练模型在整个数据集（含测试集）上训练，不可用于 IPB 测试评测。

## 10. 官方实现的坑与未决问题

1. `lstm_bi` 是 bilinear，不是 bidirectional（见 §1）；论文里的 “RoNIN LSTM” 对应 `lstm_bi`。
2. 隐藏状态 batch 在构造时固定，训练/验证必须 `drop_last=True`，测试前把 `args.batch_size` 改成 1 再构造模型；IPB 实现应去掉这一耦合。
3. `Linear1 → Linear2` 之间没有非线性，两层线性等价于一个秩 ≤ 10 的线性映射（参数量仍按两层计）。
4. 平滑参数键名错误（`'feature_sigma,'`），config 中的平滑设置不生效。
5. 论文所述线性层 dropout（keep 0.8）未在代码中实现。
6. `RandomHoriRotateSeq.__call__` 带 `@numba.jit`，且用到 `np.math`（NumPy 2 已移除）；在新版 numba/NumPy 上训练增强会失败。`np.int`（`ronin_lstm_tcn.py:316`）同样在 NumPy ≥ 1.24 报错。
7. 测试时 `pos[0]` 被覆盖为起点，首个预测速度被丢弃，轨迹相对真值有 1 帧的索引偏移（影响可忽略）。
8. config 的 `device: "cpu"` 使默认训练在 CPU 上进行。
9. 与 ResNet 共用 `--cache_path` 会读到 interval=200 的目标（缓存不校验 interval）。
10. 损失的“位置”未乘 dt，数值为米制位置的 200 倍、损失为 4×10⁴ 倍；移植时不要“修正”，否则学习率等超参数的有效尺度会改变。
