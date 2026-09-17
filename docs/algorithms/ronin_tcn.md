# RoNIN-TCN（`ronin_tcn`）

> 规格卡版本：v1（2026-09-17）· fidelity：`official-code` · 夹具：`tests/fixtures/algorithms/ronin_tcn.json`

## 1. 元信息

| 项 | 值 |
|---|---|
| 论文 | *RoNIN: Robust Neural Inertial Navigation in the Wild: Benchmark, Evaluations, & New Methods*；Herath, Yan, Furukawa；ICRA 2020；<https://arxiv.org/abs/1905.12853> |
| 官方仓库 | <https://github.com/Sachini/ronin> @ `805b7f0f28bb164ce89ada9ac05a9470dbe3d715`（本地：`/workspace/webCodex/third_party/ronin`） |
| 许可 | GPL-3.0；`source/tcn.py` 改编自 locuslab/TCN（MIT，文件头保留了 MIT 声明） |
| 框架 | PyTorch（使用旧式 `torch.nn.utils.weight_norm`） |
| fidelity | official-code |
| 参考文件 | `source/model_temporal.py`（`TCNSeqNetwork`）、`source/tcn.py`（`TemporalConvNet`/`TemporalBlock`/`Chomp1d`）、`source/ronin_lstm_tcn.py`、`source/data_glob_speed.py`、`config/temporal_model_defaults.json` |

## 2. 任务（输入）

与 `ronin_lstm` 完全相同（同一脚本 `ronin_lstm_tcn.py`、同一数据集类），此处只列要点：

| 项 | 官方 | IPB 映射 |
|---|---|---|
| 采样率 | 200 Hz | 200 Hz |
| 窗口 | 训练/验证 400 帧 seq2seq 片段；测试整条序列一次前向 | `window=400` |
| 训练步长 | 100 帧 + `randrange(−50,50)` 偏移；速度 > 3 m/s 的片段被丢弃 | `stride=100` + `time_shift(±50)` |
| 推理步长 | 不适用（逐帧） | `eval_stride=10` |
| 坐标系 / 姿态 | HACF；测试 game RV，训练 game RV 或回退源（同 `ronin_resnet18`） | `frame=gravity_world`；测试 `orientation=device` |
| 去重力 | 否 | `remove_gravity=false` |
| 通道顺序 | `[glob_gyro_xyz, glob_acce_xyz]`，batch-first `(B,T,6)`；模块内部转置为 `(B,6,T)` 做卷积 | 与 IPB 相同；包装层把 `(B,6,T)` 转成 `(B,T,6)` 再喂官方接口（或直接跳过两次转置，数值相同） |
| 额外输入 / 归一化 | 无 / 无 | — |

## 3. 输出

- 逐帧 `(B, T, 2)` 世界系水平速度（m/s），无不确定度。
- 卷积是**因果**的：第 t 帧输出只依赖 `[t−252, t]` 的输入（感受野 253 帧 ≈ 1.27 s），序列开头不足部分等价于左侧补零。导出时已数值验证：修改 t≥300 的输入不改变 t<300 的输出；第 0 帧的脉冲只影响第 0–252 帧输出。
- 官方轨迹与指标：与 `ronin_lstm` 相同（整序列前向 → `p0 + cumsum(v̂·dt)`，分量平均 ATE/RTE）。

## 4. 网络结构

`TCNSeqNetwork(input_channel=6, output_channel=2, kernel_size=3, layer_channels=[32, 64, 128, 256, 72, 36], dropout=0.2)`（通道来自 `config/temporal_model_defaults.json`；dropout 为类默认值）。

TemporalBlock(i)（`tcn.py:39-69`），膨胀 `d = 2^i`，填充 `P = (k−1)·d = 2d`：

```
branch = Dropout(PReLU(Chomp(WN-Conv1d(C, C, k=3, d, pad=P))))      # conv2 部分
       ∘ Dropout(PReLU(Chomp(WN-Conv1d(Cin, C, k=3, d, pad=P))))    # conv1 部分
res    = Conv1d(Cin, C, k=1)（有 bias） if Cin != C else x
out    = PReLU(branch(x) + res)
```

- `WN-Conv1d`：带 bias 的 Conv1d，外加 `weight_norm(dim=0)`，权重 `W = g · v / ‖v‖`（按输出通道求范数），参数为 `bias (C)`、`weight_g (C,1,1)`、`weight_v (C,Cin,3)`，注册顺序即此顺序。
- `Chomp1d(P)`：去掉时间轴最后 P 个样本，使卷积因果、长度不变。
- 每个 block 3 个 `PReLU()`（各 1 个参数，初值 0.25）。
- 本配置 6 个 block 的 `Cin ≠ C`，全部带 1×1 下采样卷积。

逐层表（输入 `(B, 400, 6)` → 内部 `(B, 6, 400)`，形状省略 batch 维）：

| # | 层 | 参数 | 归一化 | 激活 | dropout | 输出形状 |
|---|---|---|---|---|---|---|
| 0 | 转置 | (T,6)→(6,T) | — | — | — | (6, 400) |
| 1 | Block0，d=1 | WN-Conv k3 6→32 pad2 → chomp2；WN-Conv k3 32→32 pad2 → chomp2；res Conv k1 6→32 | weight norm | PReLU×3 | 0.2（两处） | (32, 400)（卷积后未裁剪时为 402） |
| 2 | Block1，d=2 | 32→64，pad4 | weight norm | PReLU×3 | 0.2 | (64, 400)（中间 404） |
| 3 | Block2，d=4 | 64→128，pad8 | weight norm | PReLU×3 | 0.2 | (128, 400)（中间 408） |
| 4 | Block3，d=8 | 128→256，pad16 | weight norm | PReLU×3 | 0.2 | (256, 400)（中间 416） |
| 5 | Block4，d=16 | 256→72，pad32 | weight norm | PReLU×3 | 0.2 | (72, 400)（中间 432） |
| 6 | Block5，d=32 | 72→36，pad64 | weight norm | PReLU×3 | 0.2 | (36, 400)（中间 464） |
| 7 | Dropout | p=0.2 | — | — | 0.2 | (36, 400) |
| 8 | Conv1d | k1，36→2，有 bias（无 weight norm） | — | — | — | (2, 400) |
| 9 | 转置 | (2,T)→(T,2) | — | — | — | (400, 2) |

- 感受野：`1 + 2·(k−1)·(2^6 − 1) = 253`（`model_temporal.py:133-134`，论文同值）。
- 参数量：**540,488**（官方 config 通道，夹具锁定）。各 block：4,003 / 20,803 / 82,563 / 328,963 / 89,643 / 14,439；输出层 74。
- **论文与代码不一致**：论文 §4.2 写通道为 16, 32, 64, 128, 72, 36；官方 config 为 32, 64, 128, 256, 72, 36。IPB 以代码 config 为准（`official`），论文通道版本 177,176 参数，记录在夹具 `paper_channels_variant`。
- 初始化（实际生效的行为，已在 torch 2.11 上数值核实）：
  - `TemporalBlock.init_weights` 对 `conv1.weight.data`/`conv2.weight.data` 做 `normal_(0, 0.01)`，但 weight norm 在注册时已由默认初始化的权重得到 `g, v`，第一次前向会用 `g·v/‖v‖` 重算 `weight`，所以**这条初始化不生效**：有效权重服从 PyTorch Conv1d 默认初始化（`kaiming_uniform_(a=√5)`），`g = ‖v‖`。
  - 1×1 下采样卷积权重 ~ N(0, 0.01)（生效），其 bias 为默认初始化。
  - 输出层权重 ~ N(0, 0.01)，bias ~ N(0, 0.001)（`model_temporal.py:129-131`）。
  - PReLU 初值 0.25；WN-Conv 的 bias 为默认初始化。

## 5. 损失与训练配方（official）

与 `ronin_lstm` 相同（Adam 3e-4、batch 72 + drop_last、1000 epoch、`--use_scheduler` 时 ReduceLROnPlateau(0.75, 10)、`RandomHoriRotateSeq`、±50 偏移、速度过滤 3 m/s、按验证损失保存、平滑不生效、无权重衰减/裁剪），差异只有损失与 dropout：

| 项 | 值 | 来源 |
|---|---|---|
| 损失 | `GlobalPosLoss(mode='part', history=253)`：`P = cumsum(targ[:,1:])`，`D_j = P_{j+253} − P_j`（`j = 0..T−2−253`），同样处理预测，`loss = mean((D̂ − D)²)`。T=400 时每段 146 个 253 帧位移项，每项等于 `Σ_{i=j+2}^{j+254} v_i`；未乘 dt | `ronin_lstm_tcn.py:35-60,131-139,190` |
| dropout | 0.2（每个 block 两处 + 输出前一处）；与论文“keep 0.8”一致 | `model_temporal.py:101` |
| epoch | config 1000；论文：约 200 epoch 收敛、30 小时 | 论文 §5 |
| 阶段切换 | 无 | — |

## 6. benchmark 适配（unified 配方）

| 改动 | 类型 | 理由 |
|---|---|---|
| 包装层 `(B,6,T)` → 官方模块 → 逐帧 `y (B,T,2)`；`vel = mean_{k=0..T−2} y_k`；`aux = {"frame_vel": y}` | 不改变结构 | 与 `ronin_lstm` 卡 §6 相同的推导：逐帧速度在 `k=0..T−2` 上的均值等于 IPB `avg_velocity` |
| `window=400`、`stride=100`、`time_shift(±50)`、`eval_stride=10`、中心时间戳 | 不改变结构 | 与官方训练片段长度一致；窗口前 252 帧输出的上下文被截断（左侧补零），与官方训练时的条件相同 |
| official/unified 损失：`GlobalPosLoss(part, history=253)`，需要视图扩展 `aux_target=frame_velocity`（定义见 `ronin_lstm` 卡）；视图不支持时退化为 `MSE(vel, avg_velocity)` 并在结果中标注 | 不改变结构 | 保留专用损失 |
| 可选流式模式 `ronin_tcn@stream`：整序列一次前向后按窗口取均值 | 不改变结构 | 与官方测试一致；因为 TCN 严格因果、感受野有限，等价于每帧都有完整 252 帧历史（序列开头除外） |
| 初始化按 §4 的“实际生效行为”实现，不要让 N(0,0.01) 真正作用到 WN 卷积上 | 不改变结构 | 忠实于官方训练起点 |
| weight norm 可用 `torch.nn.utils.parametrizations.weight_norm` 实现，但参数名/顺序会变为 `parametrizations.weight.original0/1`；参数量测试以形状多重集合比较，或实现自定义 WN 以保持 `bias, weight_g, weight_v` 顺序 | 不改变结构 | 新版 PyTorch 已弃用旧式 `weight_norm` |

## 7. 官方报告数值

论文 Table 1，RoNIN TCN 列（官方代码口径，见 `ronin_resnet18` 卡 §3；注意论文文字所述通道与代码 config 不同，无法确定表中数值对应哪一版通道）：

| 数据集 | 测试集 | ATE (m) | RTE (m) |
|---|---|---|---|
| RIDI | seen | 1.66 | 2.16 |
| RIDI | unseen | 1.66 | 2.26 |
| OxIOD | seen | 2.26 | 2.63 |
| OxIOD | unseen | 7.76 | 5.78 |
| RoNIN | seen | 4.38 | 2.90 |
| RoNIN | unseen | 5.70 | 4.07 |

## 8. 忠实性测试建议

- [ ] 参数量 540,488；`param_shapes` 与夹具逐项一致（68 个张量）；论文通道变体（若实现）177,176。
- [ ] 输入 `(4, 400, 6)` → `(4, 400, 2)`；包装层 `(4, 6, 400)` → `vel (4, 2)`。
- [ ] 感受野 = 253：对第 0 帧加脉冲，eval 模式下受影响的最后一帧索引为 252。
- [ ] 严格因果：修改 `t ≥ 300` 的输入，`t < 300` 的输出逐位相同。
- [ ] 长度不变：任意 T（例如 1、7、400、1000）输出长度等于 T。
- [ ] WN 有效权重：初始化后 `g == ‖v‖`（按输出通道），有效权重标准差与默认 Conv1d 初始化同量级（约 0.1），而不是 0.01。
- [ ] 损失：`GlobalPosLoss(part, 253)` 在 T=400 时的项数为 146；`pred = targ + δ` 时损失为 `253²·δ²`。
- [ ] eval 模式下流式输出与窗口输出在 `t ≥ 252` 的帧上一致（窗口从序列中截取，且窗口起点 ≥ 0）。

## 9. 预训练权重

同 `ronin_resnet18`（FRDR 记录页当前无文件；预训练模型在全数据集上训练，不能用于测试评测）。

## 10. 官方实现的坑与未决问题

1. 论文通道（16…）与 config 通道（32…）不一致；预训练模型目录中的 `config.json` 才能确定作者实际用哪一版，但目前无法下载核实。
2. WN 卷积的 N(0, 0.01) 初始化不生效（见 §4），原 locuslab/TCN 也有同样现象。
3. `get_receptive_field()` 公式假设每个 block 两个卷积、膨胀按 2^i 递增，只适用于本结构。
4. 旧式 `torch.nn.utils.weight_norm` 在新版 PyTorch 中发出弃用警告，行为仍然可用；参数名依赖于该实现。
5. 其余数据/训练/测试层面的坑与 `ronin_lstm` 卡 §10 第 4、6–10 条相同（平滑键名、numba/NumPy 兼容、1 帧偏移、CPU 默认设备、缓存 interval、未乘 dt）。
6. `part` 模式的位移项从第 2 帧开始累加（`targ[:,1:]` 再做差），第 0、1 帧的速度从不进入损失，第 2–253 帧只在少数项中出现；移植时保持这一加权方式。
