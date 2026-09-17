# IMUNet-EfficientNetB0（一维“EfficientNet-B0”，`imunet_efficientnetb0`）

> 规格卡版本：v1（2026-09-17）· fidelity：`official-code` · 夹具：`tests/fixtures/algorithms/imunet_efficientnetb0.json`

> **许可风险（必读）**：IMUNet 官方仓库 **没有 LICENSE 文件**（视为保留所有权利）。本卡只描述结构与行为，供研究性重实现；
> 不得复制其代码或分发其权重。文件头声明改编自 `AnjieCheng/MnasNet-PyTorch`（上游许可需另行核实）。

> **命名提醒**：该网络**不是**标准 EfficientNet-B0。它没有 SE 模块、没有 drop-connect，t=1 的块多了一个 1×1 扩张卷积，dropout 为 0.5。
> IPB 必须复现官方这一版本，并在模型文档里写明“IMUNet 仓库的一维 EfficientNetB0 变体”。

## 1. 元信息

| 项 | 值 |
|---|---|
| 论文 | 对比基线，出自 *IMUNet: Efficient Regression Architecture for Inertial IMU Navigation and Positioning*（Zeinali, Zanddizari, Chang；IEEE TIM vol. 73，2024，DOI 10.1109/TIM.2024.3381717；arXiv:[2208.00068](https://arxiv.org/abs/2208.00068)）；原始骨干为 Tan & Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*，ICML 2019 |
| 官方仓库 | https://github.com/BehnamZeinali/IMUNet @ `c57f14d0f4f5bbb8e29d836dabacc8718ca825e1` |
| 许可 | **无 LICENSE（all rights reserved）** |
| 框架 | PyTorch（`RONIN_torch/EfficientnetB0.py`）；Keras 版结构不同（§10） |
| fidelity | official-code |
| 参考文件 | `RONIN_torch/EfficientnetB0.py`；构造见 `RONIN_torch/main.py` 第 54–55 行：`EfficientNetB0(n_class=2)`；`--arch EfficientNet` |

## 2. 任务（输入）

与 `imunet.md` §2 **完全相同**：

| 项 | 官方 | IPB 映射 |
|---|---|---|
| 采样率 | 200 Hz（OxIOD 例外，按 100 Hz 使用） | 200 Hz |
| 窗口 | 200 样本 | `window=200`（全局池化，结构上可接受任意长度） |
| 训练步长 | 10，另加 ±5 随机平移 | `stride=10` + `time_shift` |
| 推理步长 | 10 | `eval_stride=10` |
| 坐标系 | 设备姿态旋到重力对齐世界系 | `frame=gravity_world` |
| 姿态来源 | 设备姿态，首帧整旋转对齐参考 | `orientation=device` |
| 去重力 | 否 | `remove_gravity=false` |
| 通道顺序 | `[glob_gyro_xyz, glob_acce_xyz]` | 与 `[gyro, acc]` 恒等 |
| 额外输入 / 归一化 | 无 / 无 | 无 |

## 3. 输出

`(B, 2)` 世界系水平平均速度（m/s），无协方差；目标与轨迹重建同 `imunet.md` §3。IPB 前向返回 `{"vel": (B, 2)}`。

## 4. 网络结构

所有卷积 `bias=False`；BN 为 PyTorch 默认（`eps=1e-5, momentum=0.1`）；激活为 Swish `x·sigmoid(x)`（即 SiLU）；**无 SE、无 drop-connect**。
官方把**同一个** Swish 模块实例放进所有 Sequential（无参数，不影响参数量；`named_modules()` 会去重，前向钩子只挂一次却被多次调用）。

**MBConv**（`InvertedResidual(inp, oup, stride s, expand t, kernel k)`，`hidden = inp·t`，`use_res = (s == 1 and inp == oup)`，
注册名 `conv.0 … conv.7`，其中 `conv.2`、`conv.5` 是共享的 Swish）：

```text
PW Conv1d(inp, hidden, k1) → BN → Swish           ← t = 1 时也存在（hidden = inp）
→ DW Conv1d(hidden, hidden, k, stride s, pad k//2, groups=hidden) → BN → Swish
→ PW Conv1d(hidden, oup, k1) → BN
out = x + F(x) 若 use_res，否则 F(x)
```

阶段配置 `(t, c, n, s, k)`：`(1,16,1,1,3) (6,24,2,2,3) (6,40,2,2,5) (6,80,3,2,3) (6,112,3,1,5) (6,192,4,2,5) (6,320,1,1,3)`，每阶段仅第一个块用步长 `s`。
stem 输出 32 通道，直接作为第一个 MBConv 的输入。

逐层表（T = 200）：

| # | 层（官方名） | 参数 | 归一化 | 激活 | 残差 | 输出形状 | 参数量 |
|---|---|---|---|---|---|---|---|
| 0 | 输入 | — | — | — | — | (B, 6, 200) | — |
| 1 | `features.0` Conv_3x3 | Conv1d 6→32, k3, s2, p1 | BN(32) | Swish | — | (B, 32, 100) | 640 |
| 2 | `features.1` MBConv | 32→16, t1, k3, s1（hidden 32，含 32→32 扩张卷积） | BN×3 | Swish×2 | 否 | (B, 16, 100) | 1,792 |
| 3 | `features.2` | 16→24, t6, k3, s2（hidden 96） | BN×3 | Swish×2 | 否 | (B, 24, 50) | 4,560 |
| 4 | `features.3` | 24→24, t6, k3, s1（hidden 144） | BN×3 | Swish×2 | 是 | (B, 24, 50) | 7,968 |
| 5 | `features.4` | 24→40, t6, k5, s2（hidden 144） | BN×3 | Swish×2 | 否 | (B, 40, 25) | 10,592 |
| 6 | `features.5` | 40→40, t6, k5, s1（hidden 240） | BN×3 | Swish×2 | 是 | (B, 40, 25) | 21,440 |
| 7 | `features.6` | 40→80, t6, k3, s2（hidden 240） | BN×3 | Swish×2 | 否 | (B, 80, 13) | 30,640 |
| 8–9 | `features.7`, `features.8` | 80→80, t6, k3, s1（hidden 480） | BN×3 | Swish×2 | 是 | (B, 80, 13) | 80,320 ×2 |
| 10 | `features.9` | 80→112, t6, k5, s1（hidden 480） | BN×3 | Swish×2 | 否 | (B, 112, 13) | 96,704 |
| 11–12 | `features.10`, `features.11` | 112→112, t6, k5, s1（hidden 672） | BN×3 | Swish×2 | 是 | (B, 112, 13) | 156,800 ×2 |
| 13 | `features.12` | 112→192, t6, k5, s2（hidden 672） | BN×3 | Swish×2 | 否 | (B, 192, 7) | 210,720 |
| 14–16 | `features.13`–`features.15` | 192→192, t6, k5, s1（hidden 1152） | BN×3 | Swish×2 | 是 | (B, 192, 7) | 453,120 ×3 |
| 17 | `features.16` | 192→320, t6, k3, s1（hidden 1152） | BN×3 | Swish×2 | 否 | (B, 320, 7) | 598,528 |
| 18 | `features.17` Conv_1x1 | Conv1d 320→1280, k1 | BN(1280) | Swish | — | (B, 1280, 7) | 412,160 |
| 19 | `features.18` AdaptiveAvgPool1d(1) | — | — | — | — | (B, 1280, 1) | 0 |
| 20 | `view(-1, 1280)` | — | — | — | — | (B, 1280) | 0 |
| 21 | `classifier.0` Dropout | **p = 0.5** | — | — | 0.5 | (B, 1280) | 0 |
| 22 | `classifier.1` Linear | 1280→2, bias | — | — | — | (B, 2) | 2,562 |

- **参数总量：3,231,906**（全部可训练）；官方配置 = benchmark 配置。152 个参数张量，BN 缓冲区 150 个（50 个 BN）。
- 计算量（本规格工程师测量）：**27.82 M MACs**/窗口。
- 初始化：`_initialize_weights()` 对一维模型**只有 `Linear` 生效**（`weight ~ N(0, 0.01²)`，`bias = 0`）；卷积与 BN 为 PyTorch 默认。
- 时间长度：200 → 100 → 100 → 50 → 25 → 13 → 13 → 7 → 7 → 1。

与标准 EfficientNet-B0 的差异（复现时**保持官方差异**）：

| 项 | 标准 B0 | 本实现 |
|---|---|---|
| SE（ratio 0.25） | 有 | **无** |
| drop-connect | 0.2 | **无** |
| t=1 块的扩张卷积 | 跳过 | **保留**（32→32 1×1 + BN + Swish） |
| 分类头 dropout | 0.2 | **0.5** |
| BN | eps 1e-3, momentum 0.01（PyTorch 记法） | PyTorch 默认 eps 1e-5, momentum 0.1 |

## 5. 损失与训练配方（official）

与 `imunet.md` §5 完全相同：`MSELoss`；Adam(`lr=1e-4`)；`ReduceLROnPlateau(factor=0.1, patience=10, eps=1e-12)`；batch 128（验证 512）；
300 epoch；无权重衰减、无梯度裁剪；`RandomHoriRotate(2π)` + ±5 样本随机平移；无阶段切换；最低验证 MSE 选模。
Dropout 只在 `train()` 模式生效。

## 6. benchmark 适配（unified 配方）

| 改动 | 类型 | 理由 |
|---|---|---|
| 输入/目标/坐标系与官方一致（`window=200, dims=2, frame=gravity_world, target=avg_velocity`） | 不改变结构 | 官方即此配置 |
| 目标用 IPB 199 间隔定义；姿态只做偏航对齐；推理时间戳取窗口中心 | 不改变结构 | IPB 协议（同 `imunet.md` §6） |
| Swish 可用 `nn.SiLU` 实现（数值相同）；每处用独立实例或共享实例均可 | 不改变结构 | 无参数；官方注释说明因 PyTorch 1.4 无 SiLU 才自定义 |
| unified 预算下 epoch/调度统一；选模改为 val ATE 且 val/test 分离 | 不改变结构 | 诚实协议 |

结论：无结构改动，夹具即官方配置。

## 7. 官方报告数值

arXiv v1 Table I（列 “EffNet”；RoNIN 按轴 RMSE 定义，与 IPB 相差 √2；TIM 2024 版未核实）：

| 数据集 | ATE (m) | RTE (m) |
|---|---|---|
| 自采（seen） | 2.67 | 3.48 |
| RoNIN seen | 3.66 | 2.79 |
| RoNIN unseen | 5.68 | 4.60 |
| OxIOD | 3.22 | 2.69 |
| RIDI | 1.67 | 2.05 |
| PX4（无人机） | 65.42 | — |

Table III（Keras→TFLite，Galaxy S10）：3.8 MB，967 µs/窗口（Keras 结构与本卡不同）。参数量/FLOPs 仅以图给出，未核实。

## 8. 忠实性测试建议

- [ ] `total_params == trainable_params == 3,231,906`；`param_shapes` 152 项逐项一致
- [ ] 输入 `(4, 6, 200)` → `(4, 2)`；各块形状同 §4 表
- [ ] `features.1` 含 3 个卷积（32→32 k1、DW 32 k3、32→16 k1）
- [ ] 残差块恰为 `features.3, 5, 7, 8, 10, 11, 13, 14, 15`（9 个）
- [ ] 激活数值等于 `x·sigmoid(x)`；模型中不存在 SE（没有全局池化后的 1×1 门控）
- [ ] 只有一个 `Dropout(p=0.5)`；`eval()` 下输出确定
- [ ] `classifier.1.bias == 0` 且权重标准差 ≈ 0.01（新建模型）

## 9. 预训练权重

同 `imunet.md` §9（Google Drive，内容未核实，无许可，不得分发）。

## 10. 官方实现的坑与未决问题

1. 数据/训练脚本问题同 `imunet.md` §10。
2. 名称误导：并非标准 EfficientNet-B0（§4 差异表）。
3. **Keras 版结构不同**（`RONIN_keras/EfficientnetB0.py`）：阶段内**重复块也用 stride 2**（例如第 3、5、7、8、13–15 个块），
   且第一个块（`block_id=1`）仍有扩张卷积；按 `(200, 6)` 实例化时可训练参数恰为 3,231,906（与 PyTorch 相同），但时间下采样倍数远大于 32；
   Keras `main.py` 用 `input_shape=(6, 200)` 构造，与数据生成器的 `(B, 200, 6)` 冲突。论文 Table III 的数值来自 Keras 模型。
