# IMUNet-MobileNetV2（一维 MobileNetV2，`imunet_mobilenetv2`）

> 规格卡版本：v1（2026-09-17）· fidelity：`official-code` · 夹具：`tests/fixtures/algorithms/imunet_mobilenetv2.json`

> **许可风险（必读）**：IMUNet 官方仓库 **没有 LICENSE 文件**（视为保留所有权利）。本卡只描述结构与行为，供研究性重实现；
> 不得复制其代码或分发其权重。文件头声明改编自 `tonylins/pytorch-mobilenet-v2`（上游许可需另行核实）。

## 1. 元信息

| 项 | 值 |
|---|---|
| 论文 | 对比基线，出自 *IMUNet: Efficient Regression Architecture for Inertial IMU Navigation and Positioning*（Zeinali, Zanddizari, Chang；IEEE TIM vol. 73，2024，DOI 10.1109/TIM.2024.3381717；arXiv:[2208.00068](https://arxiv.org/abs/2208.00068)）；原始骨干为 Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*，CVPR 2018 |
| 官方仓库 | https://github.com/BehnamZeinali/IMUNet @ `c57f14d0f4f5bbb8e29d836dabacc8718ca825e1` |
| 许可 | **无 LICENSE（all rights reserved）** |
| 框架 | PyTorch（`RONIN_torch/MobileNetV2.py`）；Keras 版见 §10 |
| fidelity | official-code |
| 参考文件 | `RONIN_torch/MobileNetV2.py`；构造见 `RONIN_torch/main.py` 第 47–48 行：`MobileNetV2()`（默认 `n_class=2, input_size=224, width_mult=1.0`） |

## 2. 任务（输入）

与 `imunet.md` §2 **完全相同**：

| 项 | 官方 | IPB 映射 |
|---|---|---|
| 采样率 | 200 Hz（OxIOD 例外，按 100 Hz 使用） | 200 Hz |
| 窗口 | 200 样本 | `window=200`（时间维全局平均，结构上可接受任意长度） |
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

所有卷积 `bias=False`；BN 为 PyTorch 默认（`eps=1e-5, momentum=0.1`）；激活 ReLU6；**无 dropout**。

**IR 块**（`InvertedResidual(inp, oup, stride s, expand_ratio t)`，`hidden = inp·t`，`use_res = (s == 1 and inp == oup)`），
块内注册名为 `conv.0 … conv.7`（t=1 时为 `conv.0 … conv.4`）：

```text
t = 1:  DW Conv1d(hidden, hidden, k3, stride s, pad 1, groups=hidden) → BN → ReLU6
        → PW Conv1d(hidden, oup, k1) → BN                         （线性瓶颈，无激活）
t > 1:  PW Conv1d(inp, hidden, k1) → BN → ReLU6
        → DW Conv1d(hidden, hidden, k3, stride s, pad 1, groups=hidden) → BN → ReLU6
        → PW Conv1d(hidden, oup, k1) → BN
out = x + F(x)  若 use_res，否则 F(x)
```

阶段配置 `(t, c, n, s)`：`(1,16,1,1) (6,24,2,2) (6,32,3,2) (6,64,4,2) (6,96,3,1) (6,160,3,2) (6,320,1,1)`；
每阶段只有第一个块用步长 `s`，其余步长 1。`width_mult=1` 时 `make_divisible(c, 8) = c`，通道不变；第一层固定 32 通道，最后一层 1280 通道。

逐层表（T = 200）：

| # | 层（官方名） | 参数 | 归一化 | 激活 | 残差 | 输出形状 | 参数量 |
|---|---|---|---|---|---|---|---|
| 0 | 输入 | — | — | — | — | (B, 6, 200) | — |
| 1 | `features.0` conv_bn | Conv1d 6→32, k3, s2, p1 | BN(32) | ReLU6 | — | (B, 32, 100) | 640 |
| 2 | `features.1` IR t=1 | 32→16, s1（DW 32 → PW 16） | BN×2 | ReLU6×1 | 否 | (B, 16, 100) | 704 |
| 3 | `features.2` IR t=6 | 16→24, s2, hidden 96 | BN×3 | ReLU6×2 | 否 | (B, 24, 50) | 4,560 |
| 4 | `features.3` IR t=6 | 24→24, s1, hidden 144 | BN×3 | ReLU6×2 | 是 | (B, 24, 50) | 7,968 |
| 5 | `features.4` IR t=6 | 24→32, s2, hidden 144 | BN×3 | ReLU6×2 | 否 | (B, 32, 25) | 9,136 |
| 6–7 | `features.5`, `features.6` IR t=6 | 32→32, s1, hidden 192 | BN×3 | ReLU6×2 | 是 | (B, 32, 25) | 13,696 ×2 |
| 8 | `features.7` IR t=6 | 32→64, s2, hidden 192 | BN×3 | ReLU6×2 | 否 | (B, 64, 13) | 19,904 |
| 9–11 | `features.8`–`features.10` IR t=6 | 64→64, s1, hidden 384 | BN×3 | ReLU6×2 | 是 | (B, 64, 13) | 51,968 ×3 |
| 12 | `features.11` IR t=6 | 64→96, s1, hidden 384 | BN×3 | ReLU6×2 | 否 | (B, 96, 13) | 64,320 |
| 13–14 | `features.12`, `features.13` IR t=6 | 96→96, s1, hidden 576 | BN×3 | ReLU6×2 | 是 | (B, 96, 13) | 114,816 ×2 |
| 15 | `features.14` IR t=6 | 96→160, s2, hidden 576 | BN×3 | ReLU6×2 | 否 | (B, 160, 7) | 151,808 |
| 16–17 | `features.15`, `features.16` IR t=6 | 160→160, s1, hidden 960 | BN×3 | ReLU6×2 | 是 | (B, 160, 7) | 314,240 ×2 |
| 18 | `features.17` IR t=6 | 160→320, s1, hidden 960 | BN×3 | ReLU6×2 | 否 | (B, 320, 7) | 468,160 |
| 19 | `features.18` conv_1x1_bn | Conv1d 320→1280, k1 | BN(1280) | ReLU6 | — | (B, 1280, 7) | 412,160 |
| 20 | 时间维均值 `x.mean(2)` | — | — | — | — | (B, 1280) | 0 |
| 21 | `classifier` Linear | 1280→2, bias | — | — | — | (B, 2) | 2,562 |

- **参数总量：2,183,330**（全部可训练）；官方配置 = benchmark 配置。158 个参数张量，BN 缓冲区 156 个（52 个 BN）。
- 计算量（本规格工程师测量）：**19.56 M MACs**/窗口。
- 初始化：`_initialize_weights()` 只检查 `nn.Conv2d / nn.BatchNorm2d / nn.Linear`；对一维模型**只有 `Linear` 生效**：
  `classifier.weight ~ N(0, 0.01²)`、`classifier.bias = 0`；所有 `Conv1d`、`BatchNorm1d` 保持 PyTorch 默认初始化。复现必须照此（不要套用原版 MobileNetV2 的 He 初始化）。
- `ReLU6(inplace=True)`；构造器中的 `assert input_size % 32 == 0` 只检查默认值 224，与实际窗口无关。
- 时间长度：200 → 100 → 100 → 50 → 25 → 13 → 13 → 7 → 7。

## 5. 损失与训练配方（official）

与 `imunet.md` §5 完全相同：`MSELoss`；Adam(`lr=1e-4`)；`ReduceLROnPlateau(factor=0.1, patience=10, eps=1e-12)`；batch 128（验证 512）；
300 epoch；无权重衰减、无梯度裁剪；`RandomHoriRotate(2π)` + ±5 样本随机平移；无阶段切换；最低验证 MSE 选模。
注意：`main.py` 的 `--arch` choices **不含** `MobileNetV2`（第 592–593 行），虽然 `get_model` 支持；要训练此变体需改 choices。

## 6. benchmark 适配（unified 配方）

| 改动 | 类型 | 理由 |
|---|---|---|
| 输入/目标/坐标系与官方一致（`window=200, dims=2, frame=gravity_world, target=avg_velocity`） | 不改变结构 | 官方即此配置 |
| 目标用 IPB 199 间隔定义；姿态只做偏航对齐；推理时间戳取窗口中心 | 不改变结构 | IPB 协议（同 `imunet.md` §6） |
| unified 预算下 epoch/调度统一；选模改为 val ATE 且 val/test 分离 | 不改变结构 | 诚实协议 |

结论：无结构改动，夹具即官方配置。

## 7. 官方报告数值

arXiv v1 Table I（列 “MobNetV2”；RoNIN 按轴 RMSE 定义，与 IPB 相差 √2；TIM 2024 版未核实）：

| 数据集 | ATE (m) | RTE (m) |
|---|---|---|
| 自采（seen） | 3.03 | 3.55 |
| RoNIN seen | 3.83 | 2.85 |
| RoNIN unseen | 6.17 | 4.69 |
| OxIOD | 3.38 | 2.89 |
| RIDI | 1.55 | 1.97 |
| PX4（无人机） | 65.86 | — |

Table III（Keras→TFLite，Galaxy S10）：2.7 MB，645 µs/窗口。参数量/FLOPs 仅以图给出，未核实。

## 8. 忠实性测试建议

- [ ] `total_params == trainable_params == 2,183,330`；`param_shapes` 158 项逐项一致
- [ ] 输入 `(4, 6, 200)` → `(4, 2)`；各块输出长度与通道同 §4 表
- [ ] 残差只出现在 §4 标“是”的 10 个块（`features.3, 5, 6, 8, 9, 10, 12, 13, 15, 16`）
- [ ] 每个 IR 块最后一个 BN 之后没有激活（线性瓶颈）
- [ ] 新建模型的 `classifier.bias` 全为 0，`classifier.weight` 标准差 ≈ 0.01；卷积权重不是 He-normal
- [ ] 无 Dropout；T=100/400 可前向

## 9. 预训练权重

同 `imunet.md` §9（Google Drive，内容未核实，无许可，不得分发）。

## 10. 官方实现的坑与未决问题

1. 数据/训练脚本问题同 `imunet.md` §10；另有 `--arch` 不含 `MobileNetV2` 的问题（§5）。
2. `_initialize_weights` 对一维层失效（§4），与原版 MobileNetV2 行为不同。
3. **Keras 版**（`RONIN_keras/MobileNetV2.py`）：第一个块只有 DW+投影（与 PyTorch 一致），可训练参数同为 2,183,330（按 `(200, 6)` 实例化），
   但多了 `Dropout(0.4)`（在全局池化之前），使用 `padding='same'`、Keras BN 默认值；Keras `main.py` 调用的是不存在的 `MobileNetV2_1D_Arch`。
4. 与常见实现相比（如 torchvision 的分类头前有 `Dropout(0.2)`）：本实现无 dropout、一维化，阶段配置 `(t, c, n, s)` 与论文原表一致。
