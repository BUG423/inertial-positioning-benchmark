# IMUNet-MnasNet（一维 MnasNet，`imunet_mnasnet`）

> 规格卡版本：v1（2026-09-17）· fidelity：`official-code` · 夹具：`tests/fixtures/algorithms/imunet_mnasnet.json`

> **许可风险（必读）**：IMUNet 官方仓库 **没有 LICENSE 文件**（视为保留所有权利）。本卡只描述结构与行为，供研究性重实现；
> 不得复制其代码或分发其权重。文件头声明改编自 `AnjieCheng/MnasNet-PyTorch`（上游许可需另行核实）。

## 1. 元信息

| 项 | 值 |
|---|---|
| 论文 | 对比基线，出自 *IMUNet: Efficient Regression Architecture for Inertial IMU Navigation and Positioning*（Zeinali, Zanddizari, Chang；IEEE TIM vol. 73，2024，DOI 10.1109/TIM.2024.3381717；arXiv:[2208.00068](https://arxiv.org/abs/2208.00068)）；原始骨干为 Tan et al., *MnasNet: Platform-Aware Neural Architecture Search for Mobile*，CVPR 2019 |
| 官方仓库 | https://github.com/BehnamZeinali/IMUNet @ `c57f14d0f4f5bbb8e29d836dabacc8718ca825e1` |
| 许可 | **无 LICENSE（all rights reserved）** |
| 框架 | PyTorch（`RONIN_torch/MnasNet.py`）；Keras 版结构不同（§10） |
| fidelity | official-code |
| 参考文件 | `RONIN_torch/MnasNet.py`；构造见 `RONIN_torch/main.py` 第 51–52 行：`MnasNet(n_class=2)`（`input_size=224, width_mult=1.0` 为默认值）；`--arch` 默认值就是 `MnasNet`（第 592 行） |

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

所有卷积 `bias=False`；BN 为 PyTorch 默认（`eps=1e-5, momentum=0.1`）；激活 ReLU6（`inplace=True`）；**无 SE 模块**。

**SepConv**（`features.1`，注册名 `0 … 4`）：`DW Conv1d(32, 32, k3, s1, p1, groups=32) → BN → ReLU6 → PW Conv1d(32, 16, k1) → BN`（线性输出，无残差）。

**MBConv**（`InvertedResidual(inp, oup, stride s, expand t, kernel k)`，`hidden = inp·t`，`use_res = (s == 1 and inp == oup)`，注册名 `conv.0 … conv.7`）：

```text
PW Conv1d(inp, hidden, k1) → BN → ReLU6
→ DW Conv1d(hidden, hidden, k, stride s, pad k//2, groups=hidden) → BN → ReLU6
→ PW Conv1d(hidden, oup, k1) → BN
out = x + F(x) 若 use_res，否则 F(x)
```

阶段配置 `(t, c, n, s, k)`：`(3,24,3,2,3) (3,40,3,2,5) (6,80,3,2,5) (6,96,2,1,3) (6,192,4,2,5) (6,320,1,1,3)`，每阶段仅第一个块用步长 `s`。
（这与无 SE 的 MnasNet-B1 配置相同。）

逐层表（T = 200）：

| # | 层（官方名） | 参数 | 归一化 | 激活 | 残差 | 输出形状 | 参数量 |
|---|---|---|---|---|---|---|---|
| 0 | 输入 | — | — | — | — | (B, 6, 200) | — |
| 1 | `features.0` Conv_3x3 | Conv1d 6→32, k3, s2, p1 | BN(32) | ReLU6 | — | (B, 32, 100) | 640 |
| 2 | `features.1` SepConv | 32→16 | BN×2 | ReLU6×1 | 否 | (B, 16, 100) | 704 |
| 3 | `features.2` MBConv | 16→24, t3, k3, s2（hidden 48） | BN×3 | ReLU6×2 | 否 | (B, 24, 50) | 2,304 |
| 4–5 | `features.3`, `features.4` | 24→24, t3, k3, s1（hidden 72） | BN×3 | ReLU6×2 | 是 | (B, 24, 50) | 4,008 ×2 |
| 6 | `features.5` | 24→40, t3, k5, s2（hidden 72） | BN×3 | ReLU6×2 | 否 | (B, 40, 25) | 5,336 |
| 7–8 | `features.6`, `features.7` | 40→40, t3, k5, s1（hidden 120） | BN×3 | ReLU6×2 | 是 | (B, 40, 25) | 10,760 ×2 |
| 9 | `features.8` | 40→80, t6, k5, s2（hidden 240） | BN×3 | ReLU6×2 | 否 | (B, 80, 13) | 31,120 |
| 10–11 | `features.9`, `features.10` | 80→80, t6, k5, s1（hidden 480） | BN×3 | ReLU6×2 | 是 | (B, 80, 13) | 81,280 ×2 |
| 12 | `features.11` | 80→96, t6, k3, s1（hidden 480） | BN×3 | ReLU6×2 | 否 | (B, 96, 13) | 88,032 |
| 13 | `features.12` | 96→96, t6, k3, s1（hidden 576） | BN×3 | ReLU6×2 | 是 | (B, 96, 13) | 114,816 |
| 14 | `features.13` | 96→192, t6, k5, s2（hidden 576） | BN×3 | ReLU6×2 | 否 | (B, 192, 7) | 171,456 |
| 15–17 | `features.14`–`features.16` | 192→192, t6, k5, s1（hidden 1152） | BN×3 | ReLU6×2 | 是 | (B, 192, 7) | 453,120 ×3 |
| 18 | `features.17` | 192→320, t6, k3, s1（hidden 1152） | BN×3 | ReLU6×2 | 否 | (B, 320, 7) | 598,528 |
| 19 | `features.18` Conv_1x1 | Conv1d 320→1280, k1 | BN(1280) | ReLU6 | — | (B, 1280, 7) | 412,160 |
| 20 | `features.19` AdaptiveAvgPool1d(1) | — | — | — | — | (B, 1280, 1) | 0 |
| 21 | `view(-1, 1280)` | — | — | — | — | (B, 1280) | 0 |
| 22 | `classifier.0` Dropout | **p = 0.5**（`nn.Dropout()` 默认） | — | — | 0.5 | (B, 1280) | 0 |
| 23 | `classifier.1` Linear | 1280→2, bias | — | — | — | (B, 2) | 2,562 |

- **参数总量：2,979,114**（全部可训练）；官方配置 = benchmark 配置。158 个参数张量，BN 缓冲区 156 个（52 个 BN）。
- 计算量（本规格工程师测量）：**24.39 M MACs**/窗口。
- 初始化：`_initialize_weights()` 只匹配 `Conv2d/BatchNorm2d/Linear`，对一维模型**只有 `Linear` 生效**（`weight ~ N(0, 0.01²)`，`bias = 0`）；卷积与 BN 为 PyTorch 默认。
- `n_class` 默认 1000，官方 `get_model` 显式传 `n_class=2`；复现时输出维度必须为 2。
- 时间长度：200 → 100 → 100 → 50 → 25 → 13 → 13 → 7 → 7 → 1。

## 5. 损失与训练配方（official）

与 `imunet.md` §5 完全相同：`MSELoss`；Adam(`lr=1e-4`)；`ReduceLROnPlateau(factor=0.1, patience=10, eps=1e-12)`；batch 128（验证 512）；
300 epoch；无权重衰减、无梯度裁剪；`RandomHoriRotate(2π)` + ±5 样本随机平移；无阶段切换；最低验证 MSE 选模。
Dropout 只在 `train()` 模式生效；验证/测试调用 `network.eval()`。

## 6. benchmark 适配（unified 配方）

| 改动 | 类型 | 理由 |
|---|---|---|
| 输入/目标/坐标系与官方一致（`window=200, dims=2, frame=gravity_world, target=avg_velocity`） | 不改变结构 | 官方即此配置 |
| 目标用 IPB 199 间隔定义；姿态只做偏航对齐；推理时间戳取窗口中心 | 不改变结构 | IPB 协议（同 `imunet.md` §6） |
| unified 预算下 epoch/调度统一；选模改为 val ATE 且 val/test 分离 | 不改变结构 | 诚实协议 |

结论：无结构改动，夹具即官方配置。

## 7. 官方报告数值

arXiv v1 Table I（列 “MnasNet”；RoNIN 按轴 RMSE 定义，与 IPB 相差 √2；TIM 2024 版未核实）：

| 数据集 | ATE (m) | RTE (m) |
|---|---|---|
| 自采（seen） | 2.75 | 3.19 |
| RoNIN seen | 3.78 | 2.75 |
| RoNIN unseen | **5.19**（该行最优） | 4.54 |
| OxIOD | 3.08 | 2.64 |
| RIDI | 1.71 | 2.10 |
| PX4（无人机） | 56.70 | — |

Table III（Keras→TFLite，Galaxy S10）：3.1 MB，654 µs/窗口（Keras 结构与本卡不同）。参数量/FLOPs 仅以图给出，未核实。

## 8. 忠实性测试建议

- [ ] `total_params == trainable_params == 2,979,114`；`param_shapes` 158 项逐项一致
- [ ] 输入 `(4, 6, 200)` → `(4, 2)`；各块形状同 §4 表
- [ ] 残差块恰为 `features.3, 4, 6, 7, 9, 10, 12, 14, 15, 16`（10 个）
- [ ] 含且仅含一个 `Dropout(p=0.5)`，位于最终 Linear 之前；`eval()` 下两次前向输出相同，`train()` 下不同
- [ ] `classifier.1.bias == 0` 且权重标准差 ≈ 0.01（新建模型）
- [ ] 无 SE、所有卷积无偏置；T=100/400 可前向

## 9. 预训练权重

同 `imunet.md` §9（Google Drive，内容未核实，无许可，不得分发）。

## 10. 官方实现的坑与未决问题

1. 数据/训练脚本问题同 `imunet.md` §10。
2. `_initialize_weights` 对一维层失效（§4）。
3. **Keras 版结构明显不同**（`RONIN_keras/MnasNet.py`）：没有 320→1280 的末端卷积；SepConv 的 BN/激活位置不同（DW 后无 BN/激活、PW 带 `strides`）；
   BN 为 `eps=1e-3, momentum=0.999`；`pooling` 不为 `'avg'` 时用全局**最大**池化；分类头无 dropout。按 `(200, 6)` 实例化时可训练参数 2,564,970
   （PyTorch 为 2,979,114）。Keras `main.py` 以 `input_shape=(6, 200)` 构造（channels-last 下是 6 个时刻 × 200 通道），与数据生成器的 `(B, 200, 6)` 矛盾。
   论文 Table III 的 MnasNet 延迟来自 Keras 模型，**不能**与本卡结构对应。
