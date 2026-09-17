# IMUNet-MobileNet（一维 MobileNetV1，`imunet_mobilenet`）

> 规格卡版本：v1（2026-09-17）· fidelity：`official-code` · 夹具：`tests/fixtures/algorithms/imunet_mobilenet.json`

> **许可风险（必读）**：IMUNet 官方仓库 **没有 LICENSE 文件**（视为保留所有权利）。本卡只描述结构与行为，供研究性重实现；
> 不得复制其代码或分发其权重。文件头声明该实现改编自 hackmd 上的一份 MobileNet 教程代码，上游许可同样未知。

## 1. 元信息

| 项 | 值 |
|---|---|
| 论文 | 对比基线，出自 *IMUNet: Efficient Regression Architecture for Inertial IMU Navigation and Positioning*（Zeinali, Zanddizari, Chang；IEEE TIM vol. 73，2024，DOI 10.1109/TIM.2024.3381717；arXiv:[2208.00068](https://arxiv.org/abs/2208.00068)）；原始骨干为 Howard et al., *MobileNets*，arXiv:1704.04861，2017 |
| 官方仓库 | https://github.com/BehnamZeinali/IMUNet @ `c57f14d0f4f5bbb8e29d836dabacc8718ca825e1` |
| 许可 | **无 LICENSE（all rights reserved）** |
| 框架 | PyTorch（`RONIN_torch/MobileNet.py`）；Keras 版 `RONIN_keras/MobileNet.py` 结构不同（见 §10） |
| fidelity | official-code |
| 参考文件 | `RONIN_torch/MobileNet.py`；构造见 `RONIN_torch/main.py` 第 49–50 行：`MobileNet(channels=[32, 64, 128, 256, 512, 1024], width_multiplier=1)` |

## 2. 任务（输入）

与 `imunet.md` §2 **完全相同**（同一训练脚本，只换网络）：

| 项 | 官方 | IPB 映射 |
|---|---|---|
| 采样率 | 200 Hz（OxIOD 例外，按 100 Hz 使用） | 200 Hz |
| 窗口 | 200 样本 | `window=200`（本网络为全局池化头，结构上可接受任意长度，但官方只用 200） |
| 训练步长 | 10，另加 ±5 随机平移 | `stride=10` + `time_shift` |
| 推理步长 | 10 | `eval_stride=10` |
| 坐标系 | 设备姿态旋到重力对齐世界系 | `frame=gravity_world` |
| 姿态来源 | 设备姿态（RoNIN 训练时可能退回 EKF/陀螺积分，测试强制 game RV），首帧整旋转对齐参考 | `orientation=device` |
| 去重力 | 否 | `remove_gravity=false` |
| 通道顺序 | `[glob_gyro_xyz, glob_acce_xyz]` | 与 `[gyro, acc]` 恒等 |
| 额外输入 / 归一化 | 无 / 无 | 无 |

## 3. 输出

`(B, 2)` 世界系水平平均速度（m/s），无协方差。目标 `(p[j+200]−p[j])[:2]/(t[j+200]−t[j])`；
轨迹重建与 `imunet.md` §3 相同（速度记在窗口起点、累加 `v̂·dts`、插值到全帧）。IPB 前向返回 `{"vel": (B, 2)}`。

## 4. 网络结构

所有卷积 `bias=False`；BN 为 PyTorch 默认（`eps=1e-5, momentum=0.1`）；激活均为 ReLU；**无残差、无 dropout**。

**DS 块**（`DSConv(c_in, c_out, stride s, padding=1)`），块内注册名 `feature.{dconv, bn1, act1, pconv, bn2, act2}`：

```text
x → Conv1d(c_in, c_in, k=3, stride=s, pad=1, groups=c_in) → BN(c_in) → ReLU
  → Conv1d(c_in, c_out, k=1) → BN(c_out) → ReLU
```

逐层表（T = 200）：

| # | 层（官方名） | 参数 | 归一化 | 激活 | dropout | 输出形状 | 参数量 |
|---|---|---|---|---|---|---|---|
| 0 | 输入 | — | — | — | — | (B, 6, 200) | — |
| 1 | `conv.conv` Conv1d | 6→32, k3, s2, p1 | `conv.bn` BN(32) | `conv.act` ReLU | — | (B, 32, 100) | 640 |
| 2 | `features.dsconv1` DS | 32→64, s1 | BN×2 | ReLU×2 | — | (B, 64, 100) | 2,336 |
| 3 | `features.dsconv2` DS | 64→128, s2 | BN×2 | ReLU×2 | — | (B, 128, 50) | 8,768 |
| 4 | `features.dsconv3` DS | 128→128, s1 | BN×2 | ReLU×2 | — | (B, 128, 50) | 17,280 |
| 5 | `features.dsconv4` DS | 128→256, s2 | BN×2 | ReLU×2 | — | (B, 256, 25) | 33,920 |
| 6 | `features.dsconv5` DS | 256→256, s1 | BN×2 | ReLU×2 | — | (B, 256, 25) | 67,328 |
| 7 | `features.dsconv6` DS | 256→512, s2 | BN×2 | ReLU×2 | — | (B, 512, 13) | 133,376 |
| 8–12 | `features.dsconv7_a` … `dsconv7_e` DS ×5 | 512→512, s1 | BN×2 | ReLU×2 | — | (B, 512, 13) | 265,728 ×5 |
| 13 | `features.dsconv8` DS | 512→1024, s2 | BN×2 | ReLU×2 | — | (B, 1024, 7) | 528,896 |
| 14 | `features.dsconv9` DS | 1024→1024, s1 | BN×2 | ReLU×2 | — | (B, 1024, 7) | 1,055,744 |
| 15 | `avgpool` AdaptiveAvgPool1d(1) + flatten | — | — | — | — | (B, 1024) | 0 |
| 16 | `linear` Linear | 1024→2, bias | — | — | — | (B, 2) | 2,050 |

- **参数总量：3,178,978**（全部可训练）；官方配置 = benchmark 配置。83 个参数张量（27 个卷积权重 + 27 个 BN 的 γ、β + Linear 的权重与偏置），BN 缓冲区 81 个。
- 计算量（本规格工程师测量）：**33.91 M MACs**/窗口（卷积+全连接权重乘加）。
- 长度随 stride 2 变化：200 → 100 → 50 → 25 → 13 → 7（`ceil`）。输出与窗口长度无关（全局平均池化），T=100/400 均可前向。
- 初始化：PyTorch 默认，无自定义初始化。
- `width_multiplier=1`：通道即 `[32, 64, 128, 256, 512, 1024]`；构造器对通道做 `int(c × width_multiplier)`。
- 源码中注释掉的 `self.output` 多层 FC 头**未启用**，不要实现。

## 5. 损失与训练配方（official）

与 `imunet.md` §5 完全相同：`MSELoss`；Adam(`lr=1e-4`，无权重衰减)；`ReduceLROnPlateau(factor=0.1, patience=10, eps=1e-12)`，按验证 MSE 每 epoch 调整；
batch 128（验证 512）；300 epoch；无梯度裁剪；增强为 `RandomHoriRotate(2π)`（仅训练）+ ±5 样本随机平移；无阶段切换；
按最低验证 MSE 保存 `checkpoint_best.pt`（RIDI/自采/OxIOD 的“验证集”即测试集）。
注意 `main.py` 的 `--arch` choices 含 `MobileNet`，可直接选择。

## 6. benchmark 适配（unified 配方）

| 改动 | 类型 | 理由 |
|---|---|---|
| 输入/目标/坐标系与官方一致（`window=200, dims=2, frame=gravity_world, target=avg_velocity`） | 不改变结构 | 官方即此配置 |
| 目标用 IPB 的 199 间隔定义；姿态只做偏航对齐；推理时间戳取窗口中心 | 不改变结构 | IPB 协议（同 `imunet.md` §6） |
| unified 预算下 epoch/调度统一；选模改为 val ATE 且 val/test 分离 | 不改变结构 | 诚实协议 |
| 若 unified 采用其他窗口长度，本网络可直接运行 | 不改变结构 | 全局平均池化头；但与官方数值的可比性下降，需在结果中注明 |

结论：无结构改动，夹具即官方配置。

## 7. 官方报告数值

arXiv v1 Table I（列 “MobNet”；指标为 RoNIN 按轴 RMSE 定义，与 IPB 相差 √2，见 `imunet.md` §7；TIM 2024 版数值未核实）：

| 数据集 | ATE (m) | RTE (m) |
|---|---|---|
| 自采（seen） | 2.98 | 3.42 |
| RoNIN seen | 4.08 | 2.83 |
| RoNIN unseen | 6.16 | 4.75 |
| OxIOD | 3.20 | 2.68 |
| RIDI | 1.73 | 2.09 |
| PX4（无人机） | 64.94 | — |

Table III（Keras→TFLite，Galaxy S10）：3.5 MB，907 µs/窗口（Keras 结构与本卡不同，见 §10）。参数量/FLOPs 仅以图给出，未核实。

## 8. 忠实性测试建议

- [ ] `total_params == trainable_params == 3,178,978`；`param_shapes` 83 项逐项一致
- [ ] 输入 `(4, 6, 200)` → `(4, 2)`；各阶段长度 100/100/50/50/25/25/13/13/7/7
- [ ] 所有 `Conv1d` 均无偏置；模型中无 Dropout、无残差
- [ ] 输入 `(2, 6, 100)`、`(2, 6, 400)` 可前向且输出 `(2, 2)`
- [ ] 模型不具旋转等变性（仅记录，不作为性质测试）

## 9. 预训练权重

同 `imunet.md` §9：README 的 Google Drive 压缩包（未核实是否含本变体），无许可，不得分发。

## 10. 官方实现的坑与未决问题

1. 数据/训练脚本的全部问题同 `imunet.md` §10（`main.py` 无法直接导入、测试集选模、OxIOD 约定错误、半窗时间错位等）。
2. **Keras 版结构不同**（`RONIN_keras/MobileNet.py`）：卷积使用 Keras 默认 `use_bias=True`、`padding='same'`，
   以 `(200, 6)` 输入实例化时可训练参数 3,189,922（比 PyTorch 多 10,944 个偏置）；且 Keras `main.py` 调用的是不存在的 `MobileNetV1_1D`。
   论文 Table III 的延迟来自 Keras/TFLite 模型。
3. 与原始 MobileNetV1 相比：完全保留 13 个 DS 块的通道与步长配置（1-D 化），无 dropout、无宽度/分辨率缩放。
