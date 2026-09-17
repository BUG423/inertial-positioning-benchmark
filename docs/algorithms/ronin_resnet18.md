# RoNIN-ResNet18（`ronin_resnet18`）

> 规格卡版本：v1（2026-09-17）· fidelity：`official-code` · 夹具：`tests/fixtures/algorithms/ronin_resnet18.json`

## 1. 元信息

| 项 | 值 |
|---|---|
| 论文 | *RoNIN: Robust Neural Inertial Navigation in the Wild: Benchmark, Evaluations, & New Methods*；Herath, Yan, Furukawa；ICRA 2020；<https://arxiv.org/abs/1905.12853> |
| 官方仓库 | <https://github.com/Sachini/ronin> @ `805b7f0f28bb164ce89ada9ac05a9470dbe3d715`（本地：`/workspace/webCodex/third_party/ronin`） |
| 许可 | GPL-3.0。IPB 移植必须是依据本卡的独立实现，不得复制官方代码；若复制则整个衍生文件受 GPL-3.0 约束 |
| 框架 | PyTorch（官方未锁版本；依赖 numpy-quaternion、numba、tensorboardX） |
| fidelity | official-code |
| 参考文件 | `source/model_resnet1d.py`（结构）、`source/ronin_resnet.py`（配置/训练/测试）、`source/data_glob_speed.py`（样本与目标）、`source/data_utils.py`（姿态源选择）、`source/transformations.py`（增强）、`source/metric.py`（ATE/RTE） |

同一仓库的 `ronin_lstm`、`ronin_tcn` 另见各自卡片；RoNIN body heading 网络不在 IPB v1 范围内。

## 2. 任务（输入）

| 项 | 官方 | IPB 映射 |
|---|---|---|
| 采样率 | 200 Hz（论文 §3：IMU 与位姿均 200 Hz） | 200 Hz，无需重采样 |
| 窗口 | 200 样本 = 1 s（`--window_size 200`） | `window=200` |
| 训练步长 | 10 样本（`--step_size 10`），每次取样再加随机偏移 `random.randrange(-5, 5)`，即 [-5, 4] 的整数 | `stride=10`，`augment` 中加 `time_shift`（±5 样本，见 §10 的边界问题） |
| 推理步长 | 代码为 `step_size`（默认 10）；论文写“每 5 帧预测一次” | `eval_stride=10`（与代码一致） |
| 坐标系 | 论文称为 HACF（heading-agnostic coordinate frame），即 z 轴与重力对齐、偏航任意但在整条序列内固定的世界系 | `frame=gravity_world` |
| 姿态来源 | 测试：只用 Android game rotation vector（`grv_only=True`）。训练/验证：若 game RV 的序列末端对齐误差 < 20°（`--max_ori_error 20`）则用 game RV，否则在 {陀螺积分、game RV、`pose/ekf_ori`} 中取对齐误差最小者（`data_utils.py:106-150`；论文 §5.2 称这种情况“用 ground-truth 姿态”）。所有姿态在第 0 帧用 Tango 姿态与标定旋转整体对齐（`data_glob_speed.py:55-58`） | 测试 `orientation=device`（IPB 在首个有效样本处补偿常值偏航，与官方首帧对齐等价到横滚/俯仰小误差）。训练见 §6 |
| 去重力 | 否（加计为含重力比力，旋到世界系后 z ≈ +9.8） | `remove_gravity=false` |
| 通道顺序 | `[glob_gyro_xyz, glob_acce_xyz]`（`data_glob_speed.py:70`） | 与 IPB `[gyro_xyz, acc_xyz]` 相同，无需置换 |
| 额外输入 | 无 | 无 |
| 输入归一化 | 无（原始物理量）；陀螺减去 `imu_init_gyro_bias`，加计做 `scale*(acc-bias)` 标定（`data_glob_speed.py:48-49`），属于数据预处理 | 由 RoNIN 转换器完成并记录在转换报告中 |

样本定义（`StridedSequenceDataset`，`data_glob_speed.py:143-185`）：对起点 `s`，输入为 `features[s : s+200]`，转置为 `(6, 200)`；目标为 `targets[s]`。

## 3. 输出

- 形状 `(B, 2)`，物理量为重力对齐世界系中的水平平均速度 `(vx, vy)`，单位 m/s，无不确定度。
- 官方目标：`v_s = (p[s+200] − p[s]) / (t[s+200] − t[s])`，只取 xy（`data_glob_speed.py:60-61,71`，`interval = window_size`）。论文 §4.3 称为 “strided velocity loss”，写作 `P_i − P_{i−200}`（在 1 s 窗口上与速度数值相同）。
- 官方轨迹重建（`ronin_resnet.py:248-261`）：以 `step_size` 滑窗得到 `v_k`；`dts = mean(ts[s_{k+1}] − ts[s_k])`（=0.05 s）；`pos[1:-1] = p0 + cumsum(v_k · dts)`，并把第 k 个累计位置赋给**窗口起点**时间 `ts[s_k]`；首尾各补一个点（`ts[0]−1e-6` 处为 `p0`，`ts[-1]+1e-6` 处复制最后位置），再线性插值到所有帧。这样积分出的位置相对真实时间提前约半个窗口（0.5 s）——IPB §5 明确禁止此做法，改为窗口中心时间戳。
- 官方指标（`metric.py`）：`ATE = sqrt(mean((est−gt)**2))` 对 N×2 个**坐标分量**求均值，等于欧氏 RMSE 的 `1/√2`；RTE 取 Δ=12000 帧（1 min @ 200 Hz）；序列短于 12000 帧时取 Δ=N−1 并乘以 `12000/N`。RTE 同样按分量平均。

## 4. 网络结构

`ResNet1D(num_inputs=6, num_outputs=2, BasicBlock1D, group_sizes=[2,2,2,2], base_plane=64, kernel_size=3, output_block=FCOutputModule, fc_dim=512, in_dim=7, dropout=0.5, trans_planes=128)`（`ronin_resnet.py:19-26,126`）。

逐层表（输入 `(B, 6, 200)`；所有卷积 `bias=False`；BN 为 PyTorch 默认 `eps=1e-5, momentum=0.1`；形状省略 batch 维）：

| # | 层 | 参数（核/步长/填充/通道） | 归一化 | 激活 | dropout | 输出形状 |
|---|---|---|---|---|---|---|
| 0 | 输入 | — | — | — | — | (6, 200) |
| 1 | Conv1d | k7 s2 p3, 6→64 | BN(64) | ReLU | — | (64, 100) |
| 2 | MaxPool1d | k3 s2 p1 | — | — | — | (64, 50) |
| 3 | group1.block1 (Basic) | conv k3 s1 p1 64→64；conv k3 s1 p1 64→64；恒等残差 | BN 于每个 conv 后 | ReLU（conv1 后；相加后） | — | (64, 50) |
| 4 | group1.block2 (Basic) | 同上 | 同上 | 同上 | — | (64, 50) |
| 5 | group2.block1 (Basic) | conv k3 **s2** p1 64→128；conv k3 s1 p1 128→128；残差 = Conv1d k1 s2 64→128 + BN | BN | ReLU | — | (128, 25) |
| 6 | group2.block2 (Basic) | 两个 conv k3 s1 p1 128→128；恒等残差 | BN | ReLU | — | (128, 25) |
| 7 | group3.block1 (Basic) | conv k3 s2 p1 128→256；conv k3 256→256；残差 k1 s2 128→256 + BN | BN | ReLU | — | (256, 13) |
| 8 | group3.block2 (Basic) | 两个 conv k3 256→256 | BN | ReLU | — | (256, 13) |
| 9 | group4.block1 (Basic) | conv k3 s2 p1 256→512；conv k3 512→512；残差 k1 s2 256→512 + BN | BN | ReLU | — | (512, 7) |
| 10 | group4.block2 (Basic) | 两个 conv k3 512→512 | BN | ReLU | — | (512, 7) |
| 11 | transition | Conv1d k1 512→128 | BN(128) | **无** | — | (128, 7) |
| 12 | flatten | — | — | — | — | (896,) |
| 13 | Linear | 896→512（有 bias） | — | ReLU | Dropout(p=0.5) | (512,) |
| 14 | Linear | 512→512 | — | ReLU | Dropout(p=0.5) | (512,) |
| 15 | Linear | 512→2 | — | — | — | (2,) |

BasicBlock 前向：`out = relu(bn1(conv1(x))); out = bn2(conv2(out)); out += downsample(x) or x; out = relu(out)`。`dilation` 参数在官方代码中被忽略（恒为 1）。

- 参数量：**4,634,882**（官方配置 = benchmark 配置，夹具锁定），全部可训练；另有 63 个 BN buffer。分段：输入 conv 2,688 + BN 128；group1 49,664；group2 181,504；group3 723,456；group4 2,888,704；transition 65,792；FC 722,946。
- 参考：同一代码的 `resnet50`（BasicBlock [3,4,6,3]，fc_dim 1024）9,256,450；`resnet101` 15,958,530；resnet18 在 window=400 时 5,028,098；输出 3 维时 4,635,395。
- 时间长度链：`L1 = ⌊(T+6−7)/2⌋+1`，`L2 = ⌊(L1−1)/2⌋+1`，此后每个 stride-2 组 `L ← ⌊(L−1)/2⌋+1`。T=200：100→50→50→25→13→7。FC 输入长度 `in_dim` 官方写死为 `window_size // 32 + 1`（见 §10）。
- 权重初始化（`model_resnet1d.py:198-217`）：Conv1d `kaiming_normal_(mode='fan_out', nonlinearity='relu')`；BN weight=1、bias=0；Linear weight~N(0, 0.01)、bias=0；`zero_init_residual=False`。

## 5. 损失与训练配方（official）

| 项 | 值 | 来源 |
|---|---|---|
| 损失 | `nn.MSELoss()`（对 batch 与 2 个分量取均值），预测 vs `v_s` | `ronin_resnet.py:135,186` |
| 优化器 | Adam，默认 betas/eps，无权重衰减 | `ronin_resnet.py:136` |
| 学习率 / 调度 | 初始 1e-4（`--lr` 默认，论文一致）；`ReduceLROnPlateau(factor=0.1, patience=10, eps=1e-12)`，每 epoch 用验证集平均 MSE 调用一次（只有提供 `--val_list` 时才调用） | `ronin_resnet.py:137,211,402` |
| batch | 训练 128；验证 512 | `ronin_resnet.py:103,111,403` |
| epoch | `--epochs` 默认 10000（实际靠手动中断）；论文：约 100 epoch 收敛、10 小时 | `ronin_resnet.py:404`；论文 §5 |
| 权重衰减 | 无 | — |
| 梯度裁剪 | 无 | — |
| 增强 | `RandomHoriRotate(2π)`：每个样本取 `θ~U[0, 2π)`，同一 2×2 旋转作用于陀螺 xy、加计 xy 与目标 xy（z 不变）；时间偏移 `randrange(-5,5)`；每 epoch 打乱 | `transformations.py:67-81`，`ronin_resnet.py:68-71` |
| dropout | FC 层 p=0.5（论文“keep probability 0.5”） | `ronin_resnet.py:20` |
| 阶段切换 | 无 | — |
| 选模/早停 | 验证 MSE 创新低时保存 `checkpoint_<epoch>.pt`（即选 best-val）；无早停；中断或结束时另存 `checkpoint_latest.pt` | `ronin_resnet.py:215-222,238-243` |
| 其他 | 训练前先对训练集（保持 train 模式前向一遍，不反传、不更新权重，但会更新 BN 滑动统计且 dropout 生效）和验证集（eval 模式）评估一次初始损失；`--feature_sigma/--target_sigma` 参数被解析但**未使用** | `ronin_resnet.py:157-173,414-415` |
| 数据划分 | `lists/list_train.txt`（72 条）、`list_val.txt`（15）、`list_test_seen.txt`（31）、`list_test_unseen.txt`（31）；RIDI 数据集用 `--dataset ridi`（`data_ridi.py`，姿态恒为 game RV） | `lists/` |

## 6. benchmark 适配（unified 配方）

| 改动 | 类型 | 理由 |
|---|---|---|
| 无结构改动：window=200 @ 200 Hz、6 通道 `[gyro, acc]`、2 维输出与官方完全一致；`forward` 返回 `{"vel": (B,2)}` | 不改变结构 | 夹具参数量 4,634,882 与官方配置相同 |
| `in_dim` 由实际特征长度决定（或对 T=200 固定为 7），不要用 `T//32+1` | 不改变结构（T=200 时数值相同） | 官方公式在 T 为 32 的倍数时算错，见 §10 |
| 目标用 IPB `avg_velocity`：`(p[s+T−1] − p[s]) / ((T−1)·dt)` | 不改变结构 | 官方用 `p[s+200]`（窗口外一个样本），两者差 1 个采样间隔，影响可忽略；统一定义才能公平比较 |
| 轨迹重建按 IPB §5（窗口中心时间戳、首尾常值外推、梯形积分） | 不改变结构 | 官方把速度赋给窗口起点，存在半窗错位 |
| 指标用 IPB 欧氏 ATE/RTE | 不改变结构 | 官方按分量平均，数值为欧氏定义的 `1/√2`；与论文对照时乘以 √2，或在报告中附 `legacy_ronin` 口径 |
| 训练姿态：official 配方用 `orientation=device`，并在设备姿态末端对齐误差 > 20° 的序列上回退到 `reference`（需要视图支持 `orientation_fallback: {source: reference, max_error_deg: 20}`）；unified 配方训练用 `orientation=reference`，测试同时报告 `device` 与 `reference` | 不改变结构 | 官方的逐序列回退规则无法用单一 `orientation` 值表达；该扩展只影响数据视图 |
| 增强：`random_yaw`（绕 z 轴 U[0,2π)，同时旋转陀螺、加计与目标的 xy）+ `time_shift(±5)` | 不改变结构 | 与官方一致；不要复刻官方起点钳制错误 |
| unified 预算下 epoch/调度改为 benchmark 统一值；official 配方保留 Adam 1e-4、ReduceLROnPlateau(0.1, 10)、batch 128、按 val 选模 | 不改变结构 | DESIGN §4 两套配方 |

## 7. 官方报告数值

论文 Table 1（“ResNset”列为 RoNIN ResNet；RTE 为 1 分钟窗口；按官方代码口径，见 §3）：

| 数据集 | 测试集 | ATE (m) | RTE (m) |
|---|---|---|---|
| RIDI | seen | 1.63 | 1.91 |
| RIDI | unseen | 1.67 | 1.62 |
| OxIOD | seen | 2.40 | 1.77 |
| OxIOD | unseen | 6.71 | 3.04 |
| RoNIN | seen | 3.54 | 2.67 |
| RoNIN | unseen | 5.14 | 4.37 |

同表基线（RoNIN seen/unseen ATE）：PDR 29.54/27.67，RIDI 17.06/15.66，IONet 31.07/32.03。注意：论文使用完整 RoNIN 数据集，公开版本只有约 50% 序列（README），在公开子集上的数值不可直接比较。

## 8. 忠实性测试建议

- [ ] `sum(p.numel()) == 4_634_882`，`trainable == total`。
- [ ] `param_shapes` 按注册顺序与夹具逐项一致（69 个张量；首项 `[64, 6, 7]`，末项 `[2]`）。
- [ ] 输入 `(4, 6, 200)` → `vel` 形状 `(4, 2)`；中间特征长度链 100/50/50/25/13/7（可用 forward hook 断言）。
- [ ] BN buffer 数量 63（含 `num_batches_tracked`）。
- [ ] 初始化：所有 Linear 的 bias 为 0，权重标准差约 0.01；BN weight 全 1。
- [ ] 偏航等变的**训练期**性质（非结构性质，仅用于检查增强实现）：`random_yaw` 增强后输入 xy 与目标 xy 的旋转角相同，z 分量不变。
- [ ] 回归防护：窗口 T=256 时模型能正常构造与前向（验证没有复刻 `T//32+1` 错误）。

## 9. 预训练权重

README 指向 <https://doi.org/10.20383/102.0543>（FRDR）；2026-09-17 访问时该记录页显示 “No files uploaded”（正在备份）。README 脚注说明**预训练模型是在整个数据集上训练的**（包含测试序列），因此即使可下载，也不得用于 IPB 的测试集评测，只能用于推理流程的冒烟测试。

## 10. 官方实现的坑与未决问题

1. **随机偏移被钳制到 ≥ window_size**（`data_glob_speed.py:172-174`）：该钳制沿用了 `DenseSequenceDataset` 的“帧号是窗口终点”语义，而 `StridedSequenceDataset` 的帧号是窗口**起点**。结果：训练时每条序列的前 200 帧（1 s）从不作为窗口起点，起点 200 的窗口被重复采样约 20 次；尾部起点不受保护，靠 `targets` 长度保证不越界。IPB 实现 `time_shift` 时应钳制到 `[0, N−T]`。
2. `randrange(-5, 5)` 不含 +5，偏移分布轻微不对称。
3. `in_dim = window_size // 32 + 1`（`ronin_resnet.py:126,291`）只在 T 不是 32 的倍数时等于实际特征长度 ⌈T/32⌉；T=256 时为 9 而实际为 8，前向会报错。
4. `get_model('resnet50'/'resnet101')` 会修改全局 `_fc_config['fc_dim']`，之后在同一进程构造 resnet18 会得到 fc_dim=1024；`resnet101` 分支没有传 `kernel_size`（默认值 3，结果相同）。
5. `GlobAvgOutputModule.forward` 调用 `self.avg()` 不带参数（`model_resnet1d.py:145`），该分支不可用；官方只用 FC 头。
6. 轨迹重建把速度赋给窗口起点（半窗错位）；ATE/RTE 按坐标分量平均（欧氏值的 1/√2）。与论文数值对照时必须说明口径。
7. 缓存 `--cache_path` 的校验只比较 `feature_dim/target_dim/aux_dim/grv_only`，不比较 `interval`：ResNet（interval=200）与 LSTM/TCN（interval=1）若共用缓存目录，会静默读到错误目标。
8. `np.int`（`ronin_resnet.py:253,307`）在 NumPy ≥ 1.24 已删除，官方测试脚本在新环境直接报错；`scipy.ndimage.filters` 为弃用路径。
9. 论文说测试时“每 5 帧预测一次”，代码默认 `step_size=10`；论文说 ResNet 末端“加一个 512 单元全连接层”，代码实际是 1×1 过渡卷积 + 两个 512 隐层的 FC 头——以代码为准（参数量 4.63 M 即由此而来）。
10. 训练时的姿态回退使用数据集自带的 `pose/ekf_ori` 或陀螺积分，而非 Tango 位姿；论文把该情形称为“ground-truth 姿态”。`pose/ekf_ori` 的生成方式本卡未核实，IPB 以 `reference` 近似。
