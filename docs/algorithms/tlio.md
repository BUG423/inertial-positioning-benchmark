# TLIO（`tlio`）

> 规格卡版本：v1（2026-09-17）· fidelity：`official-code` · 夹具：`tests/fixtures/algorithms/tlio.json`

## 1. 元信息

| 项 | 值 |
|---|---|
| 论文 | *TLIO: Tight Learned Inertial Odometry*；Wenxin Liu, David Caruso, Eddy Ilg, Jing Dong, Anastasios I. Mourikis, Kostas Daniilidis, Vijay Kumar, Jakob Engel；IEEE RA-L 5(4):5653–5660, 2020；<https://arxiv.org/abs/2007.01867> |
| 官方仓库 | <https://github.com/CathIAS/TLIO> @ `d7051c4ff14834b93931984b75b40f894e39654d`（本地：`/workspace/webCodex/third_party/TLIO`） |
| 许可 | 代码 BSD-3-Clause（`LICENSE`，Facebook/Meta）。官方数据 `golden-new-format-cc-by-nc-with-imus-v1.5` 为 CC BY-NC（仅非商业）。 |
| 框架 | PyTorch（`environment.yaml`）；EKF 为 NumPy + numba |
| fidelity | official-code（网络部分）；EKF 不在 v1 benchmark 范围内（见 §6） |
| 参考文件 | `src/network/model_resnet.py`、`src/network/model_factory.py`、`src/network/losses.py`、`src/network/train.py`、`src/network/test.py`、`src/main_net.py`、`src/dataloader/tlio_data.py`、`src/dataloader/sequences_dataset.py`、`src/dataloader/memmapped_sequences_dataset.py`、`src/dataloader/data_transform.py`、`src/tracker/imu_tracker.py`、`src/tracker/meas_source_torchscript.py`、`src/tracker/scekf.py`、`src/main_filter.py` |

仓库里还有 `resnet_seq`（逐帧输出）与 `tcn` 两种结构（`model_factory.py:13-24`），论文与默认参数（`main_net.py:34` `--arch resnet`）只用 `resnet`。本卡只规定 `resnet`。

## 2. 任务（输入）

| 项 | 官方 | IPB 映射 |
|---|---|---|
| 采样率 | 200 Hz（`main_net.py:48` `--imu_freq 200`；数据集由 1000 Hz IMU 重采样到 200 Hz，`imu0_resampled_description.json` 的 `approximate_frequency_hz=200`） | 200 Hz，不变 |
| 窗口 | 200 样本 = 1.0 s（`past_time=0`、`window_time=1.0`、`future_time=0`，`main_net.py:63-65`；数据侧硬编码 `window_size=200`，`tlio_data.py:41`） | `window=200` |
| 训练步长 | 起点索引每 10 个取一个（`decimator=10`，`memmapped_sequences_dataset.py:87`），且训练集每次取样再加随机偏移 `randint(0,10)`，上限截到 `num_rows-window-1`（`sequences_dataset.py:356-359`） | `stride=10` + 增强 `time_shift`（整数偏移 U{0..9}） |
| 推理步长 | 网络测试 10 样本（20 Hz，`test.py` 同样 `decimator=10`）；EKF 更新频率 20 Hz（`main_filter.py:59` `--update_freq 20`） | `eval_stride=10` |
| 坐标系 | **训练/网络测试**：数据文件中的 `gyr_compensated_rotated_in_World`、`acc_compensated_rotated_in_World`，即每个样本用 VIO 姿态旋到重力对齐世界系（含完整偏航），再做 U[−π, π] 随机偏航增强（`data_transform.py:344-407`）。**论文/EKF**：局部重力对齐系——取窗口**起点**克隆状态的姿态，按外旋 XYZ 欧拉分解去掉偏航 γ（`imu_tracker.py:152-166`），窗口内各样本的相对旋转由陀螺积分得到（`imu_tracker.py:168-186`） | `frame=gravity_yaw_local`（IPB 按窗口**末端**偏航去除；因训练时做全范围随机偏航，起点/末端/世界系三种锚定在分布上等价，属于不改变结构的差异，见 §6） |
| 姿态来源 | 训练与网络测试：VIO 真值姿态；EKF：滤波器自身估计 | `orientation=reference` |
| 去重力 | 否。加速度为比力（静止时世界系 z ≈ +9.81，实测 `496072817492636` 前 20000 样本均值 z=9.8136） | `remove_gravity=false` |
| 通道顺序 | `[gyr_x, gyr_y, gyr_z, acc_x, acc_y, acc_z]`（`sequences_dataset.py:235`；`meas_source_torchscript.py:32`） | 与 `[gyro_xyz, acc_xyz]` 相同，恒等置换 |
| 额外输入 | 无（仓库支持 `mag0`/`barom0`，默认 `input_sensors=["imu0"]`） | 无 |
| 输入归一化 | 无缩放（`sequences_dataset.py:316-327` 的缩放被注释掉）；非有限值置 0（`sequences_dataset.py:315`） | 无；IPB 已剔除含无效样本的窗口 |
| 偏置补偿 | 数据文件中的 IMU 已用离线标定与 VIO 偏置补偿（列名 `*_compensated_*`） | IPB 不做偏置补偿（数据格式无偏置字段）；靠偏置增强覆盖，记为已知差异 |

## 3. 输出

- `mean`：`(B, 3)`，窗口首末样本间的位移 `p[s+199] − p[s]`（米），表达在输入所用的同一重力对齐系中（`sequences_dataset.py:165`，`train.py:88` 取 `targ_dt_World[:, -1, :]`）。
- `logstd`：`(B, 3)`，对角协方差参数 `u`，`Σ = diag(exp(2u_x), exp(2u_y), exp(2u_z))`（论文式 (3)，`covariance_parametrization.py:49-68`）。
- 官方轨迹（两种）：
  1. **TLIO（主方法）**：随机克隆 EKF（`src/tracker/scekf.py`）。以 20 Hz 做状态增广与更新；量测 `h = R_γᵀ (p_j − p_i)`，γ 取第 i 个克隆状态的偏航；量测噪声 `R = 10 · diag(exp(2·max(u, −4)))`（`meas_source_torchscript.py:54-55` 先把 `u<−4` 截到 −4；`main_filter.py:88` `meascov_scale=10`）；马氏距离门限 `11.345`（χ²₃ 的 99% 分位，`scekf.py:552-557`）；更新后边缘化窗口起点之前的全部克隆。
  2. **网络积分（论文称 3D-RONIN，`test.py:60-104`）**：`v = mean / window_time`（**除以 1.0 s**），`p = cumsum(v · dts) + p_gt[0]`，`dts` 为相邻窗口末端时间差的均值（0.05 s）；时间戳取窗口**末样本**，而对照真值取窗口**中心**姿态/位置（`memmapped_sequences_dataset.py:245-285`）。README 明确说明该模式使用真值姿态，只作调试用途。

## 4. 网络结构

`ResNet1D(BasicBlock1D, in_dim=6, out_dim=3, group_sizes=[2,2,2,2], inter_dim=7)`，`inter_dim = (past+window+future)//32 + 1 = 200//32 + 1 = 7`（`train.py:200-208`）。所有卷积 `bias=False`；BN 为 `BatchNorm1d` 默认参数（eps=1e-5，momentum=0.1）。输入 `(B, 6, 200)`：

| # | 层 | 参数（核/步长/填充/通道） | 归一化 | 激活 | dropout | 输出形状 |
|---|---|---|---|---|---|---|
| 0 | `input_block.0` Conv1d | k7 s2 p3, 6→64 | — | — | — | (B,64,100) |
| 1 | `input_block.1-2` | — | BN(64) | ReLU | — | (B,64,100) |
| 2 | `input_block.3` MaxPool1d | k3 s2 p1 | — | — | — | (B,64,50) |
| 3 | `residual_groups.0` ×2 BasicBlock | 每块：conv k3 s1 p1 64→64 → BN → ReLU → conv k3 s1 p1 64→64 → BN → (+恒等) → ReLU | BN | ReLU | — | (B,64,50) |
| 4 | `residual_groups.1.0` BasicBlock | conv k3 **s2** p1 64→128 → BN → ReLU → conv k3 s1 128→128 → BN；捷径 `downsample` = conv k1 s2 64→128 + BN | BN | ReLU | — | (B,128,25) |
| 5 | `residual_groups.1.1` BasicBlock | 128→128，恒等捷径 | BN | ReLU | — | (B,128,25) |
| 6 | `residual_groups.2.0` / `.1` | 同上，128→256（第一块 s2 + 1×1 下采样捷径） | BN | ReLU | — | (B,256,13) |
| 7 | `residual_groups.3.0` / `.1` | 同上，256→512 | BN | ReLU | — | (B,512,7) |
| 8 | `output_block1.prep1` Conv1d | k1, 512→128, bias=False | — | — | — | (B,128,7) |
| 9 | `output_block1.bn1` | — | BN(128) | **无激活** | — | (B,128,7) |
| 10 | 展平 → `fc1` Linear | 896→512 | — | ReLU | Dropout(0.5) | (B,512) |
| 11 | `fc2` Linear | 512→512 | — | ReLU | Dropout(0.5) | (B,512) |
| 12 | `fc3` Linear | 512→3 → `mean` | — | — | — | (B,3) |
| 13 | `output_block2.*` | 与 8–12 完全相同的独立一份 → `logstd` | 同上 | 同上 | 同上 | (B,3) |

两头都直接接在 `residual_groups` 的输出上（同一特征），参数不共享。FcBlock 内的 ReLU 与 Dropout 模块各复用一次（两个 fc 层后共用同一模块实例）。

- 参数量：**5,424,646**（全部可训练）。分解：输入块 2,816；残差组 49,664 / 181,504 / 723,456 / 2,888,704（骨干合计 3,846,144）；每个 FcBlock 789,251（prep1 65,536 + bn 256 + fc1 459,264 + fc2 262,656 + fc3 1,539），两个头合计 1,578,502。
- 参数量随窗口变化（`inter_dim` 改变 fc1）：window=100 → 5,031,430；window=400 → 6,211,078；若把输出改为 2 维 → 5,423,620（夹具 `variants` 记录，不采用）。
- 权重初始化（`model_resnet.py:217-236`）：Conv1d `kaiming_normal_(mode="fan_out", nonlinearity="relu")`；BN weight=1、bias=0；Linear weight ~ N(0, 0.01²)、bias=0；`zero_init_residual=False`。
- 注意：`Bottleneck` 类误用 `BatchNorm2d`、`_initialize` 引用未定义的 `Bottleneck1D`，两者在默认配置下都不会执行，移植时忽略。

## 5. 损失与训练配方（official）

| 项 | 值 | 来源 |
|---|---|---|
| 损失 | 对角高斯 NLL（逐轴）：`ℓ = (d̂−d)² / (2·exp(2u')) + u'`，`u' = max(u, log 1e-3)`；按 `(B,3)` 取均值 | `losses.py:18-22, 58-70`，`train.py:102` |
| 阶段切换 | **epoch < 10**：同一 NLL，但 `u` 先 `detach()`（协方差头不收梯度，均值头的误差按当前 `exp(-2u')` 加权；初始化下 `u≈0`，等价于 `0.5·MSE`）；**epoch ≥ 10**：完整 NLL。epoch 从 1 开始计数（`range(start_epoch+1, epochs)`），故前 9 个 epoch 为第一阶段。论文写的是“先 MSE 训练 10 个 epoch 再切 NLL”，代码中的纯 MSE 分支已被注释（`losses.py:59-64`） | `losses.py:58-70`，`train.py:376` |
| 优化器 | Adam，lr=1e-4，weight_decay=0（论文 §IV-B 一致） | `train.py:334`，`main_net.py:31` |
| 调度 | 构造了 `ReduceLROnPlateau(factor=0.1, patience=10, eps=1e-12)`，但**从未调用 `scheduler.step()`**，学习率实际恒为 1e-4 | `train.py:335-337` |
| batch | 1024（训练与验证） | `main_net.py:32` |
| epoch | 默认 `--epochs 10000`（实际跑 1…9999，需人工停止）；论文：MSE 10 个 epoch + NLL 约 10 个 epoch 收敛 | `main_net.py:33`，论文 §IV-B |
| 权重衰减 | 0 | 同上 |
| 梯度裁剪 | `clip_grad_norm_(max_norm=0.1, error_if_nonfinite=True)`（出现 NaN/Inf 直接报错） | `train.py:112` |
| 增强（仅训练，按此顺序作用在已旋入重力系的整窗上） | ① 偏置：每个样本一个常值，陀螺 U[−0.05, 0.05] rad/s、加计 U[−0.2, 0.2] m/s²（高斯噪声 std=0）；② 重力扰动：绕随机水平轴（方位 U[0, 2π)）旋转 U[0, 5°]，同时作用于陀螺与加计，**目标不旋转**；③ 偏航：绕 z 轴 U[−π, π]，同时旋转陀螺、加计与位移目标（及速度） | `tlio_data.py:47-60, 191-206`，`data_transform.py` |
| 验证 | 不做增强；使用 `val_list.txt` | `train.py:393` |
| 选模/早停 | 无早停；每个 epoch 验证，验证集平均 loss（阶段对应的 NLL）创新低时保存 `checkpoint_best.pt` | `train.py:392-397` |
| 随机种子 | 未设置 | — |
| 数据划分 | 官方列表 `train_list.txt`/`val_list.txt`/`test_list.txt`（本机数据 284/36/36 条）；论文：约 400 条、60 h，80/10/10 随机划分，测试 37 条 | 数据集、论文 §IV-B |

## 6. benchmark 适配（unified 配方）

| 改动 | 类型 | 理由 |
|---|---|---|
| 输入视图 `window=200, stride=10, eval_stride=10, frame=gravity_yaw_local, orientation=reference, remove_gravity=false, dims=3` | 不改变结构 | 与官方训练/推理的采样率、窗口、步长一致。IPB 的 `gravity_yaw_local` 以窗口末端偏航为锚，官方 EKF 以起点为锚、训练数据以世界系为锚；三者只差一个全局偏航，而训练做 U[−π,π] 偏航增强，网络近似偏航不变，因此不影响结构与损失 |
| `target=displacement`（IPB 的 `p[s+T−1]−p[s]` 与官方 `targ_dt_World[:, -1]` 是同一对样本）；`forward` 输出 `vel = mean / ((T−1)·dt) = mean / 0.995 s`，`logstd_vel = logstd − log(0.995)` | 不改变结构 | 损失仍在位移尺度上计算（与官方数值一致），对外按 IPB 接口给窗口平均速度。官方网络积分除以 1.0 s 会带来 0.5% 的系统性速度低估，IPB 用精确时长，记为协议差异 |
| 保留 3 维输出（`dims=3`），评测只用水平分量 | 不改变结构 | TLIO 的贡献就是 3D 位移+不确定度，z 轴监督是官方损失的一部分；改成 2 维会改变 `fc3` 形状（参数量 5,423,620）和 NLL 的轴数。IPB 在重力系下允许 `dims=3` |
| 增强 `[bias_shift(gyro=0.05, acc=0.2), gravity_perturb(max_deg=5), random_yaw(±π), time_shift(0..9)]`，顺序同官方 | 不改变结构 | 与官方训练数据管线一致 |
| 损失阶段切换保留：epoch（1 起）< 10 时 `logstd.detach()`，之后完整 NLL，`logstd` 下限 `log(1e-3)` | 不改变结构 | DESIGN §4 要求专用损失与阶段切换与官方一致 |
| 梯度裁剪 0.1；学习率恒定 1e-4（official 配方复现“调度器未生效”）；unified 配方按 IPB 统一预算设 epoch，但 epoch 数必须 ≥ 10 才会进入 NLL 阶段 | 不改变结构 | 忠实复现官方真实行为；unified 下若使用调度器需在配置中显式声明 |
| 选模依据：official 配方用 val 平均 NLL；unified 配方用 IPB 默认 val ATE | 不改变结构 | DESIGN §6 诚实协议 |
| 不实现 EKF（v1 只评网络 + IPB 积分，等价于论文的 “3D-RONIN”/他文的 “TLIO (wo EKF)”） | 不改变网络结构，但**不是论文主结果的完整系统** | IPB v1 的推理协议是“窗口速度 → 梯形积分”；EKF 需要原始 IMU、偏置状态与滤波器姿态，可作为后续 `tlio_ekf` 单独登记 |
| 不做数据侧偏置补偿 | 不改变结构 | IPB 格式无偏置字段；偏置增强覆盖该差异 |

## 7. 官方报告数值

TLIO 论文没有给出数值表，主要结果以 CDF 图呈现（Fig. 3、8、9）。可核对的文字数值：

| 数据集 | 指标（定义） | 数值 | 来源 |
|---|---|---|---|
| TLIO 头戴数据集测试集（37 条） | 平均偏航漂移与位置漂移相对最佳 RoNIN 式基线的降低 | 27% / 33% | 论文摘要后 §I |
| 同上 | 相对 3D-RONIN-mse 的平均偏航漂移 / 位置漂移降低 | 27% / 31% | §VII-B1（与 §I 的 33% 不一致，原文如此） |
| 同上 | 网络误差落在 3σ 外的比例 | x、y：0.70%；z：0.47% | Fig. 6 说明 |
| 同上 | 马氏距离超过 χ²₃ 99% 分位的样本比例 | 0.30% | §VII-A2 |
| 同上 | 固定协方差基线使用的测量误差标准差 | diag(0.051, 0.051, 0.013) m | §VII-B1 |

指标定义（§VI-A）：ATE = 位置 RMSE；RTE-Δt（Δt=1 s）先去掉窗口起点处的偏航误差再比较相对位移；DR = 终点误差/轨迹长度。

第三方报告（RNIN-VIO 论文，ISMAR 2021，IDOL 数据集，使用真值姿态，单位 m，供量级参考）：

| 模型 | ATE B1 known / unknown | B2 known / unknown | B3 known / unknown | T-RTE B1/B2/B3 | D-RTE B1/B2/B3 | 来源 |
|---|---|---|---|---|---|---|
| TLIO (wo EKF) | 5.58 / 6.08 | 7.23 / 7.09 | 7.04 / 6.30 | 2.05 / 5.00 / 5.32 | 0.21 / 0.22 / 0.22 | RNIN Table 1、2 |
| TLIO | 3.75 / 4.14 | 6.23 / 7.17 | 5.06 / 5.23 | 2.52 / 7.25 / 4.94 | 0.28 / 0.26 / 0.29 | RNIN Table 1、2 |

## 8. 忠实性测试建议

- [ ] `ResNet1D` 等价实现在 `(B,6,200)` 输入下参数量 == 5,424,646，`param_shapes` 按注册顺序与夹具逐项一致（78 个参数张量；BN 缓冲区 66 个，见 `buffer_shapes`）。
- [ ] 输出为 `mean (B,3)` 与 `logstd (B,3)`；`layer_output_shapes` 中时间维依次为 100 → 50 → 50 → 25 → 13 → 7。
- [ ] window=100/400 时参数量分别为 5,031,430 / 6,211,078（验证 `inter_dim = T//32 + 1` 公式）。
- [ ] 损失阶段：构造固定 `mean/logstd/target`，epoch=9 时 `logstd` 的梯度为 0（或 None）、`mean` 的梯度非零；epoch=10 时两者梯度均非零。
- [ ] `logstd` 下限：`u=−10` 时损失按 `u'=log(1e-3)` 计算。
- [ ] 位移↔速度换算：`vel * 0.995 s == mean`，且 IPB `displacement` 目标与 `avg_velocity` 目标满足同一比例。
- [ ] 增强：偏置在整窗内为常值且范围正确；重力扰动不改变目标；偏航旋转同时作用于输入与目标且保持 z 分量不变。
- [ ] 偏航等变（统计性质，训练后）：把输入绕 z 旋转 θ，`mean` 的水平分量近似旋转 θ（不作为单元测试硬性要求）。

## 9. 预训练权重

官方仓库**未发布**预训练权重（README：需要自行生成数据并重新训练）。官方数据集可从 README 中的 Google Drive 链接下载（CC BY-NC）。

## 10. 官方实现的坑与未决问题

1. **MSE→NLL 实为“detach 协方差的 NLL”**：`losses.py:66-69` 在 epoch<10 时只 `detach` 协方差头，并未使用 MSE（MSE 分支在注释里）；与论文描述不同。
2. **学习率调度器未生效**：`train.py:335` 构造了 `ReduceLROnPlateau` 但从不 `step()`。
3. **CLI 参数与数据管线脱节**：`train.py:278-284` 只把 `batch_size/dataset_style/workers` 传给 `TlioData`，`--window_time/--sample_freq/--do_bias_shift/--accel_bias_range/--gyro_bias_range/--perturb_gravity*` 均不生效，数据侧使用 `tlio_data.py:40-60` 的硬编码默认值（恰与 CLI 默认值相同）。改 CLI 只会改变 `inter_dim`，可能导致形状不匹配。
4. **偏置增强加在世界系**：偏置在样本已旋到重力系之后才加入（`data_transform.py:250-257`），并非机体系偏置。
5. **网络积分的时间对齐**：`test.py` 用窗口末样本时间戳，却与窗口中心的真值比较，并用 1.0 s（而非 0.995 s）换算速度；IPB 不沿用。
6. **窗口锚定帧**：训练数据是世界系+随机偏航，EKF 用窗口起点偏航的局部系（论文 §IV-B 说“窗口起点姿态构造的重力系”）；IPB 用末端偏航，依赖偏航增强保证等价。
7. `epochs` 循环为 `range(start_epoch+1, epochs)`，实际比参数少跑一个 epoch；`checkpoint_latest.pt` 存在时会被自动续训。
8. 论文中“37 条测试序列”与公开数据列表 36 条不一致。
