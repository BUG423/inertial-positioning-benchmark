# RNIN（RNIN-VIO 的惯性网络，`rnin`）

> 规格卡版本：v1（2026-09-17）· fidelity：`official-code` · 夹具：`tests/fixtures/algorithms/rnin.json`

## 1. 元信息

| 项 | 值 |
|---|---|
| 论文 | *RNIN-VIO: Robust Neural Inertial Navigation Aided Visual-Inertial Odometry in Challenging Scenes*；Danpeng Chen, Nan Wang, Runsen Xu, Weijian Xie, Hujun Bao, Guofeng Zhang；ISMAR 2021, pp. 275–283, DOI 10.1109/ISMAR52148.2021.00043；<http://www.cad.zju.edu.cn/home/gfzhang/papers/rnin_vio.pdf>，项目页 <https://zju3dv.github.io/rnin-vio/> |
| 官方仓库 | <https://github.com/zju3dv/rnin-vio> @ `b030ecc94f151159973a9086c8ba76e22bdbc56e`（本地：`/workspace/webCodex/third_party/rnin-vio`） |
| 许可 | Apache-2.0（`LICENSE` 与文件头；README 注明知识产权属 SenseTime）。子目录 `ronin_3d/` 为 RoNIN 的 3D 扩展，带 RoNIN 自己的 GPL-3 `LICENSE`，本卡不涉及 |
| 框架 | PyTorch（`requirements.txt` 未锁版本） |
| fidelity | official-code（仅公开了惯性网络 RNIN；RNIN-VIO 的滤波融合部分未开源） |
| 参考文件 | `model/model_lstm.py`、`model/losses.py`、`model/function.py`、`dataloader/dataset.py`、`train.py`、`test.py`、`main_net.py`、`config/default.yaml`、`config/resnet_lstm.yaml`、`config/configer.py`、`utils/postprocess.py`、`utils/metric.py` |

## 2. 任务（输入）

| 项 | 官方 | IPB 映射 |
|---|---|---|
| 采样率 | 原始手机 IMU 线性插值到 **100 Hz**（`default.yaml:13` `imu_freq: 100.0`；`dataset.py:93-106`；论文 §4.4） | IPB 200 Hz；模型内部 `x[..., ::2]` 抽取到 100 Hz。100 Hz 网格是 IPB 200 Hz 网格的子集，与官方逐点线性插值等价（官方也不做抗混叠滤波） |
| 窗口 | 子窗口 100 样本 = 1.0 s（`window_time: 1.0`，`past/future_time: 0`）；LSTM 序列长度 10（`train.seq_len: 10`），一个训练样本共 10 × 100 = 1000 样本 = 10 s，**子窗口互不重叠**（`dataset.py:293-298, 339-344`） | `window=200`（被预测的最后 1 s）+ **`history=1800`**（前 9 s 上下文），模型输入 `(B, 6, 2000)`。`history` 是 IPB InputSpec 需要新增的字段，见 §6 |
| 训练步长 | `step_size = imu_freq / sample_freq = 100/20 = 5` 样本（0.05 s）（`dataset.py:148, 177-180`） | `stride=10` |
| 推理步长 | 同为 5 样本 @100 Hz（20 Hz 输出） | `eval_stride=10` |
| 坐标系 | 逐样本用完整姿态旋到重力对齐世界系：`glob = R_wb · imu`（`dataset.py:116-117`） | `frame=gravity_world` |
| 姿态来源 | 优先 `gt_q_*`（VICON/BVIO 真值）；若真值列为常值/单位四元数则改用 `vio_q_*`（`dataset.py:75-83`） | `orientation=reference` |
| 去重力 | **是**：世界系加速度减 `[0, 0, 9.805]`（`dataset.py:118`；论文式 (1)） | `remove_gravity=true`，重力常数取 9.805 m/s²（写进模型配置） |
| 偏置补偿 | 旋转前在机体系减去 VIO 估计的偏置，取**整条序列最后一行**的偏置值（`dataset.py:87-91`） | 不做（IPB 格式无偏置字段），记为已知差异 |
| 通道顺序 | `[gyro_xyz, acc_xyz]`（`dataset.py:121`） | 与 IPB 相同，恒等置换 |
| 额外输入 | 过去 9 个子窗口（共 10 个）组成的序列；LSTM 初始状态为零 | 由 `history=1800` 提供 |
| 输入归一化 | 无 | 无 |
| 样本筛选 | 训练/验证集中**使用 VIO 位姿的序列**（`get_gt=False`）剔除任一子窗口位移/1 s > 4 m/s 的样本（`dataset.py:163, 176-191`）；原始序列少于 1000 行整条丢弃，并裁掉开头 10 个、结尾约 20 个原始样本（`dataset.py:72-73, 93-95`） | 可选过滤 `max_subwindow_speed=4.0`（仅对参考位姿来自 VIO 的数据集启用） |

## 3. 输出

- `mean`：`(B, 10, 3)`，每个子窗口的 3D 位移（米），世界系。第 k 个子窗口的目标是 `d_k = p[j+(k+1)W] − p[j+kW]`（`W=100`，恰好 W 个采样间隔 = 1.0 s，终点是下一个子窗口的首样本；`dataset.py:114, 296-298`），因此 `Σ_{i≤k} d_i = p[j+(k+1)W] − p[j]` 严格成立（绝对损失依赖这一“可伸缩求和”性质）。
- `logstd`：`(B, 10, 3)`，对角协方差参数 `log σ`；只有 `forward(x)`（`compute_type=None`）才计算，`forward(x, 'dp')` 只返回 `mean`（`model_lstm.py:230-239`）。
- 官方轨迹（`test.py` + `utils/postprocess.py:24-54`）：
  1. 推理用 `fun_test_forward`（`function.py:317-337`），`epoch=1000 ≤ start_cov_epochs=2000` → `forward(x,'dp')`，`pred_cov` 置零；只取**最后一个子窗口** `mean[:, -1]`。
  2. `v = mean[:, -1] / window_time`（1.0 s）；该速度的时间戳为最后一个子窗口的中心 `j + W/2 + 9W`。
  3. `pos[0] = p_gt[该时刻]`，`pos[k+1] = pos[k] + v_k · dts`（`dts` = 相邻预测时间差均值 0.05 s），再线性插值到 100 Hz 时间戳。评测轨迹从序列开始后约 9.5 s 才开始。
  4. 指标在 **3D** 上计算：ATE（RMSE）、T-RTE（窗口 `imu_freq×60` 样本 = 60 s；序列不足 1 min 时按比例缩放）、D-RTE（参考轨迹每 1 m）（`utils/metric.py`）。
- RNIN-VIO 中网络输出以偏航锚定的相对位移约束融合进平方根逆滤波器（论文 §5.5，协方差×10），该部分代码未公开。

## 4. 网络结构

`ResNetLSTMSeqNet(cfg)`，cfg = `default.yaml` 合并 `resnet_lstm.yaml`（`configer.py:121-132`）：`input_dim=6, output_dim=3, layer_sizes=[2,2,2,2], lstm_size=256, lstm_layers=1, lstm_dropout=0.0`，`win_size = 100`，`resnet_code = 128 · int(win_size/16 + 1) = 896`。所有卷积 `bias=False`，BN 为 `BatchNorm1d` 默认参数。官方输入 `(B, 10, 6, 100)`，先 `view` 成 `(B·10, 6, 100)`：

| # | 层 | 参数（核/步长/填充/通道） | 归一化 | 激活 | dropout | 输出形状 |
|---|---|---|---|---|---|---|
| 0 | `input_block.0` Conv1d | k7 s2 p3, 6→64 | — | — | — | (B·10,64,50) |
| 1 | `input_block.1-2` | — | BN(64) | ReLU | — | (B·10,64,50) |
| — | （**无** MaxPool，区别于 TLIO/RoNIN） | | | | | |
| 2 | `residual_groups.0` ×2 ResBlock | 每块 `convs` = conv k3 s1 p1 → BN → ReLU → conv k3 s1 p1 → BN；`out = convs(x).clone() + identity` → ReLU | BN | ReLU | — | (B·10,64,50) |
| 3 | `residual_groups.1` ×2 | 第一块 conv1 **s2**，64→128，捷径 `downsample` = conv k1 s2 + BN；第二块恒等捷径 | BN | ReLU | — | (B·10,128,25) |
| 4 | `residual_groups.2` ×2 | 同上 128→256 | BN | ReLU | — | (B·10,256,13) |
| 5 | `residual_groups.3` ×2 | 同上 256→512 | BN | ReLU | — | (B·10,512,7) |
| 6 | `resnet_post_pro` | conv k1 512→128（bias=False）→ BN(128)，**无激活** | BN | — | — | (B·10,128,7) |
| 7 | 展平并 `view(B, 10, 896)` | — | — | — | — | (B,10,896) |
| 8 | `lstm` LSTM | input 896, hidden 256, 1 层, `batch_first=True`, **单向**（`bidirectional=False`）, dropout 0；h0=c0=0 | — | tanh/sigmoid | — | (B,10,256) |
| 9 | `view(B·10, 256)` → `output_block1.fcs` | Linear 256→256 → ReLU → Dropout(0.2) → Linear 256→256 → ReLU → Dropout(0.2) → Linear 256→3 | — | ReLU | 0.2 | (B·10,3) → `view(B,10,3)` = `mean` |
| 10 | `output_block2.fcs` | 与 9 相同的独立一份 | — | ReLU | 0.2 | (B,10,3) = `logstd`（仅 `compute_type=None`） |

- 参数量：**5,358,342**（全部 `requires_grad=True`；79 个参数张量，BN 缓冲区 63 个）。分解：输入块 2,816；残差组 49,664 / 181,504 / 723,456 / 2,888,704；`resnet_post_pro` 65,792；LSTM 1,181,696（`4·256·(896+256) + 2·4·256`）；每个 FcBlock 132,355，两个共 264,710（其中协方差头 132,355）。
- 若直接在 200 Hz 上用 200 样本子窗口（`imu_freq=200`），`resnet_code=1664`，参数量变为 6,144,774（夹具 `variants` 记录，**不采用**）。
- 模型支持任意序列长度 L（ResNet 逐子窗口独立处理，LSTM 按 L 展开），已验证 L=1、5 输出 `(B, L, 3)`。
- 权重初始化（`model_lstm.py:158-187`）：Conv1d `kaiming_normal_(fan_out, relu)`；BN weight=1、bias=0；Linear weight ~ N(0, 0.01²)、bias=0；LSTM `weight_ih_l0`、`weight_hh_l0` 正交初始化，`bias_ih_l0=0`，`bias_hh_l0=0` 且其遗忘门切片 `[256:512]` 置 1。
- `freeze_cov()`/`unfreeze()` 已定义但全仓库无调用。

## 5. 损失与训练配方（official）

**官方默认配置下的真实行为**：`train.epochs=201`，`train.start_cov_epochs=2000`（`default.yaml:33-34`）。`fun_train_forward` 与 `get_sequence_smooth_loss` 都以 `epoch <= start_cov_epochs` 判断阶段（`function.py:294`，`losses.py:254`），而 epoch 最大为 200，**因此训练全程停留在“位移阶段”，NLL 阶段永远不会触发**：

- 前向用 `model(feat, 'dp')`，协方差头 `output_block2` 不进入计算图，训练、验证、测试中都拿不到梯度（`.grad` 为 None，Adam 跳过），权重始终保持初始化值；`pred_cov` 被置零，`test.py` 写出的 σ 恒为 `exp(0)=1`。
- 损失（`losses.py:249-270`）：
  - 相对损失 RL：`ℓ₁ = (d̂ − d)²`，形状 `(B,10,3)`；
  - 绝对损失 AL：`ℓ₂ = 8 · (cumsum(d̂)[:,1:] − cumsum(d)[:,1:])²`，形状 `(B,9,3)`（第 0 步的累积项已包含在 RL 中，故从第 1 步开始，权重 `absolute_weight=8.0`）；
  - `loss = mean(cat([ℓ₁, ℓ₂], dim=1))`，即对 `B×19×3` 个元素取平均。
- 若把 `start_cov_epochs` 设得小于训练轮数，则 epoch > start_cov_epochs 时：`forward(x)` 同时输出 `logstd=u`；`ℓ₁ = (d̂−d)²/(2e^{2u}) + u`；`Σ_cum = cumsum(e^{2u})`（按独立假设累加方差），`ℓ₂ = 8·[(D̂−D)²/(2Σ_cum) + ½·log Σ_cum]`（仅 k=1..9）；同样 `mean(cat)`。无 detach、无下限截断。论文 §4.4 说“先 MSE 训练到收敛，再切 NLL 训练到收敛”，但**没有给出切换 epoch**，公开配置也没有实现该切换。

| 项 | 值 | 来源 |
|---|---|---|
| 损失 | 见上：RL + 8×AL（MSE 形式）；协方差头不训练 | `losses.py`，`function.py` |
| 优化器 | Adam，lr=1e-4，weight_decay=0（多卡时 lr × world_size） | `default.yaml:39-42`，`configer.py:177-179`，`main_net.py:373-374` |
| 调度 | `ReduceLROnPlateau(mode=min, factor=0.1, patience=10, eps=1e-12)`，每个 epoch 用验证集平均 loss `step()` | `train.py:76-79, 171` |
| batch | 训练 32；验证/测试 1024 | `default.yaml:32, 47, 50` |
| epoch | `range(1, 201)` 最多 200 个；学习率 < 1.1e-6 时提前停止（即第二次降学习率 1e-4→1e-5→1e-6 后停止）；论文：IDOL 上约 150 epoch 收敛 | `train.py:159, 191-192` |
| 权重衰减 | 0 | 同上 |
| 梯度裁剪 | 无 | `train.py:140-141` |
| 增强（仅训练，作用于整个 10 s 样本，按此顺序） | ① 偏航：U[0, 2π) 旋转陀螺、加计、目标的 xy 分量；② 偏置：每个样本一个 6 维常值（10 个子窗口共享），陀螺 U[−0.002, 0.002] rad/s，加计 U[−0.1, 0.1] m/s²，加在世界系、去重力后的信号上；③ “重力噪声”：绕随机水平轴（方位 U[0,2π)）旋转 U[0, 5°]，作用于陀螺与**已去重力**的加计，目标不变；④ 白噪声：陀螺 σ=1e-5、加计 σ=1e-4，逐样本 | `default.yaml:15-22`，`dataset.py:300-336` |
| 验证 | 不增强 | `dataset.py:280-281` |
| 选模/早停 | 验证 loss 创新低时保存 `checkpoints/checkpoint_<epoch>.pt`（否则若训练 loss 创新低，存到 `best_train/`）；测试时按自然排序取 `checkpoints/` 中最后一个文件，即最佳验证模型；论文：选验证 loss 最好的模型 | `train.py:172-177`，`configer.py:164-173` |
| 随机种子 | 42（`random`/`numpy`/`torch`，cudnn deterministic），DataLoader worker 种子 42+id | `default.yaml:1-3`，`main_net.py:59-67, 88-89` |
| 数据划分 | 目录 `data_train/`、`data_val/`、`data_test/`（`random_partition: False`）；论文：80/10/10 | `default.yaml:5-12` |

## 6. benchmark 适配（unified 配方）

| 改动 | 类型 | 理由 |
|---|---|---|
| **协议扩展**：InputSpec 增加 `history`（窗口前的上下文样本数）。RNIN 取 `window=200, history=1800`，模型输入 `(B, 6, history+window) = (B, 6, 2000)`；预测速度的时间戳仍按 DESIGN §5 取**被预测窗口** `[s, s+200)` 的中心 | 不改变结构（需修改 DESIGN §3/§4，属于协议层） | 官方推理只输出最后 1 s 的位移，时间戳在最后一个子窗口中心；若把 10 s 当作 IPB 窗口，时间戳会错位 4.5 s，且若对 10 个输出取平均会变成 10 s 滑动平均速度，偏离官方行为 |
| 模型内部：`x[..., ::2]` → `(B,6,1000)` → `view(B,6,10,100).permute(0,2,1,3)` → `(B,10,6,100)`，偶数偏移相对于输入首样本 `s−1800` | 不改变结构 | 精确复现官方 100 Hz、1 s 子窗口、10 步序列；参数量与官方一致（5,358,342） |
| 序列开头（`s < 1800`）推理时只用可用的完整子窗口：`L = 1 + ⌊s/200⌋`，输入最近 L 个子窗口（输入长度 `200·L`），取最后一步输出；训练只用 L=10 的完整样本 | 不改变结构 | 官方不评估前 9.5 s；IPB 要求所有模型覆盖同一时间轴（DESIGN §5），而该网络天然支持变长序列，避免零填充污染特征 |
| 输出：`vel = mean[:, −1] / 1.0 s`，`dims=3`，评测只用水平分量；official 配方**不输出** `logstd` | 不改变结构 | 与官方 `postprocess.pose_integrate` 一致；协方差头在官方默认配置下未训练，输出无意义 |
| 训练辅助目标 `seq_disp`（B,10,3）：`d_k = p[s−1800+200(k+1)] − p[s−1800+200k]`，k=0..9；最后一项需要窗口外一个样本 `p[s+200]`，视图需提供，缺失则该窗口不参与训练 | 不改变结构（视图需支持模型专用目标） | 保持官方“W 个间隔、首尾相接”的目标定义，使绝对损失中的累积位移严格等于位置差 |
| 窗口级指标仍用 IPB 标准 `avg_velocity`（`(p[s+199]−p[s])/0.995 s`） | 不改变结构 | 与模型输出（`[s, s+200]` 的 1.0 s 平均速度）只差 5 ms 采样，二者都是无偏的窗口平均速度 |
| 视图：`frame=gravity_world, orientation=reference, remove_gravity=true (g=9.805), stride=10, eval_stride=10` | 不改变结构 | 与官方输入定义一致 |
| 增强 `[random_yaw(0..2π), bias_shift(gyro=0.002, acc=0.1, per-sample), gravity_rotate(max_deg=5, after gravity removal), white_noise(gyro=1e-5, acc=1e-4)]`，顺序同官方，作用于整个 2000 样本输入 | 不改变结构 | 复现官方数据管线（包括“重力扰动加在去重力之后”这一特殊行为） |
| 损失：official 配方 = 官方默认真实行为（RL + 8·AL，协方差头不参与）；提供 `start_cov_epoch` 配置（默认“从不切换”），用于论文描述的 MSE→NLL 变体，切换点需实验者显式给出 | 不改变结构 | DESIGN §4 要求损失与阶段切换与官方一致；官方代码与论文不一致时以代码为准并如实记录 |
| official 配方：Adam 1e-4、batch 32、ReduceLROnPlateau(0.1, 10) 按 val loss、lr<1.1e-6 停止、最多 200 epoch、种子 42；unified 配方：IPB 统一预算与调度，选模用 val ATE | 不改变结构 | DESIGN §4/§6 |
| 不做 VIO 偏置补偿；速度 > 4 m/s 的筛选仅在参考位姿来自 VIO 的数据集上启用 | 不改变结构 | IPB 无偏置字段；筛选规则与官方触发条件一致 |
| 不实现 RNIN-VIO 滤波融合 | — | 代码未公开，且不属于纯惯性 benchmark 范围 |

## 7. 官方报告数值

RNIN-VIO 论文，IDOL 数据集（训练只用 Building 1，已知/未知用户），全部方法使用真值姿态；ATE、T-RTE（1 min）、D-RTE（1 m）定义沿用 IDOL 论文，单位 m（表中未注明单位，Table 5 标注为 m）。

| 数据集 | 指标 | Ours（RNIN） | Ours (wo AL) | 来源 |
|---|---|---|---|---|
| IDOL B1 known / unknown | ATE | 2.71 / 3.62 | 3.43 / 4.78 | Table 1 |
| IDOL B2 known / unknown | ATE | 6.19 / 5.23 | 9.53 / 7.41 | Table 1 |
| IDOL B3 known / unknown | ATE | 4.57 / 3.38 | 9.77 / 9.65 | Table 1 |
| IDOL B1 / B2 / B3 | T-RTE | 1.49 / 4.19 / 2.99 | 1.72 / 5.05 / 3.60 | Table 2 |
| IDOL B1 / B2 / B3 | D-RTE | 0.17 / 0.19 / 0.20 | 0.25 / 0.23 / 0.24 | Table 2 |
| IDOL（API 姿态） | ATE / T-RTE / D-RTE | 6.53 / 6.75 / 0.32 | — | Table 3 |
| IDOL（真值姿态） | ATE / T-RTE / D-RTE | 4.28 / 2.89 / 0.19 | — | Table 3 |
| 自采数据 Outdoor | ATE / T-RTE / D-RTE | 1.55 / 1.81 / 0.15 | — | Table 4 |
| 自采数据 Indoor | ATE / T-RTE / D-RTE | 1.24 / 1.43 / 0.30 | — | Table 4 |
| 户外 ~430 m 序列（纯 RNIN） | FTD (%) | Normal00 2.67，Normal01 5.35，Occlusion00 3.88，Occlusion01 4.14，Far Scenes00 4.78，Far Scenes01 5.22，Shake00 5.59，Shake01 7.29，All Challenging00 7.23，All Challenging01 2.17 | — | Table 6 |

同表中 RoNIN-ResNet 的 IDOL ATE 为 B1 2.90/4.46、B2 15.27/11.82、B3 10.03/13.74（Table 1），可作为交叉参考。论文未说明表中 RNIN 是否使用了 NLL 阶段。

## 8. 忠实性测试建议

- [ ] 官方配置实例化后参数量 == 5,358,342，79 个参数张量的 `param_shapes` 与夹具逐项一致；`lstm.weight_ih_l0` 形状 `[1024, 896]`。
- [ ] `forward(x)` 输出 `mean`、`logstd` 均为 `(B,10,3)`；`forward(x,'dp')` 只返回 `mean`，且 `output_block2` 的前向钩子不被调用。
- [ ] 变长：输入 `(B,L,6,100)`（L=1..10）输出 `(B,L,3)`；IPB 适配器对 `(B,6,200·L)` 输入给出 `vel (B,3)`。
- [ ] 抽取/切分：`(B,6,2000)` 适配后第 k 个子窗口等于 `x[:, :, 200k:200k+200:2]`。
- [ ] 默认损失数值：`pred=0`、`targ=1`、形状 `(1,10,3)` 时 loss = (30 + 3·8·Σ_{m=2}^{10} m²)/57 = 9246/57 ≈ 162.2105。
- [ ] 默认阶段：任意 epoch ≤ 2000 反向传播后，`output_block2` 所有参数 `.grad is None`，其余参数梯度非空。
- [ ] 变体阶段：`start_cov_epoch=0`、epoch=1 时协方差头获得梯度，且 AL 项使用累积方差 `cumsum(exp(2u))`。
- [ ] 增强：偏航只改变 xy 且与目标一致旋转；偏置在 2000 个样本上为同一常值；“重力噪声”不改变目标。
- [ ] 若误用 200 Hz 子窗口，参数量会变为 6,144,774——测试应能检出。

## 9. 预训练权重

README 提供了在作者自采数据上训练的预训练模型：Google Drive <https://drive.google.com/file/d/1BsnJCL1alqCT6HHoBKZ3sx6zSMd_MgDE/view?usp=sharing>，百度网盘 <https://pan.baidu.com/s/1ya3hzBHaDBLwrvkP8L6glw>（提取码 d8mm）。本卡未下载核验，不确定该权重的协方差头是否经过训练；数据集下载链接同见 README。

## 10. 官方实现的坑与未决问题

1. **协方差从未训练**（最重要）：`start_cov_epochs=2000 > epochs=201`（`default.yaml:33-34`），`function.py:294-296` 始终走 `'dp'` 分支；协方差头保持随机初始化，测试输出的 σ 固定为 1。论文描述的 MSE→NLL 切换在公开配置下不存在，切换点也未公开。
2. **“重力噪声”加在去重力之后**（`dataset.py:118` 先减 g，`dataset.py:319-329` 才旋转），只把线加速度与角速度旋转 ≤5°，并不能模拟重力方向误差导致的重力泄漏，与 TLIO 的同名增强语义不同。
3. **偏置增强加在世界系、去重力之后**（`dataset.py:312-317`），并非机体系偏置；VIO 偏置补偿用整条序列最后一行的值（`dataset.py:90-91`）。
4. `utils/postprocess.py:32` 使用已从 NumPy 1.24 移除的 `np.int`，新环境下测试脚本直接报错。
5. 训练过程中每个 epoch 都在测试集上推理并记日志（`train.py:186-190`），不参与选模，但属于“偷看 test”，IPB 不沿用。
6. `random_partition: True` 分支调用 `partition_data(..., data_paths=..., out_path=...)`（`main_net.py:301-309`），而函数签名没有这两个参数（`dataset.py:353`），会抛 `TypeError`。
7. 官方评测轨迹从序列开始后约 9.5 s 起算，速度以最后一个子窗口中心为时间戳并向前积分一个 `dts`，指标含 z 轴（3D），与 IPB 的 2D、全时间轴协议不同。
8. `resnet_code = 128·int(win/16 + 1)` 是经验公式而非由网络推导，改变窗口长度时需确认它与实际卷积输出长度一致（100→7、200→13 已验证）。
9. 配置文件 `default.yaml` 与 `resnet_lstm.yaml` 的 `model_param` 完全重复，后者覆盖前者（`configer.py:126-128`），修改时需两处同步。
10. 论文说手机 IMU 为 200/400 Hz、在线网络以 2 Hz 运行，而训练与离线测试均为 100 Hz 输入、20 Hz 输出。
