# AirIO（`airio`）

> 规格卡版本：v1（2026-09-17）· fidelity：`official-code` · 夹具：`tests/fixtures/algorithms/airio.json`
> 优先级 P1：原始论文面向**多旋翼无人机**（EuRoC / Blackbird / Pegasus 仿真），用于行人需要接口扩展（见第 6 节）。

## 1. 元信息

| 项 | 值 |
|---|---|
| 论文 | *AirIO: Learning Inertial Odometry with Enhanced IMU Feature Observability*；Y. Qiu, C. Xu, Y. Chen, S. Zhao, J. Geng, S. Scherer；IEEE RA-L 10(9):9368–9375，2025；[arXiv 2501.15659](https://arxiv.org/abs/2501.15659)（v1 2025-01-26，v2 2025-06-16） |
| 官方仓库 | https://github.com/Air-IO/Air-IO @ `7214e8a6ddee95ca07a64f01f1fa3e2f8f60df73`（本地：`/workspace/webCodex/third_party/Air-IO`） |
| 许可 | BSD-3-Clause（Copyright (c) 2025, Carnegie Mellon University）。宽松许可，保留版权声明即可 |
| 框架 | PyTorch + PyPose（`pypose` 只用于数据/姿态与 EKF，网络类本身不用）；配置为 HOCON（`pyhocon`） |
| fidelity | official-code（网络、训练、推理、EKF 均开源；提供三套数据集的预训练权重） |
| 参考文件 | 网络 `model/code.py`（`CodeNetMotionwithRot`，注册名 `codewithrot`；无姿态版 `CodeNetMotion`，注册名 `codenetmotion`），`model/__init__.py`；损失 `model/losses.py`、`model/loss_func.py`；训练 `train_motion.py`；推理 `inference_motion.py`；数据 `datasets/dataset.py`、`datasets/dataset_motion.py`、`datasets/dataset_utils.py`、`datasets/{EuRoC,BlackBird,Pegasus}dataset.py`；配置 `configs/{EuRoC,BlackBird,Pegasus}/motion_body_rot.conf`、`configs/datasets/*/*_body.conf`；评测 `evaluation/evaluate_motion.py`、`utils/velocity_integrator.py`；EKF `EKF/IMUofflinerunner.py`、`EKF/ekf.py` |

## 2. 任务（输入）

AirIO 的核心主张：**不要**把 IMU 旋到全局系，而是保留**机体系**原始 IMU（含重力），并把姿态作为**单独的一路输入**显式编码。

| 项 | 官方 | IPB 映射 |
|---|---|---|
| 采样率 | EuRoC 200 Hz（原生）；Blackbird 在加载器中重采样为 **100 Hz**（`datasets/BlackBirddataset.py:127-137`）；Pegasus 仿真 | 200 Hz |
| 窗口 | `window_size=1000` 帧（EuRoC = 5 s；Blackbird = 10 s）（`configs/datasets/*/*_body.conf`） | `window=1000`（5 s，与论文 EuRoC 设置一致；参数量与窗口无关） |
| 训练步长 | `step_size=3`（EuRoC、Blackbird train），Pegasus 10 | `stride=10`（unified）；official 配方用 3 |
| 验证（代码中叫 test）步长 | 10（Pegasus 20） | – |
| 推理 | `inference_motion.py` 的 `--whole` 永远为真 → 整条序列一次前向（`mode="inference"`，`index_map=[[seq,0,seq_len]]`） | 滑窗 `eval_stride=10`，取窗口最后一个输出 token（见第 6 节） |
| 坐标系 | `coordinate: body_coord`：IMU 保持机体系；标签速度旋到机体系 `v_b,k = R_kᵀ v_w,k`（`BlackBirddataset.py:226-229`） | 输入 `frame=body`；目标 `frame=body` |
| 姿态来源 | 训练：真值姿态；推理：默认真值（`rot_type: None`），可选 AirIMU 校正姿态或原始积分姿态 | `orientation=reference`（official），`device` 为可选 |
| 去重力 | 否（`remove_g=False`，加计保留 `R_kᵀg`） | `remove_gravity=false` |
| 通道顺序 | 网络内部 `torch.cat([acc, gyro], -1)`（**加计在前**，`model/code.py:110`），形状 `(B,T,3)` | 从 `[gyro, acc]` 置换为 `[acc, gyro]`，再转为 `(B,T,6)` |
| 额外输入 | `rot`：`(B,T,3)`，逐样本姿态（body→world）的 so(3) 对数（PyPose `SO3.Log()`，旋转向量，含**绝对偏航**）（`train_motion.py:35`） | **需要接口扩展**：`aux["q"]`，`(B,4,T)`，`body_to_world_wxyz`；模型内部转旋转向量 |
| 输入归一化 | 无 | 无 |
| 目标 | 逐样本机体系速度序列（W+1 帧，含窗口起点），按下采样对齐取样（见第 3 节） | 训练需逐样本目标（接口扩展 `batch["vel_seq_body"]`），窗口级为 `velocity_at_end`、`dims=3` |

`rot` 的时间对齐：`label["gt_rot"]` 含窗口内 W+1 个姿态（起点到终点），代码取前 W 个 `[:, :-1]`，与 W 个 IMU 样本一一对应（`datasets/dataset_motion.py:111-115`）。

## 3. 输出

- `net_vel`：`(B, T_out, 3)`，机体系速度（m/s）；`cov`：`(B, T_out, 3)`，逐轴**方差**，`cov = exp(decoder − 5)`（`model/code.py:44-46`），不是 logstd。
- 序列到序列：`T_out = ⌊(⌊(T−1)/3⌋ + 1 − 1)/3⌋ + 1`（两次 k7/s3/p3 卷积）；T=1000 → 112，T=200 → 23。
- 标签选取 `get_label`（`model/code.py:52-59`）：`s_idx = (k₀−p₀) + s₀(k₁−1−p₁) + 1 = 14`，取标签数组（W+1 帧）中下标 `14, 23, 32, …`（步长 9）；不足 `T_out` 时用最后一帧（下标 W）补齐。T=1000：110 个 + 2 个补齐（下标 1000）；T=200：`[14, 23, …, 194, 200, 200]`。
- 官方轨迹（`evaluation/evaluate_motion.py:96-125`，`utils/velocity_integrator.py`）：输出时间戳 `ts` 同样按 `get_label` 取；把稀疏的机体系速度线性插值到每个 IMU 时刻，用每个时刻的姿态（真值或 AirIMU）旋到世界系，再用**前向欧拉**（矩形）`p_{k+1} = p_k + v_k·dt_k` 从真值起点积分；不对齐。
- EKF（`EKF/IMUofflinerunner.py`）：状态 `[R, V, P, b_g, b_a]`（15 维），IMU 传播输入为 AirIMU 校正后的 IMU 与其协方差；观测为机体系速度 `R⁻¹V`，观测噪声 `R = diag(cov) × 0.1`（`obs_weight`），每当 IMU 时间到达下一个网络输出时间戳时更新一次。

## 4. 网络结构

`CodeNetMotionwithRot(conf)`（`model/code.py:72-118`），`conf.propcov=True`。`CNNEncoder` 每层为 Conv1d → BatchNorm1d → GELU，最后一层后接 **Dropout(0.5)**（`code.py:6-23`）。BN 为 PyTorch 默认。以下为 IPB 配置（`T=1000`）逐层形状：

| # | 层 | 参数（核/步长/填充/通道） | 归一化 | 激活 | dropout | 输出形状 |
|---|---|---|---|---|---|---|
| 0a | IMU 输入 `[acc, gyro]` | 转置为 (B,6,T) | – | – | – | (B,6,1000) |
| 1a | `feature_encoder.net.0` Conv1d | k7 s3 p3，6→32，有偏置 | BN(32) | GELU | – | (B,32,334) |
| 2a | `feature_encoder.net.3` Conv1d | k7 s3 p3，32→64 | BN(64) | GELU | 0.5 | (B,64,112) |
| 0b | 姿态输入 so(3) | (B,3,T) | – | – | – | (B,3,1000) |
| 1b | `ori_encoder.net.0` Conv1d | k7 s3 p3，3→32 | BN(32) | GELU | – | (B,32,334) |
| 2b | `ori_encoder.net.3` Conv1d | k7 s3 p3，32→64 | BN(64) | GELU | 0.5 | (B,64,112) |
| 3 | 拼接（通道维，IMU 在前） | 64+64 | – | – | – | (B,112,128) |
| 4 | `fcn2.0` Linear | 128→64 | BatchNorm1d(64)（在通道维上，`batchnorm2`） | GELU | – | (B,112,64) |
| 5 | `gru1` 双向 GRU | input 64，hidden 64，1 层，batch_first | – | – | – | (B,112,128) |
| 6 | `gru2` 双向 GRU | input 128，hidden 128，1 层 | – | – | – | (B,112,256) |
| 7 | `veldecoder` | Linear 256→128 → GELU → Linear 128→3 | – | GELU | – | `net_vel` (B,112,3) |
| 8 | `velcov_decoder` | Linear 256→128 → GELU → Linear 128→3，再 `exp(x − 5)` | – | GELU | – | `cov` (B,112,3) |

- 参数量（注册的全部参数）：**387,014**（官方与 IPB 配置相同；与窗口长度无关）。
- 其中**未参与前向**的注册参数 32,736 个：继承自父类 `CodeNetMotion.__init__` 的 `cnn`（15,968）以及 `fcn1`（16,512）、`batchnorm1`（256）。前向实际使用的有效参数为 **354,278**。反向传播后这些参数 `grad is None`，Adam 不会更新它们（也不施加权重衰减）。移植时**必须注册这三组参数**才能与官方参数量和 `state_dict` 一致，但不得在前向中使用。
- 分量：`feature_encoder` 15,968；`ori_encoder` 15,296；`fcn2` 8,256；`batchnorm2` 128；`gru1` 49,920；`gru2` 198,144；`veldecoder` 33,283；`velcov_decoder` 33,283。
- 无姿态版 `CodeNetMotion`（论文 “Body” 消融）：330,598（`cnn` → `gru1` → `gru2` → 两个解码器，输入只有 `[acc, gyro]`）。
- 初始化：全部为 PyTorch 默认初始化（代码中没有自定义初始化）。

## 5. 损失与训练配方（official）

| 项 | 值 | 来源 |
|---|---|---|
| 损失 | `loss = weight · ( Huber_δ=0.005(v̂ − v) + cov_weight · mean((v̂ − v)²/σ² + ln σ²) )`，`weight=1e2`，`cov_weight=1e-4`；Huber 为 `F.huber_loss(dist, 0, delta=0.005)` 的逐元素均值；`covaug=True` 时协方差项的残差**不** detach（梯度也流向速度头） | `model/losses.py:11-29`；`model/loss_func.py:59-63,87-89`；`configs/*/motion_body_rot.conf` |
| 阶段切换 | 无（协方差项从第 0 轮起就生效） | 同上 |
| 优化器 | Adam，`lr=1e-3`，`weight_decay=1e-4`（Blackbird 为 1e-3，Adam 的 L2 形式） | `train_motion.py:232-234` |
| 学习率调度 | `ReduceLROnPlateau(mode="min", factor=0.2, patience=5, min_lr=1e-5)`，每轮以“test”集损失 step | `train_motion.py:235-241,285` |
| batch | 128；训练集 shuffle | 配置 |
| epoch | 100（Blackbird 200） | 配置 |
| 梯度裁剪 | 无 | `train_motion.py` |
| 增强 | 无（`motion_collate` 的增强分支为 TODO） | `datasets/dataset_utils.py:102-107` |
| 选模 | 每轮在 `test` 集上计算 `get_motion_RMSE`（窗口内速度误差均值的 RMSE），最小者存为 `best_model.ckpt`；`eval` 集只用于日志 | `train_motion.py:58-88,286-300` |
| 精度 | 数据集张量为 float64 时网络也用 float64（`dtype=train_dataset.get_dtype()`），推理脚本显式 `.double()` | `train_motion.py:229-231`；`inference_motion.py:57` |
| 数据划分 | EuRoC：train/test 都是 MH_01、MH_03、MH_05、V1_02、V2_01、V2_03，eval 为 MH_02、MH_04、V1_03、V2_02、V1_01；Blackbird：自建 train/test/eval 目录（5 条 seen 轨迹）+ 5 条 unseen 轨迹只用于推理 | `configs/datasets/*` |

论文（arXiv v2 HTML）中的对应描述：Huber δ=0.005、λ=1e-4、Adam 初始 lr 1e-3、ReduceLROnPlateau（patience 5，衰减 0.2）、batch 128、编码器 dropout 0.5、训练用真值姿态、测试用 EKF 估计姿态，与代码一致。

## 6. benchmark 适配（unified 配方）

| 改动 | 类型 | 理由 |
|---|---|---|
| 视图 `frame=body`、`remove_gravity=false`，模型内部把 `[gyro, acc]` 置换为 `[acc, gyro]` | 不改变结构 | 与官方 `body_coord` 输入一致 |
| 接口扩展 `aux["q"]`：窗口内逐样本姿态 `(B,4,T)`（`body_to_world_wxyz`，来自 `orientation=reference` 或 `device`），模型内部转成旋转向量（角度取主值 [0, π]，与 PyPose `SO3.Log` 相同） | 不改变结构（需修订 DESIGN §4 的 `forward` 签名或 `InputSpec`） | 姿态编码是 AirIO 的核心输入，无法从 `(B,6,T)` 恢复 |
| `window=1000`（5 s） | 不改变结构 | 网络全卷积 + GRU，参数量与窗口无关；1000 帧与论文 EuRoC 设置相同 |
| 训练目标：逐样本机体系速度 `v_b,k = R_kᵀ v_w,k`，按 `get_label` 在窗口内下标 `14 + 9j` 取样，补齐用窗口末端速度；官方标签数组有 W+1 帧（补齐用下标 W），IPB 窗口只有 T 个位姿样本，补齐改用下标 T−1 | 不改变结构（需视图提供逐样本目标，例如 `batch["vel_seq_body"]`） | 保留官方 seq2seq 监督；末端下标相差 1 个样本（5 ms），在卡片中记录 |
| 窗口级输出：`vel = net_vel[:, -1]`（机体系、窗口末端），`logstd = 0.5·ln(cov[:, -1])`；目标 `target=velocity_at_end`、`frame=body`、`dims=3`；推理时用窗口末端姿态旋回世界系，再取水平分量 | 不改变结构 | IPB 要求每窗口一个速度；最后一个 token 训练时对应窗口末端速度 |
| 时间戳：`velocity_at_end` 应赋给窗口末端 | 协议说明 | 同 DIVE；若 v1 只支持窗口中心，须注明时移 |
| 推理方式差异：官方整条序列一次前向（双向 GRU 看到未来数据，非因果）；IPB 为滑窗，窗口内仍是双向 | 协议差异 | IPB 统一协议；另可提供 “whole-sequence” 变体作为对照 |
| unified 增强（可选）：随机全局偏航 `R_k ← R_z(θ)R_k`，θ~U[0,2π)，IMU 与机体系标签不变 | 不改变结构 | 行人数据集的世界系偏航是任意的；官方姿态编码含绝对偏航，不加增强容易记住航向 |
| 选模改为 IPB 的 val（官方用 “test” 集 RMSE 做选模与调度） | 不改变结构 | 诚实协议：不能用 test 选模 |
| 行人适用性 | 说明 | 论文针对无人机（机体系与运动方向强耦合）；行人手持/口袋时机体系与运动方向解耦，机体系表示未必更优，适合作为“表示方式”对照 |

## 7. 官方报告数值

来源：arXiv v2 HTML（经网页摘要工具读取，建议人工再核对一次原文表格）。ATE：估计与真值位置的 RMSE；RTE：固定时间间隔（论文称 5 s）的相对位移 RMSE；单位 m。

| 数据集 | 指标 | AirIO Net | AirIO EKF | 来源 |
|---|---|---|---|---|
| Blackbird SEEN 平均 | ATE / RTE | 0.486 / 0.312 | 0.403 / 0.299 | Table I |
| Blackbird UNSEEN 平均 | ATE / RTE | 1.300 / 0.853 | 1.309 / 0.834 | Table I |
| EuRoC 平均（MH02/MH04/V103/V202/V101） | ATE / RTE | 3.846 / 1.198 | 3.177 / 1.192 | Table II |
| EuRoC 消融（Body+Attitude） | 平均 ATE | 3.297 | – | Table III |
| Pegasus 消融（Body+Attitude） | 平均 ATE | 3.641 | – | Table III |
| Blackbird unseen 消融（Body+Attitude） | 平均 ATE | 1.294 | – | Table III |
| 实时性 | 平均推理时间 | RTX 2060：28.92 ms；Jetson AGX Orin：74.23 ms | – | Table IV |

同表中 RoNIN（EuRoC 平均 ATE/RTE 6.750/2.516）、TLIO（6.969/3.052）为作者在无人机数据上的复现，不是这些方法在行人数据上的数值。摘要：机体系表示在三个数据集上平均提升 66.7%，姿态编码再提升 23.8%。

## 8. 忠实性测试建议

- [ ] 注册参数总量 == 387,014；`param_shapes` 逐项一致（注册顺序：`cnn` → `gru1` → `gru2` → `veldecoder` → `velcov_decoder` → `feature_encoder` → `ori_encoder` → `fcn1` → `batchnorm1` → `fcn2` → `batchnorm2`；子类重新赋值的 `gru1/gru2/veldecoder/velcov_decoder` 保留在父类注册的位置）
- [ ] 一次前向 + 反向后，`cnn.*`、`fcn1.*`、`batchnorm1.*` 的梯度为 None，其余参数梯度非 None；有效参数 == 354,278
- [ ] 输出形状：T=1000 → (B,112,3)；T=200 → (B,23,3)；T=1001 → (B,112,3)
- [ ] `cov > 0` 且 `cov == exp(raw − 5)`；随机初始化时 `cov` 量级约 `e⁻⁵`
- [ ] `get_label`：W+1=201 帧的标签下标为 `[14, 23, …, 194, 200, 200]`；W+1=1001 时首三个为 `[14, 23, 32]`，末四个为 `[986, 995, 1000, 1000]`
- [ ] 通道置换：交换 IPB 输入的 gyro/acc 块后输出应改变（防止顺序写反）；把同一 IMU 按 `[acc, gyro]` 直接喂给官方模型，输出与移植模型一致（加载同一权重）
- [ ] 全局偏航不变性**不**成立（姿态编码含绝对偏航），测试应断言输出随 `R_z(θ)` 改变，以免误加不变性
- [ ] 损失：`covaug=True` 时协方差项对 `net_vel` 有梯度；`weight=100`、`cov_weight=1e-4`

## 9. 预训练权重

README 提供三个数据集的 AirIO 权重与推理结果（GitHub Release）：
[EuRoC](https://github.com/Air-IO/Air-IO/releases/download/AirIO/AirIO_EuRoC.zip)、
[Blackbird](https://github.com/Air-IO/Air-IO/releases/download/AirIO/AirIO_Blackbird.zip)、
[Pegasus](https://github.com/Air-IO/Air-IO/releases/download/AirIO/AirIO_Pegasus.zip)；
EKF 还需要 AirIMU 的结果（同页的 AirIMU 链接）。权重在无人机数据上训练，不能直接用于行人评测，只适合做“加载官方权重 → 输出一致”的移植核对。

## 10. 官方实现的坑与未决问题

1. **未使用的注册参数**：`CodeNetMotionwithRot` 调用父类构造函数，留下未使用的 `cnn`，又定义了未使用的 `fcn1`、`batchnorm1`，共 32,736 个参数计入总量（`model/code.py:25-36,72-94`）。同时 `interval=9`、`padding_num=3` 也未使用。
2. **通道顺序**：`[acc, gyro]`，与 IPB 的 `[gyro, acc]` 相反（`code.py:62,110`）。
3. **协方差是方差**：`exp(x−5)` 表示 σ²，NLL 写作 `e²/σ² + ln σ²`（没有 1/2 系数）；EKF 直接当方差用并乘 0.1。
4. **用 test 集选模**：`train_motion.py` 以 `test` 集损失做 ReduceLROnPlateau 与最优模型选择；EuRoC 配置里 `test` 与 `train` 是**同一批序列**（只是步长不同），等于用训练数据选模。
5. `inference_motion.py:45` 的 `--whole` 是 `store_true` 且默认 True，无法关闭；推理始终整条序列一次前向，双向 GRU 使用了未来数据，因此是离线非因果估计。
6. 轨迹积分为前向欧拉（梯形公式被注释掉，`utils/velocity_integrator.py:38-41`），与 IPB 的梯形积分不同。
7. EKF 更新条件写成 `io_stamp - data["timestamp"].abs() < 0.001`（`EKF/IMUofflinerunner.py:221`），应为 `abs(io_stamp - t)`；在时间单调时效果等价于“时间到达后更新”，但比较的是差值而非绝对差。
8. Blackbird 被重采样到 100 Hz，同样 1000 帧在 Blackbird 上是 10 s，而在 EuRoC 上是 5 s；README 的 Blackbird 评测命令用 `--seqlen 500`，默认值为 1000，RTE 间隔随命令而变。
9. 姿态编码包含绝对偏航，而训练无偏航增强；跨序列的世界系偏航任意时存在过拟合风险（行人数据集尤其明显）。
