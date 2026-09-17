# TartanIMU（`tartanimu`）

> 规格卡版本：v1（2026-09-17）· fidelity：`official-code`（仅公开的 ResNet-LSTM `Foundation_Model`；论文中的 LoRA 适配与在线自适应为 paper-only）· 夹具：`tests/fixtures/algorithms/tartanimu.json`

## 1. 元信息

| 项 | 值 |
|---|---|
| 论文 | *Tartan IMU: A Light Foundation Model for Inertial Positioning in Robotics*；Shibo Zhao, Sifan Zhou, Raphael Blanchard, Yuheng Qiu, Wenshan Wang, Sebastian Scherer（CMU）；CVPR 2025, pp. 22520–22529；<https://openaccess.thecvf.com/content/CVPR2025/papers/Zhao_Tartan_IMU_A_Light_Foundation_Model_for_Inertial_Positioning_in_CVPR_2025_paper.pdf>；项目页 <https://superodometry.com/tartanimu> |
| 官方仓库 | <https://github.com/superxslam/TartanIMU> @ `8a8025000f85d2c6b0692b88e15c2c91e761f7d7`（本地：`/workspace/webCodex/third_party/TartanIMU`） |
| 许可 | 代码 Apache-2.0；Hugging Face 权重卡 YAML 标 `license: mit`，正文又写 “provided for research / non-commercial use”——**权重许可存在矛盾**，benchmark 只允许本地核对，不再分发 |
| 框架 | PyTorch ≥ 2.0（依赖 einops、scipy、rich、pyyaml；训练 CLI 需 wandb） |
| fidelity | official-code（公开 LSTM 版）；README 明确 “Transformer 核心未公开”，`model_name: Transformer` 抛 `NotImplementedError`；LoRA 依赖外部 `lora` 包，仓库未提供 |
| 参考文件 | `tartan_imu/model/backbones/lstm.py`（组装）、`tartan_imu/model/lstm/trunks.py:18-274`（主干）、`tartan_imu/model/lstm/heads.py`（多头）、`tartan_imu/model/common/blocks.py:92-164`（ResBlock/FcBlock）、`tartan_imu/model/common/losses.py`、`tartan_imu/model/common/function.py`（训练/测试前向与损失调度）、`tartan_imu/dataloader/dataset_AirLab.py`、`tartan_imu/dataloader/_common.py`、`tartan_imu/evaluation/postprocess.py:125-`（积分）、`train.py`、`main_net.py`、`test.py`、`config/resnet_lstm_multihead.yaml`、`config/datasets/tartanimu/tartan_imu_dataset.yaml`；HF 配置 `config/unified.yaml` |

公开实现与论文的对应关系：

| 论文组件 | 公开代码 | 本卡处理 |
|---|---|---|
| Stage 1：ResNet 共享主干 + LSTM + 四平台多头（car/dog/drone/human） | 有（`Foundation_Model`） | official-code，夹具锁定 |
| 论文式 (1)：用已知姿态从机体系加计中减去重力、减去已知零偏 | **无**：发布配置 `use_local_coord: True`，机体系 IMU **保留重力**、不使用任何姿态（HF 卡称 “honest body frame with gravity present”） | 以代码为准，论文做法记为差异 |
| 损失：式 (2) MSE 相对速度损失 + 式 (3) 对角高斯 NLL | 代码实际为 `20·L1`，NLL 在 body 模式下从不参与（见第 5 节） | 以代码为准 |
| Stage 2：LoRA 微调（摘要 “1.1 M 可训练参数”，§3.3 又写 “5 million parameters” 的适配器） | `train.py` 中 `import lora` 失败则退化为普通微调，仓库不含 `lora` | paper-only，不实现 |
| Stage 3：在线测试时自适应 + GMM 自适应训练缓冲 | `Trainer.online_adaptation` 骨架存在，GMM 缓冲未见实现 | paper-only，不实现 |
| Transformer 版 | 仅保留注册名 | 不实现 |

## 2. 任务（输入）

| 项 | 官方 | IPB 映射 |
|---|---|---|
| 采样率 | 200 Hz（`data.imu_freq: 200.0`）；论文称所有数据集统一到 200 Hz | 200 Hz |
| 窗口 | 子窗 200 样本 / 1 s（`window_time: 1.0`），**每个样本序列 10 个首尾相接的子窗 = 2000 样本 / 10 s**（`train.seq_len: 10`）；每个子窗在送入网络前做 `[::5]` 抽取（`sample_freq: 40` → `step_size = 200/40 = 5`），得到 40 点、名义 40 Hz，**无抗混叠滤波**（`dataset_AirLab.py:569-583`） | 见第 6 节：`window=2000`（10 子窗）或 `window=200`（单子窗） |
| 训练步长 | 序列起点每 5 个样本一个（`range(0, N_targ − 9·200, step_size)`，`dataset_AirLab.py:280-286`） | `stride=5`（official）/ `stride=10`（unified） |
| 推理步长 | `test.py`：同样每 5 个样本一个序列，取最后一个子窗的输出；挑战赛脚本 `starter/tartanimu_submission.py`：把相邻不重叠子窗每 10 个打包一次，**10 个输出全部使用** | `eval_stride=5`（official）/ `10`（unified） |
| 坐标系 | **机体系**，含重力（`use_local_coord: True`，`dataset_AirLab.py:119-130`）；论文另称把各数据集机体轴统一为 X 前、Y 左、Z 上（代码不做，需数据预先对齐） | `frame=body`，`dims=3` |
| 姿态来源 | 网络输入不用姿态；测试积分时用**真值姿态**把机体速度旋到世界系 | `orientation=reference`（只用于把预测旋回世界系与构造目标） |
| 去重力 | 否（发布配置）；论文式 (1) 为是 | `remove_gravity=false` |
| 通道顺序 | npz 中 `retargetted_imu = [acc_xyz, gyro_xyz]`，读取后重排为 `[gyro_xyz, acc_xyz]`（`dataset_AirLab.py:95-96, 160-162`） | 与 IPB 相同，无需置换 |
| 额外输入 | `motion_type`（1=car, 2=dog, 3=drone, 4=human，由文件路径推断），用于选择输出头 | 行人数据恒为 `human`（4） |
| 输入归一化 | 无 | 无 |

目标定义（`dataset_AirLab.py:98-106, 169-175`；`_common.py:15-51`）：

1. 逐帧世界速度 `v_w[k] = (p[k+1] − p[k]) / (t[k+1] − t[k])`，逐帧机体速度 `v_b[k] = R(q[k])ᵀ · v_w[k]`；
2. 两者都按 15 m/s 做模长截断（只缩放超限帧）；
3. 子窗目标 `targ[i] = mean(v_b[i : i+200])`（逐帧机体速度的均值，**不是**“窗口平均世界速度再旋到机体系”）；
4. 序列样本 `j` 的 10 个目标为 `targ[j + 200·m]`，`m = 0..9`；任何一个子窗 `‖targ‖ > 5 m/s`（代码写死 `max_v_norm = 5.0`）则整个序列被丢弃；
5. 总时长 < 10 s（2000 样本）的序列不参与。

## 3. 输出

- `forward(x: (B, 10, 6, 40), motion_type=None, predict_cov=False, compute_all_heads=True)` 返回 `dict[head] → (B, 10, 3)`，为每个子窗的**机体系平均速度** `(vx, vy, vz)`（m/s）；`predict_cov=True` 时再返回同形状字典，为 `log σ`（对角标准差的对数，NLL 中方差为 `exp(2·s)`）。
- `compute_all_heads=False` 时只计算 batch 中出现的 `motion_type` 对应的头（训练默认如此）。
- 测试取值（`function.py:211-299`）：`pred = heads[最常见 motion_type][:, −1]`（最后一个子窗）；`pred *= window_time`（=1，数值不变）；`log σ += log(window_time)`；随后 `smooth_velocity_predictions(window=3)`：对**同一 batch 内按顺序排列的相邻序列**做 3 点滑动平均（复制边界填充；batch 边界处不跨越）。
- 官方轨迹（`postprocess.py:125-199`）：序列 `j` 的输出对应子窗 `[j+1800, j+2000)`，积分时刻取 `ind_intg = j + 100 + 1800`（最后子窗中心），用该时刻的**真值姿态** `R(q[ind_intg])` 把 `pred` 旋到世界系，按 `pos[k+1] = pos[k] + v_w[k]·Δt`（Δt 为相邻 `ind_intg` 的时间差，≈0.025 s）累加，起点为 `p[ind_intg[0]]`。评测（`test.py:234-360`）再把轨迹按真值路程切成 **20 m 段**（`test()` 传 `segment_length=20.0`），每段起点重新对齐真值后计算 ATE/T-RTE/D-RTE 等段内指标，并额外给出全程指标。

## 4. 网络结构

构造：`build_backbone("Foundation_Model", cfg)`，`cfg.model_param` 取 `config/resnet_lstm_multihead.yaml`：`input_dim=6, output_dim=3, split_z=True, window_time=1.0, past_time=0, future_time=0, layer_sizes=[2,2,2,2], drop_ratio=0.3, lstm_size=128, lstm_layers=2, lstm_dropout=0.1`；`cfg.data`: `imu_freq=200, sample_freq=40`；`cfg.model.pred_velocity=True`。

**主干 `ResNetLSTMSeqNet`**（`trunks.py:18-274`），每个子窗独立经过 ResNet（`(B·10, 6, 40)`），再按序列送入 LSTM：

| # | 层 | 参数（核/步长/填充/通道） | 归一化 | 激活 | dropout | 输出形状（B·S=N） | 参数量 |
|---|---|---|---|---|---|---|---|
| 0 | reshape | `(B,10,6,40) → (N,6,40)` | | | | (N, 6, 40) | |
| 1 | Conv1d | k=7, s=2, p=3, 6→64, bias=False | BN(64) | ReLU | | (N, 64, 20) | 2 688 + 128 |
| 2 | group0 ×2 | ResBlock(64→64, s=1)：Conv k3 s1 p1 → BN → ReLU → Conv k3 s1 p1 → BN；恒等捷径；相加后 ReLU | BN | ReLU | | (N, 64, 20) | 2×24 832 |
| 3 | group1.0 | ResBlock(64→128, s=2)，捷径 Conv1d k1 s2 bias=False + BN | BN | ReLU | | (N, 128, 10) | 82 688 |
| 4 | group1.1 | ResBlock(128→128, s=1) | BN | ReLU | | (N, 128, 10) | 98 816 |
| 5 | group2.0 | ResBlock(128→256, s=2)，捷径 | BN | ReLU | | (N, 256, 5) | 329 216 |
| 6 | group2.1 | ResBlock(256→256, s=1) | BN | ReLU | | (N, 256, 5) | 394 240 |
| 7 | group3.0 | ResBlock(256→512, s=2)，捷径 | BN | ReLU | | (N, 512, 3) | 1 313 792 |
| 8 | group3.1 | ResBlock(512→512, s=1) | BN | ReLU | | (N, 512, 3) | 1 574 912 |
| 9 | resnet_post_pro | Conv1d k=1, 512→128, bias=False | BN(128) | 无 | | (N, 128, 3) | 65 792 |
| 10 | reshape | `view(B, 10, −1)`（通道优先展平，128×3） | | | | (B, 10, 384) | |
| 11 | LSTM | 输入 384，隐层 128，2 层，单向，`batch_first`，层间 dropout 0.1，**初始 h/c 每次前向置零** | | tanh/sigmoid | 0.1（层间） | (B, 10, 128) | 395 264 |
| 12 | reshape | `view(−1, 128)` | | | | (N, 128) | |
| — | （未使用）`output_block1/2` | 两个 FcBlock(128→256→256→3)，Foundation 路径从不调用，但参数已注册 | | | | — | 2×99 587 |

ResBlock 内部 3×3 卷积均 `bias=False`；BN 默认 `eps=1e-5, momentum=0.1`。

**输出头 `OutputHead`**（`heads.py:18-90`），四个头 `dog, human, car, drone`（`ModuleDict` 注册顺序）结构相同，输入均为 `(N, 128)`：

| # | 子模块 | 结构 | 输出 | 参数量 |
|---|---|---|---|---|
| h0 | `velocity_scale` | Parameter `(1, 3)`，初值 1；`pred_velocity=True` 时逐轴乘到均值输出上 | — | 3 |
| h1 | `output_block1`（xy 均值） | FcBlock：Linear(128→256) → ReLU → Dropout(0.3) → Linear(256→256) → ReLU → Dropout(0.3) → Linear(256→2) | (N, 2) | 99 330 |
| h2 | `output_block2`（log σ） | 同上，最后一层 256→3 | (N, 3) | 99 587 |
| h3 | `output_block1_z`（z 均值） | 同上，最后一层 256→1 | (N, 1) | 99 073 |
| h4 | 合成 | `mean = cat[h1, h3] → view(B,10,3) → × velocity_scale`；`log σ = h2 → view(B,10,3)`（不乘 scale） | (B, 10, 3) | |

- 参数总量：官方配置 **5 698 346**（全部可训练）= 主干 4 506 374（其中 199 174 为未使用的 `model.output_block1/2`）+ 4 × 297 993（每头）。主干（含未用 FcBlock）+ human 头为 4 804 367（夹具 `benchmark_config.trunk_plus_human_head_params`），其中真正参与 human 前向的为 4 605 193。benchmark 配置参数量相同（夹具锁定）。HF 发布的 `checkpoints/expert_human.pt` 与 `checkpoints/unified.pt` 均可 `strict=True` 加载。
- 参考变体（不作为本卡目标）：`config/resnet_lstm.yaml` 单模型（lstm 64×1 层，无多头）4 193 542；`Foundation_Model` 若 `sample_freq=200`（不抽取，LSTM 输入 128×13=1664）为 6 353 706。LSTM 输入宽度 `128·int(L/16+1)` 由 `sample_freq` 决定，40 点子窗对应 3；实测抽取后长度只有 L∈[33, 48] 能跑通。
- 参数注册顺序（夹具 `param_shapes`，159 个张量）：`model.input_block` → `model.residual_groups` → `model.resnet_post_pro` → `model.lstm`（`weight_ih_l0 [512,384]`、`weight_hh_l0 [512,128]`、`bias_ih_l0`、`bias_hh_l0`、`*_l1`）→ `model.output_block1`、`model.output_block2` → `heads.dog` → `heads.human` → `heads.car` → `heads.drone`；每个头内部先 `velocity_scale`，再 `output_block1`、`output_block2`、`output_block1_z`。
- 权重初始化（`trunks.py:153-182`，只作用于主干模块）：Conv1d `kaiming_normal_(fan_out, relu)`；BN weight=1、bias=0；主干内 Linear（即未使用的两个 FcBlock）N(0, 0.01)、bias=0；LSTM 两层 `weight_ih/hh` 正交初始化，四个 bias 置零后把 `bias_hh` 的遗忘门切片 `[H, 2H)` 置 1。**头部的 FcBlock 不做自定义初始化**，保持 PyTorch `nn.Linear` 默认（Kaiming-uniform a=√5，bias U(±1/√fan_in)）；`velocity_scale` 初值全 1。

## 5. 损失与训练配方（official）

以 GitHub 配置 `config/datasets/tartanimu/tartan_imu_dataset.yaml` 为主，HF `config/unified.yaml`（发布权重实际所用）不同处单独列出。

| 项 | 值 | 来源 |
|---|---|---|
| 损失 | `use_local_coord=True` 分支：每个头 `L_h = Σ(mask · 20·|pred − targ|) / (N_h · 10 · 3)`（N_h 为该头样本数），总损失为 batch 中出现的各头 `L_h` 的**平均**；**与 epoch 无关**，协方差不参与 | `losses.py:92-125, 161-189, 269-304`；`function.py:97-182` |
| 优化器 | `torch.optim.Adam(lr=1e-4, weight_decay=0.01)`（耦合 L2，对所有参数含头部；HF 卡称 `unified.pt` 的 “wdfix2” 对头部免除衰减，但该逻辑**不在**此提交中） | `configer.py:206-217` |
| 学习率调度 | `ReduceLROnPlateau(mode=min, factor=0.1, patience=8, eps=1e-6)`，每 epoch 以 val 损失（无 val 时用 train 损失）step | `train.py:179-187, 402-410` |
| 早停 | lr < `min_lr_stop = 1.1e-6` 即停止（即第二次降学习率后停止） | `train.py:129, 448-454` |
| batch | 512（DDP 时每卡 `512 // world_size`，`drop_last=True`）；val 512 | `main_net.py:280-330` |
| epoch | 250（GitHub 配置）；HF `unified.yaml` 为 150 | 配置文件 |
| AMP | `use_amp: True`（GradScaler） | `train.py:120` |
| 梯度裁剪 | 无 | |
| 增强（仅 train，按顺序，整条 10 s 序列共用同一组随机数） | ① 绕机体 z 轴随机偏航 `θ~U[0,2π)`：旋转 gyro_xy、acc_xy、10 个目标的 xy；② 常值零偏：gyro 每轴 `U(−0.002, 0.002)` rad/s，acc 每轴 `U(−0.1, 0.1)` m/s²；③ 重力方向扰动：水平随机轴 `φ~U[0,2π)`、倾角 `U[0, 5°)`，旋转矩阵同时作用于 gyro、acc（含已加零偏）与目标；④ 白噪声：gyro σ=1e-5，acc σ=1e-4；⑤ `add_time_scaling` 未配置 → 关闭。增强在 `[::5]` 抽取之前完成 | `dataset_AirLab.py:497-566` |
| 阶段切换 | 配置 `start_cov_epochs: 200`：`epoch > 200` 时前向改为 `predict_cov=True` 并按 5 个 epoch 线性引入 NLL 权重——但该权重只在 `use_local_coord=False` 分支生效；发布配置下**协方差头永远没有梯度**。HF `unified.yaml` 的 `epochs=150 < 200`，连 `predict_cov` 都不会打开 | `function.py:105-124`；`losses.py:128-142, 177-222` |
| 选模 | 首个 epoch 必存；val 损失相对最优下降 >0.5% 存 `checkpoint_best_val_loss.pt`；否则 train 损失下降 >0.5% 存 `checkpoint_best_train_loss.pt`；每 10 个 epoch 及连续 5 个 epoch 未存时强制存档 | `train.py:779-872`；`training/checkpoint.py:101-115` |
| 数据划分 | 各平台 `train/`、`val/`、`test/` 目录（`random_partition: False`）；训练期间若提供 test loader 也会每 epoch 计算 test 损失（仅记录，不参与选模） | `main_net.py:89-329`；`train.py:416-436` |
| 数据门限 | `max_v_norm = 5.0`、GT 速度截断 15 m/s 均写死在代码中；HF `unified.yaml` 的 `max_v_norm/clip_max_speed`（drone 25 m/s）**在此提交中不被读取** | `dataset_AirLab.py:105-106, 231-233` |
| 训练数据 | 论文：>100 h、8 种平台，来源 SubT-MRS、IDOL（human）、Blackbird、UZH 等；HF：挑战赛 v3（car/dog/drone_v3/human） | 论文 §4；HF 卡 |

## 6. benchmark 适配（unified 配方）

| 改动 | 类型 | 理由 |
|---|---|---|
| 保留完整 `Foundation_Model`（4 头 + 未用 FcBlock），行人数据固定走 `human` 头（`motion_type=4`） | 不改变结构 | 参数量与官方一致（5 698 346），可 `strict` 加载官方权重做本地核对；其余三头在行人训练中无梯度 |
| 模型内部完成 `(B,6,T) → (B,S,6,200) → [..., ::5] → (B,S,6,40)` | 不改变结构 | DESIGN §3 规定输入为 `(B,6,T)`，抽取属于模型内部预处理，须按官方“先增强后抽取、直接取样本 0,5,…,195、不滤波” |
| `frame=body`、`dims=3`、`remove_gravity=false`、`orientation=reference`（仅用于目标构造与预测旋回世界系） | 不改变结构 | 与官方 `use_local_coord=True` 一致；DESIGN 规定 body 系必须 3 维 |
| **方案 A（official 配方）**：`window=2000`（S=10）；`forward` 返回 `{"vel": human[:, −1], "logstd": cov_human[:, −1]（可选）, "aux": {"seq_vel": human (B,10,3)}}`；损失对 10 个子窗目标计算 `20·L1` | 不改变结构 | 忠实复现 seq2seq 训练与“取最后一个子窗”的推理；**需要 DESIGN 扩展**：① 视图提供 `(B,10,3)` 子窗目标；② `InputSpec` 增加历史上下文（例如 `context=1800`），预测器以 `[s+1800, s+2000)` 的中心作为时间戳、目标也按该跨度构造，否则 IPB 预测器会把输出错放到 2000 样本窗口中心（错位 4.5 s） |
| **方案 B（unified 配方，扩展未就绪时）**：`window=200`、S=1，LSTM 只跑 1 步（零初始状态）；报告中标注 `tartanimu@seq1` | 不改变参数结构，但去掉 9 s 时序上下文（功能性改变） | 完全落在现有 DESIGN 接口内；参数形状与 S 无关，仍可加载官方权重 |
| 子窗目标：official 用“逐帧机体速度的均值”；unified 用 IPB `avg_velocity` 在 body 系下的定义（窗口平均世界速度按窗口末端姿态旋入机体系） | 不改变结构 | IPB 统一目标（DESIGN §3 补充约定）；二者在窗口内有转动时不同，属于协议效应，用 oracle 轨迹量化 |
| 预测旋回世界系：官方用最后子窗**中心**的真值姿态；IPB 用视图约定的姿态（窗口末端） | 不改变结构 | IPB 统一推理约定；差异为半个子窗的姿态变化 |
| 测试时 3 点滑动平均不复刻（IPB 预测器不做后处理） | 不改变结构 | IPB 对所有模型不做输出平滑；如需对照，可在 official 评测中单独开启 |
| 20 m 分段重锚定评测不复刻 | 不改变结构 | IPB 指标不做任何对齐（DESIGN §5/§6） |
| 训练步长：official 5 样本；unified 10 样本 | 不改变结构 | unified 统一预算 |
| 优化器/调度/增强：official 照第 5 节；unified 默认照搬 official 增强，如有删减须在模型配置中注明 | 不改变结构 | 忠实性底线 |
| `max_v_norm`：行人数据保持 5 m/s 门限（只影响训练样本筛选） | 不改变结构 | 与官方一致；IPB 验证/测试不做门限 |

## 7. 官方报告数值

论文（CVPR 2025）——所有方法在同一大数据集上训练、在未见数据上测试；表中未写单位与指标公式（ATE、T-RTE 数值按常规推断为米）：

| 数据集（平台） | 指标 | TartanIMU | TLIO | RNIN-VIO | AI-IMU | IMO | 来源 |
|---|---|---|---|---|---|---|---|
| IDOL [35]（Handheld/Human） | ATE / T-RTE | **4.32 / 1.95** | 6.96 / 4.82 | 7.62 / 5.61 | 8.26 / 4.89 | 10.19 / 5.67 | Table 1 |
| SubT-MRS（Wheeled/Car） | ATE / T-RTE | 6.17 / 2.52 | 8.12 / 3.73 | 7.82 / 5.06 | 7.68 / 3.33 | 8.12 / 3.73 | Table 1 |
| SubT-MRS（Legged/Dog） | ATE / T-RTE | 1.46 / 0.79 | 3.61 / 1.73 | 3.10 / 1.58 | 3.23 / 1.60 | 3.35 / 1.64 | Table 1 |
| [2]（Aerial/Drone） | ATE / T-RTE | 3.32 / 1.04 | 3.93 / 1.40 | 4.32 / 1.51 | 4.14 / 1.45 | 3.72 / 1.34 | Table 1 |
| Human：单平台训练 vs 全平台训练 | ATE / T-RTE | 6.64 / 4.24 → 4.32 / 1.95 | | | | | Table 2 |
| 四平台平均：单平台 vs 全平台 | ATE / T-RTE | 5.41 / 2.68 → 3.82 / 1.57 | | | | | Table 2 |

论文结论：相对次优模型 ATE 平均提升 35.5%、T-RTE 41.0%（Table 1 说明）；摘要称 “36% ATE 提升”、LoRA 仅 1.1 M 可训练参数、在线自适应 200 FPS。

发布权重（HF 模型卡，**与论文不可比**，数据与代码版本不同）：

| 评测 | human 数值 | 说明 |
|---|---|---|
| `test.py`，`tartan_layout` 测试集，20 m 分段 ATE | expert 1.823；unified 1.773（m） | 模型卡 Results §2 |
| 挑战赛 v4 测试（all split） | AVE 0.101 m/s；ATE20 1.023 m（unified） | 模型卡 Results §1 |
| 离线窗口 RMSE（all split） | 0.164 m/s（unified） | 模型卡 Results §3 |

## 8. 忠实性测试建议

- [ ] 参数量 == 夹具 `total_params`（5 698 346），`trainable_params` 相同；另断言“主干 + human 头”4 804 367、未用 FcBlock 199 174（夹具 `benchmark_config`），实际参与 human 前向的参数 4 605 193
- [ ] `param_shapes` 159 项逐项一致（注意头内 `velocity_scale` 排在最前）
- [ ] 形状：`(B,10,6,40)` → 每头 `(B,10,3)`；`predict_cov=True` 时额外 `(B,10,3)`；IPB 包装 `(B,6,2000)` → `vel (B,3)`（方案 A）或 `(B,6,200)` → `vel (B,3)`（方案 B）
- [ ] 抽取：包装器对 `x[..., ::5]` 与手工抽取逐元素一致；抽取后长度 ∉[33,48] 时报错
- [ ] 因果性：修改第 10 个子窗之后的输入不影响输出；修改第 1 个子窗会影响 `vel`（方案 A，LSTM 单向）
- [ ] 多头：`split_z` 合成——把 `output_block1_z` 最后一层置零且 bias=0 时 z 分量恒为 0；`velocity_scale=(2,1,1)` 时 x 分量翻倍、`logstd` 不变
- [ ] 损失：`use_local_coord=True` 下 `loss(epoch=1) == loss(epoch=300)`（与 epoch 无关），且对协方差头参数的梯度恒为 0（复现官方行为的回归测试）
- [ ] 多头损失归一化：两个头各占一半样本时，总损失 = 两头各自 masked-mean 的平均
- [ ] 增强：偏航 + 倾斜增强后 `‖gyro‖`、`‖acc − bias‖`、`‖targ‖` 保持（白噪声关闭时）；增强在抽取之前
- [ ] LSTM 初始化：`bias_hh_l{0,1}[128:256] == 1`、其余 bias 为 0；`weight_hh` 近似正交
- [ ] （可选、仅本地）加载 HF `expert_human.pt`，对同一输入与官方实现逐元素一致（eval，atol 1e-5）

## 9. 预训练权重

Hugging Face `Tartan-IMU/TartanIMU`（<https://huggingface.co/Tartan-IMU/TartanIMU>，本卡核对时 repo sha `556a9e89c85d35ad83b7d05c852194165a6601cd`）：`checkpoints/unified.pt`（63.8 MB，4 头联合，drone_v3 + wdfix2）、`checkpoints/expert_{car,dog,drone,human}.pt`（各约 59 MB，单平台专家）、`config/unified.yaml`、`config/resnet_lstm_multihead.yaml`、`inference_example.py`。checkpoint 为 dict：`model_state_dict / optimizer_state_dict / scaler_state_dict / scheduler_state_dict / trainer_state / epoch`（`expert_human.pt` 的 `epoch=18`，`unified.pt` 的 `epoch=6`）。两者均可 `strict=True` 加载到夹具结构（human 头 `velocity_scale` 学到约 `[1.11, 1.11, 1.07]`）。许可矛盾见第 1 节，不得入库或再分发。挑战赛数据集：<https://huggingface.co/datasets/Tartan-IMU/IROS-Tartan-IMU-Challenge>。

## 10. 官方实现的坑与未决问题

1. **协方差从未训练**：body 模式下损失恒为 `20·L1`（`losses.py:179-189`），`single_head_velocity_loss` 的 `propcov=False`（`losses.py:278-284`），`start_cov_epochs` 只改变是否前向计算协方差；HF 配置 `epochs=150 < start_cov_epochs=200`。发布权重的 `output_block2` 是随机初始化后只受权重衰减影响的值，**不要把 `logstd` 当作有效不确定度**。
2. **论文与代码不一致**：论文去重力（式 1）、MSE+NLL（式 2–3）、“两层全连接解码器”；代码保留重力、20·L1、三层 FcBlock（`blocks.py:133-164`）。LoRA 可训练参数论文内自相矛盾（1.1 M vs 5 M）。
3. **在线自适应冻结无效**：`freeze_backbone_parameters` 按参数名包含 `backbone/trunk/encoder` 冻结（`train.py:62-76`），而 Foundation 参数名是 `model.*`/`heads.*`，结果什么也不冻结；`lora` 包缺失时退化为全参数微调。
4. **配置键不被读取**：HF `unified.yaml` 的 `data.max_v_norm`、`data.clip_max_speed` 在此提交中无效（代码写死 5 m/s、15 m/s，`dataset_AirLab.py:105-106, 231-233`）；模型卡自己也把它列为“复现陷阱”。`train.active_heads` 放在 `train` 下时 `get_active_heads` 读不到（它读顶层 `cfg["active_heads"]`，`function.py:202-208`），回退为按 batch 中出现的平台计算。
5. **wdfix2 不在代码中**：模型卡称 `unified.pt` 对头部免除权重衰减，但 `build_trainer` 对全部参数用同一 `weight_decay`（`configer.py:206-217`）。用此提交重训无法复现 `unified.pt`。
6. **抽取无抗混叠**：`[::5]` 直接丢样本（`dataset_AirLab.py:580-581`），200 Hz 中 20–100 Hz 分量会混叠进 40 Hz 输入；复现时必须照做，不能“改进”为滤波抽取。
7. **测试平滑跨样本**：`smooth_velocity_predictions` 在测试时对 batch 维（即相邻序列）做 3 点平均（`function.py:293-295`、`losses.py:316-357`），结果依赖 test batch 大小与切分。
8. **评测分段重锚定**：`test.py` 以 20 m 为段、每段起点重新对齐真值再算 ATE（`test.py:274-290, 368`），数值远小于不对齐的全程 ATE，不可与 IPB ATE 直接比较；另有 `compute_accruacy_metrics` 中 `ATE = mean‖·‖`（非 RMSE，`metrics.py:889-891`）。
9. **目标是逐帧机体速度的均值**，与 IPB/TLIO 类“位移除以时长”不同；手持设备在 1 s 内转动时两者差异不可忽略。
10. **四个头注册顺序（dog, human, car, drone）与 `FoundationModel.types`/`motion_types` 映射（car=1, dog=2, drone=3, human=4）不同**；加载权重按名字匹配无碍，但按索引导出时易错。挑战赛 npz 的 `platform_id` 是 0 起（car=0 … human=3），喂给模型要 +1（HF 卡说明）。
11. 默认数据集配置 `tartan_imu_dataset.yaml` 的 `schemes.train: False`（只测试不训练），直接运行不会训练。
12. 论文中 human 数据来自 IDOL（手持），IPB 已有 `idol` 转换器，可作为与论文对照的首选数据集；但论文模型版本（去重力、MSE+NLL）与公开代码不同，数值只能作量级参考。
