# PedestrianDiffusion（`pedestrian_diffusion`）

> 规格卡版本：v1（2026-09-17）· fidelity：`official-code` · 夹具：`tests/fixtures/algorithms/pedestrian_diffusion.json`
> 优先级 P1：谱域条件去噪框架，需要文本条件（CLIP）与预训练 VAE，适配成本高（见第 6 节）。

> **许可警告（AGPL-3.0）**：官方仓库整体采用 GNU AGPL v3，属于强 copyleft 许可，而且 `utils/` 下还附带了 RoNIN（GPL-3.0）、TLIO（BSD）、LLIO（GPL-3.0）的代码副本。IPB 的实现只能依据本卡片与论文**独立编写**，不得复制或改写官方源码；以网络服务形式提供 AGPL 衍生代码时，必须公开源码。

## 1. 元信息

| 项 | 值 |
|---|---|
| 论文 | *PedestrianDiffusion: Multimodal Generative Denoising and Dense State Estimation for Inertial Navigation*；I-Hao Lu, Dongsoo Han；[arXiv 2607.03349](https://arxiv.org/abs/2607.03349) v1（2026-07-03），截至本卡片日期未标注正式出处 |
| 官方仓库 | https://github.com/jacklu333333/PedestrianDiffusion @ `0b6cc2d019397e390ce757335e01712304339c9b`（本地：`/workspace/webCodex/third_party/PedestrianDiffusion`） |
| 许可 | AGPL-3.0（见上方警告） |
| 框架 | PyTorch Lightning + diffusers（`envsetup.sh` 注释中的固定版本为 `diffusers==0.36.0`）+ transformers（CLIP 文本编码器）+ DeepSpeed/DDP |
| fidelity | official-code（训练、测试、VAE 预训练脚本均开源；权重在 Google Drive） |
| 参考文件 | 主模块 `utils/mmodules/PedestrianDiffusion.py`；基类 `utils/mmodules/baseDiffusionModule.py`；VAE `utils/mmodels/VAE3D.py`（`VAE3D_1024`），VAE 预训练 `utils/mmodules/VAESpectrum.py`、`trainScriptOdom_VAE_imu.py`；损失 `utils/mloss/spectrumMultiTaskLoss.py`、`utils/mloss/simclr_loss.py`、`utils/mloss/CosSimMetric.py`、`utils/mloss/NaiveDistanceError.py`；STFT 与归一化 `utils/transform.py`（`Time2Frequency`、`batchFrequencyToTime`、`batchNormalizeSensor`、`bathNormalizeRelativePosNOri`、`rotationNoise`）；数据 `utils/mdatasets/odomDataModule.py`、`mDataset.py`、`RoNINDataset.py`、`utils.py`（CLIP 编码）；训练 `trainScript.py`、配置 `config.json`、参数解析 `utils/parser.py`；测试指标 `utils/mcallbacks/TrajectoryTestResultHandler.py`、`recompute_metrics.py` |

## 2. 任务（输入）

| 项 | 官方 | IPB 映射 |
|---|---|---|
| 采样率 | 100 Hz（`sampling_rate=100`）；RoNIN 原始 200 Hz 数据**直接隔点抽取**（`acc[i::2]`，无低通；训练时 `i=0,1` 两个偏移各生成一份数据，测试只用 `i=0`）（`utils/mdatasets/RoNINDataset.py:347-375`） | 200 Hz；模型内部 `x[..., ::2]` 抽取到 100 Hz |
| 窗口 | `window_size=100`（1 s） | `window=200`（1 s） |
| 训练步长 | `stride=33`（@100 Hz） | official：`stride=66`；unified：`stride=10` |
| 测试步长 | `stride = window_size`（不重叠，`mDataset.py:232-237`） | `eval_stride=10`（IPB 协议）；另可提供不重叠的官方式稠密拼接变体 |
| 坐标系 | 世界系：用姿态四元数把机体系加计、陀螺旋到世界系（`RoNINDataset.py:316-334`） | `frame=gravity_world` |
| 姿态来源 | RoNIN：测试只用 game rotation vector（`grv_only=True`），训练时 GRV 对齐误差 > 20° 则改用对齐误差最小的来源；首帧用 Tango 姿态对齐（RoNIN 官方做法） | `orientation=device`（首帧偏航补偿，与 DESIGN §3 一致） |
| 去重力 | 否（`GravityRemoval=false`，使用含重力的 `acce`） | `remove_gravity=false` |
| 通道顺序 | `[acc_xyz, gyr_xyz]`（**加计在前**，`mDataset.py:1385`） | 从 `[gyro, acc]` 置换为 `[acc, gyro]` |
| 谱变换 | 在 Dataset 中对每个 100 样本窗口逐通道做 STFT：`n_fft=30`、`win_length=30`、Hann 窗、`hop_length=14`、`center=True`、`pad_mode="constant"`，取实部/虚部 → `(6, 16, 8, 2)`（`utils/transform.py:1051-1092`） | 模型内部完成，结果与官方相同 |
| 输入归一化 | 谱域常数缩放：加计除以 `5.78218412399292·5 = 28.9109`，陀螺除以 `2.383342266082764·5 = 11.9167`（均值 0）（`PedestrianDiffusion.py:14-20`，`transform.py:1303-1340`） | 同官方 |
| 额外输入 | CLIP 文本条件：由元数据拼成句子，经冻结的 `openai/clip-vit-base-patch32` 文本编码器得到 `(77, 512)` 的 `last_hidden_state` 与 `attention_mask`（`utils/mdatasets/utils.py:52-121,348-410`） | **需要接口扩展**：`aux["text_tokens"] (B,77,512)`、`aux["text_mask"] (B,77)`，或由模型内部根据序列元数据生成（见第 6 节） |
| 标签 | 逐样本 6 维：`[v_world (3), ω (3)]`（`mDataset.py:1378-1386`）；`v` 为整条序列位置的后向差分 × fs（首样本为 0，`mDataset.py:631-632`）；`ω` 由姿态四元数中心差分计算 `2·(q̇ ⊗ q*)` 的向量部（`utils.py:300-345`），其中 `q` 是**取逆后**的原始姿态（`RoNINDataset.py:375`） | 视图需要提供逐样本目标（接口扩展 `batch["vel_seq_world"]`、`batch["angvel_seq"]`），模型内部同样抽取到 100 Hz |
| 标签归一化 | 速度除以 `13.15231227874756·5 = 65.7616`，角速度除以 `2.802832841873169·5 = 14.0142` | 同官方 |

文本模板（注意训练与测试模板**不同**）：

- 训练（`mode=="train"`）：对属性 `person, action, device, mounted, environment, annotation` 的每个非 `unknown` 取值在 `{unknown, 取值}` 中做笛卡尔积，逐一生成
  `"This collected by {person} person doing {action} with {device} device mounted on {mounted} in {environment} environment with {annotation} annotation."`，
  最后再追加一次全部为真实取值的句子；取样时最后一句概率 0.5，其余均分 0.5（`mDataset.py:361-366`）。
- 验证/测试：只有一句，模板把 `person` 换成 **`personnel`**，属性为真实取值。
- 送入分词器前把 `_`、`-`、`/` 替换为空格；超过 77 个 token 直接报错；padding 到 77。
- RoNIN 的属性为 `person=unknown, action=walking, device=<info.json 中的设备名>, mounted=handheld（写死）, environment=unknown, annotation=ARCore`，`dataset` 字段不进入句子。

## 3. 输出

- 去噪网络输出谱域样本 `ŷ_spec`：`(B, 6, 16, 8, 2)`（`prediction_type="sample"`，即直接预测 x₀）。
- 后处理：反归一化（速度 ×65.7616，角速度 ×14.0142）→ ISTFT（同一组 STFT 参数，`length=100`）→ `(B, 6, 100)`：100 Hz 的世界系速度（m/s）与角速度（rad/s）。
- 无协方差输出。
- 官方轨迹（`utils/mcallbacks/TrajectoryTestResultHandler.py:568-640`）：测试时窗口不重叠，把每个窗口的 100 个稠密速度按时间拼接，`p = cumsum(v)/fs`（从 0 开始）；角速度同样积分得到姿态。默认输出 3D 结果，`recompute_metrics.py --2d` 重新计算 2D 指标。
- 指标代码：`ATE = sqrt(mean((est − gt)²))` 对**所有元素**取均值（3D 时等于按范数定义的 RMSE 除以 √3，2D 时除以 √2）；RTE 以 60 s（6000 样本）为间隔，同样按元素均值（`TrajectoryTestResultHandler.py:25-70,645-740`）。

## 4. 网络结构

参数所在的子模块按注册顺序为：`special_loss.log_vars`（损失中的可学习权重）→ `VAE`（只保留编码器，冻结）→ `model`（diffusers `UNet3DConditionModel`）。STFT、归一化、指标模块均无参数。

**(a) 条件编码**

| # | 层 | 参数 | 归一化 | 激活 | 输出形状 |
|---|---|---|---|---|---|
| c1 | `VAE.enc_layer` Conv3d | 6→1024，k=(16,8,2)，s=(16,8,2)，p=0，有偏置 | GroupNorm(32, 1024) | SiLU | (B,1024,1,1,1) |
| c2 | 展平 | – | – | – | (B,1024) |
| c3 | `VAE.fc_mu` / `VAE.fc_logvar` Linear | 1024→512（两个独立层） | – | – | (B,512) ×2 |
| c4 | 重参数化 `z = μ + ε·exp(0.5·logvar)`（**推理时也采样**）→ `(B,512,1)` → 转置 | – | – | – | (B,1,512) |
| c5 | 与 CLIP 文本 token 拼接；注意力掩码前面补一个 1 | – | – | – | (B,78,512) / (B,78) |

VAE 的卷积核与窗口完全绑定：输入必须正好是 16 个频点 × 8 帧 × 2，窗口长度一变就无法构建。

**(b) 去噪网络输入**：`cat([ŷ_noisy_spec, x_spec], dim=1)` → `(B,12,16,8,2)` → `rearrange("b c (f1 f2) t I -> b (c I) t f1 f2", f1=4, f2=4)` → `(B, 24, 8, 4, 4)`，即 24 通道、8 帧、4×4 “图像”（16 个频点排成 4×4）。输出再用逆 `rearrange` 变回 `(B,6,16,8,2)`。

**(c) `UNet3DConditionModel`**（`sample_size=(4,4)`，`in_channels=24`，`out_channels=12`，`block_out_channels=[128,256]`，`layers_per_block=1`，`cross_attention_dim=512`，`attention_head_dim=64`，其余为 diffusers 0.36.0 默认：`norm_num_groups=32`、`norm_eps=1e-5`、`act_fn=silu`、`downsample_padding=1`）。下表形状写成帧展开后的 `(B·8, C, H, W)`：

| # | 模块 | 结构要点 | 参数量 | 输出形状 |
|---|---|---|---|---|
| u1 | `time_proj` + `time_embedding` | 正弦时间编码 128 维 → Linear 128→512 → SiLU → Linear 512→512 | 328,704 | (B,512) |
| u2 | `conv_in` Conv2d | 24→128，k3 p1 | 27,776 | (B·8,128,4,4) |
| u3 | `transformer_in`（时间维 Transformer） | GroupNorm(32) → proj_in 128→512 → 1 个 BasicTransformerBlock（8 头 × 64，自注意力沿帧维）→ proj_out 512→128，残差 | 5,383,552 | (B·8,128,4,4) |
| u4 | `down_blocks.0` CrossAttnDownBlock3D | ResnetBlock2D(128，时间嵌入投影 512→128，GroupNorm+SiLU+Conv3×3，dropout 0) → TemporalConvLayer(4 个 GroupNorm+SiLU+Conv3d(k=(3,1,1))，后 3 个有 Dropout 0.1) → Transformer2DModel(2 头 × 64，自注意力 + 对 78 个条件 token 的交叉注意力，GEGLU FF 128→1024→128) → TransformerTemporalModel(2 头) → Downsample2D(Conv3×3 s2) | 1,531,648 | (B·8,128,2,2) |
| u5 | `down_blocks.1` CrossAttnDownBlock3D | 同上，通道 128→256，4 头，**无**下采样 | 4,865,280 | (B·8,256,2,2) |
| u6 | `mid_block` UNetMidBlock3DCrossAttn | Resnet → TempConv → Transformer2D(4 头) → TransformerTemporal → Resnet → TempConv | 7,229,440 | (B·8,256,2,2) |
| u7 | `up_blocks.0` CrossAttnUpBlock3D | 2 组 [Resnet(拼接跳连) → TempConv → Transformer2D → TransformerTemporal]，末尾 Upsample2D(最近邻 ×2 + Conv3×3) | 11,960,320 | (B·8,256,4,4) |
| u8 | `up_blocks.1` CrossAttnUpBlock3D | 同上，输出 128 通道，无上采样 | 3,293,440 | (B·8,128,4,4) |
| u9 | `conv_norm_out` + SiLU + `conv_out` | GroupNorm(32,128)；Conv2d 128→12，k3 p1 | 256 + 13,836 | (B·8,12,4,4) → (B,12,8,4,4) |

- 参数量：**37,259,798**（与论文 Table IV 的 37.26M 一致），其中去噪 UNet 34,634,252（论文 34.63M）、冻结的 VAE 编码器 2,625,536（论文 2.63M）、损失权重 `log_vars` 10。可训练参数 34,634,262。
- VAE 预训练阶段的完整 VAE（含解码器 `fc_dec`、`gn_dec`、`dec_layer`）：4,725,766。
- CLIP 文本编码器（约 63M，冻结、离线缓存编码结果）**不计入**上述参数量。
- 初始化：diffusers 默认初始化；VAE 权重从 `vae_weight_path`（默认 `./weight/VAESpectrum_weight.ckpt`）加载，去掉 `VAE.` 前缀（`utils/mcallbacks/copyFileNLoadWeightLogger.py:102-141`）。
- 调度器（替换基类中的 DDPM）：`DPMSolverMultistepScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", beta_start=1e-7, beta_end=2e-2, solver_order=2, prediction_type="sample", lower_order_final=True, rescale_betas_zero_snr=True)`（`PedestrianDiffusion.py:104-114`）。

## 5. 损失与训练配方（official）

**扩散过程的实际行为（用 diffusers 0.36.0 数值核实）**

- 训练：`timesteps = randint(0, target_time_steps=1)` → **恒为 0**；`noisy = scheduler.add_noise(y, ε, t=0)`。由于 DPMSolver 在 `set_timesteps` 之前的 `sigmas` 按时间升序、`timesteps` 按降序存放，`t=0` 实际取到的是 t=999 的 σ=4096，得到 `noisy = 2.44e-4·y + 1.0·ε`，即几乎纯噪声（没有标签泄漏）。网络的时间嵌入输入为 0。
- 推理：`set_timesteps(1)` → 时间步 `[999]`；输入 `ε ~ N(0, I)`；`scale_model_input` 为恒等；单步 DPM-Solver++（σ 终点为 0）的输出**严格等于网络输出**。所以推理就是“噪声 + 条件 → x₀ 的一次前向”，但时间嵌入输入是 **999**，训练中从未出现过。
- 因此该模型实际上是**以噪声为额外输入的单步条件回归器**，训练与测试的时间嵌入不一致（见第 10 节）。

**损失** `spectrumMultiTaskLoss(dt=1/100)`（`utils/mloss/spectrumMultiTaskLoss.py`），对 m ∈ {速度（通道 0–2），角速度（通道 3–5）}，`s = 1e-3`，`H_δ` 为 `F.huber_loss` 的均值：

| 项 | 定义 |
|---|---|
| ℓ₁ 频域重建 | `H_δ=s(ŷ_f/s, y_f/s)`（归一化谱） |
| ℓ₂ 积分一致性 | `H_δ=s(Σ_t ŷ_t·dt / s, Σ_t y_t·dt / s)`（时域，物理单位） |
| ℓ₃ 时域重建 | `H_δ=s(10·ŷ_t/s, 10·y_t/s)` |
| ℓ₄ 时域相似度 | `H_δ=s(10·(1 − cos_t)/s, 0)`，`cos_t` 为每个通道沿时间的余弦相似度，按 batch、通道求均值 |
| ℓ₅ 频域相似度 | 同 ℓ₄，作用在展平后的谱上 |
| 合成 | `L = Σ_{m,k} exp(−s_{m,k})·ℓ_{m,k} + s_{m,k}`，`s` 为 2×5 可学习参数，初值 0 |

其中 `y_t`、`ŷ_t` 为反归一化再 ISTFT 后的时域信号。

| 项 | 值 | 来源 |
|---|---|---|
| 优化器 | `AdamW(self.parameters(), lr=1e-4, fused=True)`（其余默认，weight_decay=0.01；冻结的 VAE 无梯度不更新） | `PedestrianDiffusion.py:940-955` |
| 学习率调度（阶段 1，`train_phase=initial`） | `get_cosine_schedule_with_warmup`，按 step 更新；`num_warmup_steps = 每轮 batch 数 × warm_up(0.05)`，即只预热 0.05 轮；总步数 = `max_epochs × 每轮 batch 数`（未设置 max_epochs 时为 Lightning 默认 1000 轮） | `PedestrianDiffusion.py:999-1019` |
| 阶段 2（`dual_stage=true`） | 阶段 1 结束后先在 test 上跑一次，再从最优 checkpoint 重建模型，`ReduceLROnPlateau(factor=0.5, patience=3, threshold=0.01, min_lr=1e-8)`，监控 `metric_naive_distance_error_XY/val_mean` | `trainScript.py:333-357`；`PedestrianDiffusion.py:1022-1045` |
| batch | 256（总量，按 GPU 数平分） | `config.json`；`trainScript.py:74-80` |
| 训练时长 | `max_time` 5 h（数据集名含 “RoNIN”）或 12 h；论文写的是两张 RTX 3090 共 10 h | `trainScript.py:91-93` |
| 梯度裁剪 | 1.0（L2 范数） | `config.json` |
| 精度 | `32-true`；DDP（`find_unused_parameters=True`）+ `sync_batchnorm` | `config.json`；`trainScript.py:123,264` |
| 增强 | `pre_augmentation`：概率 1，绕 z 轴随机偏航 θ~U[0, 2π)，同时旋转输入（加计、陀螺）与标签（速度、角速度）；`augmentation.*` 概率均为 0 | `odomDataModule.py:42-56`；`transform.py:141-200` |
| EMA | 关闭（`ema.enable=false`） | `config.json` |
| 早停 / 选模 | `EarlyStopping(monitor="metric_naive_distance_error_XY/val_mean", patience=10, min_delta=1e-4)`；保存该指标最优的 3 个 checkpoint；指标为 `‖Σ_t v̂_xy − Σ_t v_xy‖ / fs / (L/fs) × 1`（每窗口位移误差折算到每秒） | `trainScript.py:148-230`；`utils/mloss/NaiveDistanceError.py` |
| 随机种子 | 42 | `trainScript.py:40-41` |
| 数据 | `config.json` 默认 `RoNINs`（RoNIN seen + unseen 列表）；论文主结果为 `hybrid`（RIDI、RoNIN、OxIOD VICON、OxIOD Tango、TLIO 联合训练） | `odomDataModule.py:218-246,385-466` |
| VAE 预训练 | `VAESpectrum`：完整 VAE3D_1024 重建 IMU 谱，`ControlVAELoss(target_kl=0.1)`，AdamW + 余弦预热 + EMA；README 命令 `trainScriptOdom_VAE_imu.py --encoding False -b 256` | `utils/mmodules/VAESpectrum.py`；`trainScriptOdom_VAE_imu.py` |

## 6. benchmark 适配（unified 配方）

| 改动 | 类型 | 理由 |
|---|---|---|
| 视图 `frame=gravity_world`、`orientation=device`、`remove_gravity=false`、`window=200`；模型内部置换为 `[acc, gyr]` 并 `x[..., ::2]` 抽取到 100 样本 | 不改变结构 | 官方网络与 VAE 卷积核硬性要求 100 样本 / 8 帧；官方 RoNIN 加载器同样是无滤波的隔点抽取 |
| 模型内部 STFT（n_fft 30、hop 14、Hann、center、常数填充）与常数归一化；输出反归一化 + ISTFT(length=100) | 不改变结构 | 与官方数据管线逐项一致 |
| 目标：逐样本世界系速度与角速度（接口扩展 `vel_seq_world`、`angvel_seq`，200 Hz，由模型内部抽取）；速度按官方定义用位置后向差分 × fs 在整条序列上计算后再切窗 | 不改变结构（需视图扩展） | 官方是稠密 6D 监督，角速度通道参与损失 |
| 窗口级输出：`vel = mean_t(v̂_t[:2])`（100 个稠密速度的均值 = 窗口位移 / 1 s）；`target=avg_velocity`、`dims=2`；网络仍输出 6 通道 | 不改变结构 | IPB 每窗口一个速度；`dims=2` 只做切片，不改变 `out_channels=12` |
| 文本条件：模型根据序列元数据生成句子并调用冻结的 CLIP 文本编码器（结果缓存）；属性映射 `device ← device_id`、`mounted ← placement`、`annotation ← position_source`、`action=walking`、其余 `unknown`；训练/测试模板与采样概率照搬官方 | 不改变结构（需元数据接口） | 论文 Table V 显示“已知属性”明显优于“未知属性”；另报告全部为 `unknown` 的变体，便于跨数据集公平比较 |
| 确定性推理：每个窗口的初始噪声 ε 与 VAE 采样噪声都用 `torch.Generator(seed=global_seed·1_000_003 + window_index)` 生成，单样本（K=1），与官方一次采样一致；可选 K=8 均值作为消融 | 协议补充 | 官方推理有两处随机性（ε 与 VAE 重参数化），IPB 要求结果可复现 |
| 推理时间嵌入：official 配方保持官方行为（t=999）；另报告 t=0 变体（与训练一致） | 不改变结构 | 第 10 节第 1 条的训练/测试不一致 |
| VAE 权重：official 配方加载官方 `VAESpectrum_weight.ckpt`；unified 配方在 IPB 训练集上按 `VAESpectrum` 配方重新预训练 | 不改变结构 | 官方 VAE 在 hybrid 数据（含 RoNIN/RIDI/OxIOD/TLIO）上训练，其划分与 IPB 的 test 是否重叠需要核查，否则存在泄漏风险 |
| 选模：IPB val 的 ATE（DESIGN §6），不再用 `metric_naive_distance_error_XY/val_mean` | 不改变结构 | 统一 fitness |
| 训练预算：unified 用 IPB 统一 epoch 预算，余弦总步数按实际 epoch 计算（不沿用默认 1000 轮） | 不改变结构 | 官方调度长度与实际训练长度不符 |

## 7. 官方报告数值

来源：arXiv v1 HTML（经网页摘要工具读取，建议人工再核对一次原文表格）。PD = PedestrianDiffusion。注意第 3 节所说的 ATE/RTE 实现方式（按元素均值），与 IPB 的定义相差 √3（3D）或 √2（2D）倍。

| 数据集 | 指标 | 数值 | 来源 |
|---|---|---|---|
| RoNIN seen（3D） | ATE / RTE (m) | 3.17±1.32 / 2.24±1.23 | Table I |
| RoNIN unseen（3D） | ATE / RTE (m) | 4.82±3.24 / 3.71±1.76 | Table I |
| RoNIN seen（2D，与 SOTA 对比） | 位置 ATE / RTE (m) | 3.15 / 2.23 | Table II |
| RoNIN unseen（2D） | 位置 ATE / RTE (m) | 4.79 / 3.70 | Table II |
| hybrid 多数据集（位置，m） | ATE / RTE：OxIOD(V) 0.56/0.30，OxIOD(T) 1.60/1.34，RIDI 1.38/1.39，RoNIN 3.02/2.19，RoNIN(U) 4.79/3.64，TLIO 1.21/0.98 | – | Table III |
| 计算量 | 参数 37.26M，FLOPs 23.37G，每窗口 100 帧，GPU 107.71 ms（1.08 ms/帧），CPU 987.79 ms | – | Table IV |
| 文本条件消融（RIDI） | 位置 ATE：未知属性 1.70±1.10，已知属性 1.38±0.79 | – | Table V |

Table II 中对比方法：CTIN 4.62/2.81（seen）、EqNIO(O(2)) 3.45/2.78（作者用官方代码复现）；Table I 中作者复现的 RoNIN-ResNet 为 4.14/2.38。这些数值使用的是作者自己的评测代码。

## 8. 忠实性测试建议

- [ ] 总参数量 == 37,259,798；可训练 == 34,634,262；冻结 == 2,625,536；`param_shapes` 逐项一致（631 个张量，顺序为 log_vars → VAE → UNet）
- [ ] VAE 编码器对 `(B,6,16,8,2)` 输出 `(B,512,1)`；输入帧数不是 8 时应报错
- [ ] STFT：100 样本 → `(6,16,8,2)`；ISTFT(length=100) 重建误差 < 1e-5（实测 6e-7）
- [ ] 去噪网络输出形状与输入的 y 部分相同 `(B,6,16,8,2)`
- [ ] 调度器：`add_noise(y, ε, t=0)` 的信号系数 ≈ 2.44e-4、噪声系数 ≈ 1.0；`set_timesteps(1)` 得到 `[999]`；单步 `step` 的输出等于网络输出
- [ ] 确定性：固定种子时两次推理结果完全相同；更换 VAE 采样种子时结果改变（证明重参数化被保留）
- [ ] 偏航等变性（近似）：对输入 `[acc, gyr]` 与目标同时施加 `R_z(θ)`，训练数据分布不变（增强覆盖全角度）；网络本身不保证等变，测试只检查增强实现把 6 个通道按 3 维一组正确旋转
- [ ] 损失：`log_vars` 为 0 时 `L = Σ ℓ`；`ℓ₂` 使用 `dt = 0.01`；速度通道与角速度通道分别计算
- [ ] 文本编码：训练模板用 `person`，测试模板用 `personnel`；`_ - /` 替换为空格

## 9. 预训练权重

README 提供权重压缩包：`gdown 1vQP0zt6s2oYgjrdWdDAqWg4m5zoYhjUn`（[Google Drive](https://drive.google.com/file/d/1vQP0zt6s2oYgjrdWdDAqWg4m5zoYhjUn/view?usp=sharing)），`.gitignore` 中的 `weight/` 目录即为其解压位置（默认 VAE 路径 `./weight/VAESpectrum_weight.ckpt`）。本卡片编写时**未下载核对**压缩包内容；另需自动下载 `openai/clip-vit-base-patch32`（HuggingFace）。

## 10. 官方实现的坑与未决问题

1. **训练/测试时间步不一致**：训练时时间步恒为 0（`target_time_steps=1`），测试时 `set_timesteps(1)` 给出 999，网络在测试时看到的时间嵌入从未训练过（`PedestrianDiffusion.py:640-646,815-820`）。
2. **DPMSolver 的 `add_noise` 索引问题**：`set_timesteps` 之前 `sigmas` 升序、`timesteps` 降序，`t=0` 取到的是 t=999 的噪声水平，所以训练输入几乎是纯噪声。这恰好让训练输入与测试输入的分布一致；如果移植时“修正”成真正的 t=0 加噪，会把标签几乎原样泄漏给网络。
3. **推理带随机性**：`VAE3D_1024.encode` 在 eval 模式下仍然做重参数化采样（`utils/mmodels/VAE3D.py:89-101`），初始噪声同样随机，官方没有固定种子。
4. **文本模板不一致**：训练用 “person”，测试用 “personnel”（`utils/mdatasets/utils.py:372,391`）；训练句子列表把全属性句子多追加了一次；RoNIN 的 `mounted` 写死为 `handheld`。
5. **疑似四元数顺序混用**：同一个 `ori` 数组在旋转 IMU 时按 `quaternion.from_float_array`（w,x,y,z）解释，在生成角速度标签时又按 `scipy Rotation.from_quat`（x,y,z,w）解释并取逆（`RoNINDataset.py:316-334,375`；`mDataset.py:1378-1383`）。按 RoNIN 官方约定该数组是 w,x,y,z，角速度标签可能是用错位的四元数算出来的，需要用真实数据核实。速度标签不受影响。
6. **学习率调度长度不对**：`warm_up=0.05` 表示 0.05 轮；余弦总步数按 `max_epochs`（未设置时为默认 1000 轮）计算，而训练实际被 `max_time`（5 h / 12 h）或早停截断，学习率基本停留在峰值附近。
7. **指标定义**：ATE/RTE 按元素求均值，比按范数定义的 RMSE 小 √3（3D）或 √2（2D）倍；早停监控的是窗口位移误差而不是轨迹误差。
8. `dual_stage=true` 时，阶段 1 结束会先在 test 集上跑一次，再继续阶段 2 训练；阶段 2 仍以 val 指标选模，但 test 被运行了不止一次（违反 IPB 诚实协议，移植时去掉）。
9. RoNIN 的 200→100 Hz 降采样没有抗混叠滤波；训练时两个偏移各生成一份数据，数据量翻倍。
10. 仓库依赖很重（deepspeed、cuml、mamba 等，`config.json` 中的 `mamba_*` 参数在该模型中未使用），`utils/mdatasets/utils.py` 在 import 时就会下载并加载 CLIP。
11. 许可混合：AGPL 仓库中包含 GPL-3.0（RoNIN、LLIO）与 BSD（TLIO）代码副本。
