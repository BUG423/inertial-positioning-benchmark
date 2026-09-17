# TinyOdom（`tinyodom`）

> 规格卡版本：v1（2026-09-17）· fidelity：`official-code` · 夹具：`tests/fixtures/algorithms/tinyodom.json`

## 1. 元信息

| 项 | 值 |
|---|---|
| 论文 | *TinyOdom: Hardware-Aware Efficient Neural Inertial Navigation*；Swapnil Sayan Saha, Sandeep Singh Sandha, Luis Antonio Garcia, Mani Srivastava；Proc. ACM IMWUT 6(2), Article 71，2022 年 7 月；DOI [10.1145/3534594](https://dl.acm.org/doi/10.1145/3534594)；开放全文 [PMC10957141](https://pmc.ncbi.nlm.nih.gov/articles/PMC10957141/)（未找到 arXiv 版本） |
| 官方仓库 | <https://github.com/nesl/tinyodom> @ `7958781baeddf1b99a4f365fb54fb7bac5e96845`（本地：`/workspace/webCodex/third_party/tinyodom`） |
| 许可 | BSD-3-Clause。依赖 `keras-tcn==3.3.0`（MIT） |
| 框架 | Keras / TensorFlow 2.5（`requirements.txt`）；TCN 骨干来自 `keras-tcn==3.3.0`；部署用 TFLite Micro（Mbed） |
| fidelity | official-code（Jupyter 笔记本 + keras-tcn）。NAS 选出的最终结构**未公开**，本卡锁定的是仓库中唯一的具体 RoNIN 结构，见 §4 |
| 参考文件 | `RoNIN/TinyOdom_RoNIN.ipynb`（cell 2 数据参数、cell 8 NAS 目标函数与模型、cell 9 搜索空间、cell 11 训练最终模型、cell 16 轨迹与指标）；`RoNIN/data_utils.py`（`import_ronin_dataset`、`random_rotate`、`Cal_TE`）；`RoNIN/hardware_utils.py`；`tinyodom_tcn/model.cc`（嵌入的 TFLite flatbuffer）；`tinyodom_tcn/main.cpp`；keras-tcn 3.3.0 `tcn/tcn.py`；其他领域：`OxIOD/`、`AQUALOC/`、`EuRoC MAV/`、`Gundog/`、`robust_depth_filter/` |

**领域范围**：仓库覆盖行人（OxIOD 100 Hz、窗口 200、步长 10；RoNIN 200 Hz、窗口 400、步长 20）、水下机器人（AQUALOC）、无人机（EuRoC，窗口 50、步长 5）和动物（GunDog，40 Hz、窗口 10）。它们的骨干与输出头完全相同，只是输入通道和窗口不同（论文表 2）。本卡针对 **RoNIN 行人设置**。OxIOD 设置与之相同，通道为 `[acc(=Lin_Acc+Grav), gyro, mag, step_mask]`（`OxIOD/data_utils.py:65-94`），但其中 acc 的 y、z 两个通道有复制 bug，见 §10。

## 2. 任务（输入）

| 项 | 官方（RoNIN 笔记本） | IPB 映射 |
|---|---|---|
| 采样率 | 200 Hz（cell 2 `sampling_rate = 200`） | 200 Hz |
| 窗口 | 400 个样本（2 s，cell 2） | `window=400` |
| 训练步长 | 20 个样本（0.1 s），用 giotto-tda `SlidingWindow` 生成 | `stride=20`（unified 可用 IPB 默认值 10，结构不变） |
| 推理步长 | 20 个样本（测试集同样步长 20 滑窗，cell 5/16） | `eval_stride=10`（IPB 统一值；官方为 20） |
| 坐标系 | **机体系原始读数**，网络不接收任何姿态（`data_utils.py:73-76` 直接读 `synced/{acce,gyro,magnet}`）；目标却在 Tango 世界系（见 §3） | `frame=body`（见 §6） |
| 姿态来源 | 无（网络不使用；轨迹重建也不使用） | 网络不使用。IPB 用 `orientation=reference` 把目标旋到窗口末端机体系，并在推理时把预测旋回世界系 |
| 去重力 | 否（`acce` 为含重力的比力） | `remove_gravity=false` |
| 通道顺序 | `(B, 400, 10)`，batch-first：`[acce_x, acce_y, acce_z, gyro_x, gyro_y, gyro_z, magnet_x, magnet_y, magnet_z, step_mask]`（`data_utils.py:24`、`:174`） | IPB `(B,6,T)` 为 `[gyro, acc]` → 置换为 `(B,T,6)`，顺序 `[acc_xyz, gyro_xyz]`（即通道下标 `[3,4,5,0,1,2]`） |
| 额外输入 | 磁力计 3 维（µT，RoNIN `synced/magnet`）；物理元数据通道 `step_mask`（步伐峰值位置为 1，其余为 0，见下） | 无磁力计 → 丢弃；`step_mask` 丢弃（见 §6） |
| 输入归一化 | 无 | 无 |

`step_mask` 的官方生成方式（`useStepCounter=True`，`data_utils.py:78-85`，依赖 `pydometer==2019.2.8`，README 要求手工修补该包的 DataFrame 构造 bug）：对**整条序列**的加速度计求模 `|a|`，做 5 阶 Butterworth 低通（截止 5 Hz，`filtfilt` 零相位），减去整条序列的均值，再用 `scipy.signal.find_peaks(height=整条序列的标准差)` 找峰，峰值位置置 1。这个过程是非因果的，而且用到序列级统计量。论文式 (11) 描述的是“局部方差步伐检测”，与代码实现不同。

## 3. 输出

- 官方：两个标量头 `velx`、`vely`，形状均为 `(B,1)`（cell 11）。**目标名为 velocity，实际是窗口位移**：`vx = tango_pos[-1,x] − tango_pos[0,x]`（`data_utils.py:119-123`），即 399 个采样间隔（1.995 s）内的 Tango 世界系水平位移，单位 m；无协方差。
- 官方轨迹（cell 16，论文式 17）：从参考轨迹首点 `(x0, y0)` 出发，每个窗口累加 `L_t = L_{t−1} + v_t · s/(n−s) = v_t / 19`（`s=20, n=400`）。这里没有时间轴，每个窗口前进一步，相当于每 0.1 s 一步。严格来说系数应为 `s/(n−1) = 20/399`，官方 1/19 使步长偏大约 5%。**参考轨迹也按同样方式**由窗口目标累加得到，指标比较的对象是这条重建轨迹，而不是原始 Tango 轨迹。
- IPB：`{"vel": (B,3)}`，由 `velx, vely, velz` 拼接而成，表示窗口末端机体系下的平均速度（`frame=body` 要求 `dims=3`）。推理时用姿态旋回世界系后取水平分量，再按 DESIGN §5 积分。

## 4. 网络结构

**结构来源**：笔记本的模型由 NAS 超参数决定（cell 8/11），论文只在图 8 里定性展示了 RoNIN 的 NAS 结果，**没有表格给出最终超参数**。仓库 `tinyodom_tcn/model.cc` 中嵌入了一个 TFLite 模型（428,808 字节），输入 `[1,400,10]`，属于 RoNIN。按张量名称与 padding 反推，其超参数为：`nb_filters=30, kernel_size=12, dilations=[1,2,4,8,64], use_skip_connections=False`，无归一化，输出 `velx, vely`。用 keras-tcn 3.3.0 在 Keras 3 下重新实例化该结构，拷入 flatbuffer 中的卷积核与 Dense 核，与 TFLite 解释器对随机输入的输出相比，最大绝对误差为 **5.7e-6**，说明逐层语义完全一致。需要注意，该 flatbuffer 的权重正好是截断 he_normal 初值（最大值等于截断界 ±0.1695，std 0.0746 = √(2/360)），bias 全为 0，**是一个未训练的 HIL-NAS 候选**，而不是发布的训练好的模型；它的大小（428.8 kB）也与论文表 5 中任何一个 RoNIN 模型（50.8/65.3/147.1/253.9 kB）都对不上。IPB 以它作为 `tinyodom` 的**参考结构**；其 dropout 率从推理图中无法得知，unified 配方取 0.0（NAS 搜索范围 {0,…,0.4} 的下界）。

**keras-tcn 3.3.0 残差块**（`tcn.py:81-164`），输入通道 `C_in`，滤波器数 `F`，核 `k`，膨胀 `d`：
- 主支：`Conv1D(F, k, dilation=d, padding='causal', he_normal, bias)` → `ReLU` → `SpatialDropout1D(p)` → `Conv1D(F, k, d, causal)` → `ReLU` → `SpatialDropout1D(p)` → `ReLU`（第三个 ReLU 由 `tcn.py:134` 追加，对非负输入是恒等，可以省略但应在 docstring 注明）。`use_batch_norm` 时每个卷积后插入 BN，本结构未使用。
- 捷径：`C_in ≠ F` 时为 `Conv1D(F, 1, padding='same')`（`matching_conv1D`），否则为恒等（`tcn.py:116-128`）。
- 输出：`ReLU(shortcut + branch)`；块还返回 `skip_out = branch`（相加前的主支输出）。
- causal padding 在左侧补 `(k−1)·d` 个零，输出长度保持 T。
- TCN 层（`tcn.py:309-331`）：若 `use_skip_connections=True`，输出 = 各块 `skip_out` 之和（丢弃主干输出，求和后不再激活）；否则为最后一块的输出。`return_sequences=False` 时取最后一个时间步 `x[:, -1, :]`。默认值：`nb_stacks=1, padding='causal', activation='relu', kernel_initializer='he_normal'`。

逐层表（IPB 基准配置，Keras 形状 `(B, T, C)`，B=1、T=400；PyTorch 移植时卷积在 `(B, C, T)` 上进行）：

| # | 层 | 参数（核/步长/填充/通道） | 归一化 | 激活 | dropout | 输出形状 |
|---|---|---|---|---|---|---|
| 0 | 通道置换（IPB 适配） | `(B,6,T)[gyro,acc]` → `(B,T,6)[acc,gyro]` | — | — | — | (1,400,6) |
| 1 | block0.conv1D_0 (2,190) | k=12, d=1, s=1, 左侧补 11，6→30 | 无 | ReLU | SpatialDropout1D(p) | (1,400,30) |
| 2 | block0.conv1D_1 (10,830) | k=12, d=1, 左侧补 11，30→30 | 无 | ReLU（+ 冗余 ReLU） | SpatialDropout1D(p) | (1,400,30) |
| 3 | block0.matching_conv1D (210) | k=1, 6→30，作用在块输入上 | — | — | — | (1,400,30) |
| 4 | block0 相加 | shortcut + branch | — | ReLU | — | (1,400,30) |
| 5 | block1 (21,660) | 两个卷积 k=12, d=2, 左侧补 22, 30→30；恒等捷径；相加后 ReLU | 无 | ReLU | p | (1,400,30) |
| 6 | block2 (21,660) | 同上，d=4，补 44 | 无 | ReLU | p | (1,400,30) |
| 7 | block3 (21,660) | 同上，d=8，补 88 | 无 | ReLU | p | (1,400,30) |
| 8 | block4 (21,660) | 同上，d=64，补 704 | 无 | ReLU | p | (1,400,30) |
| 9 | 取末时间步 | `x[:, -1, :]`（无 skip 求和） | — | — | — | (1,30) |
| 10 | Reshape | `(B,30) → (B,30,1)`（笔记本为 `tf.reshape(m,[-1,F,1])`） | — | — | — | (1,30,1) |
| 11 | MaxPooling1D | pool=2, stride=2, valid；**沿滤波器轴**两两取最大（F 为奇数时丢掉最后一个） | — | — | — | (1,15,1) |
| 12 | Flatten | — | — | — | — | (1,15) |
| 13 | Dense `pre` (512) | 15→32 | — | **linear（无激活）** | — | (1,32) |
| 14 | Dense `velx` / `vely` / `velz` (各 33) | 32→1，三头并联 | — | linear | — | 各 (1,1) |

- 参数总量：官方 RoNIN 配置（10 通道，2 个头）为 **102,008**；IPB 基准配置（6 通道，3 个头）为 **100,481**，由夹具锁定；6 通道 2 头（`dims=2` 消融）为 100,448。参数量与窗口长度 T 无关。
- 感受野：每块两个卷积，实际感受野为 `1 + 2·(k−1)·Σd = 1 + 2·11·79 = 1739` 个样本，大于 400。keras-tcn 自带的 `receptive_field` 属性（`tcn.py:246`）给出的是 `1+k·Σd`，与实际不符，不要用它校验。
- 权重初始化：卷积为 `he_normal`（TF 实现为截断正态，σ = √(2/fan_in)/0.8796，在 ±2σ 处截断），卷积 bias 为 0；Dense 为 `glorot_uniform`，bias 为 0。
- 参数注册顺序（夹具 `param_names`，Keras 顺序）：`block0.conv1D_0.{kernel,bias}`、`block0.conv1D_1.{kernel,bias}`、`block0.matching_conv1D.{kernel,bias}`，然后 block1…block4 各自的 `conv1D_0`、`conv1D_1`，最后是 `pre`、`velx`、`vely`、`velz`。Keras Conv1D 核形状为 `(k, in, out)`，对应 PyTorch 的 `(out, in, k)`，需要做 `permute(2,1,0)`；Dense 核形状为 `(in, out)`，对应 PyTorch Linear 的 `(out, in)`，需要转置。

## 5. 损失与训练配方（official）

| 项 | 值 | 来源 |
|---|---|---|
| 损失 | `{'velx':'mse','vely':'mse'}` 等权求和。论文式 (16) 带权重 κ，代码中等价于 κ=1 | cell 11；论文 §4.0.3 |
| 优化器 | `tf.keras.optimizers.Adam()` 默认参数（lr 1e-3，β=(0.9,0.999)，ε=1e-7） | cell 11；论文 §6.1 |
| 学习率 / 调度 | 1e-3，常数，无调度 | cell 11 |
| batch | 256 | cell 11 |
| epoch | 900（`model_epochs`，RoNIN/OxIOD）；其他领域为 300 | cell 7；论文表 2 |
| 权重衰减 | 无 | cell 11 |
| 梯度裁剪 | 无 | cell 11 |
| 增强 | RoNIN 笔记本 `AugmentationCopies=0`，即**不做增强**。论文 §4.0.1 描述的“受控随机旋转 + 多元高斯噪声”在 `random_rotate` 中只实现了旋转：每份拷贝抽一组欧拉角 U(0,π)³，对该序列所有窗口的 acc/gyro/mag 做同一个 3D 旋转（`data_utils.py:198-214`），没有实现噪声 | cell 3-5；`data_utils.py` |
| 阶段切换 | 无 | — |
| 选模/早停 | `ModelCheckpoint(monitor='loss', save_best_only=True)`，按**训练损失**选最优 epoch；训练阶段不跑验证 | cell 8/11；论文 §6.3 |
| NAS | Mango `Tuner`（高斯过程 + UCB），迭代 50 次（`NAS_epochs`），`initial_random=5`，`batch_size=1`。搜索空间（cell 9）：`nb_filters∈[2,63]`，`kernel_size∈[2,15]`，`dropout∈{0,0.1,0.2,0.3,0.4}`，`use_skip_connections∈{T,F}`，`norm_flag∈{0}`，膨胀为 {1,2,…,256} 中长度 3–8 的升序子集。评分：HIL 时为 `−(RMSE_x+RMSE_y) + 0.01·(RAM/maxRAM + Flash/maxFlash) − 0.05·Latency`；使用代理时最后一项换成 `−0.05·FLOPs/30e6`；装不下目标硬件的候选记 −5。验证集只用于计算评分 | cell 8/9；论文 §6.1-6.2 |
| 划分 | RoNIN 官方划分；仓库自带 `list_train.txt`(69)/`list_val.txt`(15)/`list_test_seen.txt`(31)/`list_test_unseen.txt`(31)。作者称只能拿到约 50% 的 RoNIN 数据；论文表 2 给出的比例为 70/5/25 | `RoNIN/dataset_download_and_splits/`；论文表 1/2 |

## 6. benchmark 适配（unified 配方）

| 改动 | 类型 | 理由 |
|---|---|---|
| 固定参考结构 `F=30, k=12, d=[1,2,4,8,64]`，无 skip、无归一化，dropout 取 0.0，**不跑 NAS** | 不改变结构（固定一个官方候选） | 官方没有公开最终 NAS 结果；NAS 依赖 MCU 硬件在环，不属于 benchmark 范畴。`model.cc` 是仓库内唯一的具体 RoNIN 结构，且已做数值校验 |
| 输入通道 10→6：去掉磁力计 3 维和 `step_mask` 1 维；`block0.conv1D_0` 与 `matching_conv1D` 的 `in_channels` 由 10 改为 6（参数少 1,560） | 改变结构 | IPB 格式没有磁力计；`step_mask` 依赖整条序列的非因果统计量和第三方包，不能在窗口内复现。可选变体 `physics_channel=true`：在模型内部按 §2 的算法对窗口 acc 计算 `step_mask`（统计量改用窗口内值），`in_channels=7`，默认关闭 |
| 通道置换 `[gyro,acc]→[acc,gyro]`，`(B,6,T)→(B,T,6)` | 不改变结构 | 保持官方通道语义 |
| 视图 `frame=body, dims=3, target=avg_velocity, orientation=reference`；输出头 2→3（新增 `velz`，Dense 32→1） | 改变结构（+33 参数） | 官方靠磁力计从机体系输入推断世界系航向；没有磁力计时，世界系航向对网络不可观测。改为预测窗口末端机体系速度，这样网络本身仍不需要姿态，姿态只用于目标构造与轨迹重建。DESIGN §3 规定 `body` 系必须 `dims=3` |
| 消融 `tinyodom_world2d`（不注册为主模型）：`frame=gravity_world, dims=2`，保留官方 2 个头，网络不接收任何姿态 | 改变结构（仅输入通道） | 用来复现官方语义在无磁力计时的失败模式，供分析用 |
| 目标缩放：官方 1.995 s 窗口位移 → IPB 平均速度 | 不改变结构 | 线性缩放 |
| 窗口 400；训练步长用 IPB 默认 10（官方为 20）；推理步长 10 | 不改变结构 | 参数量与 T、步长无关；统一推理协议 |
| 轨迹：DESIGN §5 的时间轴积分，替代官方的 `v/19` 逐窗累加 | 不改变结构（评测协议） | 官方系数有约 5% 偏差，且没有时间轴 |
| 选模：val ATE，替代训练损失 | 不改变结构（训练协议） | IPB 诚实协议 |
| 损失：3 个头 MSE 等权求和；Adam lr 1e-3 常数；batch 256；无增强。official 配方 900 epoch；unified 配方按 IPB 统一预算 | 不改变结构 | 与官方一致 |
| 可选增强 `random_rotation3d`：输入 acc/gyro 与**机体系目标**做同一个旋转 | 不改变结构 | 官方 RoNIN 设置不做增强；如果开启，必须同时旋转目标，否则会破坏机体系目标的一致性 |
| Keras 的 `tf.reshape` → `Reshape((F,1))` 层 | 不改变结构 | Keras 3 不允许对 KerasTensor 直接调用 `tf.reshape`，两者数值等价 |

## 7. 官方报告数值

论文表 5（ATE/RTE 单位 m；SRAM/Flash 单位 kB；FLOPs 单位 M）。§6.5 的指标定义：ATE 是整条轨迹上预测与真实位置误差的“average RMSE”（式 34）；RTE 是在 1 分钟时间间隔上计算的同类误差。代码实现（`Cal_TE`）见 §10。

| 数据集 | 方法（硬件） | SRAM | Flash | FLOPs | ATE | RTE |
|---|---|---|---|---|---|---|
| RoNIN | TinyOdom (STM32F446RE) | 36.2 | 50.8 | 4.15 | 28.3 | 7.76 |
| RoNIN | TinyOdom (STM32L476RG) | 56.2 | 65.3 | 7.80 | 23.9 | 6.74 |
| RoNIN | TinyOdom (STM32F407VET6) | 138.3 | 147.1 | 26.54 | 27.7 | 6.20 |
| RoNIN | TinyOdom (STM32F746ZG) | 257.3 | 253.9 | 49.44 | 27.36 | 5.84 |
| RoNIN | RoNIN TCN（# 用完整 RoNIN 训练的官方模型） | 2046.3 | 2195.5 | 440 | 4.73 | 1.21 |
| RoNIN | IONet（作者复现） | 976.3 | 782.0 | – | 22.52 | 7.63 |
| RoNIN | L-IONet（作者复现） | 159.0 | 182.9 | 26.8 | 24.73 | 14.84 |
| RoNIN | PDR | 10.8 | 49.6 | – | 34.81 | 23.62 |
| OxIOD | TinyOdom (STM32F446RE) | 52.4 | 71.6 | 4.64 | 3.30 | 1.24 |
| OxIOD | TinyOdom (STM32L476RG) | 72.5 | 89.6 | 6.65 | 3.59 | 1.37 |
| OxIOD | TinyOdom (STM32F407VET6) | 90.1 | 117.6 | 8.92 | 6.82 | 1.28 |
| OxIOD | TinyOdom (STM32F746ZG) | 55.5 | 71.0 | 4.92 | 2.80 | 1.26 |
| OxIOD | RoNIN TCN | 2046.3 | 2195.5 | 220 | 1.95 | 0.42 |

论文表 6 列出了不做微调时的跨数据集 RTE（单位 m，括号内为训练集）：TinyOdom(RoNIN) 在 RoNIN 上为 6.74、在 OxIOD 上为 3.16；TinyOdom(OxIOD) 在 OxIOD 上为 1.26、在 RoNIN 上为 97.2。论文图 11 的消融（在 OxIOD 与 AQUALOC 上）显示，去掉磁力计或速度形式会显著增大 ATE/RTE。图中只有柱状图，没有数值表，本卡不摘录具体数值。

## 8. 忠实性测试建议

- [ ] `total_params == 100,481`，`trainable_params == 100,481`；把 `in_channels` 改为 10、头数改为 2 后应得 102,008（官方配置）
- [ ] `param_shapes` 与夹具 30 项逐项一致；PyTorch 移植需先把 Keras 形状换算为 PyTorch 形状（卷积核 `permute(2,1,0)`，Dense 核转置）
- [ ] 输入 `(B,6,400)` → `vel (B,3)`；输入 `(B,6,T)` 取任意 T≥1 都能运行，且参数量不变
- [ ] **因果性与感受野**：输出只取最后一个时间步，感受野为 1739 个样本。取 T=2000，修改前 `T−1739=261` 个样本中的任意一个，输出应不变；修改第 262 个样本起的任意样本，输出一般会改变。T=400 时全部样本都在感受野内
- [ ] **数值对拍**（可选，需 TF）：用 `model.cc` 解码出的卷积核与 Dense 核（bias 全 0）搭建官方 10 通道、2 头结构，对随机输入与 TFLite 解释器输出对比，误差应小于 1e-4（本卡实测 5.7e-6）；对拍脚本放在 `tests/` 之外，不作为必需测试
- [ ] MaxPool 沿滤波器轴：构造 `pre` 前的 30 维特征，验证池化结果为 `max(f[2i], f[2i+1])`
- [ ] 残差块捷径：只有 block0 含 1×1 卷积，其余块为恒等（参数形状中恰好出现一次 `[1,6,30]`）
- [ ] 通道置换：`[gyro,acc]` 输入按下标 `[3,4,5,0,1,2]` 重排后送入骨干（可以用单位核的 1×1 探针验证）

## 9. 预训练权重

没有发布训练好的权重。`tinyodom_tcn/model.cc` 中的 flatbuffer 是**未训练**的 NAS 候选（权重为初始化值、bias 全 0，见 §4），只能用来确定结构和做数值对拍，不能作为预训练模型。

## 10. 官方实现的坑与未决问题

1. **链式赋值导致测试集泄漏**：RoNIN 笔记本 cell 4/5 写成 `X_val, … = X, Y_disp, … = import_ronin_dataset(...)`，Python 会把同一结果**同时**赋给 `X, x_vel, y_vel…`。从头运行时，这些训练变量最终被 **seen 测试集**覆盖，于是 cell 8（NAS 候选训练）和 cell 11（最终训练）的 `model.fit(x=X, y=[x_vel, y_vel])` 实际在测试集上训练，而 cell 14/16 又在同一个测试集上评测。GunDog 笔记本的测试集 cell 有同样问题；OxIOD 笔记本没有。
2. **最终训练 cell 无法运行**：cell 11 读取 `results['best_params']['dilations']`，而搜索空间的键名是 `'dil_list'`（cell 9），会抛出 `KeyError`。cell 23 引用了未定义的 `X_tr`。cell 18/20 使用了残留的循环变量 `i` 来取 `x0_list_test[i]`。
3. **搜索空间与论文不符**：`norm_flag` 取 `np.arange(0,1)`，即只有 0，归一化从未被搜索（论文称搜索 Weight/Layer/Batch）；dropout 只到 0.4（论文写 0–1.0）；滤波器数只到 63、核只到 15（论文写 64、16）。
4. **论文与代码的残差块不一致**：论文式 (25) 描述的是门控残差 `tanh(W_f*x) ⊙ σ(W_g*x)`；代码使用的 keras-tcn 3.3.0 是普通 ReLU 残差块。本卡以代码为准。
5. **“velocity” 实为 2 s 窗口位移**，轨迹系数 `1/19` 相对精确值 `20/399` 偏大约 5%；参考轨迹也按同样方式重建，所以官方 ATE/RTE 是相对重建轨迹计算的。
6. **指标定义与 RoNIN 不同**：`Cal_TE`（`data_utils.py:241-269`）中，ATE 是逐窗位置误差的**均值**，不是 RMSE；RTE 是**前 1 分钟内**（前 `int((200·60−400)/20)=580` 个窗口）绝对位置误差的均值，并不是按区间的相对误差；序列不足 1 分钟时，RTE = ATE × 580/窗口数。笔记本最后输出的是各序列的中位数。
7. **RoNIN 上航向在原理上不可观测**：目标位于 Tango 世界系，按 RoNIN 数据说明，其偏航由每条序列 Tango 会话的起点决定，与磁北没有固定关系，所以磁力计无法锚定该偏航。这是本卡的分析判断，也能解释 RoNIN 上 ATE 23.9–28.3 m 远差于 OxIOD。
8. **选模看训练损失**（`monitor='loss'`），验证集只在 NAS 外层循环中使用。
9. **MaxPool 沿滤波器轴池化**，`pre` 层为线性激活，所以 `pre` 与输出头合起来是一个线性映射（输出头总秩 ≤ 2）；F 为奇数时最后一个滤波器被丢弃。
10. **keras-tcn 细节**：残差主支末尾有一个冗余 ReLU（`tcn.py:134`）；`use_skip_connections=True` 时对各块“相加前的主支输出”求和，并丢弃主干；`receptive_field` 属性公式有误（见 §4）；3.3.0 默认 `use_skip_connections=False`（笔记本显式传参）。
11. **`model.cc` 不是发布模型**：它是 HIL 评估时在训练**之前**转换出的候选（cell 8 中 `convert_to_tflite_model` 在 `fit` 之前调用），权重为初值。`main.cpp` 中的 `numChannels=6` 与该模型的 10 通道输入不一致（`main.cpp:16`）；HIL 控制器每次运行都会按当前窗口和通道数改写这两行（`RoNIN/hardware_utils.py:36-47`），所以仓库里的 `main.cpp` 与 `model.cc` 不是同一次运行的产物。
12. **OxIOD 加速度计通道复制 bug**：`OxIOD/data_utils.py:84-85` 写成 `acc_y = acc_x.reshape(...)`、`acc_z = acc_x.reshape(...)`，所以 OxIOD 输入中的三个 acc 通道全都是 x 分量。论文中的 OxIOD 结果（表 5）就是在这一输入下得到的。RoNIN 的数据管线没有这个问题。
13. **环境**：原始环境为 TF 2.5 + keras-tcn 3.3.0；在 TF 2.21/Keras 3 下，keras-tcn 3.3.0 仍能构建，只有笔记本里的 `tf.reshape` 需要换成 `Reshape` 层。
