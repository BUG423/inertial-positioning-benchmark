# IPB v1 算法总表 / Algorithm Catalogue

> 状态：v1 规格基线（2026-09-17）。本文件是 `docs/algorithms/*.md` 规格卡的索引，机读版本为
> [`docs/algorithms/index.yaml`](algorithms/index.yaml)。规格卡供“算法移植工程师”**不看官方代码**、照卡实现；
> 忠实性由 `tests/fixtures/algorithms/<name>.json` 夹具（参数量与逐项参数形状）锁定。
> 本文件只描述规格，不代表 `src/` 中已有实现。

## 1. 约定

- **fidelity**
  - `official`：实例化官方仓库模型（CPU、随机权重、固定种子）导出夹具，卡片内容以官方代码为准，与论文不一致处逐条注明。
  - `paper-only`：没有可用的官方源码，只依据论文写重实现规格。论文未给出的细节在卡片中标为假设，并给出推荐默认值；参数量是按卡片规格推导的，不能作为忠实性证据。
  - `定义`：经典非学习基线，按 IPB 自己的定义实现，只有少量标定标量。
- **参数量**：“官方”指官方发布的配置；“IPB”指按卡片 §6 适配到 IPB 协议（200 Hz、IPB 窗口、必要时改输出维度）后的配置，由夹具的 `total_params` 锁定。两者相同时只写一个数。
- **依赖外部姿态**：指官方**推理**时网络输入或轨迹重建是否需要外部姿态（设备 game RV、VIO 或滤波器姿态）。
- **状态**：`可移植` 表示卡片足以直接实现；`许可风险` 表示仓库无 LICENSE 或为强 copyleft，只能照卡独立实现，不得复制代码或再分发权重；`部分` 表示论文或代码缺少关键细节，卡中有待核实的假设。
- **夹具格式**：必需键为 `name, source_repo, commit, input_shape, output_shape, total_params, trainable_params, param_shapes, notes`。`param_shapes` 按 `named_parameters()` 的注册顺序排列（Keras 模型按 `model.weights` 顺序）。`input_shape`/`output_shape` 在单张量时为列表，多输入/多输出时为“名称 → 形状”的字典。各夹具另带 `param_names`、`buffer_shapes`、`official_config`、`benchmark_config` 等补充键；参数名仅作核对参考，移植实现不必沿用。
- **官方代码来源**：所有官方仓库浅克隆在 `/workspace/webCodex/third_party/<repo>`（不进入本仓库），提交号见 §5。本仓库不包含任何官方代码或权重。

## 2. 总表

按 P0（行人、官方代码可用）→ P1（跨平台或需接口扩展）→ paper-only 重实现 → 经典基线排序。

| 名称 | 优先级 | fidelity | 许可 | 输入（官方） | 输出（官方） | 参数量（官方 / IPB） | 依赖外部姿态 | 官方数据集 | 状态 |
|---|---|---|---|---|---|---|---|---|---|
| [`ronin_resnet18`](algorithms/ronin_resnet18.md) | P0 | official | GPL-3.0 | 200 Hz×200；`gravity_world`；`[gyro,acc]` | 2D 平均速度 | 4,634,882 | 是 | RoNIN, RIDI, OxIOD | 可移植 |
| [`ronin_lstm`](algorithms/ronin_lstm.md) | P0 | official | GPL-3.0 | 200 Hz×400 序列（batch-first）；`gravity_world`；`[gyro,acc]` | 逐帧 2D 速度 | 216,620 | 是 | RoNIN, RIDI, OxIOD | 可移植 |
| [`ronin_tcn`](algorithms/ronin_tcn.md) | P0 | official | GPL-3.0 | 200 Hz×400 序列（batch-first）；`gravity_world`；`[gyro,acc]` | 逐帧 2D 速度 | 540,488 | 是 | RoNIN, RIDI, OxIOD | 可移植 |
| [`tlio`](algorithms/tlio.md) | P0 | official | BSD-3 | 200 Hz×200；`gravity_yaw_local`；`[gyro,acc]` | 3D 位移＋logstd | 5,424,646 | 是 | TLIO | 可移植 |
| [`rnin`](algorithms/rnin.md) | P0 | official | Apache-2.0 | 100 Hz，10×100 子窗；`gravity_world`、去重力；`[gyro,acc]` | 10 步 3D 位移＋logstd | 5,358,342 | 是 | IDOL, RNIN | 可移植 |
| [`imunet`](algorithms/imunet.md) | P0 | official | 无 LICENSE ⚠ | 200 Hz×200；`gravity_world`；`[gyro,acc]` | 2D 平均速度 | 3,661,618 | 是 | RoNIN, RIDI, OxIOD, IMUNet, PX4 | 可移植（许可风险） |
| [`imunet_mobilenet`](algorithms/imunet_mobilenet.md) | P0 | official | 无 LICENSE ⚠ | 同 IMUNet | 2D 平均速度 | 3,178,978 | 是 | RoNIN, RIDI, OxIOD, IMUNet, PX4 | 可移植（许可风险） |
| [`imunet_mobilenetv2`](algorithms/imunet_mobilenetv2.md) | P0 | official | 无 LICENSE ⚠ | 同 IMUNet | 2D 平均速度 | 2,183,330 | 是 | RoNIN, RIDI, OxIOD, IMUNet, PX4 | 可移植（许可风险） |
| [`imunet_mnasnet`](algorithms/imunet_mnasnet.md) | P0 | official | 无 LICENSE ⚠ | 同 IMUNet | 2D 平均速度 | 2,979,114 | 是 | RoNIN, RIDI, OxIOD, IMUNet, PX4 | 可移植（许可风险） |
| [`imunet_efficientnetb0`](algorithms/imunet_efficientnetb0.md) | P0 | official | 无 LICENSE ⚠ | 同 IMUNet | 2D 平均速度 | 3,231,906 | 是 | RoNIN, RIDI, OxIOD, IMUNet, PX4 | 可移植（许可风险） |
| [`eqnio_ronin`](algorithms/eqnio_ronin.md) | P0 | official | 无 LICENSE ⚠ | 200 Hz×200；`gravity_world`；`[gyro,acc]`（内部 O(2) 规范化） | 2D 速度（＋规范帧） | 5,230,466 | 是 | RoNIN, RIDI, OxIOD | 可移植（许可风险） |
| [`eqnio_tlio`](algorithms/eqnio_tlio.md) | P0 | official | 无 LICENSE ⚠ | 200 Hz×200；`gravity_yaw_local`；`[gyro,acc]`（内部 O(2) 规范化） | 3D 位移＋logstd（规范帧内） | 6,020,230 | 是 | TLIO, Aria Everyday | 可移植（许可风险） |
| [`llio`](algorithms/llio.md) | P0 | official | GPL-3.0 | 100 Hz×100（README 注释）；`gravity_yaw_local`；`[gyro,acc]` | 3D 位移＋logstd | 7,191,654 | 是 | unknown | 部分（假设待核实） |
| [`tinyodom`](algorithms/tinyodom.md) | P0 | official | BSD-3 | 200 Hz×400；机体系原始读数；`[acc,gyro,mag,step_mask]` | 2 s 位移 (x, y) | 102,008 / 100,481 | 否（IPB 适配后仅用于目标与积分） | RoNIN, OxIOD, AQUALOC, EuRoC MAV, GunDog | 可移植 |
| [`deepils`](algorithms/deepils.md) | P0 | official | 无 LICENSE ⚠ | 200 Hz×200；`gravity_world`；`[gyro,acc]` | 2D 平均速度 | 2,290,226 | 是 | RoNIN, RIDI, OxIOD, IMUNet, KIOD, INAIOD | 可移植（许可风险） |
| [`tartanimu`](algorithms/tartanimu.md) | P0 | official | Apache-2.0 | 200 Hz，10×1 s 子窗各抽到 40 点；`body`；`[gyro,acc]`＋平台头 | 逐子窗机体系 3D 速度＋logstd | 5,698,346 | 网络否；积分需要 | SubT-MRS, IDOL, Blackbird, UZH-FPV, TartanDrive 等 | 可移植 |
| [`dive`](algorithms/dive.md) | P1 | official | MIT | 400 Hz×1400；局部重力系、去重力；`[Log(C), acc−g]`＋窗末姿态 | 窗末 3D 速度＋logstd | 12,367,110 / 7,516,420 | 是 | DIDO, Blackbird | 可移植 |
| [`airio`](algorithms/airio.md) | P1 | official | BSD-3 | 200 Hz×1000；`body`＋逐样本姿态编码；`[acc,gyro]` | 逐帧机体系 3D 速度＋方差 | 387,014 | 是 | EuRoC, Blackbird, Pegasus | 可移植 |
| [`nio_lie_events_ronin`](algorithms/nio_lie_events.md) | P1 | official | 无 LICENSE ⚠ | 200 Hz×200 → 12 通道 SE(3) 李事件；需逐样本姿态、v0 | 2D 速度 | 4,637,570 | 是 | RoNIN, RIDI, OxIOD | 可移植（许可风险） |
| [`nio_lie_events_tlio`](algorithms/nio_lie_events.md) | P1 | official | 无 LICENSE ⚠ | 同上，`gravity_yaw_local` | 3D 位移＋logstd | 5,427,334 | 是 | TLIO, Aria Everyday | 可移植（许可风险） |
| [`pedestrian_diffusion`](algorithms/pedestrian_diffusion.md) | P1 | official | AGPL-3.0 ⚠ | 100 Hz×100 → STFT；`gravity_world`；`[acc,gyro]`＋CLIP 文本 | 逐帧 3D 速度与角速度 | 37,259,798 | 是 | RoNIN, RIDI, OxIOD, TLIO | 可移植（许可风险） |
| [`ionet`](algorithms/ionet.md) | paper-only | paper-only | 无代码 | 100 Hz×200；机体系；`[acc,gyro]` | 极坐标 (Δl, Δψ) | 论文未报告 / 推导 302,978 | 否（官方） | IONet 自采 Vicon 集、OxIOD | 可移植 |
| [`ctin`](algorithms/ctin.md) | paper-only | paper-only | 占位仓库（MIT，无源码） | 200 Hz×200；`gravity_world`；`[gyro,acc]` | 逐帧 2D 速度＋logstd | 论文 0.557 M / 推导 477,572 | 是 | RIDI, OxIOD, RoNIN, IDOL, CTIN | 部分（假设待核实） |
| [`imot`](algorithms/imot.md) | paper-only | paper-only | 占位仓库（MIT，无源码） | 200 Hz×200；世界系（论文未明示）；`[acc,gyro]` | 2D 速度 | 论文 14.49 M / 推导 7,137,482 | 是（来源未明示） | RIDI, RoNIN, OxIOD, IDOL | 部分（假设待核实） |
| [`rio`](algorithms/rio.md) | paper-only | paper-only | 无代码 | 200 Hz×200；`gravity_world`；`[gyro,acc]` | 2D 速度 | 论文未报告 / 推导 4,634,882 | 是 | RoNIN, OxIOD, RIDI, IPS | 可移植 |
| [`ionext`](algorithms/ionext.md) | paper-only | paper-only | 无代码 | 200 Hz×200；`gravity_world`；`[acc,gyro]` | 2D 速度 | 论文 ≈1.1×10⁷ / 推导 10,641,506 | 是 | IMUNet, RoNIN, RIDI, OxIOD, RNIN, TLIO | 部分（假设待核实） |
| [`gnio`](algorithms/gnio.md) | paper-only | paper-only | 无代码 | 200 Hz×200（1 s 窗、0.1 s 步长）；`gravity_yaw_local`；`[gyro,acc]` | 3D 位移＋logstd（门控头） | 论文 4.90 M / 推导 4,934,153 | 是 | OxIOD, RIDI, RoNIN, IDOL, TLIO | 可移植 |
| [`velobins`](algorithms/velobins.md) | paper-only | paper-only | 无代码 | 100 Hz×100；`body`；`[acc,gyro]`＋逐旋翼转速/PWM | 逐轴 512 箱分布 → 机体系 3D 速度＋方差 | 论文 ≈76.6 k / 推导 69,968（无转速支路） | 否（网络）；积分需要 | AI-IO, NanoBench, TII-RATM, NeuroBEM | 部分（假设待核实） |
| [`pdr`](algorithms/pdr.md) | baseline | 定义 | — | 200 Hz 连续序列；加计＋姿态 | 2D 平均速度 | 0（标定标量） | 是 | — | 可移植 |
| [`mean_speed_heading`](algorithms/mean_speed_heading.md) | baseline | 定义 | — | 姿态（航向）；门控变体用加计 | 2D 平均速度 | 0（标定标量） | 是 | — | 可移植 |

`nio_lie_events_ronin` 与 `nio_lie_events_tlio` 共用一张卡 [`nio_lie_events.md`](algorithms/nio_lie_events.md)：两者的事件生成完全相同，只是骨干不同；夹具仍按注册名分开。
IMUNet 的四个变体指仓库中的一维 MobileNet / MobileNetV2 / MnasNet / EfficientNetB0 实现，并非标准视觉版本（差异见各卡 §10）。

## 3. 协议扩展（状态）

DESIGN §3/§4 原本规定模型只接收 `(B, 6, T)` 并返回窗口级 `vel`。下列扩展已在 DESIGN §3/§4/§5 中落地
（**已落地**一列给出配置键与文档位置）；尚未落地的扩展，对应卡片仍给出降级方案，并要求在结果中标注。

| 扩展 | 用途 | 涉及算法 | 状态 |
|---|---|---|---|
| 逐帧目标 `target=frame_velocity`，输出布局 `(B,T,D)` | seq2seq 专用损失（RoNIN 的 latent velocity loss、CTIN 的 IVL/CNL、AirIO 的逐帧机体系速度） | `ronin_lstm`、`ronin_tcn`、`ctin`、`airio`、`pedestrian_diffusion` | **已落地**：DESIGN §3.1；`target=frame_velocity`，损失按 `mask` 跳过无效帧 |
| 多步目标 `target=multi_displacement` + `output_steps`，输出布局 `(B,H,D)` | 一次预测多步位移（RNIN 的 10 步） | `rnin` | **已落地**：DESIGN §3.1；窗口等分为 `output_steps` 段，各段按自己的跨度换算速度 |
| 重叠预测合并 `overlap` | 逐帧/多步输出在滑窗下重叠，需要确定的合并规则 | 同上 | **已落地**：DESIGN §5 第 1 条；`overlap=mean`（默认）/ `center` |
| 历史上下文 `history` / `history_stride`：输入 `(B,H_in,6,T)` | 多窗口序列输入（10 个 1 s 子窗） | `rnin`（`history=10, history_stride=100`）、`tartanimu`（方案 A） | **已落地**：DESIGN §3.2；有效性按整个输入跨度判断 |
| 额外输入 `extra_inputs`：逐样本姿态、重力、初始速度 v0 | 输入不能只由 `[gyro, acc]` 构造 | `dive`、`airio`、`nio_lie_events_*`、`pdr`、`mean_speed_heading` | **已落地**：DESIGN §3.3；`orientation` / `gravity` / `init_velocity`（**特权输入**，结果中标记并单列） |
| 序列级 / 有状态预测器钩子 | 整序列流式推理、上一窗口预测作为下一窗口输入、测试时训练 | `ronin_lstm@stream`、`ronin_tcn@stream`、`nio_lie_events_ronin`、`ionet`（官方协议模式）、`rio`（A-TTT）、`pdr` | **已落地**：DESIGN §4 的 `SequenceModel.predict_sequence(seq, view, starts)` 与 `calibrate(views, split)` |
| 时间戳 / `body_frame` 等序列级元信息 | PDR 的前向轴查表 | `pdr`、`mean_speed_heading` | **已落地**：序列级模型直接拿到 `Sequence` 与 `SequenceView`（含 `attrs`） |
| 训练姿态回退 `orientation_fallback`（设备姿态末端对齐误差 > 20° 时改用参考姿态） | RoNIN 系官方训练姿态规则 | `ronin_*`、`eqnio_ronin`、`nio_lie_events_ronin` | 未落地（卡片给降级方案） |
| 窗口速度上限过滤（例如 3 m/s、4 m/s） | 官方训练样本筛选 | `ronin_lstm`、`ronin_tcn`、`rnin` | 未落地（卡片给降级方案） |
| `dims=3` 与 `gravity_yaw_local` 的锚点 | TLIO 系官方以**窗口起点**偏航建局部系，IPB 以末端建系；非等变骨干在偏航增强下近似等价 | `tlio`、`llio`、`eqnio_tlio`（严格等变，可直接用 `gravity_world`）、`nio_lie_events_tlio`、`dive` | 未落地（仍用末端锚点；航向定义见 DESIGN §3） |
| 可复用输出头：`gated`（Softplus 幅值 × Tanh 门控）、`bins`（逐轴分箱分类 + 期望/argmax 解码） | 在任意骨干上做“回归 vs 门控 / 回归 vs 分箱”的对照消融 | `gnio`、`velobins` | 未落地（卡片给降级方案） |
| 分箱范围 `R` 的训练期统计与记录（逐轴 `max|v|×1.1`，只用训练划分），以及解码方式开关 | 分箱输出头的语义依赖数据集相关的箱网格 | `velobins` | 未落地（卡片给降级方案） |

两个经典基线已实现并注册：[`pdr`](algorithms/pdr.md)（`cfg/models/pdr.yaml`，
`src/inertial_benchmark/models/pdr/`）与 [`mean_speed_heading`](algorithms/mean_speed_heading.md)
（`cfg/models/mean_speed_heading.yaml`，`src/inertial_benchmark/models/mean_speed_heading/`）。
两者都是 `SequenceModel`，标定标量（`pdr` 的 `K`、`δ`；`mean_speed_heading` 的 `s̄`、`s̄_move`、`v̄`、`δ`）
只在 train 划分上拟合（传入其他划分直接报错），并随 checkpoint 与 `metrics.json` 的 `calibration` 一起发布。
测试见 `tests/test_baselines.py`（合成走路信号上的步数、振幅、距离误差、航向约定与标定恢复）。

## 4. 跨算法的官方实现问题（摘要）

逐条证据（文件与行号）见各卡 §10。

1. **窗口起点时间戳 / 按分量平均的 ATE**：RoNIN、IMUNet、DeepILS 都把窗口速度的时间戳记在窗口起点，积分轨迹因此滞后半个窗口；ATE/RTE 对 N×2 个坐标分量求均值，数值为欧氏定义的 1/√2（PedestrianDiffusion 为 3D 分量平均，是 1/√3）。TinyOdom 的 ATE 用的是平均误差而非 RMSE。IPB 按 DESIGN §5/§6 统一计算，与论文数值对照时须换算。
2. **阶段切换或调度器实际不生效**
   - TLIO：前 9 个 epoch 用的是“logstd 被 detach 的 NLL”，并非论文所说的 MSE；`ReduceLROnPlateau` 从未 `step()`。
   - EqNIO-TLIO、NIO-TLIO：调度器同样从未调用；EqNIO 的规范帧架构从不调用 `zero_grad`，梯度一直累积。
   - RNIN：`start_cov_epochs=2000` 大于 `epochs=201`，协方差头从未训练，测试时 σ 恒为 1。
   - TartanIMU：机体系分支的损失恒为 20·L1，协方差头拿不到梯度。
   - EqNIO-RoNIN：发布的训练循环只跑 1 个 epoch。
   - GNIO（论文）：正文自称“两阶段 MSE→NLL”，式 13 却是固定权重的静态加和（λ_MSE=1e2、λ_NLL=1e-4），没有任何按 epoch 的切换。
   - VeloBins（论文）：**完全不优化 NLL**，不确定度只由误差条件高斯标签的 KL 监督学到——移植时不要“顺手补一个 NLL 损失”。
3. **通道语义**
   - RoNIN 的 `lstm_bi` 指 bilinear 层，LSTM 本身是**单向**的。
   - EqNIO 在网络内部把骨干输入重排为 `[a', s·ω']`（O(2)）；陀螺必须按**赝矢量**参与反射，否则等变性不成立。夹具中给出了实测误差：正确变换时 ≤1e-12，把陀螺当普通矢量时约 0.1–0.9。
   - TinyOdom 的官方输入是 10 通道 `[acc, gyro, mag, step_mask]`；AirIO、PedestrianDiffusion、iMoT、IONext 的通道顺序为 `[acc, gyro]`；DIVE 的输入是 `[Log(C), acc−g]`，不是原始 `[gyro, acc]`。
4. **测试数据参与选模或训练**
   - IMUNet（RIDI、自采数据集、OxIOD）、DeepILS（RIDI/IMUNet/KIOD/INAIOD）、AirIO（EuRoC）用测试集做验证或选模。
   - TinyOdom 的 RoNIN 笔记本因链式赋值，实际在 seen 测试集上训练。
   - RoNIN 的预训练模型是在整个数据集（含测试序列）上训练的。
5. **论文与代码不一致**：RoNIN-TCN 通道（论文 16… 对代码 32…）；IMUNet 的结构表与代码不符；TartanIMU（论文去重力、MSE+NLL，代码保留重力、20·L1）；EqNIO 规范帧向量的顺序与符号；LLIO 回归 MLP 的 dropout 写死为 0.5。一律以**官方代码实际行为**为准，论文版本作为变体记录。
6. **无法直接运行的官方脚本**：`np.int`/`np.math`（NumPy ≥ 1.24/2.0）、`ReduceLROnPlateau(verbose=True)`（torch 2.11）、缺失模块导入、调用签名错误（NIO 训练期事件堆叠）、键名错误（TinyOdom `dil_list`、RoNIN `'feature_sigma,'`）。

## 5. 官方仓库、提交与许可

| 仓库 | 提交 | 许可 | 本地路径 / 说明 |
|---|---|---|---|
| <https://github.com/Sachini/ronin> | `805b7f0f28bb164ce89ada9ac05a9470dbe3d715` | GPL-3.0 | `third_party/ronin` |
| <https://github.com/CathIAS/TLIO> | `d7051c4ff14834b93931984b75b40f894e39654d` | BSD-3-Clause（数据 CC BY-NC） | `third_party/TLIO` |
| <https://github.com/zju3dv/rnin-vio> | `b030ecc94f151159973a9086c8ba76e22bdbc56e` | Apache-2.0（`ronin_3d/` 为 GPL-3） | `third_party/rnin-vio` |
| <https://github.com/BehnamZeinali/IMUNet> | `c57f14d0f4f5bbb8e29d836dabacc8718ca825e1` | **无 LICENSE** | `third_party/IMUNet` |
| <https://github.com/RoyinaJayanth/EqNIO> | `97b7a60a5e5cc8f0132bba343c83de114211f827` | **无 LICENSE** | `third_party/EqNIO` |
| <https://github.com/RoyinaJayanth/NIO_Lie_Events> | `1e87470f6479c198875c5f1a32b392a46eb5af74` | **无 LICENSE** | `third_party/NIO_Lie_Events` |
| <https://github.com/i2Nav-WHU/LightweightLearnedInertialOdometer> | `e225ab8f3080c35c92481673b6af1d68692941f6` | GPL-3.0 | `third_party/LightweightLearnedInertialOdometer` |
| <https://github.com/nesl/tinyodom> | `7958781baeddf1b99a4f365fb54fb7bac5e96845` | BSD-3-Clause | `third_party/tinyodom`（Keras；导出用 keras-tcn 3.3.0 重建并与 TFLite 对拍） |
| <https://github.com/OmerTariq-KAIST/DeepILS-IoT-Journal-2025> | `4fd65aa9564075810dab494d43610857dc3e1d01` | **无 LICENSE**（文件头 All Rights Reserved） | `third_party/DeepILS-IoT-Journal-2025` |
| <https://github.com/superxslam/TartanIMU> | `8a8025000f85d2c6b0692b88e15c2c91e761f7d7` | Apache-2.0（HF 权重标签 MIT，但正文写仅限研究用途） | `third_party/TartanIMU` |
| <https://github.com/decargroup/DIVE> | `e982ddd0643c9d831de77a95467b282b07cceec1` | MIT | `third_party/DIVE` |
| <https://github.com/Air-IO/Air-IO> | `7214e8a6ddee95ca07a64f01f1fa3e2f8f60df73` | BSD-3-Clause | `third_party/Air-IO` |
| <https://github.com/jacklu333333/PedestrianDiffusion> | `0b6cc2d019397e390ce757335e01712304339c9b` | **AGPL-3.0**（内含 GPL 代码副本） | `third_party/PedestrianDiffusion` |
| <https://github.com/bingrao/ctin> | `1441c726811ac87283903471562023fd429e3a08` | MIT（README 徽章写 Apache-2.0） | 占位仓库，只有 README/LICENSE，未克隆 |
| <https://github.com/Minh-Son-Nguyen/iMoT> | `7f275702bd9b0dbdb60bb4d5dae802f8e9ab9dac` | MIT | 占位仓库（“coming soon”），未克隆 |

IONet、RIO、IONext、GNIO、VeloBins 没有公开代码（RIO 的 CVPR 论文链接到华为云 AI Gallery 页面，该页面需要 JS 渲染，内容未能核实；IONext 承诺审稿后发布；VeloBins 摘要称“录用后发布”；GNIO 未提供任何链接）。

**许可策略**：
- GPL/AGPL 仓库：只依据规格卡独立实现，不得复制代码，否则衍生文件受 copyleft 约束。
- 无 LICENSE 的仓库：默认保留全部权利；本仓库只保存规格卡与形状夹具（不含代码与权重），权重不得再分发。
- 夹具中的 `pretrained_golden`/`checkpoint_check` 只记录官方权重的 sha256、严格加载结果和少量推理输出，用于本地核对，不包含权重本身。

## 6. 排除项

| 项目 | 排除理由 |
|---|---|
| RepViT | 视觉骨干（面向图像的移动端 CNN/ViT），没有惯性里程计模型、任务或协议可复现 |
| CPUBone | 面向 CPU 推理的视觉骨干，不是惯性定位方法 |
| MoRPINet | 面向轮式/蛇形运动平台，运动模型不适用于行人 IPB 数据集 |
| TALOS-NIO | 没有论文、没有许可，既无可引用的规格来源，也不能合法再分发 |
| AI-IO、IMO、DIDO | 需要旋翼转速或推力等执行器输入，行人数据集没有这类信号 |

`velobins` 的官方配置同样需要旋翼转速/PWM，本来符合上面的排除理由；我们仍然登记它，是因为它的**分箱输出头**是我们研究线的直接竞品，
可以在行人骨干上作为对照实现。登记的是明确标注的 **IMU-only 变体**（去掉执行器支路，属于改变结构），论文的百分比收益不适用于该变体。

## 7. 未完成与待核实

- **论文全文不可得**：LLIO、DeepILS、DIVE 的论文全文在 IEEE 付费墙后，卡中的训练配方与报告数值标为“未核实”。其中 LLIO 状态为“部分”，拿到 PDF 后应补齐 §5、§7。
- **paper-only 的参数量与论文不符**：CTIN 推导值比论文少 14%，iMoT 只到论文值的约 49%，IONext 需在 v1/v2 两版描述之间取舍。这些卡片状态为“部分”，参数量不能作为忠实性锁定依据。
- **未下载核对的权重**：RoNIN（FRDR 记录当前无文件）、IMUNet、RNIN、PedestrianDiffusion 的预训练权重，以及 NIO 其余 3 个 TLIO 权重。
- **TinyOdom**：官方未公布 NAS 最终结构，卡片固定仓库 `model.cc` 中的 RoNIN 候选结构（未训练权重，已与 TFLite 输出对拍）。
- **DIVE**：IPB 配置把窗口改为 200 @ 200 Hz、输出改为 2 维，属于改变结构，参数量与官方 3.5 s @ 400 Hz 配置不同（12,367,110 → 7,516,420）。
- **GNIO**：注意力维度/头数与特征池化方式未给出，推导参数量 4,934,153（论文 4.90 M）依赖 §4.3 的假设；增强、权重衰减、划分未写明，按 TLIO 补齐并标注。
- **VeloBins**：编码器沿用 AI-IO 且未复述细节，只有 6,848 参数的分箱解码器可精确复现；箱中心是否均匀、速度目标取窗口哪个时刻、越界速度如何处理均未规定。
