# DeepILS（`deepils`）

> 规格卡版本：v1（2026-09-17）· fidelity：`official-code` · 夹具：`tests/fixtures/algorithms/deepils.json`

## 1. 元信息

| 项 | 值 |
|---|---|
| 论文 | *DeepILS: Toward Accurate Domain-Invariant AIoT-Enabled Inertial Localization System*；Omer Tariq, Muhammad Bilal Akram Dastagir, Muhammad Bilal, Dongsoo Han；IEEE Internet of Things Journal, vol. 12, no. 11, pp. 17153–17168, 2025；DOI 10.1109/JIOT.2025.3538938；<https://ieeexplore.ieee.org/document/10873819>（README 中标题写作 “Towards Accurate Domain Invariant …”） |
| 官方仓库 | <https://github.com/OmerTariq-KAIST/DeepILS-IoT-Journal-2025> @ `4fd65aa9564075810dab494d43610857dc3e1d01`（本地：`/workspace/webCodex/third_party/DeepILS-IoT-Journal-2025`） |
| 许可 | **无 LICENSE 文件**；`src/models/ResNet1D_dws.py`、`src/tflite_evaluation.py` 文件头写有 “All Rights Reserved / Unauthorized copying or redistribution … is strictly prohibited”。**许可风险高**：只能按本卡洁净室重实现，不得复制代码，不得再分发官方权重 |
| 框架 | PyTorch（`DeepILS.yml`：CUDA 11.8 conda 环境）；另有 ONNX → TFLite 导出脚本 |
| fidelity | official-code（结构、损失、配方均取自官方代码；论文全文为付费墙，未能核对） |
| 参考文件 | `src/models/DeepILS.py`（网络）、`src/main.py`（构建、训练、测试）、`src/preprocess/data_processor.py`（数据与增强）、`src/metric.py`（ATE/RTE）、`dataset/preprocess_dataset.py`（KIOD/INAIOD 预处理，源自 IMUNet）、`results/*/checkpoint_*.pt`（发布权重） |

DeepILS 本质上是 **RoNIN-ResNet18 的轻量变体**：3×3 卷积全部换成深度可分离卷积（depthwise k=3 + pointwise 1×1），每个残差块末尾串接 CBAM 式的通道注意力与时间（“空间”）注意力，输入层卷积核 7→5。数据管线、目标定义、测试流程与 IMUNet 仓库（RoNIN 修改版）一致。

## 2. 任务（输入）

| 项 | 官方 | IPB 映射 |
|---|---|---|
| 采样率 | 200 Hz（RoNIN/RIDI 原生；KIOD/INAIOD/IMUNet 由 `preprocess_dataset.py` 插值到 200 Hz） | 200 Hz |
| 窗口 | 200 样本 / 1 s（`--window_size 200`；FC 头要求时间长度为 7，仅 T∈[191, 222] 可运行） | `window=200` |
| 训练步长 | 10 样本（`--step_size 10`），另加随机平移 `random_shift = step_size//2 = 5`（`main.py:116`） | `stride=10`（随机平移见第 5 节增强） |
| 推理步长 | 10 样本 | `eval_stride=10` |
| 坐标系 | 重力对齐世界系：`R(q)·gyro`、`R(q)·acc`（`data_processor.py:130-133, 269-273`） | `frame=gravity_world` |
| 姿态来源 | 训练：RoNIN 用 `select_orientation_source(max_ori_error=20)`（game_rv 误差<20° 用 game_rv，否则取 gyro 积分/game_rv/EKF 中误差最小者）；测试：RoNIN 强制 `grv_only=True`（game_rv）。RIDI/IMUNet/KIOD/INAIOD 训练与测试都用 game_rv，并用首帧 Tango 姿态做常值对齐 `init_rotor = q_tango[0]·q_rv[0]^{-1}` | 官方测试为 `orientation=device`；unified 配方默认 `reference`，需单独报告 `device` 结果 |
| 去重力 | 否（加计为含重力比力） | `remove_gravity=false` |
| 通道顺序 | `[glob_gyro_xyz, glob_acce_xyz]`（`data_processor.py:137`） | 与 IPB `[gyro, acc]` 相同，无需置换 |
| 额外输入 | 无（磁力计虽被预处理进 csv，但不进入网络） | 无 |
| 输入归一化 | 无（代码中逐通道归一化被注释掉，`data_processor.py:521-522`） | 无 |

窗口/目标的精确定义（`StridedSequenceDataset`，`interval = window_size`）：

- 帧索引 `f` 取 `range(0, N-200, 10)`；特征 `feat = features[f : f+200]`（200 帧）；
- 目标 `targ = (p[f+200] − p[f])_{xy} / (t[f+200] − t[f])`，即跨 200 个采样间隔的水平平均速度；
- IPB `avg_velocity` 为 `(p[s+T−1] − p[s]) / ((T−1)·dt)`，与官方差 1 个采样间隔（0.5%），属于协议差异，**不改变结构**。

## 3. 输出

- `(B, 2)`：重力对齐世界系水平速度 `(vx, vy)`，单位 m/s；无协方差/不确定度输出。
- 官方轨迹重建（`main.py:296-310`）：测试时 `step=10` 滑窗，`dts = mean(t[ind[1:]] − t[ind[:-1]])`（≈0.05 s），`pos[k] = p0 + Σ v_k·dts`，时间戳取**窗口起点** `t[ind]`，首尾各补一个常值点后对全序列线性插值。IPB §5 明令禁止“时间戳赋给窗口起点”，因此 IPB 复现时一律用我们的预测器（窗口中心 + 梯形积分），官方做法只用于解释数值差异。

## 4. 网络结构

`DeepILS(num_inputs=6, num_outputs=2, block=BasicBlock1D, group_sizes=[2,2,2,2], base_plane=64, output_block=FCOutputModule, kernel_size=3, fc_dim=512, in_dim=7, dropout=0.5, trans_planes=128)`（`main.py:43, 51-52`）。

**残差块 `DWBlock(in, out, stride)`**（`DeepILS.py:15-19, 54-95`），按执行顺序：

1. `conv1` = depthwise `Conv1d(in, in, k=3, stride=stride, pad=1, groups=in, bias=False)` → pointwise `Conv1d(in, out, k=1, bias=False)`；
2. `BatchNorm1d(out)` → `ReLU`；
3. `conv2` = depthwise `Conv1d(out, out, k=3, stride=1, pad=1, groups=out, bias=False)` → pointwise `Conv1d(out, out, k=1, bias=False)`；
4. `BatchNorm1d(out)`（其后**无** ReLU）；
5. 通道注意力 `CA`：`w = σ( MLP(AvgPool1d→1) + MLP(MaxPool1d→1) )`，`MLP` 为**共享**的 `Conv1d(out, out//16, 1, bias=False) → ReLU → Conv1d(out//16, out, 1, bias=False)`；`y = y · w`（缩减比恒为 16，构造参数 `ratio` 被忽略）；
6. 时间注意力 `SA`：`s = σ( Conv1d(2, 1, k=7, pad=3, bias=False)( cat[mean_c(y), max_c(y)] ) )`；`y = y · s`；
7. 捷径：`stride≠1` 或 `in≠out` 时为 `Conv1d(in, out, 1, stride, bias=False) → BatchNorm1d(out)`，否则恒等；
8. `out = ReLU(y + shortcut)`。

逐层表（输入 `(B, 6, 200)`，无 dropout 的层 dropout 列留空）：

| # | 层 | 参数（核/步长/填充/通道） | 归一化 | 激活 | dropout | 输出形状 | 参数量 |
|---|---|---|---|---|---|---|---|
| 0 | 输入 | `[gyro, acc]` 世界系 | | | | (B, 6, 200) | |
| 1 | Conv1d | k=5, s=2, **p=3**, 6→64, bias=False | BN(64) | ReLU | | (B, 64, 101) | 1 920 + 128 |
| 2 | MaxPool1d | k=3, s=2, p=1 | | | | (B, 64, 51) | 0 |
| 3 | group0.block0 | DWBlock(64→64, s=1)，恒等捷径，CA 隐层 4 | BN×2 | ReLU | | (B, 64, 51) | 9 358 |
| 4 | group0.block1 | DWBlock(64→64, s=1) | BN×2 | ReLU | | (B, 64, 51) | 9 358 |
| 5 | group1.block0 | DWBlock(64→128, s=2)，1×1 s=2 捷径+BN，CA 隐层 8 | BN×3 | ReLU | | (B, 128, 26) | 36 174 |
| 6 | group1.block1 | DWBlock(128→128, s=1) | BN×2 | ReLU | | (B, 128, 26) | 36 110 |
| 7 | group2.block0 | DWBlock(128→256, s=2)，捷径，CA 隐层 16 | BN×3 | ReLU | | (B, 256, 13) | 141 966 |
| 8 | group2.block1 | DWBlock(256→256, s=1) | BN×2 | ReLU | | (B, 256, 13) | 141 838 |
| 9 | group3.block0 | DWBlock(256→512, s=2)，捷径，CA 隐层 32 | BN×3 | ReLU | | (B, 512, 7) | 562 446 |
| 10 | group3.block1 | DWBlock(512→512, s=1) | BN×2 | ReLU | | (B, 512, 7) | 562 190 |
| 11 | transition | Conv1d k=1, 512→128, bias=False | BN(128) | 无 | | (B, 128, 7) | 65 792 |
| 12 | flatten | `view(B, −1)`（通道优先展平） | | | | (B, 896) | 0 |
| 13 | Linear | 896→512 | | ReLU | Dropout(0.5) | (B, 512) | 459 264 |
| 14 | Linear | 512→512 | | ReLU | Dropout(0.5) | (B, 512) | 262 656 |
| 15 | Linear | 512→2 | | | | (B, 2) | 1 026 |

- 以 group1.block0 为例的参数形状（注册顺序）：`conv1.0 [64,1,3]`、`conv1.1 [128,64,1]`、`bn1 [128]×2`、`conv2.0 [128,1,3]`、`conv2.1 [128,128,1]`、`bn2 [128]×2`、`ca.fc.0 [8,128,1]`、`ca.fc.2 [128,8,1]`、`sa.conv1 [1,2,7]`、`downsample.0 [128,64,1]`、`downsample.1 [128]×2`。完整列表见夹具 `param_shapes`（共 109 个张量；BN 的 running 统计量另有 63 个 buffer）。
- 参数总量：官方配置 **2 290 226**（全部可训练）；benchmark 配置相同（夹具锁定）。与官方发布的 `results/{ronin,ridi,imunet,kiod,inaiod}/checkpoint_best.pt` 及 7 个 OxIOD 权重 `strict=True` 加载一致；TinyIO（arXiv 2507.15293）Table I 报告 DeepILS 为 2.291 M 参数、15.287 M FLOPs，吻合。
- 权重初始化（`DeepILS.py:261-270`）：所有 `Conv1d`（含注意力内卷积）`kaiming_normal_(mode="fan_out", nonlinearity="relu")`；`BatchNorm1d` weight=1、bias=0；`Linear` weight ~ N(0, 0.01)、bias=0；`zero_init_residual=False`。
- 注意事项：`conv_dw` 的 `dilation` 参数未使用；输入层 k=5 却用 p=3（非对称“same”），必须照抄以保证长度 101/51/26/13/7；BN 使用 PyTorch 默认 `eps=1e-5, momentum=0.1`。

## 5. 损失与训练配方（official）

| 项 | 值 | 来源 |
|---|---|---|
| 损失 | `L = MSE(pred, targ) + mean(|pred − targ|)`，两项权重均为 1，均为对 batch 与 2 个分量的均值；日志中把 L1 项称为 “Differential Loss” | `main.py:226-231` |
| 优化器 | Adam，lr=1e-4，betas/eps 为默认，**无权重衰减** | `main.py:199, 564` |
| 学习率调度 | `ReduceLROnPlateau(factor=0.1, patience=10, eps=1e-12)`，每 epoch 以 **val MSE**（两分量均值）为指标 step | `main.py:200, 279` |
| batch | 训练 128（`shuffle=True`）；验证 512 | `main.py:175, 183, 565` |
| epoch | 默认 200（`--epochs 200`）；发布权重实际停在 RoNIN 63、RIDI 34、IMUNet 184、KIOD 192、INAIOD 198（checkpoint 中的 `epoch` 字段） | `main.py:566`；`results/*/checkpoint_best.pt` |
| 权重衰减 | 0 | |
| 梯度裁剪 | 无 | |
| AMP | 无 | |
| 增强 | ① `RandomHoriRotate(2π)`：每个样本抽 `θ~U[0, 2π)`，对 gyro 的 xy、acc 的 xy、目标 xy 同时左乘 2D 旋转（z 分量不变）；② 帧索引随机平移 `f += randrange(−5, 5)` 后 `f = max(200, min(f, N_targ−1))`；验证/测试不增强 | `data_processor.py:37-51, 516-518`；`main.py:114-118` |
| 阶段切换 | 无 | |
| 选模/早停 | 无早停；每个 epoch 若 **val 平均 L1**（两分量均值）创新低则保存 `checkpoint_best.pt`；训练结束另存 `checkpoint_latest.pt`（OxIOD 测试用 latest） | `main.py:281-290, 694` |
| 数据划分 | RoNIN：`list_train.txt` / `list_val.txt` / seen+unseen 测试；RIDI、IMUNet、KIOD、INAIOD：**val 列表直接用测试列表**；KIOD/INAIOD 的 train 列表与 test 列表**完全相同**；OxIOD：`train/` 训练，`validation/` 同时作 val 与 test | `main.py:599-636, 641-697`；`dataset/KIOD/list_*.txt`、`dataset/INAIOD/list_*.txt` |

## 6. benchmark 适配（unified 配方）

| 改动 | 类型 | 理由 |
|---|---|---|
| 输入 `(B,6,200)`，`frame=gravity_world`、`dims=2`、`target=avg_velocity`、200 Hz | 不改变结构 | 官方设定本来就与 IPB 默认一致，参数量不变（2 290 226） |
| 目标跨度由 200 个采样间隔改为 IPB 的 199 个 | 不改变结构 | IPB 统一目标定义（DESIGN §3），差 0.5% 时长，属协议效应，可由 oracle 轨迹分离 |
| 轨迹重建改用 IPB 预测器（窗口中心时间戳 + 梯形积分 + 首尾常值外推） | 不改变结构 | 官方以窗口起点为时间戳，半窗错位（DESIGN §5 禁止） |
| 姿态：unified 默认 `orientation=reference`，另报 `device` | 不改变结构 | IPB 默认协议；官方测试为 device（game_rv），两者都要可复现 |
| 选模依据改为 val ATE；划分使用 IPB 官方划分/分组 val | 不改变结构 | 官方以测试集作 val 选模（RIDI/IMUNet/KIOD/INAIOD），违反诚实协议 |
| 随机平移下限 `max(200, ·)` 不复刻 | 不改变结构 | 官方钳位导致序列前 200 帧的窗口被折叠到第 200 帧，属实现瑕疵；unified 用 IPB 的 `time_shift` 增强（±5 样本、越界丢弃） |
| `official` 配方保留：MSE+L1 损失、Adam 1e-4、无衰减、Plateau(0.1, 10) on val MSE、batch 128、200 epoch、随机水平旋转 | 不改变结构 | 忠实性底线 |
| 若需其他窗口长度：FC 输入 `128 × L_out` 随 T 变化 | 改变结构（不建议） | 仅 T∈[191,222] 与官方参数量一致；其他 T 必须把 `in_dim` 设为 `L_out(T)` 并重新锁定参数量 |

## 7. 官方报告数值

论文全文（IEEE 付费墙、ResearchGate 403）未能读取，**原文表格数值未核实**。README/摘要只给出定性结论：在 4 个公开基准 + 2 个自采数据集上“比 SOTA 定位精度提高 70%”。

可核对的数值来源：

| 数据集 | 指标（定义） | 数值 | 来源 |
|---|---|---|---|
| KIOD（8 条序列，**与训练集相同**） | ATE / RTE（官方 `metric.py`，见下） | 01: 1.727/2.443；02: 0.947/1.783；03: 1.669/1.811；04: 1.806/2.530；05: 1.941/2.697；06: 1.991/2.603；07: 1.992/2.067；08: 1.980/2.643（m） | 仓库内 `src/output/Test_out/KIODDeepILS/losses.csv`（非论文表格，且测试=训练） |
| RoNIN / RIDI / OxIOD / IMUNet | ATE / RTE（m） | 5.80/3.90；1.89/2.36；3.29/1.23；7.86/5.59 | **第三方**：TinyIO（arXiv 2507.15293）Table I，非 DeepILS 原文，未说明是否复跑 |
| — | Params / FLOPs / 峰值显存 | 2.291 M / 15.287 M / 70.979 MB | 同上（第三方） |

官方指标定义（与 RoNIN 仓库相同，`metric.py:3-61`）：`ATE = sqrt(mean((p̂ − p)²))`，均值取遍 **N×2 个元素**，因此数值等于 IPB ATE（`sqrt(mean‖·‖²)`）的 `1/√2`；RTE 取 Δ = 12000 帧（200 Hz 下 60 s），短于 60 s 的序列用全长端点漂移按 `12000/N` 缩放，同样按元素求均值（`1/√2` 关系同样成立）。与 IPB 数值比较前必须乘 `√2`。

## 8. 忠实性测试建议

- [ ] 参数量 == 夹具 `total_params`（2 290 226），`trainable_params` 相同
- [ ] `param_shapes` 109 项逐项一致（注册顺序：input conv/BN → 4 组×2 块（每块 conv1.dw, conv1.pw, bn1, conv2.dw, conv2.pw, bn2, ca.fc.0, ca.fc.2, sa.conv1, [downsample.conv, downsample.bn]）→ transition conv/BN → fc×3）
- [ ] 输出形状：`(B,6,200) → {"vel": (B,2)}`；中间形状 101/51/51/26/13/7（可用 forward hook 断言）
- [ ] T∉[191,222] 时应抛出形状错误（或模型在构造时校验 `window`）
- [ ] 注意力数学：给定 `y`，`CA(y)` 形状 `(B,C,1)` 且值域 (0,1)；`SA(y)` 形状 `(B,1,L)`；将 `sa.conv1`、`ca.fc.*` 权重置零后 CA/SA 输出恒为 0.5
- [ ] 损失：`loss(pred, targ) == mse + l1` 的数值单元测试（手算小例子）
- [ ] 增强：`random_yaw` 后 `‖acc_xy‖`、`‖gyro_xy‖`、`‖v_xy‖` 不变，z 分量不变
- [ ] 初始化：Linear 权重标准差 ≈ 0.01，偏置为 0；BN weight=1
- [ ] （可选、仅本地）把官方 `checkpoint_best.pt` 按夹具 `param_names` 映射加载到移植模型，对同一随机输入输出一致（eval 模式，atol 1e-5）；权重不得入库

## 9. 预训练权重

仓库内直接提供（无许可声明，**仅限本地核对，不得再分发**）：`results/{ronin,ridi,imunet,kiod,inaiod}/checkpoint_best.pt`（`model_state_dict` + `optimizer_state_dict` + `epoch`），`results/oxiod/<场景>/checkpoint_diff_loss.pt`（handbag、handheld、large scale、pocket、running、slow walking、trolley 七个场景各一），以及 `onnx_outputs/model.onnx`（9.19 MB，按导出代码输入为 `(1,6,200)`，对应数据集不明；未安装 onnx 包，未逐张量核对）。全部权重均可 `strict=True` 加载到夹具结构。另有 Android App（Google Drive 链接，README）。

## 10. 官方实现的坑与未决问题

1. **许可**：无 LICENSE 且文件头声明保留所有权利（见第 1 节）——只能做洁净室重实现，报告中注明。
2. **测试集泄漏**：RIDI、IMUNet、KIOD、INAIOD 训练时 `val_list` 即测试列表，并以其 L1 选模（`main.py:607, 614, 621, 628, 281-285`）；KIOD/INAIOD 的 `list_train.txt` 与 `list_test.txt` 完全相同；OxIOD 的 val 与 test 都是 `validation/`。官方数值（尤其 KIOD/INAIOD/RIDI/IMUNet）不能当作泛化结果引用。
3. **指标尺度**：官方 ATE/RTE 按元素求均值，比 IPB 定义小 `√2` 倍（`metric.py:17, 43`）。
4. **半窗时间错位**：轨迹积分时间戳取窗口起点（`main.py:302-308`）。
5. **OxIOD 读取器有误**：`quaternion.from_float_array` 需要 `[w,x,y,z]`，代码传入 `[rv_x, rv_y, rv_z, rv_w]`；把 `[gyro, 0]`、`[acce, 0]` 当作纯四元数，实际把 `gx`/`ax` 放进了实部（`data_processor.py:329, 338-339`），导致 OxIOD 特征旋转错误；且 `target_dim=3` 与实际 2 维目标不符。OxIOD 权重文件名 `checkpoint_diff_loss.pt` 与 `main.py` 保存名不同，说明发布权重来自另一版本代码。
6. **FC 头只支持窗口 200**：`train()` 使用默认 `in_dim=7`，只有 `test_sequence()` 会按 `window//32+1` 重设（`main.py:43, 345`），改窗口会在训练时直接报错。`get_model(arch)` 忽略其参数，读全局 `args`（`main.py:47-49`）。
7. **与新版 PyTorch 不兼容**：`ReduceLROnPlateau(..., verbose=True)` 在 torch 2.11 上抛 `TypeError`（`main.py:200`，已实测）。`val_list` 为 `None` 时 `val_loader` 未定义（`main.py:255`）。
8. **随机平移钳位**：`max(window_size, …)` 使前 200 帧起点的样本都被映射到 200（`data_processor.py:518`），并非对称平移。
9. **通道注意力缩减比写死为 16**（`DeepILS.py:26-28`），`ratio` 形参无效；512 通道时隐层 32、64 通道时隐层 4。
10. `data_processor.py` 注释称“与 RoNIN 完全相同”，实际是 IMUNet 的修改版（新增 `ProposedSequence`、RIDI/OxIOD 读取器，RIDI 用 game_rv + 首帧 Tango 对齐）；RoNIN 训练姿态按 `max_ori_error=20` 选择来源，测试强制 game_rv——训练/测试姿态来源不同。
11. 论文正文（结构图中的层数、注意力位置、训练超参数）未能核对；本卡以代码为准。若日后取得论文，需要复核：是否在论文中报告了 KIOD/INAIOD 的训练/测试划分，以及“70%”提升的基线与指标定义。
