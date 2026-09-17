# DIVE（`dive`）

> 规格卡版本：v1（2026-09-17）· fidelity：`official-code` · 夹具：`tests/fixtures/algorithms/dive.json`
> 优先级 P1：原始论文面向**四旋翼**，用于行人需要适配（见第 6 节）。

## 1. 元信息

| 项 | 值 |
|---|---|
| 论文 | *DIVE: Deep Inertial-Only Velocity Aided Estimation for Quadrotors*；A. Bajwa, C. C. Cossette, M. A. Shalaby, J. R. Forbes；IEEE RA-L 9(4):3728–3734，2024；DOI [10.1109/LRA.2024.3370006](https://doi.org/10.1109/LRA.2024.3370006)（无 arXiv 版本，正文未能在线核读，见第 7 节） |
| 官方仓库 | https://github.com/decargroup/DIVE @ `e982ddd0643c9d831de77a95467b282b07cceec1`（本地：`/workspace/webCodex/third_party/DIVE`） |
| 许可 | MIT（Copyright (c) 2023 Angad Bajwa）。宽松许可，重实现只需保留署名 |
| 框架 | PyTorch + Lightning（`lightning.pytorch`）；滤波依赖 `pymlg`（SO(3)/SE_2(3) 工具） |
| fidelity | official-code（网络、数据预处理、损失、EKF 均开源；未发布权重与划分文件） |
| 参考文件 | 网络 `network/resnet1d/resnet1d.py`；训练入口 `network/train.py` → `network/pl_trainer.py` → `network/modules.py`（`VelocityVectorRegressor`）；损失 `network/resnet1d/loss.py`；数据 `network/dido_preprocessor.py`、`network/preprocessing/data_segmenter.py`、`data/hdf5_loader.py`、`data/imu_preprocessing.py`；滤波 `modelling/quad.py`、`filtering/measurement_models.py`（`VelocityVector`）、`filtering/ekf.py`、`filtering/process_models.py`；运行脚本 `syn_pseudo/velocity_regressor_*.py` |

## 2. 任务（输入）

DIVE 回归**窗口末端时刻**的 3D 速度，表达在“窗口末端偏航去除后的重力对齐系”中。网络的 6 个输入通道**不是** `[gyro, acc]`，而是
`[φ_k (3), a_k (3)]`：

- `φ_k = Log(C_γᵀ · C_k)`：第 k 个样本的机体→局部重力对齐系姿态的旋转向量（so(3) 对数，弧度）。
  `C_k` 由窗口末端姿态 `C_end` 出发，用陀螺**向后积分**得到：`C_{k} = C_{k+1} · Exp(ω_k · Δt_k)ᵀ`，其中 `Δt_k = t_{k+1} − t_k`
  （`network/dido_preprocessor.py:134-147`）。`C_γ = R_z(γ)`，`γ = atan2(C_end[1,0], C_end[0,0])`，即 ZYX 欧拉角的偏航
  （`dido_preprocessor.py:118-127`）。因此窗口末端 `C_γᵀC_end` 只含横滚/俯仰。
- `a_k = C_γᵀ C_k · f_k − [0, 0, g]`：旋到同一局部系并**去除重力**的加速度（`g = scipy.constants.g = 9.80665`，世界系 z 轴向上，`dido_preprocessor.py:151-156`）。
- 陀螺本身**不作为通道输入**（`gyro_ga` 被计算但未使用，`dido_preprocessor.py:153`），它只通过 `φ_k` 间接进入。

| 项 | 官方 | IPB 映射 |
|---|---|---|
| 采样率 | 名义 400 Hz（`nominal_imu_frequency=400`，DIDO/Blackbird 数据） | 200 Hz |
| 窗口 | 训练默认 3.5 s = 1400 样本（`network/train.py:17`，`int(3.5/(1/400))=1400`）；EKF 脚本默认 0.5 s（Blackbird/EuRoC）、1 s（DIDO、VI-failure）、3 s（cloud）；可视化脚本 `visualizers/hyperparameter_study/01_hyperparams.py` 比较了 0.5/1.5/2.5/3.5 s | `window=200`（1 s，与 DIDO EKF 默认一致） |
| 训练步长 | `sampling_frequency=20 Hz` → `int((1/20)/(1/400)) = 20` 样本（0.05 s，`data_segmenter.py:30-34`，`Tensor.unfold`） | `stride=10`（0.05 s@200 Hz，时长一致） |
| 推理步长 | EKF 每 `update_frequency`（10 或 20 Hz）调用一次网络（`modelling/quad.py:69`） | `eval_stride=10`（20 Hz） |
| 坐标系 | 输入与目标均在“窗口末端偏航去除的重力对齐系” | 目标 `frame=gravity_yaw_local`；输入需在模型内部由机体系 IMU + 末端姿态构造（见第 6 节） |
| 姿态来源 | 训练：末端**真值**姿态 + 加噪陀螺向后积分；EKF：滤波器当前姿态估计 `x_prior` + 去偏陀螺（`measurement_models.py:391-405`） | `orientation=reference`（或 `device`），只使用**窗口末端**一个姿态 |
| 去重力 | 是（在局部系减 `[0,0,g]`） | 模型内部完成；视图 `remove_gravity=false` |
| 通道顺序 | `[φ_xyz, a_xyz]` | 由 `[gyro_xyz, acc_xyz]`（机体系）+ `q_end` 计算得到，不是置换 |
| 额外输入 | 窗口末端姿态 `C_end`（3×3） | **需要接口扩展**：`aux["q_end"]`，形状 `(B,4)`，`body_to_world_wxyz` |
| 输入归一化 | 无 | 无 |
| 目标 | 窗口**末端样本**速度 `v_end` 旋到局部系：`C_γᵀ v_end`（`dido_preprocessor.py:287,308`；`train_raw_velocity=True` 时不归一化） | `target=velocity_at_end`，`dims=2`（见第 6 节） |

训练时数据来自 DIDO 的 `data.hdf5`：`self_augment=True`（默认）时使用**真值**角速度 `gt_gyr` 与由真值加速度合成的比力
`C_kᵀ(a_world − [0,0,−g])`（`data/imu_preprocessing.py:69-76`），再叠加随机噪声与偏置（第 5 节），而不是真实 IMU 读数。
序列开头丢弃 `start_idx=50` 个样本。

## 3. 输出

- `vel`：`(B,3)`，窗口末端速度（m/s），局部重力对齐系。
- `logstd`：`(B,3)`，对角标准差的对数 `s`，`σ = exp(s)`，协方差 `Σ = diag(exp(2s))`（`loss.py:47-69 gen_cov_diag_only`）；两个头结构相同、参数独立。
- 官方轨迹：**不做网络直接积分**。EKF（SE_2(3) 位姿 + 陀螺/加计偏置，15 维误差态，默认左扰动）用 IMU 预积分传播，
  每 `1/update_frequency` 秒用网络输出作速度伪观测更新：观测模型 `h(x) = C_γ(C)ᵀ v`（`measurement_models.py:268-298`），
  观测噪声 `R = cov_scaling · diag(exp(2s))`，`cov_scaling` 默认 10（EuRoC 与 `vi_failure_vis` 脚本为 100）。
  首次凑满窗口时（`reinitialize_after_inertial_window=True`）用真值重置状态（`modelling/quad.py:141-214`）。
- IPB 中：用同一 `C_γ` 把 `vel` 旋回世界系，按第 6 节的时间戳约定积分；EKF 不在 v1 范围内。

## 4. 网络结构

TLIO 风格 ResNet1D（`network/resnet1d/resnet1d.py`），`BasicBlock1D`，`group_sizes=[d,d,d,d]`，官方 `d = residual_block_depth = 3`
（`network/train.py:26`，经 `pl_trainer.py:72-97` 传入；旧的非 Lightning 入口 `network/net.py:89-91` 用 `[2,2,2,2]`，不是当前入口）。
所有卷积 `bias=False`；BN 为 PyTorch 默认（`eps=1e-5, momentum=0.1`）。下表为 IPB 适配配置（输入 `(B,6,200)`，`dims=2`）：

| # | 层 | 参数（核/步长/填充/通道） | 归一化 | 激活 | dropout | 输出形状 |
|---|---|---|---|---|---|---|
| 0 | 输入 | `[φ, a]` | – | – | – | (B,6,200) |
| 1 | `input_block.0` Conv1d | k7 s2 p3，6→64 | – | – | – | (B,64,100) |
| 2 | `input_block.1-2` | – | BN(64) | ReLU | – | (B,64,100) |
| 3 | `input_block.3` MaxPool1d | k3 s2 p1 | – | – | – | (B,64,50) |
| 4 | group1：3×BasicBlock | 每块 conv3(s1,p1) 64→64 + conv3 64→64，恒等捷径 | BN | ReLU（块内一次、相加后一次） | – | (B,64,50) |
| 5 | group2：块1 | conv3 s2 64→128，conv3 128→128；捷径 conv1 s2 64→128 + BN | BN | ReLU | – | (B,128,25) |
| 6 | group2：块2-3 | conv3 128→128 ×2，恒等捷径 | BN | ReLU | – | (B,128,25) |
| 7 | group3：块1 | conv3 s2 128→256，conv3；捷径 conv1 s2 + BN | BN | ReLU | – | (B,256,13) |
| 8 | group3：块2-3 | 同上无下采样 | BN | ReLU | – | (B,256,13) |
| 9 | group4：块1 | conv3 s2 256→512，conv3；捷径 conv1 s2 + BN | BN | ReLU | – | (B,512,7) |
| 10 | group4：块2-3 | 同上无下采样 | BN | ReLU | – | (B,512,7) |
| 11a | `output_block1.prep1` Conv1d | k1，512→128，无偏置 | BN(128) | **无**（BN 后直接展平） | – | (B,128,7) |
| 12a | `output_block1.fc1` Linear | 128·7=896 → 512 | – | ReLU | 0.5 | (B,512) |
| 13a | `output_block1.fc2` Linear | 512→512 | – | ReLU | 0.5 | (B,512) |
| 14a | `output_block1.fc3` Linear | 512→2 | – | – | – | `vel` (B,2) |
| 11b–14b | `output_block2`（结构同 11a–14a，独立参数） | 同上 | 同上 | 同上 | 同上 | `logstd` (B,2) |

- FcBlock 的展平长度 `in_dim = window_samples // 32 + 1`（`resnet1d.py:7-15`；官方 1400 → 44，IPB 200 → 7）。
  这条公式对 200/1400 样本恰好等于最后一层特征长度；移植时应直接用实际特征长度并加断言。
- 参数量：
  - 官方配置（1400 样本，3D，d=3）：**12,367,110**；
  - IPB 适配（200 样本，2D，d=3）：**7,516,420**（夹具锁定；`input_block` 2,816，residual_groups 5,936,128
    [74,496 / 280,320 / 1,117,696 / 4,463,616]，每个头 788,738：prep1 65,536 + bn 256 + fc1 459,264 + fc2 262,656 + fc3 1,026）；
  - 其他参照：200 样本 3D 为 7,517,446；400 样本（400 Hz、1 s）3D 为 8,303,878；旧入口 `[2,2,2,2]`、1400 样本 3D 为 10,274,310。
- 权重初始化（`resnet1d.py:226-245`）：Conv1d `kaiming_normal_(mode="fan_out", nonlinearity="relu")`；BN weight=1、bias=0；
  Linear `normal_(0, 0.01)`、bias=0；`zero_init_residual=False`（该分支引用了未定义的 `Bottleneck1D`，但不会执行）。
- 前向：`x1 = output_block1(x)`（均值），`x2 = output_block2(x)`（`s`），无额外激活。

## 5. 损失与训练配方（official）

| 项 | 值 | 来源 |
|---|---|---|
| 损失 | `epoch < 10`：MSE `mean((v̂ − v)²)`（对 3 轴与 batch 求均值）；`epoch ≥ 10`：TLIO 式对角 NLL `mean((v̂ − v)² / (2·exp(2(s+1e-7))) + (s+1e-7))` | `loss.py:125-133,148-155,171-175`；`modules.py:88-91`（`VelocityVectorRegressor.training_step`） |
| 阶段切换 | 由 Lightning `self.current_epoch`（从 0 开始）决定，第 0–9 轮 MSE，第 10 轮起 NLL；MSE 阶段 `logstd` 头不在计算图中、无梯度，AdamW 跳过其更新 | `modules.py:91` |
| 优化器 | `torch.optim.AdamW(lr=1e-4)`，其余为 PyTorch 默认（**weight_decay=0.01**，betas=(0.9,0.999)） | `modules.py:123-124`；`train.py:25` |
| 学习率调度 | Lightning 路径**无**调度（旧 `net.py` 的 ReduceLROnPlateau(factor 0.1, patience 10) 不在当前入口中） | `pl_trainer.py` |
| batch | 32；训练集 shuffle，验证集不 shuffle | `train.py:29`；`pl_trainer.py:52-63` |
| epoch | 100（`max_epochs`） | `train.py:30` |
| 梯度裁剪 | 无 | `pl_trainer.py:115-119` |
| 精度 | Lightning 默认 32 位，`accelerator="gpu"` | 同上 |
| 增强 | 每次 `__getitem__` 重新采样（验证集同样加噪）：陀螺白噪声连续谱密度 `σ_g ~ U(1e-3, 2e-3]`、加计 `σ_a ~ U(6e-3, 2e-2]`，离散标准差 `σ/√Δt`（Δt=1/400）；常值偏置 `b_g ~ U(−0.01, 0.01]`、`b_a ~ U(−0.05, 0.05]`（逐轴，`0.01 − 0.02·rand`）；轴失准旋转被计算但**未施加**；无随机偏航（输入/目标本就去除偏航） | `dido_preprocessor.py:19-89` |
| 选模 | `ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)`；另存最新一轮 | `pl_trainer.py:102-113` |
| 数据 | 训练/验证：DIDO（`network/splits/train.txt`、`val_original_formatting.txt`，未随仓库发布） | `train.py:47-48` |

## 6. benchmark 适配（unified 配方）

| 改动 | 类型 | 理由 |
|---|---|---|
| 200 Hz、`window=200`（1 s），FcBlock `in_dim=7` | 改变结构（仅 fc1 输入宽度） | IPB 统一 200 Hz；官方窗口本身是超参（0.5–3.5 s 均有使用），1 s 与 DIDO EKF 默认一致；`in_dim` 按官方公式由窗口推出 |
| 接口扩展 `aux["q_end"]`（窗口末端姿态，`orientation=reference` 或 `device`），视图输入取 `frame=body` 的 `[gyro, acc]` | 不改变结构（需修订 DESIGN §4 的 `forward` 签名或 `InputSpec`） | DIVE 的输入通道依赖姿态历史，无法从 `(B,6,T)` 的 `[gyro, acc]` 单独恢复；只需要末端一个姿态，其余由陀螺向后积分，与官方完全一致 |
| 模型内部预处理：`γ = atan2(R[1,0], R[0,0])`；`C_T = R_z(γ)ᵀR_end`；向后积分 `C_k = C_{k+1} Exp(ω_k Δt)ᵀ`（Δt=1/200）；`φ_k = Log(C_k)`；`a_k = C_k f_k − [0,0,9.80665]`；拼成 `[φ, a]` | 不改变结构 | 等价复现 `generate_gravity_aligned_input`；Log 取主值（角度 ∈ [0, π]） |
| 目标：`frame=gravity_yaw_local`（末端偏航），`target=velocity_at_end` | 不改变结构 | 与官方标签一致；推理时用同一 `γ` 旋回世界系 |
| `dims=2`：两个头的 `fc3` 输出 3→2 | 改变结构（每头 −513 参数） | IPB 指标在水平面；局部系 z 轴与世界 z 一致，水平分量可独立监督；对角 NLL 按轴独立，2D 化不改变损失形式 |
| 时间戳：`velocity_at_end` 的速度应赋给**窗口末端**而非中心 | 协议说明 | DESIGN §5 的“窗口中心”约定针对 `avg_velocity`；若 v1 只支持中心赋值，须在结果中注明半窗时移 |
| 训练输入：用数据集真实 IMU（IPB 无真值 IMU），保留噪声/偏置注入作为增强（`augment: [imu_noise, imu_bias]`，同分布参数） | 不改变结构 | 官方 `self_augment` 依赖合成的无噪 IMU；行人数据集没有该字段 |
| unified：AdamW(1e-4, wd 0.01)、MSE→NLL 于第 10 轮切换保留；epoch/batch 按 IPB 统一预算 | 不改变结构 | 阶段切换属于忠实性底线 |
| 行人适用性 | 说明 | 四旋翼速度变化平滑、与姿态强相关；行人手持/口袋场景姿态与运动方向解耦，预期效果低于行人专用方法；作为“姿态显式编码”的跨平台对照 |

## 7. 官方报告数值

论文正文（IEEE Xplore）未能在线获取，表格数值**未核实**。可核实的只有摘要中的相对提升：

| 数据集 | 指标（定义） | 数值 | 来源 |
|---|---|---|---|
| 分布内测试集（DIDO） | 定位精度相对 SOTA（TLIO 类学习惯性里程计）提升 | 42% | 摘要（IEEE Xplore / Semantic Scholar 页面） |
| 分布外测试集（Blackbird） | 同上 | 22% | 摘要 |
| VIO 失效场景 | 相对纯 IMU 递推的提升 | 43% | 摘要 |

仓库中的评测为相对位姿平移误差（`visualizers/hyperparameter_study/01_hyperparams.py`，“Relative Pose Translational Error (m)”），具体数值需从论文表格补录。

## 8. 忠实性测试建议

- [ ] IPB 配置参数量 == 7,516,420；`param_shapes` 逐项一致（夹具注册顺序：input_block → residual_groups → output_block1 → output_block2）
- [ ] 官方配置（1400 样本、3D）参数量 == 12,367,110
- [ ] 输出：`vel (B,2)`、`logstd (B,2)`；两个头参数不共享
- [ ] 预处理：静止、水平、偏航 γ 的合成窗口（ω=0，f=R_endᵀ[0,0,g]）→ `φ ≡ 0`、`a ≡ 0`（误差 < 1e-5）
- [ ] 偏航不变性：对 `q_end` 左乘任意 `R_z(θ)`（机体系 IMU 不变）→ 预处理输出 `[φ, a]` 完全不变（< 1e-6），因此 `vel` 不变
- [ ] 向后积分：用恒定 ω 构造 `R_k = R_end · Exp(−ω(T−1−k)Δt)`，预处理得到的 `C_k` 与解析值一致
- [ ] 损失：`epoch=9` 时 `logstd` 头梯度为 None/0，`epoch=10` 时非零；NLL 在 `s=0`、残差 r 时等于 `mean(r²/2 + 1e-7)`（考虑 1e-7 偏移）
- [ ] 初始化：Linear 权重标准差 ≈ 0.01，偏置为 0

## 9. 预训练权重

仓库与 README 均未提供预训练权重或下载链接；`syn_pseudo/*.py` 中的 `--model_path` 指向作者本机的 Lightning checkpoint（如 `final_velReg_augment_1_best_val_loss.ckpt`）。DIDO 训练/验证列表文件也未发布。

## 10. 官方实现的坑与未决问题

1. **通道语义**：6 个输入通道是 `[Log(C_g0_b), 去重力局部系加速度]`，不是陀螺+加计；照搬“`[gyro, acc]` 输入”会得到另一个模型（`dido_preprocessor.py:158-160`）。
2. `augment_data` 的 `else` 分支引用未定义的 `sigma_acc_ct/sigma_gyro_ct/dt`，`--no-self_augment` 训练会直接 `NameError`（`dido_preprocessor.py:77-87`）；因此官方实际只能用真值合成 IMU + 噪声训练。
3. 轴失准旋转 `C_ax_misalignment`、初始姿态误差 `delta_phi` 与参数 `--initial_orientation_error` 均被计算/声明但**未使用**（`dido_preprocessor.py:67-71,131-132`；`train.py:38`）。
4. 验证集同样逐次随机加噪，`val_loss` 不确定；且第 0–9 轮是 MSE、之后是 NLL（可为负），`ModelCheckpoint` 跨阶段比较两种量纲，最优模型几乎必然落在 NLL 阶段。
5. AdamW 默认 `weight_decay=0.01` 实际生效（配置中未显式写出）。
6. 训练默认窗口 3.5 s，而 EKF 脚本默认 0.5/1/3 s；网络 `in_dim` 与窗口绑定，加载 checkpoint 时窗口必须与训练一致。EKF 的 IMU 缓冲最多保留 `range+1` 个样本（`modelling/quad.py:97-100`），对 400 样本窗口实际喂入 401 个样本（特征长度仍为 13，不报错）。
7. `filtering/measurement_models.py:12` 导入了 `model_resnet_tlio.ResNet1D`，但实际加载走 `network.modules.VelocityVectorRegressor`（`resnet1d.py` 版本），前者未被使用。
8. 训练脚本 `net.py` 在每个 batch 调用 `torchviz.make_dot`（仅旧入口），且导入不存在的 `metrics` 包路径；以 `train.py` + Lightning 为准。
9. 偏航提取用 `atan2(C[1,0], C[0,0])`，俯仰接近 ±90° 时不稳定（四旋翼与手持设备均可能出现），移植时保留同一定义并记录。
