# PDR：步检测 + Weinberg 步长 + 设备航向（`pdr`）

> 规格卡版本：v1（2026-09-17）· fidelity：`definition`（经典非学习基线，本卡即定义）· 夹具：无（没有神经网络参数）

## 1. 元信息

| 项 | 值 |
|---|---|
| 方法来源 | 行人航位推算（Pedestrian Dead Reckoning）的经典三段式：步检测 → 步长估计 → 航向。步长模型：H. Weinberg, *Using the ADXL202 in Pedometer and Personal Navigation Applications*, Analog Devices Application Note AN-602, 2002（`L = K·(a_max − a_min)^{1/4}`）。综述：R. Harle, *A Survey of Indoor Inertial Positioning Systems for Pedestrians*, IEEE Communications Surveys & Tutorials 15(3):1281–1293, 2013；A. R. Jiménez et al., *A comparison of pedestrian dead-reckoning algorithms using a low-cost MEMS IMU*, IEEE WISP 2009, pp. 37–42 |
| 官方仓库 | 无（IPB 自定义的参考实现规格） |
| 许可 | 不适用（方法为公知技术，本卡为 IPB 原创定义） |
| 框架 | numpy / scipy（`scipy.signal.butter`、`filtfilt`、`find_peaks`），不依赖 torch |
| fidelity | `definition` |
| 目的 | 给学习方法提供“不学习也能做到多少”的下界；同时暴露“设备朝向 ≠ 行走方向”在各数据集上的影响 |

## 2. 任务（输入）

PDR 是**序列级**方法：步检测需要跨窗口的连续滤波与峰值检测，不能逐窗口独立运行。它在整条序列上检测步，再把步汇总到 IPB 的窗口网格上（见第 3 节），从而与所有模型共用 §5 轨迹重建。

| 项 | 定义 | IPB 映射 |
|---|---|---|
| 采样率 | 200 Hz | 200 Hz |
| 窗口 | 只用于输出汇总：与被比较模型相同 | `window=200` |
| 训练步长 | 只用于标定（第 5 节） | `stride=10` |
| 推理步长 | 与被比较模型相同 | `eval_stride=10` |
| 坐标系 | 比力取 `gravity_world` 视图的加计通道（索引 3–5），其范数与竖直分量都与偏航无关 | `frame=gravity_world` |
| 姿态来源 | 航向必须来自姿态。`device`：`imu/orientation`（按 DESIGN §3 补充约定，只在首个有效样本处用参考姿态对齐常值偏航）；`reference`：`pose/orientation`（“理想姿态 PDR”） | 跟随视图 `orientation`；主表建议两种都报 |
| 去重力 | 否（内部自行减 `g₀`） | `remove_gravity=false` |
| 通道 | 只用加计 `imu[:, 3:6]`；陀螺不直接使用（只通过姿态间接使用） | `[gyro, acc]` 中取后 3 个 |
| 额外输入 | 每个样本的姿态四元数 `q_wb`（`body_to_world_wxyz`，(N,4)），来源同上；序列的 `body_frame` 根属性（决定前向轴） | 视图需要提供 `orientation` 张量（第 6 节） |
| 有效性 | 只在 `valid/imu` 连续为真的区段内滤波与检测；区段之间不连通 | — |

## 3. 输出

- 每个 IPB 窗口 `[s, s+T)` 的水平平均速度 `v_k ∈ R²`（世界系，m/s），**不含**不确定度。
- 定义：设 `t_s = timestamp[s]`，`t_e = timestamp[s+T−1]`，
  `v_k = ( Σ_{i: t_i ∈ (t_s, t_e]} L_i · [cos ψ_i, sin ψ_i] ) / ((T−1)·dt)`，
  其中 `t_i`、`L_i`、`ψ_i` 为第 i 步的时间、步长、航向（第 4 节）。分母与 IPB `avg_velocity` 的定义一致。
- 轨迹：把 `v_k` 交给 IPB §5 统一重建（窗口中心时间戳、首尾常值外推、梯形积分）。在滑窗步长 `eval_stride` 下每一步大约被 `(T−1)/eval_stride` 个窗口计入，积分后的总位移在期望上等于 `Σ L_i`（单步会有约 ±1/19 的计数抖动，平均后抵消）。
- 诊断模式 `native`（不进主表）：`p_i = p_{i−1} + L_i·[cos ψ_i, sin ψ_i]`，在步时刻更新位置，用于检查 §5 重建带来的差异。

## 4. 网络结构（不适用 → 算法定义）

没有可学习网络。流水线与默认参数如下（**除 K、δ 外全部固定，不在 test 上调**；若要调，只能在 val 上调并记录）：

| # | 步骤 | 定义 | 默认值 |
|---|---|---|---|
| 1 | 标量信号 | `signal=norm`：`a(t) = ‖f_w(t)‖ − g₀`；`signal=vertical`：`a(t) = f_w,z(t) − g₀`。`f_w` 为世界系比力 | `norm`，`g₀ = 9.80665 m/s²` |
| 2 | 低通 | 4 阶 Butterworth 低通，截止 `f_c`，**零相位** `filtfilt`（对每个有效区段整体滤波；离线评测允许非因果）。选项 `causal=true` 改用 `lfilter`（会引入群延迟） | `f_c = 3.0 Hz`，`order = 4` |
| 3 | 峰值检测 | `scipy.signal.find_peaks(s, height=h_min, prominence=p_min, distance=round(Δ_min·fs))`，每个峰值即一步，`t_i = timestamp[peak_i]` | `h_min = 1.0 m/s²`，`p_min = 0.5 m/s²`，`Δ_min = 0.3 s`（60 样本，步频上限 3.33 Hz） |
| 4 | 步区间 | 第 i 步的区间 `I_i = (max(peak_{i−1}, peak_i − W), peak_i]`；首步 `I_1 = [max(seg_start, peak_1 − W), peak_1]` | `W = 1.0 s`（200 样本） |
| 5 | 振幅 | `a_max,i = max_{I_i} s`，`a_min,i = min_{I_i} s` | — |
| 6 | 步长（Weinberg） | `L_i = K · (a_max,i − a_min,i)^{1/4}` | K 由第 5 节标定；不设默认值 |
| 7 | 前向轴 | 按序列 `body_frame` 查表：`android_device`（x 右、y 指向机顶、z 出屏）主轴 `e_fwd = +y`，备用轴 `e_alt = −z`（竖持拍照时机背朝前）。其他 `body_frame` 必须在数据集配置中显式给出，否则报错 | 见左 |
| 8 | 步航向 | `u(t) = R(q_wb(t))·e_fwd`，取水平分量 `u_xy`；`ū = Σ_{t∈I_i} u_xy(t)`；若 `‖ū‖ < 0.2·|I_i|`（机顶与竖直夹角 < 约 11.5°，水平投影不可靠）则改用 `e_alt` 重算；`ψ_i = atan2(ū_y, ū_x) + δ` | 阈值 0.2；δ 由第 5 节标定 |
| 9 | 汇总到窗口 | 第 3 节公式 | — |

说明：
- 对平放在手上、机顶朝前的手机，`R = R_z(yaw)` 时 `u = [−sin yaw, cos yaw, 0]`，所以航向 = `yaw + 90°`（Android 机体系的前向是 +y，而偏航从 x 轴量起）。
- 已知局限：口袋、包、摆臂、打电话等姿态下“设备前向轴”与行走方向之间的夹角既不是 0，也不恒定。单一常值 δ 只能修正数据集平均偏差，这正是该基线要暴露的误差来源。**不得**按 `placement` 等元数据分别标定 δ（测试时不应使用这些信息）。
- 可学习参数：0；标定标量：2 个（K、δ）。

## 5. 损失与训练配方（不适用 → 标定流程，只用 train 划分）

| 项 | 定义 |
|---|---|
| K（步长系数） | 对 train 划分每条序列，按时间切成互不重叠、全部样本有效的 10 s 段 `s`。段内步的振幅项之和 `S_s = Σ_{t_i∈s} (a_max,i − a_min,i)^{1/4}`；参考路径长 `D_s = Σ_j ‖p_xy(τ_j + 1 s) − p_xy(τ_j)‖`（1 s 分辨率，与 IPB PLR 一致）。去掉 `S_s = 0` 的段，做过原点的最小二乘：`K = Σ_s S_s·D_s / Σ_s S_s²`。记录 K、段数、R²、逐段长度 RMSE。每个训练数据集一个 K（`k_mode=per_dataset`）。 |
| δ（航向偏置） | 用与被比较模型相同的训练窗口网格（`window=200`，`stride=10`），只取全部样本有效且 `‖v_gt‖ > 0.5 m/s` 的窗口：`ψ_gt,k = atan2(v_gt,k)`，`ψ_dev,k` 为第 4 节第 8 步在该窗口上的航向（不加 δ）；`δ = atan2(Σ_k sin(ψ_gt,k − ψ_dev,k), Σ_k cos(ψ_gt,k − ψ_dev,k))`。同时记录平均合向量长度 `R̄ ∈ [0,1]`（越小说明偏置越不恒定）。选项 `delta=none` 表示 δ=0。 |
| 跨数据集 | 在数据集 A 上标定、在 B 上测试时使用 A 的 K、δ（与学习方法的跨数据集协议相同） |
| 其余超参 | 固定为第 4 节默认值 |
| 诚实性 | 标定函数只接受 train 划分的序列 ID；传入 val/test 序列时报错 |

## 6. benchmark 适配

| 项 | 类型 | 说明 |
|---|---|---|
| 以序列级预测器实现，输出对齐到 IPB 窗口网格，再走 §5 重建 | 不改变结构（方法本身无结构） | 所有方法共用同一时间轴与积分器 |
| 需要视图额外提供逐样本姿态 `q_wb` 与 `body_frame` | 接口扩展 | DESIGN §4 的 `forward(imu)` 只有 `(B,6,T)`；建议增加 `BaselinePredictor.predict_sequence(seq, view) -> (K, 2)` 或 `extra_inputs=["orientation"]` 钩子 |
| 含无效样本的窗口 | 不改变结构 | 仍输出速度，由 §5 第 5 条替换；步检测不跨无效区段 |
| `orientation=reference` 与 `device` 分别报告 | 不改变结构 | 区分“姿态误差”与“步长/朝向模型误差” |
| 非手机机体系（如头戴设备） | 需配置 | 前向轴查表必须补充；未配置时该数据集跳过并在结果中注明 |

## 7. 官方报告数值

无（本卡为定义，不对应任何论文实现）。不同论文里的 “PDR” 基线实现各异，数值不可引用为本基线的参照。

## 8. 忠实性测试建议（合成数据，确定性）

以 fs=200 Hz、20 s、`f = [0, 0, g₀ + A·sin(2π f_step t)]`（世界系，设备平放、姿态恒定）为基本信号：

- [ ] `A = 2 m/s²`，`f_step = 2 Hz` → 检测到 **40** 步；第 2 步起每步振幅 `a_max − a_min = 2A·|H(2 Hz)|² = 4/(1 + (2/3)^8) = 3.8498`（±1%），步长 `L = K·3.8498^{1/4} = 1.40075·K`。（首步区间被序列起点截断，振幅偏小，测试时跳过首步。）
- [ ] `f_step = 1 Hz` → 20 步，振幅 `4/(1 + (1/3)^8) = 3.9994`
- [ ] `f_step = 4 Hz`（高于截止频率）→ 滤波后振幅约 0.18 < `h_min` → **0 步**
- [ ] 静止：`f = [0, 0, g₀] + N(0, 0.05²)`（种子 0）→ 0 步 → 所有 `v_k = 0`
- [ ] 最小步间隔：任意输入下相邻步时间差 ≥ 0.3 s
- [ ] 航向：姿态 `R = R_z(30°)`、`body_frame=android_device`、δ=0 → 所有 `ψ_i = 120°`；把 `q_wb` 左乘 `R_z(α)` → 所有 `ψ_i` 加 α、所有 `v_k` 旋转 α
- [ ] 备用轴：设备竖直（机顶朝上、机背朝 `R_z(30°)·e_y` 方向）→ 使用 `−z` 轴，航向 = 120°
- [ ] 窗口汇总：人工给定步序列（时间、长度、航向），`v_k` 与第 3 节公式逐元素一致；时间正好等于 `t_s` 的步不计入该窗口，等于 `t_e` 的计入
- [ ] K 标定：用已知 `K = 0.45` 生成参考路径长，标定结果误差 < 1e-6；δ 标定：已知偏置 0.3 rad，误差 < 1e-3
- [ ] 泄漏保护：把 test 序列 ID 传给标定函数时抛出异常
- [ ] 常速直线：40 步、每步 `L`、航向恒定 → §5 重建的终点位移 ≈ `40·L`（误差 < 5%，来自首尾外推与计数抖动）
- [ ] `signal=vertical` 与 `signal=norm` 在上述竖直信号下结果相同

## 9. 预训练权重

不适用。标定结果（每个训练数据集的 K、δ、R̄、段数）写入运行目录 `metrics.json`，随结果一起发布。

## 10. 坑与未决问题

1. 前向轴依赖 `body_frame`：IPB 各转换器必须正确填写该属性；Android/iOS 手机机体系都是 “x 右、y 机顶、z 出屏”，其他设备需要单独确认。
2. `filtfilt` 在每个有效区段首尾有边缘效应（首步振幅偏小）；区段很短（< 约 0.5 s，小于滤波器 padlen）时直接跳过该区段。
3. 设备姿态的偏航漂移会直接变成航向误差；`orientation=device` 与 `reference` 的差即为姿态误差的影响。
4. 单一 K 对不同步态（跑、慢走、上下楼）和不同携带方式有系统偏差；这是基线的已知局限，不做分类器扩展，以保持定义简单、可复现。
5. 默认阈值（1.0 / 0.5 m/s²、0.3 s、3 Hz）是常见经验值，不是为某个数据集调出来的；如果某数据集检测率明显异常（例如 R² < 0.5），应在报告中说明，而不是在 test 上调参。
