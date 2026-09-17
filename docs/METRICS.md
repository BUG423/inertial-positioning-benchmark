# IPB 评测指标 / Metrics

> 状态：v1 开发基线。实现见 `src/inertial_benchmark/metrics/`，解析解测试见 `tests/test_metrics.py`。
> 本文与 [DESIGN.md](DESIGN.md) 第 5–6 节配套；两者冲突时先修正文档再改代码。

## 1. 约定

| 符号 | 含义 |
|---|---|
| `t_i`, `i = 0…N−1` | 序列统一时间轴，`t_i = i / r`，`r = 200 Hz` |
| `p_i` | 参考位置（`pose/position`），世界系 |
| `p̂_i` | 预测轨迹（DESIGN 第 5 节重建：窗口速度 → 首尾常值外推 → 梯形积分 → 锚定首个有效参考位置） |
| `p̃_i` | oracle 轨迹：把窗口**目标**当作预测、用同一流程积分得到 |
| `m_i` | 参考位姿有效掩码 `valid/pose`（轨迹指标只在 `m_i = 1` 处取样） |
| `v̂_k`, `v_k` | 第 `k` 个窗口的预测 / 目标速度，已换算到世界系 |
| `w_k` | 窗口有效掩码：窗口内全部样本 `valid/imu ∧ valid/pose` |
| `D` | 计算维度，默认 `metric_dims = 2`（只取 x、y，水平面） |

- 所有指标**逐序列**计算，再在序列间聚合（第 5 节）。
- 无法定义时返回 `NaN`（例如没有有效样本），聚合时剔除并报告有效序列数 `n`。
- 距离单位米，速度 m/s，角度度（°）。

## 2. 轨迹级指标

### 2.1 ATE（绝对轨迹误差）

```
ATE = sqrt( mean_{i: m_i} ‖p̂_i − p_i‖² )
```

不做任何对齐（起点已锚定）。`ATE_aligned`（键 `ate_aligned`）先对 `p̂` 做最小二乘刚体对齐：
2D 时为 SE(2)（绕 z 旋转 + 平移），3D 时为 4 自由度（绕 z 旋转 + 三维平移），旋转角闭式解
`θ = atan2(Σ (a_x b_y − a_y b_x), Σ a·b)`，`a`、`b` 为去均值后的预测与参考水平坐标。有效样本少于 2 个时为 `NaN`。

> 与 RoNIN 官方代码的关系：官方 `compute_absolute_trajectory_error` 对 `N×2` 个**坐标分量**求均值，
> 即 `ATE_ronin_code = ATE / √2`（2D）。IPB 采用逐点欧氏距离（TUM 定义），与论文表格对比时需注意该因子。

### 2.2 RTE（RoNIN 定义）与 T-RTE@τ

```
Δ = round(τ · r)
RTE(τ) = sqrt( mean_{i: m_i ∧ m_{i+Δ}} ‖(p̂_{i+Δ} − p̂_i) − (p_{i+Δ} − p_i)‖² )
```

- `RTE`（键 `rte`）取 `τ = rte_delta = 60 s`，起点为每一个样本（步长 1）。
- **统一规则按有效跨度决定是否换算**（`valid_span = (i1 − i0)/r`，`i0`/`i1` 为首、末有效样本，
  中间的缺口计入跨度）：
  1. 只要存在至少一对两端均有效的 `(i, i+Δ)`，就用上式对**全部**这样的对求 RMSE，`rte_scaled = 0`；
  2. 否则（有效跨度不足 τ，或缺口把序列切成的片段都短于 τ）只取 `(i0, i1)` 这一对并线性换算：
     `RTE = ‖e(i0, i1)‖ · τ / valid_span`，`rte_scaled = 1`；
  3. 有效样本少于 2 个或 `valid_span = 0` 时为 `NaN`。
- 指标输出额外包含 `rte_scaled`（0/1，聚合后即“被换算的序列比例”）、`valid_span_s`（有效跨度，秒）
  与每个 τ 的 `t_rte_<τ>s_scaled`。
- 早期实现按**样本数**（`N − 1 < Δ`）判断短序列，于是 70 s 序列若有效跨度不足 60 s 就得到 `NaN`、
  40 s 序列却按比例放大给出数值：长序列中最难的那些被悄悄剔除，聚合结果存在选择偏差。现按有效跨度统一处理。
- RoNIN 官方代码使用 `(0, N−1)` 这一对并乘以 `12000 / N`，与按实际跨度换算相差 `N/(N−1)` 倍（可忽略）。
- `T-RTE@τ`（键 `t_rte_1s`、`t_rte_10s`，由 `t_rte` 配置）使用同一公式与同一换算规则。

### 2.3 D-RTE@d（按距离的相对误差）

1. 参考累计距离 `s_i`：在 1 s 分辨率折线上累计（只累计两端均有效的线段），再线性插值回每个样本；
2. 起点沿参考路径每 `1 m` 取一个：`i_k = min{ i ≥ i_first : s_i ≥ k·1m }`（避免静止段被重复计数）。
   `i_first` 为首个有效样本：`s_i` 在它之前恒为 0，若从样本 0 开始搜索，序列开头无效时 `k = 0`
   的起点会落在无效样本上而被整段丢掉（早期实现的缺陷）；
3. 终点 `j_k = min{ j : s_j ≥ s_{i_k} + d }`；两端都有效才计入；
4. `D-RTE@d = sqrt( mean_k ‖(p̂_{j_k} − p̂_{i_k}) − (p_{j_k} − p_{i_k})‖² )`。

默认 `d = 10 m`（键 `d_rte_10m`，由 `d_rte` 配置）。参考总距离小于 `d` 时为 `NaN`。

### 2.4 PDE（终点漂移）

```
PDE = ‖p̂_e − p_e‖ / L_ref × 100      (%)
```

`e` 为最后一个有效样本；`L_ref` 为 1 s 分辨率的参考路径长度（见 2.5）。`L_ref < 1 m` 时为 `NaN`。

### 2.5 路径长度与 PLR

路径长度 `L_res(x)`：从首个有效样本起每 `res` 秒取一个样本（并补上末个有效样本），对相邻采样点的折线段求和，
**只累计两端均有效的线段**。同一序列的所有长度使用**同一组采样点**（由参考掩码决定）。

| 键 | 定义 |
|---|---|
| `plr` | `L_1s(p̂) / L_1s(p)` |
| `plr_dense` | `L_1s(p̂) / L_dense(p)`，`L_dense` 为 200 Hz 稠密折线长度 |
| `plr_oracle` | `L_1s(p̂) / L_1s(p̃)` |
| `oracle_ratio` | `L_1s(p̃) / L_dense(p)` |
| `ate_oracle` | oracle 轨迹的 ATE（协议本身能达到的下限参考） |

`ate_oracle` 属于默认报表列（`MAIN_METRICS`），必须与 `ate` 一起解读。含多秒 `valid` 缺口的序列上，
缺口内的窗口速度是插值来的，而参考位置在缺口两端可能不连续，于是 oracle 轨迹自身就偏离参考，
`ate_oracle` 可以超过模型的 `ate`（真实数据实例：RIDI `huayi_bag2` 有 4.42 s 的 IMU/位姿缺口，
oracle ATE 5.37 m，而模型 ATE 3.31 m）。这类序列上的 `ate`/`pde` 度量的是缺口而非模型，报告时应单独列出。

恒等式 `plr_dense = oracle_ratio × plr_oracle` 精确成立（测试锁定）。`oracle_ratio < 1` 反映窗口平均速度
与 1 s 采样对路径的平滑程度（协议效应），`plr_oracle` 反映模型效应。参考长度为 0 时对应比值为 `NaN`。

## 3. 窗口级速度指标

只统计 `w_k = 1` 的窗口，取前 `D` 维。

| 键 | 定义 |
|---|---|
| `vel_rmse` | `sqrt( mean_k ‖v̂_k − v_k‖² )` |
| `speed_bias` | `mean_k (‖v̂_k‖ − ‖v_k‖)` |
| `speed_ratio` | `Σ_k ‖v̂_k‖ / Σ_k ‖v_k‖`（比值的聚合形式，避免低速窗口的除零放大；分母为 0 时 `NaN`） |
| `dir_err_mean` / `dir_err_median` | 夹角 `atan2(‖v̂_k × v_k‖, v̂_k · v_k)` 的均值 / 中位数，只统计 `‖v_k‖ > min_speed`（默认 0.2 m/s）；预测速度为零时记为 90° |
| `along_bias` / `along_rmse` | 误差在目标方向 `u_k = v_k/‖v_k‖` 上的分量 `a_k = (v̂_k − v_k)·u_k` 的均值 / RMSE（同样只统计运动窗口） |
| `cross_rmse` | 垂直分量 `‖(v̂_k − v_k) − a_k u_k‖` 的 RMSE |
| `num_windows` / `num_moving_windows` | 参与统计的窗口数 |

纯旋转误差 `θ` 时：`dir_err = θ`，`along_bias = mean‖v‖(cos θ − 1)`，`cross_rmse = sin θ · rms‖v‖`；
纯缩放 `(1+s)` 时：`speed_ratio = 1+s`，`cross_rmse = 0`（均有测试）。

窗口速度的来源：`avg_velocity` 与 `velocity_at_end` 目标直接是速度；`displacement` 目标除以窗口跨度
`(T−1)/r` 换算；`gravity_yaw_local` 与 `body` 坐标系先用窗口末端姿态旋回世界系。

## 4. 效率指标

| 键 | 定义 |
|---|---|
| `params` | 全部参数个数 |
| `flops` | batch=1、单窗口 `(1, 6, T)` 前向的乘加次数（MACs）；`flops_backend` 记录计数后端 |
| `latency_ms_cpu` / `latency_ms_cuda` | batch=1 单窗口前向延迟：预热 10 次后 30 次的中位数（CUDA 用事件计时并同步）；CPU 线程数 ≤ 8 |

FLOPs 后端优先 `thop`，其次 `fvcore`，都不可用时使用内置钩子：
Conv `= 输出元素数 × (C_in/groups × Πk) (+ 输出元素数，若有偏置)`，Linear `= 行数 × (in × out + out)`，
BN/LayerNorm `= 2 × 输出元素数`，LSTM/GRU `= 步数 × 方向数 × Σ_层 [门数 × (in·h + h² + 2h) + 逐元素项]`，
MultiheadAttention `= 4·L·E² + L²·E`。不同后端约定略有差异（ResNet18@200 帧：thop 38.25 M，钩子 38.11 M），
比较时请使用同一后端。

## 5. 聚合与统计（`utils/reports.py`）

- **跨序列**：均值、中位数、标准差（`ddof=1`），跳过 `NaN`，同时报告 `n`。
- **分组 bootstrap 95% 置信区间**：按 `group_id` 有放回地重采样组（默认 2000 次，种子 0），
  每次对被抽中组内全部序列的指标求均值，取 2.5% / 97.5% 分位数。只有一个组时区间退化为点。
- **多种子**：先对每个种子求跨序列均值，再报告种子间的 `均值 ± 标准差`（`ddof=1`）。
- **配对 Wilcoxon 符号秩检验**：模型 A、B 在同一数据集同一划分的相同序列上配对，逐序列值先在种子间取均值；
  双侧检验（`scipy.stats.wilcoxon`，`zero_method="wilcox"`），报告 `n`、统计量、p 值与差值中位数。
  配对少于 5 对时不给出 p 值。

- **评测协议一致性**：汇总前核对各 run 的 `metrics.json` → `protocol` 块，评测协议键
  （`eval_stride`、`metric_dims`、`rte_delta`、`t_rte`、`d_rte`、`min_speed`）必须完全一致，
  否则报错（见 [DESIGN.md](DESIGN.md) 第 6 节）。
- **特权输入单列**：使用特权输入（`metrics.json` 的 `privileged_inputs`，例如来自参考真值的
  `init_velocity`）的方法在 `summary.csv` 中带 `privileged` 列、在 `summary.md` 里单独一张表，
  LaTeX 主表只含纯 IMU 方法——两者不可直接比较。

## 6. 诚实协议

训练期间只在 `val` 上计算上述指标并据此选模型（`fitness`，默认 `ate`，越小越好，配置解析时即校验）；
每个越小越好的指标另有中位数型变体 `<指标>_median`（`ate_median` 等），取逐序列该指标的中位数，
供 val 划分小而偏斜的数据集使用（RIDI 自动生成的 val 只有 1 名受试者、含 2 条 train 中不存在的
携带方式，均值型选模信号很噪）；缺省不变，改用时须在结果中注明，详见 `docs/CLI.md` §4.3。
`test` 及其官方子集只在最终评测（`ipb val split=test` 或 `ipb benchmark`）中运行。
序列级/标定型方法（`pdr`、`mean_speed_heading`）的标定标量只在 `train` 划分上拟合，
结果写入 `metrics.json` 的 `calibration` 块随结果一起发布。
