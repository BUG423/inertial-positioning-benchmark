# IPB v1 开发交接文档 / Handoff

> 状态快照：2026-09-18。分支 `dev/v1`，提交 `ffb4d63`，700 个测试通过，`ruff check src tests` 0 错误。
> 本文件面向**接手开发的人或 Agent**，回答四个问题：为什么做、要求是什么、做到哪了、接下来怎么做。
> 设计决策的权威来源是 [`DESIGN.md`](DESIGN.md)（设计宪法），本文件不重复其内容，只做导航与状态记录。

---

## 1. 开发目的

把一套原本私有、不能开源的惯性定位研究代码（下称"旧系统"），**从零重写**为一个像 Ultralytics 那样
完备、可复现、可扩展的开源基准框架，使得：

1. 任何公开惯性定位数据集都能转换成**同一种数据事实**（统一单位、坐标系、时间基、采样率、真值来源标注）；
2. 任何公开算法都能在**同一套协议**下训练、推理、评测，接口差异由框架消化，而不是各自为政；
3. 评测结果**可审计**：每个数字都能追溯到数据版本、划分、配置、代码提交与随机种子。

它是一项基础设施工作，不作为创新点，但决定了后续研究结论是否站得住。

## 2. 开发要求（硬约束）

1. **第一性原理优先**：遇到"官方这么做"和"物理/统计上正确"冲突时，先写清事实，再选择，并把理由写进文档。
2. **洁净室**：不得打开 `/workspace/webCodex/Begin`（旧系统，私有）与 `/workspace/webCodex/open-inertial-benchmark`
   （早期被推翻的版本）下的**源码**（`*.py`、`*.sh`）。公开算法一律依据论文与官方仓库**独立重写**，
   保持本仓库自己的代码风格与模块组织。规格卡 `docs/algorithms/*.md` 就是为此准备的——
   移植者只看卡片与夹具，不看官方代码。
3. **忠实性有锁**：每个公开算法的参数量与逐项参数形状必须与官方实现一致，并由单元测试锁定
   （夹具在 `tests/fixtures/algorithms/*.json`，由实例化官方模型导出）。改变结构的适配必须显式标注。
4. **诚实协议**：开发与选模只看 val；test 只在最终评测时跑一次。不得用 test 选择变体、阈值或检查点。
5. **不静默修数据**：任何单位、轴、四元数约定、时间戳的修正都要写进转换报告与数据卡片。
6. **测试先行**：每个模块都要有确定性单元测试；真实数据测试在数据缺失时自动跳过。
7. **隐私**：任何网络请求不得包含用户邮箱或个人信息（曾发生过一次子任务把邮箱放进 Unpaywall 请求，已纠正）。
8. **提交规范**：小步提交，英文祈使句，结尾空一行加
   `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`；不提交数据、权重、运行输出。

## 3. 环境与部署

### 3.1 机器与资源

- 本机是**共享**的：3 张 A100 80GB，GPU0 常年被其他容器占满，GPU1/GPU2 也有外部负载（常驻约 57 GB、算力接近满载）。
- **约定只用 GPU2**：`CUDA_VISIBLE_DEVICES=2 CUDA_DEVICE_ORDER=PCI_BUS_ID`，单任务进程数 ≤2、每进程显存 <6–8 GB。
- CPU 256 核、内存 2 TB、`/workspace` 剩余约 18 TB。

### 3.2 代码与数据位置

| 路径 | 内容 |
|---|---|
| `/workspace/webCodex/inertial-positioning-benchmark` | 本仓库主检出，分支 `dev/v1`（已推送到 `origin`） |
| `/workspace/webCodex/datasets/ipb/` | 转换后的 IPB v1 数据（8 个数据集，约 7.1 GB），**不进 git** |
| `/workspace/webCodex/datasets/imu_odometry/raw/` | 原始公开数据集（只读） |
| `/workspace/webCodex/third_party/` | 官方仓库浅克隆，仅供写规格卡时核对，**不进 git**，移植者不得打开 |
| `/workspace/webCodex/ipb-runs/` | 训练与评测输出，**不进 git** |

环境变量 `IPB_DATASETS=/workspace/webCodex/datasets/ipb` 必须设置，否则默认根目录是 `~/datasets/ipb`
（曾因此误判"划分文件不存在"）。

### 3.3 安装与自检

```bash
cd /workspace/webCodex/inertial-positioning-benchmark
python -m venv --system-site-packages .venv && .venv/bin/pip install -e . --no-deps   # 或直接 PYTHONPATH=src
export IPB_DATASETS=/workspace/webCodex/datasets/ipb
PYTHONPATH=src python -m pytest -q          # 期望：700 passed
ruff check src tests                        # 期望：0 错误
PYTHONPATH=src python -m inertial_benchmark info
```

系统 Python 3.10 已有 torch 2.11(cu128)、numpy、h5py、scipy、pandas、matplotlib、pyyaml、pyarrow。

### 3.4 常用命令

```bash
ipb convert dataset=ronin source=<raw> output=$IPB_DATASETS   # 原始数据 → IPB v1
ipb check   data=ronin full=true hash=true                    # 校验格式、重力、划分泄漏、内容重复
ipb train   model=ronin_resnet18 data=ridi device=2 amp=false
ipb val     model=runs/train/exp/weights/best.pt data=ridi split=test
ipb predict model=... source=<seq.h5>
ipb benchmark cfg=batchA device=2                             # 矩阵；同命令即续跑
ipb report  runs=<运行目录> out=<报表目录>                      # 部分完成也能汇总
```

**训练默认 `amp=false`**——这既是吞吐考虑，也是正确性考虑，见 §6.1。

## 4. 已完成的工作

### 4.1 数据层

- **IPB v1 格式**（DESIGN §2）：统一 200 Hz、机体系 IMU（比力含重力）+ 参考姿态（重力对齐 z 轴向上）、
  区分参考姿态与设备自身姿态、逐样本有效掩码、数据集级清单与转换报告、gzip 逐字节可复现。
- **8 个转换器**：`ronin, ridi, oxiod, imunet, tlio, rnin, idol, pedlocdata`，每个都带物理自检
  （重力对齐、速度量级、陀螺与参考姿态一致性、时钟对齐），不合格序列拒收并记原因。
- **已转换数据**：3192 条序列、196.45 h、稠密 2D 路程 536.5 km；`ipb check full=true hash=true` 8 个数据集 0 错误。
- 转换中查明并修正的关键事实（详见各数据卡 `docs/datasets/*.md`）：
  - RoNIN 的 `tango_ori` 是胸前 Tango 的姿态（另一个刚体），参考姿态改用起点对齐后的设备姿态；
  - IMUNet 的 ARCore 设备四元数需 `R_x(+90°) ⊗ ori ⊗ R_z(+90°)`，旧流水线因此把重力放到 −x 轴；
  - OxIOD 比力为 `−(gravity+user_acc)·9.80665`，Vicon 与手机的外参逐序列估计（约 169–179°），
    两个时钟存在 −0.18…+1.70 s 偏移与约 −490 ppm 速率差；
  - IDOL 的 `orient` 是 Stencil 设备姿态，需估计到 iPhone 机体的常值外参；iOS 加速度符号与比力相反；
  - TLIO 的 npy IMU 在世界系且已用 VIO 在线补偿零偏（属参考信息泄漏，已在卡片标注）；
  - PedLocData 官方划分存在会话级泄漏（213 条 test 切片中 201 条的紧邻切片在 train/val），
    默认改用分组划分，官方划分以 `official_*` 保留。
- **划分策略**（DESIGN §2.4）：官方划分优先；官方泄漏时改用分组划分；val 按条件分层生成，
  保证 val 中出现的条件在 train 中存在；train 中不存在的条件移入 `test_ood_*` 子集单独报告。

### 4.2 框架

`src/inertial_benchmark/` 分层：`cfg`（YAML + `key=value` 覆盖）、`data`（格式/重采样/转换/划分/视图/增强/数据集）、
`nn`（注册表/积木/输出头/损失）、`models`（21 个算法包）、`engine`（NIO 门面/训练器/验证器/推理器/矩阵）、
`metrics`、`utils`、`cli`。要点：

- **任务视图**：窗口、步长、坐标系（`gravity_world`/`body`/`gravity_yaw_local`）、姿态来源、去重力、
  目标类型（平均速度/位移/末端速度/**逐帧**/**多步**）、2D/3D、增强，全部是配置。
- **协议扩展**（为忠实复现而加）：逐帧与多步输出布局、历史上下文 `history`、额外输入 `extra_inputs`
  （含**特权输入**标记，如来自真值的初速度，报表单列）、序列级 `SequenceModel`（PDR 等有状态方法）。
- **推理与重建**（DESIGN §5）：窗口中心时间戳、首尾常值外推、梯形积分、起点锚定、重叠输出按 `overlap` 合并、
  同时积分窗口目标得到 oracle 轨迹。
- **评测协议锁定**：模型 YAML 不得改写评测协议键；生效协议写入 `args.yaml` 与 `metrics.json`，
  `ipb report` 会比对不同 run 的协议是否一致。
- **等窗口预算**：`train_windows_budget`（默认 3.2e7）按每轮窗口数换算 epoch，使大小数据集获得同等优化预算。
- **训练诊断**：`train_eval_gap` 守卫在训练结束比较 `train()`/`eval()` 两模式的窗口损失并写入 `metrics.json`
  （当前为单侧判据，待改双侧，见 §6.1）。

### 4.3 算法

21 个算法包、31 个注册名，全部参数量与官方夹具或卡片推导值逐项一致：

- RoNIN（ResNet18/50/101、LSTM、TCN）、TLIO、RNIN、IMUNet（+MobileNet/MobileNetV2/MnasNet/EfficientNetB0）、
  DeepILS、EqNIO（RoNIN 版与 TLIO 版）、LLIO、TinyOdom、RIO、IONet、TartanIMU、CTIN、iMoT、IONext、
  DIVE、AirIO、NIO-LieEvents（两个骨干）、**GNIO**、**VeloBins**，以及两个经典基线 PDR 与常速航向。
- 可复用输出头：速度头、对角高斯头、极坐标头、**门控头 `gated`**（GNIO 式）、**分箱头 `bins`**
  （VeloBins 式，解码方式可热切换），可插到任意骨干做消融。
- 规格卡 `docs/algorithms/*.md`（29 张）记录论文、官方仓库与提交号、许可、逐层结构、损失、训练配方、
  IPB 适配差异与理由、官方报告数值、忠实性测试建议。**移植新算法的唯一入口就是先写卡片。**
- 卡片中记录的官方实现"坑"（例如 TLIO 调度器从不 step、RNIN 协方差头因阈值设置从未训练、
  多个仓库用测试集选模），`official` 配方如实复现，`unified` 配方给出修正版本，两者都有测试。

### 4.4 评测与报表

ATE（欧氏定义）、ATE_aligned、RTE（RoNIN 定义）、T-RTE、D-RTE、PDE、长度比（含 oracle 分解）、
速度偏差、方向误差、沿/横迹误差、窗口 RMSE、效率（参数量/FLOPs/延迟）；
逐序列表、按组 bootstrap 置信区间、多种子聚合、配对 Wilcoxon；轨迹叠加、误差 CDF、误差随时间、
箱线图、长度比散点、帕累托图。**注意**：本仓库的 ATE 与 RTE 都是欧氏定义，是 RoNIN 官方代码数值的 √2 倍
（已核实官方 `metric.py` 对坐标分量求均值），与文献对比时必须换算，`docs/METRICS.md` §2.2 有说明。

### 4.5 已验证的正确性证据

- 独立重写的指标脚本与框架实现最大相对误差 **1.9e-14**；独立重写的轨迹重建与框架**逐位相同**。
- RIDI 上 `ronin_resnet18` 完整训练后 test ATE 2.98 m（换算成 RoNIN 官方口径 2.10 m），与文献同量级。
- 8 个数据集抽检：1 s 分辨率路径长度与原始数据差 ≤0.16%，重力对齐加速度均值差 ≤0.004 m/s²。

## 5. 未完成的工作（按优先级）

### 5.1 立刻要做的

1. **分层划分的收尾**（提交 `ffb4d63` 标注为 partially complete）：
   - 重跑一次划分审计表，证明"val 条件被 train 覆盖"的问题确实消失；
   - 更新 `docs/DESIGN.md` §2.4（分层规则与理由）与 `docs/DATASETS_V1.md`（新划分规模、`test_ood_*` 子集说明）；
   - 数据集指纹已随重算更新（`tests/data/test_converted_datasets.py`），如再次重算需同步更新。
2. **`train_eval_gap` 守卫改双侧**（规格已定，见 §6.1）。
3. **重启基准矩阵**：`cfg/benchmarks/batchA.yaml` 已存在（12 模型 × 3 数据集 × seed 0，只评 val）。
   现在模型已扩到 31 个注册名，建议重新拟定模型清单。**重要教训：矩阵必须跑在固定代码版本的独立检出上**
   （例如 `git worktree add /workspace/webCodex/ipb-bench <commit>`），否则开发中的改动会让前后 run 不可比。
   上一批只完成 1 个 run 就因划分修正而作废。

### 5.2 中期

4. **第三批算法**：PedestrianDiffusion（AGPL，37M 参数，频域扩散）等尚未移植；
   `docs/ALGORITHMS.md` §3 列出仍未落地的协议扩展（有状态递推逐帧输出、TTT、EKF 融合、
   `orientation_fallback`、窗口速度上限过滤）。
5. **框架缺口**：`frame=body` 下的 `random_yaw` 会同时旋转输入与目标（自相抵消），
   真正需要的是"只旋转姿态支路"的算子；`CHOICES["loss"]` 仍是白名单，命令行无法覆盖模型专用损失。
6. **多种子与 test 评测**：矩阵目前设计为单种子、只评 val；最终主表需要 3 个种子，test 一次性评测。

### 5.3 长期

7. 其余数据集接入（Mobile-GVIO 原始包已在本地未转换；Aria/Nymeria/TartanIMU Challenge 需授权）。
8. 排行榜与版本化结果发布；README 与 `docs/` 的英文版本。

## 6. 接手者必须知道的坑

### 6.1 AMP 会让 RoNIN 式全连接头静默塌陷（最重要）

实测：`ronin_resnet18` 在 `amp=true` 下训练，推理时**速度幅值只剩真值的 0.37–0.48**，轨迹长度比同步塌到 0.35；
`amp=false` 下同样配置是 0.88–0.91。四条独立测量（两套代码状态、两位工程师、基准矩阵）互相印证，
数据划分、测量口径、训练时长均已排除。机理：fp16 前向 + GradScaler 把输出头推进"预激活几乎全负"的退化解，
训练期唯一把该层推过零点的是 dropout 的噪声基座，推理时噪声消失，输出塌陷；**换 bf16 同样塌陷**。

后果与要求：
- **主表所有 run 必须 `amp=false`**（现已是默认值）；
- 任何 `amp=true` 的历史结果都应视为可疑；
- `train_eval_gap` 守卫要改成**双侧**：`gap = max(r, 1/r) > 1.5` 告警、`> 3` 报错，
  对有符号损失用差值判据；同时记录两种模式的 `speed_ratio`/`plr`，并做 dropout/BN 通道归因
  （`dropout_only` 用 K=8 的 MC 平均，单次采样会系统性抬高幅值比）；
- 可豁免的模型：dropout 紧接仿射输出层者（`imunet_mnasnet`、`imunet_efficientnetb0`）、
  无 dropout 者（`imunet`、`imunet_mobilenet`、`imunet_mobilenetv2`）、`tinyodom`（p=0）。
  其余 17/20 个配置的 dropout 上游含非线性，守卫必须开启。

### 6.2 其他

- **验证集代表性**：RIDI 曾出现 val 只有 1 名受试者、且其携带方式在 train 中完全不存在，
  导致方向误差中位数 53°、均值 ATE 11 m 而中位数 1.58 m。已用分层划分修正，
  并提供 `fitness_stat=median` 选项；新接入数据集时务必先看条件覆盖。
- **缺口序列**：IDOL/RIDI/OxIOD 有多秒缺口，跨缺口的速度插值会产生不可信轨迹；
  `ate_oracle`（协议误差下限）已进默认报表列，缺口序列上它可能超过模型自身 ATE。
- **续训可复现**：每轮用 `(seed, epoch)` 重置随机流，"连续训练 N 轮"与"中断续训到 N 轮"权重逐位相同，有测试锁定。
- **`unified` 配方统一什么**：统一训练预算、数据划分、窗口与评测协议；**不统一**算法固有的增强与输入表示
  （EqNIO 的等变规范帧本就替代偏航增强，RIO 的旋转监督即其方法），理由见 DESIGN §4。

## 7. 文档导航

| 文件 | 内容 |
|---|---|
| [`DESIGN.md`](DESIGN.md) | 设计宪法：第一性原理、架构、数据格式、任务视图、模型接口、推理重建、评测、开发纪律 |
| [`FORMAT.md`](FORMAT.md) | v0.1 数据规范（历史版本，v1 在 DESIGN §2） |
| [`METRICS.md`](METRICS.md) | 每个指标的精确公式与边界情况，含与 RoNIN 官方口径的 √2 关系 |
| [`CLI.md`](CLI.md) | 命令行契约、配置键、吞吐实测、`train_windows_budget`、`fitness`/`fitness_stat` |
| [`DATASETS_V1.md`](DATASETS_V1.md) | 数据发布说明：规模、真值来源、划分、许可、已知限制 |
| [`datasets/*.md`](datasets/) | 8 张数据卡：字段、约定、标定、物理自检统计、被拒序列 |
| [`ALGORITHMS.md`](ALGORITHMS.md) · [`algorithms/*.md`](algorithms/) | 算法总表与 29 张规格卡 |
| [`REPOSITORY_LAYOUT.md`](REPOSITORY_LAYOUT.md) | 目录职责与新组件准入规则 |
