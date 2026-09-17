# IPB 命令行 / CLI

> 实现见 `src/inertial_benchmark/cli.py`；设计背景见 [DESIGN.md](DESIGN.md)，指标定义见 [METRICS.md](METRICS.md)。

安装后（`pip install -e ".[train]"`）提供 `ipb` 命令，也可用 `python -m inertial_benchmark` 调用。

```bash
ipb <command> key=value [key=value ...]
```

## 1. 参数写法

- 一律 `key=value`，与 Python API 的关键字参数同名（`ipb train epochs=2` ⇔ `NIO(...).train(epochs=2)`）；`--key=value` 亦可。
- 取值解析：`true/false` → 布尔；`none/null` 或空 → 空值；整数与浮点（含 `1e-3`）→ 数字；
  以 `[` 或 `{` 开头 → YAML 列表/字典（`augment=[random_yaw,time_shift]`、`model_args={dropout: 0.1}`）；其余为字符串。
  含空格的列表需加引号：`"t_rte=[1, 10]"`。
- 训练/评测相关的键必须出现在 `default.yaml` 中，否则报错并给出相近键名（例如 `epoch` → `epochs`）；类型不符也会报错。
  只有 `data`、`source`、`pretrained`、`loss`、`device`、`name` 这几个“明确可空”的键接受 `none`；
  其余键写 `none` 直接报错（`epochs=none` 不会静默退回默认值）。`fitness` 在解析时按“越小越好的验证指标”校验。
- 序列 id 列表（`only=`）一律按字符串解析：`only=[010]` 是 `"010"` 而非八进制 8，`only=007` 是 `"007"`；
  空列表、括号不匹配、空元素等以退出码 `2` 报错。
- 合并顺序：`default.yaml` < 模型 YAML 的 `input` < 模型 YAML 的 `recipes[recipe]` < 命令行。
  使用 checkpoint 时，模型输入规格（`window/frame/orientation/remove_gravity/target/dims/rate`、
  `overlap/history*/extra_inputs`）来自 checkpoint，不可改写。
- **评测协议不属于模型**：`eval_stride`、`metric_dims`、`rte_delta`、`t_rte`、`d_rte`、`min_speed`、`split`
  只能来自 `default.yaml`、benchmark 配置或命令行；模型 YAML 的 `input`/`recipes` 里出现它们（或 `device`、
  `project` 等运行键）会报错。生效协议写入 `runs/*/args.yaml` 与 `metrics.json` 的 `protocol` 块，
  `ipb report` 会核对各 run 的评测协议是否一致，不一致即报错。

环境变量：`IPB_DATASETS`（转换后数据根目录，缺省 `~/datasets/ipb`）、`IPB_VERBOSE=0`（静默）、
`IPB_PLUGINS`（逗号分隔的插件模块名或 `.py` 路径，导入后其中 `@register_model` / `@register_augmentation`
注册的模型与增强即可按名字使用；插件中的模型需配合模型 YAML 或已注册的名字，checkpoint 评测时同样需要设置）、
`CUDA_VISIBLE_DEVICES`（可见 GPU；`device=0` 指可见列表中的第一张）、`OMP_NUM_THREADS`（CPU 线程，缺省最多 8）。

退出码：`0` 成功；`1` 检查未通过或无序列被接收；`2` 参数/配置/文件错误。

## 2. 命令

### `ipb convert` — 原始数据 → IPB v1

```bash
ipb convert dataset=ronin source=/raw/ronin                         # 输出到 $IPB_DATASETS/ronin
ipb convert dataset=ridi source=/raw/ridi output=/data/ipb/ridi workers=8
ipb convert dataset=ronin source=/raw/ronin only=[a000_1,a001_3] overwrite=false
```

| 键 | 缺省 | 说明 |
|---|---|---|
| `dataset` | 必需 | 转换器名（`data/converters/<dataset>.py`），也是数据集名 |
| `source` | 必需 | 原始数据目录 |
| `output` | `$IPB_DATASETS/<dataset>` | 输出目录 |
| `only` | 全部 | 只转换这些 `sequence_id`（列表或逗号分隔） |
| `workers` | `min(8, 核数/2)` | 并行进程数 |
| `overwrite` | `true` | `false` 时跳过已成功转换的序列 |
| `converter` | 按 `dataset` | 转换器模块名或 `.py` 文件路径（测试/私有数据用） |
| `compression` | `gzip` | `gzip`（字节可复现）/ `lzf` / `none` |
| `gap_threshold` | `0.05` | 缺口阈值（秒），见 DESIGN 2.3 |
| `min_duration` | `2.0` | 短于该时长（秒）的序列拒收 |
| `val_fraction`, `seed` | `0.1`, `0` | 官方无 val 时按 `group_id` 抽取的比例与种子 |

输出 `sequences/*.h5`、`splits/*.txt`、`dataset.json`（统计、sha256、IMU 内容哈希、指纹、划分策略）、
`conversion_report.json`（接收/拒绝原因、警告、裁剪量、泄漏与重复检查）。

划分规则（DESIGN 2.4）：默认写官方划分（缺 val 时按 `group_id` 抽取）；转换器声明 `OFFICIAL_SPLITS_LEAK` 时，
默认 `train/val/test` 改为转换器的分组划分，官方划分另存为 `official_*.txt`；转换器声明的附加子集
（如 `test_unseen_subject`）原样写出；不属于任何划分的已转换序列列在 `dataset.json` 的 `unassigned` 中，不会并入 train。
读取 IDOL 需要 `pip install -e ".[idol]"`（pyarrow）。

### `ipb check` — 校验已转换数据集

```bash
ipb check data=ronin                       # 清单、文件、划分泄漏
ipb check data=ronin full=true hash=true save=check_ronin.json
```

`full=true` 逐条读取并执行完整校验（含重力检查）并重算 IMU 内容哈希；`hash=true` 重算文件 sha256；`save` 写出 JSON 报告。

- **错误**：文件缺失、默认划分之间的序列重叠、`unseen/unknown/novel` 子集与 train/val 共享 `group_id`、
  IMU 内容相同的不同序列（重复发布）、逐条校验失败、哈希不符；
- **警告**：默认划分之间的 `group_id` 重叠（seen-subject 等官方设定，如 RoNIN `test_seen`）、
  `official_*` 划分中的已声明泄漏、不属于任何默认划分的序列（`unassigned`）。

### `ipb train` — 训练

```bash
ipb train model=ronin_resnet18 data=ronin epochs=40 device=0
ipb train model=ronin_resnet18 data=ronin recipe=official device=0 name=ronin_official
ipb train model=runs/train/exp/weights/best.pt data=ridi epochs=10     # 以 checkpoint 为 pretrained 微调
ipb train resume=true name=exp                                          # 断点续训（或 resume=path/to/last.pt）
```

训练只使用 `train` 与 `val` 划分；每 `val_interval` 轮在 val 上计算全部指标，按 `fitness` + `fitness_stat`（缺省 `ate` 的逐序列均值，越小越好）保存 `best.pt`。
结束后用 best 权重在 val 上评测一次并写出 `metrics.json`，其中 `train_eval_gap` 记录同一权重在
`train()` 与 `eval()` 两种模式下的 val 窗口损失比（见第 4.2 节）。

### `ipb val` — 评测

```bash
ipb val model=runs/train/exp/weights/best.pt data=ronin split=test_unseen device=0
```

`split` 为逻辑划分名（经数据集 YAML 的 `splits` 映射）或 `splits/<file>.txt` 的文件名。
test 及其官方子集只应在最终评测时运行。

### `ipb predict` — 推理

```bash
ipb predict model=best.pt source=/data/ipb/ronin/sequences/a000_1.h5
ipb predict model=best.pt source=/data/ipb/ronin save=false
```

`source` 为单个 `.h5`（v1 或 v0.1）或目录；`save=true`（缺省）时写出 `predictions/<seq>.npz` 与轨迹图。

### `ipb benchmark` — 模型 × 数据集 × 种子矩阵

```bash
ipb benchmark cfg=main dry_run=true                      # 内置 cfg/benchmarks/main.yaml，只打印计划
ipb benchmark cfg=benchmarks/my.yaml device=0            # 自定义 YAML；其余 key=value 覆盖所有运行
```

`cfg` 为 YAML 路径；路径不存在时按文件名在 `cfg/benchmarks/` 中查找（`cfg=main` 即内置主基准：
`ronin_resnet18` × 八个核心数据集 × 种子 0/1/2；`cfg=batchA` 见下）。

```yaml
name: main
project: runs/benchmark            # 输出 <project>/<name>/<label>/<dataset>/seed<k>/{train,<split>}/
models:
  - ronin_resnet18                   # 名称 / 模型 YAML / checkpoint（checkpoint 不再训练）
  - {label: ronin_official, model: ronin_resnet18, overrides: {recipe: official, epochs: 100}}
datasets: [ronin, ridi, {data: /data/ipb/oxiod, overrides: {batch: 256}}]
seeds: [0, 1, 2]
splits:                            # 缺省为数据集 YAML 的 test_splits；[val] = 只评 val
overrides: {train_windows_budget: 3.2e+7, device: 0}
report: true                       # 结束后汇总到 <project>/<name>/report
continue_on_error: false           # true 时单项失败只记录进 benchmark.json 并继续
```

完成标志为 `train/metrics.json` 与 `<split>/metrics.json`：重复运行会跳过已完成项；训练中断（有 `last.pt` 无 `metrics.json`）会自动续训；
数据集缺少某个测试子集时跳过并告警。进度写入 `<project>/<name>/benchmark.json`。
命令行的 `project=<dir>` 会覆盖 YAML 里的输出根目录（同一份矩阵可以写到别的盘而不改配置）。

每个 run 的开始与结束各写一行结构化日志，便于 `grep` 与自动监控：

```text
benchmark run start: {"i": 3, "n": 36, "model": "tlio", "dataset": "rnin", "seed": 0, "dir": ...}
benchmark run done:  {"i": 3, ..., "train": "trained", "epochs": 69, "seconds": 8123.4,
                      "fitness": "median ate", "val_fitness": 3.41, "test": false,
                      "splits": {"val": "evaluated"}}
```

`dry_run=true` 只打印计划，并且**已经给出每个 run 的 epoch 数**（等窗口预算按 train 划分的窗口
数换算；惰性读取掩码，不载入序列）：

```text
benchmark run planned: {"i": 1, "n": 36, "model": "ronin_resnet18", "dataset": "ridi",
                        "seed": 0, "epochs": 334, "splits": ["val"]}
```

`continue_on_error: true`（或命令行 `continue_on_error=true`）时单个组合失败不终止矩阵：状态里
带 `error`，写进 `benchmark.json`，命令最终以退出码 `1` 结束。

#### 内置 `batchA`

`cfg/benchmarks/batchA.yaml`：十个学习型算法 + 两个经典基线 × `ridi` / `imunet` / `rnin` × 种子 0，
`recipe: unified`、`fitness=ate` + `fitness_stat=median`、`amp=false`、`deterministic=true`、
`train_windows_budget=3.2e7`（`budget_max_epochs=400`），`splits: [val]` —— **只评 val**，遵守
DESIGN 第 6 节的诚实协议。

```bash
CUDA_VISIBLE_DEVICES=2 CUDA_DEVICE_ORDER=PCI_BUS_ID ipb benchmark cfg=batchA   # 启动/续跑
ipb benchmark cfg=batchA dry_run=true                                          # 只看计划
ipb report runs=<project>/batchA out=<project>/batchA/report                    # 部分完成即可汇总
```

最终一次性评 test（**需先确认**）：复制 `batchA.yaml`、保留同一个 `name` 与 `project`、删掉
`splits`（改用各数据集 YAML 的 `test_splits`），再跑同一条命令——训练目录已有 `metrics.json`，
训练会被跳过，只做一次 test 评测。

加入经典基线作为下界参照（`pdr` / `mean_speed_heading`，见 [ALGORITHMS.md](ALGORITHMS.md) §3）：

```yaml
models:
  - {label: pdr, model: pdr, overrides: {orientation: reference}}          # “理想姿态 PDR”
  - {label: pdr_device, model: pdr}                                         # 需要 imu/orientation
  - {label: mean_speed_heading, model: mean_speed_heading,
     overrides: {orientation: reference}}
```

前提：`pdr` / `mean_speed_heading` 的前向轴按序列根属性 `body_frame` 查表，目前内置
`android_device` / `ios_device`；其他机体系必须显式配置
（`model_args={body_axes: {<body_frame>: [[fx,fy,fz],[ax,ay,az]]}}`），否则该数据集会报错。
`orientation=device` 只能用于提供了 `imu/orientation` 的数据集。

### `ipb report` — 汇总

```bash
ipb report runs=runs/benchmark/main out=reports/main
ipb report runs=runs/val metrics=[ate,rte,pde,plr] plots=false
```

递归查找所有评测目录（含 `metrics.json` 与 `sequences.csv`，训练目录除外），输出
`per_sequence.csv`、`summary.csv`（均值/中位数/标准差/分组 bootstrap CI/种子间标准差）、`summary.md`、`summary.tex`、
`wilcoxon.csv`（同一数据集划分上的模型两两配对检验）、`report.json` 与箱线图、误差 CDF、长度比散点、参数量–精度帕累托图。
统计定义见 [METRICS.md](METRICS.md) 第 5 节。

汇总前先核对所有 run 的 `metrics.json` → `protocol` 块：评测协议键（`eval_stride`、`metric_dims`、
`rte_delta`、`t_rte`、`d_rte`、`min_speed`）必须完全一致，否则以退出码 `2` 报错并指出差异键与目录；
旧结果没有 `protocol` 块时同样报错（重跑评测即可）。一致的协议写入 `report.json` 的 `protocol` 字段。

### `ipb info` / `ipb cfg` / `ipb version`

```bash
ipb info                                  # 已有模型配置、数据集配置、转换器与数据根目录
ipb info model=ronin_resnet18             # 结构参数、输入规格、参数量、FLOPs、论文与仓库
ipb info data=ronin                       # 数据集路径、划分映射、转换统计与指纹
ipb cfg model=ronin_resnet18 recipe=official   # 打印合并后的完整配置
```

## 3. 输出目录

```text
runs/<mode>/<name>/            # mode = train / val / predict；name 缺省 exp，已存在时 exp2、exp3…（exist_ok=true 则复用）
├── args.yaml                  # 完整解析后的配置，外加 protocol 块（生效的输入规格与评测协议）
├── env.json                   # git 提交与是否有未提交修改、依赖版本、GPU、数据集指纹与 dataset.json 哈希
├── log.txt
├── weights/{best,last}.pt     # 训练：模型配置、输入规格、完整 cfg、epoch、优化器/调度器状态（best 去掉优化器）、git 提交
├── results.csv                # 训练：每轮 train/loss、val/*、lr、fitness、耗时
├── metrics.json               # 聚合指标（mean/median/std/count）、protocol、privileged_inputs、calibration、
│                              # 效率指标、train_eval_gap（train/eval 模式的窗口损失比，见第 4.2 节）
├── sequences.csv              # 逐序列指标
├── predictions/<seq>.npz      # 逐帧预测/参考/oracle 轨迹，窗口级 vel_pred/vel_target/(logstd)，其他模型输出 out_<键>
└── plots/*.png                # 轨迹叠加、误差 CDF、误差随时间、箱线图、长度比、训练曲线
```

## 4. 全部配置键（`default.yaml`）

| 键 | 缺省 | 说明 |
|---|---|---|
| **任务** | | |
| `mode` | `train` | train / val / predict / benchmark |
| `model` | `ronin_resnet18` | 模型名（cfg/models/<name>.yaml）、模型 YAML 或 checkpoint (.pt) |
| `model_args` | `{}` | 覆盖模型 YAML 中 args 的结构参数 |
| `data` | `（空）` | 数据集名（cfg/datasets/<name>.yaml）、数据集 YAML 或已转换目录 |
| `split` | `val` | val / predict 使用的划分（逻辑名或 splits/<file>.txt 的文件名） |
| `source` | `（空）` | predict 的输入：序列 .h5 文件或目录 |
| `recipe` | `unified` | 训练配方：official（论文/官方仓库）/ unified（benchmark 统一预算） |
| `pretrained` | `（空）` | 仅加载模型权重的 checkpoint |
| `resume` | `false` | true（续训 <project>/train/<name>/weights/last.pt）或 last.pt 路径 |
| **训练** | | |
| `epochs` | `100` | `train_windows_budget > 0` 时由框架改写，见第 4.4 节 |
| `train_windows_budget` | `0` | >0 时启用等窗口预算：固定“看过的训练窗口总数”，框架据此算 `epochs`（0 = 关闭） |
| `budget_max_epochs` | `200` | 等窗口预算换算出的 epoch 上限（下界固定为 1） |
| `batch` | `128` |  |
| `lr` | `0.001` |  |
| `optimizer` | `adamw` | sgd / adam / adamw |
| `momentum` | `0.9` | sgd 动量；adam/adamw 的 beta1 |
| `weight_decay` | `0.0001` |  |
| `scheduler` | `cosine` | none / cosine / step / plateau |
| `lr_final` | `0.01` | cosine 调度的最终学习率 = lr × lr_final |
| `step_size` | `30` | step 调度周期（epoch） |
| `gamma` | `0.1` | step / plateau 的衰减系数 |
| `plateau_patience` | `10` | plateau 调度的耐心（验证次数） |
| `warmup_epochs` | `0` | 线性预热轮数 |
| `grad_clip` | `0.0` | 梯度范数裁剪阈值，0 表示关闭 |
| `loss` | `（空）` | 覆盖模型默认损失：mse / mse_sum / mse_l1 / gaussian_nll / mse_then_nll / nll_detach_then_nll |
| `loss_switch_epoch` | `10` | mse_then_nll / nll_detach_then_nll 切换到完整 NLL 的轮次（0 起） |
| `patience` | `30` | 早停：fitness 连续多少个 epoch 无改善即停止，0 表示关闭 |
| `seed` | `0` |  |
| `deterministic` | `true` | 固定算法以逐位复现（**可复现性优先**，默认保持开启）；与 `amp=true` 同时开启时小 batch 上很慢，见下 |
| `device` | `（空）` | 空 = 自动（优先 CUDA）；cpu / 0 / cuda:0 |
| `workers` | `4` |  |
| `amp` | `false` | 自动混合精度，仅 CUDA 生效；与 `deterministic=true` 叠加是四种组合里最慢的，因此默认关闭，见下 |
| `cache` | `true` | true 把序列载入内存；false 逐窗口读取 HDF5（train 与 val 划分都生效） |
| `val_interval` | `1` | 每多少个 epoch 验证一次（最后一轮总会验证） |
| `fitness` | `ate` | 模型选择**指标名**（越小越好），只在 val 上计算；ate / rte / loss / vel_rmse … |
| `fitness_stat` | `mean` | `fitness` 的跨序列聚合统计量：mean / median，见第 4.3 节 |
| `save_period` | `0` | 每多少个 epoch 额外保存 epoch<k>.pt，0 表示关闭 |
| `val_batch` | `512` | 推理批大小 |
| **任务视图（DESIGN 第 3 节）** | | |
| `window` | `200` | 窗口样本数（200 Hz） |
| `stride` | `10` | 训练滑窗步长 |
| `eval_stride` | `10` | 验证/推理步长 |
| `frame` | `gravity_world` | gravity_world / body / gravity_yaw_local |
| `orientation` | `reference` | reference / device |
| `remove_gravity` | `false` |  |
| `target` | `avg_velocity` | avg_velocity / displacement / velocity_at_end / frame_velocity（逐帧） / multi_displacement（多步） |
| `dims` | `2` | 2（水平）/ 3 |
| `output_steps` | `0` | `multi_displacement` 的步数 H（窗口等分为 H 段） |
| `overlap` | `mean` | 重叠预测的合并策略：mean（同一时刻取平均）/ center（取最靠窗口中心的） |
| `history` | `0` | 历史子窗口个数 H_in（0 = 不使用历史，输入 `(C,T)`；>0 时输入 `(H_in,C,T)`） |
| `history_stride` | `0` | 子窗口起点间隔（样本），`history > 1` 时必须 ≥ 1 |
| `extra_inputs` | `[]` | 额外输入：orientation / gravity / init_velocity（**特权输入**，结果中标记并单列） |
| `augment` | `[random_yaw, time_shift]` |  |
| `rate` | `200.0` | 统一采样率（Hz），与 IPB v1 数据一致 |
| **评测（DESIGN 第 5–6 节）** | | |
| `metric_dims` | `2` | 轨迹与速度指标使用的维度 |
| `rte_delta` | `60.0` | RTE 的时间窗（秒） |
| `t_rte` | `[1.0, 10.0]` | T-RTE 的时间窗（秒） |
| `d_rte` | `[10.0]` | D-RTE 的距离（米） |
| `min_speed` | `0.2` | 方向误差与沿/横向误差的速度阈值（m/s） |
| `save_predictions` | `true` | 保存 predictions/<seq>.npz |
| `plots` | `true` |  |
| `max_plots` | `12` | 最多绘制的单序列轨迹图数量 |
| `efficiency` | `true` | 记录参数量 / FLOPs / 单窗口延迟 |
| **输出** | | |
| `project` | `runs` | 输出根目录：<project>/<mode>/<name>/ |
| `name` | `（空）` | 运行名，缺省 exp，已存在时自动递增 |
| `exist_ok` | `false` |  |
| `verbose` | `true` |  |

### 4.1 `amp` 与 `deterministic` 的吞吐代价

`deterministic=true` 会调用 `torch.use_deterministic_algorithms`，迫使 cuDNN 选择确定性卷积算法；
该路径在半精度下尤其慢，与 `amp=true` 叠加后小 batch 上出现明显反效果。A100 80GB（与他人共享）上
`ronin_resnet18`、`window=200` 的训练步吞吐实测（窗口/s，`tools` 无关的纯前反向微基准）：

| `deterministic` | `amp` | batch 128 | batch 512 |
|---|---|---:|---:|
| true | true | **2 670** | 16 232 |
| true（默认） | false（默认） | 6 722 | 15 242 |
| false | true | 5 927 | 20 337 |
| false | false | 6 958 | 17 921 |

端到端复核（RIDI，96 084 窗口/轮，batch 128，workers 8）：`amp=true` 57.3 s/轮，`amp=false` 28.4 s/轮。

**因此 `amp` 的缺省值是 `false`**：`deterministic=true` + `amp=true` 是四种组合里最慢的一种
（batch 128 下 2 670 窗口/s，比 `amp=false` 的 6 722 慢 2.5 倍），而 IPB 把逐位可复现放在吞吐之前，
所以保留 `deterministic=true`、关掉 `amp`。想用混合精度时显式 `amp=true`；若同时要吞吐又不需要
逐位复现，用 `deterministic=false amp=true`（batch 512 下 20 337 窗口/s，是四种组合里最快的）。
两者都不改变配方超参数，`batch` 属于配方，不应为了速度单独调大。

### 4.3 `fitness` 与 `fitness_stat`

选模标量由**两个正交的键**决定：`fitness` 只写**指标名**（越小越好），`fitness_stat` 决定该指标
怎么跨序列聚合——`mean`（缺省，逐序列均值）或 `median`（逐序列中位数）：

```bash
ipb train data=ridi fitness=ate                       # 缺省：逐序列 ATE 的均值
ipb train data=ridi fitness=ate fitness_stat=median   # 逐序列 ATE 的中位数
```

什么时候该用 `median`：**val 划分小、构成偏斜**时。例如 RIDI 自动生成的 val 只有 1 名受试者，
其中 2 条序列的携带方式在 train 中根本不存在——这两条的误差远高于其余序列，均值几乎完全由它们
决定，选模信号非常噪；中位数不受少数极端序列主导，在这种划分上更稳。反过来，val 划分大且均衡时
均值用到全部序列的信息、方差更小，所以**缺省仍是 `mean`**，改用中位数需要显式声明并在结果中注明。

注意：
- 统计量是独立的键，**不是**指标名的后缀。因此 `dir_err_median`（序列内窗口角度的中位数）仍然是一个
  普通指标名，`fitness=dir_err_median fitness_stat=median` 的含义没有歧义；
- `ate_oracle` 与比值型指标（`plr` 等）都不能做 `fitness`（前者与模型无关，后者不是越小越好），
  配置解析时直接报错；`fitness_stat` 的取值同样在解析时校验；
- `loss` 是按窗口数加权的窗口级损失，`fitness_stat=median` 对它没有额外意义（仍按均值字典取值）；
- `results.csv` 的**列集合固定**，不随这两个键变化：每轮只写一列 `fitness`（本轮用于选模的标量），
  其含义由 `args.yaml` 里记录的 `fitness` / `fitness_stat` 决定——这样跨 run 汇总不会错位。

### 4.4 `train_windows_budget`：等窗口预算

固定 epoch 数在规模差一个数量级的数据集之间不是等算力预算。实测每轮训练窗口数
（`window=200`、`stride=10`、只算窗口内全部样本有效的窗口）：

| 数据集 | train 划分时长 | 每轮训练窗口数 |
|---|---:|---:|
| RIDI | 1.35 h | 96 084 |
| IMUNet | 4.84 h | 346 977 |
| RNIN | 6.59 h | 470 135 |
| PedLocData | 约 89 h（全集） | 百万级 |

按固定 `epochs` 跑，算力就按数据集大小分配；`train_windows_budget` 改为固定**每个 run 看过的
训练窗口总数** `B`，epoch 数由框架计算：

```text
epochs = clip(ceil(B / 每轮训练窗口数), 1, budget_max_epochs)
```

```bash
ipb train model=ronin_resnet18 data=ridi   train_windows_budget=3.2e+7   # → 334 轮
ipb train model=ronin_resnet18 data=rnin   train_windows_budget=3.2e+7   # → 69 轮
```

- “每轮训练窗口数”即日志里 `train: <n> windows` 的数字（`len(train_set)`，窗口内含无效样本的
  窗口已被剔除）；换算发生在 `Trainer.setup()` 里、写 `args.yaml` 之前，因此 `args.yaml` 的
  `epochs` 就是实际跑的轮数，`metrics.json` 的 `train_budget` 块另记
  `{train_windows_budget, windows_per_epoch, epochs_uncapped, epochs, budget_max_epochs,
  planned_windows}`，可以审计“这个 run 到底看过多少窗口”。
- 上限被触发（`epochs < epochs_uncapped`）时告警，`planned_windows` 会小于预算——此时这个 run
  **没有**用完预算，跨数据集不再等预算，必须调大 `budget_max_epochs` 或在报告中说明。
- 这是**配置层**能力：模型 YAML 不得为此硬编码 `epochs`（`official` 配方里的 `epochs` 是论文
  超参数，与本键无关；同时给出两者时 `train_windows_budget` 优先）。
- 序列级模型（`pdr`、`mean_speed_heading` 等 `SequenceModel`）只在 train 划分上标定一次，
  不适用等窗口预算：框架记录一条 info 并保留配方里的 `epochs`。
- `patience`（早停）照常生效：预算给出的是**上界**，实际轮数以 `results.csv` 为准。
- YAML 里写科学计数法要带指数符号（`3.2e+7`）；PyYAML 会把 `3.2e7` 当字符串。

### 4.2 `train_eval_gap`：train/eval 失配诊断

dropout 与 BatchNorm 让同一组权重在 `train()` 与 `eval()` 下行为不同。训练结束时框架在 val 的一条
序列上用两种模式各算一次窗口损失，把 `{loss_eval, loss_train, ratio}` 写进 `metrics.json` 的
`train_eval_gap`；`ratio > 2` 时给出告警。诊断在模型副本上进行，不改动权重、BN 统计与随机数状态。

`ratio` 明显大于 1 意味着**报出的指标不是这组权重真实的能力**。实测例子：`ronin_resnet18` 在
`recipe=unified` 下训练 4 轮后，RoNIN val 窗口损失 eval 模式 0.155、train 模式 0.020（比值 7.7），
速度幅值只有训练模式的 1/3（`speed_ratio` 0.35 对 0.94），于是 `plr` 掉到 0.32–0.35、ATE 虚高到 15 m。
根因是 RoNIN 头部的 `Dropout(0.5) → Linear → ReLU`：训练时 dropout 噪声让 ReLU 更容易激活，
关掉 dropout 后同一层的输出被系统性压低（逐项排除：把 BN 单独切回 train 模式、或重估 BN running
统计都不改善，只有 dropout 开着才恢复）。`plr` / `speed_ratio` 远离 1 是同一问题的下游表现。

## 5. Python 等价写法

```python
from inertial_benchmark import NIO

model = NIO("ronin_resnet18")
model.train(data="ronin", epochs=40, device=0)          # ipb train ...（trainer=Trainer 子类可定制训练流程）
result = model.val(data="ronin", split="test")          # ipb val ...（返回 RunResult）
traj = model.predict("seq.h5")                          # ipb predict ...（返回 Trajectory）
model.info(); model.benchmark()                         # 参数量 / FLOPs / 延迟

from inertial_benchmark.data.convert import convert_dataset
from inertial_benchmark.engine.benchmark import run_benchmark
from inertial_benchmark.utils.reports import build_report
```
