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

训练只使用 `train` 与 `val` 划分；每 `val_interval` 轮在 val 上计算全部指标，按 `fitness`（缺省 `ate`，越小越好）保存 `best.pt`。
结束后用 best 权重在 val 上评测一次并写出 `metrics.json`。

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
`ronin_resnet18` × 八个核心数据集 × 种子 0/1/2）。

```yaml
name: main
project: runs/benchmark            # 输出 <project>/<name>/<label>/<dataset>/seed<k>/{train,<split>}/
models:
  - ronin_resnet18                   # 名称 / 模型 YAML / checkpoint（checkpoint 不再训练）
  - {label: ronin_official, model: ronin_resnet18, overrides: {recipe: official, epochs: 100}}
datasets: [ronin, ridi, {data: /data/ipb/oxiod, overrides: {batch: 256}}]
seeds: [0, 1, 2]
splits:                            # 缺省为数据集 YAML 的 test_splits
overrides: {epochs: 100, device: 0}
report: true                       # 结束后汇总到 <project>/<name>/report
```

完成标志为 `train/metrics.json` 与 `<split>/metrics.json`：重复运行会跳过已完成项；训练中断（有 `last.pt` 无 `metrics.json`）会自动续训；
数据集缺少某个测试子集时跳过并告警。进度写入 `<project>/<name>/benchmark.json`。

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
├── metrics.json               # 聚合指标（mean/median/std/count）、protocol、privileged_inputs、calibration、效率指标
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
| `epochs` | `100` |  |
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
| `deterministic` | `true` |  |
| `device` | `（空）` | 空 = 自动（优先 CUDA）；cpu / 0 / cuda:0 |
| `workers` | `4` |  |
| `amp` | `true` | 自动混合精度，仅 CUDA 生效 |
| `cache` | `true` | true 把序列载入内存；false 逐窗口读取 HDF5 |
| `val_interval` | `1` | 每多少个 epoch 验证一次（最后一轮总会验证） |
| `fitness` | `ate` | 模型选择指标（越小越好），只在 val 上计算；ate / rte / loss / vel_rmse … |
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
