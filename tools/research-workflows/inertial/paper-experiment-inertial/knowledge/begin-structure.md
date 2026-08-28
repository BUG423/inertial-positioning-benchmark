# Begin 代码结构 (先验知识)

## 仓库概览

Begin 是基于 IMU 的行人航位推算研究代码库。任务: 6 轴 IMU (3 陀螺+3 加速度) → 3D 速度向量。

```text
Begin/
├── bash/              # 训练启动脚本 (冒烟测试 / baseline 一键训练 / 单算法脚本)
├── load_dataset/      # 数据集加载与预处理 (6 数据集 + 统一入口 + 缓存机制)
├── network/           # 神经网络模型库 (对比算法 + 自研算法)
├── plot/              # 结果可视化 (轨迹图/CDF/PDE/性能对比/特征分析)
├── train_test/        # 训练/测试主流程 (main.py + Train_Test.py + losses.py)
├── pyproject.toml     # uv 依赖管理
└── README.md
```

## 核心概念

- **任务**: 6 轴 IMU → 3D 速度向量
- **输入/输出**: `feat (B, 6, T)` → `pred (B, 3)`，T 默认 200 帧
- **坐标系**: Global (世界系速度) 或 Body (载体系速度)
- **评估指标**: ATE (绝对轨迹误差)、RTE (相对轨迹误差)、T-RTE、D-RTE
- **环境**: uv 管理依赖，`uv sync` 一键安装，`uv run python` 运行

## network/ — 模型库结构

每个算法一个目录，固定结构:

```text
network/<算法名>/
├── README.md          # 论文信息 + 核心思想 + arch 名→文件映射表
└── model/
    ├── model.py       # 核心模型 (或其他命名)
    └── components.py  # 可选: 内部组件
```

模型工厂 `network/model_factory.py` 通过 `args.network` 用相对导入创建模型:
```python
elif arch == "算法名":
    from .<算法名>.model.<文件> import <类名>
    network = <类名>(input_dim, output_dim, ...)
```

模型接口契约:
- 构造参数: 从 `args` / `input_dim` / `output_dim` / `net_config` 获取
- 前向签名: `forward(feat, targ)` — `feat (B,6,T)`, `targ (B,3)`
- 输出: `pred (B,3)` 或 `(pred, pred_cov)` 二元组

已收录算法 (24+):
- 惯性里程计专用: RoNIN (3 变体), TLIO, RNIN, CTIN, IMUNet, EqNIO, DeepILS, NanoMST, iMOT, SBIPTVT
- 通用高效骨干: MobileNet/MobileNetV2/MnasNet/EfficientNet, MobileNetV3-Small, MobileNetV4-Medium, RepViT-M0.6, EfficientViM, ViT
- 自研: FTIN, TinyIO (含 deploy 和消融变体), IONext (7 变体), FDIO/MambaIO, StarNet-MCA, CKANIO, PostDiffIO, ModeMoEIO
- 消融变体通过不同 `arch` 名区分 (e.g. `TinyIO` / `TinyIO_deploy` / `TinyIO_No_DPAA`)

## train_test/ — 训练/测试主流程

三个文件:
| 文件 | 职责 |
|------|------|
| `main.py` | CLI 入口，argparse 解析参数，训练循环，checkpoint 保存/加载 |
| `Train_Test.py` | `train_f()` / `test_f()` 单 epoch 逻辑，`process_metrics()` 指标计算，轨迹可视化 |
| `losses.py` | 算法专用损失 (TLIO NLL / RNIN 两阶段 / CTIN 马氏距离 / 其余默认 MSE) |

关键 CLI 参数 (argparse):
| 参数 | 默认 | 说明 |
|------|------|------|
| `--network` | TLIO-resnet18 | 算法 arch 名 |
| `--dataset_type` | RoNIN | 数据集类型 |
| `--Coordinate` | Global | 坐标系 |
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 1024 | 批大小 |
| `--lr` | 1e-4 | 初始学习率 (Adam + ReduceLROnPlateau) |
| `--window_size` | 200 | 滑窗长度 |
| `--out_dir` | None | 输出目录，存 checkpoint/日志/指标 |
| `--resume` | True | 断点续训 |
| `--test` | False | 仅推理 |
| `--step_size` | — | 滑窗步长 |
| `--seed` | — | 随机种子 |
| `--device` | — | 设备 |
| `--cache_path` | — | 特征缓存目录 |

checkpoint 结构: `{model_state_dict, epoch, optimizer_state_dict, scheduler_state_dict}`
重参数化网络 (TinyIO) 额外在 `<out_dir>_deploy/` 保存部署态权重。

## load_dataset/ — 数据加载

支持 6 数据集: RoNIN, RIDI, OxIOD, IMUNet, TLIO, RNINv2

统一入口: `load_dataset/get_dataset.py` → `get_dataset(args)` 返回 `(train_loader, val_loader, test_loader)`
返回固定: `(dataloader_list, input_dim=6, output_dim=3, net_config=7)`

数据目录约定:
```text
<DATA_ROOT>/<数据集>v2/
├── all/          # 所有样本子目录
├── train.txt     # 每行一个样本名
├── val.txt
└── test.txt
```

## bash/ — 启动脚本

| 脚本 | 用途 |
|------|------|
| `baseline.sh` | 一键训练全部第三方对比算法 (数据集×算法 笛卡尔积) |
| `smoke_dataset.sh` | 数据加载冒烟测试 (1 epoch + 大步长) |

脚本范式: 配置区 (DATA_ROOT/算法/超参) → 执行区 (`for` 循环调 `main.py`)
自研算法用最小模板: 改 `NETWORK` 名 + `DATA` 路径 + 超参 → 调 `main.py`

## plot/ — 可视化

6 个子目录:
| 子目录 | 内容 |
|--------|------|
| `analysis/` | 指标统计 (均值/标准差表格) |
| `traj/` | 轨迹可视化 (全量总览 + 精选网格图) |
| `cdf_pde/` | CDF 曲线与 PDE 分布图 |
| `perf/` | ATE vs 参数量散点图/折线图/雷达图 |
| `feature/` | ERF/t-SNE/PCA/频率分析 |
| `utils/` | FLOPs 计算 / SQLite→CSV 导出 |

所有脚本顶部有 `RESULTS_DIR` 变量指向 `results/`，首次使用须手动修改。

## results/ 输出结构

```text
results/<算法名>/<Coordinate>/<数据集>/<算法>/
├── checkpoints/    # checkpoint_best.pt + checkpoint_latest.pt
├── training.log    # 训练日志
└── ...             # 评估指标 JSON / 轨迹图
```
