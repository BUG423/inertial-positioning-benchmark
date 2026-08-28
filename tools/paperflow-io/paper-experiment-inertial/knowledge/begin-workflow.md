# Begin 操作工作流 (先验知识)

## 1. 新增算法全流程

### 1.1 建目录骨架
```text
network/<算法名>/
├── README.md
└── model/
    ├── model.py
    └── components.py  # 可选
```

### 1.2 实现模型代码
- 必须实现 `forward(feat, targ) → pred (B,3)` 或 `→ (pred, pred_cov)`
- `feat` 只取前 6 通道 (gyro 3 + acce 3)
- 若涉及重参数化: `deploy=True/False` 控制训练/推理模式
- 消融变体: 通过构造参数 (`use_xx=True/False`) 或独立类区分，不同变体在 model_factory 注册为不同 `arch`

### 1.3 写 README.md (固定模板)
```markdown
# 算法名
## 论文信息
- 标题 / 作者 / 出处
## 核心思想
<1-3 段中文>
## 本目录实现
| 工厂 arch 名 | 入口 | 文件 | 说明 |
| --- | --- | --- | --- |
| 算法名 | 类名(...) | model/model.py | 主模型 |
| 算法名_deploy | 类名(..., deploy=True) | model/model.py | 部署态 |
| 算法名_No_XX | 类名(..., use_xx=False) | model/model.py | 消融: 去 XX |
<目录树>
```

### 1.4 在 model_factory.py 注册
在 `get_model()` 中添加 `elif arch == "算法名":` 分支，用相对导入。每个消融变体各加一个分支。

### 1.5 自定义损失 (如需要)
1. `train_test/losses.py` 末尾加 `get_<算法>_loss` 函数
2. `Train_Test.py` 的 `train_f` **和** `test_f` **两处**加分支 — 漏一处训练和推理损失不同

### 1.6 冒烟验证
```bash
uv run python train_test/main.py \
    --network <算法名> --dataset_type RoNIN \
    --train_list <path>/train.txt --val_list <path>/val.txt \
    --test_seen_list <path>/test.txt --dataset_root_dir <path>/all \
    --out_dir results/smoke/<算法名> --epochs 1 --batch_size 64
```
预期: 无报错，checkpoint 正常落盘。有错查 `training.log`。

## 2. 全量训练流程 (单算法)

### 2.1 训练脚本模板 (bash/<算法名>.sh)
```bash
#!/bin/bash
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1

PYTHON="${PYTHON:-python3}"
DATA_ROOT="/path/to/data"   # ← 改
OUT_BASE="$REPO_ROOT/results/<算法名>"
SEED=42

for DS in "RoNIN" "TLIO" "IMUNet"; do
    DATA_SUB="${DATA_ROOT}/${DS}v2"
    $PYTHON train_test/main.py \
        --seed $SEED --network <算法名> --dataset_type $DS \
        --train_list "${DATA_SUB}/train.txt" \
        --val_list "${DATA_SUB}/val.txt" \
        --test_seen_list "${DATA_SUB}/test.txt" \
        --test_unseen_list "${DATA_SUB}/test.txt" \
        --dataset_root_dir "${DATA_SUB}/all" \
        --window_size 200 --batch_size 1024 --epochs 100 --lr 1e-4 \
        --device cuda:0 --Coordinate Global \
        --out_dir "${OUT_BASE}/Global/${DS}/<算法名>"
done
```

### 2.2 训练参数标准值 (Begin 默认)
| 参数 | 默认值 |
|------|--------|
| epochs | 100 |
| batch_size | 1024 |
| lr | 1e-4 |
| window_size | 200 |
| Coordinate | Global |
| optimizer | Adam (weight_decay=1e-5) |
| scheduler | ReduceLROnPlateau (factor=0.1, patience=10) |

## 3. Baseline 复现

### 3.1 一键全跑 (推荐先跑)
```bash
# 改 bash/baseline.sh 顶部 DATA_ROOT
PYTHON="uv run python3" bash bash/baseline.sh
```
脚本内 `algorithms` 数组含所有第三方对比算法。单个失败不中断，结束打印汇总。

### 3.2 单独跑某个 baseline
```bash
uv run python train_test/main.py \
    --network RoNIN-resnet18 \
    --dataset_type RoNIN \
    --train_list ... --val_list ... --test_seen_list ... \
    --dataset_root_dir ... \
    --out_dir results/baseline/Global/RoNIN/RoNIN-resnet18 \
    --epochs 100 --batch_size 1024 --lr 1e-4 --device cuda:0
```

## 4. 消融实验

每个消融变体在 model_factory 注册为独立 `arch`，用同一训练命令切换 `--network`:
```bash
# 主模型
--network <算法名>
# 消融: 去掉 XX
--network <算法名>_No_XX
# 消融: 只保留 YY
--network <算法名>_Only_YY
```

## 5. 结果分析

### 5.1 指标统计
```bash
RESULTS_DIR=<path/to/results/算法名> python plot/analysis/analysis_average.py
```
输出每个数据集×方法的 ATE mean±std

### 5.2 轨迹对比图
```bash
python plot/traj/plot_select_traj_<算法名>.py   # 如有专用脚本
python plot/traj/plot_all_trajectories.py        # 通用全量图
```

### 5.3 ATE vs 参数量散点图 (轻量化方向核心图)
```bash
python plot/perf/plot_ate_params.py
```

### 5.4 CDF 曲线
```bash
python plot/cdf_pde/plot_cdf_curves_<算法名>.py
```

## 6. 常见训练问题排查

| 症状 | 可能原因 | 检查方法 |
|------|---------|---------|
| 模型创建报错 | model_factory import 路径错 | 检查 arch 名拼写 + import 路径 |
| forward 报 shape mismatch | feat 只取前 6 通道？输出 shape 非 (B,3)？ | print feat.shape, pred.shape |
| ATE 比预期差 20%+ | window_size 设错？Coordinate 设错？ | 检查 main.py 报的 feat shape |
| checkpoint 不保存 | out_dir 权限/磁盘满 | `ls results/.../checkpoints/` |
| TinyIO 部署态 ATE 和训练态差太多 | 重参数化 merge 有 bug | 对比训练态和部署态的 pred |
| 内存不足 | batch_size 太大 | 降到 512 或 256 |

## 7. 默认配置速查

- **6 数据集**: RoNIN, RIDI, OxIOD, IMUNet, TLIO, RNINv2
- **16+ baseline**: RoNIN(3变体), TLIO, RNIN, CTIN, IMUNet, EqNIO, DeepILS, NanoMST, iMOT, SBIPTVT, MobileNet 系列(6), RepViT, EfficientViM, ViT
- **1×3090**: 所有 IO 模型训 RoNIN 100 epoch < 3 小时
- **固定输入**: 6 通道 IMU, 200 帧窗口, 3 维速度输出
