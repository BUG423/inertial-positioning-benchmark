# IPB v1 数据发布说明

> 状态：v1 首批发布（2026-09-17）｜格式规范：[DESIGN.md 第 2 节](DESIGN.md)｜命令行：[CLI.md](CLI.md)
> ｜指标：[METRICS.md](METRICS.md)
>
> 本文件描述**已转换数据的事实**：每个数据集接收了什么、划分怎么定、检查结果如何，以及如何在自己的机器上
> 复现同一份数据。**本仓库不提交任何数据文件**（`.h5`、划分文件、清单都在 `$IPB_DATASETS` 下，不入版本库）；
> 原始数据请各自从官方入口获取并遵守其许可。

本机数据根目录：`/workspace/webCodex/datasets/ipb`（环境变量 `IPB_DATASETS`）。
合计 **8 个数据集、3192 条序列、196.46 h、525.6 km**，统一 200 Hz、统一格式（IPB v1.0）。

## 1. 总表

| 数据集 | 序列（接收/发布） | 时长 (h) | 距离 (km) | 平台 / 携带方式 | 真值来源 | 默认划分 train/val/test | 官方划分是否泄漏 | 许可 | 转换器 | 已知限制（详见数据卡片） |
|---|---:|---:|---:|---|---|---|---|---|---|---|
| [ronin](datasets/ronin.md) | 151 / 152 | 23.11 | 61.94 | 安卓手机，自然携带（handheld/pocket/bag，`mixed`） | 胸前 Tango 手机的 VIO（**另一刚体**）；参考姿态为起点对齐后的手机姿态 | 71 / 16 / 64（= `test_seen` 32 + `test_unseen` 32） | 否（`test_seen` 与 train 共享受试者属官方 seen-subject 设定） | RoNIN Data License（非商业科研，禁止再分发） | `ronin@1.0` | 只公开约 50%；位置是胸前设备的位置；参考姿态有偏航漂移（官方末端误差最大 19.6°） |
| [ridi](datasets/ridi.md) | 94 / 94 | 2.71 | 10.30 | 单台 Tango 手机；handheld / pocket(leg) / bag / body | 同设备 Tango VIO | 43 / 6 / 23 | 否（train/test 按序列划分，受试者跨划分） | 未声明（学术使用，引用论文） | `ridi@1.1` | 加速度计 1–3% 比例误差；IMU 已被官方插值到位姿时间；22 条未列出序列作为附加测试子集 |
| [oxiod](datasets/oxiod.md) | 126 / 128 | 13.60 | 36.90 | iPhone 7 Plus / iPhone 5 / iPhone 6 / Nexus 5；handheld/pocket/bag/trolley + 慢走/跑 | Vicon 动捕（外参与时钟逐序列估计） | 54 / 7 / 9 | 否（官方 test 与 train 同属一个会话，组重叠） | 牛津大学学术/非商业使用 | `oxiod@1.1` | 外参/时钟为估计值；杆臂未知；`syn/`、`large scale/`、`test/` 未转换；57 条无官方划分的序列作为附加测试子集 |
| [tlio](datasets/tlio.md) | 354 / 354 | 31.60 | 37.63 | VR 头显（Bosch BMI055），`head` | 头显上的视觉惯性滤波（MSCKF），非独立真值 | 283 / 35 / 36 | 否（随机划分，9 台设备跨划分） | CC BY-NC 4.0 | `tlio@1.0.0` | IMU 已被参考滤波器补偿（“纯惯性”结果偏乐观）；无受试者标签；大量低速/原地序列 |
| [idol](datasets/idol.md) | 130 / 130 | 19.89 | 64.94 | iPhone 8 固定在 Kaarta Stencil 上，`handheld` | Stencil 的 LiDAR-视觉-惯性 SLAM（位置已被作者平滑） | 38 / 5 / 87（= `test_known` 51 + `test_unknown` 36） | 否（known 与 train 共享受试者是 known 的定义） | CC BY 4.0 | `idol@1.0.0` | 世界系倾斜（已调平）与时变残余倾斜；5 条序列开头 3–6 s 世界系未收敛；位置被平滑；需 `pyarrow` |
| [rnin](datasets/rnin.md) | 300 / 301 | 8.49 | 25.56 | 多款安卓手机，5 名采集者 | 逐序列不同：外部 gt（75 条，“VICON 或其他设备”）或 BVIO（226 条） | 240 / 51 / 9 | 否（官方目录划分，启发式会话跨划分） | 未单独声明（仓库 Apache-2.0，数据知识产权归商汤） | `rnin@1.1.0` | 采样率与时钟不统一（200/250/400 Hz）；`train_179` 与 `train_178` 逐字节重复（已拒收）；许可不明确 |
| [imunet](datasets/imunet.md) | 113 / 126 | 7.91 | 30.80 | Tango / Galaxy S10 / S21 / 小米，4 名受试者 | 同设备 Tango VIO 或 ARCore VIO（原生约 30 Hz，官方插值到 200 Hz） | 65 / 16 / 32 | 否（按序列划分，受试者跨划分） | 未声明（学术使用，引用论文） | `imunet@1.0` | 小米 11 条加速度计比例 0.90–0.92（整类拒收，不做非官方校正）；1 条逐字节重复已拒收；ARCore 位姿为插值 |
| [pedlocdata](datasets/pedlocdata.md) | 1924 / 2161 | 89.14 | 257.58 | 手机 + 未公开的参考系统，27 名受试者（YT）+ Demo | 未公开来源（Zenodo 只写 “Ground truth position/quaternion”；Demo 文件名暗示 Livox Mid-360） | 1315 / 399 / 210（**分组划分**） | **是**（按切片随机分配，329/474 个录制跨划分；官方划分保留为 `official_*.txt`） | CC BY 4.0 | `pedlocdata@1.1.0` | 官方划分泄漏；参考姿态在 5–10 s 尺度漂移（237 条切片被拒收）；真值来源未公开 |

拒收合计 1/152（RoNIN）、0/94（RIDI）、2/128（OxIOD）、0/354（TLIO）、0/130（IDOL）、1/301（RNIN）、
13/126（IMUNet）、237/2161（PedLocData）；每条的原因都在对应数据卡片的“被拒序列”一节与
`conversion_report.json` 中。

### 1.1 附加测试子集与官方子集

| 数据集 | 划分文件 | 序列 | 时长 (h) | 含义 |
|---|---|---:|---:|---|
| ronin | `test_seen.txt` / `test_unseen.txt` | 32 / 32 | 4.93 / 5.07 | 官方 seen / unseen 受试者测试集（`test` 为二者并集） |
| ridi | `test_unseen_subject.txt` | 11 | 0.30 | 官方列表外、受试者未出现过（`ruixuan`、`shali`）；`list_crosssubject.txt` 的超集 |
| ridi | `test_unlisted_seen_subject.txt` | 11 | 0.29 | 官方列表外、受试者出现过（seen-subject 设定） |
| oxiod | `test_unseen_subject.txt` | 30 | 2.96 | multi users（user2–user5，官方场景只有 user1） |
| oxiod | `test_unseen_device.txt` | 26 | 2.04 | multi devices（iPhone 5 / iPhone 6 / Nexus 5） |
| idol | `test_known.txt` / `test_unknown.txt` | 51 / 36 | 8.16 / 5.93 | 受试者是否出现在 train 中（`test` 为二者并集） |
| idol | `test_known_building1.txt` / `test_unknown_building1.txt` | 15 / 14 | 2.36 / 2.15 | 论文 Table 2 的 building1 设定 |
| pedlocdata | `test_yt.txt` / `test_demo.txt` | 204 / 6 | 7.87 / 0.41 | 默认 `test` 按来源文件拆分 |
| pedlocdata | `official_{train,val,test,test_yt,test_demo}.txt` | 1340 / 390 / 194 / 188 / 6 | 61.39 / 18.16 / 9.59 / 9.09 / 0.50 | **泄漏的**官方划分，只用于与文献对照 |

附加测试子集绝不并入 `train`；不属于任何默认划分的已转换序列会列在 `dataset.json` 的 `unassigned`
（本批全部为空）。

### 1.2 数据指纹（可复现性）

同一原始数据 + 同一转换器版本 + gzip(4)+shuffle 压缩 ⇒ 逐字节相同的输出，因此指纹可直接比对：

| 数据集 | 转换器 | `dataset.json` 的 `fingerprint`（前 16 位） |
|---|---|---|
| ronin | `ronin@1.0` | `f29e5c58e71935cc` |
| ridi | `ridi@1.1` | `6e47893cb3b6dbf0` |
| oxiod | `oxiod@1.1` | `559c7c3de007887e` |
| tlio | `tlio@1.0.0` | `c2d0ddbe9d6ea11f` |
| idol | `idol@1.0.0` | `dadeed68bbc3f119` |
| rnin | `rnin@1.1.0` | `6d7f93e5734c88b5` |
| imunet | `imunet@1.0` | `3fe0f633264ab36a` |
| pedlocdata | `pedlocdata@1.1.0` | `79b8e288ac4be804` |

## 2. 划分策略（DESIGN 2.4 的落地）

1. **官方划分优先**：7 个数据集直接使用官方列表/目录；官方没有 val 时，由统一流水线从 train 中按
   `group_id` 分组抽取约 10%（固定种子 0，按时长加权），方法与实际比例记在 `dataset.json` 的 `split_method`。
   本批生成 val 的是 ridi（受试者 `ma`，占 train 时长 11.5%）、oxiod（会话 `slow_walking_data1`，14.5%）、
   imunet（受试者 `subject2`，17.2%）、idol（受试者 `subject03`/`subject05`，13.3%）；组的粒度较粗时实际比例
   会高于 10%（分组不可拆分）。
2. **泄漏优先于“官方”**：PedLocData 的官方划分把同一次录制的相邻切片分散到 train/valid/test
   （329/474 个 YT 录制、10/13 个 Demo 录制跨划分，26/27 名受试者出现在全部三个划分）。转换器声明
   `OFFICIAL_SPLITS_LEAK = True` 并提供 `grouped_splits()`，流水线把分组划分写为默认 `train/val/test`，
   官方划分降级为 `official_*.txt`。**不要**用默认 train 训练后在 `official_test` 上评测（两者共享序列）。
3. **不在官方划分中的序列不并入 train**：RIDI 22 条、OxIOD 57 条通过转换器的 `extra_splits()`
   写成附加测试子集（见 1.1）。
4. **seen-subject 设定**：RoNIN 的 `test_seen`、RIDI/IMUNet 的按序列划分、TLIO 的设备随机划分、
   RNIN 的会话重叠都会让 `group_id` 跨划分出现。这是官方设定，`ipb check` 报告为**警告**而不是错误；
   名字含 `unseen`/`unknown`/`novel` 的子集与 train/val 共享组才会被判为错误。评测报告中应注明所用划分是
   seen 还是 unseen 设定。

## 3. 校验结果

对每个数据集执行 `ipb check data=<name> full=true hash=true`（逐条读取 + 完整校验 + 重力检查 +
重算 sha256 与 IMU 内容哈希）：

| 数据集 | 结果 | 警告 | 说明 |
|---|---|---:|---|
| ronin | 通过 | 5 | 全部为 seen-subject 组重叠 |
| ridi | 通过 | 3 | 组重叠（含 `test_unlisted_seen_subject`/train） |
| oxiod | 通过 | 3 | 组重叠 2 条 + 1 条重力勉强通过（倾角 2.57°） |
| tlio | 通过 | 3 | 设备组重叠 |
| idol | 通过 | 33 | 6 条组重叠 + 22 条重力勉强通过（加速度计 +1.9% 比例误差）+ 5 条静止段/全段不一致 |
| rnin | 通过 | 15 | 3 条组重叠 + 12 条重力勉强通过（\|g\| 偏差 0.30–0.42 m/s²） |
| imunet | 通过 | 2 | 组重叠 |
| pedlocdata | 通过 | 3 | 官方划分的已声明泄漏（默认划分无任何重叠） |

八个数据集都是 **0 错误、无重复内容、无未分配序列**；逐条重力检查、时间轴均匀性、四元数与属性完整性全部通过。

## 4. 一致性抽检

每个数据集随机抽 5 条序列（固定种子 0），比较转换器解析结果（原生时钟）与 200 Hz 输出：
**1 s 分辨率路径长度差异 ≤ 0.16%（40 条中 36 条 < 0.01%）**，时长差 ≤ 5 ms，样本数与 ⌊时长×200⌋+1 完全一致，
重力对齐加速度均值差 ≤ 0.004 m/s²，把 200 Hz 结果插回原生时刻后的水平位置 RMS ≤ 1.1 mm。
逐序列的数字见各数据卡片的“转换结果”一节。唯一显著的差异是 RNIN 250 Hz 序列的**稠密**（逐样本）路径长度
低 1.2–1.6%：多相抗混叠滤波去掉了参考位置的高频抖动，这是统一采样率的预期效果，对 1 s 尺度指标无影响。

## 5. 如何自己转换

```bash
# 1) 安装（读取 IDOL 的 .feather 需要 idol 可选依赖；训练另需 train）
git clone <repo> && cd inertial-positioning-benchmark
pip install -e ".[idol]"            # 或 pip install -e ".[idol,train,test]"

# 2) 原始数据：按各数据卡片第 1 节的入口自行获取并解压（仓库不再分发）
#    本机布局（只读）：
#      /workspace/webCodex/datasets/imu_odometry/raw/_staging/{RoNIN,RIDI,OxIOD,TLIO,IDOL,RNIN}
#      /workspace/webCodex/datasets/imu_odometry/raw/IMUNet/IMUNet_dataset
#      /workspace/webCodex/datasets/imu_odometry/raw/PedLocData

# 3) 选定输出根目录
export IPB_DATASETS=/workspace/webCodex/datasets/ipb

# 4) 逐个数据集转换（workers 视机器而定；共享机器上全部进程总数建议 ≤ 32）
RAW=/workspace/webCodex/datasets/imu_odometry/raw
ipb convert dataset=ronin      source=$RAW/_staging/RoNIN          output=$IPB_DATASETS/ronin      workers=8
ipb convert dataset=ridi       source=$RAW/_staging/RIDI           output=$IPB_DATASETS/ridi       workers=8
ipb convert dataset=oxiod      source=$RAW/_staging/OxIOD          output=$IPB_DATASETS/oxiod      workers=8
ipb convert dataset=tlio       source=$RAW/_staging/TLIO           output=$IPB_DATASETS/tlio       workers=16
ipb convert dataset=idol       source=$RAW/_staging/IDOL           output=$IPB_DATASETS/idol       workers=16
ipb convert dataset=rnin       source=$RAW/_staging/RNIN           output=$IPB_DATASETS/rnin       workers=16
ipb convert dataset=imunet     source=$RAW/IMUNet/IMUNet_dataset   output=$IPB_DATASETS/imunet     workers=8
ipb convert dataset=pedlocdata source=$RAW/PedLocData              output=$IPB_DATASETS/pedlocdata workers=24

# 5) 校验（逐条读取 + 重力检查 + 哈希；退出码 0 表示通过）
for d in ronin ridi oxiod tlio idol rnin imunet pedlocdata; do
  ipb check data=$d full=true hash=true save=/tmp/check_$d.json
done

# 6) 核对指纹与本文件 1.2 节的表格是否一致
ipb info data=tlio
```

要点与注意事项：

- **原始数据只读**：转换器不会写入 `source`；不要把原始数据或转换结果提交到版本库（`.gitignore` 已排除 `*.h5`）。
- **磁盘**：转换结果共 7.1 GB（pedlocdata 2.9 GB、tlio 1.2 GB、ronin 921 MB、idol 796 MB、oxiod 545 MB、
  rnin 342 MB、imunet 313 MB、ridi 113 MB）。原始数据另需约 61 GB（RoNIN 19 GB、PedLocData 15 GB、
  TLIO 8.5 GB、IMUNet 5.5 GB、RNIN 5.2 GB、RIDI 4.0 GB、OxIOD 2.7 GB、IDOL 1.5 GB）。
- **耗时**：本机（256 核共享）每个数据集 1–5 分钟；PedLocData（2161 条切片）在 24 进程下约 4 分钟，
  8 个数据集的 `full=true` 校验并行跑完约 2.5 分钟。
- **单条失败不影响整体**：转换器解析异常或子进程被杀时，该序列记为 `rejected`（原因写入
  `conversion_report.json`），清单与报告照常写出；`overwrite=false` 可续转。
- **部分转换**：`only=[seq1,seq2]` 只转换指定序列（会与上次的报告合并）。
- **真实数据抽样测试**（数据缺失时自动跳过）：
  `PYTHONPATH=src python -m pytest tests/data`，它会用原始文件核对列含义、约定、重复表与拒收清单。
- **重现划分**：划分只依赖官方列表、`group_id` 与固定种子，不依赖转换顺序或机器；
  `dataset.json` 的 `split_method` 记录每个划分的来源。

## 6. 数据发布内容

每个数据集目录（`$IPB_DATASETS/<name>/`）包含：

```text
dataset.json            # 版本、转换器、采样率、划分策略与方法、按划分统计、逐序列 sha256 与 imu_sha256、指纹
sequences/<id>.h5       # IPB v1.0 序列（200 Hz，gzip(4)+shuffle，字节可复现）
splits/*.txt            # train/val/test + 官方子集 + 附加测试子集 (+ official_* )
conversion_report.json  # 逐条接收/拒收原因、警告、裁剪量、物理自检统计、泄漏与重复检查
```

序列内容与属性的完整契约见 [DESIGN.md 2.2](DESIGN.md)；每条序列的 `notes`（`conversion_notes` 属性）
记录了所有已应用的修正（单位、坐标系、调平、时延、裁剪）与物理自检数值，便于审计。
