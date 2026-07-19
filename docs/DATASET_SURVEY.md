# 惯性定位公开数据集调研（第一版）

> 状态：持续更新  
> 最近核验：2026-07-19  
> 目标：为 Inertial Positioning Benchmark 确定数据集范围、接入优先级和统一格式需求。

## 1. 调研范围与判定标准

“包含 IMU 的定位数据集”并不等于“惯性定位数据集”。本调研按以下三层划分：

1. **核心惯性定位数据集**：原始加速度计/陀螺仪可作为主要输入，具有轨迹、速度或位姿真值，适合评测 IMU-only 方法。
2. **扩展平台数据集**：面向足绑、车辆、轮载、多 IMU、无人机等特定平台，可用于扩展 benchmark 的覆盖范围。
3. **辅助多模态数据集**：包含高质量 IMU 与真值，但原始目标通常是视觉—惯性、激光—惯性或多传感器融合。可用于跨域泛化，不能与纯惯性主榜直接混排。

纳入前需要逐项核验：

- 数据是否仍可下载，下载是否需要登录或签署协议；
- 原始 IMU、设备姿态、磁力计等字段的定义与单位；
- 时间戳时基、采样频率、丢帧和同步方式；
- 设备坐标系、世界坐标系、重力方向和四元数顺序；
- 真值来源、精度、覆盖范围及是否存在伪真值；
- 官方训练/验证/测试划分和被后续论文实际采用的划分；
- 数据与代码许可证是否允许再分发、转换格式和公开 benchmark 结果。

## 2. 第一阶段最值得接入的数据集

| 数据集 | 平台/设备 | 规模与多样性 | 真值 | 主要价值 | 初步优先级 |
|---|---|---|---|---|---|
| [RIDI](https://yanhangpublic.github.io/ridi/) | 手持/口袋/包内智能手机 | 约 150 分钟，200 Hz，多种人体运动与携带方式 | Google Tango 视觉—惯性轨迹 | 早期学习式惯性定位代表数据集；格式和规模适合首批适配 | P0 |
| [OxIOD](https://deepio.cs.ox.ac.uk/) | 4 类消费级手机；手持、口袋、手提包、推车 | 158 条序列，总距离超过 42 km；5 名用户，多速度模式 | 多数序列为光学动捕；长距离序列含其他参考来源 | 携带方式、设备、用户和运动模式较丰富，文献使用广泛 | P0 |
| [RoNIN](https://ronin.cs.sfu.ca/) | 人体携带智能设备 | 超过 40 小时、100 名受试者、自然人体运动 | 3D 轨迹真值 | 当前行人神经惯性导航最重要的公共基准之一；具有 seen/unseen 泛化设置 | P0 |
| [UTIAS Foot-Mounted INS](https://starslab.ca/foot-mounted-inertial-navigation-dataset/) | 足绑 IMU | 5 名受试者，总行程超过 7.6 km | 多种位置参考 | 足绑 ZUPT/INS 的高价值公共数据；可形成独立 foot-mounted 轨道 | P0 |
| [pyShoe Hallway](https://github.com/utiasSTARS/pyshoe) | 足绑 IMU | 39 次走/跑试验，3 条走廊 | 路径上的中间真值位置 | 数据、算法和代码结合紧密，适合作为足绑流水线的最小可运行样例 | P0 |
| [IPIN 2019 Track 4](https://zenodo.org/records/3937220) | 足绑 IMU | IPIN 2019 离线竞赛数据 | 竞赛配套参考与评分材料 | 有真实竞赛协议，适合验证 leaderboard 与盲测式评测设计 | P1 |
| [Pedestrian Inertial Navigation Dataset with Wearable Sensor](https://recherche.data.gouv.fr/en/dataset/pedestrian-inertial-navigation-dataset-with-wearable-sensor) | 可穿戴惯性传感器 | 2 名受试者、6 条真实路线、总计超过 2 km；校园、办公楼、城市、树林、商场停车场 | 数据页提供配套定位参考 | 场景跨度较好，但规模较小，适合外部测试 | P1 |
| SIMD | 智能手机 | 论文报告超过 4,500 条轨迹、约 190 小时、超过 700 km | 需进一步核验真值生成与公开下载内容 | 规模很大，若数据和许可完整，可能成为重要训练集 | P1，待核验 |

### 2.1 RIDI

RIDI 在 ECCV 2018 论文中提出。论文报告数据约 150 分钟、200 Hz，覆盖手机从手持到口袋、包内等放置变化；真值由 Google Tango 设备的视觉—惯性系统产生。它不是最精确或最大的集合，但具有以下 benchmark 价值：

- 是学习式行人惯性定位发展史上的关键数据集；
- 文件规模适中，适合先建立下载、校验、预处理和缓存机制；
- 后续 RoNIN、EqNIO 等工作仍将其用于跨数据集测试。

风险：真值属于视觉—惯性估计而非全程独立高精度动捕；需要核验各序列的手机姿态、坐标变换和时间同步定义。

主要来源：[项目主页](https://yanhangpublic.github.io/ridi/)、[ECCV 2018 论文](https://openaccess.thecvf.com/content_ECCV_2018/papers/Hang_Yan_RIDI_Robust_IMU_ECCV_2018_paper.pdf)、[官方代码](https://github.com/higerra/ridi_imu)。

### 2.2 OxIOD

OxIOD 是专门面向深度惯性里程计的数据集。官方论文报告：

- 158 条序列，总距离超过 42 km；
- 4 种携带方式：手持、口袋、手提包、推车；
- 4 种运动模式：停止、慢走、正常行走、跑步；
- 5 名用户、4 类消费级手机；
- 主要为室内场景；多数序列使用光学动捕提供位置、速度和姿态标签。

它适合评测携带方式变化、设备差异以及二维/三维输出的兼容性。需要特别注意：并非所有序列的真值来源和质量完全一致，不能在预处理时将其无差别合并。

主要来源：[官方数据页](https://deepio.cs.ox.ac.uk/)、[论文](https://arxiv.org/abs/1809.07491)。

### 2.3 RoNIN

RoNIN 将任务定义为从 IMU 序列估计移动主体的位置与方向。论文报告超过 40 小时的数据、100 名受试者以及自然人体运动下的 3D 轨迹真值。它的重要性不仅在规模，还在于强调跨人员和跨场景泛化。

首版实现时应保留官方序列级划分，禁止随机切窗后再随机分配训练/测试，否则同一轨迹会发生数据泄漏。benchmark 至少应分别报告：

- seen subjects/scenes；
- unseen subjects/scenes；
- 跨数据集 zero-shot 测试；
- 端到端轨迹指标与固定时间窗相对误差。

主要来源：[论文](https://arxiv.org/abs/1905.12853)。

### 2.4 SIMD

《Smartphone-Based Pedestrian Inertial Tracking: Dataset, Model, and Deployment》报告了一个大规模 Smartphone Inertial Measurement Dataset：超过 4,500 条行走轨迹、约 190 小时、总距离超过 700 km。其规模非常有吸引力，但目前必须先完成以下核验再决定是否接入：

- 官方下载入口是否长期可访问；
- 公开部分是否包含全部轨迹和足够的真值；
- 采集设备、采样率和时间同步细节；
- 数据许可是否允许建立自动下载器或只允许用户手动申请；
- 论文中的训练划分是否能够复现。

因此，SIMD 暂列 P1，而不是在仅凭论文规模描述的情况下直接列为 P0。

## 3. 足绑惯性导航轨道

足绑 IMU 与自由携带手机的运动约束完全不同：足部周期性静止使零速更新（ZUPT）成为核心机制。因此应建立独立任务，不与手机神经惯性里程计共享排行榜。

建议首批接入：

1. [UTIAS Foot-Mounted Inertial Navigation Dataset](https://starslab.ca/foot-mounted-inertial-navigation-dataset/)；
2. [pyShoe Hallway Dataset](https://github.com/utiasSTARS/pyshoe)；
3. [IPIN 2019 Competition Track 4](https://zenodo.org/records/3937220)；
4. 早期高精度光学参考足绑数据集（需继续确认官方下载与许可）。

建议任务：

- 零速区间检测；
- 姿态/航向估计；
- 纯足绑 INS 轨迹估计；
- 不同步态（走、跑）与不同受试者泛化。

## 4. 特定载体与新型 IMU 配置

这些数据不应进入第一阶段的行人主榜，但适合后续建立独立 track。

| 数据集 | 平台 | 特点 | 建议用途 |
|---|---|---|---|
| [IO-VNBD](https://doi.org/10.1016/j.dib.2021.106885) | 汽车 | 文献报告超过 58 小时、约 4,400 km 的惯性与里程计数据 | 车辆惯性/里程计融合轨道；需核验原始下载入口 |
| [AI-IMU Dead-Reckoning](https://github.com/mbrossar/ai-imu-dr) | 轮式车辆 | IMU-only 车辆航位推算代码与配套数据流程 | 学习辅助滤波基线与车辆轨道 |
| [Multiple and Gyro-Free Inertial Datasets](https://pmc.ncbi.nlm.nih.gov/articles/PMC11450167/) | 移动机器人、乘用车、转台 | 54 个惯性传感器组成 9 个 IMU；支持 MIMU/GFINS；约 45 小时并带真值 | 多 IMU、无陀螺仪结构研究 |
| [Wheel-Mounted Inertial Datasets](https://www.nature.com/articles/s41597-025-06224-w) | 乘用车、移动机器人 | 14 个 IMU，其中包含轮载 IMU；约 73.75 分钟采集，按全部 IMU 计约 578 分钟数据 | 轮载 IMU 与车体 IMU 对比 |
| [i2Nav-Robot](https://github.com/i2Nav-WHU/i2Nav-Robot) | 轮式机器人 | 大尺度室内外、多传感器、提供 ROS bag 与原始文本格式 | 多传感器导航扩展轨道与格式适配参考 |
| [UrbanNav](https://www.polyu-ipn-lab.com/) | 城市车辆 | 面向城市峡谷的 GNSS/INS/多传感器定位 | GNSS 退化与融合导航，不宜放入纯 IMU 主榜 |
| [UrbanLoco](https://github.com/weisongwen/UrbanLoco) | 城市车辆 | 13 条轨迹、超过 40 km，含 LiDAR、相机、IMU、GNSS | 城市多模态定位扩展 |
| [KITTI Raw](https://www.cvlibs.net/datasets/kitti/raw_data.php) | 汽车 | 广泛使用的相机、激光、GPS/IMU 数据 | 生态兼容性测试；并非纯惯性专用数据集 |

## 5. 视觉—惯性与多模态辅助数据集

这些数据具有高质量同步 IMU 和位姿真值，可用于验证统一读取接口、六自由度输出和跨平台泛化，但其采集目标不是 IMU-only 定位。建议单独标记为 **auxiliary**。

| 数据集 | 平台与规模 | 可用于惯性研究的价值 | 限制 |
|---|---|---|---|
| [ADVIO](https://github.com/AaltoVision/ADVIO) | 手持智能手机；室内外、楼梯、扶梯、电梯、商场、地铁站等 | 自然人类移动、多场景、手机级 IMU | 原始任务是视觉—惯性里程计；真值和纯 IMU可观测性需分开讨论 |
| [EuRoC MAV](https://projects.asl.ethz.ch/datasets/euroc-mav/) | 微型飞行器；双目、同步 IMU、精确真值 | 高频动态、标准化格式、6-DoF 轨迹 | 仅 11 条主要序列，平台与行人差异大 |
| [TUM-VI](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset) | 手持相机—IMU 装置；室内外多类序列 | 时间戳、标定、IMU 坐标系与真值组织规范，适合作为格式设计参考 | 许多长序列只在起止段提供动捕真值 |
| [UZH-FPV](https://fpv.ifi.uzh.ch/) | 激进飞行无人机；27+ 序列、10+ km | 极端角速度、加速度和高速 6-DoF 运动 | 非行人域；许可为非商业使用 |
| [Aria Everyday Activities](https://www.projectaria.com/datasets/aea) | Aria 眼镜；143 条日常活动序列、5 个地点 | 头戴式 IMU、全局对齐高频轨迹、真实日常活动 | 下载需接受专用协议；许可限制非商业研究及特定研究领域 |
| [InCrowd-VI](https://incrowd-vi.cloudlab.zhaw.ch/) | Aria 眼镜；58 条序列、约 5 km、1.5 小时 | 室内人群遮挡条件下的人体导航与高质量轨迹 | 主要为 VIO/SLAM 评测，真值来自机器感知服务而非独立动捕 |

## 6. 明确不纳入定位主榜的数据

以下数据即使包含 IMU，也不应仅凭“有加速度和陀螺仪”而加入定位 benchmark：

- 只有活动类别、没有连续轨迹或位姿真值的 HAR 数据集；
- 只有步数、步态、关节角或医学标签的数据集；
- 只有原始 IMU、没有可靠定位参考的数据；
- 仅提供论文统计、没有可访问数据文件的数据；
- 真值与 IMU 时间不同步且无法修复的数据；
- 许可禁止目标用途、结果公开或必要格式转换的数据。

这类数据未来可以用于 IMU 表征预训练，但必须与定位评测数据分离。

## 7. 初步结论

### 7.1 第一阶段不宜覆盖所有载体

如果一开始同时支持手机、足绑、汽车、无人机、多 IMU 和视觉—惯性，统一格式会迅速变成一个过度抽象的容器，评测协议也无法保持公平。

建议第一版聚焦：

> **人体携带式、IMU-only、学习式惯性里程计**

首批 P0 数据集：

1. RIDI；
2. OxIOD；
3. RoNIN。

随后新增一个独立的足绑轨道：

4. UTIAS Foot-Mounted；
5. pyShoe；
6. IPIN 2019 Track 4。

### 7.2 “统一”不等于强行同质化

统一格式应该保留三层信息：

- **raw**：尽可能忠实保存原始字段；
- **canonical**：统一时间戳、单位、坐标约定和真值接口；
- **derived**：重采样、重力对齐、滑窗、速度/位移标签等可再生成特征。

不能只发布处理后的网络输入，否则会丢失复现实验和改变预处理策略的能力。

### 7.3 统一评测至少需要两类协议

- **in-domain**：遵循数据集官方或文献通用划分；
- **cross-domain**：在一个或多个数据集训练，在完全不同数据集上测试。

其中 train/val/test 必须按受试者、场景或完整序列切分，不能先切滑窗再随机划分。

## 8. 下一轮核验任务

- [ ] 下载并检查 RIDI 的真实目录结构、字段、单位、坐标系与许可；
- [ ] 下载并检查 OxIOD，不同子集分别记录真值来源；
- [ ] 获取 RoNIN 官方数据与 split 文件，核对 seen/unseen 定义；
- [ ] 查找 SIMD 的官方长期下载入口、许可和完整元数据；
- [ ] 对 UTIAS、pyShoe、IPIN 2019 做足绑数据字段对照；
- [ ] 建立数据集许可证与再分发矩阵；
- [ ] 统计每个数据集的实际序列数、时长、距离、采样率、缺失率和文件体积；
- [ ] 根据前三个 P0 数据集反推 unified schema，而不是预先设计过度通用的格式。

## 参考入口

- [RoNIN paper](https://arxiv.org/abs/1905.12853)
- [RIDI paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Hang_Yan_RIDI_Robust_IMU_ECCV_2018_paper.pdf)
- [OxIOD paper](https://arxiv.org/abs/1809.07491)
- [TLIO paper](https://arxiv.org/abs/2007.01867)
- [EqNIO / ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6554c4151c2ccbd36da2d55c015b2039-Abstract-Conference.html)
- [ADVIO repository](https://github.com/AaltoVision/ADVIO)
- [EuRoC official page](https://projects.asl.ethz.ch/datasets/euroc-mav/)
- [TUM-VI official page](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset)
- [Aria Everyday Activities](https://www.projectaria.com/datasets/aea)
- [Multiple and Gyro-Free Inertial Datasets](https://pmc.ncbi.nlm.nih.gov/articles/PMC11450167/)
- [Wheel-Mounted Inertial Datasets](https://www.nature.com/articles/s41597-025-06224-w)
