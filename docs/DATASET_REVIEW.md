# 惯性定位 Dataset Review

> 最近更新：2026-07-19  
> 完整调研与证据链见 [DATASET_SURVEY.md](DATASET_SURVEY.md)。

## 核心清单

| 数据集 | 首次发布 | 平台/任务 | 规模摘要 | 真值 | 开放状态 | 接入状态 | 最后核验 |
|---|---:|---|---|---|---|---|---|
| RIDI | 2018 | 手机、2D 行人 IO | 约 150 分钟，200 Hz | Google Tango VIO | 公开 | P0 待下载 | 2026-07-19 |
| OxIOD | 2018/2019 | 手机、2D/3D IO | 158 序列，42+ km | 多数 Vicon，部分其他参考 | 公开 | P0 待下载 | 2026-07-19 |
| RoNIN | 2019/2020 | 人体携带、2D IO | 40+ 小时，100 人 | 3D 轨迹参考 | 约 50% 公开 | P0 待下载 | 2026-07-19 |
| TLIO | 2020 | 头戴式、3D IO | 400 序列，约 60 小时 | MSCKF/VIO 状态 | EqNIO 提供 golden v1.5 | P0 待下载与许可核验 | 2026-07-19 |
| IDOL | 2021 | 手机方向与位置 | 20+ 小时，15 人，3 建筑 | 设备 pose | Zenodo 公开 | P0 待下载 | 2026-07-19 |
| SenseINS / RNIN-VIO | 2021 | 多手机、3D 人体运动 | 自采约 7 小时，5 人 | BVIO；部分 Vicon | 公开 | P0/P1 待下载 | 2026-07-19 |
| IMUNet_dataset | 2024 | Android 手机 IO | 规模待实物统计 | ARCore/Tango 参考 | Google Drive 公开 | P1 待核验 | 2026-07-19 |
| Aria Everyday Activities | 2024 | 头戴式、跨设备 3D | 143 序列，5 地点 | 全局对齐轨迹 | 协议约束公开 | P1 | 2026-07-19 |
| Blackbird | 2018 | UAV、高动态 3D | 168 飞行，10+ 小时 | 360 Hz 动捕 | 公开 | UAV P1 | 2026-07-19 |
| EuRoC MAV | 2016 | UAV、VIO/IO | 11 条主序列 | 激光/动捕真值 | 公开 | UAV P1 | 2026-07-19 |
| GrandTour | 2026 | 四足机器人 | 49+ 环境/任务 | RTK-GNSS/全站仪 | 公开 | Legged P1 | 2026-07-19 |
| Go2 | 2026 | 四足机器人 IO | 待核验 | 待核验 | 未确认 | P2 | 2026-07-19 |

## Review 模板

- Identity：名称、版本、论文、作者和官方入口；
- Scope：平台、任务、场景、人员/设备；
- Sensors：型号、频率、单位、坐标系、同步；
- Ground truth：来源、频率、精度、覆盖范围；
- Files：目录、格式、大小、校验和；
- Splits：官方 train/val/test 与泄漏风险；
- License：下载、使用、转换、再分发、商业限制；
- Known issues：缺帧、漂移、时间偏移、坏序列；
- Adoption：使用该数据集的论文与常用协议；
- Integration：下载器、转换器、测试和状态；
- Verification：最后核验日期与证据链接。

## 状态流

待发现 → 已找到官方来源 → 已确认可下载 → 已下载 → 已解析 → 已转换 → 已验证 → 已接入

任何数据集只有在“已验证”后才能进入正式 benchmark。
