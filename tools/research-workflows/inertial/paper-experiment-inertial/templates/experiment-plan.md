> stage: experiment | updated: <YYYY-MM-DD>

# 实验记录 — <选定 idea 标题>

## 二次校验结果
- 校验日期: <YYYY-MM-DD>
- 文献再检索: <新增相关 paper 数> / 抢跑风险: <low/medium/high>
- 代码可行性: <score>/5 — <一句话>
- 缺失 baseline: <列表, 如无则写"无">

## 实验环境
- 代码: Begin (路径: <path>)
- 算力: 1×3090
- 环境: uv + PyTorch CUDA 12.1
- 数据集: RoNIN / TLIO / IMUNet / RIDI / OxIOD / RNINv2 (默认全部)

## 选定执行路径
- 路径: A (用户自主) / B (Claude 协助执行)
- 代码路径: <path> (路径 B 必填)
- 数据路径: <path> (路径 B 必填)
- 服务器信息: <如有> (路径 B 选填)

## 新增算法骨架
- 目录: `network/<算法名>/`
- model_factory arch 名: <主模型>, <deploy>, <No_XX>, ...
- 是否需要自定义损失: <是/否>
- 冒烟测试: <通过/未通过>

## 对比实验结果 (必做: ≥3 数据集 × ≥6 算法 × {ATE,RTE} = ≥36 格, 目标 ≥90% SOTA)
- 数据集 (≥3): <列出>  |  对比算法 (≥6): <列出>
- 战略性排除: <无 / 排除了哪些数据集·算法 + 理由>
- SOTA 命中: <x> / <总格数 ≥36>  ( <百分比>% )  →  ≥90% 即成功，不追求 100%

| 数据集 | 方法 | ATE (seen) | ATE (unseen) | RTE | 参数量 | SOTA? |
|--------|------|-----------|-------------|-----|--------|:--:|
| RoNIN | 你的方法 | | | | | |
| RoNIN | <baseline1> | | | | | |
| RoNIN | <baseline2> | | | | | |
| TLIO | ... | | | | | |
| RIDI | ... | | | | | |

## 消融结果 (模型能拆模块才做)
| 变体 (arch 名) | ATE | Δ vs full model |
|------|-----|----------|
| Full (<算法名>) | | — |
| No XX | | |
| No YY | | |

## 可视化产物 (必做)
- CDF + PDE 图: 数据集 <X>, 对比算法 <列出>, 路径 `plot/cdf_pde/...` | 我的 CDF 是否更靠左上: <是/否>
- 轨迹对比图: 精选误差低的 <N> 条轨迹, 路径 `plot/traj/...` | 是否更贴合 GT: <是/否>
- ATE vs 参数量图 (轻量方向, 可选): `plot/perf/...`

## H1 验证结果
- 假设: <H1 原文>
- 验证方法: <怎么验证的>
- 结果: <成立/不成立>
- 如果不成立: <pivot 到了哪里, 日期>

## 关键发现
- <写 paper 时最想强调的 1-3 个发现>
- <审稿人可能挑的 1-2 个弱点 + rebuttal 准备>

## 下一步
- paper-writing: 故事线方向
- 补充实验 (如需要): 待跑清单

## 检索元信息
- 二次检索 query: <terms>
- 二次检索日期: <YYYY-MM-DD>
- 新增发现: <N> 篇 | 威胁等级: <none/low/medium/high>
