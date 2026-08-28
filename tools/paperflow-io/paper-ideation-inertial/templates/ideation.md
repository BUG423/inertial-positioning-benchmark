> stage: ideation | updated: YYYY-MM-DD

# 研究方向讨论 — 深度学习惯性定位 (IO)

## 用户处境
- 领域: 深度学习惯性定位 (IO) / 子方向: <direction>
- 资源: 算力=1×3090 (锁定) | 数据=<data_availability> | deadline=<deadline>
- 入场状态: <bucket A 无想法 / B 模糊方向 / C 一篇 paper 触发 / D 多候选>

## 选定方向 (主推 + Backup)

### 🎯 主推 I<n>: <一句话标题>
- 核心 claim (C1): <在什么设定下，相比什么，能做到什么>
- 核心 claim (C2，可选): ...
- 关键假设 H1: <如果不成立则整个方向失败，实验阶段优先验证>
- 综合评级: <S/A/B>
- 评分依据 (3 维):
  - 新颖性 ★×N (Lv?): ...
  - 可行性 ★×N: 1×3090 + 数据...
  - 影响力 ★×N: 谁会在乎...

### 🛡️ Backup I<m> (deadline < 6 个月时必给)
- 切换触发条件: <e.g. 第 8 周仍未拿到 X>
- 综合评级: <A/B>

## 完整 10 个候选清单

| # | 标题 | 贡献类型 | 综合评级 | 推荐排名 |
|---|------|---------|---------|---------|
| I1 | ... | 新架构 / 新训练范式 / 新泛化方法 / 新评测 / 新理论分析 / 新传感器融合 / 新效率方法 | S | #1 🎯 主推 |
| I2 | ... | ... | A | #2 |
| ... | ... | ... | ... | ... |
| I10 | ... | ... | C | 不推荐 |

## 种子文献 (来自 Step 4 检索或知识库，均已核实)
- [shortkey1] <标题>, <venue> <年> — <与本方向关系> — arXiv:<id>
- [shortkey2] ...
- [shortkey3] ...

## 目标 venue (初步)
- 主选: <ICRA / IROS / CVPR / ICRA + RA-L / TIM>
- 备选: <另一个>
- 截稿月: <如用户提供>

## 交接 (进入实验阶段)
- 本阶段产出: 本文件 + idea-spec-I<n>.md (每个选定 idea 一份) + references.md (均在 paperflow/)
- 下一步: 直接启动实验 skill `paper-experiment-inertial`，它会自动检索并加载 paperflow/ 下的文件。
- 实验做什么、怎么做，由实验 skill 负责，本阶段不规定。

## 检索元信息
- 检索 query: <query terms>
- 时间窗: 近 <years> 年
- 检索到论文: <N> 篇 (其中 <M> 篇用作 idea 依据)
- 检索时间: <YYYY-MM-DD HH:MM>
- 用过的源: arXiv + OpenAlex + S2 (S2 是否限流: yes/no)
- 知识库 fallback: yes/no
