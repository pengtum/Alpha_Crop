# 论文计划书：After transfer（新主线）

2026-09-27 写成。供作者定稿和后续写作使用。

**依据**
- 定位简报：`REPOSITION_AFTER_CHENG2026_2026-09-27.md`，含重合审计、审稿人视角、三条主线和三份评审
- 方案：冻结方案 `ANALYSIS_PLAN_guidance.md` 及 rev1–rev4
- 结果文档：`G_RESULTS`、`CONCLUSIONS_2026-09-26`、`STRATEGY_FINDINGS_2026-09-26`、`CHECK1_RESULTS`、`CHECK3_RESULTS`、`PIVOT_EVALUATION_2026-09-25`、`MODEL_DIAGNOSIS_2026-09-25`、`TESTSET_AUDIT_2026-09-26`
- 各视角的只读输出：`verify/reposition/*`、`verify/strategy/*`、`verify/pivot_eval/*`、`verify/model_diag/*`
- Cheng et al. 2026 原文 PDF：已核对的引文和数字见附录 B

**写作边界**
- 没有读取 `results/guidance_G_ml/` 下的任何文件。写作时 rev2 进程 `multilearner_G.py --all` 仍在运行。
- **2026-09-27 晚补记（rev2 解盲后）**：rev2 完成（每个学习器 1,848 行，0 失败），完整性检查通过；`analyze_ML.py`、`analyze_ML_rev3.py` 已正式运行；D2 已对四个学习器运行（`cheng_comparable/unblinded_4learners/`）。独立重算（不导入 analyze_*）逐项复现了全部 headline 数字，并用概率文件重算了全部 5,544 行指标（最大误差 1.4e−14），未发现会改变结论的问题。本计划中 rev2、rev3-B1/B2/B3、D2 的占位已据此填入，填入处标〔rev2 已填〕。图：`L/figures/fig_multilearner.png`（`plot_ML.py` 重画版式）、`L/figures/fig_ML_rev3.png`（`plot_ML_rev3.py`）。
- 没有训练模型，也没有修改项目文件。
- 新做的只有两项：一是只读的类别计数（`class_counts.py`，见 §4.1 与附录 B）；二是核对 Cheng PDF 的原文。

**路径缩写**
- `L/` = `/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion/`
- `V/` = `/Users/perng_mini/AlphaCrops_local/verify/`
- `R/` = `/Users/perng_mini/AlphaCrops_local/results/`

---

## 0. 约定

**证据等级**（沿用 `CONCLUSIONS_2026-09-26` 的定义）

| 等级 | 含义 |
|---|---|
| A | 预注册的主判定，核查通过 |
| B | 方向稳健，但有重要限定，例如只用了一个学习器、划分没有缓冲、属于次要指标 |
| B− | 否定性结论，区间宽，或只在部分设置下成立 |
| C | 事后分析，或用了已复用的测试集，没有确认 |
| D | 已被推翻，或本身有缺陷，不能写 |

**标注**
- 【预注册】【事后】
- 【复用】：G 的 s0/s1 测试集在主分析之后已多次使用
- 【非盲】：rev3 §A 列明的 LR 行
- 【LGB】：只有 LightGBM 的结果

**占位符**
- 格式为 `⟦待定·来源：内容⟧`，来源取 rev2、rev3-B1/B2/B3、D1、D2、D3、确认运行、作者决定之一。
- 摘要和正文中的待定数字一律用占位符，不预估。

**数字的最终来源**
- 论文中的数字最终都从 D3 的 `NUMBERS.md` / `NUMBERS.csv` 引用。
- 本计划的数字都摘自上列文档，或由本计划只读计数得到，出处都已标注。D3 生成后要逐项对表。

---

## 1. 标题、核心问题、研究问题

### 1.1 标题候选

| # | 标题 | 使用条件 |
|---|---|---|
| 1（推荐） | *After transfer: what a few hundred target-country labels can and cannot fix in multi-source cross-country crop-type mapping*。副题可选：*Evidence from 11 European countries with AlphaEarth embeddings and Sentinel-2 phenological statistics* | 现在就能用 |
| 2 | *Class composition is cheap, domain residual is not: collecting and using target-country labels after cross-country crop-type transfer* | 两个条件都满足才用：① D1 完成、定义表统一了"域残差 / 决策层可挽回部分 / 表征层剩余"；② rev2/rev3 显示后半句不是 LightGBM 特有的现象。否则不用。〔rev2 已填：条件 ② 不满足——B2 判据 2/4，LR 下 200 个标签去 RO、PT 仍带来 +4.4 pp BA，"域残差不便宜"只对树模型成立 → 不用标题 2，用标题 1〕 |
| 3（备选） | *Spending a few hundred labels after cross-country crop-type transfer: gap anatomy, sample selection and label use in 11 European countries* | 候选 1 太长时使用 |

**标题规则**（REPOSITION §2.2）
- 前半句不出现 AlphaEarth/AE。
- 不写 "pre-registered strategy benchmark"。
- 不用"决策层"作标题框架。
- 基线写 "Sentinel-2 phenological statistics"，不写 SITS。

### 1.2 一句话核心问题

> 用 10 国约 5 万个外国地块训练的作物类型模型迁到新国家，目标国只能标 50–400 个地块时：这些标签能补上迁移缺口的哪一部分？该标哪些地块？标签该怎么用？AlphaEarth 在其中任何一步是否比 Sentinel-2 手工物候统计更有帮助？

英文草稿：
> When a crop-type model trained on ~50,000 parcels pooled from ten foreign countries is transferred to a new country that can afford only 50–400 labels, which part of the transfer gap can those labels close, which parcels should be labelled, how should the labels be used, and does an Earth-observation foundation-model embedding (AlphaEarth) help at any of these steps relative to Sentinel-2 phenological statistics?

### 1.3 研究问题

| RQ | 问题 | 对应结论 | 结果节 | 主要证据 | 待定 |
|---|---|---|---|---|---|
| RQ1 | 迁移缺口由哪些部分组成？少量目标标签能补上哪一部分，要多少？ | H1、H3（附 D1） | 3.1、3.2 | B（检验 3 分解、模型诊断 M3、G 的 SRS 曲线）；上界与 D1 为 C | D1；rev3-B2 |
| RQ2 | 该标哪些地块？ | H4，以及 H2 中"指导"的部分 | 3.3 | A/B（θ_E1）；B−（几何法≈SRS）；C（predstrat） | rev2 θ_E1(L) |
| RQ3 | 标签该怎么用？ | H5（附 D1） | 3.4 | B（先验估计）；C（决策头、D1） | D1 |
| RQ4 | AE 在任何一步是否比 S2 更有帮助？分三处看：作选样指导、作选源距离、作分类特征 | H2、H6 | 3.3、3.5、3.6 | A/B（指导）；B（选源）；C（特征） | rev2；rev3-B1/B3；D2 |

### 1.4 贡献：新颖性只写四点

依据 REPOSITION §6 B11 与评审 1 第四条。
1. 用少量目标标签估计类别先验，不依赖普查或作物统计。先验偏移分解本身不新：Saerens 2002、Kluger et al. 2021 已有，Cheng 引言也列为已有方法。
2. OA 与 BA 的迁移缺口来源不同，所以"标签怎么用"取决于交付什么产品。
3. 设定：11 国多源外国池，加 50–400 个目标标签，自然类别比例，空间分块并加缓冲的测试。
4. 把缺口解剖、选样、标签用法放在同一协议下联动评估。AE 在其中只作为被检验的因子。

**不主张**
- "首个"作物主动学习；"首个"用嵌入选样。
- 分解概念的新颖性。
- "随机难打败"是新发现。

---

## 2. 摘要骨架

### 2.1 准入规则

- 摘要只放 A/B 级结论：H1、H2、H3 中的 B 级部分，以及 H5 中的先验估计部分（REPOSITION §4）。
- 下面这些一律不进摘要：
  - H4：几何法≈SRS 为 B−，predstrat 为 C；
  - 决策头；
  - D1；
  - 缓冲格网上的决策层上界；
  - 标签当量与"汇率"；
  - AE 作为特征的优势（H6，C）；
  - 单源 |Δlat| 反向结果。
- 如果做了确认运行，并且结果通过预先登记的判据，再按结果改写这一条。

### 2.2 英文骨架

逐句列出，句后注等级和出处。字数上限和 Highlights 规则按 RSE Guide for Authors 核对（本计划未核实）。

1. **背景**：Cross-country transfer is a common route to crop-type maps in label-scarce countries. A recent comparison showed that embeddings with more stable crop-type separability still require target-region validation and local labels over long transfer distances (Cheng et al., 2026).
   - 引文只作动机，不附数字。
2. **问题**：We ask what a few hundred target-country labels can and cannot fix after multi-source transfer, and whether AlphaEarth embeddings change the answer relative to Sentinel-2 phenological statistics.
3. **设计**（事实）：We use 611,519 parcels from 11 European countries (EuroCrops, 2021, 10 classes), a pooled foreign training set of 50,000 parcels, 0–400 target labels, natural class proportions, and spatially blocked test sets (10-km grid, 1-km buffer). Both overall accuracy (OA) and balanced accuracy (BA) are reported.
   - 611,519 来自本计划的只读计数（附录 B）。
4. **H1**〔B；检验 3 预注册，随机无缓冲划分〕：The transfer gap has different origins for the two metrics. Class composition accounts for 44–67% of the OA loss, depending on the training setting, whereas class-set differences explain only 11–15% of the BA loss. The remaining domain residual was not detectably smaller with AlphaEarth (OA +0.00 pp; BA +0.73 pp).
5. **H1 续**〔B；事后，缓冲格网，LGB/MLP/LR〕：Training on ~30,000 target labels alone exceeded foreign-only training by 13–14 pp BA in all 22 country × feature combinations, and a gap of similar or larger size appeared with MLP and logistic-regression learners in a five-country pilot.
6. **H5 先验估计**〔B；检验 3 预注册次要〕：About 50 random target labels recovered 85–97% of the OA gain attainable with true class priors (label-free EM: 31–58%). The same correction lowers BA, so the intended product (area/OA vs per-class maps) determines how labels should be used.
7. **H3**〔B；LGB〕：When labels were merged into foreign training, 200 random labels raised BA by 2.4–3.1 pp on average, but only by ~0.7 pp outside the two countries with extreme composition mismatch or domain residual. 〔rev2 已填：B2 判据 2/4 满足（LightGBM、RF），LR、MLP 不满足 → 按预设措辞改为〕 "With LightGBM, …; the gain depended on the learner: outside those two countries it stayed below ~0.7 pp with the two tree ensembles but reached 4.4 pp (95% CI 2.0–6.7) with logistic regression."
   - 句中 LightGBM 的 2.4–3.1 不变；LR 的 +4.4 为【非盲】【复用】，C 级描述，不作推断。MLP 为 +1.1 [−0.4, +2.7]，不必进摘要。
8. **H2**〔方向 A，等价 B；G 预注册〕：Replacing Sentinel-2 by AlphaEarth as the geometry that guides which parcels to label changed BA by −0.05 pp (90% CI −0.21 to +0.10), within a pre-registered ±0.3 pp equivalence margin. 〔rev2 已填：情形 3，见 2.4〕 "The same equivalence held with an MLP ensemble and a random forest, whereas with logistic regression the result was inconclusive (+0.17 pp, 95% CI −0.12 to +0.45)."
   - 必须同时交代：只对 BA（ba_ge30）成立；OA 版 θ_E1 在四个学习器下都"不确定"（正文，不进摘要）。
9. **H2 续**〔B；检验 1 预注册〕：AlphaEarth distances did not rank source countries better than Sentinel-2 distances (Δρ = +0.02, 95% CI −0.02 to +0.06) or geographic distance. Better separability of the embedding (10-NN BA 46.6% vs 38.8%) thus did not translate into better labelling decisions.
10. **结论**：A first batch of ~50 spatially dispersed random labels serves validation and prior estimation. Beyond that, the choice of representation did not change where labels should be spent, and the BA gap is dominated by a domain residual that a few hundred labels merged into foreign training do not close.
    - 〔rev2 已填〕后半句只对树模型成立（B2 判据 2/4）。改为："…that a few hundred labels merged into foreign training did not close with tree ensembles; how much they helped depended on the learner." 或删去后半句。
    - 这里不写"决策层可廉价修正"。D1 完成前也不写"少量标签即可修正决策层"。

### 2.3 Highlights 草稿（字符数待核）

- OA and BA transfer gaps across 11 countries have different origins
- ~50 random target labels recover most of the prior-correction OA gain
- Merged into foreign training, a few hundred random labels add little BA〔rev2 已填：只对树模型成立；改为 "…add little BA with tree ensembles" 或换一条〕
- AlphaEarth and Sentinel-2 guidance for label selection are equivalent〔rev2 已填：改为 "…are equivalent for balanced accuracy with three of four learners"〕
- Embedding separability did not translate into better labelling decisions

### 2.4 H2 的学习器稳健性：三种情形的备用句

rev2 §4 冻结的先后顺序：先判情形 1；不成立再判情形 2；其余归情形 3。

| 情形 | 判据（rev2 §4） | 摘要用句 |
|---|---|---|
| 1 等价不依赖学习器 | 四个学习器都是：90% CI 落在 ±0.3 内，且 95% CI 含 0 | "…and this equivalence held for all four learners tested (LightGBM, logistic regression, an MLP ensemble and random forest)." |
| 2 至少一个学习器有方向性差异 | 任一学习器的 95% CI 不含 0 | "…; with ⟦L⟧, AlphaEarth guidance was ⟦better/worse⟧ by ⟦x⟧ pp (⟦CI⟧), ⟦该学习器自己的判定，如 'statistically better but by less than 0.3 pp'⟧." 不挑最有利的学习器，按实际方向写 |
| 3 部分等价 | 其余情况 | "…equivalent with ⟦L1, L2⟧ and inconclusive with ⟦L3⟧." 逐个列出 |

rev2 出结果之前，全文不出现"跨学习器"字样。

**〔rev2 已填〕结果：情形 3（部分等价）。**
- 情形 1 不成立：LR 的 90% CI [−0.06, +0.40] 越出 +0.3。
- 情形 2 不成立：四个学习器的 95% CI 都含 0。
- 各学习器（ba_ge30，11 国，df=10）：

| 学习器 | θ_E1 [95% CI] | 90% CI | TOST p | 判定 | >0 国家 |
|---|---|---|---|---|---|
| LightGBM（预注册主分析） | −0.054 [−0.244, +0.136] | [−0.208, +0.101] | 0.008 | 等价 | 4/11 |
| Logistic 回归【非盲】 | +0.168 [−0.115, +0.451] | [−0.062, +0.398] | 0.162 | 不确定 | 8/11 |
| MLP 集成 | +0.018 [−0.251, +0.288] | [−0.201, +0.238] | 0.021 | 等价 | 5/11 |
| 随机森林 | −0.072 [−0.237, +0.092] | [−0.206, +0.061] | 0.006 | 等价 | 5/11 |

- OA 版 θ_E1：四个学习器都"不确定"（LGB −0.15，LR +0.18，MLP −0.01，RF −0.16；90% CI 都越出 ±0.3）。
- 配对差（描述）：θ(LR) − θ(LGB) 为 BA +0.22 [−0.03, +0.47]、OA +0.33 [+0.03, +0.63]；MLP、RF 与 LGB 的差都在 ±0.14 以内。
- 摘要用句取情形 3 那一行："…equivalent with LightGBM, an MLP ensemble and a random forest, and inconclusive with logistic regression."
- **解读上必须写的限定（独立核验提出）**：两个树模型对目标标签本身就不敏感（去 RO、PT 后 G(200) ≈ +0.6 pp），因此在它们下面判"等价"，部分反映的是标签整体作用小；标签作用最大的 LR 恰好是"不确定"的那个，点估计偏向 AE。这一点写进 3.3 与局限（测定敏感性），不得省略。

---

## 3. 引言逐段要点

**P1 背景与问题**
- 作物类型图需要目标区标签，而很多国家标签稀缺，所以常用"从有标签国家迁移"。
- 地球观测基础模型的嵌入（AlphaEarth Foundations；TESSERA）被期待有更好的可迁移性。EuroCrops/HCAT 使欧洲多国的协调标签可用。
- 候选引文（需核实书目）：Brown et al. 2025（AEF）；Feng et al. 2025（TESSERA）；EuroCrops；Kluger et al. 2021。

**P2 Cheng 2026 的结论**（已核对原文，附录 B）
- 2576 个训练–测试对，欧洲主实验为 567 个跨区迁移，RF 为主分类器。
- 本地测试：AEF 平均 OA 比 SITS 低 0.02（0.94 对 0.96，表 3）。
- 跨区迁移：AEF 平均 OA 0.71，SITS 0.75（表 3）。
- 每增加 1000 °C·d 的 |ΔGDD|，AEF 的 OA 降 0.14，SITS 降 0.23。|ΔGDD| 超过约 1400–1500 °C·d（欧洲约 9–10° 纬度）后，AEF 更有竞争力。
- 机制：AEF 更好地保持了作物类型在特征空间中的相对位置和可分性（JMmin 下降更慢）。
- 写法：Cheng 是动机前提，不是对手。AE 在本文中只是被检验的因子。

**P3 Cheng 在 §4.3 Practical implications 的建议**（第 15–16 页；草稿中的"§4.1"是错的，统一改为 §4.3）
- 转述：远距离迁移应先用目标区地面真值验证，然后 "followed by local adaptation or the inclusion of additional local labels when necessary"。
  - 全文只直引这一段，不超过 15 词，其余一律转述。
- 衔接句：本文从这一步开始，回答"这些本地标签补什么、标哪里、怎么用"。

**P4 钩子：SITS-LDA5D**（Cheng §3.4.3、Fig. 12b）
- 转述：Cheng 用所有非测试欧洲区域六类作物的标签拟合 LDA，把 SITS 投影到 5 维。它在欧洲跨区迁移中取得最高平均 OA（0.83），ALR 与 AEF 相当。作者据此认为，作物标签监督的降维能改善 SITS 的可迁移性，尤其是 LDA 用多个非测试区域的标签拟合时。
- 我们的引申：多区域标签本身就能提供被归功于 AEF 的迁移稳健性。这正好为"多源外国池 + 少量目标标签"的设定提供动机。
- 注意：
  - Cheng 正文对 LDA5D 的 ALR 只写"与 AEF 相当"。已读 Fig. 12b 图例核实：SITS-LDA5D slope −0.14（R²=0.37），即 ALR 0.14，与同图 AEF 的 0.14 相同，SITS 为 0.23。REPOSITION 写的"ALR 0.14"是对的，引用时注明"Fig. 12b 图中读数"。
  - §2.6.3 写 LDA 用"all non-test European regions"的标签拟合，是否对每个迁移对排除其测试区域，正文没有说清，引用前读其补充材料。
  - 另外，Cheng 用单源分类器，LDA5D 只是降维；我们的设定是多源合并训练。两者不等同，要写成"motivates"，不写成"shows"。

**P5 Cheng 设计留下的空白与四维对照表**（放引言末尾）

| 维度 | Cheng et al. 2026 | 本研究 |
|---|---|---|
| 目标区标签 | 不用 | 50–400 个，研究补什么、标哪里、怎么用 |
| 训练源 | 单源到单目标 | 10 国合并的外国池（约 5 万个地块） |
| 类别先验 | 每区-年-类最多 5000 点，近均衡；表 3 中 OA≈BA | 自然先验，OA 与 BA 双报，并量化类别构成通道 |
| 测试划分 | 区内按类分层 70/10/20 随机划分 | 10 km 格网加 1 km 缓冲，主分析预注册 |

**P6 相关工作定位**
- 先验/标签偏移：Saerens 2002；Kluger et al. 2021（RSE 262:112488，两种偏移）；Reuss 2026 DirPA。Cheng 引言已把先验偏移调整列为需要作物比例数据的方法（Cheng 第 3 页）。
- 主动学习与主动域适应，包括作物主动学习：Kaijage 2024 等。只写"本文不宣称方法新"。
- 预算采样与嵌入选样：Betti 2026 "Mapping on a budget"；GeoSpOT。
- 少样本作物基准：EuroCropsML。
- 这些书目的年份与卷期，在 `L/novelty_refs_2026-09-24.bib`、`L/sampling_survey_refs.bib` 里逐条核实。投稿前 4–6 周做增量检索（§9）。

**P7 本文做什么**
- 列出 RQ1–RQ4。
- 设计要点：11 国；外国池 5 万；k ∈ {0, 50, 100, 200, 400}；AE 与 S2 各自既作选样指导，也作分类特征；预注册与测试集审计。
- 定位句（英文，沿用 REPOSITION §3 的中性版）：
  > Cheng et al. (2026) showed, on class-balanced single-source transfers without target labels, that AEF better preserves crop-type separability across thermal gradients. We ask the question their design leaves open: under natural class proportions and multi-source training, how much of the cross-country loss can a few target labels recover, and does a more stable representation change where and how those labels should be spent?
- D1 完成之前，不改回"sits in a decision layer that a few target labels can recalibrate"。

---

## 4. 数据与方法要点

旧稿 `Manuscript/Paper_Alpha_Earth/0001_draft_v1/01_main.tex` 第 120–215 行（研究区、EuroCrops、特征提取）可以改写复用，但全部改成 dataset_v2 口径。旧稿写的"约 640k"是修复前的数字。

### 4.1 数据（dataset_v2）

- 11 国：AT、BE、DE、EE、LT、LV、PT、RO、SE、SI、SK。数据为 2021 年 LPIS（EuroCrops HCAT），归并为 10 个 Tier-B 类：
  - grassland、maize、orchards、other_crops、potatoes、root_crops、strawberries、tree_wood_forest、vegetables、winter_cereals。
- 总地块 **611,519**，每国 41,791–96,727（本计划只读计数，`dataset_v2_class_counts.csv`）。
- 缺类（必须在数据节写明）：
  - **RO 只有 3 类**：grassland、other_crops、tree_wood_forest。原因是泛称类 arable_crops 被归入 other_crops，属标签映射伪影（CHECK1）。
  - **LV 无 potatoes、strawberries**：S2 第 0008 批按类别选择性缺失，已排除（CHECK1）。
  - **LT 也无 potatoes、strawberries**（本计划计数发现）。已读文件中没有说明原因，需要作者核实后披露（§8 D6）。
- 每国测试集中 ≥30 个地块的类数（`ba_ge30` 实际平均的类数，两个划分相同）：

| 国家 | AT | BE | DE | EE | LT | LV | PT | RO | SE | SI | SK |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 测试集 ≥30 地块的类数 | 9 | 10 | 10 | 8 | 7 | 5 | 8 | 3 | 8 | 7 | 8 |

- 主文表 2（新表）：每国地块数、存在的类、测试集大小、≥30 地块的类数。数据来自 `V/storyline/paper_plan/dataset_v2_class_counts.csv` 与 `G_test_class_counts.csv`。

### 4.2 特征

| 特征 | 内容 |
|---|---|
| AE | AlphaEarth V1 年度嵌入，64 维，取 30 m 圆形邻域的 mean 与 stdDev，共 128 列 |
| S2 | Sentinel-2 手工物候统计：年统计加季度统计，168 列。其中只有 5 个变量有 4 个时间步（REPOSITION 附录 A §1c） |

- 统一称 "Sentinel-2 hand-crafted phenological statistics"，不称 SITS。
- 与 Cheng 的对照：Cheng 的 SITS 为 105 维（S2 7 个 40 天合成、Landsat 6 个 48 天合成、S1 14 个 20 天合成），AEF 取单点 10 m 像元的 64 维（附录 B）。

### 4.3 迁移协议（G 协议）

依据 `MODEL_DIAGNOSIS` §1、`TESTSET_AUDIT` §2 和本计划计数。
- **目标国划分**：
  - 按 (lon, lat, area) 去重后划 10 km 格网，随机取 30% 的格子作测试；
  - 非测试格中，距测试地块 >1 km 的地块作候选。
- **规模**：
  - 测试集每单元 11,890–30,251 个地块；
  - 缓冲候选集 27,673–65,357 个；
  - 从候选集中抽 3 万作候选池，EE、LV 不足 3 万，两个重复用的是同一个池。
- **外国池**：其余 10 国各 5,000 个，共 5 万。
- **单元**：11 国 × 划分 {0, 1} × 候选池重复 {0, 1} = 44 个。
  - k=0 的模型在两个重复间共用。
  - rep0 与 rep1 的候选池高度重合，推断单位是国家（df=10）。
- **预算** k ∈ {0, 50, 100, 200, 400}。
  - G 的主估计量只用 k ∈ {50, 200}；
  - SRS 学习曲线用全部五档。
- **E_far**：每个划分内，离缓冲候选集最远的三分之一测试地块。

### 4.4 分类器与训练

- LightGBM 多分类：固定 300 轮、不早停，63 叶，learning_rate 0.05，min_data_in_leaf 20，bagging 0.8/1，feature_fraction 0.8，max_bin 63。
- **权重**：
  - 类别权重 cw 只按外国池频率计算；
  - 外国样本权重为 cw[y]，目标样本为 50·cw[y]（r=50）。
  - r 的网格为 {1, 5, 20, 50}，r=50 落在网格上边界，而且是在 PT、BE、SI 的测试集上、用 S2 分类器与 SRS 选定的（必须披露）。
- **种子**：1000·split + rep。
- **阶段 0**：外国全量模型给出目标（候选 + 测试）的概率；按外国 5 组留国家组拟合标量温度。

### 4.5 学习器稳健性（rev2，预先指定，相对冻结方案是事后追加）

- **学习器**：多项 Logistic 回归（C 只用外国池、按留国家组规则选定）、MLP 集成（3 个种子）、RF（500 棵）。超参数见 rev2 §1 表。
- **范围**：
  - 训练集、权重、选样与 G 完全相同，只重训已保存的 880 个选样；
  - 每个学习器 1,804 次拟合；
  - 每个学习器应有 1,848 行（完整性条件）。
- **主估计量**：θ_E1(L)，定义与冻结方案相同。
- **判定**：三种情形按冻结的先后顺序判（§2.4）。
- **补充估计量**（rev3 B，预先指定、描述性）：
  - B1 Δ_feat(L,k)；
  - B2 G(L,k)；
  - B3 RF k=0 的 AE−S2。
- **库版本**：scikit-learn 1.9.1、numpy 2.5.3、scipy 1.18.1、lightgbm 4.7.0、Python 3.12.13。线程数不影响结果：代码审查实测 1 线程与 4 线程逐位相同。

### 4.6 各实验与测试集

主文方法节放一张表；证据上限由测试集性质决定。

| 实验 | 用途（本文节） | 测试集 | 预注册 | 证据上限 |
|---|---|---|---|---|
| 检验 1：单源两两迁移，无标签选源 | 3.5 | 国别留出，目标国全部地块 | 是（check1） | B（单种子、LGB） |
| 检验 3：跨境损失链式分解与先验估计 | 3.1、3.4 | 随机 70/30，无缓冲；另有 20 km 格网敏感性，同样无缓冲 | 是（08b3d271 + rev1 fc6aeffd）；容量检验为事后 | B |
| G：选样指导 | 3.3 | 10 km 格网 + 1 km 缓冲，s0/s1 | 是（ff55edf8）；rev1 为事后 | 方向 A，等价 B |
| G 的 SRS 学习曲线、方法 − SRS | 3.2、3.3 | 同 G | SRS 曲线是预注册的标签当量用途；方法 − SRS 是 R3 描述 | B/B− |
| 模型诊断（tgt300、容量） | 3.1 | 同 G【复用】 | 否 | B（方向）/ C |
| 转向评估（决策头、R2V） | 3.4 | 同 G【复用】；α、β、γ、λ 都在这里选定 | 否 | C |
| 重定位分析（缓冲上界、feat_by_k、标签价值对距离与构成） | 3.1、3.2、3.6 | 同 G【复用】 | 否 | C |
| rev2 多学习器 | 3.3、3.6 | 同 G【复用】 | 预先指定（2589217a） | 由判定决定；非确认性 |
| rev4 D1–D3 | 3.1、3.4、补充材料 | 同 G【复用】 | 运行前冻结（7e90ccd5） | C |
| 确认运行（可选） | 3.3、3.4 | 从未测试过的格子，F1/F2 | 需新登记 | 可到 A/B |

### 4.7 预注册与哈希

补充材料 S1 附完整时间线（`L/plan_hashes.txt`）。正文列这几项：

| 时间 | 文件 | sha256[:16] | 说明 |
|---|---|---|---|
| 09-24 09:47 / 10:10 | ANALYSIS_PLAN_check3 / rev1 | 08b3d2710f0aefa7 / fc6aeffd8bd0a81 | 检验 3 训练前；rev1 在看结果之前 |
| 09-24 18:47 | ANALYSIS_PLAN_guidance | ff55edf89528dd07 | G 冻结，冻结前没有在真实数据上训练 |
| 09-25 08:48 | rev1 | 3ee376b498ed1b2d | 看过层级 1+2 之后写的事后补充 |
| 09-27 11:05 | rev2、multilearner_G.py、analyze_ML.py、清单 | 2589217a7f2f2ae9 / f32183af651d337a / ee5fad2732057aed / 5d196f93d945db2d | 正式运行前 |
| 09-27 11:56 | rev3、analyze_ML_rev3.py | 6dac580ff2b54a9e / 776314e09f5ef257 | MLP、RF 未被查看；LR 的 SRS/k0 行从 11:35 起非盲 |
| 09-27 14:48 | rev4 | 7e90ccd5b1c3828f | D1–D3 运行前 |

检验 1 的方案 `ANALYSIS_PLAN_check1.md` 不在 plan_hashes 中。要么补登记时间，要么注明它的哈希没有登记。

### 4.8 指标与统计

- **主指标 ba_ge30**：测试集中 ≥30 个地块的类的平均召回，不做先验处理（P0）。
- **OA 同时报告**：OA 与 BA 是两个产品，不做事后的工作点调整。
- **P1/P2**（只用于机制拆分）：
  - P1 = 设计先验；
  - P2 = 候选池真实先验。
  - 术语按 rev1 B5：Δ(P0) − Δ(P2) 称"全局先验校正可去除的部分"，Δ(P2) 称"先验校正后的剩余部分"，不用"条件信息"。
- **推断**：
  - 以国家为单位的单样本 t 检验（df=10）。
  - 置信水平统一以 95% 为主；90% 只用于 TOST 与冻结方案第 8 节的判定。
- **等价**：SESOI ±0.3 pp，判定表见冻结方案第 8 节。
- **去掉 RO、PT**：每个均值都附这一版本（9 国，df=8）。
- **事后检验**：一律标为描述性，或按族做 Holm 校正。正文不引用朴素 p。
- **不混用测试集**：pairwise_v2（随机划分）的数字不与 G（缓冲格网）的数字放进同一个比较。

### 4.9 术语定义表（主文表 3）

必须在方法节给出，用来调和检验 3"BA 近乎全是域残差"与缓冲上界"BA 差距的 71–78% 可在决策层补回"这两种说法。

| 术语 | 定义 | 操作化与出处 | 注意 |
|---|---|---|---|
| 迁移缺口 | 同一目标测试集上，某参照模型与"只用外国"模型的精度差 | 参照有两种：检验 3 的本国模型 W0/W-nat（随机划分）；G 协议的 tgt300（只用目标池 3 万训练，300 轮，**只有划分 0**，`V/model_diag/fit/table_upper_bound.csv`） | 两种参照数值不同，不混用 |
| 类别构成 / 类别集合 / 样本量 / 域残差 | 检验 3 的训练时链式替换。OA 链：L0→PM→CMN→W-nat；BA 链：L0→LS→CMB→W0 | `CHECK3_RESULTS`；预注册 | "域残差"只剩"数据来自哪个国家"一项，其中**包含**可由温度与偏置吸收的部分。"≥100%"是链式分解的产物，不引用 |
| **决策层可挽回部分**（rev4 D1） | 在冻结的外国模型上，用目标标签只重新拟合温度与类别偏置所能得到的增益 | D1：`calib_diag.fit_bcts`，λ=0.1（预先固定），类均衡 NLL（每个出现的类总权重相同），从约 3 万候选池（EE、LV 为全部候选）中简单随机抽 k 个，k ∈ {50, 100, 200, 400, 1000, 全池}；报告 Δhead(k)。BA：k=50/200/400 为 +4.48/+6.17/+6.53，全池 +6.55 [+3.99, +9.12]（D1） | 类均衡权重面向 BA。这个头下的 OA 不是 OA 的决策层上界（全池 OA −0.79，去掉 RO、PT 为 −4.02）。λ 决定 BA–OA 工作点：λ=1e-4 时全池 BA +9.25、OA −11.92（D1 复核，事后，C 级，不用于选 λ）。与缓冲格网上界不是同一个量 |
| **表征层剩余**（rev4 D1） | 只用目标池全部标签训练的模型（tgt300），与决策层上界之间的差距 | tgt300 − 头(全池)。⟦待定·D1⟧ | tgt300 只有划分 0，所以表征层剩余只能在划分 0 上计算 |
| 域残差 ⊇ 决策层可挽回部分 + 表征层剩余（概念关系） | 链式分解中的"域残差"包含这两部分（rev4 D1）；检验 3（随机划分）与 D1、tgt300（G 缓冲格网，tgt300 只有划分 0）的数值不能直接相加 | — | 这一句写进方法节，并在 3.1 中引用 |
| 决策层上界（另一种操作化） | 缓冲格网上，用缓冲候选集的全部标签拟合温度与偏置 | `V/reposition/decision_vs_representation/oracle_G_buffered.py`：自然权重、无岭罚；BA 版在拟合后按候选池先验重加权到均匀先验 | 拟合集是**全部缓冲候选集**（每单元 27,673–65,357 个），不是"约 3 万"。实现与 D1 的全池也不同（权重、λ、拟合集），数值不可直接比较。建议放补充材料作为另一种操作化，正文以 D1 为准 |
| BIAS_ORACLE | 检验 3（随机划分、早停）中只校准决策层的上界 | `CHECK3_RESULTS`；C | 只在早停设置下做过，基线偏弱 |
| 标签价值 G(L,k) | 同一单元、同一分类器特征下，ba_ge30(SRS, k) − ba_ge30(k=0) | rev3 B2 | 是"合并重训"这种用法下的价值，不是标签的内在价值 |
| 特征效应 Δ_feat(L,k) | 同一单元、同一 k、同一批 SRS 标签下，ba_ge30(AE 分类器) − ba_ge30(S2 分类器) | rev3 B1 | 相对的是手工 S2 统计 |
| θ_E1 | 只用几何的通路上，AE 指导减 S2 指导的精度差，按单元、方法、分类器、k 等权平均后取 11 国均值 | 冻结方案 §1 | 等价的是等权平均，不是每个国家、每个方法 |
| 类均衡决策头 | 冻结外国模型，只用 k 个目标标签按类均衡权重拟合温度与偏置 | PIVOT F3b 是与重训的混合（α=0.4）；D1 是"只用头"，不混合 | D1 的 Δhead(k) 不能与 F3b 的 +0.94 对照或混称 |
| 标签当量 | 锚定定义 f⁻¹(f(k)+Δ) − f⁻¹(f(k))，f 为 PAV 单调化的 SRS 学习曲线 | rev1 B4，事后改的定义；C | 必须带删失比例 |
| predstrat_srs | 按外国模型的预测类做 sqrt 分配，层内随机抽 | 冻结方案 §4 | — |
| **标签预算 k**（label budget；2026-09-29 作者认可） | 新增目标国标签的个数，每个地块成本相同（室内目视解译情景） | 正文主轴；G、rev2、D1 都用这一口径，k ∈ {0, 50, 100, 200, 400} | 外国池、目标国已有初始样本 S0、测试集都不计入；约 50 个随机标签兼作先验估计与目标区验证，计入 k，不另算 |
| **调查成本预算 B**（survey cost budget；2026-09-29 作者认可） | 实地调查情景下的补采成本，按簇计：簇 = 10 km 格，每簇标 M 个地块；S0 已调查区域内每簇成本 c1，区域外 c2 | 只用于 Betti 模块；相对成本单位，不用货币；c2/c1 事先固定为 {1, 2}（Betti 图 5 的范围）；同时报"同成本下的精度"与"所得标签数"；SPR= 臂（与 EXT 同簇数、全国分散）把"放在哪"与"标多少"分开 | 与 k 不混用；全文不单独用"预算"一词，须写明是哪一种 |
| **初始样本 S0** | 目标国在补采之前已有的标签；Betti 式按簇、只覆盖少数区域的有偏样本，或随机样本 | n0 ∈ {100, 500, 2000}（对应 Betti 的 USA / Togo / India），每簇 M = 10 / 20 / 25 | 是场景变量，不是预算；学习曲线横轴可同时标 n0 + k |

### 4.10 证据分级与披露表（主文或补充材料）

- 逐条 headline（H1–H6 与主要次要结果）列出：等级、是否预注册、测试集是否复用、学习器、国家数、去掉 RO/PT 后的值、出处。
- 由 D3 的 `NUMBERS.csv` 生成。
- 重定位阶段 5 个视角新做的全部事后分析，都进这张表，标 C（评审 1 第二条）：overlap_audit、rse_reviewer、label_budget_line、single_vs_multi、decision_vs_representation。

---

## 5. 结果各节

图表编号是建议。主文图：沿用 3 张现成图，新做 4 张，另做 1 张研究区与设计示意图。现成图的图中文字是中文（`plot_G.py`、`plot_strategy.py`），投稿前都要出英文版，并统一期刊风格。

### 3.1 缺口解剖（RQ1；H1）

**主张与措辞**
1. 〔B；检验 3 预注册，随机无缓冲，LGB〕按 OA 看，跨境损失的 44–67% 来自类别构成：
   - 早停设置 AE 63% [41, 76]、S2 67% [53, 78]；
   - 固定 300 棵树 46% [22, 69] / 44% [26, 60]；
   - 20 km 分块 66% / 72%。
   
   按 BA 看，类别集合只解释 11–15%，BA 的损失主要来自域残差：
   - 早停下 ba_ge30 总损失 AE 17.8、S2 18.7 pp；
   - 固定 300 棵为 12.9 / 13.5 pp。
   
   类别构成部分主要由 RO（+41）、SI（+22）、SK（+15）驱动；PT 以域残差为主（+19 至 +24）。
2. 〔B〕AE 与 S2 的域残差未检出差异：OA +0.00（p=0.58），ba_ge30 +0.73（p=0.70），功效有限。
   - 固定 300 棵下，OA 中 AE 的域残差小 2.4 pp（8/11，p=0.054），是事后的边缘信号（C）。
3. 〔B；事后，缓冲格网【复用】〕同一 LightGBM 只用目标池约 3 万标签，比只用外国高约 13 pp BA、11–12 pp OA（划分 0）。
   - 划分 1 上高 14.4/14.4 pp BA，22/22 个组合为正。
   - 换成 MLP 为 15 pp，LR 为 19–23 pp（只有 5 国试点，划分 1；`MODEL_DIAGNOSIS` M3；质疑者建议由 A 降为 B）。
4. 〔C；缓冲格网上界，另一种操作化〕决策层上界：
   - OA +6.66 / +6.38（11/11、10/11 国为正），BA +10.45 / +9.28（均 11/11）。
   - 划分 0 上，以 tgt300 差距为分母：OA 57% / 53%，BA 78% / 71%；去掉 RO、PT 后 OA 37% / 38%，BA 87% / 80%。
   - 剩余：OA 4.62 / 5.54 pp，BA 2.78 / 3.73 pp。
   - **正文只报 pp**，份额随设置在 32–78% 之间变化。
   - SE、LT、LV、DE、SK 的 BA 剩余出现负值，所以 tgt300 不是严格上限。
5. 〔C；D1〕决策层可挽回部分随 k 的变化（BA，两特征国家值平均）：Δhead(k) 在 k=50/100/200/400/1000 为 +4.48 [+1.64, +7.32]（11/11）/+5.24/+6.17/+6.53 [+4.05, +9.01]/+6.74，全池 +6.55 [+3.99, +9.12]；去掉 RO、PT 后为 +3.21 至 +5.89，全池 +5.67（9/9）。
   - 与同样 k 个标签并入重训的对照：Δretrain(k) 为 +1.52/+1.87/+2.75/+3.33（去掉 RO、PT 为 +0.17 至 +1.06）；Δhead(k) − Δretrain(k) 为 +2.95 [+2.02, +3.88]（10/11）至 +3.42（k=200，9/11），去掉 RO、PT 为 +3.04 至 +4.56（9/9）。OA 上头 − 重训为 −2.92 至 −7.07。
   - 按 D1 同口径报告全池上界与 k≤400 的 Δhead(k)（k=200 已达全池的 94%），同时报 OA 代价；不使用"差一个数量级"的说法（REPOSITION 风险表已按此更正）。
6. 〔C〕国家分型（按结果事后分组，n=2），只作讨论与示例：
   - PT 属表征层：决策层校准后仍差 OA 17.3 / 15.6、BA 12.9 / 9.8。
   - RO 属决策层：OA 上界 k0→校准 AE 24.3→69.2、S2 30.7→66.1。但 RO 只有 3 类，部分是映射伪影。

**图表**
- **图 2a（现成）**：`L/figures/fig_check3_decomposition.png|pdf`。检验 3 的链式分解，要改英文。
- **图 2b（新做，不需训练）**：逐国 × 特征的堆叠条或阶梯图。
  - 内容：k0 水平 → 同样 k 个标签重训的实际兑现（SRS k=200/400）→ 只用头的实际兑现（D1，k=200/400）→ 决策层上界（D1 全池）→ tgt300。
  - 标出 RO、PT。
  - 数据：D1 输出（⟦待定·D1 输出路径⟧）；G 的 `R/guidance_G/rows_t*.csv`（经 `analyze_G.load_rows`，method ∈ {srs, foreign_only}）；`V/model_diag/fit/table_upper_bound.csv`（tgt300，只有划分 0）。
  - 缓冲上界（`oracle_G_buffered_rows.csv`）放补充材料的平行图。
- **表（补充材料）**：AE − S2 在 k0、决策层可挽回量、校准后水平、剩余四个量上的差：
  - 缓冲上界下：dec_oa +0.29 [−1.95, +2.52]，dec_ba +1.17 [−1.37, +3.71]；ora_oa +1.33 [+0.17, +2.50]，ora_ba +3.59 [+1.83, +5.36]（均 10/11）；left_oa −0.91 [−1.73, −0.10]；left_ba −0.95 [−3.65, +1.75]（C，朴素区间）。

**待定**
- D1 的全部数字；
- 定义表中"决策层上界"取 D1 全池还是缓冲上界（作者决定，建议取 D1）；
- rev2 与 M3 的学习器补充只是旁证（M3 本身已含 MLP、LR）。

**写作注意**
- 不写"大头在决策层"。
- 不写"少量标签即可修正决策层"，除非 D1 显示 k≤400 的 Δhead(k) 实质为正，并且写明是 C 级。
- 不报单一份额。

### 3.2 要多少标签（RQ1；H3）

**主张与措辞**
1. 〔B；LGB，合并重训这种用法〕随机目标标签的边际价值小：
   - k=0→50 增益中位数 +0.31 pp；22 条 SRS 学习曲线中 13 条不单调；
   - k=0→200：S2 分类器 +2.41，AE 分类器 +3.10；去掉 RO、PT 后只有 +0.66 / +0.71；
   - k=0→400：中位数 +1.41，均值 3.33，由 RO（+21 / +19）和 PT（+8.5 / +5.9）拉高；
   - DE（AE 分类器）、EE、LT、LV、SE 的增益不到 1 pp。
   
   措辞：写"在合并进 5 万外国样本、LightGBM 重训这种用法下"，不写"标签不值钱"。
2. 〔C，描述〕标签价值与到外国池的平均距离无关（但与"最小纬度差"正相关，见下；属分叉路径，两者都报）：
   - 平均纬度差、大圆距离与 k=50/200/400 增益的 Spearman ρ 在 −0.31 至 +0.33 之间，朴素 p 均 >0.3（`V/reposition/label_budget_line/labelvalue_vs_distance.log`）。
   - EE、SE 到外国池的平均纬度差（8.43°、8.51°）与 RO（7.96°）相近，仅次于 PT（11.44°），但标签价值约为 0。
3. 〔C，描述，n=2〕目标国与外国池类别构成的 L1 距离：RO 1.41、PT 0.86，其余 9 国 0.19–0.56。
   - 高标签价值只出现在 RO、PT；其余 9 国内部没有梯度（ρ 在 −0.47 至 +0.46 之间）。
   - 50 个随机标签估出的 L1 与真值的秩相关 ρ=0.99，但估计值偏高，例如 SK 真值 0.19，估计中位数 0.30。
   - PT 按检验 3 属域残差型，不能归入构成通道。不写因果。
4. 〔C〕学习器依赖：k 从 0 到 200，LR 的增益比 LightGBM 多 +3.2 / +2.9 pp（M8，只有 2 个单元）。以 rev3 B2 为准。

**图表**
- **图 3（新做，不需训练）**：
  - (a) 分国 SRS 学习曲线，k=0/50/100/200/400，AE 与 S2 两种分类器，标出 RO、PT。数据：`R/guidance_G/rows_t*.csv`（method ∈ {srs, foreign_only}）。
  - (b) 标签价值（k0→200，BA 与 OA）对构成 L1 和对平均 |Δlat|。数据：`V/reposition/label_budget_line/composition_by_target.csv`、`labelvalue_vs_distance.csv`。
  - (c) 各学习器的 G(L,k)：✓ `L/figures/fig_ML_rev3.png` 的 (c)(f) 面板（B2 输出）。

**待定**
- 〔rev2 已填〕rev3-B2 判据：LightGBM、RF 满足，LR、MLP 不满足（2/4）→ **依学习器而定**。主张 1 的措辞限定为"LightGBM（及 RF）"，不写"跨学习器"。
  - G(L,200)，两特征平均，去 RO、PT（9 国，df=8）：LGB +0.68 [+0.12, +1.25]；LR +4.37 [+2.02, +6.72]；MLP +1.14 [−0.40, +2.68]；RF +0.56 [−0.12, +1.24]。
  - 11 国合并时 G(L,200) 为 2.8–5.4 pp，由 RO、PT 主导；k=400 时 LR 为 +6.82。
  - 各学习器的 SRS 曲线（ba_ge30，11 国均值，AE / S2 特征）：k=0 → 400，LGB 45.3→48.7 / 42.9→46.2；LR 43.0→50.0 / 40.9→47.5；MLP 50.0→54.7 / 48.0→51.4；RF 42.2→45.9 / 39.3→43.1。
  - 主张 4（M8 的"LR 增益更大"）由此得到 11 国的支持，但仍是 C 级描述（【非盲】【复用】）。
  - 图 3(c) 用 `L/figures/fig_ML_rev3.png` 的 (c)(f) 面板。
- LR 的 SRS 曲线已非盲（rev3 §A），文中标【非盲】。
- 〔rev2 之后新看到的，C 级描述，方案外〕学习器本身的效应大于表征与选样：k=0 时 MLP 的 BA 比 LightGBM 高约 5 pp（AE 50.0 对 45.3；S2 48.0 对 42.9），但 OA 低约 7 pp（64.6 对 71.4）。这与 D1 决策头"BA 升、OA 降"方向一致，提示差别主要落在决策层（类间取舍）。只作讨论材料；MLP、LR 的超参数是预先固定的，未调参，不能写"MLP 更好"。

### 3.3 标哪些地块（RQ2、RQ4；H2 的指导部分与 H4）

**主张与措辞**
1. 〔方向 A，等价 B；G 预注册〕θ_E1 = −0.054 pp：
   - 95% CI [−0.244, +0.136]，90% CI [−0.208, +0.101]，p=0.54，4/11 国为正；
   - TOST p=0.008，能通过 TOST 的最小界值为 0.21 pp；
   - 逐一去掉每个国家，11/11 仍判等价（均值 −0.108 至 +0.012）；去掉开发国后为 +0.030，90% CI [−0.112, +0.173]；
   - 两档预算合并时，所有已报指标上 AE 优势的 90% 上界都 ≤ +0.27 pp（最大为 f1_ge30 的 +0.26）；按 k 拆开，f1_ge30 在 k=50 为 +0.36，其余指标各档 ≤ +0.24。
   
   **必须附的五条限定**（G_RESULTS §1）：
   - 等价的是等权平均（PT −0.71，SE +0.48）；
   - OA 为"不确定"：−0.149，90% CI [−0.432, +0.133]；k=200 单看也是"不确定"：90% CI [−0.323, +0.083]，由 PT 造成；
   - ±0.3 pp 在标签尺度上不小：k=50 时约合 30–90 个随机标签（C）；
   - 本设定对几何指导质量的敏感性没有得到确立：预注册的 ORACLE H = −0.014 [−0.54, +0.51]，判"不确定"；
   - 以稳健预处理为前提：RAWZ 一致；NATIVE +0.41，90% CI [−0.07, +0.90]，判不确定。
2. 〔B〕可分性前提复现了，但没有传导到决策：
   - 候选池 10 近邻 BA：AE 46.6、S2 38.8，11/11 国 AE 更高。这一项设计审查在冻结前已看过（高 8.1 pp），不是 G 的新发现。
   - 外国模型在候选池上的准确率 71.1% 对 70.0%，NLL 0.95 对 0.98，margin 识别误判的 AUROC 都是 0.743。
   - 国家层面有事后正相关 r=+0.62，依赖 SE（C），所以不写"两者无关"。
3. 〔B；预注册次要〕γ = −0.008 [−0.109, +0.093]，没有同源交互。
   - 可靠性 R1：AE 0.301、S2 0.280，差 +0.021 [−0.032, +0.075]。
4. 〔B−；R3 描述〕4 种几何覆盖法相对 SRS：AE 指导 −0.156 [−0.73, +0.42]，S2 指导 −0.102 [−0.65, +0.44]；4 种方法之间 F(3,30)=1.01，p=0.40（重复 0 的 22 单元；全部 44 单元为 F=0.80，p=0.51）。
   - 措辞：写"未检出优于 SRS"，不写"约等于随机"。
5. 〔B−/C〕风险在尾部：kcenter 的 OA −1.76 [−4.31, +0.79]，只有 1/11 国为正，损失集中在 RO −12.55、PT −4.32、SI −1.26；其余 9 国均值 −0.27。
6. 〔C〕predstrat_srs 相对 SRS：
   - 合并 +1.04 [+0.17, +1.90]，10/11 国为正；AE 指导 +1.23，S2 指导 +0.84；
   - 16 项族 Holm 校正后不成立（最小 p=0.198）；按冻结方案做收缩估计，AE 指导收缩到 +0.41；去掉开发国后为 +1.01 [−0.25, +2.27]；
   - k=50 时 OA −0.41 至 −0.52。
7. 〔C〕机制：
   - 类别构成剂量反应：每单位 Δlog(1+n)，该类召回 +3.3 pp，逐国 11/11 为正；
   - 空间摆放不能解释方法差异：扣除构成后与残差的 r=+0.07；
   - 策略排序依赖标签用法：重训与偏置头之间方法效应的相关，BA Pearson 0.12（Spearman 0.23），OA Spearman 0.09（Pearson 0.42）。
8. 〔C，作为测定敏感性的阳性例子〕同一测定能检出约 1 pp 的效应：ORACLE2 的 typiclust +0.93 [+0.34, +1.52]（10/11）；predstrat +1.04。
   - 设计阶段估计的 MDE80 约 0.35 pp。

**图表**
- **图 4（现成）**：`L/figures/fig_G_results.png|pdf`。面板 C 保留"事后、探索性"标注，图中文字改英文。
- **图 5（现成，二选一）**：
  - `V/strategy/benchmark/fig1_methods_vs_srs.png`；
  - 或 `L/figures/fig_strategy_benchmark.png|pdf`（含逐国热图与风险面板）。
  - 图注写明：几何法 44 个单元，预测类与混合类方法 22 个单元；方法 − SRS 是 R3 描述，不是预注册假设。
- **补充材料**：
  - `V/strategy/mechanism/fig_mech1_composition_regime.png`：只用面板 A、B；面板 C 的标题"偏离代表性的代价"不成立。
  - `fig2_consistency_mechanism.png`：面板 (c)"BA 与 OA 排名互不相关"是聚合伪影，要改画成国内相关或删掉。
  - `fig_mech2_strategy_x_usage.png`：只放讨论或补充材料。

**待定**
- 〔rev2 已填〕θ_E1(L) 与判定见 §2.4 的表：情形 3，LGB、MLP、RF 等价，LR 不确定；OA 版四个都不确定。θ(LR) − θ(LGB) BA +0.22 [−0.03, +0.47]。各学习器下几何法 − SRS 与胜率见 `R/guidance_G_ml/report_ML.txt`"几何方法相对 SRS"一节（R3 描述量），由 D3 数字表统一引用。
- 写作要点：等价在三个学习器下成立，但两个树模型本身对标签不敏感；LR 是唯一标签作用大的学习器，其 θ 点估计 +0.17、8/11 国为正。正文写成"指导空间的选择在我们测到的范围内不重要；在最依赖标签的学习器下，结果不足以排除 AE 的小幅优势（≤0.45 pp）"。
- 已知的非盲方向要披露：LR（C=1）在 k=200 下 θ≈+0.12 [−0.15, +0.40]，与 LightGBM 的差中差为 +0.24（p=0.067）。见 rev2 §5。

### 3.4 标签怎么用（RQ3；H5）

**主张与措辞**
1. 〔B；检验 3 预注册次要，随机划分〕用少量随机目标标签估计类别先验：
   - 50 个标签能拿回按真实先验修正所得 OA 增益的 85–97%（早停 95%/97%，固定容量 85%/93%）；200 个标签约 100%。
   - 不用标签的 EM 只有 31% / 58%；BBSE 不稳定。
   - 恢复率被 RO 拉动（RO 占先验增益的 31–42%），所以另报 pp 缺口：50 个标签比真实先验平均少 0.3–0.9 pp。
   - 先验估计的 L1 误差：50 个标签 0.254，200 个标签 0.107。措辞不写"估准"。
   - 标签只来自 1 个区域时恢复约 80%，来自 3 个区域时 91–94%（C；"区域"= 候选池按坐标 KMeans 分成 20 簇后取的簇；检验 3 随机划分；C 级，不进摘要；改用 50 簇时 1 个区域 78%/80%、3 个区域 89%/92%）。
   - 面积统计的精度没有评估。
2. 〔B；定义推论〕同一先验修正使 BA 下降：ba(PO) − ba(LS) 为 −12.21 / −12.77，11/11 国。
   - 不作为新发现，只作为"先定产品"的依据。
3. 〔C；转向评估【复用】，参数在同一测试集上选定〕类均衡决策头与重训的混合（F3b，α=0.4），相对同一批标签的 SRS 重训：
   - +0.94 [+0.69, +1.19]，11/11，OA −0.05 [−0.15, +0.05]；
   - 缩水情况：同一配置（SRS 标签，k=200）留一国选 α 为 +0.95 [+0.71, +1.19]，11/11，OA −0.07；+0.46 至 +1.50 是四种配置（SRS / predstrat_AE × k=50/200）留一国结果的范围，不是缩水；λ 敏感性只在 α=0.5 下算过（λ=0.02/0.1/0.5 时为 +2.46/+1.28/+0.25，frontier3_report），headline 的 α=0.4 没有做；PT 只有 +0.10 至 +0.27。
   - 机制对照：按自然先验加权（F3n）BA −0.42、OA +0.58；不用标签的均衡化（F3u）BA +1.97，但 OA −1.03，0/11 国为正。
   - 不要引用"+0.9 至 +2.5"，它混用了两种基线。
4. 〔C；V9〕同一批标签用在校准层，比并入重训：9 国 +1.95 至 +6.61，RO −0.48，PT +0.21。
   - 与重训增益集中在 RO、PT 正好相反。
5. 〔C；D1〕标签效率曲线（BA）：Δhead(k)/Δhead(全池) 在 k=50/100/200/400/1000 为 0.68/0.80/0.94/1.00/1.03，Δretrain(k)/Δhead(全池) 为 0.23/0.29/0.42/0.51（国家均值之比，描述）；去掉 RO、PT 后重训只有 0.03–0.19。OA 上类均衡头的方向为负（去掉 RO、PT：−2.21 至 −3.17，0–1/9 国为正），重训为 +0.53 至 +1.85。
   - 这决定能否写"少量标签用在决策层比并入重训更划算"。即使能写，也只到 C。

**图表**
- **图 6a（现成，要扩展）**：`V/pivot_eval/prior_shift/fig_prior_shift_frontier.png`（BA–OA 前沿）。扩展时加上 D1 的点，注明参数来源。
- **图 6b（新做，不需训练）**：D1 曲线，横轴 k（对数），纵轴 Δhead(k) 与 Δretrain(k)（BA 与 OA 两个面板），加上全池水平线，标出 RO、PT。数据：D1 输出。
- **表 4（新做）**：先验恢复率（PK50/PK200/EM/BBSE；1 区域对 3 区域）。数据：`R/check3_report.txt`、`R/check3_capacity_report.txt`。

**待定**：D1。

**写作注意**
- "先定产品，再定标签用在哪里"是本节的组织句。
- 决策头相关的一律标"探索性（C），待确认"。

### 3.5 决策 0：无标签选源（RQ4；检验 1，简写）

**主张与措辞**
1. 〔B；预注册，单源两两迁移，LGB，单种子〕各距离与迁移 BA 的平均 Spearman ρ：
   - 地理距离 0.588 [0.37, 0.76]、S2-C2ST 0.541、AE-MMD 0.532 [0.27, 0.71]、S2-MMD 0.511 [0.23, 0.71]。
   - AE-MMD − S2-MMD：Δρ = +0.021 [−0.018, +0.062]，6/11，p=0.43。
   - AE-MMD − 地理距离：Δρ = −0.055，4/11，p=0.25。
   - 稳健性：Kendall τ +0.009；去掉 RO +0.011。
   - 措辞：写"没有检出优于"，这不是等价检验。
2. 〔B，描述〕AE-MMD 与地理距离的秩相关 0.92，S2-MMD 为 0.81。
   - 不写"AE 的域信息只是地理，而 S2 不是"：S2-C2ST 与地理距离也有 0.90。
3. 〔C；次要指标，无区间〕regret：AE-MMD 约 3.2 pp、S2-MMD 约 7.3、地理距离 2.9–4.0、随机 12.9。
4. 不把"选源直接用地理距离即可"推广到多源合并训练。

**图表**
- 正文用一张小表（ρ 与 Δρ），或新做一张点图。数据：`R/check1_per_target.csv`、`R/check1_report.txt`。
- 两两矩阵放补充材料（`R/pairwise_v2.csv`）。

**待定**：无。可选：用 E-OBS 2021 GDD 作附加描述量，先写方案、登记哈希。

### 3.6 表征还是标签：AE 作为特征与学习器稳健性（RQ4；H6 + rev2）

**现有结果**〔C；LGB；缓冲格网【复用】；`V/reposition/label_budget_line/feat_by_k.csv`〕

同一批 SRS 标签下，AE 分类器减 S2 分类器：

| k | ba_ge30 | OA |
|---|---|---|
| 0 | +2.42 [−0.36, +5.20]，8/11，p=0.081 | +1.05，9/11 |
| 50 | +3.02 | +2.79 [+1.36, +4.22]，11/11 |
| 200 | +3.11 [+0.49, +5.74]，8/11 | +2.54，10/11 |
| 400 | +2.47 [+0.22, +4.73]，7/11 | +2.09，10/11 |

- 去掉 RO、PT 后，BA 差在 +2.81 至 +3.33 之间。
- PT、SE、SI 在五档 k 下 BA 都为负（PT −2.2 至 −5.1，SE −1.4 至 −3.1，SI −0.1 至 −0.9）；AT 在 k=400 也为负（−0.14）。
- k 从 0 到 400，这个差只变了 +0.05 [−0.92, +1.02]。这只是没检出变化，不是等价，没有做 TOST。
- LightGBM 的 SRS 学习曲线平行：k 从 0 到 400 的 BA 增益，AE +3.36、S2 +3.31。
- 缓冲上界下，决策层可挽回量两者相同，校准后的水平 AE 更高（3.1 的补充材料表）。

**待定**
- 〔rev2 已填〕rev3-B1 整体平移判据：**0/4 满足 → 依学习器而定**，不能写成 B 级"整体平移"。
  - 四个学习器 k=0 与 k=400 都同号（都为正）；不满足的是"0→400 变化的 95% CI 在 ±1 内"：LGB +0.05 [−0.92, +1.02]，LR +0.31 [−1.24, +1.86]，MLP +1.24 [−1.11, +3.59]，RF −0.05 [−1.07, +0.98]。LGB、RF 只差一点，LR、MLP 区间宽。
  - 能写的：未检出 AE 特征优势随 k 变化（变化量点估计 −0.05 至 +1.24 pp），但区间约 ±1–2.4 pp，不足以确立恒定。
  - Δ_feat(L,k) 的 BA 值：k=0 为 LGB +2.42、LR +2.15、MLP +2.03、RF +2.86；k=400 为 +2.47、+2.46、+3.27、+2.81。11 国下 k=0 的 BA 优势对任何学习器都不显著（p 0.06–0.18）。去 RO、PT 后，LGB、MLP 在五档 k 下 95% CI 都在 0 以上，RF 在 k=0、200、400 在 0 以上，LR 在任何 k 下都含 0（朴素区间，未校正多重比较）。
  - OA 的特征效应依学习器差别大：k=0 时 LGB +1.05 [−0.98, +3.07]，LR +4.94 [+1.90, +7.98]，MLP +0.94 [−4.65, +6.53]，RF +4.17 [+2.47, +5.86]；去 RO、PT 后四个都为正（LGB +2.25 … LR +6.50）。
- 〔rev2 已填〕rev3-B3（RF，k=0，AE − S2，描述）：OA +4.17 [+2.47, +5.86]，10/11 国为正（RO −0.47）；去 RO、PT +4.94 [+3.42, +6.47]，9/9。BA +2.86 [−0.15, +5.86]，不显著，必须同时报。约 4 pp 的 OA 差依学习器而定（LGB 只有 +1.05，MLP +0.94）。RF 的 argmax 并列（k=0 最多占测试集 0.92%）换打破方式后 OA 仍为 +4.16、BA 为 +2.88–2.90，可忽略。
- 〔rev2 已填〕学习器表（表 5）：θ_E1(L) 与判定见 §2.4；技术失败 0；LR 不收敛 0/1804；LR 的 C* 为 3e−4（11 个国家×划分×特征）或 1e−3（33 个），无一落在网格边界。
- 〔rev2 已填〕D2（Cheng 可比口径，补充材料，描述）：`storyline/cheng_comparable/unblinded_4learners/cheng_metrics_summary.txt`。k=0 AE − S2：
  - 全类 BA：LGB +1.98 [−0.69, +4.65]；LR +2.17 [−0.94, +5.27]；MLP +2.80 [−0.26, +5.87]；RF +2.41 [−0.40, +5.22]。
  - 4 类子任务 OA：LGB +1.03；LR +6.16；MLP +1.61；RF +5.39（11 国，CI 都含 0；RO 只剩 grassland，4 类指标退化）。去 RO、PT：LGB +0.62、LR +2.23、MLP +2.06 [+0.79, +3.33]、RF +1.40 [+0.17, +2.64]。
  - 4 类子任务 BA：LGB +1.36；LR +5.22 [+0.27, +10.17]；MLP −0.48；RF +3.62。
  - 写法：在 Cheng 最接近的口径（RF、k=0、4 类）下，AE 相对手工 S2 统计的 OA 差去 RO、PT 后为 +1.4 pp；我们的基线不是 SITS，不能拿来检验 Cheng 的"AEF 比 SITS 低 0.04"。
- **调和 M9 与 rev2**：
  - M9 写"LR 在 k=0 时 C=1 为 −1.48、C=0.01 为 +0.49"，这是 5 国、划分 1 的试点；同为 LR 的 11 国试点（`V/model_diag/altmodels/part1_rows.csv`，划分 0 重复 0）为 +0.08。
  - rev3 §A.2 记录的 LR k=0 AE−S2 为 +2.15 [−1.06, +5.35]（【非盲】，C 由规则选定）。
  - 要在方法或补充材料中说明设定差异，或删去"符号不稳"的说法。

**图表**
- **图 7（✓ 中文版已做）**：各学习器下 Δ_feat(L,k) 随 k 的变化（BA 与 OA），加上去掉 RO、PT 的版本。`L/figures/fig_ML_rev3.png|pdf`（`plot_ML_rev3.py`，只读 `summary_ML_rev3.csv`；(a)(b)(d)(e) 为图 7，(c)(f) 为图 3(c)）。
- **表 5（数据已有）**：学习器稳健性表。数据：`report_ML.txt`、`summary_ML.csv`；配图 `L/figures/fig_multilearner.png`（`plot_ML.py` 重画版式；`analyze_ML.py` 登记了哈希，不改，其原图例压住曲线，重跑后要再运行 `plot_ML.py`）。
  - `L/figures/fig_multilearner_NOISE.png` 是噪声模式的流程测试图，指标为随机数，不是结果，不能用。

**写作注意**
- 全文把 AE 的优势写成"相对手工 S2 统计"。
- "标签汇率"不进结果主张。要写就放讨论，并同时报两组数：
  - 11 国：AE k0 − S2 k400 为 BA −0.89、OA −5.57；
  - 去掉 RO、PT：BA +1.99、OA +0.31。
  
  另外注明学习曲线平坦会放大当量。

---

## 6. 讨论

### 6.1 与 Cheng 的三处对话

1. **类别构成通道**
   - Cheng 引言把先验偏移调整列为需要作物比例数据的方法；它每区-年-类最多 5000 点的均衡设计，在构造上消掉了这一通道（表 3 中 OA≈BA：0.71/0.71、0.75/0.75）。
   - 我们量化了这一通道：OA 损失的 44–67% 来自类别构成（B）；约 50 个随机目标标签就能拿回先验修正 OA 增益的 85–97%（B）。
   - 附 D2（补充材料，描述，C 级）：k=0 下 AE−S2 的全类平衡精度 +1.98 [−0.69, +4.65]（7/11），4 类平衡精度 +1.36 [−3.28, +5.99]（8/11），4 类 OA +1.03 [−0.92, +2.98]（8/11）；4 类齐全的 8 国为 +0.11 [−6.18, +6.40]（方案外）。各版本 CI 都同时含 0 和 Cheng 只用 S2 对照的 +4 pp，只能说方向与只用 S2 的对照一致；本数据检验不了"比完整 SITS 低 0.04"。
2. **可分性是否传导到决策**
   - 前提复现了：10 近邻 BA 46.6 对 38.8，11/11（B）。
   - 但在三处决策上都没有兑现：
     - 选样指导等价（θ_E1，A/B）；
     - 选源时 AE 距离不优于 S2 距离（Δρ +0.021，B）；
     - margin 识别误判的 AUROC 两者都是 0.743（B）。
   - 旁证：Cheng Fig. 11b 中 JMmin 差与 OA 差显著正相关，但只解释约 18% 的方差（已读图核实：图例"slope=0.111, R²=0.18***"，RF，567 例；正文只写"positively associated"）。引用时注明是图中读数，并说明 Cheng 以此支持其机制。
   - 要如实写出唯一与 Cheng 同向的信号（C，多个对比中的一项）：
     - 决策层校准后，AE 剩余的 OA 差距比 S2 小：缓冲格网 −0.91 [−1.73, −0.10]；检验 3 的早停设置 −1.21、固定 300 棵 −1.76。
     - 它只有决策层可挽回 OA 量（+6.4 至 +13.5 pp）的约 1/15 至 1/4。
3. **"远距离需要本地标签"在多源设定下的形态**
   - 标签价值与到外国池的平均距离（平均纬度差、平均大圆距离）无关（ρ 在 −0.31 至 +0.33 之间，C）；但同一脚本算的"到外国池的最小纬度差"给出 ρ=+0.53 至 +0.77（朴素 p 0.006–0.095，原日志未打印），属分叉路径，不能泛称"与距离无关"。
   - 高价值只出现在 RO、PT（n=2，C）：RO 是构成失配加映射伪影，PT 是表征层问题。

### 6.2 一致之处与不一致之处

**一致之处（独立印证）**
- 协议效应远大于特征效应：Cheng 从本地到跨区约掉 23 pp，AEF 与 SITS 只差 0.02–0.04；我们原稿 P3 从 ID 到 LOCO 掉 12–27 pp，AE 与 S2 差不到 1 pp。注意 P3 是修复前的数据，B 级。
- AE 不能普遍替代常规特征。
- 距离能预测迁移损失（检验 1 中地理距离 ρ=0.588）。
- 可分性更好不等于决策更好。

**不一致之处与调和**
- **本地精度方向相反**：
  - Cheng AEF 低 0.02；我们 pairwise_v2 国内测试 AE 55.54、S2 53.35。
  - 归因于基线不同：Cheng 自己只用 S2 的 SITS 跨区 OA 为 0.67，低于 AEF 的 0.71。
  - 这两组数字来自不同划分，只作定性调和，不放进同一个比较。
- **距离斜率**：只放补充材料 S5，统一用这句措辞：
  > 在我们的设定下未复现 Cheng 的正斜率；按其数据换算的预测 +1.35 pp/° 被国家块 bootstrap 的 95% 上界 +0.04 排除；点估计为负，95% CI 含 0。
  
  同时说明：逐国的表征层 AE−S2 差值随训练设置翻号（PT 为 +4.9 或 −11.9），所以国家级数据无法检验热量梯度。

### 6.3 七项设定差异表（放讨论正文；Cheng 一列已对原文核对）

| # | 设定 | Cheng et al. 2026 | 本研究 |
|---|---|---|---|
| 1 | 基线特征 | SITS 与 AEF 输入对齐：S2+S1+Landsat，105 维时间序列合成 | S2 手工年度与季度统计，168 列 |
| 2 | 类别 | 欧洲 6 类：MFP、玉米、马铃薯、冬油菜、冬大麦、冬小麦 | 10 个 Tier-B 类，含异质类与稀有类；RO 只有 3 类 |
| 3 | 类别先验 | 每区-年-类最多 5000 点，近均衡，OA≈BA | 自然先验，OA≠BA，双报 |
| 4 | 训练源 | 单源区域到单目标区域，零目标标签 | 10 国合并的外国池（5 万），加 k 个目标标签 |
| 5 | 空间单元 | 7 国 10 个 NUTS 区（9 个训练区），样本为地块内单点 | 11 个国家，地块级 30 m 邻域统计 |
| 6 | 年份 | 2017–2023，7 年 | 只有 2021 年 |
| 7 | 测试划分 | 区内按类分层 70/10/20 随机划分 | 10 km 格网 + 1 km 缓冲，预注册，并做测试集审计 |

另外可以加一行"分类器"：Cheng 以 RF 为主，另有 LinearSVM、XGB 与两种深度模型；我们以 LightGBM 为主，另有 LR、MLP、RF 的稳健性分析。与 Cheng 重叠的国家只有 AT、DE、PT、SE，而且 Cheng 的 DE 是勃兰登堡。

### 6.4 给实践者的预算方案（放讨论末尾，每条标等级）

1. **先采约 50 个空间分散的随机标签，来自 ≥3 个区域**〔先验估计为 B；空间分布效应为 C〕。它们同时承担三件事：
   - 满足 Cheng 所说的目标区验证；
   - 估计类别先验：按 OA 或面积交付时回报最确定；
   - 检查构成是否严重失配（C）。
2. **先定产品**〔B〕：
   - 面积统计或 OA：做先验修正；
   - 逐类制图或 BA：不做先验修正，否则 BA 会降约 12 pp。
3. **超过 50 个以后**〔B，LGB〕：并入外国样本重训时，k≤200 的回报很小（9 国约 +0.7 pp BA），构成失配或表征层问题突出的国家例外。〔rev2 已填：只对树模型成立（LGB、RF）；LR 去 RO、PT 后 k=200 增益 +4.4 pp，MLP +1.1。改写为"用树模型并入重训时回报很小；回报大小依学习器而定"〕
4. **选样**：
   - 默认 SRS〔B−〕；
   - 以 BA 为目标、k≈200 时，可按外国模型预测类做 sqrt 分层〔C〕；
   - 避免 kcenter 一类的最远点方法〔B−/C〕；
   - 几何覆盖法不值得投入工程成本〔B−〕；
   - 先检查外国模型的预测类是否覆盖目标国的类，没覆盖到的类分层帮不上〔C〕。
5. **指导空间**：用 AE 还是 S2 都可以，取便宜的那个〔A/B，限 BA 类指标；rev2：LGB、MLP、RF 下等价，LR 下不确定（≤+0.45 pp）〕。
6. **标签用法**：以 BA 为目标时，冻结外国模型、只拟合一个类均衡决策头，可能比并入重训更划算〔C，待确认；rev4 D1：相对同样 k 个标签重训 BA +2.95 至 +3.42，OA −2.92 至 −7.07（k=50–400）〕。以 OA 或面积为目标时不适用。
7. **表征层问题**（PT 这一型）：靠少量标签补不上，需要大量目标标签〔方向 B；分型为 C〕。

### 6.5 局限

1. **没有 TESSERA，也没有对齐的 SITS。**
   - Cheng 表 5 中 TESSERA 的 RF 迁移 OA 为 0.83，AEF 为 0.70；这是 2017、2018 两年、162 个跨区迁移的子集。
   - TESSERA 在 2021 年欧洲是否有覆盖，尚未核实。
   - 本文结论不依赖"哪个表征最好"，AE 只是被检验的因子。
2. **单年（2021）、国家级粒度、农户申报标签。** 国家内部异质性大，例如 SE 地块纬度的 p10–p90 为 56.2–61.5°。
3. **RO 标签映射伪影**：只有 3 类。LV、LT 缺马铃薯与草莓。每个均值都附去掉 RO、PT 的值。
4. **测试集复用**：G 的 s0/s1 在预注册主分析之后被 rev1、G_audit、转向评估、模型诊断、重定位分析和 rev2–rev4 多次使用。正向结论都是 C。
5. **事后分析多**：朴素 p 只作描述。标签当量的定义是事后改的。
6. **学习器**：G、检验 1、检验 3 都只用 LightGBM；r=50 和 300 轮沿用阶段 A 的设置。rev2 只回答选样指导与 SRS 曲线的学习器稳健性，不覆盖 predstrat、决策头。
   - 〔rev2 已填〕LR、MLP、RF 的超参数预先固定、未调（LR 的 C 按外国留国家组规则选；MLP 固定 20 轮）。LR 的 C 在约 4 万样本的训练折上选定、用于 5 万 + k 的训练集，sklearn 的正则按样本权重和缩放，最终拟合的正则约弱 20%（影响小，披露）。rev2 写的"RF 每进程 4 线程"实际为单线程（joblib 在子进程中降为 1），结果确定，不受影响。
   - 〔rev2 已填〕**测定敏感性与学习器**：两个树模型本身对目标标签不敏感（去 RO、PT 后 G(200) ≈ +0.6 pp），它们下面的"等价"部分反映的是标签整体作用小；标签作用最大的 LR 结果不确定。这一条与第 7 条（等价的测定敏感性有限）合写。
7. **等价的测定敏感性有限**：
   - SRS 曲线平坦；ORACLE 阳性对照判为"不确定"；
   - ±0.3 pp 在 k=50 时约合 30–90 个标签；
   - 所以写"未发现优势，并给出上界"，不写"信息相同"。
8. **没有评估面积精度**；**每地块等成本**的假设；**方法覆盖面窄**：
   - 构成设计只有 sqrt 分层；不确定性法只有单轮 margin；
   - 没有整群、行政区分层或成本模型；k≤400；
   - 每个单元只有一次 SRS 抽样。
9. **缓冲与转导**：
   - 1 km 缓冲按质心近似，大圆距离下略不足（SE 约 0.90 km）；
   - 标准化与 PCA 在含测试地块的全体地块上拟合（不用标签，对 AE、S2 对称）。

---

## 7. 补充材料清单

| 编号 | 内容 | 状态 | 来源 |
|---|---|---|---|
| S1 | 预注册与修订时间线：各方案的哈希、冻结时是否已看过结果 | 现成，整理即可 | `L/plan_hashes.txt`；rev1–rev4 |
| S2 | 证据分级与披露总表（逐条 headline） | 待 D3 | D3 `NUMBERS.csv`；§8 |
| S3 | 数据：各国各类地块数；各单元测试集、候选集大小，以及 ≥30 地块的类 | 现成 | `V/storyline/paper_plan/*.csv` |
| S4 | 测试集审计：固定性、各臂逐位相同、缓冲细节、复用情况、rep 重合 | 现成 | `TESTSET_AUDIT_2026-09-26.md`；`V/testset_audit/` |
| S5 | 单源 \|Δlat\| 反向分析（C） | 先登记方案哈希，并注明纬度代理结果已看过；可选改用 E-OBS GDD | `V/reposition/overlap_audit/`（`fig_posthoc_distance.png`、`posthoc_summary.json`） |
| S6 | 两两迁移矩阵与检验 1 的完整表 | 现成 | `R/pairwise_v2.csv`、`R/check1_report.txt` |
| S7 | Cheng 可比口径（D2）：全类平衡精度；4 类子集的 OA 与平衡精度；逐国每类测试样本数 | ✓ 四学习器已算 | `storyline/cheng_comparable/unblinded_4learners/`；样本数见附录 B |
| S8 | 多学习器完整输出：θ_E1(L)、各指标、SRS 曲线、失败与不收敛计数、C* 分布；rev3 B1–B3 | ✓ 已有 | `report_ML.txt`、`report_ML_rev3.txt`、`fig_multilearner.png`、`fig_ML_rev3.png` |
| S9 | 检验 3 的固定 300 棵稳健性与 20 km 分块；决策层上界在两种设置下的对照 | 现成 | `R/check3_capacity_report.txt`；`V/reposition/decision_vs_representation/oracle_c300.log`、`layer_split.log` |
| S10 | 缓冲格网决策层上界（另一种操作化）与逐国表 | 现成 | `oracle_G_buffered_rows.csv`、`oracle_G_buffered.log` |
| S11 | G 的完整次要结果：通路、γ、XY/S2XY、ORACLE/ORACLE2、P2 拆分、E_far、按 k 的 TOST、留一国 | 现成 | `R/guidance_G/report_G.txt`；`G_RESULTS.md` |
| S12 | 预处理敏感性（RAWZ、NATIVE） | 现成 | 同上 |
| S13 | 标签当量换算（锚定定义、删失比例、跨国平均曲线） | 现成 | `G_RESULTS.md` §5 |
| S14 | 策略基准逐方法表、风险画像、剂量反应、空间摆放 | 现成（部分图要改） | `V/strategy/benchmark/key_table_methods.csv`、`V/strategy/mechanism/*` |
| S15 | 转向评估：决策头的参数敏感性（α、λ）、机制对照（F3n、F3u、F3z）、R2V | 现成（C） | `V/pivot_eval/prior_shift/*`、`PIVOT_EVALUATION` |
| S16 | 模型诊断：拟合阶梯、轮数与容量、替代学习器 | 现成 | `MODEL_DIAGNOSIS`；`V/model_diag/fit/report_fit.txt` |
| S17 | 检验 2 试点（4 国、无缓冲）与阶段 A 的 PT 期中结果，作背景 | 现成 | `CHECK2_RESULTS.md`；`fig_stageA_interim_PT.png` |
| S18 | 标签价值对距离、对构成的描述统计 | 现成 | `V/reposition/label_budget_line/*` |
| S19 | D1 的完整表：逐国、逐 k，包括 k=1000 与全池 | 待 D1 | D1 输出 |

---

## 8. 必须披露清单

汇总自 rev1–rev4、TESTSET_AUDIT、CONCLUSIONS、REPOSITION 与评审。主文方法节放 1–2 段摘要，全部条目放补充材料 S1/S2/S4。

### A. 预注册与分析路径

- A1. G 的预注册主分析只用了一次 s0/s1 测试集。之后被 rev1（层级 3/4）、G_audit、转向评估（选定 α、β、γ、λ 与头的参数化）、模型诊断（按测试结果选轮数和 τ）、重定位分析、rev2–rev4 复用。相应数字一律标为探索性。
- A2. rev1 在看过层级 1+2 的全部结果之后才写：
  - ORACLE 几何与方案不符（没做逐列 z 截断，缺席类用最小值填充），ORACLE2 为修正版，与原版并列；
  - 混合模型没有用 statsmodels，原来的"REML 随机效应"一行实为逆方差加权元分析；
  - 层级 3 只补跑了 RAWZ、NATIVE；
  - verdict() 措辞已对齐；
  - 设计审查预写的两句解读不适用。
- A3. 标签当量的锚定定义是 rev1 事后修改的；设计审查对"k=200 时 0.3 pp ≈ 预算 27–45%"的事前预期偏低。
- A4. 术语改动：rev1 B5 废弃了"条件信息"一词。
- A5. rev2 相对冻结方案是事后追加的学习器稳健性分析，不是确认性检验：
  - LR 的 C 网格在冻结前冒烟测试（只用外国数据）后向下扩展；
  - MLP 种子方案在冻结前改定；
  - rev2 §5 列出的格子冻结前已看过：LR（C=1）E1×k=200 共 704 次拟合；LR/MLP 的 SRS k=0/200 s0r0、s1r0；RF 的 SRS k=200 s0r0 与 AT k=0；
  - 已知方向：LR k=200 θ≈+0.12，差中差 +0.24（p=0.067）。
- A6. rev3 §A 的非盲事件：
  - 09-27 11:35–11:38，一个重定位代理读取了 LR 的全部 SRS 与 k=0 行，并算了 AE−S2。没有读 E1 行，没有算 θ_E1。
  - 作者侧分析者读到了 LR k=0 AE−S2 的 +2.15。
  - 三个视角都看过 LightGBM 的 Δ_feat。
  - 论文中 LR 的 SRS 曲线、AE−S2 特征差以及 rev3 B 中 LR 的全部量，都标"非盲"。
- A7. rev4 的 D1–D3 在运行前冻结，但仍在复用测试集上，属 C 级。D2、D3 中涉及 LR、MLP、RF 的部分，在 rev2 正式分析之后才计算。
- A8. 重定位阶段 5 个视角的事后分析（overlap_audit、rse_reviewer、label_budget_line、single_vs_multi、decision_vs_representation）全部列入披露表，标 C，不能只挑有利的。
- A9. 检验 3 的容量检验（固定 300 棵）是事后补充（`ANALYSIS_ADDENDUM_check3_capacity.md`）。早停设置下"63–67%"不作效应大小。
- A10. 检验 1 的方案哈希没有登记在 plan_hashes.txt 中（§4.7），需说明。

### B. 设定、参数与已知数字

- B1. r=50 是在 PT、BE、SI 的测试集上、用 S2 分类器与 SRS 选定的，落在网格上边界。AE 分类器直接沿用。
- B2. G 中 SRS k=50、S2 分类器的 6 个数字，与冻结前已看过的阶段 A rgrid 逐位相同：PT、BE、SI 的 s0、s1 为 27.18、24.58、49.56、45.84、39.19、39.50。
- B3. 冻结前已知的信息：
  - 检验 2 的 AE 对 S2（+0.02 pp）；
  - 设计审查的几何诊断读过标签（kNN 纯度；10 近邻 BA 在 11 国 AE 高 8.1 pp）；
  - `tree_check.csv` 中存有 PT、SE 的对比，作者冻结前没有查看；
  - 保留 kherding 只依据耗时。
- B4. PT、BE、SI 是阶段 A 的开发国，PT 做过期中查看；阶段 A 与 G 共用 PT 的单元，SRS 抽样逐位相同。
- B5. 300 棵树与 LightGBM 参数沿用阶段 A；k=0 的测试概率以 float16 保存（最多影响 1 个 argmax）。

### C. 测试集与隔离

- C1. 检验 2、3 用随机 70/30 划分，没有空间缓冲：测试地块到最近候选的距离中位数 221–737 m；PT 有 20.6% 在 100 m 以内；每国有 0–7 个测试地块与候选地块特征完全相同。另外，PT、SE、SK、LV 上检验 3 的测试集与检验 2 的 test_idx 逐位相同（TESTSET_AUDIT §2–3），检验 3 预注册分析运行时这 4 国的测试集已在检验 2 中评估过。
- C2. 1 km 缓冲在单一 lat0 的近似投影上按质心计算：
  - 大圆距离下 22 个单元都略低于 1 km（SE 约 0.90 km）；
  - RO 的大地块用等面积圆近似时，有 211–221 个候选的外缘离测试地块不到 0.5 km；
  - 外国池中有 0–7 个地块离目标测试地块不到 1 km。
- C3. 稳健 z 标准化与 PCA 在包含测试地块特征的全体地块上拟合：不用标签的转导，对 AE 与 S2 对称。
- C4. rep1 不独立：rep0 与 rep1 候选池的 Jaccard，EE、LV 为 1，只有 BE、DE 约 0.3。EE、LV 两个重复的候选池相同。推断单位是国家。
- C5. E_far 的类集合取整个测试集中 ≥30 的类，其中部分类在 E_far 子集内不足 30。
- C6. 检验 1 已评估过每国的全部地块，所以严格意义上不存在"从未被任何评估碰过"的地块。

### D. 数据

- D1. dataset_v2 修复了映射与 BE_WAL；原稿数字来自修复前的数据：9.7% 的地块被误归入 other_crops，LV 的 S2 缺 4 类，BE_WAL 的 AE 丢了 4,995 行。
- D2. LV 的 S2 第 0008 批按类别选择性缺失，已排除，因此 LV 缺马铃薯和草莓。
- D3. RO 只有 3 类，原因是 arable_crops 被归入 other_crops（映射伪影）。
- D4. SK 有 825 个地块没有作物信息。CHECK1 写"build 脚本已改为排除，下次重建数据集生效"；需确认当前 dataset_v2 是否已排除，并按实际情况披露。
- D5. 检验 3 训练集短缺：PT 的 orchards 缺 459 个，RO 的 tree_wood_forest 缺 346 个；早停集也有短缺。
- D6. **新发现，需核实原因**：LT 在 dataset_v2 中马铃薯与草莓都是 0（本计划计数）。已读文件中只解释了 LV 的原因。

### E. 实现与口径

- E1. 缓冲格网决策层上界用的是缓冲候选集的**全部**标签（每单元 27,673–65,357 个），不是"约 3 万"；实现是自然权重、无岭罚、事后重加权。rev4 D1 的全池用 3 万候选池、类均衡权重、λ=0.1。两者要分开标注（§4.9）。
- E2. D1 的"只用头"与 PIVOT 的 F3b（α=0.4 的混合）不是同一个量。
- E3. 决策头的 α、β、γ、λ 都是在 G s0/s1 上选定的；LAM0=0.1 在 calib_diag 中注明"预先固定，未在测试集上调"。
- E4. 与 Cheng 对照的事后分析是在看到 Cheng 之后做的。

---

## 9. 投稿前待办与时间线

### 9.1 待办（按优先级；已完成的标 ✓）

**P0（已完成）**
- ✓ rev3 披露补遗与补充估计量，11:56 登记。
- ✓ rev4（D1–D3）冻结，14:48 登记。

**A. 等 rev2（✓ 09-27 晚全部完成；独立重算核验通过）**
1. ✓ rev2 完成后先跑完整性检查，再依次正式运行 `analyze_ML.py` → `analyze_ML_rev3.py`，按冻结顺序报告。
2. ✓ 据此填入的占位符（标〔rev2 已填〕）：
   - §2.2 第 7、8 句与 §2.4；
   - 3.2 的 B2；
   - 3.3 的 θ_E1(L)；
   - 3.6 的 B1/B3 与学习器表。
3. 凡依赖"LightGBM 学习曲线平坦"的说法（重训稀释、AE 特征优势抵得上 400 个标签），都以各学习器的结果为准；符号不同的就降级或删除。

**B. 只分析已有数据（不训练，线程 ≤2）**
4. **D1**（现在就能做，只用 LightGBM 阶段 0 概率）：填 3.1、3.4 与定义表。
5. ✓ **D2**：LightGBM 部分已做；四学习器版本 09-27 晚已做（`cheng_comparable/unblinded_4learners/`）。
   - RO 在 4 类子集中只剩 grassland 一类，4 类指标退化，要单列或剔除并说明。
   - LT、LV 实际是 3 类子集（无马铃薯）。
6. **D3**：`numbers_table.py` → `NUMBERS.md` / `NUMBERS.csv`。在 rev2、D1、D2 都完成后生成，论文数字全部改为从这张表引用。顺带统一：
   - 单源斜率一处写 95% CI [−0.91, +0.04]，另一处写 90% CI [−0.78, −0.04]，统一为 95%；
   - pairwise_v2 与 G 的数字不混用。
7. 证据分级与披露表（S2）；每个均值附去掉 RO、PT 的值。
8. **调和 M9 与 rev2**（LR k=0）。
9. **测定敏感性段落**：只报 AE 优势的 90% 上界（两档 k 合并时 ≤+0.27；f1_ge30 在 k=50 为 +0.36），给出标签尺度（C），加阳性例子（ORACLE2 typiclust、predstrat，C）和 MDE80 约 0.35 pp。
10. **写作清理**：
    - Cheng 引文统一改为 §4.3；
    - 相对 Kluger 2021、Saerens 2002、DirPA 重写新颖性，并补上 Betti 2026、EuroCropsML 与作物主动学习文献的定位；
    - 删除原稿中已不成立的主张（§10）。
11. **图件**：
    - 3 张现成图改英文；
    - 新做 4 张：图 2b、图 3、图 6b、图 7；
    - 新做研究区与设计示意图（图 1）；
    - 统一期刊风格。
12. **与 Cheng 相关的核对**：
    - 把 PDF 复制进 `reference/`（附录 B）；
    - ✓ 已读图核实：Fig. 11b 图例 slope=0.111、R²=0.18***；Fig. 12b 图例 SITS-LDA5D slope −0.14（ALR 0.14，R²=0.37），AEF −0.14，SITS −0.23（见 doc_fixes/DOC_FIXES.md I 节）；
    - 读 Cheng 的补充材料（Table S3/S4、Methods S3）；读之前不写"未处理区域对之间的依赖"。
13. **数据核实**：LT 马铃薯与草莓为 0 的原因（§8 D6）；SK 的 825 个无作物信息地块是否已排除（§8 D4）。
14. **增量文献检索**：投稿前 4–6 周做，覆盖 Cheng 2026、Betti 2026、GeoSpOT、EuroCropsML、DirPA 的前向引用，确认没有人先发表"迁移场景下的目标标签预算与选样"。

**C. 作者要做的唯一大取舍：确认运行**（TESTSET_AUDIT §5）
15. 做法：
    - 在从未测试过的格子上重新划分，用新的随机数命名空间，把储备切成 F1/F2 两半，只用 rep0；
    - 训练前冻结 α、β、γ、λ、big 类集合和 E_far；
    - 只比较 SRS、predstrat 与决策头；每个单元做 ≥5 次 SRS 抽样。
    
    需要重建 stage0 并训练，耗时待估。
16. 判断规则（REPOSITION §6 C14）：
    - rev2 显示各学习器的 SRS 曲线都平、等价跨学习器成立 → 可以不做，直接投，正向建议写"未确认"；
    - rev2 结果混杂 → 确认运行是提升接受概率最大的单项。
    - 〔rev2 已填〕实际结果属"混杂"：等价 3/4（LR 不确定），B1 0/4、B2 2/4 满足。按本规则应认真考虑确认运行；是否做由作者决定（作者 09-25 曾要求"暂时先不做新的实验"）。

**本轮不做**：去近源的合并训练；用 RF 重跑两两迁移；对齐的 SITS；TESSERA；跨年分析。

### 9.2 时间线（相对 T0 = rev2 完成）

REPOSITION 估计 rev2 完成后约 3–4 周可出完整初稿。下表是建议安排，不含确认运行。

| 时间 | 内容 |
|---|---|
| T0 之前（现在即可做） | D1；D2 的 LightGBM 部分；图 2a、4、5 改英文；方法节、引言初稿；定义表（D1 数字留空）；数据核实（LT、SK）；Cheng 图表数字核对 |
| T0 当天 | 完整性检查 → analyze_ML.py → analyze_ML_rev3.py；按冻结顺序判定；确定摘要第 7、8 句与标题 2 能否用 |
| T0 + 1 周 | D2 的其余学习器；D3 数字表；结果 3.1–3.6 初稿；图 2b、3、6b、7 |
| T0 + 2 周 | 讨论、补充材料、披露与分级表；决定是否做确认运行 |
| T0 + 3–4 周 | 完整初稿；按 §10 做一次内部对抗式核查；增量文献检索（投稿前 4–6 周要再做一次）；投稿信中说明与 Cheng 的关系 |
| 若做确认运行 | 先写方案并登记哈希，再重建 stage0 → 训练 → 分析。完成后按结果改写摘要与 3.3、3.4 |

---

## 10. 不能写的说法清单

**关于 AE 与 S2 作为选样指导（G）**
- "AE 与 S2 提供同等的指导信息""精度相同"。
- 一般意义上的"AE 嵌入没有指导价值"。
- "OA 非劣效"。
- "AE 在预测通路更好"及其机制解释（+0.34 只能写成需要确认的探索性信号）。
- "连神谕都无提升""任务对选样不敏感""几何选样几乎不影响精度"。
- "结论不依赖预处理""AE 的原生几何优于 S2"。
- 不限定几何通路的"指导来源是次要因素"。
- 不带删失比例的标签当量；把"标签汇率"写成 headline 或标题级结论。
- 用 Jaccard≈0.009 证明两者选出了不同的地块。
- rev2 出结果之前出现"跨学习器"。〔rev2 已填：结果为情形 3，因此任何时候都不能写"跨学习器等价""等价不依赖学习器""across four learners"；只能逐个列出学习器。〕
- 不附"两个树模型对标签不敏感"这一限定，就用 RF 的等价来加强结论。

**关于选样策略**
- "30 多种方法≈随机"（只有 PT 的期中结果）；几何法"约等于随机"（改写"未检出优于 SRS"）。
- "策略无关紧要"；"BA 与 OA 的方法排名互不相关"。
- "分布匹配法、typiclust 或 kherding 的 OA 最佳"。
- 泛指所有策略的"策略优势随预算增大"。
- 引用这些数字：方差份额 78.5% / 3.2% 和 66.8% / 6.9%（改用每自由度均方比）、"8/8 个设置"、64 格中的 +1.87。
- 下面这些作 headline："predstrat 约等于标签翻倍""AE 指导下策略更有效""OA 上某方法显著优于 SRS"。
- "RO 损失来自类内代表性""离群比例导致 kcenter 受损""空间摆放不重要"。
- 不附设定的神谕上限或真实类配额外推（+1.7 至 +2.0）。

**关于缺口、标签价值与用法**
- 把 BA"域残差 ≥100%"当数字引用；把检验 3 早停设置下的"63–67%"当效应大小；"剩余 11–13 pp 需要改变表示"当一般数字。
- 报单一的"决策层份额"；"大头在决策层"。
- D1 之前写"少量标签即可修正决策层"。
- 把缓冲上界说成"用测试标签拟合的"，实际用的是候选集标签；或说成"用约 3 万标签拟合"，实际是 2.8–6.5 万。
- "标签不值钱"（改写"合并重训这种用法下的边际价值小"）。
- "标签价值集中在构成失配国家"作因果（n=2）；把 PT 归入构成通道。
- "预算向 RO、PT 型国家倾斜"：这类国家事前无法识别，而且 RO 是伪影。
- "50 个标签就能把类别构成估准"：L1 误差 0.254。
- "50 标签 + 决策头 ≈ 400 标签重训"：OA 低 1.3。
- 决策头"+0.9 至 +2.5"：混用了基线。
- 把 D1 的 Δhead(k) 与 F3b 的 +0.94 当成同一个量。
- "面积统计已解决"：面积精度没有评估。
- 把"先验修正使 OA 升、BA 降"或"类别构成对 OA、BA 方向相反"写成新发现。
- 把"选源直接用地理距离即可"推广到多源合并训练；"AE 的域信息只是地理，而 S2 不是"。

**关于 AE 作为特征**
- rev3 B1 判据满足之前，把"整体平移"写成 B 级结论；没有做 TOST 就写"平移不变"或"等价"。〔rev2 已填：B1 判据 0/4 满足，"整体平移"不能写成结论，只能写"未检出随 k 的变化，区间宽"。〕
- "用 RF（Cheng 的主分类器）时 AE 比 S2 高约 4 pp"而不附 BA 不显著（+2.86 [−0.15, +5.86]）与学习器依赖（LGB +1.05、MLP +0.94）。
- "几百个随机标签的边际价值小"不限定学习器（B2 判据 2/4）。
- "MLP 更好""学习器比表征更重要"作结论：MLP、LR 未调参，且 MLP 的 BA 优势伴随 OA 下降。
- 把"AE 更好"写成一般结论。只能写"相对手工 S2 统计"，并注明依学习器与国家而变。

**关于 Cheng**
- "Cheng 错了"；"方向相反且稳健"（改写"未复现"）。
- 读 Cheng 补充材料之前，写"Cheng 没有处理区域对之间的依赖"。
- 引文标 §4.1（应为 §4.3）。
- 把我们的基线称为 SITS。
- 把 SITS-LDA5D 写成"shows"（只能写 "motivates"）；引 LDA5D 的 ALR（0.14）和 Fig. 11b 的 R²（0.18）时不注明"图中读数"；把 Fig. 11b 写成"Cheng 显示可分性与精度联系弱"。
- 在同一个比较中混用 pairwise_v2（随机划分）与 G（缓冲格网）的数字。

**原稿中已不成立或已被抢先的**
- "首个欧洲两两迁移矩阵"；"≥10 国 + 协调标签 + 诊断"首创；任何"首个"类主张，包括首个作物主动学习、首个用嵌入选样。
- 标题或 headline 中的"Stronger geographic encoding"与地理编码叙事。
- "AlphaEarth 是唯一以预计算嵌入发布的 EOFM"：TESSERA 已有。
- CKA 1.5×、国家探针 94.4/88.8、UMAP、坐标消融作为机制。
- 马铃薯 +11.9（LV 数据伪影）；对称性；OLS 决定因素；"跨境优势消失，有统计支持"；LV −18.6 是国家离群；同源交互 γ≈+0.29；把 Fusion 作为推荐。
- 与 Rauch 2025 形成"直接反差"；用"外国池让模型已经胜任"解释与 Kaijage 2024 的差别。
- 标题或摘要写 "pre-registered strategy benchmark"。

**统计口径**
- 正文引用事后分析的朴素 p，例如 ρ=−0.755、p=0.007（改用平均公里距离后 ρ=−0.28）。
- 把 rep 当独立重复（推断单位是国家）。

---

## 附录 A：关键数字索引（D3 之前的临时版；论文最终以 NUMBERS.csv 为准）

| 量 | 值 | 等级 | 出处 |
|---|---|---|---|
| θ_E1（ba_ge30） | −0.054；95% CI [−0.244, +0.136]；90% CI [−0.208, +0.101]；TOST p=0.008 | A/B | `R/guidance_G/report_G.txt`；G_RESULTS §1 |
| AE 优势的 90% 上界 | 两档 k 合并时所有指标 ≤ +0.27（ba_ge30 +0.10、OA +0.13、f1 +0.26、ba_P1 +0.14、oa_P2 +0.19）；按 k 拆开：ba_ge30 k=50 +0.15、k=200 +0.08；f1_ge30 k=50 +0.36（超过 0.27） | B | CONCLUSIONS 附录 C |
| OA 版 θ_E1 | −0.149；90% CI [−0.432, +0.133]，不确定 | B | G_RESULTS §1 |
| 10 近邻 BA | AE 46.6、S2 38.8，11/11 | B | G_RESULTS §2 |
| 选源 Δρ | AE−S2 +0.021 [−0.018, +0.062]；AE−地理 −0.055 | B | CHECK1_RESULTS |
| OA 损失中类别构成的份额 | 44–67%（早停 63/67，固定 300 棵 46/44，20 km 分块 66/72） | B | CHECK3_RESULTS；CONCLUSIONS C3a |
| BA 损失中类别集合的份额 | 11–15% | B | 同上 |
| 域残差 AE−S2 | OA +0.00（p=0.58），BA +0.73（p=0.70） | B | CHECK3_RESULTS |
| 只用目标池 − 只用外国 | 约 13–14 pp BA，22/22；MLP 15，LR 19–23（MLP、LR 为 5 国试点） | B | MODEL_DIAGNOSIS §3.4 |
| PK50 恢复率 | OA 85–97%；EM 31/58%；1 区域约 80%，3 区域 91–94%（区域 = 20 簇之一；50 簇时 78–80% / 89–92%，C） | B（空间分布效应为 C） | CHECK3_RESULTS |
| 先验修正对 BA | −12.21 / −12.77，11/11 | B | CONCLUSIONS C3c |
| SRS k0→50 | 中位数 +0.31；13/22 条曲线不单调 | B | G_RESULTS §5 |
| SRS k0→200 | +2.41（S2）/ +3.10（AE）；去掉 RO、PT 后 +0.66 / +0.71 | B | CONCLUSIONS #6 |
| 几何法 − SRS | AE −0.156 [−0.73, +0.42]；S2 −0.102 [−0.65, +0.44] | B− | STRATEGY ① |
| predstrat − SRS | +1.04 [+0.17, +1.90]，10/11；收缩后 +0.41 | C | STRATEGY ② |
| F3b 混合头（α=0.4）− 重训 | +0.94 [+0.69, +1.19]，11/11，OA −0.05 | C | PIVOT §2.3；frontier2_report |
| 只用类均衡头 − 重训（rev4 D1，λ=0.1） | BA +2.95 至 +3.42，OA −2.92 至 −7.07（k=50–400） | C | `V/storyline/decision_curve/decision_curve_summary.txt` |
| 缓冲上界 | OA +6.66 / +6.38；BA +10.45 / +9.28 | C | `oracle_G_buffered.log` |
| Δ_feat（LGB） | k0 +2.42；k200 +3.11；k400 +2.47；0→400 变化 +0.05 [−0.92, +1.02] | C | `feat_by_k.csv`；REPOSITION H6 |
| D1、D2、rev2、rev3-B1/B2/B3 | ⟦待定⟧ | — | — |

---

## 附录 B：本计划新算或核对的内容

**1. 只读类别计数**（`V/storyline/paper_plan/class_counts.py` → `class_counts.log`、`dataset_v2_class_counts.csv`、`G_test_class_counts.csv`；单线程、不训练）
- dataset_v2 共 611,519 个地块。
- 缺类：LT 与 LV 的马铃薯、草莓为 0；RO 只有 grassland、other_crops、tree_wood_forest。
- 各单元规模：测试集 11,890–30,251；缓冲候选集 27,673–65,357；外国池均为 50,000。
- **D2 的 4 类子集，测试集样本数**（s0 / s1）：

| 国家 | 草地 | 玉米 | 马铃薯 | 冬季谷物 |
|---|---|---|---|---|
| AT | 8119 / 8017 | 1000 / 1025 | 172 / 151 | 1412 / 1505 |
| BE | 16893 / 14784 | 4754 / 3970 | 1255 / 1012 | 3319 / 3005 |
| DE | 13979 / 13826 | 5137 / 5025 | 614 / 532 | 5576 / 5863 |
| EE | 7094 / 7337 | 68 / 74 | 231 / 252 | 1176 / 1375 |
| LT | 5914 / 6997 | 94 / 114 | **0 / 0** | 2791 / 2841 |
| LV | 7150 / 7143 | 60 / 71 | **0 / 0** | 1396 / 1621 |
| PT | 4503 / 4075 | 1524 / 1030 | 183 / 122 | 42 / 40 |
| RO | 1594 / 1797 | **0 / 0** | **0 / 0** | **0 / 0** |
| SE | 10244 / 10252 | 60 / 73 | 62 / 75 | 885 / 916 |
| SI | 8945 / 8962 | 1095 / 1763 | 277 / 308 | 787 / 1352 |
| SK | 7684 / 7418 | 1101 / 1042 | 145 / 172 | 1753 / 1756 |

- 解读：
  - 除缺类外，各类都 ≥30，但波罗的海三国与 SE 的玉米、SE 的马铃薯、PT 的冬季谷物只有 40–114 个。
  - REPOSITION 中"几百个""PT winter_cereals 156"指的是**全国总数**（例如 PT 全国 156），不是测试集数。D2 报告时以测试集数为准。
  - RO 在 4 类子集中只剩 1 类，指标退化；LT、LV 实为 3 类。

**2. Cheng et al. 2026 原文核对**
- PDF 位置：`/Users/perng_mini/Library/Mobile Documents/com~apple~CloudDocs/Research_icloud/1-s2.0-S0034425726004645-main.pdf`（不在项目 `reference/` 中，建议复制过去）。
- 已核对：
  - §4.3 Practical implications 在第 15–16 页，含本文引用的建议句；
  - 表 3：本地 0.94 / 0.96，迁移 0.71 / 0.75，OA≈BA；
  - 摘要：ALR 0.14 / 0.23 每 1000 °C·d，交叉点约 1400–1500 °C·d（约 9–10°）；
  - §3.1.2：只用 S1 / S2 / Landsat 的 SITS 跨区 OA 为 0.72 / 0.67 / 0.62；
  - 表 5：TESSERA RF 迁移 0.83、AEF RF 0.70，为 2017–2018 两年、162 个跨区迁移的子集；
  - §3.4.3：SITS-LDA5D 的平均 OA 为 0.83（最高），ALR"与 AEF 相当"；
  - §2.1：欧洲 10 个 NUTS 区、7 国，9 个区用于训练；每区-年-类最多 5000 点；区内按类分层 70/10/20；
  - SITS 105 维（S2 7 个 40 天、Landsat 6 个 48 天、S1 14 个 20 天合成）；AEF 为 10 m 像元 64 维。
- **正文没有、已从图中核实**：
  - Fig. 11b 的 R²=0.18（图例"slope=0.111, R²=0.18***"，第 14 页）；
  - SITS-LDA5D 的 ALR 0.14（Fig. 12b 图例"slope=−0.14, R²=0.37***"，第 15 页；正文只写"与 AEF 相当"）。

**3. 缓冲格网上界的拟合集**（读 `oracle_G_buffered.py`）
- 拟合用 `S['cand']`，即全部缓冲候选集，每单元 27,673–65,357 个。
- 用 `fit_bcts(Pc, yC, np.ones(nc))`，即自然权重、无岭罚；BA 版在拟合后按候选池先验重加权到均匀先验。
- 与 rev4 D1 的全池（3 万候选池，类均衡权重，λ=0.1）不同，见 §4.9 与 §8 E1。
