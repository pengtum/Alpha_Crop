# 预测模型盘点与多学习器扩展方案（2026-09-27）

# 答复：目前用了几个预测模型，论文里能否多放几个

## 1. 目前做了几个

**两篇论文的正文和所有预注册主实验（G、阶段 A、检验 1–3）都只用了 LightGBM 一种模型。** 算上已作废的试点，实际跑过的学习器共 9 种。另外 8 种只出现在原论文里没写进手稿的稳健性检验、3 月的 Flanders 试点，或 9 月的事后诊断里。

| 模型 | 所在层级 | 覆盖情况 |
|---|---|---|
| LightGBM | 两篇正文，以及全部预注册主实验 | G 里的"两种分类器"是同一个 LightGBM：一个输入 AE 的 128 列，一个输入 S2 的 168 列。层级 3 的 AE⊕S2 融合分类器在 rev1 里取消了，没有运行 |
| XGBoost | 原论文稳健性检验，没进手稿 | 用的是旧的 18 国数据。exp2b 共 306 对，XGBoost 比 LightGBM 高约 1 pp。exp1 只跑了 AE，1 次 |
| 多项 LR | 原论文 exp10（没进手稿），加上事后诊断 | exp10 用的是 11 国 2021 版数据，共 48 次拟合，全部收敛；诊断覆盖 1–4 个单元 |
| sklearn MLP | 事后诊断 | 2 个单元 |
| RF | 事后诊断（未调参），加上 3 月 BE 试点 | part1 只跑了 s0r0 |
| ExtraTrees、kNN | 事后诊断 | part1 只跑了 s0r0 |
| HistGradientBoosting | 事后诊断（只有 AT），加上 BE 试点 | — |
| nearest centroid | BE 试点（已作废） | — |
| TabPFN | 只有代码 | 从未运行 |

- BBSE、PEM、bcts、R2V 和"冻结外国模型 + 决策头"都是在 LightGBM 输出上做的调整或变体，不算新的学习器。
- 所有事后诊断都做在 G 的 s0/s1 测试集上，而这两个测试集已经被多次使用。

## 2. 能否在论文里多放几个：可以

**推荐清单：** 正文放 LightGBM 加 3 个学习器，kNN 放补充材料。

| 学习器 | 代表什么 | 理由 | 位置 |
|---|---|---|---|
| LightGBM | 主学习器 | 3,782 个概率文件都已缓存，不需要重训 | 正文 |
| **多项 LR** | 线性探针。嵌入评估的惯例做法，AlphaEarth 原文、Corley、Galileo、PANGAEA、Presto 都这样做 | 它和树模型的归纳偏置差别最大。k 从 0 增到 200，它的增益比 LightGBM 多约 3 pp，所以对"两者没有差别"是最严格的检验 | 正文，第一优先 |
| **MLP 集成（3 个种子）** | 深度方法 / 非线性头。Ball、Koukos、Lisaius、Tong、Hou 都用了 | Ball 指出 GFM 嵌入需要非线性头。诊断中它相对 LightGBM(τ=0.5) 的 Δba 为 +1.69 / +1.76，OA 持平或略高 | 正文 |
| **RF** | 遥感惯例。Ma、Mughees、Lisaius、Koukos、Tong、BreizhCrops、Sheykhmousa 都用了 | 主要作用是让读者熟悉。它和 LightGBM 同属树集成，新信息较少。诊断中未调参的 RF 比 LightGBM 低 3.33 pp | 正文 |
| kNN | 嵌入几何探针 | 不支持样本权重，只能按权重 w 加权投票来近似；诊断中 ba 低 5.1–5.4 pp。不能当作稳健性检验 | 补充材料 |
| HGB | 第二种 GBDT 实现 | 和 LightGBM 冗余：诊断中两者差 ≤ 约 1 pp，但只有 AT 一国的存档 | 可选 |
| ExtraTrees | — | 诊断中比 LightGBM 低 10.63 pp | 可选，优先级最低 |

**不推荐，或需要你批准：**
- **MLP 早停版**：它的验证集是从整个训练集里随机抽 10%，k=200 时会拿走约 20 个目标样本的标签。
- **LinearSVC / SGD**：在 S2 上不收敛，没有真正的概率输出，而且和 LR 同属线性族。
- **XGBoost、CatBoost、TabICL、TabPFN**：都需要你批准安装，也都没实测过。
  - XGBoost 和 CatBoost 能完全复现现有的权重设置，但和 LightGBM 同属 GBDT，冗余。
  - TabICL 不支持样本权重，数据规模也超出了它的推荐范围。
- **TempCNN、Transformer 等深度时序模型：不适合现有特征。**
  - S2 的 168 列 = 84 个统计量 × 邻域 {mean, stdDev}，只有 5 个变量有 4 个时间步；AE 没有时间维。
  - 要用这类模型，得重新从 GEE 提取约 64.5 万个地块的密集序列，还需要 GPU 和 torch。
  - 更关键的是，那样表征和学习器的效应会混在一起。建议写进局限性段，引用 Tong & Wang。

## 3. 实施方案（只重训，不重新选样）

**数据准备**
- 层级 1 的 1,232 个选样文件都能找到，清单在 `tier1_sel_index.csv`。LightGBM 不需要重训。
- 训练集和样本权重与 G 保持一致：外国样本 5 万 + k 个目标样本。
  - 权重：外国样本 w = cw[y]，目标样本 w = 50·cw[y]，cw 只按外国样本的类别频率计算。
  - LR、MLP、RF、ET、HGB 直接用原生的 sample_weight（LR 先把 w 除以均值），RF 设 class_weight=None。
  - kNN 让每个近邻按自己的 w 加权投票，票数归一化后当作概率。

**超参草稿**（`learners_rev2_draft.json`）
- LR：C 按规则选，在外国池上按 `G.fold_groups` 做 5 组留国家组验证，从 {1e-3…10} 中选。
- MLP：结构 (256,128)，固定训练 20 轮、不早停，3 个种子的概率取平均。
- RF：500 棵树，max_features=sqrt，min_samples_leaf=1。
- kNN：K=10，K=30 作次要结果。

**覆盖范围：推荐方案 A 作正文。**
- 方案 A 包括三部分，每种特征 902 次拟合，每个学习器 1,804 次：
  - E1：704 个选样，即 4 种几何方法 × AE/S2 两种指导 × k∈{50, 200} × 11 国 × 4 个单元；
  - SRS：176 个选样；
  - k=0：22 次拟合。
- E1 里已经包含 k=50，正好补上 LR 判断等价时缺的 k=50。
- 方案 B 里的 PRED、HYB（352 个选样）建议放附录，并写明："这些选样由 LightGBM 指导，这里只是换别的学习器重训"。
- 方案 C（只做 SRS 和 k=0）信息量小，而且很多格子的结果已经看过，只适合当作退路。

**实测单次耗时**（4 线程，ae / s2，单位秒）
- 共 114 次计时运行，全部正常，没有读取测试标签。

| 学习器 | 耗时 (s) |
|---|---|
| LightGBM（参照） | 18.4–19.5 / 21.3–22.3 |
| LR | 1.4–1.7 / 2.6–3.3 |
| MLP 集成 | 8.8–8.9 / 9.3 |
| RF | 21.8–28.3 / 24.5–31.0 |
| kNN | 0.5–1.2 / 0.6–1.4 |
| HGB | 19.8–20.6 / 25.0–26.9 |
| ET（min_samples_leaf=5） | 6.3–6.4 / 7.0–7.2 |

**总墙钟时间**（10 核机器用 8 核）

| 方案 | 4 个核心学习器 | 加 HGB 和 ET |
|---|---|---|
| A | **7.8–8.8 h（一个晚上）** | 13.7–16.1 h |
| B | 10.9–12.2 h | 19.1–22.4 h |
| C | 1.7–1.9 h | 3.0–3.5 h |

- 时间大头是 RF，方案 A 下约 6.9 h。
- 并行方式：LR、MLP、kNN、HGB 用单线程多进程；RF、ET 用 2 进程 × 4 线程。"8×1"那一列是从 4 路并发线性外推的，没有实测。
- 存储：方案 A 每个学习器约 1.1 GB，磁盘还剩 34 GB。

**指标**
- 新学习器只报 argmax 指标：ba_ge30、OA、E_far。kNN 和 RF 的概率不适合做先验调整。
- LR 的 ba 提升主要来自工作点移动：part1 中 Δba +4.09、ΔOA −8.60；用 τ=0.5 对齐后，优势约为 0。所以 LR 要同时报 ba 和 OA，或者所有学习器统一做 τ 调整。

## 4. 定位与预登记

**定位**
- G 的 s0/s1 测试集已经多次使用，所以这些结果只能写成"预先指定的学习器稳健性分析（相对 G 原方案为事后追加）"，不能写成确认性结果。

**需要逐条披露已经看过结果的格子**

| 学习器 | 已看过结果的格子 |
|---|---|
| LR | E1 × k=200 的全部 704 个格子；SRS k=0/200 的 s0r0、s1r0 |
| MLP 早停版 | E1 × k=200 的 s0r0 |
| MLP 集成 | SRS k=0/200 的 s0r0、s1r0 |
| kNN | SRS k=0/200 的 s0r0 |
| RF、ET | SRS k=200 的 s0r0；AT 的 k=0 |
| HGB | 只有 AT |

**看结果之前要冻结（由你执行）**
1. 写 `ANALYSIS_PLAN_guidance_rev2.md`，冻结以下内容：
   - 学习器清单和超参数，以及"只在外国池上按留国家组调参"的规则；
   - 种子取 1000·split+rep，MLP 的三个种子在此基础上加 0、1、2；
   - 库版本；
   - 选样清单 `tier1_sel_index.csv`，其 sha256[:16] = 66252ed9025cb7a4；
   - 估计量：每个学习器各自的 θ_E1，沿用 ±0.3 的 TOST 和 90%/95% 区间；
   - 稳健的判定规则；
   - "所有学习器都报告"，剔除条件只限事先写明的技术失败；
   - 上面那张已看过的格子清单。
2. 把方案和运行代码的哈希写进 `plan_hashes.txt`。运行脚本启动时校验哈希，做法同 `guidance_G.main`。
3. 冻结后如有任何改动，另写 rev3。

**如果以后要做确认实验**
- 必须在从未当过测试集的格子上重新划分，要重算 stage0（每个国家 × 划分约 680 s，2 线程）和选样（层级 1 的全部选样累计 0.31 h）。
- 到时把学习器作为一个因子并入设计，按学习器分层估计 θ_E1，并检验学习器 × 指导的交互。

## 5. 对论文的价值

**它回应的审稿质疑**："只用了 LightGBM，结论是否依赖学习器？"
- 同类评估论文一般比较 2–5 个学习器：Ma 2 个，Brown 3 个，Ball 4 个，Hou 5 个，Lisaius 5 个。
- 我们的结论讲的是表征和选样策略的优劣。Lowell 和 Lu 已经表明，这类结论会随模型变化，所以不能照搬业务制图论文只用一个学习器的做法。

**可能出现的结果和对应写法**

| 情形 | 写法 |
|---|---|
| 每个学习器的 θ_E1 的 90% 区间都在 ±0.3 以内 | 写成"AE 与 S2 作为选样指导等价，且不依赖评估用的学习器"，主结论加强 |
| LR 下偏向 AE，且区间越过 +0.3。现有 k=200 证据是 +0.12 [−0.15, +0.40]，与 LightGBM 的差中差为 +0.24（p=0.067），所以有这种可能 | 如实报告为学习器之间的异质性："等价在树模型下成立，在线性探针下 AE 指导略优"。同时给出 OA 或 τ 调整后的结果，把工作点效应区分开 |
| 各学习器方向不一致 | 结论降级为"依学习器而定"，不挑最好的那个 |
| predstrat 在 LR、MLP 下仍然最好 | 支持"收益来自类别构成、与学习器无关"的机制解释，证据很强 |
| margin 等不确定性方法在 LR 下变差 | 这符合 Lu et al. 的预期：查询模型和任务模型不一致时，不确定性采样会退化。写成"LightGBM 选出的样本迁移到其他学习器时效果有限"，并注明这不等于"每个学习器用自己的不确定性去选样" |
| MLP 的绝对精度高于 LightGBM | 如实报告，但不在事后更换主学习器；说明本文结论的对象是表征和选样，不是学习器 |
| RF 低于 LightGBM | 注明 RF 未调参 |

**局限性段**
- 写明结论限定在"冻结特征 + 表格学习器"的设定内，端到端时序模型不在范围内。
- 学习器之间的差异通常小于协议之间的差异：BreizhCrops 中 RF 与最好的深度模型只差约 2–4 pp（OA）；Long et al. 发现 RF 在向相似域迁移时有竞争力。

## 6. 需要你决定的事项

1. **是否破例做新实验。** 方案 A 的 4 个核心学习器约 8–9 h，一个晚上能跑完。如果不做，RSE 那篇只能在补充材料里描述已有的诊断结果（1–2 个单元，测试集已被复用），证据较弱。
2. **学习器清单。** 我建议正文放 LR、MLP 集成、RF，补充材料放 kNN。HGB 和 ET 是否加入？加入后方案 A 会多出约 6–7 h。
3. **覆盖范围。** 正文用方案 A。B 里的 PRED、HYB 是否进附录？4 个核心学习器会多 3.1–3.4 h。
4. **LR 的 C 怎么定。** 按留国家组规则选，还是固定 C=1（诊断里用的是 C=1）？
5. **稳健的判定规则要写死**，例如"每个学习器 θ_E1 的 90% 区间都在 ±0.3 以内"。
6. **LR 的工作点怎么处理。** 所有学习器统一做 τ 先验调整，还是同时报 ba 和 OA？
7. **是否批准安装 xgboost、catboost。** 我的建议是不需要，它们和 LightGBM 冗余。TabPFN、TabICL 不建议用。
8. **以后是否做确认实验**：在新划分上做，把学习器作为一个因子。
9. **原论文（Paper_Alpha_Earth）不用新算就能补的内容：**
   - exp10 的 LR 探针用的是 11 国 2021 版数据，可以直接放进补充材料，但需要补记当时用的 solver。
   - XGBoost 的结果基于旧的 18 国数据，要么不用，要么装 xgboost 后重跑。
   - 另外，手稿写 S2 特征是 168 维，但列出的清单加起来只有 84 个，漏写了邻域 mean/stdDev 这一步的 ×2，建议顺手改正。

草稿和计时文件都在这几个目录（没有改动任何项目文件）：
- `/Users/perng_mini/AlphaCrops_local/verify/multilearner/inventory/model_inventory.csv`
- `/Users/perng_mini/AlphaCrops_local/verify/multilearner/feasibility_compute/`：其中 `learners_rev2_draft.json` 是学习器规格草稿，`tier1_sel_index.csv` 是选样清单，`estimate_plans.csv` 是三种方案的估算
- `/Users/perng_mini/AlphaCrops_local/verify/multilearner/reviewer_lit/`

---

# 附录 A：模型盘点

**模型盘点（只统计有结果文件为证的运行）**

两篇论文的正文和所有预注册实验都只用了 LightGBM 一种模型。另外 7 种学习器确实跑过：XGBoost、多项 LR、sklearn MLP、RF、ExtraTrees、kNN、HistGradientBoosting。它们只出现在原论文未写进手稿的稳健性检验、3 月的 Flanders 试点，或 9 月的事后诊断里。TabPFN 只有代码，从未运行。

路径缩写：R = 项目根目录，L = `/Users/perng_mini/AlphaCrops_local`。

## 一、原论文（R/Manuscript/Paper_Alpha_Earth/0001_draft_v1/01_main.tex）

**1. LightGBM：跑过，是唯一写进正文的分类器**
- 参数（config.LGBM_PARAMS）：500 棵，学习率 0.05，63 叶，balanced 类别权重，50 轮早停。
- 特征：AE、S2、Fusion 三种。
- 覆盖：exp0/1/2/2b/3/4/5/6/8/9 全部用它，包括国家探针和 168 次消融拟合。
- 手稿第 222 行写明只用 LightGBM，第 528 行把"单一分类头"列为局限。
- 证据等级：原论文主实验，没有预注册。

**2. XGBoost：跑过（4 月 8–9 日），但没写进手稿（01_main.tex 中查不到 XGBoost）**
- 参数（config.XGB_PARAMS）：500 棵，深度 6，学习率 0.05，50 轮早停。
- 数据是旧的 18 国、未按年份对齐版本，不是现行的 11 国 2021 版。
- exp1 Pooled：只跑了 AE，1 次重复，n=1,089,914。BA 73.19；同数据上 LightGBM 为 70.15±0.26（5 次重复）。
  - 结果：R/results/experiments/exp1_id_baseline/tier_b_nocoords_xgb.json
- exp2b 两两迁移：18 国，306 个有序对，AE/S2/Fusion 三种都跑了。
  - 我在同一批国家对上配对重算的 BA（XGB 对 LGB）：
    - AE：30.13 对 29.05，XGB 胜 230/306
    - S2：30.47 对 29.42，胜 251/306
    - Fusion：32.12 对 31.14，胜 227/306
  - Wilcoxon 检验 p<1e-18。只取 11 国子集（110 对）时，XGB 高 0.8–1.0 pp。
  - 结果：R/results/experiments/exp2b_country_transfer/transfer_{ae,s2,fusion}_nocoords_xgb_tier_b.csv，另有 delta 图、胜负地图
- LOCO 没有用 XGB 跑过。现在的 venv 里也没装 xgboost。
- 证据等级：原论文稳健性检验，未写入手稿，数据版本已过期。

**3. 多项 LR 线性探针（exp10）：跑过，未写入手稿**
- 模型：LogisticRegression，balanced 类别权重，先做 StandardScaler。
- 覆盖：11 国 2021 版，AE/S2/Fusion × (Pooled 5 次 + LOCO 11 国) = 48 次拟合，全部收敛（我已核对）。
- BA（探针 / 同条件 LightGBM）：

| 协议 | AE | S2 | Fusion |
|---|---|---|---|
| Pooled | 69.75 / 68.48 | 68.35 / 67.67 | 74.00 / 71.45 |
| LOCO 逐国平均 | 44.58 / 42.71 | 45.08 / 43.17 | 48.08 / 45.20 |

- 结论：跨境的 AE−S2 差异在两种分类头 × 4 种样本数阈值下都不显著，共 8 个检验（SESSION_STATE）。
- 缺陷：用的 solver 没有记录。
- 结果：R/results/experiments/exp10_linear_probe/（json 与 48 个 parquet）

**4. RF：只在 3 月的 Flanders 单国试点里跑过，已被取代**
- 设置：只有 BE，3,235 个地块，18 个 Tier-A 类，按纬度南北切分当作 OOD。
- 结果：RF 分布内 BA 47.46、OOD 32.98；同设置 LightGBM 为 46.70 / 31.00。
  - 文件：R/outputs/all_data_experiments_20260305_1520/summary_metrics.csv，以及 R/results/rf_rerun_20260305_1025/
- 同一批试点里还有：
  - HistGradientBoosting(200)：结果文件名叫 `summary_alphaearth_lgbm.json`，但里面实际是 HGB；
  - nearest centroid（R/results/quick_pilot/）。
- 11 国实验里没用过 RF：`get_trainer` 只提供 lgbm/xgb/tabpfn 三个选项。

**5. TabPFN：从未运行，只有代码**
- 代码：`experiment_utils.train_tabpfn`，另有 8 个 run_*.py 脚本在 `--classifier` 选项里列了它。
- 我在 results/、RESULTS_SUMMARY/、outputs/ 和手稿里检索 "tabpfn"，没有任何命中，也没有输出文件。
- 本地 venv 没装 tabpfn。
- PROGRESS_REVIEW 记录 settings.local.json 里有 TABPFN_TOKEN，说明可能尝试过配置，但没有留下任何结果。

## 二、标签扩充（RSE 那篇，label_expansion/）

**6. LightGBM：预注册主实验的唯一学习器**
- G：300 轮固定，63 叶，学习率 0.05，max_bin 63，类别权重按外国频率，r=50。
  - 覆盖：11 国 × s0/s1 × r0/r1，层级 1–4。
  - L/results/guidance_G/rows_t*.csv 共 3,872 行，clf 只有两种取值：ae 1,936 行、s2 1,936 行。
- 阶段 A：只用 S2 分类器（`FEAT='s2'`），PT/BE/SI × s0–s2，r 网格 {1, 5, 20, 50}，513 行。
- 检验 1：LightGBM 500 轮加早停，AE/S2/Fusion，11×11 国（L/results/pairwise_v2.csv）。
- 检验 2 与 headroom：LightGBM，4 国（pilot_selection.csv、headroom.csv）。
- 检验 3：LightGBM，11 国，37 个臂（check3_arms.csv）。其中 BBSE、PEM、bcts 是加在 LightGBM 输出上的标签偏移或校准调整，不是另一种模型。
- 不算作物分类器的：`domain_distance.py` 里的 LR 只作 C2ST 域判别器。

**G 里"两种分类器"的确切含义**
- 方案原文（ANALYSIS_PLAN_guidance.md 第 104 行）："LightGBM，分 S2 特征与 AE 特征两种，同一选样、同一种子各训练一次"。
- 也就是同一个 LightGBM、同样的超参，只是输入分别为 AE 128 列和 S2 168 列。
- 层级 3 的 AE⊕S2 融合分类器（第 105 行）在 rev1 第 28 行被明确取消。rows 文件里没有任何 fus 行；kcenter_foreign、k=1000 和通路交叉也都没跑。
- 另有两处 LightGBM 只用来给选样提供指导，不参与评估：
  - stage0 的外国模型（袋外概率 + 温度缩放）；
  - ORACLE 在候选池内交叉拟合的 AE⊕S2 LightGBM（几何部分另用 LDA）。

**7. 事后诊断（L/verify/model_diag/）**
主要单元是 s0r0，都在已被多次使用的 s0/s1 测试集上做。

- **part1**（altmodels/part1_rows.csv）：11 国 × {ae, s2} × k∈{0, 200}（k=200 用 G 的 SRS 选样）。
  - 模型：LR（C=1）、MLP (256,128) 和 (512,256,128)（早停）、kNN10/30、IW-LR、只用目标样本的 LGB/LR/RF/kNN。
  - 覆盖不全的：RF(150)、ET(300)、IW-LGB 在 k=200 覆盖 11 国，k=0 只有 AT；HGB 只有 AT。
  - IW 模型的域判别器是二分类 LightGBM。
  - k=200 相对 LightGBM 的 Δba（ae/s2 平均）：

| 模型 | Δba | ΔOA |
|---|---|---|
| LR | +4.09 | −8.60 |
| MLP 早停 | +4.0 | −6.0 |
| RF | −3.33 | – |
| ET | −10.63 | – |
| kNN | −5.1 至 −5.4 | – |
| IW-LGB | −1.60 | – |

  - 用不看标签的先验调整（τ=0.5）把 LightGBM 的工作点对齐后，LR 和早停 MLP 的优势约为 0。
- **did_lr**（verify_altmodels/did_lr_rows.csv）：用 LR 在 G 的 E1 选样上重训，k=200，4 个单元 × 4 种方法。
  - θ(AE−S2)：LR +0.12 [−0.15, +0.40]，LightGBM −0.12；差中差 +0.24，p=0.067。结论没有反转。
- **mlp_fair**（mlp_fair_rows_s0r0 / s1r0.csv）：3 种子 MLP 集成（不早停，20 轮）。
  - 相对 LightGBM(τ=0.5) 的 Δba 为 +1.69 / +1.76，OA 持平或略高。只覆盖 2 个单元、k=200。
- **fit/**：LightGBM 的容量变体（15 / 63 / 255 叶，正则，最多 1000 轮）以及 3 种子集成。

**8. 事后探索（L/verify/pivot_eval/）**
都在 s0/s1 上做，s2/s3 上的确认没跑。
- R2V：AE、S2 两个 LightGBM 重训模型的对数概率等权平均。
- 冻结外国模型 + 类均衡决策头：bias、bcts、mat、feat 几种形式，都是加权交叉熵加向外国模型收缩的 L2 正则；另有 gbft（4 叶、60 轮的 LightGBM 头）和 XV-Cal 两视图校准层。
- 头 + R2V 相对 R2V：BA +2.51，OA +0.06（review_exec/ctrl_viewsplit_report.txt）。
- TabPFN、TabICL、RealMLP 等只做了文献调研，没有实测。

## 汇总表

| 模型 | 预注册主实验 / 手稿正文 | 原论文稳健性（未进手稿） | 事后诊断 / 探索 | 已作废试点 | 只有代码 |
|---|---|---|---|---|---|
| LightGBM | ✔（两篇全部） | – | ✔（容量、种子、R2V、gbft 头） | ✔ | – |
| XGBoost | – | ✔（exp1 只有 AE；exp2b 18 国） | – | – | – |
| 多项 LR | – | ✔（exp10，48 次拟合） | ✔（part1、did_lr、mlp_fair） | – | – |
| MLP（sklearn） | – | – | ✔（2 个单元） | – | – |
| RF | – | – | ✔（part1，未调参） | ✔（BE） | train_rf.py |
| ExtraTrees | – | – | ✔（part1） | – | – |
| kNN | – | – | ✔（part1） | – | – |
| HistGradientBoosting | – | – | ✔（只有 AT） | ✔（BE） | – |
| nearest centroid | – | – | – | ✔（BE） | – |
| TabPFN | – | – | – | – | ✔（从未运行） |
| G 的 AE⊕S2 融合 LightGBM | 方案里有，rev1 取消 | – | – | – | ✔ |

我还在 L/verify/multilearner/inventory/model_inventory.csv 存了一份这张表的草稿（CSV）。

**一句话答案：** 算上已作废的 Flanders 试点里的 nearest centroid，实际跑过的学习器共 9 种；但在预注册主实验和两篇手稿正文里只有 LightGBM 一种。XGBoost（旧 18 国数据）和 LR 线性探针在原论文里做过稳健性检验但没写进手稿；LR、MLP、RF、ET、kNN、HGB 只在 1–2 个单元上做过事后诊断，这些单元的测试集已被多次使用；TabPFN 和 G 的融合分类器只有代码或方案，从未运行。

---

# 附录 B：可行性与算力

**结论**：只用已安装的库、不重新选样，论文里可以再加 4 个学习器：多项 LR、MLP 3 种子集成、RandomForest、kNN。另有 2 个可选：HistGradientBoosting 和 ExtraTrees。按主问题最小集（方案 A）算，4 个核心学习器约 8–9 小时，一个晚上能跑完；加上 2 个可选学习器约 14–16 小时。LightGBM 的概率已全部缓存，不需要重训。但 G 的 s0/s1 测试集已被反复使用，这些新结果只能写成"预先指定的学习器稳健性分析"，不能写成确认性结果。

## 1. 候选学习器与权重的实现方式
训练集统一为外国 5 万 + k 个目标样本。权重 w = cw[y]（外国）和 50·cw[y]（目标），cw 只按外国频率计算。

| 学习器 | 预设参数 | 权重怎么实现 | 建议 |
|---|---|---|---|
| LR | 标准化；lbfgs 多项、L2；tol 1e-4；max_iter 1000。C 用规则选：在外国池上按 `G.fold_groups` 的 5 组留国家组，从 {1e-3…10} 中选 | 原生 sample_weight，w/mean(w) | 纳入 |
| MLP 集成（mlp_f3） | 标准化；(256,128)；α 1e-4；batch 256；lr 1e-3；固定 20 轮、不早停；3 个种子概率平均 | 原生 sample_weight（sklearn 1.9） | 纳入 |
| MLP 早停版 | early_stopping，10% 验证 | 验证集是整个训练集的随机 10%（已查 sklearn 源码）。k=200 时会拿走约 20 个目标标签，且停止与否主要由外国样本决定 | 不作主模型 |
| RF | 500 棵，sqrt，min_samples_leaf=1，bootstrap | 原生 sample_weight，class_weight=None | 纳入 |
| ET | 500 棵，sqrt，min_samples_leaf=5，不 bootstrap | 原生 | 可选 |
| HGB | lr 0.05，300 轮，63 叶，min_samples_leaf 20，63 bins，不早停 | 原生。没有行采样和按树的特征采样，与 LightGBM 不完全相同 | 可选，作第二种 GBDT 实现 |
| kNN | 标准化，欧氏距离，brute，K=10（K=30 为次要，同一次搜索） | 不支持权重；改为每个近邻按其 w 加权投票，票数归一化作概率 | 纳入 |
| LinearSVC / SGD | C=1，dual=False；SGD 用 modified_huber | 原生 | 不纳入：S2 上不收敛，没有真概率，与 LR 同属线性族 |
| XGBoost / CatBoost（需批准安装） | – | 原生权重，可以完全复现 r=50 和类别权重 | 未实测，装好后先做 12 次计时试点 |
| TabICL（需批准安装） | – | 不支持权重；168 列、5 万行上下文都超出推荐规模，只能抽样上下文并降维，与现设计不等价 | 未实测，只宜作外部基准 |

## 2. 实测（BE 为最大测试集，3.0 万；LV 为最小，1.2 万）
- 范围：ae/s2 × k=0/200，G 的 s0r0 SRS 选样，拟合加预测。
- 共 114 次运行，全部正常，概率都合法。
- 计时时只读了测试集特征用于预测，没有读测试标签，也没有算任何精度。
- 测量时系统负载 2–5（系统后台进程），没有其他 Python 任务。

| 学习器 | 4 线程单次 (s)，ae / s2 | 1 线程、4 路并发 (s)，BE k200 | 峰值内存（含约 0.6 GB 共享数据） |
|---|---|---|---|
| LightGBM 参照 | 18.4–19.5 / 21.3–22.3 | 44 / 54 | ≤0.92 GB |
| LR | 1.4–1.7 / 2.6–3.3（307–625 次迭代，都收敛了） | 1.7 / 3.0 | 0.72 GB |
| LR 选 C（每个国家×划分×特征一次，25 次内层拟合） | 16–17 / 31–35 | – | 0.74 GB |
| MLP 早停版 | 1.6–2.8 / 2.7–4.4（12–31 轮就停了） | 2.5 / 4.5 | 0.75 GB |
| MLP 集成 | 8.8–8.9 / 9.3 | 9.7 / 10.2（多给线程几乎不加速） | 0.74 GB |
| RF（min_samples_leaf=1） | 21.8–28.3 / 24.5–31.0 | 89 / 100 | 1.5–1.7 GB，模型约 0.8 GB |
| RF（min_samples_leaf=5） | 19.3–23.8 / 21.8–26.3 | – | 0.9–1.2 GB |
| ET（min_samples_leaf=1） | 8.5–8.8 / 9.7–9.9 | 37 / 40 | 2.7–3.5 GB，模型 2.6–2.9 GB |
| ET（min_samples_leaf=5） | 6.3–6.4 / 7.0–7.2 | – | 1.7–1.9 GB |
| HGB | 19.8–20.6 / 25.0–26.9 | 58 / 83 | ≤0.8 GB |
| kNN | 0.5–1.2 / 0.6–1.4 | 4.0 / 4.3 | 0.74 GB |
| LinearSVC | 10.1–10.7 / 110–147（BE 的 S2 跑满 1000 次迭代仍未收敛） | 14 / 164 | 0.92 GB |
| SGD | 0.7–1.1（每次都跑满 50 轮，未收敛） | 3.0 / 4.8 | 0.72 GB |

k=0 与 k=200 的耗时几乎相同。LR、MLP、kNN、HGB 用单线程多进程并行更划算；RF、ET 用"2 进程 × 4 线程"和单线程多进程效率相同，但单线程多进程的内存不够。

## 3. 三种方案的算力估算
每个学习器都算 ae、s2 两种特征。k=0 每个（国家, 划分）只拟合一次，两次重复共用。

| 方案 | 构成 | 每个学习器的拟合次数 |
|---|---|---|
| A | E1 704 + SRS 176 + k0 22，每种特征 902 | 1,804 |
| B | 层级 1 全部 1,232 + k0 22 | 2,508 |
| C | SRS 176 + k0 22 | 396 |

墙钟时间（10 核机器用 8 核）：

| 学习器 | A | B | C |
|---|---|---|---|
| LR（8×1 外推 / 4×1 实测） | 0.16 / 0.32 h | 0.22 / 0.45 h | 0.04 / 0.07 h |
| MLP 集成 | 0.62 / 1.25 h | 0.87 / 1.73 h | 0.14 / 0.27 h |
| kNN | 0.14 / 0.27 h | 0.19 / 0.38 h | 0.03 / 0.06 h |
| RF（2×4） | 6.9 h | 9.6 h | 1.5 h |
| HGB（8×1 / 2×4） | 4.2 / 5.7 h | 5.9 / 7.9 h | 0.9 / 1.2 h |
| ET（min_samples_leaf=5，2×4） | 1.7 h | 2.3 h | 0.4 h |
| **4 个核心合计** | **7.8–8.8 h** | **10.9–12.2 h** | **1.7–1.9 h** |
| **加 HGB 和 ET** | **13.7–16.1 h** | **19.1–22.4 h** | **3.0–3.5 h** |

- **8×1 未实测**：这一列是从 4 路并发线性外推的（线程上限 4）。
- **内存**：ET（min_samples_leaf=1）每进程约 3.4 GB，最多只能同时跑 2–3 个。
- **参照**：若重训 LightGBM，A 方案约 5.1 h。G 原来层级 1 的 2,464 次训练累计 40.9 进程小时（5 进程 × 2 线程），墙钟约 8 小时。
- **存储**：按 G 的 probs 目录推算（6.5 GB / 3,782 个文件，每个文件存 3 个轮数），每次拟合存一个 float32 概率矩阵约 0.6 MB。A 方案每个学习器约 1.1 GB，磁盘剩余 34 GB。

## 4. 能复用的缓存
1. **LightGBM**：3,782 个概率文件已全部存好，不需要重训。R2V 和 200/250 轮等变体也可以从缓存零成本算出，但它们是 LightGBM 的变体，不是新学习器。
2. **k=0**：两次重复共用，只需 44 次拟合，而不是 88 次。
3. **"同一选样在不同指导下相同时复用"实际用不上**：
   - 层级 1 的 1,232 个选样文件全部找到；
   - 528 对 AE/S2 指导的选样中，完全相同的是 0 对；
   - 跨方法、跨条件的重复哈希也是 0 个。
   缓存键仍可沿用（单元, 特征, k, 选样哈希）。
4. **kNN 外国近邻缓存可行但不值得**：外国部分只算一次 0.5 s，每个选样再合并 0.07 s。但 kNN 本来每次就不到 1.5 s，还要改成只用外国统计量标准化。
5. 各进程读数据的准备时间：冷启动 1.3 s，之后 0.02 s，可以忽略。耗时几乎全在 RF、HGB、ET 三个树模型上。

## 5. 方案 B 的效度问题
- **E1 和 SRS 与学习器无关**：选样只用 PCA 几何，不用任何分类器。
- **PRED、HYB 的选样由 stage0 的 LightGBM 概率 P_T 指导**：新学习器在这些选样上的结果，只能解释为"LightGBM 指导的选样被别的学习器复用"，不能写成"用 RF 做 margin 选样"。
- **建议**：方案 A 作正文，B 中的 PRED、HYB 部分放附录并注明这一点。
- **概率类次要指标**：先验调整 P1、P2 需要校准的概率，kNN 的投票概率和 RF 的概率都不合适。新学习器建议只报 argmax 指标：ba_ge30、OA、E_far。

## 6. 风险：测试集复用与预先登记
**定位**：写成"预先指定的学习器稳健性分析（相对 G 原方案为事后追加）"，不能写成确认性结果。理由：
- G 的 s0/s1 测试集已被多次使用（见测试集审计）；
- 有些格子的结果已经看过，需要逐条披露：
  - **LR**：E1×k=200 的全部 704 个格子都算过精度（part2 的 s0r0 加 did_lr 的其余 3 个单元），SRS k=0/200 在 s0r0 和 s1r0 上也算过；
  - **MLP 早停版**：E1×k=200 的 s0r0；
  - **MLP 集成**：SRS k=0/200 的 s0r0、s1r0；
  - **kNN**：SRS k=0/200 的 s0r0（11 国）；
  - **RF、ET**：SRS k=200 的 s0r0（11 国），以及 AT 的 k=0；
  - **HGB**：只跑过 AT。

**预先登记步骤**（本目录只读，以下由作者执行）：
1. 写 `ANALYSIS_PLAN_guidance_rev2.md`，冻结以下内容：
   - 学习器清单与超参数，以及只在外国池留国家组上调参的规则；
   - 种子 1000·split+rep，MLP 的三个种子在此基础上加 0、1、2；
   - 线程数，库版本（scikit-learn 1.9.1、lightgbm 4.7.0、numpy 2.5.3、scipy 1.18.1）；
   - 选样清单 `tier1_sel_index.csv`，sha256[:16] = 66252ed9025cb7a4；
   - 估计量：每个学习器各自的 θ_E1，沿用 ±0.3 的 TOST、90%/95% 区间和国家层推断；
   - "稳健"的判定规则，例如每个学习器的 90% 区间都落在 ±0.3 之内；
   - 所有学习器都报告，不挑最好的；唯一的剔除条件是事先写明的技术失败（不收敛、概率不合法）；
   - 上面列出的已看过的格子。
2. 在任何正式运行之前，把方案与运行代码的 sha256[:16] 连同时间戳写入 `plan_hashes.txt`。运行脚本启动时校验这个哈希，做法同 `guidance_G.main`。
3. 冻结前只允许做不读标签的计时冒烟测试，即本次这种。冻结后如有任何改动，另写 rev3。

**如果要确认性结果**：必须按测试集审计的建议，在从未作过测试的格子上重新划分。这要重算 stage0 和选样，超出了"不重新选样"的范围。参考成本：
- stage0 每个（国家, 划分）680 s（2 线程）；
- 层级 1 的全部选样累计只用了 0.31 h；
- 主要成本是训练。

## 7. 文件
所有文件都在 `/Users/perng_mini/AlphaCrops_local/verify/multilearner/feasibility_compute/`：
- `bench_one.py`、`driver.py`：单次计时脚本和驱动
- `bench_4thr.jsonl`、`bench_1thr_c4.jsonl`、`bench_4thr_extra.jsonl`：原始计时记录
- `count_fits.py`、`tier1_sel_index.csv`：选样清单与重复统计
- `estimate2.py`、`estimate_plans.csv`：三种方案的估算
- `learners_rev2_draft.json`：待冻结的学习器规格草稿

---

# 附录 C：审稿人期望与文献

**结论：** 现行主实验（G、阶段 A、检验 2、检验 3）只用了 LightGBM 一种模型，分别在 AE 特征和 S2 特征上各训一个。线性模型、MLP、RF 等只在事后诊断里试过 1–2 个单元。近年同类评估论文一般比较 2–5 个学习器，只报 LightGBM 低于惯例，审稿人大概率会问。建议论文正文保留 LightGBM 作主学习器，另补三个：**多项逻辑回归**（线性探针）、**训练充分的 MLP 集成**和 **RF**，都在已保存的选样文件上重训，不需要重新选样。kNN 只作补充的几何探针。TempCNN、Transformer 等深度时序模型不适合现有特征。以下只用本次实际打开过的来源；打不开的会注明。

## (1) 文献里常比较哪些模型（均已打开）

| 文献（年份，刊物或状态） | 场景 | 比较的模型 | 与本项目相关的要点 |
|---|---|---|---|
| Brown et al. 2025，arXiv [2507.22291](https://arxiv.org/html/2507.22291)（AlphaEarth 原文） | 嵌入评估，含 3 个作物数据集 | kNN（k=1、k=3）和线性层，每项评测取最好的一种 | 嵌入产品自己的评估就只用 kNN 和线性探针 |
| Corley et al. 2026，arXiv [2603.02080](https://arxiv.org/html/2603.02080) | 嵌入的汇聚方式 | kNN（k=5，余弦）和多项 LR | 作者把两种探针定位为分别考察几何结构与线性可分性，并承认结论会随探针变化 |
| Tseng et al.（Presto），arXiv [2304.14065](https://arxiv.org/html/2304.14065) | 时序嵌入，CropHarvest | 在嵌入上训 LR 或线性回归、RF、kNN，另有微调 | 简单学习器接嵌入是惯例，理由是部署成本低 |
| Tseng et al. 2025（Galileo），arXiv [2502.09356](https://arxiv.org/html/2502.09356) | GFM，含 CropHarvest 和 Breizhcrops | kNN（k=20）、线性探针、微调；CropHarvest 用 sklearn 默认参数 | 同上 |
| Marsocci et al. 2024（PANGAEA），arXiv [2412.04204](https://arxiv.org/html/2412.04204v1) | GFM 基准，含 PASTIS 作物数据 | 冻结编码器；分类任务用线性探针，分割用 UPerNet；从零训练的 UNet 和 ViT 作基线；时序用 L-TAE | 冻结编码器加线性探针是 GFM 领域的标准 |
| Mughees et al. 2026，arXiv [2609.17138](https://arxiv.org/html/2609.17138) | AE，缅因耕地二分类 | LR、RF、LightGBM、XGBoost，另有 SVM-RBF、kNN、堆叠和最近类中心 | 全量训练时四种模型只差 0.27 pp；专门设计"分类器梯子"来证明主要是嵌入起作用 |
| Ball et al. 2026，*Science of Remote Sensing*（[bioRxiv v3](https://www.biorxiv.org/content/10.64898/2026.02.23.707022v3)） | AE 和 TESSERA，树种分类 | 线性、RF、小型 NN、深层 NN | 用在 GFM 嵌入上的线性分类器，不如用在常规合成影像上的 NN，所以非线性头是必要的 |
| Koukos et al. 2026，arXiv [2609.03480](https://arxiv.org/abs/2609.03480) v3 | AE、TESSERA、手工特征，丹麦树种 | RF、XGBoost、MLP | 三种输入下 MLP 都优于树模型；手工特征加 MLP 最好 |
| Lisaius et al. 2026，arXiv [2601.16900](https://arxiv.org/html/2601.16900) | AE、TESSERA、原始时序，塞内加尔作物 | RF、XGBoost、SVM、LR、MLP，各跑 200 个种子 | 所有表征都用同一组学习器评估 |
| Tong & Wang，arXiv [2509.03497](https://arxiv.org/html/2509.03497) v3（2026） | 8 国跨国作物分类 | 手工特征和嵌入：RF（500 棵）、MLP（8×128）；5 天中值时序：CropNet、TempCNN、ResNet50 等 | 在地理偏移下，时序 CNN 优于 GFM 嵌入加 RF 或 MLP |
| Ma et al. 2026，*JAG* 105258（[arXiv 2601.00857](https://arxiv.org/abs/2601.00857)，PDF 已读） | AE，美国三个农业任务 | RF、XGBoost（各 200 棵） | 明确不用深度学习，理由是样本少，且树模型在表格数据上不逊于深度学习 |
| Hou et al. 2026，RSE（[EarthArXiv](https://eartharxiv.org/repository/view/13036/)） | AE，人口制图 | 摘要写的是线性、RF、深度模型；具体为 Ridge、ElasticNet、RF、HistGBT、MLP，据项目调研 | 这是 RSE 上的 AE 评估范文，用了 5 个模型 |
| Rußwurm et al. 2020（BreizhCrops），ISPRS Archives（[arXiv 1905.11893](https://arxiv.org/pdf/1905.11893)，PDF 已读） | S2 时序作物分类 | RF 加 7 种深度模型（TempCNN、MSResNet、InceptionTime、OmniscaleCNN、LSTM、StarRNN、Transformer） | RF 的 OA 为 0.76–0.78，Transformer 为 0.80–0.81；RF 的输入是 71 个插值日期 × 10–13 个波段 |
| Rußwurm & Körner 2020，ISPRS JPRS 169（[arXiv 1910.10536](https://arxiv.org/abs/1910.10536)） | 原始 S2 时序 | 1D 卷积、RNN、自注意力、RF | 数据预处理对所有模型都有提升 |
| Pelletier et al. 2019，*Remote Sensing* 11:523（[arXiv 1811.10166](https://arxiv.org/abs/1811.10166)） | 46 期影像序列 | TempCNN、RF、RNN | TempCNN 优于 RF 和 RNN |
| Long et al. 2026，*Comput. Electron. Agric.* 246（[arXiv 2507.12590](https://arxiv.org/html/2507.12590)） | 大范围作物制图与迁移 | RF 加 10 种 RNN 类和 Transformer | Transformer 最好；RF 在向相似域直接迁移时有竞争力 |
| Reuss et al. 2026，*ISPRS Open J.* 19（[arXiv 2504.11022](https://arxiv.org/abs/2504.11022)） | EuroCropsML，含 EE、LV、PT | 元学习、迁移学习、自监督 | 地理上相近的区域之间迁移收益最大 |
| Kluger et al. 2021，RSE 262（[arXiv 2109.01246](https://arxiv.org/abs/2109.01246)） | 跨区作物迁移中的先验偏移与特征偏移 | LDA 为主，另在 RF 上演示 | 迁移研究里单一基学习器也被接受，前提是学习器不是结论的对象 |
| Li et al. 2024，IEEE TGRS 62（[PDF](https://www.nass.usda.gov/Research_and_Science/Cropland/docs/Zhengwei_2024_Automated_Crop-Type_Mapping.pdf)，已读） | 美国作物制图 | 只用 RF | 同上：业务制图论文常只用一个学习器 |
| Sheykhmousa et al. 2020，IEEE JSTARS 13（[HZDR](https://www.hzdr.de/publications/Publ-31960)） | 251 篇遥感论文的元分析 | RF 与 SVM | RF 和 SVM 是遥感领域的两大传统学习器 |
| Grinsztajn et al. 2022，[arXiv 2207.08815](https://arxiv.org/abs/2207.08815) | 中等规模表格数据 | 树模型与深度学习 | 树模型仍是最优 |
| Gorishniy et al. 2021，NeurIPS（[arXiv 2106.11959](https://arxiv.org/abs/2106.11959)） | 表格深度学习 | MLP、ResNet、FT-Transformer | 没有普遍优于 GBDT 的方法 |
| Lowell et al.，[arXiv 1807.04801](https://arxiv.org/abs/1807.04801) | 主动学习 | — | 主动学习得到的数据集与选样用的模型绑定；换一个模型再训练，不一定优于随机样本 |
| Lu et al. 2025，TMLR（[arXiv 2306.08954](https://arxiv.org/abs/2306.08954)） | 表格数据主动学习 | — | 查询模型与任务模型不一致时，不确定性采样会退化 |

以下几篇没打开，只能引用项目已有调研，不作为本文的论据：Wijesingha 2024（ScienceDirect 返回 403）、Zhong 2019 RSE、Wang 2019 RSE（都拿不到正文）、Hou 全文（PDF 超过 10 MB）。

## (2) 审稿人期望的最少模型

- **惯例**：评估型论文一般用 2–5 个学习器。数一下上表：Ma 2 个，Tong 在嵌入上 2 个，Corley 2 个，Brown 3 个，Koukos 3 个，Ball 4 个，Mughees 4–9 个，Hou 5 个，Lisaius 5 个。
- **只用一个学习器的论文**：Li 2024、Kluger 2021 这类业务或方法论文，学习器本身不是结论的对象。我们的结论讲的是"表征或选样策略的优劣"，而文献（Lowell、Lu）已经说明这类结论会随模型变化，所以不能按这类论文的做法处理。
- **最小可信集合**是四个：
  - LightGBM，作主学习器；
  - **多项 LR**，即线性探针，是嵌入评估的惯例（Brown、Corley、Galileo、PANGAEA、Presto）；
  - **RF**，是遥感的惯例（Ma、Mughees、Lisaius、Koukos、Tong、BreizhCrops、Sheykhmousa）；
  - **MLP**，代表深度方法（Koukos、Lisaius、Tong、Ball、Hou）。Ball 的论文直接指出 GFM 需要非线性头，熟悉这篇的审稿人会要求 MLP。
- **可选项**：
  - kNN：代表"嵌入几何"的惯例；
  - SVM：遥感的历史惯例，但 5 万样本上的 RBF 核很贵；
  - XGBoost 或 HistGB：和 LightGBM 同属 GBDT，冗余。Mughees 中 LightGBM 与 XGBoost 只差 0.03 pp，项目诊断中 HistGB 与 LightGBM 差 ≤ 约 1 pp（只存档了 AT 一国）。

## (3) 哪些学习器最能回应"只用了 LightGBM"

**结论一："AE 与 S2 作为选样指导等价"**

主估计量 θ_E1 用的四种方法（lpm2_nogeo、kcenter、typiclust、kherding）只用几何，选样时不用任何模型。所以 LightGBM 只出现在评估端，拿已存的选样重训其他学习器，比较是干净的。

1. **LR，第一优先。**
   - 它代表"嵌入质量"惯例，而且学习曲线最陡：k 从 0 到 200，增益比 LightGBM 多约 3 pp。选样效应在 LR 下最容易被放大，因此是对"没有差别"最严格的检验。
   - 已有证据：k=200、4 个单元，θ_LR 为 +0.12 [−0.15, +0.40]，LightGBM 下为 −0.12；两者之差 +0.24，p=0.067。
   - 要判等价，还需要补 k=50。
   - LR 的 ba 提升主要来自工作点移动，因此要同时报告 ba 和 OA，或者两边都做同样的 τ 先验调整。
2. **MLP 集成**：回应 Ball 与 Koukos 的"非线性头"论点。诊断中它在 k=200 时比 LightGBM 的 ba 高约 1.3–1.8 pp，而且 OA 不降。
3. **RF**：主要作用是让读者熟悉。它和 LightGBM 都是树集成，提供的新信息较少；诊断中未调参的 RF 比 LightGBM 低 3.3 pp，文中需注明"未调参"。
4. **kNN**：只作几何探针，因为 AE 的邻域可分性更高（10 近邻 BA 46.6 对 38.8）。它不能作为稳健性检验：sklearn 的 kNN 不支持样本权重，无法复现 r=50 和类别权重，且诊断中 ba 低约 5 pp。

**结论二：样本扩充策略的比较**

margin、badge_c、gist_margin、predstrat_srs 都用了外国 LightGBM 的预测来选样，也就是说查询模型就是 LightGBM。

- 用 LR、MLP、RF 在同一批选样上重训，回答的正是 Lowell 提出的问题：LightGBM 选出的样本对其他学习器还有没有用。
- **LR 最有信息量**：它的归纳偏置与树模型差别最大，也最能利用目标标签。predstrat 的收益来自类别构成，按机制应当不依赖学习器；如果它在 LR 和 MLP 下仍然最好，证据就很强。
- **必须写明的限定**：固定选样再重训，不等于"每个学习器用自己的不确定性去选"。后者需要重新选样，超出范围。
- Lu et al. 预期，查询模型和任务模型不一致时不确定性采样会退化，所以 margin 在 LR 下变差不应过度解读。几何法与 SRS 选样时不依赖模型，这部分对比最干净。

**实现与统计上的限定**

- 我在 sklearn 1.9.1 中核对过：LR、RF、MLP、SVC 的 `fit` 都支持 `sample_weight`，KNeighborsClassifier 不支持。
- 测试集审计指出 s0/s1 的测试集已被多次使用。新增学习器的结果如果要当确认性证据，必须在从未当过测试集的格子上评估；否则只能写成敏感性分析或描述。

## (4) 深度时序模型为什么不适合

- **S2 特征基本没有时间轴。** 我读了 `data/s2/s2_AT.parquet` 的列名：168 列 = 84 个基础统计 × 30 m 邻域的 {mean, stdDev}。
  - 84 个基础统计包括：年度统计 30 个，植被指数统计 24 个，Q1–Q4 季度统计 28 个（NDVI 和 EVI 的中值与标准差，B4、B8、B11 的中值），greenup 2 个。
  - 只有 5 个变量有 4 个时间步，其余都是全年聚合量。
  - TempCNN、Transformer 设计时面对的是密集序列：BreizhCrops 为 71 个日期，Pelletier 为 46 期，Tong 为 5 天中值。4 个时间步上，卷积或注意力几乎没有可学的东西。
- **AE 没有时间维。** 它是年度嵌入，我们只取了空间邻域的 mean 和 stdDev。能对应到"深度方法"的只有表格 MLP、ResNet 或 FT-Transformer（Gorishniy）。PANGAEA 用 L-TAE 做时序聚合的做法对年度 AE 也不适用。
- **补做的代价会改变研究问题。** 要用时序模型，需要重新从 GEE 提取约 64.5 万个地块的密集 S2 序列，还需要 GPU 和 torch（本机没有，也不得安装）。S2 这一臂也不再是 G 用的那套特征。更关键的是，这会变成"原始时序加 TempCNN"对"AE 加 MLP"，表征和学习器的效应混在一起，而 G 的设计正是为了避开这种混淆。
- **论文中的写法**：放进局限性段。Tong & Wang 发现，在地理偏移下时序 CNN 优于 GFM 嵌入加 RF 或 MLP，这正是审稿人可能追问的依据。结论应限定为"冻结特征加表格学习器"的设定，并说明端到端时序模型不在范围内。另外，BreizhCrops 中 RF 与最好的深度模型只差约 2–4 pp（OA）或约 5 pp（平均精度）；Long et al. 发现 RF 在向相似域迁移时有竞争力。这两点可以用来说明学习器差异通常小于协议差异。

## 附：旁注与文件

- **原稿数字不一致**：`01_main.tex` 中 S2 特征清单的数字相加是 84 个（30+24+28+2），正文却写 168 维，漏写了邻域 mean/stdDev 这一步的 ×2。建议修正。
- **草稿文件**，都在 `/Users/perng_mini/AlphaCrops_local/verify/multilearner/reviewer_lit/`：
  - `breizhcrops.txt`
  - `ma2026.txt`
  - `li2024tgrs.txt`

  这三个是从 PDF 抽出的全文，没有修改任何项目文件或结果文件。