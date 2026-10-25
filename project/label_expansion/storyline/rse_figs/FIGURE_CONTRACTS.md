# 图件契约（RSE 故事线，2026-09-30）

每张图先回答"这张图要捍卫哪一句话"。图中文字一律英文（投稿用）；本文件说明用中文。

## 0. 全局规则（所有图必须遵守）

**代码与运行**
- 每张图一个脚本 `rse_figs/figN_xxx.py`，开头 `import nstyle as ns; ns.setup()`，用 `ns.save(fig, 'figN_xxx')` 导出。导出格式为 SVG、PDF、TIFF 600 dpi 与 PNG 300 dpi，都写到 `rse_figs/out/`。
- 运行命令：`cd rse_figs && PYTHONDONTWRITEBYTECODE=1 ~/AlphaCrops_local/.venv/bin/python figN_xxx.py`。只用 Python/matplotlib/geopandas。
- 只读 `rse_figs/source_data/` 下的文件，路径用 `ns.src('xxx.csv')`。不读其他结果目录，不重算统计量，不训练。
- 代码注释用中文。线程 ≤ 2。不修改 `nstyle.py`、`prep_data.py`、`source_data/`。若 nstyle 需要改，在报告里提出。
- 图中文字、图例与导出文件中，不得出现本机路径、私有文件名或内部代号（"G 实验""检验 1""PC-1""v5"等）。改用中性描述，例如 "pre-registered guidance experiment" 与 "source-selection analysis"。

**尺寸与字体**
- 宽度：双栏 183 mm（`ns.W2`），单栏 89 mm（`ns.W1`），1.5 栏 120 mm（`ns.W15`）。高度按内容定，≤ 170 mm。
- 字体为 Arial，字号 5.5–7 pt：刻度 6、轴标签 6.5、面板内标注 ≥ 5.5。面板标号为小写粗体 8 pt（`ns.panel(ax, 'a')`）。
- 去掉上、右轴线。不画网格；需要时只画极淡的参考线。背景白色。
- 不用双 y 轴。不用彩虹色图。图中不写标题句，结论放在图例文字里。

**颜色语义**（全文固定，见 `nstyle.C`）

| 实体 | 键 |
|---|---|
| 随机标签（场景 A，零起点 SRS） | `LABEL` #0F4D92 |
| 已有调查 + 全国随机补采（S0 + SRS） | `SPREAD` #3775BA |
| 已有调查 + Betti 分配（S0 + BA） | `BETTI` #7884B4 |
| 集中样本：已有调查本身 S0；S0 + 原区扩采 EXT | `CONC` #9A4D8E |
| AlphaEarth 特征 | `AE` #42949E |
| Sentinel-2 物候统计特征 | `S2` #767676 |
| 零标签迁移 Z0 | `REF_Z0` #4D4D4D |
| 只用目标国训练（约 3 万标签）TGT | `REF_TGT` #272727 |

- 绿/红（`UP`/`DOWN`）只用于方向提示，不用于类别。
- 学习器用标记形状区分：LightGBM 实心圆，RF 空心圆，MLP 方块，LR 三角。
- RO 与 PT 不在 9 国主集中，一律用空心或灰色虚线，并在图中直接标注国家代码。

**统计标注**
- 国家为推断单位。确认运行的区间为 95% t CI（df = n − 1）；交互量为 90% CI；X3 为 90% 国家层 bootstrap 区间。每处都要在图例文字里写明。
- 能写 "n_pos/n countries > 0" 的地方写上，例如 "9/9"。
- 数字保留 2 位小数，负号用 U+2212。

**术语**（图中统一用英文写法）

| 概念 | 图中写法 |
|---|---|
| FA | "BA at matched OA"。y 轴写 "ΔBA at matched OA (pp)" |
| Z0 | "zero-label transfer" |
| RT-k | "+k random target labels" |
| TGT | "target-only model (~30k labels)" |
| S0 | "existing survey (100 clustered labels)" |
| EXT | "extend surveyed area" |
| SRS（场景 B） | "spread nationally (random)" |
| BA（分配） | "Betti allocation"。注意不要与 balanced accuracy 混淆：分配策略一律写 "Betti allocation"，指标写 "BA" |

**每张图的交付**
1. 脚本；
2. `out/` 下四种格式；
3. `out/figN_xxx_legend.txt`：英文图例，含每个面板的数据定义、n、区间类型。写出图中出现的每一个数字及其来源 CSV 与行。

交付前自检：
- 用 Read 工具查看 PNG，确认没有文字重叠、裁切或出界；
- 确认 SVG 中文字为 `<text>` 且 font-family 为 Arial；
- 确认最小字号 ≥ 5.5 pt；
- 核对图中每个数字与源数据一致。

---

## Fig. 1 设计 `fig1_design`（双栏，约 183 × 90 mm；schematic-led composite）

**核心句：** 两个平行场景共用同一个迁移协议与同一个评估：外国池来自 10 国；目标国要么零标签起步，要么已有一次集中的局部调查；评估在带缓冲的空间测试格上按同 OA 比较 BA。

**a｜研究区地图**（约 55 mm 宽）
- 数据：`countries.geojson`（CNTR_CODE, study）与 `country_counts.csv`。
- 投影：EPSG:3035。范围约为经度 −10…33、纬度 35…70。
- 11 国用浅蓝灰填色（#B4C0E4），9 国主集与 RO/PT 区分开：RO、PT 用同色斜线填充，或更浅的填色。其他国家为 #EEEEEE 填充、白色边线。
- 在国家质心处标 2 字母代码，6 pt。
- 角落写图注："11 target countries; each is mapped from a pool of the other 10"。在空白处列出各国地块数，n_parcels 以千为单位。

**b｜迁移与评估流程示意**（约 50 mm 宽）
用 FancyBboxPatch 画 4 个圆角框，竖向排列，框之间用箭头连接：
1. "Foreign pool: 10 countries × 5,000 parcels"
2. "+ target labels (weight 50): Scenario A — none → k random; Scenario B — existing survey + k1 new"
3. "LightGBM (main) · Random forest (check)"
4. "Test: 10-km cells of the target country, 1-km buffer · Metric: BA at matched OA"

- 框线用 `REF_Z0`，框内文字 6 pt。框 2 中的场景 A、B 用各自颜色的小色块标示。

**c｜示例目标单元（比利时，一半储备格）**（约 75 mm 宽，2×2 小图）
- 数据：`example_unit_layout.csv`（role ∈ test、candidate、S0、SRS100、EXT100、BA100，坐标 x_km、y_km）、`example_unit_cells.csv`（cx、cy、region、S0_region、S0_cell）、`example_unit_test_cells.csv`。
- 每个小图的底图相同：
  - candidate 为 #D8D8D8 小点，s ≈ 0.3；
  - 测试格为 10 km 方格，填 #4D4D4D，alpha 0.18，并在第一个小图的图例里标 "test cells"；
  - 等比例坐标、无刻度，加 20 km 比例尺。
- 4 个小图：
  1. "Scenario A: random labels"：SRS100 为 `LABEL` 实心点，s ≈ 4；
  2. "Scenario B: existing survey"：S0 为 `CONC` 实心点。S0 所在的 2 个调查区，其格子画 #9A4D8E、alpha 0.10 的淡色方格，标 "surveyed regions"；
  3. "+ extend surveyed area"：S0 画浅 CONC，EXT100 画 `CONC` 实心点；
  4. "+ Betti allocation"：S0 画浅 CONC，BA100 画 `BETTI` 实心点。
- 在小图 2 的角落写："100 labels = 10 cells × 10 parcels in 2 of 8 regions"。

**图例文字须说明**：示例单元的 test 与 candidate 点是抽样显示（test 4,000 个，candidate 12,000 个）；这里显示的补采是 k1 = 100、第 1 次抽样的结果。

---

## Fig. 2 为什么要在同 OA 下比较 `fig2_evaluation`（双栏，约 183 × 62 mm）

**核心句：** 加标签和换表征都会移动工作点（OA 与 BA 相互交换），固定在 τ = 0 读数会把两根杠杆的 BA 回报读小；在同 OA 窗口内比较 BA，才能把前沿提升与沿前沿的移动分开。

**a｜示例单元：标签怎样移动前沿**（约 58 mm）
- 数据：`example_unit_curves.csv`（unit BE_F1，learner lgb，feat ae）与 `example_unit_fa.csv`。
- 画 OA（x）对 BA（y）的曲线，每条曲线是一个臂的 τ 族：
  - Z0：`REF_Z0`；
  - RT200 r0：`LABEL`；
  - RT400 r0：`LABEL` 虚线，可选；
  - TGT：`REF_TGT`。
- 显示范围约 OA 70–92、BA 35–76。曲线按 τ 排序连线；曲线在窗口外出界属于正常。
- 用实心点标出 τ = 0 的点，并标注 "τ = 0"。
- 窗口 [OA★ − 4, OA★] 画浅灰竖带（`BAND`），标 "matched-OA window"；OA★ 取 example_unit_fa 中 ae 行的 OA_star。
- 用双向箭头或短标注写出本单元的 FA 差值 "+3.18 pp"：取 `example_unit_diffs.csv` 中 name dFA_RT200、feat ae、r 0 的 value = 3.18，即窗口内逐点差的平均。
- 在 τ = 0 处另标本单元两点的 BA 差与 OA 差（由曲线文件读 τ = 0 行计算），说明固定工作点读数不同。
- 右上角写：TGT "target-only (~30k labels)"。

**b｜同一单元：表征怎样移动前沿**（约 50 mm）
- 画 Z0 的 AE 曲线（`AE`）与 S2 曲线（`S2`），各标出 τ = 0 的点。
- 窗口锚点取 min(OA★_ae, OA★_s2) = 81.73，窗口为 [77.73, 81.73]。标注本单元 D = "+6.38 pp"：取 `example_unit_diffs.csv` 中 name D_Z0 的 value。
- 面板角落标 "illustrative unit (Belgium)"。

**c｜两种读法对比**（约 70 mm，森林图）
数据来自 `confirm_estimates.csv`，每行给均值与 95% CI，并标 n_pos/n：

| 组 | 行 | est_id | 取值条件 |
|---|---|---|---|
| "+200 random labels (vs zero-label), 9 countries" | ΔBA at τ = 0 | M1_RT200-Z0_BA | feat both、set 9 |
| | ΔOA at τ = 0 | M1_RT200-Z0_OA | 同上 |
| | ΔBA at matched OA | H1 | feat both、set 9 |
| "AE instead of S2 (zero labels), 11 countries" | ΔBA at τ = 0 | M1_Z0_AE-S2_BA | feat ae-s2、set 11 |
| | ΔOA at τ = 0 | M1_Z0_AE-S2_OA | 同上 |
| | ΔBA at matched OA | H2 | set 11 |

- 两组分别用 `LABEL` 与 `AE` 着色。"matched OA" 行用实心大点，τ = 0 的两行用空心点。
- 0 处画参考线。x 轴写 "Difference (pp)"。

**图例文字须说明**：
- τ 族定义：ℓ = log P − τ·log π̂，τ ∈ [−2, 2]，步长 0.05；
- OA★ 为 Z0 在 τ = 0 时的 OA，窗口为 [OA★ − 4, OA★]；
- 9 国 = 去掉 RO、PT；
- H1、H2 为 Holm 校正的确认性检验，两者都拒绝 H0。

---

## Fig. 3 RQ1：从零起步，标签能买到什么 `fig3_labels_zero`（双栏，约 183 × 62 mm）

**核心句：** 多源迁移之后，50–400 个随机目标标签让同 OA 下的 BA 提高 0.8–3.7 pp，9 个主集国家全部为正；这是真实的增益，但最多只补上本国全量训练缺口的四分之一。RF 给出同样的答案。

**a｜剂量曲线（主面板，约 75 mm）**
- 数据：`confirm_country_values.csv`，取 learner lgb、feat both，name 为 dFA_RT50、dFA_RT100、dFA_RT200、dFA_RT400，k = 0 处值为 0。均值与 CI 取 `confirm_estimates.csv` 中 A1_RT* 的 both、set 9 行。
- x 轴为 k（0、50、100、200、400），用线性刻度，标签写 "Random target labels (k)"。
- 9 个主集国家各画一条细线（`LABEL`，alpha 0.25，lw 0.6）；RO、PT 用灰色虚线，在线末端标国家代码。
- 均值用粗线（lw 1.4）加 95% CI 误差棒，颜色 `LABEL`。
- 在 k = 200 处标注："+2.34 pp [1.54, 3.14], 9/9 (confirmatory, Holm)"。
- 在 k = 400 的 11 国均值处，用灰色小字另标："11 countries: +2.23 [−0.27, 4.73]"。

**b｜补上缺口的比例（约 55 mm）**
- 数据：`confirm_a2_ratio.csv`（set 9）；TGT 取 A2_TGT（both、set 9）= +16.39 [13.66, 19.12]。
- 横条图：
  - 最上一条为 "target-only model (~30k labels)"，全长 16.39 pp，`REF_TGT` 浅填色。
  - 下面 4 条对应 k = 50、100、200、400，长度为 ΔFA，颜色 `LABEL`。条后写百分比：5%、8%、14%、23%。
- x 轴写 "ΔBA at matched OA (pp)"，范围 0–20。

**c｜换学习器检验（约 45 mm）**
- 数据：`confirm_country_values.csv` 中 dFA_RT200、feat both 的 lgb 与 rf 两组国家值，11 国。
- 散点图：x 为 LightGBM，y 为 RF。画 y = x 虚线。主集国家为 `LABEL` 实心点，RO、PT 为空心点，每个点标国家代码。
- 角落写：
  - "LightGBM +2.34 [1.54, 3.14], 9/9"；
  - "RF +2.03 [1.02, 3.04], 9/9"（A3_RF_RT200，both、set 9）。

**图例文字须说明**：
- ΔBA at matched OA = FA(arm) − FA(Z0)，AE 与 S2 两种特征国家值取平均；
- 每国 2 个半区 × 3 次抽样；
- 9 国（去掉 RO、PT）为主集，理由是这两国零标签 OA★ 仅约 27–35%。

---

## Fig. 4 RQ2：已有调查的价值与补采位置 `fig4_existing_survey`（双栏，约 183 × 64 mm）

**核心句：** 已有的 100 个集中样本约值 100 个随机标签；在原区继续补采，价值递减；在全国补采（随机或 Betti 分配）则与零起点随机设计打平，而 Betti 分配并不比随机好。

**a｜总标签数—增益曲线（主面板，约 80 mm）**
- 数据取 `confirm_estimates.csv`（feat both、set 9）的均值与 95% CI：

| 曲线 | 颜色 | 点 |
|---|---|---|
| Random（场景 A） | `LABEL` | (0,0)、(50,A1_RT50)、(100,A1_RT100)、(200,A1_RT200)、(400,A1_RT400) |
| Concentrated（S0、S0+EXT） | `CONC` | (0,0)、(100,X2_B_S0)、(200,X2_B_EXT100)、(400,X2_B_EXT300) |
| S0 + spread nationally | `SPREAD` | (100,X2_B_S0)、(200,X2_B_SRS100)、(400,X2_B_SRS300) |
| S0 + Betti allocation | `BETTI` | (100,X2_B_S0)、(200,X2_B_BA100)、(400,X2_B_BA300) |

- 各族点在 x 方向错开 ±5，避免误差棒重叠。
- x 轴写 "Target labels in total"，y 轴写 "ΔBA at matched OA (pp)"。
- 图例放在左上角，或者在曲线末端直接标注。

**b｜新标签补在哪里（约 50 mm，森林图）**
- est_id：B1_SRS-EXT100、B1_SRS-EXT300、B1_BA-SRS100、B1_BA-SRS300（both、set 9）。
- 行标签：
  - "Spread − extend, +100"
  - "Spread − extend, +300"
  - "Betti − spread, +100"
  - "Betti − spread, +300"
- 背景画 (−1, +1) 的浅灰带，标 "<1 pp (pre-specified reading)"。每行右侧写 n_pos/9。

**c｜同总数下与零起点随机设计比较（约 50 mm，森林图）**
- est_id：X1_S0-RT100、X1_SRS100-RT200、X1_EXT100-RT200、X1_BA100-RT200、X1_SRS300-RT400、X1_EXT300-RT400、X1_BA300-RT400（both、set 9）。
- 行标签按族着色，例如 "Survey (100) vs 100 random"、"Survey + spread 100 vs 200 random"。
- 0 处画参考线，每行写 n_pos/9。

**图例文字须说明**：
- S0 的构造：8 个 KMeans 调查区中按 PPS 抽 2 个区 → 10 个格 → 每格 10 个地块；
- EXT 只在 S0 所在区内补采；
- Betti 分配按边际增益 γ_g[(n+1)^−½ − (n+2)^−½] 分配到各区；
- 全部配置共用同一个零标签锚点。

---

## Fig. 5 RQ2：交换率与成本规则 `fig5_exchange_cost`（双栏，约 183 × 62 mm）

**核心句：** 集中样本在总数 200 时约值 0.83 个随机标签，在总数 400 时约值 0.72 个。只有当全国分散采一个样本的成本超过集中采样的约 1.2–1.4 倍时，扩大原调查区才更划算。

**a｜各配置值多少个随机标签（约 62 mm）**
- 数据：`confirm_x3_equiv.csv` 中 kind == 'keq' 的行（point、lo90、hi90；inf 表示 > 400）。
- x 为该配置的总标签数：B_S0 = 100；*100 = 200；*300 = 400。y 为 k_eq。
- 画 y = x 虚线，标注 "worth one random label each"。
- 颜色：B_S0 与 EXT 用 `CONC`；SRS 用 `SPREAD`；BA 用 `BETTI`。同一 x 上的点错开 ±6。
- hi90 = inf 时，误差棒画到 400 并加向上箭头；point = inf（B_BA300）时，在 y = 400 处画向上三角，并标 "> 400"。
- 轴范围 0–430。

**b｜集中样本的交换率与盈亏平衡成本比（约 55 mm）**
- 数据：`confirm_x3_equiv.csv` 中 kind == 'conc_diff'（item 100、200、400）与 'conc_rate' 的行。
- 画集中曲线与随机曲线之差（pp）及 90% 区间，x 为 100、200、400。
- 在点旁标注：
  - 100："not distinguishable (rate not reported)"；
  - 200："rate 0.83 [0.75, 0.91] → break-even cost ratio 1.20"；
  - 400："rate 0.72 [0.66, 0.77] → 1.38"。
- break-even = 1 / rate，是派生量，图例里须说明。
- 0 处画参考线。y 轴写 "Concentrated − random (pp)"。

**c｜成本效益（约 62 mm，3 个并排小图：c = 1、2、3）**
- 数据：`confirm_x3_cost.csv` 中 kind == 'budget' 的行（c、name、cost、value、best）。
- 每个小图：x 为预算（100、200、400，单位为"一个集中样本的成本"），y 为 ΔBA at matched OA。
- 4 条线：random（`LABEL`）、concentrated（`CONC`）、S0+SRS（`SPREAD`）、S0+BA（`BETTI`）。best == True 的点加黑色描边星号。
- 小图标题写 "c = 1"、"c = 2"、"c = 3"，并在旁边写 "(c = cost of a spread label ÷ cost of a concentrated label)"。
- 3 个小图共用 y 轴。

**图例文字须说明**：
- k_eq 的算法：PAV 单调化随机曲线后线性插值；
- 区间为 90% 国家层 bootstrap（2,000 次）；
- 成本效益只报点估计，按方案族内线性插值，不外推。

---

## Fig. 6 RQ3：换表征与加标签 `fig6_representation`（双栏，约 183 × 115 mm，两行）

**核心句：** 零标签时，把 S2 物候统计换成 AE 嵌入，同 OA 下 BA 提高 5.4 pp，超过 400 个随机标签能买到的增益。这一优势在加标签后仍然存在，与标签可叠加，换成 RF 也成立。但 AE 当作"选哪些地块去标"的指导时，并不比 S2 更好。

**第一行**

**a｜两根杠杆放在同一尺度（主面板，约 70 mm）**
- 横向点图加 95% CI，9 国主集（同一国家集合）：
  - "+50 / +100 / +200 / +400 random labels"：A1_RT*（both、set 9），颜色 `LABEL`；
  - "AE instead of S2 (zero labels)"：H2 的 set 9 行 +6.21 [2.60, 9.82]，颜色 `AE`。
- 另起一行，灰色小字写 11 国值 "+5.37 [2.01, 8.73] (11 countries, confirmatory)"。
- x 轴写 "ΔBA at matched OA (pp)"，0 处画参考线。

**b｜逐国 AE − S2（约 60 mm）**
- 数据：`confirm_country_values.csv` 中 feat ae-s2、name D_Z0 的 lgb 与 rf 国家值，11 国，按 LightGBM 值排序。
- 横向点图：LightGBM 实心 `AE`，RF 空心 `AE`。RO、PT 的行标签用灰色。
- 底部写均值：
  - "LightGBM +5.37 [2.01, 8.73], 9/11"；
  - "RF +6.84 [4.18, 9.51], 11/11"（E1_RF_D_Z0，set 11）。

**c｜加标签后仍在（约 45 mm，森林图，11 国）**
- 行：
  - "zero labels"：H2，set 11；
  - "+200 random labels"：E1_D_RT200，set 11；
  - "survey + 100 spread"：E1_D_S0+SRS100，set 11。
- 颜色 `AE`，0 处画参考线。

**第二行**

**d｜标签对两种特征的回报相同（约 55 mm）**
- 数据：A1_RT* 的 feat ae 与 feat s2 行（set 9），均值与 95% CI。
- 两条剂量曲线：AE 用 `AE`，S2 用 `S2`，x = 0、50、100、200、400。
- 注意：y 是各自相对于本特征 Z0 的增益，不是绝对水平。

**e｜交互量接近 0（约 55 mm，森林图，90% CI）**
- 行：
  - E2_RT200：9 国主，11 国为灰色附行；
  - E2_B1_SRS-EXT100、E2_B1_SRS-EXT300、E2_B1_BA-SRS100、E2_B1_BA-SRS300：9 国。
- 行标签：
  - "label gain (+200)"
  - "spread − extend (+100)"
  - "spread − extend (+300)"
  - "Betti − spread (+100)"
  - "Betti − spread (+300)"
- x 轴写 "AE − S2 difference in effect (pp)"。

**f｜AE 作选样指导（约 60 mm）**
- 数据：`guidance_theta.csv`（metric ba_ge30），4 个学习器。
- 画 θ_E1 = BA(AE-guided) − BA(S2-guided)（pp）及 90% CI；等价带 ±0.3 pp 画浅灰带，标 "equivalence margin ±0.3 pp"。
- 学习器用形状区分（见全局规则）。每行右侧写结论："equivalent (TOST p = 0.008)"、"equivalent (p = 0.006)"、"equivalent (p = 0.021)"、"inconclusive"。
- 面板副标题写："separate pre-registered experiment; 11 countries"。

**图例文字须说明**：
- D 的锚点为 min(OA★_ae, OA★_s2)；
- 交互量 = AE 国家值 − S2 国家值；
- 面板 f 来自另一次预注册的选样指导实验（同 11 国，另一组空间测试划分），θ_E1 在 11 国上平均，TOST 的等价界为 ±0.3 pp。

---

## Fig. 7 给实践者的决策图 `fig7_decision`（双栏，约 183 × 78 mm；示意图）

**核心句：** 先换表征，再决定标签：
- 没有调查时，随机采几百个；
- 已有调查时，保留它，新标签按成本比决定分散还是扩区；
- 用同 OA 的 BA 评估。

**版式**：三列，从左到右用粗箭头连接，每列一个标题框加 2–3 个内容框。

1. **"1 · Representation"**
   - 内容框："Use AE embeddings, not S2 statistics: +5.4 pp BA (11 countries); more than 400 labels buy"。颜色 `AE`。
   - 小字注意："Weaker on small parcels; PT the exception"（C-grade, supplementary）。
2. **"2 · Labels"**，分叉成两个框：
   - "No survey → k random labels: +2.3 pp for 200, +3.7 pp for 400 (≤ 23% of the gap). Smarter geometric sampling ≈ random."，颜色 `LABEL`。
   - "Existing survey → keep it (100 clustered ≈ 95 random). Add new labels nationally unless a spread label costs > 1.2–1.4× a concentrated one; Betti allocation ≈ random."，颜色 `CONC`/`SPREAD` 渐变或双色边。
3. **"3 · Evaluate"**
   - 内容框："Compare BA at matched OA; fixed-operating-point readings hide most of the label gain (+0.4 vs +2.3 pp)"。颜色 `REF_Z0`。
   - 另加一个灰框："Not supported: AE as a sampling guide or as a source-selection distance"。

**要求**：
- 框内文字 6–6.5 pt，每框不超过 3 行。框为圆角，填色用对应颜色的 12% 透明度，边框为对应颜色。
- 不画任何图标。

---

## 补充图

### Fig. S1 逐国结果热图 `figS1_country_heatmap`（双栏，约 183 × 100 mm）
- 数据：`confirm_country_values.csv`。
- 行：
  - dFA_RT50…400（lgb、both）；
  - dFA_TGT；
  - dFA_B_S0、dFA_B_SRS100、dFA_B_SRS300、dFA_B_EXT100、dFA_B_EXT300、dFA_B_BA100、dFA_B_BA300；
  - B1 四项；
  - D_Z0、D_RT200、D_B_SRS100（ae-s2）；
  - rf dFA_RT200、rf D_Z0。
- 列：9 国主集；空一列；RO、PT。
- 分组标题：
  - "Scenario A (random labels)"
  - "Scenario B (existing survey)"
  - "Placement contrasts"
  - "AE − S2"
  - "Random forest"
- 发散色：负为 `DOWN`，0 为 #F7F7F7，正为 `LABEL`，截断在 ±8。TGT 行超出色域的格子加 "▲" 标注。
- 每格写数值，5.5 pt，1 位小数。

### Fig. S2 AE 优势来自哪里（C 级）`figS2_ae_drivers`（双栏，约 183 × 62 mm）
- **a｜按类：** `ae_by_class.csv`。用 11 国列：d11、lo11、hi11、pos11、n11。按 d11 排序画横向森林图，行标签写类名与 "(n countries)"。颜色 `AE`，0 处画参考线。x 轴写 "AE − S2 recall at matched OA (pp)"。
- **b｜按地块大小：** `ae_by_size.csv`，组为 <0.5、0.5–1、1–2、2–4、≥4，单位是核面积 π·30² ≈ 0.28 ha 的倍数。画 d11 与 95% CI；x 轴下方另标各组地块占比 share11。y 轴写 "AE − S2 BA at matched OA (pp)"。
- 图例须写明：事后分析，测试集为另一组已多次使用的划分，只作描述，C 级。

### Fig. S3 AE 作为选源距离 `figS3_source_selection`（单栏，89 × 60 mm）
- 数据：`source_selection_rho.csv`，7 个描述量。
- 显示名：
  - MMD (AE)
  - MMD (S2)
  - C2ST (AE)
  - C2ST (S2)
  - geographic distance
  - climate difference
  - class-composition JS
- 画每个目标国的 Spearman ρ（−distance 与迁移 BA）均值，加 95% bootstrap CI。AE 描述量用 `AE`，S2 描述量用 `S2`，其余用 `REF_Z0`。
- 注明：AE-MMD − S2-MMD 的 Δρ = +0.021 [−0.018, 0.062]，6/11 国更好；数据为 11 国、110 个单源迁移对。

### Fig. S4 选样策略对比随机 `figS4_strategies`（1.5 栏，120 × 70 mm）
- 数据：`strategy_vs_srs.csv`，只画 guidance 为 AE 与 S2 两种；AE+S2 不画。
- 8 种方法 − SRS 的 BA（pp）加 95% CI。每个方法一行，AE 与 S2 两个点上下错开。背景画 ±1 pp 带，0 处画参考线。
- 显示名：
  - predstrat_srs → "prediction-stratified"
  - kcenter → "k-center"
  - kherding → "kernel herding"
  - lpm2_nogeo → "LPM (feature space)"
  - typiclust → "TypiClust"
  - margin → "margin"
  - gist_margin → "GIST-margin"
  - badge_c → "BADGE"
- 图例写：预注册的选样指导实验；2 个分类器 × 2 个 k 等权合并；11 国；Holm 校正后没有方法显著优于随机。
