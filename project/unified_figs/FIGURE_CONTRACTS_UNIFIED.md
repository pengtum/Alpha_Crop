# 三任务统一图件契约（作物 crop · 城市形态 lcz · 土地覆盖 lucas）

**目标**：作物、LCZ、LUCAS 三个任务走同一条分析流程，用同一套图件呈现。每张图由一个脚本生成，脚本对 `us.TASKS` 循环，三个任务的版式、尺寸、颜色和文字规则完全相同。只有一张跨任务总览图把三个任务放在一起。图中文字一律英文；本文件用中文说明。

## 0. 全局规则

**代码与运行**
- 脚本放在 `unified_figs/ufN_xxx.py`。开头 `import ustyle as us; us.setup()`，导出用 `us.save(fig, f'ufN_xxx_{task}')`（跨任务图不加任务后缀），图例文字用 `us.write_legend(name, text)` 写出。
- 运行：`cd unified_figs && PYTHONDONTWRITEBYTECODE=1 ~/AlphaCrops_local/.venv/bin/python ufN_xxx.py`。只用 matplotlib、geopandas、numpy、pandas。
- 数据只读 `unified_figs/source_data/`：用 `us.read(task, what)`、`us.est(...)`、`us.dom(...)`、`us.row(...)`、`us.src(...)`。不重算统计量，不训练。
- 不修改 `ustyle.py`、`prep_unified.py`、`source_data/`，也不修改 RSE 的 `rse_figs/` 下任何文件。若共享组件需要改，在报告中提出。线程数 ≤ 2。代码注释用中文。
- 图中不得出现本机路径、私有文件名或内部代号（"G 实验""试点 v4""T1/T2"等）。任务名用 `us.META[task]['title']`。

**样式**（沿用 RSE 的 nstyle）
- 尺寸：宽度用 `us.W2`（183 mm）、`us.W15`（120 mm）或 `us.W1`（89 mm），高度 ≤ 170 mm。
- 字体为 Arial，字号 5.5–7 pt。面板标号用 `us.panel(ax, 'a')`。
- 不加上、右轴线，不画网格，不用双 y 轴，图中不写结论标题句。

**颜色语义**（全文固定，见 `us.C`）

| 实体 | 颜色键 |
|---|---|
| 随机目标标签 | `LABEL` |
| AE | `AE` |
| S2 | `S2` |
| 零标签迁移 | `REF_Z0` |
| 参照臂 REF | `us.REF_COLOR` |

- 参照臂 REF：作物为只用目标国训练的模型（约 3 万标签）；LCZ、LUCAS 为用尽全部候选标签。文字用 `us.META[task]['ref']` 或 `ref_short`。
- 绿、红只表示方向。

**统计显示（统一规则）**
- 域数 n ≥ `us.CI_MIN_N`（5）时，画均值与 95% t CI；交互量画 90% CI。
- n < 5 时，不画区间，改画各域的值（`us.forest` 与 `us.dose` 已内置）。
- 作物 n = 9（主集），LCZ、LUCAS 各 n = 3。
- 作物的 RO、PT 不在主集：用灰色或空心表示，并标出国家代码。
- 数字保留 2 位小数，负号用 U+2212（`us.fmt_pp`）。

**术语**

| 概念 | 图中写法 |
|---|---|
| 同 OA 下的 BA | "BA at matched OA"；y 轴写 "ΔBA at matched OA (pp)" |
| Z0 | "zero-label transfer" |
| RT-k | "+k random target labels" |
| D | "AE instead of S2" |
| 交互 | "AE − S2 difference in label gain" |

**源数据**（`source_data/{task}_{what}.csv`）
- `estimates`：name、feat、set、n、mean、lo95、hi95、lo90、hi90、n_pos。
- `domain_values`：domain、name、feat、value。
- `unit_rows`：unit、name、feat、r、value。
- `arm_rows`：τ = 0 的 OA、BA，以及 OA★。
- `example_curves`：示例单元的 τ 族 OA–BA，臂为 Z0、RT200、RT400、REF，只有 r0。
- `example_layout`：role ∈ test / candidate / RT200 / buffer，坐标为 x_km、y_km。
- `example_test_cells`：cx、cy、size_km、off_km；格子左下角 = off_km + c × size_km。
- `class_rows`、`class_summary`：按类的 AE − S2 召回（同 OA 点）；summary 含 cls_name、n、mean、lo95、hi95、n_pos。
- `size_rows`、`size_summary`：只有作物与 LUCAS。
- `{task}_domains`：地图用，含 domain、n、role，LCZ 另有 lon、lat。
- `world_lsib.geojson`：底图，字段 name。
- `crop_study_regions_de.geojson`、`crop_study_regions_ro.geojson`。
- `tasks_meta.json`：primary、all、example_unit。

**name 与 feat 的取值**
- name：
  - dFA_RT50、dFA_RT100、dFA_RT200、dFA_RT400、dFA_REF；
  - D_Z0、D_RT200、D_RT400；
  - t0_RT200-Z0_BA、t0_RT200-Z0_OA、t0_Z0_AE-S2_BA、t0_Z0_AE-S2_OA。
- feat：ae、s2、both（两特征平均）、inter（AE − S2）、ae-s2（D 与 t0_Z0）。

---

## UF0 跨任务总览 `uf0_overview`（双栏，183 × 约 110 mm，两行三列）

**核心句**：在三个任务中，迁移之后换用 AE 带来的 BA 增益都大于 400 个随机目标标签。

**第一行（a–c，每列一个任务）：剂量曲线与 AE 参考线**
- 用 `us.dose(ax, task)` 与 `us.ae_band(ax, task, xmax=400)`。
- 标题为 `META title`；只在 a 写 y 轴标签。
- 三列各用自己的 y 范围，但刻度风格相同。

**第二行（d–f）：两根杠杆同尺度**，用 `us.forest`，行自上而下为：
- "+50 random labels" … "+400 random labels"：dFA_RT{k}、both，颜色 `LABEL`；
- REF：dFA_REF、both，颜色 `us.REF_COLOR`，行标签用 `ref_short`；
- "AE instead of S2 (0 labels)"：D_Z0、ae-s2，颜色 `AE`，实心大点。

**图例文字须说明**：
- 三个任务的域和样本单位；
- 作物为 9 国主集，LCZ、LUCAS 各 3 个目标域；
- CI 显示规则；
- REF 在不同任务中的含义。

## UF1 设计 `uf1_design_{task}`（双栏，183 × 约 75 mm）

**a｜研究区地图**（约 70 mm）
- 底图 `world_lsib.geojson`，国界 #FFFFFF 0.3 pt，陆地 #EEEEEE。
- **crop 与 lucas**：EPSG:3035，范围约经度 −11…35、纬度 34…71。
  - 目标域国家填 `#B4C0E4`。lucas 的只作源池的国家填 `#E1E6F3`。
  - crop：11 国都作过目标。RO、PT 用同色斜线填充；DE 与 RO 叠加实际研究区，做法参照 RSE 的 `rse_figs/fig1_design.py`。
  - 在目标国中心标 2 字母代码，6 pt。
- **lcz**：世界地图，Equal Earth（EPSG:8857）。29 个源城市为小点（`S2` 色，s ≈ 6）；3 个目标城市为 `LABEL` 色大点，并标城市名。
- 角落小字说明：
  - crop："11 target countries; each mapped from a pool of the other 10"
  - lcz："3 target cities (one per region); each mapped from the other 29"
  - lucas："3 target countries (south, central-west, north); each mapped from the other 25"

**b｜流程示意**（约 45 mm）：4 个圆角框，竖排，用箭头连接。
1. `META pool`
2. "+ k random target labels (weight 50), k = 50–400; reference: " + `META ref`
3. "LightGBM · features: AE (64-d) or S2 (84 statistics)"
4. `META test` + " · Metric: BA at matched OA"

**c｜示例目标域布局**（约 60 mm）：示例单元为 `tasks_meta.json` 中的 example_unit，作物为 BE 的一个半区。
- 测试格画方块：#4D4D4D，alpha 0.15，无边线。
- candidate 为 #D8D8D8 小点；test 为 #9A9A9A 小点；RT200 为 `LABEL` 实心点，s ≈ 4。
- 等比例，无刻度，加比例尺：crop、lucas 50 km，lcz 5 km。
- 标题写："Example target: {domain} — 200 random labels (draw 1)"。

## UF2 评估 `uf2_evaluation_{task}`（双栏，183 × 62 mm）

**a｜示例单元：标签怎样移动前沿**
- 数据 `example_curves`，feat 取 ae。
- 曲线：Z0（`REF_Z0`）、RT200（`LABEL`）、RT400（`LABEL` 虚线）、REF（`us.REF_COLOR`，细线）。
- τ = 0 的点画实心点，并标 "τ = 0"。
- 窗口 [OA★ − 4, OA★] 画浅灰竖带（`C['BAND']`），OA★ 取该单元 Z0 ae 在 τ = 0 的 OA。
- 标注本单元的 dFA_RT200：取 `unit_rows` 中 unit = 示例单元、name = dFA_RT200、feat = ae、r = 0 的值。
- 坐标轴为 OA (%)、BA (%)，范围按数据自定。

**b｜示例单元：表征怎样移动前沿**
- Z0 的 AE 曲线（`AE`）与 S2 曲线（`S2`）。
- 窗口锚定在 min(OA★_ae, OA★_s2)。
- 标注本单元的 D_Z0（`unit_rows`）。

**c｜两种读法对比**：用 `us.forest`，两组：
- "+200 random labels vs zero labels"：
  - t0_RT200-Z0_BA（both），写作 "ΔBA at τ = 0"；
  - t0_RT200-Z0_OA（both），写作 "ΔOA at τ = 0"；
  - dFA_RT200（both），写作 "ΔBA at matched OA"。
- "AE instead of S2 (zero labels)"：
  - t0_Z0_AE-S2_BA、t0_Z0_AE-S2_OA；
  - D_Z0。
- matched-OA 行用实心点，τ = 0 行用空心点。x 轴写 "Difference (pp)"。

**图例文字须说明**：τ 族定义；窗口定义；数据单位。

## UF3 标签与表征 `uf3_labels_representation_{task}`（双栏，183 × 62 mm）

- **a｜剂量曲线**（约 75 mm）：`us.dose` 加 `us.ae_band`。在 k = 200 处标注 `us.ci_text(est(dFA_RT200, both))`。
- **b｜两根杠杆同尺度**（约 55 mm）：行与 UF0 第二行相同。
- **c｜逐域对比**（约 45 mm）：
  - 每个域一行：D_Z0（`AE` 实心）与 dFA_RT400（both，`LABEL` 空心）。
  - 按 D_Z0 排序；非主集域的行标签为灰色。
  - x 轴写 "ΔBA at matched OA (pp)"。

## UF4 互补性 `uf4_complementarity_{task}`（双栏，183 × 62 mm）

- **a｜标签对两种特征的回报**：
  - `us.dose(feat='ae', color=C['AE'], show_domains=False, offset=-4)` 与 `us.dose(feat='s2', color=C['S2'], show_domains=False, offset=+4)`；
  - 各自相对本特征 Z0，图注写明这一点；
  - 域数 < 5 时，用细线画各域（可另写辅助代码）。
- **b｜AE 优势在加标签后仍在**：用 `us.forest`，行为 D_Z0 "zero labels"、D_RT200 "+200 random labels"、D_RT400 "+400 random labels"，颜色 `AE`。
- **c｜交互**：用 `us.forest(level=90)`，行为 dFA_RT50…RT400 与 dFA_REF 的 inter。
  - x 轴写 "AE − S2 difference in label gain (pp)"。
  - 图例写明 90% CI（n ≥ 5 时）。

## UF5 逐域热图 `uf5_domain_heatmap_{task}`（双栏，183 × 约 85 mm）

**行**分 4 组：
- "Random labels"：dFA_RT50…RT400（both）、dFA_REF（both）；
- "AE instead of S2"：D_Z0、D_RT200、D_RT400；
- "Interaction"：dFA_RT200、dFA_RT400、dFA_REF 的 inter；
- "Readings at τ = 0"：t0_RT200-Z0_BA（both）、t0_Z0_AE-S2_BA。

**列**：主集域。作物另空一列后放 RO、PT。

**格式**：
- 发散色：负为 `DOWN`，0 为 #F7F7F7，正为 `LABEL`，截断在 ±8；超出色域的格子加 "▲" 或 "▼"。
- 每格写数值，5.5 pt，保留 1 位小数。
- 列标签用 `us.dlabel`。

## UF6 AE 优势来自哪里 `uf6_ae_drivers_{task}`

**尺寸**：crop 与 lucas 为双栏 183 × 62 mm，两个面板；lcz 为 1.5 栏 120 × 70 mm，只有面板 a。

**a｜按类**
- 数据：`class_summary`，各域的值取自 `class_rows` 按 domain 的平均。
- 行标签为 "类名 (n domains)"，按 mean 排序。
- n ≥ 5 时画 95% CI；否则画各域的点。
- 颜色 `AE`，x 轴写 "AE − S2 recall at matched OA (pp)"。

**b｜按样本面积**（crop、lucas）
- 数据 `size_summary`：组为 <0.5 … ≥4，单位为 π·30² m² 的倍数；画均值，以及 CI 或各域的点。
- 在 x 轴下方标各组占比。
- y 轴写 "AE − S2 BA at matched OA (pp)"。

**图例须写明**：同 OA 点 o_mid = min(OA★_ae, OA★_s2) − 2；只计各单元的 BA 类别；结果为描述性。

---

## 第二轮统一规则（2026-10-05，覆盖前文冲突之处）

**共享组件已更新**（ustyle.py），各脚本改用共享组件，并删除做同样事情的局部替代代码。

**1. 森林图**：一律用 `us.forest(ax, rows, level, cols=True)`。
- 轴右侧固定两列：均值（n ≥ 5 时附 CI，CI 端点不带 + 号）与 "> 0" 计数。列头是 "mean [95% CI]" 或 "mean"，以及 "> 0"。
- 不再在点上方写数字。
- 右上角角注一律用 `us.forest_note(ax, task, level)`。
- 某一行需要放大时，在 row dict 中设 `ms`：例如 AE 行与 matched-OA 行设 4.6。
- n < 5 时，各域点画在均值之上。
- 作物的 RO、PT 由 `us.row(...)` 自动带出 others，画灰色空心圈并标代码（5.5 pt）。只在逐域相关的面板显示，可用 `show_others=False` 关闭。
- 需要分组时（UF2c），用两个上下相邻的轴，各调一次 forest，组标题写在轴上方。

**2. 统一文本**：用 `us.ci_text(e)`，格式为 "+2.34 [1.54, 3.14], 9/9 > 0" 或 "+0.88, 2/3 > 0"。

**3. 术语**：用 `us.TERM`。
- 行名写 "+{k} random labels"；剂量曲线的 x 轴写 "Random target labels (k)"。
- "AE instead of S2 (0 labels)"，各图一律如此，不写 "zero labels"。
- "zero-label transfer"。
- 主集一律称 "primary set"，不用 "main set"。
- 图例文字中的 source pool 一律写 "source pool"；作物可写 "foreign source pool (other 10 countries)"，不要嵌套括号。

**4. 域名**：一律用 `us.dname(task, d, unit)`。
- 作物、LUCAS 用国家全名（Belgium、Sweden……），LCZ 用城市名。
- 作物示例单元写 "Belgium (one spatial half)"。
- 热图列标签与逐域点图的行标签用 `us.dlabel` 的短码（IT、UK、SE、Amsterdam……），作物同样用国家代码。

**5. 剂量面板**（UF0、UF3a、UF4a）
- 统一 `ax.set_xlim(us.DOSE_XLIM)`。
- 调用 `us.ae_band(ax, task)`，使用默认的 xmax 与 x_dom。
- 不再用局部包装改字号或偏移，共享组件已处理。

**6. OA–BA 曲线**（UF2）：REF 臂用 `ls=us.REF_LS`（点线）；零标签迁移为实线。

**7. 尺寸**
- UF5：n < 5 的任务（lcz、lucas）宽度固定为 `us.W15`；crop 为 `us.W2`。
- UF6：lcz 为 `us.W15`。
- UF1：三列宽约 58 / 44 / 69 mm。由数据触发的放大窗口（lcz 的区域放大、lucas 的测试格放大）保留。

**8. 图例文字**
- 设计常量取 `us.DESIGN`；样本短词取 `META[task]['samples_short']`；特征取样支撑取 `META[task]['support']`。
- 示例布局中的候选池大小 = candidate 数 + RT200 数。作物的灰点是显示用的随机子样本，要写明。

## 第三轮补充（2026-10-05）

- **剂量面板**：n < 5 时，`us.ae_band` 只画 AE 均值线，各域的 AE − S2 只在同图的森林图中显示；`us.dose` 在单域线末端标域名。UF0、UF3 的局部补丁随之删除。
- **UF4a 不画 `ae_band`**：其 y 轴是各特征相对自身零标签模型的增益，D_Z0 已在 b 中。
- **森林图**：列头基线 = 轴顶；`forest_note` 固定在列头上方一行。删除各脚本中为错开两者所加的 ylim 补丁。
- **others（RO、PT）**：只在逐域面板显示。UF6 的 a、b 是跨域汇总，不画。
- **行名**
  - 组标题已写 "AE instead of S2" 时，组内行名写 "0 labels"、"+200 random labels"、"+400 random labels"（UF4b、UF5 一致）；单独出现时写 `TERM['ae0']`。
  - REF 行名一律用 `META['ref_short']`；长形式只用在图例文字中。
- **S2、AE 的描述**一律用 `us.DESIGN['s2_desc']`、`us.DESIGN['ae_desc']`。作物源池写 "Foreign source pool"（已改入 META）。
- **UF4**：三个任务的数值列位置按三任务共同的最长文本（以 mm 计）固定，保证三张图同宽。
- **UF1 lucas**：放大窗的比例尺放到窗外，不加底色。
- **UF2**：竖向间距小于两个箭头头长时，改画无头短竖线。
