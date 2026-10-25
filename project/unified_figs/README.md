# 三任务统一分析与配图（作物 · LCZ · LUCAS）

同一条分析流程、同一套图件，覆盖三个任务：
- 作物类型（EuroCrops，RSE 确认运行，11 国）；
- 城市形态（So2Sat LCZ42，NC 试点，3 个目标城市）；
- 土地覆盖（LUCAS 2018，NC 试点，3 个目标国）。

## 流程

1. **数据层**：`prep_unified.py` 从三个任务各自的测试集预测概率出发，用同一份代码算出下列各量：
   - 逐单元的 dFA_RT{k}、dFA_REF、D_Z0、D_RT200、D_RT400；
   - τ = 0 读数；
   - 按类、按面积的 AE − S2。

   度量函数直接 import RSE 冻结的 `analyze_confirm`。运行时与已登记结果对账，不一致即中止：
   - 作物 352 个国家值与 `confirm_v1` 一致，最大差 3.6e-15；
   - LCZ、LUCAS 各 63 个域值与试点的一次性分析一致。

   输出写到 `source_data/`。
2. **样式与组件**：`ustyle.py`。
   - 沿用 RSE 的 nstyle：字体、颜色语义、导出格式。
   - 提供任务元数据、术语、森林图、剂量曲线、AE 参考线等组件。
   - 统一规则：域数 ≥ 5 时画 t 置信区间，否则画各域的值（作物 9 国，LCZ、LUCAS 各 3 个域）。
3. **图件**：`uf0_overview.py` … `uf6_ae_drivers.py`。每个脚本用同一段代码为三个任务出图，输出写到 `out/`，格式为 svg、pdf、tiff、png，另附英文图例 `_legend.txt`。
4. **汇编**：`make_gallery.py`，生成 `out/unified_figures_gallery.pdf`（19 页）。

契约与逐轮修订记录见 `FIGURE_CONTRACTS_UNIFIED.md`。

## 运行

```bash
cd unified_figs
~/AlphaCrops_local/.venv/bin/python -B prep_unified.py
for f in uf0_overview uf1_design uf2_evaluation uf3_labels_representation uf4_complementarity uf5_domain_heatmap uf6_ae_drivers; do ~/AlphaCrops_local/.venv/bin/python -B $f.py; done
~/AlphaCrops_local/.venv/bin/python -B make_gallery.py
```

## 图件

| 图 | 内容 |
|---|---|
| UF0 | 跨任务总览：剂量曲线与 AE 参考线；两根杠杆放在同一尺度 |
| UF1 | 设计：研究区地图、流程、示例目标域布局 |
| UF2 | 评估：示例单元的 OA–BA 曲线；τ = 0 读数与同 OA 读数的对比 |
| UF3 | 标签与表征放在同一尺度：剂量曲线、两根杠杆、逐域对比 |
| UF4 | 互补性：两种特征的标签回报；加标签后 AE 的优势；交互 |
| UF5 | 逐域热图 |
| UF6 | AE 优势的来源：按类；按面积（作物、LUCAS） |

## 与 RSE 作物图的关系

作物图中依赖以下内容的面板，LCZ、LUCAS 试点没有对应数据，统一图中不包含：
- 情景 B（已有调查）；
- 交换率与成本；
- 随机森林检验；
- 选样指导；
- 选源距离。

底图为 GEE `USDOS/LSIB_SIMPLE/2017`，经 3 km 简化后存为 `source_data/world_lsib.geojson`。
