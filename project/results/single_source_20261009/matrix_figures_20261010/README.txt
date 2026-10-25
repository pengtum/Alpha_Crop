单源跨国迁移矩阵（独立扩展，2026-10-10）

实验与读图
11个国家，每次以一国为source、另一国为target，共110个有方向的国家配对，每对两个空间评估单元。每个source提供5,000个训练标签；增加的目标标签与AE/S2严格配对。另有22个等样本量多源空间配置，不纳入这两张单源矩阵。行=source，列=target；A→B和B→A是不同任务。

01_zero_label_transfer_accuracy
Scientific question: How do absolute transfer accuracies vary across all source–target pairs before any target labels are added?
a/b: AE/S2 overall accuracy (OA). c/d: AE/S2 balanced accuracy (BA). LightGBM, no target labels. Cells average the two frozen spatial evaluation units; all 110 directed pairs are retained in each panel. The same 0–100% colour scale is used throughout. Displayed values are rounded to integer percentages; unrounded means are in absolute_source_data.csv. BA averages recall only over classes with at least 30 test parcels in the spatial unit; the eligible class set can differ between target countries. OA and BA are evaluated at the default decision rule, not the matched-OA envelope. No uncertainty intervals or significance tests are encoded: these are descriptive pair means across two spatial units, not independent country replications.

02_paired_representation_advantage
Scientific question: Is the AE advantage widespread across country pairs, robust to learner choice, and retained after adding target labels?
a: No target labels, LightGBM. b: No target labels, random forest. c: 200 random target labels, LightGBM. d: 100 existing clustered labels plus 100 nationally random labels, LightGBM.
Each cell is the within-unit paired BA difference (AE minus S2) on the frozen common-OA window. Positive/teal values favour AE; negative/warm values favour S2. The common colour scale is −30 to +30 percentage points and includes every finite observation without clipping. Repetitions are first averaged within spatial units, then the available spatial-unit means are averaged within each country pair. Zero-label arms have one fit per feature per unit; augmented arms have three paired draws per unit. The source data retain valid draw and unit counts. A common-OA coverage below 50% is unavailable, not zero, and no missing cell is imputed. The self-transfer diagonal (—) was not run; × indicates no available matched-OA comparison; a small dot marks incomplete split/draw support. These dots are not significance markers. Means are descriptive; no cell-wise inferential intervals or multiple-testing claims are made. The country pairs share source/target countries and must not be treated as 110 independent replicates.

可用配对：a 98/110（77为正）；b 104/110（88为正）；c 95/110（48为正）；d 95/110（48为正）。
部分重复可用的配对数：a 1，b 6，c 30，d 25。缺失和负向结果全部显示，不据此筛选国家。

Country codes: AT Austria; BE Belgium; DE Germany; EE Estonia; LT Lithuania; LV Latvia; PT Portugal; RO Romania; SE Sweden; SI Slovenia; SK Slovakia.

绘图约定
Python/matplotlib；183×190 mm；Arial优先，最小字号5.5 pt；白底，低饱和顺序/双向色标，PDF/SVG可编辑文字，TIFF 600 dpi，PNG 300 dpi。两图分别为quantitative grid。原科研绘图样式仅作风格继承；矩阵按当前配对数据重新构建。

所有图只写独立实验目录，未加入论文，未修改冻结RSE或训练/分析数据。源数据CSV和绘图脚本支持复现。
