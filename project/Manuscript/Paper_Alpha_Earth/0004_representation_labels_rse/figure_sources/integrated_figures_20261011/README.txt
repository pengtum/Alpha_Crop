合并图复现与溯源

plot_dataset.py 重排原空间聚合、类别比例和类别总数，加入真实 Tielt 地块及标签实例。plot_concept.py 在上一版图 2 基础上加入同一地块的质心特征流程。

运行环境：/Users/perng_mini/AlphaCrops_local/.venv/bin/python -B；设置 MPLCONFIGDIR=/private/tmp/rse_matplotlib。out/ 保存 PDF、SVG、PNG 300 dpi 和 TIFF 600 dpi。PDF/SVG 保留可编辑文字。

只读源数据：../parcel_reference_20261011/source_data/；../regional_allocation/concept_20261008/；项目 label_expansion/storyline/rse_figs/revision_20261007/source_data/。figure_contract.json 定义证据角色；qa_report.json 记录最终核验和人工复查。

示例涵盖窗口内全部 39 个真实地块，P1 为 BE_VLG_456196。正射影像未改颜色和亮度；标签来自作物申报。欧洲类别数值来自完整 611,519 个配对样本。邻域网格为名义几何，概念特征空间和曲线不代表实测精度。
