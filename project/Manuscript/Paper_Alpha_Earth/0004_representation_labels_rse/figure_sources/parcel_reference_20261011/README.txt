地块参考标签与质心特征提取：两张候选配图

第一张：真实地块边界及其作物标签。
同一地区并排展示官方正射影像与类别着色地块，以一个可追溯的冬小麦地块说明申报名称、EuroCrops 统一类别和本文分析标签的关系。

第二张：质心和邻域统计。
使用同一地块的实际边界和生产流水线保存的质心，说明每个输入通道的邻域均值、标准差及最终维度。网格明确为示意，未把名义 30 m 圆当作 Earth Engine 实际像元配准的重建，也未画成整块田的面平均。

建议排布
第一张可作为现有图 1 的局部地块放大示例；地图与不均衡统计仍由现有图 1 承担。第二张适合数据预处理小节，或在正文图件过多时作为补充图。现有图 2 继续负责迁移实验设计。
本轮先输出两张独立候选图，不改论文编号与正文布局。

文件说明
out：PDF、可编辑 SVG、PNG 预览、600 dpi TIFF 与 QA 报告。
source_data：示例地区地块、原始属性、选中地块的实际缓存特征、官方影像和元数据。
prepare_example.py：可追溯地准备示例数据；支持从官方公开压缩包读入以避免云占位文件。
plot_figures.py：继承既有论文样式绘图。
captions.txt：英文图注。
figure_contract.json：示例选择、数据含义、导出与完整性约束。

数据和方法边界
影像来自官方航空正射影像服务，不是 Sentinel-2，也不作为实验输入或标签来源。
标签来自农户申报记录，不能把所有记录解释为逐地块现场独立核验。
该地区按地理位置选作说明，不以模型预测性能筛选，也不代表欧洲或全国类别比例。
没有新增训练、重算特征或修改原始记录。

公开来源
EuroCrops v11：https://zenodo.org/records/14094196
Digitaal Vlaanderen：https://www.vlaanderen.be/datavindplaats/catalogus/orthofotomozaiek-middenschalig-winteropnamen
2021影像层：OMWRGB21VL，https://geo.api.vlaanderen.be/OMW/wms
