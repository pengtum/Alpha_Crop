2026-10-10 独立分析修复

原冻结代码 label_expansion/single_source_v1 完全保留，不改 frozen.json，不重训。
修复 analyze.py 仅改变代码定位、独立输出目录和报告用 logit_abs_max 的显式数值转换。
原 CSV 用 keep_default_na=False 读取，RF 的空诊断使混合列为字符串；空值保留为 NaN，非空非法文本直接报错。
原训练参数、选择、预测、OA–BA函数、统计汇总及原分析输出均不修改。

输出：results/single_source_20261009/analysis_repaired_20261010

从项目根目录运行（Python 为 /Users/perng_mini/AlphaCrops_local/.venv/bin/python -B）：
1. check_inputs.py：验证源配置、空间隔离、元数据、冻结输入和RSE校验。
2. analyze.py：重新验证18,392份预测并完整分析；MPLCONFIGDIR=/private/tmp/alpha_mpl XDG_CACHE_HOME=/private/tmp/alpha_cache。
3. audit.py：核对原/新9份指标CSV逐字节一致、混淆矩阵与OA一致，生成补充诊断和完整REPORT.txt。
4. plot_summary.py：生成与原论文样式一致的独立证据摘要图和图注。

辅助检查曾在临时目录进行试运行；最终有效结果以独立输出目录中的validation.json为准。
prior_status.json保留原分析失败状态；全部验证成功后才更新实验status.json为complete。
报告中的国家权重区间为敏感性范围，不是110个独立重复的显著性检验。
所有负面、非单调、无法比较及数值诊断结果保留；是否纳入RSE由用户讨论决定。
