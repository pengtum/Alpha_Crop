"""
只读（不训练）：同 OA 的 AE−S2（Z0，W4_min）与国家协变量的 Spearman（描述，朴素 p，未校正）。
依赖本目录 matched_oa_units.csv 与 country_covariates.csv（先运行 ae_s2_country.py 与 ae_s2_matched_oa.py）。
"""
import pandas as pd
from scipy import stats
C = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']
C9 = [c for c in C if c not in ('RO', 'PT')]
U = pd.read_csv('matched_oa_units.csv').groupby('target').W4_min_d.mean().loc[C]
cov = pd.read_csv('country_covariates.csv', index_col=0).loc[C]
preds = ['mean_dlat_to_pool', 'min_dlat_to_pool', 'comp_L1_vs_foreign', 'grass_share_test', 'knn10_ae_minus_s2',
         'sep_ratio_ae', 'sep_ratio_ae_over_s2', 'foreign_in_rate_ae', 'mmd_ae_mean', 'tgt300_ba_ae_minus_s2',
         'domestic_ba_ae_minus_s2', 'median_area_ha', 'share_lt_0p29ha', 'ae_mean_norm_median', 'lat', 'lat_span_p10_p90']
print('同 OA 的 AE−S2（Z0，W4_min）逐国：', U.round(2).to_dict())
for p in preds:
    r = stats.spearmanr(cov[p], U); r9 = stats.spearmanr(cov.loc[C9, p], U.loc[C9])
    print(f'  {p:26s} ρ11={r[0]:+.2f} (p={r[1]:.3f})  ρ9={r9[0]:+.2f} (p={r9[1]:.3f})')
