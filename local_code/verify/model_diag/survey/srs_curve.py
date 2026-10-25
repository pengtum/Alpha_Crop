# 只读：由 G 的 rows_t1 计算 SRS 学习曲线（国家×分类器，4 个单元平均）
import pandas as pd, glob, numpy as np
d = pd.concat([pd.read_csv(f) for f in glob.glob('/Users/perng_mini/AlphaCrops_local/results/guidance_G/rows_t1_*.csv')])
s = d[d.method.isin(['srs', 'foreign_only'])]
c = s.groupby(['target', 'clf', 'k']).ba_ge30.mean().unstack('k')
g = pd.DataFrame({'d0_50': c[50]-c[0], 'd50_400': c[400]-c[50], 'd0_400': c[400]-c[0]})
print(g.round(2).to_string())
print('median over 22 curves:\n', g.median().round(2).to_string())
print('mean:\n', g.mean().round(2).to_string())
o = s.groupby(['target','clf','k']).oa.mean().unstack('k')
print('OA d0_400 median', round((o[400]-o[0]).median(),2), 'ba mean by clf/k:')
print(s.groupby(['clf','k'])[['ba_ge30','oa']].mean().round(2).to_string())
print('t_train sec median by clf:', d.groupby('clf').t_train.median().round(1).to_dict(), 'n rows', len(d))
