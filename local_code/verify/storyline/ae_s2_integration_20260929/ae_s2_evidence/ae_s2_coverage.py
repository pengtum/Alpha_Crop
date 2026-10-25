"""
只读（不训练）：dataset_v2 中 AE / S2 的可用性（S2 缺失是否按国家、类别选择性），以及地块面积相对 30 m 邻域核的比例。
运行：/Users/perng_mini/AlphaCrops_local/.venv/bin/python -B ae_s2_coverage.py
"""
import pandas as pd
d = pd.read_parquet('/Users/perng_mini/AlphaCrops_local/data/dataset_v2.parquet',
                    columns=['country', 'label', 'area_ha', 'has_ae', 'has_s2'])
print('总地块', len(d))
t = d.groupby('country').agg(n=('has_ae', 'size'), ae=('has_ae', 'mean'), s2=('has_s2', 'mean'),
                             both=('has_ae', lambda s: (s & d.loc[s.index, 'has_s2']).mean()))
print(t.round(4).to_string())
x = d[d.has_ae]
m = x.groupby(['country', 'label']).has_s2.mean().unstack().round(3)
print('\n有 AE 的地块中有 S2 的比例（国家 × 类别；<0.9 的格标出）')
print(m.to_string())
low = m.stack()
print('\n<0.90 的格：', low[low < 0.9].round(3).to_dict())
b = d[d.has_ae & d.has_s2]
print('\n进入分析的地块（AE∩S2）：', len(b))
print('面积 <0.29 ha（约等于 30 m 圆形核的覆盖面积）比例：%.3f；<0.9 ha（Cheng 的面积门槛）比例：%.3f；中位面积 %.3f ha'
      % ((b.area_ha < 0.29).mean(), (b.area_ha < 0.9).mean(), b.area_ha.median()))
