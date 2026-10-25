"""用原审查者的 V2 复训行（只含 LV/PT/RO）复算其 H_V2 数字；并展示"只替换 3 国"的混合性质。只读。"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import analyze_G as A
OUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f'{OUT}/rows_t*_*.csv'))], ignore_index=True)
V = pd.concat([pd.read_csv(p) for p in glob.glob('/Users/perng_mini/AlphaCrops_local/verify/G_audit/oracle/a6_retrain_V2_*.csv')])
V = V.rename(columns={'tgt': 'target'})
key = ['target', 'split', 'method', 'k', 'clf']
R2 = R.copy()
m = R2.cond.eq('ORACLE') & R2.rep.eq(0)
Vi = V.set_index(key).ba_ge30
idx = R2[m].set_index(key).index
hit = idx.isin(Vi.index)
old = R2.loc[m, 'ba_ge30'].values.copy()
new = old.copy(); new[hit] = Vi.reindex(idx[hit]).values
R2.loc[m, 'ba_ge30'] = new
print('替换行数', hit.sum())
d = pd.DataFrame({'old': old[hit], 'new': new[hit]}, index=idx[hit])
print('V2 − 原（ba_ge30）按国家×方法：'); print((d.new - d.old).groupby(['target', 'method']).mean().round(3).to_string())
for lab, RR in (('原始', R), ('V2 替换 3 国', R2)):
    for ms in (['lpm2_nogeo', 'typiclust', 'margin'], ['typiclust'], ['lpm2_nogeo']):
        s = A.t_summary(A.country_values(A.deltas(RR, 'ORACLE', 'S2', ms, reps=[0])))
        print(lab, ms, f"{s['mean']:+.3f} [{s['lo95']:+.3f}, {s['hi95']:+.3f}] p={s['p']:.3f} {s['n_pos']}/11")
cv = A.country_values(A.deltas(R2, 'ORACLE', 'S2', ['typiclust'], reps=[0]))
print('V2 替换后 typiclust 各国：', cv.round(3).to_dict())
