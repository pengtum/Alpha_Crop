# 只读：由 fa_rows_B2.csv 重算 ΔFA(RT200)=FA(RT200)−FA(Z0) 的 AE−S2 逐国值（新网格），看 9/11 国翻号由谁驱动；给截尾均值与符号翻转 p
import os; os.environ["OMP_NUM_THREADS"]="2"
import itertools, numpy as np, pandas as pd
from scipy import stats
C=['AT','BE','DE','EE','LT','LV','PT','RO','SE','SI','SK']; C9=[c for c in C if c not in ('RO','PT')]
fr=pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/step0/partB2/fa_rows_B2.csv')
fr=fr[fr.grid.astype(str).str.contains('new')] if fr.grid.astype(str).str.contains('new').any() else fr
print('grid 值：',fr.grid.unique()[:5])
def sf(v):
    v=np.asarray(v); obs=abs(v.mean()); return np.mean([abs((v*np.array(s)).mean())>=obs-1e-12 for s in itertools.product([1,-1],repeat=len(v))])
def tsum(v):
    v=pd.Series(v).dropna(); n=len(v); m=v.mean(); se=v.std(ddof=1)/np.sqrt(n); h=stats.t.ppf(.975,n-1)*se
    return f"{m:+.2f} [{m-h:+.2f}, {m+h:+.2f}] >0 {int((v>0).sum())}/{n}"
out=[]
p=fr.pivot_table(index=['target','unit','clf'],columns='arm',values='FA')
for arm in ['RT200','PK200','HD200']:
    if arm not in p.columns: continue
    d=(p[arm]-p['Z0']).unstack('clf'); d=(d['ae']-d['s2']).groupby('target').mean().reindex(C)
    out.append(f"ΔFA({arm}) AE−S2 逐国：{d.round(2).to_dict()}")
    out.append(f"   11国 t {tsum(d)}；截尾均值 {np.sort(d.values)[1:-1].mean():+.2f}；中位 {d.median():+.2f}；符号翻转 p {sf(d.values):.3f} | 9国 t {tsum(d.loc[C9])}")
txt="\n".join(out); print(txt)
open('/private/tmp/ae_s2_methods_review_20260929/interaction_by_country_out.txt','w').write(txt+"\n")
