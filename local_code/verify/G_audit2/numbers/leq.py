import sys
sys.path.insert(0, '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion')
sys.argv=['x']
import numpy as np, pandas as pd
from scipy import stats
import analyze_G as A
from guidance_G import E1
R=A.load_rows()
sE1=A.t_summary(A.country_values(A.deltas(R,'AE','S2',E1)))
names={'θ_E1':sE1['mean'],'L95':sE1['lo95'],'U95':sE1['hi95'],'+0.3':0.3,'−0.3':-0.3}
La=A.label_equiv_detail(R,names,A.label_equiv_anchored)
cs=A.censored_summary(La); print(cs.to_string())
Lp=A.label_equiv_detail(R,names,A.label_equiv); print(A.censored_summary(Lp).to_string())
# SRS gains per curve
S=R[R.method.isin(['srs','foreign_only'])]
g=S.pivot_table(index=['target','clf','split','rep'],columns='k',values='ba_ge30')
print(g.columns.tolist())
gain=(g[400]-g[50]).groupby(['target','clf']).agg(['mean','std','count'])
gain['se']=gain['std']/np.sqrt(gain['count']); gain['t']=gain['mean']/gain['se']; gain['p']=2*stats.t.sf(gain.t.abs(),gain['count']-1)
print(gain.round(3).to_string())
g0=(g[50]-g[0]).groupby(['target','clf']).mean(); print('k0->50 median', g0.median())
gg=g.groupby(['target','clf']).mean(); print('50->400 median', (gg[400]-gg[50]).median(), 'mean', (gg[400]-gg[50]).mean())
nonmono=(gg[[0,50,100,200,400]].diff(axis=1).iloc[:,1:]<0).any(axis=1); print('nonmono', nonmono.sum(), nonmono[nonmono].index.tolist())
