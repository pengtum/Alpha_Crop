import sys; sys.path.insert(0,'.')
from load import ci, fmt
import numpy as np, pandas as pd
from scipy import stats
pd.set_option('display.width',250)
d=pd.read_csv('/Users/perng_mini/AlphaCrops_local/results/stageA/stageA_merged_interim_PT.csv')
print(d.method.nunique(), sorted(d.method.unique())); print(d.groupby(['k','r']).size())
d=d[d.r==50] if 50.0 in d.r.unique() else d
key=['target','split','rep','k']
s=d[d.method=='srs'].set_index(key)
x=d[d.method!='srs'].copy(); kk=pd.MultiIndex.from_frame(x[key])
for m in ['ba_ge30','oa','comp_L1','ba_P2','oa_P2','n_classes_sel']: x['d_'+m]=x[m].to_numpy()-s[m].reindex(kk).to_numpy()
T=x.groupby(['k','method'])[['d_ba_ge30','d_oa','d_ba_P2','d_oa_P2','d_comp_L1','comp_L1','d_n_classes_sel']].mean()
for k in sorted(x.k.unique()):
    t=T.xs(k).sort_values('d_ba_ge30')
    print(f'\nk={k}'); print(t.round(2).to_string())
    r1=stats.spearmanr(t.d_comp_L1,t.d_ba_ge30); r2=stats.spearmanr(t.d_comp_L1,t.d_oa)
    print(f'  方法层 Spearman：Δcomp_L1 vs ΔBA ρ={r1.statistic:+.2f} (p={r1.pvalue:.3f})；vs ΔOA ρ={r2.statistic:+.2f} (p={r2.pvalue:.3f})')
T.to_csv('a8_stageA_PT.csv')
