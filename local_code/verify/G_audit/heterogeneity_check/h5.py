import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from scipy import stats
sys.path.insert(0, '.')
import analyze_G as A
from guidance_G import E1
R = A.load_rows()
# 22% claim
for lab,mu,tau,se in (('R0 pooled',-0.0537,0.2056,0.0853),('REML own var',-0.0263,0.0944,0.0521)):
    s=np.sqrt(tau**2+se**2)
    pn = stats.norm.cdf((-0.3-mu)/s)+stats.norm.sf((0.3-mu)/s)
    pt = stats.t.cdf((-0.3-mu)/s,9)+stats.t.sf((0.3-mu)/s,9)
    ptau = stats.norm.cdf((-0.3-mu)/tau)+stats.norm.sf((0.3-mu)/tau)
    print(lab,'P(|D|>0.3) normal(tau+se) %.3f, t9 %.3f, tau-only %.3f'%(pn,pt,ptau))
# Q-profile CI for tau (pooled v known)
v=A.country_values(A.deltas(R,'AE','S2',E1)).to_numpy(); vw=A.within_var(A.deltas(R,'AE','S2',E1))
S=((v-v.mean())**2).sum()
for q in (0.975,0.025):
    c=stats.chi2.ppf(q,10); t2=max(0,S/c-vw); print('Q-profile tau bound (chi2 %.3f): %.3f'%(q,np.sqrt(t2)))
# F-based (v estimated with df=33): tau^2 = v*(F/Fcrit -1)
F=S/10/vw
for q in (0.975,0.025):
    c=stats.f.ppf(q,10,33); t2=max(0,vw*(F/c-1)); print('F-based tau bound (%.3f): %.3f'%(q,np.sqrt(t2)))
# SE typiclust diagnostics
g=R[(R.target=='SE')&(R.method.isin(['typiclust','srs','kcenter']))&(R.cond.isin(['AE','S2','none']))]
cols=['split','rep','cond','method','clf','k','ba_ge30','ba_P2','comp_L1','n_classes_sel']
print(g[cols].sort_values(['method','k','clf','split','rep','cond']).to_string(index=False))
