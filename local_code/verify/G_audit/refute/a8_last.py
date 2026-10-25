import numpy as np, pandas as pd
from scipy import stats
exec(open('/Users/perng_mini/AlphaCrops_local/verify/G_audit/refute/a2_main.py').read().split("d=deltas('AE','S2',E1); v=cv(d)")[0])
import sys; sys.path.insert(0,'.')
import analyze_G as A
d=deltas('AE','S2',E1); v=cv(d)
ins=['SE','EE','LV','LT','DE']; sens=['AT','SK','BE','SI','PT','RO']
show('label-insensitive 5 [探索]', ts(v[ins])); show('label-sensitive 6 [探索]', ts(v[sens]))
print('sign test 0/6:', stats.binomtest(0,6).pvalue)
# fragility: PT value needed to flip lo90 below -0.3
for x in np.arange(-0.7,-2.5,-0.05):
    vv=v.copy(); vv['PT']=x; s=ts(vv)
    if s['lo90']<-0.3: print('PT value flipping TOST:', round(x,2)); break
for x in np.arange(0.48,3,0.05):
    vv=v.copy(); vv['SE']=x; s=ts(vv)
    if s['hi90']>0.3: print('SE value flipping TOST upper:', round(x,2)); break
# implied prob new country |delta|>0.3 under R0 (t, df=9)
mu,tau,se=-0.054,0.206,0.085; sd=np.sqrt(tau**2+se**2)
print('P(new<-0.3)=',stats.t.cdf((-0.3-mu)/sd,9),' P(new>0.3)=',stats.t.sf((0.3-mu)/sd,9))
print('WCB p', A.webb_wcb(v.values))
# SRS country-level placebo: SRS rep0 - rep1 country means (over split, clf, k in 50/200)
sr=R[(R.method=='srs')&R.k.isin([50,200])].pivot_table(index=['target','split','clf','k'],columns='rep',values='ba_ge30')
pl=(sr[0]-sr[1]).groupby('target').mean()
print('SRS rep0-rep1 country means (placebo-ish, includes pool change):', pl.round(2).to_dict(), ' SD', pl.std().round(3))
print('country theta SD', v.std().round(3))
# per-country fraction of cells where |Δ|>0.3
print('cell |Δ|>0.3 fraction:', (d.d.abs()>0.3).mean().round(3), '  |Δ|>1:', (d.d.abs()>1).mean().round(3))
