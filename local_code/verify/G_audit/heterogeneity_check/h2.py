import sys, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, '.')
import analyze_G as A
from guidance_G import E1
R = A.load_rows()
d = A.deltas(R, 'AE', 'S2', E1)
u = A.unit_level(d).rename('u').reset_index()
# nested ANOVA: country / split / rep
gm = u.u.mean()
c = u.groupby('target').u.mean()
cs = u.groupby(['target','split']).u.mean()
nC=11
SSc = 4*((c-gm)**2).sum()
SSs = 2*((cs - c.reindex(cs.index.get_level_values(0)).to_numpy())**2).sum()
SSr = ((u.set_index(['target','split']).u - cs.reindex(pd.MultiIndex.from_frame(u[['target','split']])).to_numpy())**2).sum()
MSc, MSs, MSr = SSc/10, SSs/11, SSr/22
print('MS country %.4f  MS split(c) %.4f  MS rep(split) %.4f'%(MSc,MSs,MSr))
print('F country vs split(c): %.3f p=%.4f'%(MSc/MSs, stats.f.sf(MSc/MSs,10,11)))
print('F split(c) vs rep: %.3f p=%.4f'%(MSs/MSr, stats.f.sf(MSs/MSr,11,22)))
print('F country vs pooled units (=report approach): %.3f p=%.4f'%(MSc/((SSs+SSr)/33), stats.f.sf(MSc/((SSs+SSr)/33),10,33)))
# variance components
s2_r = MSr; s2_s = max(0,(MSs-MSr)/2); s2_c = max(0,(MSc-MSs)/4)
print('var comp: rep %.4f split %.4f country %.4f ; tau_c=%.3f'%(s2_r,s2_s,s2_c,np.sqrt(s2_c)))
# correct var of country mean = s2_s/2 + s2_r/4
vtrue = s2_s/2 + s2_r/4
print('var of country mean (nested) %.4f vs pooled var/4 %.4f'%(vtrue, A.within_var(d)))
v = A.country_values(d).to_numpy()
w=1/vtrue; Q=((v-v.mean())**2).sum()/vtrue
print('Q with nested var: %.2f p(chi2)=%.4f'%(Q, stats.chi2.sf(Q,10)))
pi = A.prediction_interval(v, np.full(11, vtrue)); print('PI nested var', {k:round(x,3) for k,x in pi.items()})
# leave-one-out heterogeneity: drop PT, drop SE, drop both
vv = A.country_values(d)
for drop in (['PT'],['SE'],['PT','SE'],['RO']):
    ud = u[~u.target.isin(drop)]
    grp=[g.u.to_numpy() for _,g in ud.groupby('target')]
    print('drop',drop,'oneway F', stats.f_oneway(*grp))
