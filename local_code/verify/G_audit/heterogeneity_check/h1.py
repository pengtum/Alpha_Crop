import sys, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, '.')
import analyze_G as A
from guidance_G import E1
R = A.load_rows()
d = A.deltas(R, 'AE', 'S2', E1)
print('n cells', len(d), d.groupby('target').size().to_dict())
v = A.country_values(d)
print(v.round(3).to_dict())
u = A.unit_level(d)
print(u.unstack(['split','rep']).round(3))
# within var pooled
vw = A.within_var(d)
print('pooled within var', vw, 'sd', np.sqrt(vw))
ug = u.groupby('target')
vc = (ug.var(ddof=1)/ug.size())
print('country-specific within var:', vc.round(4).to_dict())
def Qstat(y, vv):
    w = 1/vv; mu=(w*y).sum()/w.sum(); Q=(w*(y-mu)**2).sum(); k=len(y)
    p = stats.chi2.sf(Q, k-1); I2=max(0,(Q-(k-1))/Q)
    return Q,p,I2,mu
y=v.to_numpy()
print('Q pooled:', Qstat(y, np.full(len(y), vw)))
print('Q country-specific:', Qstat(y, vc.reindex(v.index).to_numpy()))
# F-test version (one-way ANOVA on unit values, country effect): equivalent to pooled
grp=[g.to_numpy() for _,g in u.groupby('target')]
print('one-way ANOVA units ~ country:', stats.f_oneway(*grp))
# Kruskal
print('Kruskal:', stats.kruskal(*grp))
# PI
print(A.prediction_interval(y, np.full(len(y), vw)))
print(A.prediction_interval(y, vc.reindex(v.index).to_numpy()))
