import sys
import numpy as np, pandas as pd
from scipy import stats
sys.argv=[sys.argv[0]]
import analyze_G as A
d = pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/G_audit/verify_pred_loo/pred_cells.csv')
v = A.country_values(d)
wv = A.within_var(d)
pi = A.prediction_interval(v.to_numpy(), np.full(len(v), wv))
print('pooled within var', round(wv,4), 'REML/HKSJ pooled:', {k: round(x,3) for k,x in pi.items()})
mu, se = pi['mu'], pi['se_hksj']; q=stats.t.ppf(0.975,10)
print('  mu 95%% CI HKSJ [%.3f, %.3f], p=%.4f' % (mu-q*se, mu+q*se, 2*stats.t.sf(abs(mu/se),10)))
u = A.unit_level(d).groupby('target')
vc = (u.var(ddof=1)/u.size()).reindex(v.index)
pm = A.prediction_interval(v.to_numpy(), vc.to_numpy())
print('REML per-country var:', {k: round(x,3) for k,x in pm.items()})
mu, se = pm['mu'], pm['se_hksj']
print('  mu 95%% CI HKSJ [%.3f, %.3f], p=%.4f' % (mu-q*se, mu+q*se, 2*stats.t.sf(abs(mu/se),10)))
# LOO for predstrat_srs alone and margin alone
for m in ['predstrat_srs','margin']:
    vm = A.country_values(d[d.method==m])
    ps = [A.t_summary(vm.drop(c))['p'] for c in vm.index]
    print(m, 'full p', round(A.t_summary(vm)['p'],4), 'LOO max p', round(max(ps),4), 'LOO p>0.05 count', sum(p>0.05 for p in ps))
# Huber M-estimate of location as robust check
from scipy.stats import bootstrap
x = v.to_numpy()
def hodges_lehmann(x):
    w = [(x[i]+x[j])/2 for i in range(len(x)) for j in range(i,len(x))]
    return np.median(w)
print('Hodges-Lehmann', round(hodges_lehmann(x),3))
# Compare fragility of geom pathway for context: LOO p of theta_E1
from guidance_G import E1, FAMILIES
R = A.load_rows()
for name in ['geom','hyb']:
    vv = A.country_values(A.deltas(R,'AE','S2',FAMILIES[name], reps=[0] if name!='geom' else None))
    print(name, 'LOO mean range', round(min(A.t_summary(vv.drop(c))['mean'] for c in vv.index),3), round(max(A.t_summary(vv.drop(c))['mean'] for c in vv.index),3))
# pred pathway with k split and clf split
for k in (50,200):
    s=A.t_summary(A.country_values(d[d.k==k])); print('pred k=%d: %+.3f p=%.3f pos %d' % (k,s['mean'],s['p'],s['n_pos']))
for c in ('ae','s2'):
    s=A.t_summary(A.country_values(d[d.clf==c])); print('pred clf=%s: %+.3f p=%.3f pos %d' % (c,s['mean'],s['p'],s['n_pos']))
