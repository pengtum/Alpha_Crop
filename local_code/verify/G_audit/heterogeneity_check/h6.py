import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from scipy import stats
sys.path.insert(0, '.')
import analyze_G as A
from guidance_G import E1
R = A.load_rows()
d = A.deltas(R, 'AE', 'S2', E1)
v = A.country_values(d)
def s(v): x=A.t_summary(v); return '%+.3f 90%%[%+.3f,%+.3f] %s'%(x['mean'],x['lo90'],x['hi90'],A.verdict(x))
print('full', s(v))
for c in v.index: print(' drop', c, s(v.drop(c)))
for m in E1:
    print(' drop method', m, s(A.country_values(d[d.method!=m])))
    print(' only method', m, s(A.country_values(d[d.method==m])))
# drop SE typiclust cell only
dd = d[~((d.target=='SE')&(d.method=='typiclust'))]
print(' drop SE-typiclust cell', s(A.country_values(dd)))
# PT unit-level t
u=A.unit_level(d)
for c in ('PT','SE','RO'):
    x=u.loc[c].to_numpy(); t=x.mean()/(x.std(ddof=1)/2); print(c, x.round(3), 't=%.2f p=%.3f'%(t, 2*stats.t.sf(abs(t),3)))
# country-level per-country t, count p<0.05
ps=[]
for c in v.index:
    x=u.loc[c].to_numpy(); t=x.mean()/(x.std(ddof=1)/2); ps.append(2*stats.t.sf(abs(t),3))
print('countries p<0.05 (4-unit t):', sum(p<0.05 for p in ps), dict(zip(v.index, np.round(ps,3))))
