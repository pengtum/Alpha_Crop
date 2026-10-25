import numpy as np, pandas as pd
from scipy import stats
exec(open('/Users/perng_mini/AlphaCrops_local/verify/G_audit/refute/a2_main.py').read().split("d=deltas('AE','S2',E1); v=cv(d)")[0])
import sys; sys.path.insert(0,'.')
from analyze_G import reml_tau2, prediction_interval, within_var
d=deltas('AE','S2',E1); v=cv(d)
u=d.groupby(UNIT).d.mean()
vc=(u.groupby('target').var(ddof=1)/4).reindex(v.index)
vp=float(vc.mean())
print('pooled within var', vp, 'sqrt', np.sqrt(vp))
print('country within SE', np.sqrt(vc).round(3).to_dict())
pi=prediction_interval(v.values,np.full(11,vp)); print('R0 pooled', {k:round(x,3) for k,x in pi.items()})
pi2=prediction_interval(v.values,vc.values); print('R0 own', {k:round(x,3) for k,x in pi2.items()})
# Q test
for lab,var in [('pooled',np.full(11,vp)),('own',vc.values)]:
    w=1/var; mu=(w*v.values).sum()/w.sum(); Q=(w*(v.values-mu)**2).sum(); p=stats.chi2.sf(Q,10)
    I2=max(0,(Q-10)/Q)
    print(f'Q({lab})={Q:.2f} p={p:.4f} I2={I2:.2f}')
# DerSimonian-Laird tau
# proportion of countries |v|>0.3
print('|country|>0.3:', (v.abs()>0.3).sum(), v[v.abs()>0.3].round(3).to_dict())
# method x country interaction test: unit-level (country, method, unit) means over clf x k
um=d.groupby(['target','method','split','rep']).d.mean().reset_index()
# two-way anova with replication (4 units per cell)
gm=um.d.mean()
cm=um.groupby('target').d.transform('mean'); mm=um.groupby('method').d.transform('mean'); cellm=um.groupby(['target','method']).d.transform('mean')
SSint=((cellm-cm-mm+gm)**2).sum(); SSres=((um.d-cellm)**2).sum(); SSc=((cm-gm)**2).sum(); SSm=((mm-gm)**2).sum()
dfint=(11-1)*(4-1); dfres=len(um)-44; dfc=10; dfm=3
F=(SSint/dfint)/(SSres/dfres); print(f'country x method interaction F={F:.2f} df=({dfint},{dfres}) p={stats.f.sf(F,dfint,dfres):.4f}')
Fc=(SSc/dfc)/(SSres/dfres); print(f'country main F={Fc:.2f} p={stats.f.sf(Fc,dfc,dfres):.4f}')
Fm=(SSm/dfm)/(SSres/dfres); print(f'method main F={Fm:.2f} p={stats.f.sf(Fm,dfm,dfres):.4f}')
# caveat: units within country shared across methods -> residual not indep across methods; alternative: country main with unit-level error
uu=d.groupby(UNIT).d.mean().reset_index()
cm2=uu.groupby('target').d.transform('mean'); SSb=((cm2-uu.d.mean())**2).sum(); SSw=((uu.d-cm2)**2).sum()
F2=(SSb/10)/(SSw/33); print(f'country effect on unit-level values (one-way ANOVA): F={F2:.2f} p={stats.f.sf(F2,10,33):.4f}')
# Kruskal
print('Kruskal country:', stats.kruskal(*[g.d.values for _,g in uu.groupby('target')]))
# typiclust SE, kcenter BE, kherding RO detail
for c,m in [('SE','typiclust'),('BE','kcenter'),('RO','kherding'),('PT','typiclust'),('PT','kcenter'),('PT','lpm2_nogeo')]:
    x=d[(d.target==c)&(d.method==m)].pivot_table(index=['split','rep'],columns=['clf','k'],values='d')
    print(c,m); print(x.round(2).to_string())
