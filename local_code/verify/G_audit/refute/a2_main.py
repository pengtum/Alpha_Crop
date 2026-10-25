import numpy as np, pandas as pd
from scipy import stats
R=pd.read_pickle('/Users/perng_mini/AlphaCrops_local/verify/G_audit/refute/R.pkl')
E1=['lpm2_nogeo','kcenter','typiclust','kherding']
UNIT=['target','split','rep']; KEY=UNIT+['method','clf','k']
def deltas(a,b,methods,metric='ba_ge30',reps=None):
    A=R[(R.cond==a)&R.method.isin(methods)].set_index(KEY)[metric]
    B=R[(R.cond==b)&R.method.isin(methods)].set_index(KEY)[metric]
    d=(A-B).dropna().rename('d').reset_index()
    if reps is not None: d=d[d.rep.isin(reps)]
    return d
def cv(d): return d.groupby(['target','method','clf','k']).d.mean().groupby('target').mean()
def ts(v):
    v=np.asarray(v,float); n=len(v); m=v.mean(); se=v.std(ddof=1)/np.sqrt(n)
    q95,q90=stats.t.ppf(.975,n-1),stats.t.ppf(.95,n-1)
    return dict(n=n,mean=m,se=se,lo95=m-q95*se,hi95=m+q95*se,lo90=m-q90*se,hi90=m+q90*se,p=2*stats.t.sf(abs(m/se),n-1),npos=int((v>0).sum()))
def verdict(s):
    if s['lo95']>0: return 'AE更好'
    if s['hi95']<0: return 'S2更好'
    if s['lo90']>-.3 and s['hi90']<.3: return '等价'
    return '不确定'
def show(lab,s): print(f"{lab:38s} {s['mean']:+.3f} 95%[{s['lo95']:+.3f},{s['hi95']:+.3f}] 90%[{s['lo90']:+.3f},{s['hi90']:+.3f}] p={s['p']:.3f} n+={s['npos']}/{s['n']} -> {verdict(s)}")
d=deltas('AE','S2',E1); v=cv(d); s=ts(v); show('theta_E1',s); print(v.round(3).to_dict())
print('n delta per country', d.groupby('target').size().to_dict())
# TOST p-values
se=s['se']; df=10
p_lo=stats.t.sf((s['mean']+.3)/se,df); p_hi=stats.t.cdf((s['mean']-.3)/se,df)
print('TOST p lower, upper', p_lo, p_hi, 'max', max(p_lo,p_hi))
# minimal equivalence margin achievable: max(|lo90|,|hi90|)
print('smallest margin for TOST pass (90%):', max(-s['lo90'], s['hi90']))
print('margin at 95% two-sided (TOST alpha=.025):', max(-(s['mean']-stats.t.ppf(.975,10)*se), s['mean']+stats.t.ppf(.975,10)*se))
# per country within CI: unit-level (4 units, df=3) and cell-level
print('\n=== 各国国内：单元值 (4 单元) t(df=3) CI 与格子层面 ===')
u=d.groupby(UNIT).d.mean().reset_index()
rows=[]
for c,g in u.groupby('target'):
    x=g.d.values; m=x.mean(); sd=x.std(ddof=1); se=sd/2
    q=stats.t.ppf(.975,3)
    dc=d[d.target==c].d
    rows.append(dict(c=c,mean=m,unit_vals=np.round(x,2).tolist(),lo95=m-q*se,hi95=m+q*se,p=2*stats.t.sf(abs(m/se),3),
                     cell_sd=dc.std(),cell_pos=(dc>0).mean(),n_cells=len(dc)))
print(pd.DataFrame(rows).round(3).to_string())
# method x country
print('\n=== 方法 × 国家 国家值 ===')
mc=d.groupby(['target','method','clf','k']).d.mean().groupby(['target','method']).mean().unstack()
print(mc.round(2).to_string())
print('\nmethod mean, sd over countries, n>0, 90% CI:')
for m in E1:
    ss=ts(mc[m]); show(m,ss)
# clf x k
print('\n=== 分类器 × k ===')
for clf in ['s2','ae']:
    for k in [50,200]:
        show(f'clf={clf} k={k}', ts(cv(d[(d.clf==clf)&(d.k==k)])))
for k in [50,200]:
    for m in E1:
        show(f'k={k} {m}', ts(cv(d[(d.k==k)&(d.method==m)])))
# split/rep
print('\n=== 划分 / 重复 ===')
for col in ['split','rep']:
    for val in [0,1]:
        show(f'{col}={val}', ts(cv(d[d[col]==val])))
# leave one out
print('\n=== 留一国 ===')
for c in v.index:
    show(f'drop {c}', ts(v.drop(c)))
show('drop PT+SE', ts(v.drop(['PT','SE'])))
# metrics
print('\n=== 不同指标 ===')
for met in ['ba_ge30','oa','f1_ge30','ba_P1','oa_P1','ba_P2','oa_P2','ba_far','oa_far','ba_i200','ba_i250','oa_i200','oa_i250','f1_i200','f1_i250']:
    show(met, ts(cv(deltas('AE','S2',E1,met))))
