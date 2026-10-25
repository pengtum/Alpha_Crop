import numpy as np, pandas as pd
from scipy import stats
exec(open('/Users/perng_mini/AlphaCrops_local/verify/G_audit/refute/a2_main.py').read().split("d=deltas('AE','S2',E1); v=cv(d)")[0])
# SRS learning curve
print('=== SRS 学习曲线（单元均值，ba_ge30）===')
S=R[R.method.isin(['srs','foreign_only'])]
lc=S.groupby(['target','clf','k']).ba_ge30.mean().unstack('k')
print(lc.round(2).to_string())
g=lc.copy()
print('\n增益：k50-k0, k200-k50, k400-k200 （各国×分类器）')
gain=pd.DataFrame({'50-0':g[50]-g[0],'100-50':g[100]-g[50],'200-50':g[200]-g[50],'400-200':g[400]-g[200]})
print(gain.round(2).to_string()); print(gain.describe().round(2).to_string())
print('中位 400-200:', gain['400-200'].median(), ' 400-200 < 0.3 的比例:', (gain['400-200']<0.3).mean())
# method - SRS by guidance
print('\n=== 方法 − SRS（国家值 t）===')
srs=R[R.method=='srs'].set_index(UNIT+['clf','k']).ba_ge30
def vs(cond,methods,reps=None):
    a=R[(R.cond==cond)&R.method.isin(methods)].copy()
    if reps is not None: a=a[a.rep.isin(reps)]
    a['d']=a.ba_ge30.values-srs.reindex(pd.MultiIndex.from_frame(a[UNIT+['clf','k']])).values
    return a.dropna(subset=['d'])
for cond in ['AE','S2']:
    for m in E1+['margin','predstrat_srs','gist_margin','badge_c']:
        dd=vs(cond,[m]); show(f'{cond} {m} - SRS', ts(cv(dd)))
    show(f'{cond} E1 - SRS', ts(cv(vs(cond,E1))))
for cond in ['ORACLE','XY','S2XY']:
    for m in ['lpm2_nogeo','typiclust','margin','kcenter']:
        dd=vs(cond,[m])
        if len(dd): show(f'{cond} {m} - SRS', ts(cv(dd)))
print('\n=== ORACLE − S2 / ORACLE − AE，逐方法（rep0）===')
for m in ['lpm2_nogeo','typiclust','margin']:
    show(f'ORACLE-S2 {m}', ts(cv(deltas('ORACLE','S2',[m],reps=[0]))))
    show(f'ORACLE-AE {m}', ts(cv(deltas('ORACLE','AE',[m],reps=[0]))))
    show(f'AE-S2 {m} rep0', ts(cv(deltas('AE','S2',[m],reps=[0]))))
# noise floor: SRS rep0 vs rep1 (same split, different pool+seed)
print('\n=== 噪声基线 ===')
sr=R[R.method=='srs'].pivot_table(index=['target','split','clf','k'],columns='rep',values='ba_ge30')
dsr=(sr[0]-sr[1])
d=deltas('AE','S2',E1)
print('cell SD of AE-S2 (E1):', d.d.std().round(3), ' mean |Δ|:', d.d.abs().mean().round(3))
print('SD of SRS rep0-rep1 (k50/200):', dsr[dsr.index.get_level_values('k').isin([50,200])].std().round(3), ' /sqrt2 =', (dsr[dsr.index.get_level_values('k').isin([50,200])].std()/np.sqrt(2)).round(3))
# SD across methods within guidance and SD of method-SRS
for cond in ['AE','S2']:
    x=vs(cond,E1)
    print(cond,'cell SD of method-SRS:',x.d.std().round(3),' mean:',x.d.mean().round(3))
# within unit/clf/k: range of accuracy across 8 selections (4 methods x 2 guidances) + srs
X=R[(R.cond.isin(['AE','S2'])&R.method.isin(E1))|(R.method=='srs')]
X=X[X.k.isin([50,200])]
sp=X.groupby(UNIT+['clf','k']).ba_ge30.agg(['std','min','max','count'])
print('\n同一 (单元, 分类器, k) 内 9 种选样（4方法×2指导+SRS）的精度 SD 中位数:', sp['std'].median().round(3), ' 极差中位数:', (sp['max']-sp['min']).median().round(3))
print(sp['count'].value_counts())
# Compare: k effect variance vs selection variance
# Variance decomposition of ba_ge30 across E1 AE/S2 rows at fixed unit,clf,k: method vs guidance vs interaction
Y=R[R.cond.isin(['AE','S2'])&R.method.isin(E1)].copy()
Y['cell']=Y[UNIT+['clf','k']].astype(str).agg('|'.join,axis=1)
Y['r']=Y.ba_ge30-Y.groupby('cell').ba_ge30.transform('mean')
mm=Y.groupby(['cell','method']).r.transform('mean'); gg=Y.groupby(['cell','cond']).r.transform('mean')
print('within-cell var total', Y.r.var().round(3), ' method part', mm.var().round(3), ' guidance part', gg.var().round(3), ' resid(int)', (Y.r-mm-gg).var().round(3))
