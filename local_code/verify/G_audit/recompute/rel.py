import pandas as pd, numpy as np, glob
from scipy import stats
exec(open('recompute.py').read().split("print('=== H1')")[0])
key=['target','unit','clf','k']
srs=df[(df.method=='srs')&df.k.isin(K)][key+['ba_ge30']].rename(columns={'ba_ge30':'srs'})
out={}
for c in ['AE','S2']:
    a=df[(df.cond==c)&df.method.isin(E1)&df.k.isin(K)].merge(srs,on=key,how='left')
    assert a.srs.notna().all() and len(a)==11*4*4*2*2
    a['dm']=a.ba_ge30-a.srs
    out[c]=a
    print(c,'mean dm %.3f  P(<SRS) %.3f  P(<-0.5) %.3f'%(a.dm.mean(),(a.dm<0).mean(),(a.dm<-0.5).mean()))
r1={c:out[c].assign(bad=out[c].dm<-0.5).groupby('target').bad.mean() for c in out}
print('R1 AE mean %.3f S2 mean %.3f'%(r1['AE'].mean(),r1['S2'].mean()))
ttest((r1['AE']-r1['S2']).values,'H4 R1 diff')
print('wilcoxon',stats.wilcoxon((r1['AE']-r1['S2']).values))
print('wilcoxon exact/approx', stats.wilcoxon((r1['AE']-r1['S2']).values, method='approx'), stats.wilcoxon((r1['AE']-r1['S2']).values, zero_method='zsplit'))
print((r1['AE']-r1['S2']).round(4).to_dict())
# country-level R3
for c in out:
    g=out[c].groupby('target')
    print(c,'country-avg: mean dm %.3f P<SRS %.3f P<-0.5 %.3f'%(g.dm.mean().mean(),g.dm.apply(lambda x:(x<0).mean()).mean(),g.dm.apply(lambda x:(x<-0.5).mean()).mean()))
# R2 candidates
def r2(f):
    v=np.log(f(out['AE'])/f(out['S2'])); ttest(v.values,'R2 cand')
print('SD of dm across all cells within country'); r2(lambda a:a.groupby('target').dm.std(ddof=1))
print('SD of ba across all cells within country'); r2(lambda a:a.groupby('target').ba_ge30.std(ddof=1))
print('SD of dm across units within (country,method,clf,k), averaged'); r2(lambda a:a.groupby(['target','method','clf','k']).dm.std(ddof=1).groupby('target').mean())
print('SD of ba across units within cell, averaged'); r2(lambda a:a.groupby(['target','method','clf','k']).ba_ge30.std(ddof=1).groupby('target').mean())
print('RMS SD across units within cell'); r2(lambda a:np.sqrt(a.groupby(['target','method','clf','k']).ba_ge30.var(ddof=1).groupby('target').mean()))
print('SD of method-mean ... '); r2(lambda a:a.groupby(['target','method']).dm.mean().groupby('target').std(ddof=1))
