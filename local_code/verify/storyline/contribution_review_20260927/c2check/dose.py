import numpy as np, pandas as pd
from scipy import stats
exec(open('/private/tmp/c2check/theta.py').read().split("Rs={'LGB'")[0])
R=load_lgb()
meth=['lpm2_nogeo','typiclust']
out={}
for a in ['AE','ORACLE','ORACLE2']:
    d=deltas(R,a,'S2',meth); d=d[d.rep==0]
    v=cv(d); out[a]=v; print(a,'-S2 (rep0, lpm2+typiclust):',ts(v.values))
    for m in meth:
        vm=d[d.method==m].groupby(['target','clf','k'])['d'].mean().groupby('target').mean(); print('    ',m,ts(vm.values))
ok=pd.read_csv('/private/tmp/c2check/oracle_knn.csv').groupby('target').oracle_zclip.mean()
gq=pd.read_csv('/Users/perng_mini/AlphaCrops_local/results/guidance_G/guidance_quality_G.csv')
knn=gq.pivot_table(index='target',columns='feat',values='knn10_ba')
print('mean knn S2 %.1f AE %.1f ORACLE %.1f'%(knn.s2.mean(),knn.ae.mean(),ok.mean()))
# pooled within-country dose-response: stack (knn diff vs S2, effect) for AE and ORACLE
X=pd.concat([ (knn.ae-knn.s2).rename('dk'), out['AE'].rename('eff')],axis=1).assign(g='AE')
Y=pd.concat([ (ok-knn.s2).rename('dk'), out['ORACLE'].rename('eff')],axis=1).assign(g='ORACLE')
Z=pd.concat([X,Y]); sl=stats.linregress(Z.dk,Z.eff); print('pooled slope pp per knn-pp: %.4f (se %.4f) intercept %.3f'%(sl.slope,sl.stderr,sl.intercept))
r=out['ORACLE'].mean()/ (ok-knn.s2).mean(); r2=out['ORACLE2'].mean()/(ok-knn.s2).mean()
print('ratio slope ORACLE: %.4f ; ORACLE2 (using ORACLE knn as proxy): %.4f'%(r,r2))
print('implied AE effect: %.3f / %.3f pp for AE knn gain %.2f'%(r*(knn.ae-knn.s2).mean(), r2*(knn.ae-knn.s2).mean(), (knn.ae-knn.s2).mean()))
