import glob, numpy as np, pandas as pd
OUT='/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R=pd.concat([pd.read_csv(f) for f in glob.glob(OUT+'/rows_t*_*.csv')],ignore_index=True)
g=R[R.method=='srs'].copy()
g['x']=np.log2(g.k)
# within-curve slope using unit-level data with unit fixed effects
res=[]
for (t,c),d in g.groupby(['target','clf']):
    d=d.copy(); d['yc']=d.ba_ge30-d.groupby(['split','rep']).ba_ge30.transform('mean'); d['xc']=d.x-d.x.mean()
    b=(d.yc*d.xc).sum()/(d.xc**2).sum()
    r=d.yc-b*d.xc; dof=len(d)-4-1
    se=np.sqrt((r**2).sum()/dof/(d.xc**2).sum())
    sdk=d.groupby('k').ba_ge30.std(ddof=1).mean()
    res.append(dict(t=t,clf=c,slope=b,se=se,sd_units_at_k=sdk))
S=pd.DataFrame(res); print(S.round(3).to_string())
print('median SE of per-curve slope',S.se.median().round(3),' median unit SD at fixed k',S.sd_units_at_k.median().round(3))
