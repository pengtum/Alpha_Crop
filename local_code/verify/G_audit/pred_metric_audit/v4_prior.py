import glob, os
import numpy as np, pandas as pd
from scipy import stats
OUT='/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R=pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT,'rows_t*_*.csv'))],ignore_index=True)
KEY=['target','split','rep','method','clf','k']; PRED=['margin','predstrat_srs']
def dl(metric):
    a=R[(R.cond=='AE')&R.method.isin(PRED)&(R.rep==0)].set_index(KEY)[metric]
    b=R[(R.cond=='S2')&R.method.isin(PRED)&(R.rep==0)].set_index(KEY)[metric]
    return (a-b).dropna()
def cv(s): return s.groupby(['target','method','clf','k']).mean().groupby('target').mean()
def ts(v):
    v=np.asarray(v,float); n=len(v); m=v.mean(); se=v.std(ddof=1)/np.sqrt(n); t=stats.t.ppf(.975,n-1)
    return f'{m:+.3f} [{m-t*se:+.3f},{m+t*se:+.3f}] p={2*stats.t.sf(abs(m/se),n-1):.3f} pos={int((v>0).sum())}/{n}'
print('Δoa(P0)−Δoa(P2)', ts(cv(dl('oa')-dl('oa_P2'))))
print('Δba(P0)−Δba(P2)', ts(cv(dl('ba_ge30')-dl('ba_P2'))))
T=pd.DataFrame({m:cv(dl(m)) for m in ['ba_ge30','oa','ba_P2','oa_P2']})
print(T.loc[['RO','SE','SI','PT']].round(3).to_string())
# 按方法拆
for m in PRED:
    for met in ['ba_ge30','oa','ba_P2','oa_P2']:
        a=R[(R.cond=='AE')&(R.method==m)&(R.rep==0)].set_index(KEY)[met]; b=R[(R.cond=='S2')&(R.method==m)&(R.rep==0)].set_index(KEY)[met]
        print(m,met,ts(cv((a-b).dropna())))
