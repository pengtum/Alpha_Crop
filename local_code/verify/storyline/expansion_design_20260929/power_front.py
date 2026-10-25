import numpy as np, pandas as pd
from scipy import stats
R=pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/storyline/contribution_review_20260927/c3_skeptic/retrain_tau_rows.csv')
key=['target','split','rep','clf']
def ba_at(c,oa_t):
    c=c.sort_values('tau'); o=c.oa.values; b=c.ba_ge30.values
    cand=[]
    for i in range(len(o)-1):
        lo,hi=sorted((o[i],o[i+1]))
        if lo<=oa_t<=hi and o[i]!=o[i+1]:
            cand.append(b[i]+(oa_t-o[i])/(o[i+1]-o[i])*(b[i+1]-b[i]))
    if abs(o[0]-oa_t)<1e-9: cand.append(b[0])
    return max(cand) if cand else np.nan
def area(c,lo,hi,n=21):
    g=np.linspace(lo,hi,n); v=[ba_at(c,x) for x in g]; return np.nanmean(v) if not np.isnan(v).any() else np.nan
out=[]
for u,g in R.groupby(key):
    base=g[g.src=='base']; oa0=base[base.tau==0].oa.iloc[0]; ba0=base[base.tau==0].ba_ge30.iloc[0]
    for k in sorted(g[g.src=='retrain'].k.unique()):
        r=g[(g.src=='retrain')&(g.k==k)]; r0=r[r.tau==0].iloc[0]
        a=ba_at(r,oa0)                      # retrain curve at OA0
        c=ba_at(base,r0.oa)                 # base frontier at retrain native OA
        ar=area(r,oa0-4,oa0); ab=area(base,oa0-4,oa0)
        out.append(dict(zip(key,u),k=k,raw=r0.ba_ge30-ba0,rawOA=r0.oa-oa0,dF_OA0=a-ba0,dC=r0.ba_ge30-c,dArea=ar-ab,reach=int(not np.isnan(a))))
o=pd.DataFrame(out)
def ci(v,lvl=.95):
    v=np.asarray(v,float); v=v[~np.isnan(v)]; m=v.mean(); se=v.std(ddof=1)/np.sqrt(len(v)); t=stats.t.ppf(1-(1-lvl)/2,len(v)-1)
    return f'{m:+.2f} [{m-t*se:+.2f},{m+t*se:+.2f}] sd={v.std(ddof=1):.2f} n={len(v)}'
for k in sorted(o.k.unique()):
    ok=o[o.k==k]
    print('k',k,'unreachable units',int((ok.reach==0).sum()),'/',len(ok))
    cm=ok.groupby(['target','clf'])[['raw','rawOA','dF_OA0','dC','dArea']].mean().groupby('target').mean()
    for nm,x in (('11',cm),('9',cm.drop(['RO','PT']))):
        print(' ',nm,' | '.join(f'{c} {ci(x[c])}' for c in x.columns))
