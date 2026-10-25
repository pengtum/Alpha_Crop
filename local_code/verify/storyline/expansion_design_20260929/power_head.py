import numpy as np, pandas as pd
from scipy import stats
exec(open('power_front.py').read().split("out=[]")[0])
U=pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/storyline/decision_curve/decision_curve_units.csv')
H=U.pivot_table(index=key+['k'],columns='metric',values='head').reset_index()
Z=pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/storyline/contribution_review_20260927/c3_skeptic/zerolabel_rows.csv'); Z=Z[Z.design=='tau']
def ci(v,lvl=.95):
    v=np.asarray(v,float); v=v[~np.isnan(v)]
    if len(v)<2: return 'n<2'
    m=v.mean(); se=v.std(ddof=1)/np.sqrt(len(v)); t=stats.t.ppf(1-(1-lvl)/2,len(v)-1)
    return f'{m:+.2f} [{m-t*se:+.2f},{m+t*se:+.2f}] sd={v.std(ddof=1):.2f} n={len(v)}'
out=[]
for _,h in H[H.k.isin([50,200,400])].iterrows():
    u=tuple(h[c] for c in key)
    zb=Z.set_index(key).loc[u].reset_index()
    base0=zb[zb.tau==0].iloc[0]
    f0=ba_at(zb,h.oa)
    rr=R[(R.target==u[0])&(R.split==u[1])&(R.rep==u[2])&(R.clf==u[3])&(R.src=='retrain')&(R.k==h.k)]
    fr=ba_at(rr,h.oa)
    out.append(dict(zip(key,u),k=int(h.k),raw=h.ba_ge30-base0.ba_ge30,rawOA=h.oa-base0.oa,vsZL=h.ba_ge30-f0,vsRetTau=h.ba_ge30-fr))
o=pd.DataFrame(out)
for k in (50,200,400):
    ok=o[o.k==k]; print('k',k,'nan vsZL',int(ok.vsZL.isna().sum()),'nan vsRet',int(ok.vsRetTau.isna().sum()))
    cm=ok.groupby(['target','clf'])[['raw','rawOA','vsZL','vsRetTau']].mean().groupby('target').mean()
    for nm,x in (('11',cm),('9',cm.drop(['RO','PT']))):
        print(' ',nm,' | '.join(f'{c} {ci(x[c])}' for c in x.columns))
