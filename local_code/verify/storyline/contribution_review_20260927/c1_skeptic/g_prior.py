import numpy as np, pandas as pd, os
from scipy import stats
D='/Users/perng_mini/AlphaCrops_local/data/cache_guidance_G/'
S0='/Users/perng_mini/AlphaCrops_local/results/guidance_G/stage0/'
y=np.load(D+'y.npy')
def metrics(yt,pp,big):
    rec=np.array([(pp[yt==c]==c).mean() if (yt==c).any() else np.nan for c in range(10)])
    return 100*(yt==pp).mean(), 100*np.nanmean(rec[big])
def adjust(P,pi,pitr):
    Q=P*(pi/np.maximum(pitr,1e-12)); return Q/Q.sum(1,keepdims=True)
def em(P,pitr,iters=200,tol=1e-6):
    pi=pitr.copy()
    for _ in range(iters):
        Q=adjust(P,pi,pitr); new=Q.mean(0)
        if np.abs(new-pi).max()<tol: pi=new; break
        pi=new
    return pi
rows=[]
for fn in sorted(os.listdir(S0)):
    tgt,sp=fn[:-4].split('_'); split=int(sp[1])
    S=np.load(S0+fn)
    yt=y[S['test']]; yc=y[S['cand']]; yf=y[S['foreign']]
    nF=np.bincount(yf,minlength=10); pitr=np.where(nF>0,1.0,0.0); pitr/=pitr.sum()
    big=np.where(np.bincount(yt,minlength=10)>=30)[0]
    n_cand=len(S['cand'])
    rng0=np.random.default_rng(10_000*split+100*0+7); pool=np.sort(rng0.choice(n_cand,min(30000,n_cand),replace=False))
    ypool=yc[pool]
    pi_test=np.bincount(yt,minlength=10)/len(yt); pi_cand=np.bincount(yc,minlength=10)/len(yc)
    for f in ['ae','s2']:
        P=S[f'P0test_{f}_i300'].astype(np.float64); P/=P.sum(1,keepdims=True)
        r=dict(target=tgt,split=split,feat=f)
        r['L0_oa'],r['L0_ba']=metrics(yt,P.argmax(1),big)
        r['POt_oa'],r['POt_ba']=metrics(yt,adjust(P,pi_test,pitr).argmax(1),big)
        r['POc_oa'],r['POc_ba']=metrics(yt,adjust(P,pi_cand,pitr).argmax(1),big)
        pe=em(P,pitr); r['EM_oa'],r['EM_ba']=metrics(yt,adjust(P,pe,pitr).argmax(1),big)
        oas=[];bas=[];l1=[]
        for s in range(20):
            rg=np.random.default_rng(1000+s); lab=ypool[rg.choice(len(ypool),50,replace=False)]
            pk=(np.bincount(lab,minlength=10)+1.0); pk/=pk.sum(); l1.append(np.abs(pk-pi_test).sum())
            o,b=metrics(yt,adjust(P,pk,pitr).argmax(1),big); oas.append(o); bas.append(b)
        r['PK50_oa'],r['PK50_ba'],r['PK50_L1']=np.mean(oas),np.mean(bas),np.mean(l1)
        r['L1_test_cand']=np.abs(pi_test-pi_cand).sum()
        rows.append(r)
df=pd.DataFrame(rows); df.to_csv('/private/tmp/c1_skeptic/g_prior_rows.csv',index=False)
c=df.groupby('target').mean(numeric_only=True)
def tci(x):
    x=np.asarray(x); n=len(x); m=x.mean(); se=x.std(ddof=1)/np.sqrt(n); t=stats.t.ppf(.975,n-1)
    return f"{m:+.2f} [{m-t*se:+.2f},{m+t*se:+.2f}] {int((x>0).sum())}/{n}"
for lab,m in [('11',c.index),('exROPT',[t for t in c.index if t not in ['RO','PT']])]:
    cc=c.loc[m]
    g=cc.POt_oa-cc.L0_oa
    print(f'[{lab}] (feat avg, split avg) PO_test-L0 OA {tci(g)} | PO_cand-L0 {tci(cc.POc_oa-cc.L0_oa)} | PK50-L0 {tci(cc.PK50_oa-cc.L0_oa)} | EM-L0 {tci(cc.EM_oa-cc.L0_oa)}')
    print(f'    recovery PK50 {((cc.PK50_oa-cc.L0_oa).sum()/g.sum()*100):.1f}%  PO_cand {((cc.POc_oa-cc.L0_oa).sum()/g.sum()*100):.1f}%  EM {((cc.EM_oa-cc.L0_oa).sum()/g.sum()*100):.1f}%')
    print(f'    BA: PO_test-L0 {tci(cc.POt_ba-cc.L0_ba)} PK50-L0 {tci(cc.PK50_ba-cc.L0_ba)}')
    print(f'    L1(test vs cand) mean {cc.L1_test_cand.mean():.3f}; PK50 L1 vs test {cc.PK50_L1.mean():.3f}')
for f in ['ae','s2']:
    cf=df[df.feat==f].groupby('target').mean(numeric_only=True)
    g=cf.POt_oa-cf.L0_oa
    print(f, 'per-country PO_test-L0 OA', g.round(2).to_dict())
    print(f, 'recovery PK50 11:', round((cf.PK50_oa-cf.L0_oa).sum()/g.sum()*100,1), ' EM:', round((cf.EM_oa-cf.L0_oa).sum()/g.sum()*100,1), ' RO share of PO gain', round(g['RO']/g.sum()*100,1))
print('L0 OA mean', c.L0_oa.mean().round(2), 'L0 BA mean', c.L0_ba.mean().round(2))
