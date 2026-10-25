import numpy as np, pandas as pd, os
from scipy import stats
D='/Users/perng_mini/AlphaCrops_local/data/cache_guidance_G/'
S0='/Users/perng_mini/AlphaCrops_local/results/guidance_G/stage0/'
PM='/Users/perng_mini/AlphaCrops_local/results/guidance_G_ml/probs/'
y=np.load(D+'y.npy')
def metrics(yt,pp,big):
    rec=np.array([(pp[yt==c]==c).mean() if (yt==c).any() else np.nan for c in range(10)])
    return 100*(yt==pp).mean(), 100*np.nanmean(rec[big])
def adjust(P,pi,pitr):
    Q=P*(pi/np.maximum(pitr,1e-12)); return Q/np.maximum(Q.sum(1,keepdims=True),1e-300)
def em(P,pitr,iters=200,tol=1e-6):
    pi=pitr.copy()
    for _ in range(iters):
        new=adjust(P,pi,pitr).mean(0)
        if np.abs(new-pi).max()<tol: return new
        pi=new
    return pi
def tci(x):
    x=np.asarray(x); n=len(x); m=x.mean(); se=x.std(ddof=1)/np.sqrt(n); t=stats.t.ppf(.975,n-1)
    return f"{m:+.2f} [{m-t*se:+.2f},{m+t*se:+.2f}] {int((x>0).sum())}/{n}"
rows=[]
for fn in sorted(os.listdir(S0)):
    tgt,sp=fn[:-4].split('_'); split=int(sp[1]); S=np.load(S0+fn)
    yt=y[S['test']]; yc=y[S['cand']]; yf=y[S['foreign']]
    pitr=(np.bincount(yf,minlength=10)>0).astype(float); pitr/=pitr.sum()
    big=np.where(np.bincount(yt,minlength=10)>=30)[0]
    n_cand=len(S['cand']); rng0=np.random.default_rng(10_000*split+7)
    ypool=yc[np.sort(rng0.choice(n_cand,min(30000,n_cand),replace=False))]
    pi_test=np.bincount(yt,minlength=10)/len(yt)
    for L in ['lgb','lr','mlp','rf']:
        for f in ['ae','s2']:
            if L=='lgb': P=S[f'P0test_{f}_i300'].astype(np.float64)
            else: P=np.load(f'{PM}{L}/{tgt}_s{split}_{f}_k0.npz')['P'].astype(np.float64)
            P=np.clip(P,1e-12,None); P/=P.sum(1,keepdims=True)
            r=dict(L=L,target=tgt,split=split,feat=f)
            r['L0_oa'],r['L0_ba']=metrics(yt,P.argmax(1),big)
            r['PO_oa'],r['PO_ba']=metrics(yt,adjust(P,pi_test,pitr).argmax(1),big)
            r['EM_oa'],r['EM_ba']=metrics(yt,adjust(P,em(P,pitr),pitr).argmax(1),big)
            o=[];b=[]
            for s in range(20):
                lab=ypool[np.random.default_rng(1000+s).choice(len(ypool),50,replace=False)]
                pk=np.bincount(lab,minlength=10)+1.0; pk/=pk.sum()
                a1,a2=metrics(yt,adjust(P,pk,pitr).argmax(1),big); o.append(a1); b.append(a2)
            r['PK_oa'],r['PK_ba']=np.mean(o),np.mean(b); rows.append(r)
df=pd.DataFrame(rows); df.to_csv('/private/tmp/c1_skeptic/g_prior_ml_rows.csv',index=False)
for L in ['lgb','lr','mlp','rf']:
    c=df[df.L==L].groupby('target').mean(numeric_only=True)
    print(f'==== {L}: k0 OA {c.L0_oa.mean():.1f} BA {c.L0_ba.mean():.1f}')
    for lab,m in [('11',list(c.index)),('exROPT',[t for t in c.index if t not in ['RO','PT']])]:
        cc=c.loc[m]; g=cc.PO_oa-cc.L0_oa
        print(f' [{lab}] PO-L0 OA {tci(g)} | PK50-L0 OA {tci(cc.PK_oa-cc.L0_oa)} | rec PK50 {((cc.PK_oa-cc.L0_oa).sum()/g.sum()*100):.0f}% EM {((cc.EM_oa-cc.L0_oa).sum()/g.sum()*100):.0f}%')
        print(f'      BA: PO-L0 {tci(cc.PO_ba-cc.L0_ba)} | PK50-L0 {tci(cc.PK_ba-cc.L0_ba)}')
    g=c.PO_oa-c.L0_oa; print('  RO share of PO gain', round(g['RO']/g.sum()*100,1))
