import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
C=10; D='/Users/perng_mini/AlphaCrops_local/results/check3'
def laplace(yl,a=1.0):
    n=np.bincount(yl,minlength=C).astype(float); return (n+a)/(n.sum()+a*C)
def bage(yt,yh):
    n=np.bincount(yt,minlength=C); return 100*np.mean([(yh[yt==c]==c).mean() for c in np.flatnonzero(n>=30)])
def ld(t,strat,k,s,f):
    d=np.load(f'b5_probs/{t}_{strat}_{k}_{s}_{f}.npz'); p=d['test'].astype(float); return p/p.sum(1,keepdims=True), d['lab']
rows=[]
for t in ['PT','SE','SK','LV']:
    sp=np.load(f'{D}/{t}_split.npz'); yp,yt=sp['y_pool'],sp['y_test']; piT=np.bincount(yp,minlength=C)/len(yp)
    for f in ('ae','s2'):
        for s in range(3):
            p,_=ld(t,'foreign_only',0,s,f); r=dict(target=t,feat=f,seed=s,FO=bage(yt,p.argmax(1)),PO=bage(yt,(p*piT).argmax(1)))
            for k in (50,200):
                pr,lab=ld(t,'random',k,s,f); pk=laplace(yp[lab])
                r[f'PK{k}']=bage(yt,(p*pk).argmax(1)); r[f'R{k}']=bage(yt,pr.argmax(1)); r[f'R{k}+PK']=bage(yt,(pr*pk).argmax(1))
            rows.append(r)
X=pd.DataFrame(rows).groupby(['feat','target']).mean(numeric_only=True).drop(columns='seed')
G=X.sub(X.FO,axis=0).drop(columns='FO'); G.insert(0,'FO',X.FO)
print('ba_ge30（相对 FO 的变化，pp）'); print(G.round(2).to_string())
for f in ('ae','s2'): print(f, G.loc[f].mean().round(2).to_dict())
