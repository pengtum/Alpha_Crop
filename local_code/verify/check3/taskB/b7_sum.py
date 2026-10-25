import numpy as np, pandas as pd, glob, os, warnings; warnings.filterwarnings('ignore')
C=10; D='/Users/perng_mini/AlphaCrops_local/results/check3'
def laplace(yl,a=1.0):
    n=np.bincount(yl,minlength=C).astype(float); return (n+a)/(n.sum()+a*C)
def em(p,it=5000,tol=1e-6):
    pt=np.full(C,.1); pi=pt.copy()
    for _ in range(it):
        a=p*(pi/pt); a/=a.sum(1,keepdims=True); new=a.mean(0)
        if np.abs(new-pi).max()<tol: pi=new; break
        pi=new
    return pi
def bage(yt,yh):
    n=np.bincount(yt,minlength=C); return 100*np.mean([(yh[yt==c]==c).mean() for c in np.flatnonzero(n>=30)])
rows=[]
for path in sorted(glob.glob('b7_probs/*_A.npz')):
    t,f,_=os.path.basename(path).split('_')
    if not os.path.exists(f'b7_probs/{t}_{f}_B.npz'): continue
    sp=np.load(f'{D}/{t}_split.npz'); yp,yt=sp['y_pool'],sp['y_test']; piT=np.bincount(yp,minlength=C)/len(yp)
    for v in 'AB':
        d=np.load(f'b7_probs/{t}_{f}_{v}.npz'); p=d['test']/d['test'].sum(1,keepdims=True); pp=d['pool']/d['pool'].sum(1,keepdims=True)
        assert (p.max(0)>0).all()
        oa=lambda q: 100*(q.argmax(1)==yt).mean()
        pk=[oa(p*laplace(yp[np.random.default_rng(5000+s).choice(len(yp),50,replace=False)])) for s in range(20)]
        rows.append(dict(target=t,feat=f,v=v,iters=int(d['iters']),maxdiff=float(d['maxdiff_vs_loco']),L0=oa(p),PO=oa(p*piT),PK50=np.mean(pk),EM=oa(p*em(pp)),
                         meanmax=p.max(1).mean(), ba_L0=bage(yt,p.argmax(1)), ba_PO=bage(yt,(p*piT).argmax(1))))
X=pd.DataFrame(rows)
pd.set_option('display.width',250)
X['PO−L0']=X.PO-X.L0; X['PK50%']=100*(X.PK50-X.L0)/X['PO−L0']; X['EM%']=100*(X.EM-X.L0)/X['PO−L0']; X['baPO−L0']=X.ba_PO-X.ba_L0
print(X[['target','feat','v','iters','maxdiff','meanmax','L0','PO','PO−L0','PK50%','EM%','ba_L0','baPO−L0']].round(3).to_string(index=False))
for v in 'AB':
    for f in ('ae','s2'):
        x=X[(X.v==v)&(X.feat==f)]
        if len(x)==0: continue
        print(f'{v} {f}: n={len(x)} L0 均值 {x.L0.mean():.2f}  PO−L0 均值 {x["PO−L0"].mean():+.2f}  合并 PK50 {100*(x.PK50-x.L0).sum()/(x.PO-x.L0).sum():.0f}%  EM {100*(x.EM-x.L0).sum()/(x.PO-x.L0).sum():.0f}%  PO 后 OA {x.PO.mean():.2f}  ba(PO)−ba(L0) {x["baPO−L0"].mean():+.2f}')
X.to_pickle('b7_sum.pkl')
