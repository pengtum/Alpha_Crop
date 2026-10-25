import numpy as np, pandas as pd, glob
from sklearn.metrics import roc_auc_score
from scipy.spatial import cKDTree
R='/Users/perng_mini/AlphaCrops_local/results/guidance_G/'
y=np.load('/Users/perng_mini/AlphaCrops_local/data/cache_guidance_G/y.npy')
C=['AT','BE','DE','EE','LT','LV','PT','RO','SE','SI','SK']
E1=['lpm2_nogeo','kcenter','typiclust','kherding']; OTH=['margin','predstrat_srs','gist_margin','badge_c']
jac=[]; q=[]
def pool(S,split,rep):
    n=len(S['cand']); g=np.random.default_rng(10000*split+100*rep+7)
    return np.sort(g.choice(n,min(30000,n),replace=False))
def knnba(X,yy,k=10):
    _,idx=cKDTree(X).query(X,k=k+1); idx=idx[:,1:]
    pred=np.array([np.bincount(yy[i],minlength=10).argmax() for i in idx])
    return 100*np.mean([(pred[yy==c]==c).mean() for c in np.unique(yy) if (yy==c).sum()>=10])
for t in C:
    for s in (0,1):
        S=dict(np.load(R+f'stage0/{t}_s{s}.npz'))
        for r in (0,1):
            for m in E1+(OTH if r==0 else []):
                for k in (50,200):
                    a=np.load(R+f'sel/{t}_s{s}_r{r}_AE_{m}_k{k}.npz')['sel']; b=np.load(R+f'sel/{t}_s{s}_r{r}_S2_{m}_k{k}.npz')['sel']
                    jac.append(dict(t=t,s=s,r=r,m=m,k=k,j=len(np.intersect1d(a,b))/len(np.union1d(a,b))))
        pp=pool(S,s,0); yy=y[S['allT'][pp]]; pi=np.bincount(yy,minlength=10)/len(yy)
        ref=np.random.default_rng(99).choice(len(pp),3000,replace=False)  # different subsample on purpose
        for f in ('ae','s2'):
            P=S[f'P_T_{f}'][pp].astype(float)
            ss=np.sort(P,1); wrong=P.argmax(1)!=yy
            q.append(dict(t=t,s=s,f=f,nll=-np.log(np.clip(P[np.arange(len(yy)),yy],1e-300,None)).mean(),
                          auroc=roc_auc_score(wrong,1-(ss[:,-1]-ss[:,-2])),
                          pL1=np.abs(np.bincount(P.argmax(1),minlength=10)/len(yy)-pi).sum(),
                          knn=knnba(np.asarray(S[f'G_{f}'][pp][ref],float),yy[ref])))
J=pd.DataFrame(jac); print('Jaccard by method,k mean:'); print(J.groupby(['m','k']).j.mean().round(3).to_string()); print('range of means',J.groupby(['m','k']).j.mean().min(),J.groupby(['m','k']).j.mean().max())
Q=pd.DataFrame(q); print(Q.groupby('f')[['nll','auroc','pL1','knn']].mean().round(3))
w=Q.pivot_table(index=['t','s'],columns='f',values='knn'); print('knn ae>s2 countries', (w.groupby('t').mean().ae>w.groupby('t').mean().s2).sum())
