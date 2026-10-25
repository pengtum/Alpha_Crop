# 方格几何中心作"格质心"时的区划分（= 独立实现 sens_square），大 R 下与主实现划分比较 clus n0=100 的估计目标
import os, sys
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[_v]='2'
sys.dont_write_bytecode=True
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recompute as RCP
C11=RCP.CTR; C9=[c for c in C11 if c not in ('RO','PT')]; RR=200
D=RCP.G.GData(); out=[]
for t in C11:
    for s in (0,1):
        U=RCP.build(D,t,s)
        S=U['S']; lat0=float(S['lat0']); cand=U['cand']
        x=np.asarray(D.lon[cand],float)*111.32*np.cos(np.radians(lat0)); yk=np.asarray(D.lat[cand],float)*110.57
        ij=np.c_[np.floor(x/10.0),np.floor(yk/10.0)]
        cij=np.zeros((len(U['cnt']),2)); cij[U['cell']]=ij
        uq,inv=np.unique(cij,axis=0,return_inverse=True); inv=inv.ravel()   # 字典序（独立实现）
        cen=(uq+0.5)*10.0; fit_u=np.bincount(inv,weights=(U['cnt']>=10).astype(float))>0
        km=KMeans(8,n_init=10,random_state=0).fit(cen[fit_u])
        d=((cen[:,None]-km.cluster_centers_[None])**2).sum(-1); zu=d.argmin(1); zu[fit_u]=km.labels_
        creg_sq=zu[inv]
        for part,creg in (('main',U['creg']),('square',creg_sq)):
            reg=creg[U['cell']]; rs=np.bincount(reg,minlength=8); n0=100; M=10; ncl=10; el=U['cnt']>=M
            rng=np.random.default_rng(515151+1000*C11.index(t)+10*s)
            for r in range(RR):
                ro=RCP.es(rng,rs); m=2
                while m<8 and (el&np.isin(creg,ro[:m])).sum()<ncl+20: m+=1
                pool=np.flatnonzero(el&np.isin(creg,ro[:m])); cells=pool[RCP.es(rng,U['cnt'][pool])[:ncl]]
                S0=np.concatenate([rng.choice(U['members'][c],M,replace=False) for c in cells])
                s0m=np.zeros(U['n'],bool); s0m[S0]=True; S1=rng.choice(np.flatnonzero(~s0m),200,replace=False)
                e0,e1=RCP.lap(U['yc'][S0]),RCP.lap(U['yc'][S1]); row=dict(target=t,split=s,part=part,r=r)
                for f in ('ae','s2'):
                    o0=RCP.score(U,f,e0)[0]; row[f'oa_{f}']=o0; row[f's1ms0_{f}']=RCP.score(U,f,e1)[0]-o0
                out.append(row)
        print(t,s,flush=True)
df=pd.DataFrame(out)
for col in ('oa_ae','oa_s2','s1ms0_ae','s1ms0_s2'):
    res={}
    for part in ('main','square'):
        g=df[df.part==part].groupby(['target','split'])[col]; m=g.mean().groupby('target').mean().reindex(C11)
        se=(np.sqrt((g.var(ddof=1)/g.count()).groupby('target').sum())/2).reindex(C11); res[part]=(m,se)
    d=res['main'][0]-res['square'][0]; sed=np.sqrt(res['main'][1]**2+res['square'][1]**2)
    print(f'n0=100 {col:9s} 11国 main {res["main"][0].mean():+.3f} square {res["square"][0].mean():+.3f} 差 {d.mean():+.3f} (MC SE {np.sqrt((sed**2).sum())/11:.3f})；9国 差 {d[C9].mean():+.3f}；单国最大 {d.abs().idxmax()} {d[d.abs().idxmax()]:+.3f}，|z|>3 {list(d.index[(d/sed).abs()>3])}')
