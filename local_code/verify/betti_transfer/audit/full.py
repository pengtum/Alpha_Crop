# 全部 44 单元 × 2 空间：KMeans 分别用 float64 / float32 输入，重算 K=8、q95/none、w=1/50、k∈{50,200,400}
# 目的：确认主实现 = float64 版本、独立实现 = float32 版本，两者差异完全来自 KMeans 输入精度
import os, sys
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[v]='2'
sys.dont_write_bytecode=True
import zipfile, time
import numpy as np, pandas as pd
from numpy.lib import format as npf
from sklearn.cluster import KMeans
LE='/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, LE)
from guidance_G import pool_positions
ST='/Users/perng_mini/AlphaCrops_local/results/guidance_G/stage0'
OUT=os.path.dirname(os.path.abspath(__file__))
def hdr(f):
    v=npf.read_magic(f)
    return npf.read_array_header_1_0(f) if v==(1,0) else npf.read_array_header_2_0(f)
def read_rows(z,key,n=None):
    with z.open(key+'.npy') as f:
        shape,fo,dt=hdr(f); assert not fo
        n=shape[0] if n is None else n
        cols=int(np.prod(shape[1:])) if len(shape)>1 else 1
        need=n*cols*dt.itemsize; b=f.read(need); assert len(b)==need
        a=np.frombuffer(b,dtype=dt).copy()
        return (a.reshape(n,cols) if len(shape)>1 else a), shape
def greedy(gamma,cap,n0,k):
    n=np.zeros(len(gamma))
    for _ in range(int(min(k,cap.sum()))):
        a=n0+n; g=np.where(n<cap, gamma*((a+1)**-0.5-(a+2)**-0.5), -np.inf)
        n[int(np.argmax(g))]+=1
    return n
rows=[]; t0=time.time()
for c in ['AT','BE','DE','EE','LT','LV','PT','RO','SE','SI','SK']:
    for s in [0,1]:
        with zipfile.ZipFile(f'{ST}/{c}_s{s}.npz') as z:
            cand,_=read_rows(z,'cand'); nc=len(cand)
            G={f:read_rows(z,f'G_{f}',nc)[0] for f in ['ae','s2']}
            GF={f:read_rows(z,f'GF_{f}')[0].astype(np.float64) for f in ['ae','s2']}
        for r in [0,1]:
            pos=pool_positions({'cand':cand},s,r)
            for f in ['ae','s2']:
                X32=G[f][pos]; X64=X32.astype(np.float64)
                for prec,X in [('f64',X64),('f32',X32)]:
                    km=KMeans(n_clusters=8,n_init=10,random_state=0).fit(X)
                    lab=km.labels_; C=km.cluster_centers_.astype(np.float64)
                    dp=np.sqrt(((X64-C[lab])**2).sum(1))
                    cap=np.bincount(lab,minlength=8).astype(float); gam=cap/cap.sum()
                    d2=(GF[f]**2).sum(1)[:,None]+(C**2).sum(1)[None]-2*GF[f]@C.T
                    gs=d2.argmin(1); ds=np.sqrt(np.maximum(d2[np.arange(len(gs)),gs],0))
                    for thr in ['q95','none']:
                        rg=np.array([np.quantile(dp[lab==g],0.95) for g in range(8)]) if thr=='q95' else np.full(8,np.inf)
                        m=np.bincount(gs[ds<=rg[gs]],minlength=8).astype(float)
                        for k in ([50,200,400] if thr=='q95' else [200]):
                            nt=greedy(gam,cap,m/50,k); nn=greedy(gam,cap,np.zeros(8),k)
                            rows.append(dict(country=c,split=s,rep=r,space=f,prec=prec,thr=thr,k=k,
                                             tv_a=0.5*np.abs(nt-nn).sum()/k, tv_b=0.5*np.abs(nt-gam*k).sum()/k,
                                             fr=m.sum()/50000))
        print(c,s,f'{time.time()-t0:.0f}s',flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/full_recalc.csv',index=False)
print('done')
