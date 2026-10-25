# 完全自写的 τ 族 + OA/BA（不经 step0B），与 ae2_curves.csv 比较；并用自写包络重算 D
import os, sys
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[v]='2'
sys.dont_write_bytecode=True
import numpy as np, pandas as pd
R3='/Users/perng_mini/AlphaCrops_local/results/check3'
OUT='/Users/perng_mini/AlphaCrops_local/verify/preconf/ae/indep_ae_mechanism'
TAUS=np.round(np.arange(-2,2+1e-9,0.05),2)
cur=pd.read_csv(f'{OUT}/ae2_curves.csv', float_precision='round_trip')
units=pd.read_csv(f'{OUT}/ae2_units.csv')
C=['AT','BE','DE','EE','LT','LV','PT','RO','SE','SI','SK']
maxd=0; dd=[]
def curve(P,y):
    P=P.astype(np.float64); P/=P.sum(1,keepdims=True)
    with np.errstate(divide='ignore'): Z=np.log(P)
    pi=P.mean(0)
    lp=np.log(np.where(pi>0,pi,1.0))
    n=np.bincount(y,minlength=10); big=np.flatnonzero(n>=30)
    oa=[];ba=[]
    for t in TAUS:
        yh=(Z-t*lp).argmax(1)
        oa.append(100*(yh==y).mean()); ba.append(100*np.mean([(yh[y==c]==c).mean() for c in big]))
    return np.array(oa),np.array(ba)
def env(oa,ba,o):
    best=-np.inf
    for j in range(len(oa)-1):
        a0,a1,b0,b1=oa[j],oa[j+1],ba[j],ba[j+1]
        if min(a0,a1)-1e-9<=o<=max(a0,a1)+1e-9:
            v=max(b0,b1) if abs(a1-a0)<1e-9 else b0+min(max((o-a0)/(a1-a0),0),1)*(b1-b0)
            best=max(best,v)
    return best if np.isfinite(best) else np.nan
for tgt in C:
    y=np.load(f'{R3}/{tgt}_split.npz')['y_test']
    for cap,suf in (('c300','_c300'),('es','')):
        for mk,m in (('dom','within'),('xb','loco')):
            cv={}
            for f in ('ae','s2'):
                cv[f]=curve(np.load(f'{R3}/{tgt}_{f}_{m}{suf}_test.npy'),y)
                x=cur[(cur.target==tgt)&(cur.cap==cap)&(cur.model==mk)&(cur.clf==f)].sort_values('tau')
                maxd=max(maxd,np.abs(cv[f][0]-x.oa.values).max(),np.abs(cv[f][1]-x.ba_ge30.values).max())
            i0=40
            anc=min(cv['ae'][0][i0],cv['s2'][0][i0])
            grid=[anc-4+0.1*i for i in range(41)]
            a=np.array([env(*cv['ae'],o) for o in grid]); s=np.array([env(*cv['s2'],o) for o in grid])
            ok=np.isfinite(a)&np.isfinite(s)
            D=np.mean(a[ok]-s[ok]) if ok.mean()>=0.5 else np.nan
            ref=units[(units.target==tgt)&(units.cap==cap)&(units.model==mk)].D.iloc[0]
            dd.append(abs(D-ref))
print('曲线最大|Δ|',maxd,'；D 最大|Δ|',max(dd), '单元数', len(dd))
