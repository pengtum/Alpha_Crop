# 诊断：RO 的 Z0+ 在两实现间的差异来源（log 裁剪 vs 精确 log）
import os, sys
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[v]='2'
sys.dont_write_bytecode=True
import numpy as np, pandas as pd
L='/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0,L); os.chdir(L)
import guidance_G as G
from check3_priors import em_prior, reweight
R_G='/Users/perng_mini/AlphaCrops_local/results/guidance_G'
C=10
TAUS=np.round(np.arange(-0.5,2.0+1e-9,0.05),2)
OFFS=np.round(np.arange(-4.0,1e-9,0.1),1)
D=G.GData(); y=np.asarray(D.y)
def curve(Z, lp, yT, big):
    oa=[];ba=[]
    for t in TAUS:
        ph=(Z-t*lp).argmax(1)
        oa.append(100*(ph==yT).mean()); ba.append(100*np.mean([(ph[yT==c]==c).mean() for c in big]))
    return np.array(oa),np.array(ba)
def env(curves, grid):
    best=np.full(len(grid),-np.inf)
    for oa,ba in curves:
        for i in range(len(oa)-1):
            o0,o1,b0,b1=oa[i],oa[i+1],ba[i],ba[i+1]
            for g,o in enumerate(grid):
                if min(o0,o1)-1e-9<=o<=max(o0,o1)+1e-9:
                    v=max(b0,b1) if abs(o1-o0)<1e-9 else b0+(np.clip((o-o0)/(o1-o0),0,1))*(b1-b0)
                    best[g]=max(best[g],v)
    best[~np.isfinite(best)]=np.nan
    return best
for tgt in ['RO','PT','AT']:
  for split in (0,1):
    S=np.load(os.path.join(R_G,'stage0',f'{tgt}_s{split}.npz'))
    n_cand=len(S['cand']); test=S['test']; fr=S['foreign']; yT=y[test]
    big=np.flatnonzero(np.bincount(yT,minlength=C)>=G.MIN_N)
    yF=y[fr]; cw=G.class_weights_from(yF); pit=np.bincount(yF,weights=cw[yF],minlength=C)/cw[yF].sum()
    for f in ('ae','s2'):
        PT=S[f'P_T_{f}'][n_cand:].astype(np.float64); P0=PT/PT.sum(1,keepdims=True)
        pe,it,conv,_=em_prior(P0,pit); Q,_=reweight(P0,pe,pit)
        pihQ=Q.mean(0)
        with np.errstate(divide='ignore'):
            Zex=np.log(Q); lpex=np.where(pihQ>0,np.log(np.where(pihQ>0,pihQ,1)),0)
        Zcl=np.log(np.clip(Q,1e-12,None)); lpcl=np.log(np.clip(pihQ,1e-12,None))
        Z0=np.log(P0); lp0=np.log(P0.mean(0))
        c0=curve(Z0,lp0,yT,big); cex=curve(Zex,lpex,yT,big); ccl=curve(Zcl,lpcl,yT,big)
        grid=c0[0][TAUS==0][0]+OFFS
        fz=np.nanmean(env([c0],grid)); fex=np.nanmean(env([c0,cex],grid)); fcl=np.nanmean(env([c0,ccl],grid))
        tiny=np.flatnonzero(pihQ<1e-12)
        print(f'{tgt} s{split} {f}: OA*={grid[-1]:.2f} pi_train={np.round(pit,4).tolist()} pi_em={np.array2string(pe,precision=2)} '
              f'|test counts={np.bincount(yT,minlength=C).tolist()} big={big.tolist()} tiny π̂ classes={tiny.tolist()} '
              f'min Q={Q.min():.1e} frac Q<1e-12={(Q<1e-12).mean():.3f} | FA Z0={fz:.3f} Z0+exact={fex:.3f} Z0+clip={fcl:.3f} '
              f'| EM curve OA range exact [{cex[0].min():.1f},{cex[0].max():.1f}] clip [{ccl[0].min():.1f},{ccl[0].max():.1f}]; '
              f'maxdiff curve BA {np.abs(cex[1]-ccl[1]).max():.2f} first τ differing {TAUS[np.flatnonzero(np.abs(cex[0]-ccl[0])>1e-9)[:1]].tolist()}')
