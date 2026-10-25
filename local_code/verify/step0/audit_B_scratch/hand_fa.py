# 手算 FA：从原始概率独立重算 4 个 (单元, 特征, 臂)，另做 τ=0 全量对账
import os, sys, ast, hashlib, glob
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[v]='2'
sys.dont_write_bytecode=True
import numpy as np, pandas as pd
from scipy.optimize import minimize
L='/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0,L); os.chdir(L)
import guidance_G as G
from check3_priors import em_prior, reweight
R_G='/Users/perng_mini/AlphaCrops_local/results/guidance_G'
CALIB='/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/prior_shift/calib_diag.py'
src=open(CALIB).read(); ns=dict(np=np,minimize=minimize,C=10)
for node in ast.parse(src).body:
    if isinstance(node,ast.FunctionDef) and node.name in ('fit_bcts','weights'):
        exec(compile(ast.get_source_segment(src,node),CALIB,'exec'),ns)
fit_bcts,weights=ns['fit_bcts'],ns['weights']
C=10
TAUS=[round(-0.5+0.05*i,2) for i in range(51)]
D=G.GData(); y=np.asarray(D.y)
MAIN=pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/step0/partB/fa_rows.csv'); MAIN=MAIN[MAIN.anchor=='Z0']
def pts(Z, yT, big):
    P=np.exp(Z-np.max(np.where(np.isfinite(Z),Z,-1e300),1,keepdims=True)); P/=P.sum(1,keepdims=True); pi=P.mean(0)
    out=[]
    for t in TAUS:
        with np.errstate(invalid='ignore'):
            sc=Z-t*np.where(pi>0,np.log(np.where(pi>0,pi,1)),0)
        ph=sc.argmax(1)
        out.append((100*(ph==yT).mean(),100*np.mean([(ph[yT==c]==c).mean() for c in big])))
    return out
def fa(curves, oastar):
    vals=[]
    for j in range(41):
        o=oastar-4+0.1*j
        best=None
        for cv in curves:
            for (o0,b0),(o1,b1) in zip(cv[:-1],cv[1:]):
                if min(o0,o1)-1e-9<=o<=max(o0,o1)+1e-9:
                    v=max(b0,b1) if abs(o1-o0)<1e-9 else b0+min(max((o-o0)/(o1-o0),0),1)*(b1-b0)
                    best=v if best is None else max(best,v)
        vals.append(best)
    ok=[v for v in vals if v is not None]
    cov=len(ok)/41
    return (np.mean(ok) if cov>=0.5 else np.nan), cov
GR=pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(R_G,'rows_t1_*.csv'))])
GR=GR[(GR.cond=='none')&(GR.method=='srs')]
def cell(tgt,split,rep,f,arm):
    S=np.load(os.path.join(R_G,'stage0',f'{tgt}_s{split}.npz')); nc=len(S['cand'])
    test=S['test']; yT=y[test]; big=np.flatnonzero(np.bincount(yT,minlength=C)>=30)
    PT=S[f'P_T_{f}'][nc:].astype(np.float64)
    with np.errstate(divide='ignore'):
        z0=np.log(PT/PT.sum(1,keepdims=True))
    c0=pts(z0,yT,big); oastar=c0[TAUS.index(0.0)][0]
    unit=f'{tgt}_s{split}_r{rep}'
    if arm=='Z0+':
        yF=y[S['foreign']]; cw=G.class_weights_from(yF); pit=np.bincount(yF,weights=cw[yF],minlength=C)/cw[yF].sum()
        P0=PT/PT.sum(1,keepdims=True); pe=em_prior(P0,pit)[0]; Q=reweight(P0,pe,pit)[0]
        with np.errstate(divide='ignore'):
            curves=[c0,pts(np.log(Q),yT,big)]
    else:
        k=int(arm[2:]); g=GR[(GR.target==tgt)&(GR.split==split)&(GR.rep==rep)&(GR.clf==f)&(GR.k==k)].iloc[0]
        if arm.startswith('RT'):
            Pr=np.load(os.path.join(R_G,'probs',f'{unit}_{f}_k{k}_{g.sel_hash}.npz'))['i300'].astype(np.float64)
            with np.errstate(divide='ignore'):
                curves=[pts(np.log(Pr/Pr.sum(1,keepdims=True)),yT,big)]
        else:
            pp=G.pool_positions(S,split,rep); sp=np.load(os.path.join(R_G,'sel',f'{unit}_none_srs_k{k}.npz'))['sel_pos']
            ys=y[S['allT'][pp][sp]]
            s,b=fit_bcts(np.log(np.clip(S[f'P_T_{f}'][pp][sp].astype(np.float64),1e-12,None)),ys,weights(ys,np.ones(len(ys)),'bal'),0.1)
            curves=[pts(s*np.log(np.clip(PT,1e-12,None))+b,yT,big)]
    F,cov=fa(curves,oastar)
    m=MAIN[(MAIN.unit==unit)&(MAIN.clf==f)&(MAIN.arm==arm.replace('-',''))].iloc[0]
    print(f'{unit} {f} {arm}: 手算 OA*={oastar:.4f} FA={F:.6f} 可达={cov:.4f} | 主实现 OA*={m.OA_star:.4f} FA={m.FA:.6f} 可达={m.reach:.4f} | ΔFA={F-m.FA if np.isfinite(F) else float("nan"):.2e}')
cell('AT',0,0,'ae','RT200')
cell('LT',1,0,'s2','HD400')
cell('LT',1,1,'s2','HD200')
cell('RO',1,0,'ae','Z0+')
cell('SI',1,0,'ae','HD50')
