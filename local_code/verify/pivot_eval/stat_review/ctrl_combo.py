# 评审对照 3：组合方案的零训练预估。XV-Cal（两视图校准层，只用 k 个标签）与"两视图重训集成"按 γ 做对数概率混合，
# 与公平基线（AE+S2 重训集成，同一批标签）以及 AE 单重训比较。只读 G 的 sel/probs/stage0；stacker 从 multiview 目录只读导入。
import os, sys; os.environ['OMP_NUM_THREADS']='2'; sys.dont_write_bytecode=True
import numpy as np, pandas as pd, glob
R='/Users/perng_mini/AlphaCrops_local'; GG=f'{R}/results/guidance_G'
sys.path.insert(0,f'{R}/verify/pivot_eval/multiview'); import stacker as st
Y=np.load(f'{R}/data/cache_guidance_G/y.npy')
rows=pd.concat([pd.read_csv(f) for f in glob.glob(f'{GG}/rows_t1_*.csv')])
def pool_positions(n_cand, split, rep):
    rng=np.random.default_rng(10_000*split+100*rep+7); return np.sort(rng.choice(n_cand,min(30000,n_cand),replace=False))
def sc(yT,L,big):
    ph=L.argmax(1); return 100*(ph==yT).mean(), 100*np.mean([(ph[yT==c]==c).mean() for c in big])
def cen(L): return L-L.mean(1,keepdims=True)
GAM=(0.0,0.25,0.4,0.5,1.0)
out=[]
for (lab_cond,lab_m) in (('none','srs'),('AE','predstrat_srs')):
  for k in (50,200):
    g=rows[(rows.cond==lab_cond)&(rows.method==lab_m)&(rows.k==k)]
    for (t,s,r),gg in g.groupby(['target','split','rep']):
        z=np.load(f'{GG}/stage0/{t}_s{s}.npz'); nc=len(z['cand']); pp=pool_positions(nc,s,r)
        yT=Y[z['test']]; big=np.flatnonzero(np.bincount(yT,minlength=10)>=30); yP=Y[z['allT'][pp]]
        Pa,Ps=z['P_T_ae'].astype(np.float64),z['P_T_s2'].astype(np.float64)
        sp=np.load(f'{GG}/sel/{t}_s{s}_r{r}_{lab_cond}_{lab_m}_k{k}.npz')['sel_pos']
        L={}
        for clf in ('ae','s2'):
            h=gg[gg.clf==clf].sel_hash.iloc[0]
            L[clf]=np.log(np.clip(np.load(glob.glob(f'{GG}/probs/{t}_s{s}_r{r}_{clf}_k{k}_{h}.npz')[0])['i300'].astype(np.float64),1e-6,None))
        ens=cen((L['ae']+L['s2'])/2)
        rec0=dict(lab=f'{lab_cond}_{lab_m}',k=k,t=t,s=s,r=r)
        oa,ba=sc(yT,L['ae'],big); rec0.update(oa_ae=oa,ba_ae=ba)
        oa,ba=sc(yT,ens,big); rec0.update(oa_ens=oa,ba_ens=ba)
        for mode in (0.5,'bal'):
            th=st.fit(st._ll(Pa[pp][sp]),st._ll(Ps[pp][sp]),yP[sp],mode=mode,gw=5.0,gb=1.0,views='both')
            Lx=cen(np.log(np.clip(st.predict(th,st._ll(Pa[nc:]),st._ll(Ps[nc:])),1e-6,None)))
            for gm in GAM:
                oa,ba=sc(yT,(1-gm)*ens+gm*Lx,big)
                out.append(dict(rec0,mode=str(mode),gamma=gm,oa=oa,ba=ba))
df=pd.DataFrame(out); df.to_csv(f'{R}/verify/pivot_eval/stat_review/ctrl_combo_rows.csv',index=False)
def f(v):
    v=np.asarray(v); m=v.mean(); se=v.std(ddof=1)/np.sqrt(len(v)); return f'{m:+.2f} [{m-2.228*se:+.2f},{m+2.228*se:+.2f}] {int((v>0).sum())}/11'
for (lab,k,mode,gm),d in df.groupby(['lab','k','mode','gamma']):
    cm=d.groupby('t').mean(numeric_only=True)
    ex=cm.drop(index=['PT','RO'])
    print(f'{lab:17s} k={k:3d} mode={mode:3s} γ={gm:.2f} | vs ENS dBA {f(cm.ba-cm.ba_ens)} dOA {f(cm.oa-cm.oa_ens)} | vs AE dBA {(cm.ba-cm.ba_ae).mean():+.2f} dOA {(cm.oa-cm.oa_ae).mean():+.2f} | exRO/PT vsENS dBA {(ex.ba-ex.ba_ens).mean():+.2f} dOA {(ex.oa-ex.oa_ens).mean():+.2f}')
