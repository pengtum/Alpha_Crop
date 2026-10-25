# 评审对照：两视图"重训集成"（AE 重训 + S2 重训，同一批 SRS 标签，对数概率平均）能否拿到 XV-Cal 的"两视图增益"。
# 只读 G 的 rows/probs/stage0 与 multiview/d3_paired_rows.csv；线程 ≤2。
import os; os.environ['OMP_NUM_THREADS']='2'
import numpy as np, pandas as pd, glob
R='/Users/perng_mini/AlphaCrops_local'; GG=f'{R}/results/guidance_G'
Y=np.load(f'{R}/data/cache_guidance_G/y.npy')
rows=pd.concat([pd.read_csv(f) for f in glob.glob(f'{GG}/rows_t1_*.csv')])
d3=pd.read_csv(f'{R}/verify/pivot_eval/multiview/d3_paired_rows.csv')
def sc(yT,P,big):
    ph=P.argmax(1); return 100*(ph==yT).mean(), 100*np.mean([(ph[yT==c]==c).mean() for c in big])
out=[]
for (lab_cond,lab_m) in (('none','srs'),('AE','predstrat_srs')):
  for k in (50,200):
    g=rows[(rows.cond==lab_cond)&(rows.method==lab_m)&(rows.k==k)]
    for (t,s,r),gg in g.groupby(['target','split','rep']):
        z=np.load(f'{GG}/stage0/{t}_s{s}.npz'); yT=Y[z['test']]; big=np.flatnonzero(np.bincount(yT,minlength=10)>=30)
        P={}
        for clf in ('ae','s2'):
            h=gg[gg.clf==clf].sel_hash.iloc[0]
            fs=glob.glob(f'{GG}/probs/{t}_s{s}_r{r}_{clf}_k{k}_{h}.npz')
            if not fs: break
            P[clf]=np.load(fs[0])['i300'].astype(np.float64)
        if len(P)<2: continue
        la,ls=np.log(np.clip(P['ae'],1e-6,None)),np.log(np.clip(P['s2'],1e-6,None))
        rec=dict(lab=f'{lab_cond}_{lab_m}',k=k,t=t,s=s,r=r)
        rec['oa_ae'],rec['ba_ae']=sc(yT,P['ae'],big)
        rec['oa_s2'],rec['ba_s2']=sc(yT,P['s2'],big)
        rec['oa_ens'],rec['ba_ens']=sc(yT,la+ls,big)
        rec['gB_ae']=gg[gg.clf=='ae'].ba_ge30.iloc[0]
        c=d3[(d3.target==t)&(d3.split==s)&(d3.rep==r)&(d3.cond==lab_cond)&(d3.method==lab_m)&(d3.k==k)&(d3.alpha==0.5)]
        for v in ('both','ae'):
            cc=c[c.views==v]
            if len(cc): rec[f'oa_xv_{v}'],rec[f'ba_xv_{v}']=cc.oa.iloc[0],cc.ba.iloc[0]
        out.append(rec)
df=pd.DataFrame(out); df.to_csv(f'{R}/verify/pivot_eval/stat_review/ctrl_ens_rows.csv',index=False)
print('check G reproduce max|diff|:',(df.ba_ae-df.gB_ae).abs().max())
def f(v):
    v=np.asarray(v); m=v.mean(); se=v.std(ddof=1)/np.sqrt(len(v)); return f'{m:+.2f} [{m-2.228*se:+.2f},{m+2.228*se:+.2f}] {int((v>0).sum())}/{len(v)} sd={v.std(ddof=1):.2f}'
for (lab,k),d in df.groupby(['lab','k']):
    cm=d.groupby('t').mean(numeric_only=True)
    print(f'\n== labels={lab} k={k} units={len(d)} ==')
    for a,b in (('ens','ae'),('xv_both','ae'),('xv_ae','ae'),('xv_both','ens'),('xv_both','xv_ae')):
        if f'ba_{a}' not in cm: continue
        print(f'{a:8s}-{b:6s} dBA {f(cm[f"ba_{a}"]-cm[f"ba_{b}"])} | dOA {f(cm[f"oa_{a}"]-cm[f"oa_{b}"])}')
