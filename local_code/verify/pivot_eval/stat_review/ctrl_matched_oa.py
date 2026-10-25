# 评审对照 2：同 OA 下比较。对重训模型（AE 单视图 / AE+S2 集成）做"无标签"的先验压平 log p − β·log p̄（p̄ 为测试集平均预测，β∈[0,2]），
# 得到 BA–OA 曲线，在 XV-Cal 的 OA 处插值出 BA，差值 = 在相同 OA 下"标签"带来的 BA（剔除单纯移动工作点的部分）。
import os; os.environ['OMP_NUM_THREADS']='2'
import numpy as np, pandas as pd, glob
R='/Users/perng_mini/AlphaCrops_local'; GG=f'{R}/results/guidance_G'
Y=np.load(f'{R}/data/cache_guidance_G/y.npy')
rows=pd.concat([pd.read_csv(f) for f in glob.glob(f'{GG}/rows_t1_*.csv')])
d3=pd.read_csv(f'{R}/verify/pivot_eval/multiview/d3_paired_rows.csv')
def sc(yT,L,big):
    ph=L.argmax(1); return 100*(ph==yT).mean(), 100*np.mean([(ph[yT==c]==c).mean() for c in big])
BETAS=np.round(np.arange(0,2.01,0.1),2)
out=[]
for (lab_cond,lab_m) in (('none','srs'),('AE','predstrat_srs')):
  for k in (50,200):
    g=rows[(rows.cond==lab_cond)&(rows.method==lab_m)&(rows.k==k)]
    for (t,s,r),gg in g.groupby(['target','split','rep']):
        z=np.load(f'{GG}/stage0/{t}_s{s}.npz'); yT=Y[z['test']]; big=np.flatnonzero(np.bincount(yT,minlength=10)>=30)
        P={}
        for clf in ('ae','s2'):
            h=gg[gg.clf==clf].sel_hash.iloc[0]
            P[clf]=np.load(glob.glob(f'{GG}/probs/{t}_s{s}_r{r}_{clf}_k{k}_{h}.npz')[0])['i300'].astype(np.float64)
        la,ls=np.log(np.clip(P['ae'],1e-6,None)),np.log(np.clip(P['s2'],1e-6,None))
        c=d3[(d3.target==t)&(d3.split==s)&(d3.rep==r)&(d3.cond==lab_cond)&(d3.method==lab_m)&(d3.k==k)&(d3.alpha==0.5)]
        for name,L,xv in (('ae',la,'ae'),('ens',(la+ls)/2,'both')):
            q=np.exp(L-L.max(1,keepdims=True)); q/=q.sum(1,keepdims=True); pbar=q.mean(0)
            cur=np.array([sc(yT,L-b*np.log(np.clip(pbar,1e-6,None)),big) for b in BETAS])
            oaT,baT=c[c.views==xv].oa.iloc[0],c[c.views==xv].ba.iloc[0]
            o=np.argsort(cur[:,0]); oa_s,ba_s=cur[o,0],cur[o,1]
            ba_at=np.interp(oaT,oa_s,ba_s)
            out.append(dict(lab=f'{lab_cond}_{lab_m}',k=k,t=t,s=s,r=r,ret=name,oa_xv=oaT,ba_xv=baT,ba_ret_at=ba_at,
                            oa0=cur[0,0],ba0=cur[0,1],clipped=int(oaT<oa_s.min() or oaT>oa_s.max()),
                            ba_max_curve=cur[:,1].max()))
df=pd.DataFrame(out); df.to_csv(f'{R}/verify/pivot_eval/stat_review/ctrl_matched_oa_rows.csv',index=False)
def f(v):
    v=np.asarray(v); m=v.mean(); se=v.std(ddof=1)/np.sqrt(len(v)); return f'{m:+.2f} [{m-2.228*se:+.2f},{m+2.228*se:+.2f}] {int((v>0).sum())}/{len(v)}'
for (lab,k,ret),d in df.groupby(['lab','k','ret']):
    cm=d.groupby('t').mean(numeric_only=True)
    print(f'{lab:18s} k={k:3d} retrain={ret:3s}  raw dBA(xv-ret) {f(cm.ba_xv-cm.ba0)} | matched-OA dBA {f(cm.ba_xv-cm.ba_ret_at)} | clipped units {int(d.clipped.sum())}/{len(d)}')
