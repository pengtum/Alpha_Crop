# 只读、轻量（不训练树模型）：在 G 的 10 km 格网 + 1 km 缓冲划分上，用 stage0 已缓存的外国模型概率
# （P_T，300 轮 + 标量温度；覆盖候选集与测试集）重做"决策层上界"：
#   用候选集全部标签拟合温度 + 类偏置（与 analyze_check3.fit_bcts 同一实现），应用到测试集；
#   OA 版 = 自然先验拟合；BA 版 = 拟合后除以候选集先验、回到目标国出现类上的均匀先验（与检验 3 的 ORACLE_ba 同一做法）。
# 与 model_diag 的"只用目标池训练"（tgt300，划分 0）对照，得到决策层可修复量与表征层剩余量。
# 事后分析，测试集已被多次复用（TESTSET_AUDIT）→ C 级。
import os, sys
for v in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS"]: os.environ[v]="1"
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import wilcoxon, ttest_1samp
LE="/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion"
sys.path.insert(0, LE)
from check3_priors import trained_mask, reweight
G="/Users/perng_mini/AlphaCrops_local/results/guidance_G/stage0"
OUT="/Users/perng_mini/AlphaCrops_local/verify/reposition/decision_vs_representation/"
y=np.load("/Users/perng_mini/AlphaCrops_local/data/cache_guidance_G/y.npy")
C=10; CT=['AT','BE','DE','EE','LT','LV','PT','RO','SE','SI','SK']
def fit_bcts(p,y,w):  # 与 analyze_check3.fit_bcts 逐行相同
    k=trained_mask(p); z=np.log(np.clip(p[:,k],1e-12,None))
    Y=np.eye(k.sum())[np.searchsorted(np.flatnonzero(k),y)]; w=w/w.sum()
    def f(th):
        s,b=th[0],th[1:]; l=s*z+b; l=l-l.max(1,keepdims=True); q=np.exp(l); q/=q.sum(1,keepdims=True)
        nll=-(w*np.log(np.clip((q*Y).sum(1),1e-300,None))).sum(); g=(q-Y)*w[:,None]
        return nll, np.r_[(g*z).sum(), g.sum(0)]
    r=minimize(f,np.r_[1.0,np.zeros(k.sum())],jac=True,method='L-BFGS-B',bounds=[(1e-3,None)]+[(None,None)]*k.sum())
    def apply(pp):
        l=r.x[0]*np.log(np.clip(pp[:,k],1e-12,None))+r.x[1:]; l=l-l.max(1,keepdims=True); q=np.exp(l)
        out=np.zeros_like(pp); out[:,k]=q/q.sum(1,keepdims=True); return out
    return apply
def scores(yT,ph,big):  # 与 guidance_G._scores 同口径：OA、ba_ge30（big = 测试集中 ≥30 的类）
    rec=np.array([(ph[yT==c]==c).mean() if (yT==c).any() else np.nan for c in range(C)])
    return 100*float((ph==yT).mean()), 100*float(np.nanmean(rec[big]))
rows=[]
for t in CT:
    for s in (0,1):
        S=np.load(os.path.join(G,f'{t}_s{s}.npz'))
        cand,test=S['cand'],S['test']; yC,yT=y[cand],y[test]; nc=len(cand)
        big=np.bincount(yT,minlength=C)>=30
        nC=np.bincount(yC,minlength=C); piC=nC/nC.sum(); present=nC>0
        for f in ('ae','s2'):
            P=S[f'P_T_{f}'].astype(np.float64); P/=P.sum(1,keepdims=True)
            Pc,Pt=P[:nc],P[nc:]
            P0=S[f'P0test_{f}_i300'].astype(np.float64)
            oa0,ba0=scores(yT,P0.argmax(1),big)
            oa0t,ba0t=scores(yT,Pt.argmax(1),big)   # 温度不改变 argmax，作核对
            ora=fit_bcts(Pc,yC,np.ones(nc)); q=ora(Pt)
            oa_o,_=scores(yT,q.argmax(1),big)
            qb=reweight(q,present/present.sum(),piC)[0]; _,ba_o=scores(yT,qb.argmax(1),big)
            rows.append(dict(target=t,split=s,feat=f,k0_oa=oa0,k0_ba=ba0,chk_oa=oa0t-oa0,chk_ba=ba0t-ba0,ora_oa=oa_o,ora_ba=ba_o))
d=pd.DataFrame(rows)
print("核对：P_T 测试部分与 P0test_i300 的 argmax 指标最大差 OA %.3f BA %.3f（float16 舍入）"%(d.chk_oa.abs().max(),d.chk_ba.abs().max()))
d['dec_oa']=d.ora_oa-d.k0_oa; d['dec_ba']=d.ora_ba-d.k0_ba
ub=pd.read_csv("/Users/perng_mini/AlphaCrops_local/verify/model_diag/fit/table_upper_bound.csv").rename(columns={'tgt':'target'})
d=d.merge(ub[['feat','target','k0_ba','k0_oa','tgt300_ba','tgt300_oa']].rename(columns={'k0_ba':'ub_k0_ba','k0_oa':'ub_k0_oa'}),on=['feat','target'],how='left')
d.to_csv(OUT+"oracle_G_buffered_rows.csv",index=False)
s0=d[d.split==0].copy()
print("核对：划分 0 的 k0 与 model_diag table_upper_bound 的最大差 OA %.2f BA %.2f"%((s0.k0_oa-s0.ub_k0_oa).abs().max(),(s0.k0_ba-s0.ub_k0_ba).abs().max()))
# 国家值：两个划分平均
cv=d.groupby(['target','feat'])[['k0_oa','k0_ba','ora_oa','ora_ba','dec_oa','dec_ba']].mean()
for f in ('ae','s2'):
    z=cv.xs(f,level='feat')
    print(f"[{f}] 11 国（2 划分平均）k0 OA {z.k0_oa.mean():.1f} BA {z.k0_ba.mean():.1f} | 决策层上界 OA {z.dec_oa.mean():+.2f}（{int((z.dec_oa>0).sum())}/11 为正）BA {z.dec_ba.mean():+.2f}（{int((z.dec_ba>0).sum())}/11）")
    zz=s0[s0.feat==f]
    tot_oa=zz.tgt300_oa-zz.k0_oa; tot_ba=zz.tgt300_ba-zz.k0_ba
    print(f"     划分 0：只用目标池 3 万（tgt300）− k0：OA {tot_oa.mean():+.2f} BA {tot_ba.mean():+.2f}；决策层上界 OA {zz.dec_oa.mean():+.2f} BA {zz.dec_ba.mean():+.2f}；"
          f"合并比例 OA {zz.dec_oa.sum()/tot_oa.sum():.0%} BA {zz.dec_ba.sum()/tot_ba.sum():.0%}；剩余（tgt300 − 上界）OA {(zz.tgt300_oa-zz.ora_oa).mean():+.2f} BA {(zz.tgt300_ba-zz.ora_ba).mean():+.2f}")
    m=~zz.target.isin(['RO','PT'])
    print(f"     划分 0 去 RO、PT：合并比例 OA {zz.dec_oa[m].sum()/tot_oa[m].sum():.0%} BA {zz.dec_ba[m].sum()/tot_ba[m].sum():.0%}")
print("\n逐国（划分 0）：")
s0['left_oa']=s0.tgt300_oa-s0.ora_oa; s0['left_ba']=s0.tgt300_ba-s0.ora_ba
print(s0[['target','feat','k0_oa','ora_oa','tgt300_oa','left_oa','k0_ba','ora_ba','tgt300_ba','left_ba']].round(1).sort_values(['feat','target']).to_string(index=False))
print("\nAE − S2（国家值，11 国 t 检验与 Wilcoxon；正 = AE 更大）：")
w=cv.unstack('feat')
for k in ['k0_oa','k0_ba','dec_oa','dec_ba','ora_oa','ora_ba']:
    v=w[(k,'ae')]-w[(k,'s2')]; tt=ttest_1samp(v,0)
    ci=tt.confidence_interval(0.95)
    print(f"  {k}: 均值 {v.mean():+.2f} [{ci.low:+.2f}, {ci.high:+.2f}]，AE 更大 {int((v>0).sum())}/11，t p={tt.pvalue:.3f}，Wilcoxon p={wilcoxon(v).pvalue:.3f}")
w0=s0.set_index(['target','feat'])[['left_oa','left_ba']].unstack('feat')
for k in ['left_oa','left_ba']:
    v=w0[(k,'ae')]-w0[(k,'s2')]; tt=ttest_1samp(v,0); ci=tt.confidence_interval(0.95)
    print(f"  剩余 {k}（划分 0，tgt300 − 上界）: 均值 {v.mean():+.2f} [{ci.low:+.2f}, {ci.high:+.2f}]，AE 剩余更大 {int((v>0).sum())}/11，Wilcoxon p={wilcoxon(v).pvalue:.3f}")
