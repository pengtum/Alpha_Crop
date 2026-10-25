# 只读、轻量（不训练树模型）：在固定 300 棵树（c300）的跨境模型缓存概率上重做 BIAS_ORACLE
# （温度 + 偏置，用目标 pool 全部标签拟合，与 analyze_check3.py 完全同一实现），
# 检验"决策层可挽回的上界"是否依赖早停基线。先用早停缓存复现 check3_arms.csv 中的 ORACLE 值作核对。事后，C 级。
import os, sys
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"; os.environ["VECLIB_MAXIMUM_THREADS"]="1"
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import wilcoxon
LE="/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion"
sys.path.insert(0, LE)
from check3_priors import as64, trained_mask, reweight
D="/Users/perng_mini/AlphaCrops_local/results/check3"
OUT="/Users/perng_mini/AlphaCrops_local/verify/reposition/decision_vs_representation/"
C=10; MIN_N=30
CT=['AT','BE','DE','EE','LT','LV','PT','RO','SE','SI','SK']
def P(t,f,m,part): return as64(np.load(os.path.join(D,f'{t}_{f}_{m}_{part}.npy')))
def metrics(y,yhat):
    n=np.bincount(y,minlength=C)
    rec=np.array([(yhat[y==c]==c).mean() if n[c] else np.nan for c in range(C)])
    return 100*float((y==yhat).mean()), 100*float(np.nanmean(rec[n>=MIN_N]))
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
arms=pd.read_csv("/Users/perng_mini/AlphaCrops_local/results/check3_arms.csv"); arms=arms[arms.split=="random"]
cap=pd.read_csv("/Users/perng_mini/AlphaCrops_local/results/check3_capacity_arms.csv")
rows=[]
for t in CT:
    sp=np.load(os.path.join(D,f'{t}_split.npz')); y_pool=sp['y_pool']; y_test=sp['y_test']
    n_pool=np.bincount(y_pool,minlength=C); pi_T=n_pool/n_pool.sum(); present=n_pool>0
    for f in ['ae','s2']:
        for capn,m in [('es','loco'),('c300','loco_c300')]:
            Lp,Lt=P(t,f,m,'pool'),P(t,f,m,'test')
            ora=fit_bcts(Lp,y_pool,np.ones(len(y_pool))); q=ora(Lt)
            oa_o,_=metrics(y_test,q.argmax(1))
            qb=reweight(q,present/present.sum(),pi_T)[0]; _,ba_o=metrics(y_test,qb.argmax(1))
            oa0,ba0=metrics(y_test,Lt.argmax(1))
            rows.append(dict(target=t,feat=f,cap=capn,L0_oa=oa0,L0_ba=ba0,ORA_oa=oa_o,ORA_ba=ba_o))
d=pd.DataFrame(rows)
# 核对：早停 ORACLE 与 check3_arms 一致
chk=arms[arms.arm.isin(['ORACLE_oa','ORACLE_ba','L0'])].pivot_table(index=['target','feat'],columns='arm',values=['oa','ba_ge30'])
e=d[d.cap=='es'].set_index(['target','feat'])
print("核对 早停 ORACLE_oa 最大差", float(np.max(np.abs(e.ORA_oa-chk[('oa','ORACLE_oa')]))),
      " ORACLE_ba 最大差", float(np.max(np.abs(e.ORA_ba-chk[('ba_ge30','ORACLE_ba')]))),
      " L0 oa 最大差", float(np.max(np.abs(e.L0_oa-chk[('oa','L0')]))))
# c300 的本国参照
cw=cap[(cap.cap=='c300')&cap.arm.isin(['Wnat','W0','L0'])].pivot_table(index=['target','feat'],columns='arm',values=['oa','ba_ge30'])
c=d[d.cap=='c300'].set_index(['target','feat'])
print("核对 c300 L0 oa 与 capacity_arms 最大差", float(np.max(np.abs(c.L0_oa-cw[('oa','L0')]))))
c=c.join(pd.DataFrame({'Wnat_oa':cw[('oa','Wnat')],'W0_ba':cw[('ba_ge30','W0')]}))
ew=arms[arms.arm.isin(['Wnat','W0'])].pivot_table(index=['target','feat'],columns='arm',values=['oa','ba_ge30'])
e=d[d.cap=='es'].set_index(['target','feat']).join(pd.DataFrame({'Wnat_oa':ew[('oa','Wnat')],'W0_ba':ew[('ba_ge30','W0')]}))
out=[]
for name,x in [('es',e),('c300',c)]:
    x=x.copy()
    x['oa_total']=x.Wnat_oa-x.L0_oa; x['oa_gain']=x.ORA_oa-x.L0_oa; x['oa_left']=x.Wnat_oa-x.ORA_oa
    x['ba_total']=x.W0_ba-x.L0_ba; x['ba_gain']=x.ORA_ba-x.L0_ba; x['ba_left']=x.W0_ba-x.ORA_ba
    x['cap']=name; out.append(x.reset_index())
    for f in ['ae','s2']:
        z=x.xs(f,level='feat')
        print(f"[{name} {f}] L0 OA {z.L0_oa.mean():.1f} BA {z.L0_ba.mean():.1f} | OA 总损失 {z.oa_total.mean():+.2f}，决策层上界 {z.oa_gain.mean():+.2f}（合并比例 {z.oa_gain.sum()/z.oa_total.sum():.0%}），剩余 {z.oa_left.mean():+.2f} | "
              f"BA 总损失 {z.ba_total.mean():+.2f}，决策层上界 {z.ba_gain.mean():+.2f}（合并比例 {z.ba_gain.sum()/z.ba_total.sum():.0%}），剩余 {z.ba_left.mean():+.2f}")
    for k in ['oa_gain','oa_left','ba_gain','ba_left']:
        v=(x.xs('ae',level='feat')[k]-x.xs('s2',level='feat')[k])
        print(f"   AE−S2 {k}: 均值 {v.mean():+.2f}，AE 更大 {int((v>0).sum())}/11，Wilcoxon p={wilcoxon(v).pvalue:.3f}")
pd.concat(out).to_csv(OUT+"oracle_es_vs_c300_by_country.csv",index=False)
