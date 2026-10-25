# label.py 的抽查核对（不 import label.py / step0B*，独立实现）：L4 的 τ̂（sklearn balanced_accuracy_score）与 W2/W8 的 FA（逐段循环的包络）
import os, sys
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[v]='2'
sys.dont_write_bytecode=True
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
RG='/Users/perng_mini/AlphaCrops_local/results/guidance_G'
OUT='/Users/perng_mini/AlphaCrops_local/verify/preconf/label'
y=np.load('/Users/perng_mini/AlphaCrops_local/data/cache_guidance_G/y.npy')   # guidance_G.GData 的标签缓存（只读）
TAUS=np.round(np.arange(-2,2+1e-9,0.05),2)
L4=pd.read_csv(f'{OUT}/label_l4_rows.csv', float_precision='round_trip')
CUR=pd.read_csv(f'{OUT}/label_curves.csv', float_precision='round_trip')
FR=pd.read_csv(f'{OUT}/label_rows.csv', float_precision='round_trip')
def pool_positions(n_cand, split, rep):
    rng=np.random.default_rng(10_000*split+100*rep+7); return np.sort(rng.choice(n_cand, min(30000,n_cand), replace=False))
bad=0; n=0
for unit in ['AT_s0_r0','DE_s1_r1','SE_s0_r1','RO_s1_r0','LV_s1_r0']:
    t,s,r=unit.split('_'); s=int(s[1]); r=int(r[1])
    S=np.load(f'{RG}/stage0/{t}_s{s}.npz'); nc=len(S['cand'])
    pp=pool_positions(nc,s,r)
    for f in ('ae','s2'):
        P=S[f'P_T_{f}'].astype(np.float64); P/=P.sum(1,keepdims=True)
        PT=P[nc:]; pih=PT.mean(0)   # Z0 的 π̂：softmax(log P)=P
        for k in (50,200,400):
            z=np.load(f'{RG}/sel/{unit}_none_srs_k{k}.npz'); sel=z['sel']; sp=z['sel_pos']
            assert np.array_equal(S['allT'][pp][sp], sel)
            Ps=P[pp][sp]; ys=y[sel]
            with np.errstate(divide='ignore'): lz=np.log(Ps)
            oa=[]; ba=[]
            for tau in TAUS:
                ph=(lz - tau*np.log(pih)).argmax(1)
                oa.append(100*(ph==ys).mean()); ba.append(100*balanced_accuracy_score(ys, ph))
            oa=np.array(oa); ba=np.array(ba); i0=40
            feas=np.flatnonzero(oa>=oa[i0]-2-1e-9); best=ba[feas].max()
            top=feas[ba[feas]>=best-1e-9]; j=top[np.lexsort((TAUS[top],np.abs(TAUS[top])))[0]]
            row=L4[(L4.unit==unit)&(L4.clf==f)&(L4.k==k)&(L4.arm=='Z0')&(L4['mode']=='rule')].iloc[0]
            n+=1
            if abs(row.tau-TAUS[j])>1e-9 or abs(row.ba_lab-ba[j])>1e-9: bad+=1; print('L4 不一致', unit,f,k,row.tau,TAUS[j],row.ba_lab,ba[j])
            # 测试集 OA/BA 于 τ̂：与曲线一致
            c=CUR[(CUR.unit==unit)&(CUR.clf==f)&(CUR.learner=='lgb')&(CUR.arm=='Z0')&(np.isclose(CUR.tau,TAUS[j]))].iloc[0]
            assert c.oa==row.oa_test and c.ba_ge30==row.ba_test
print(f'L4 Z0 τ̂ 独立核对：{n} 个（单元×特征×k），不一致 {bad}')
# W2/W8 FA 独立包络（逐段循环）
def env1(oa,ba,grid,eps=1e-9):
    out=np.full(len(grid),-np.inf)
    for i in range(len(oa)-1):
        o0,o1,b0,b1=oa[i],oa[i+1],ba[i],ba[i+1]
        lo,hi=min(o0,o1)-eps,max(o0,o1)+eps
        for g,x in enumerate(grid):
            if lo<=x<=hi:
                if abs(o1-o0)<eps: v=max(b0,b1)
                else:
                    lm=min(max((x-o0)/(o1-o0),0),1); v=b0+lm*(b1-b0)
                out[g]=max(out[g],v)
    out[~np.isfinite(out)]=np.nan; return out
mx=0; cnt=0
for unit in ['AT_s0_r0','PT_s1_r1','EE_s0_r1']:
    for f in ('ae','s2'):
        for lrn,arm in (('lgb','RTPK200'),('lgb','PK200'),('lr','RT200'),('mlp','RTPK200')):
            c=CUR[(CUR.unit==unit)&(CUR.clf==f)&(CUR.learner==lrn)&(CUR.arm==arm)].sort_values('tau')
            z0=CUR[(CUR.unit==unit)&(CUR.clf==f)&(CUR.learner==lrn)&(CUR.arm=='Z0')&(np.isclose(CUR.tau,0))].oa.iloc[0]
            for w in (2,4,8):
                grid=z0+np.round(np.arange(-w,1e-9,0.1),1)
                b=env1(c.oa.to_numpy(),c.ba_ge30.to_numpy(),grid)
                cov=np.isfinite(b).mean(); fa=np.nanmean(b) if cov>=0.5 else np.nan
                r=FR[(FR.unit==unit)&(FR.clf==f)&(FR.learner==lrn)&(FR.arm==arm)&(FR.width==w)].iloc[0]
                d=abs(fa-r.FA) if np.isfinite(fa) else (0 if np.isnan(r.FA) else 99)
                mx=max(mx,d); cnt+=1
print(f'FA 独立包络核对：{cnt} 个，最大 |Δ| {mx:.2e}')
