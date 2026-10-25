"""只读复核 altmodels 的 frontier_curves.npz：用 Pareto 包络 E(x)=max{ba: OA>=x}（τ 网格 −3..3）代替 at_oa 的"最接近 OA"匹配。
at_oa 在 OA(τ) 非单调时可能落到 τ<0 的劣支上。这里比较：在对方原生 OA 处，本模型包络能达到的 ba。
注意：包络用测试标签描出，是能力诊断，不是可部署规则。"""
import numpy as np, pandas as pd
from scipy import stats
Z=np.load('/Users/perng_mini/AlphaCrops_local/verify/model_diag/altmodels/frontier_curves.npz')
TAUS=np.round(np.arange(-3,3.001,0.05),2); n0=int(np.flatnonzero(TAUS==0)[0])
def env(C,x):
    ok=C[:,1]>=x-1e-9
    return C[ok,2].max() if ok.any() else np.nan
def tci(v):
    v=np.asarray(v,float); v=v[~np.isnan(v)]; n=len(v); m=v.mean(); h=stats.t.ppf(.975,n-1)*v.std(ddof=1)/np.sqrt(n)
    return f'{m:+.2f} [{m-h:+.2f}, {m+h:+.2f}] {int((v>0).sum())}/{n}'
rows=[]
for key in Z.files:
    tgt,feat,k,model=key.split('|')
    if model=='lgb': continue
    L=Z[f'{tgt}|{feat}|{k}|lgb']; M=Z[key]
    oaL,baL=L[n0,1:]; oaM,baM=M[n0,1:]
    # 在 M 原生 OA 处：M 原生 ba vs LGB 包络
    rows.append(dict(target=tgt,feat=feat,k=int(k),model=model,
        ba_m=baM, oa_m=oaM, ba_lgb=baL, oa_lgb=oaL,
        lgb_env_at_mOA=env(L,oaM), m_env_at_lgbOA=env(M,oaL),
        lgb_ba_tau05=L[np.argmin(abs(TAUS-0.5)),2], lgb_oa_tau05=L[np.argmin(abs(TAUS-0.5)),1],
        m_ba_tau05=M[np.argmin(abs(TAUS-0.5)),2], m_oa_tau05=M[np.argmin(abs(TAUS-0.5)),1],
        m_env_at_lgb05OA=env(M,L[np.argmin(abs(TAUS-0.5)),1]),
        # 公共 OA 网格上的包络差（两者都可达的 OA 范围）
        ))
R=pd.DataFrame(rows)
R.to_csv('frontier_envelope_rows.csv',index=False)
for k in (0,200):
    for model in ('lr','mlp2'):
        g=R[(R.k==k)&(R.model==model)].groupby('target').mean(numeric_only=True)
        print(f'--- {model} k={k} (国家层面, ae/s2 平均) ---')
        print(f'  原生 m − LGB 包络@m原生OA：Δba {tci(g.ba_m-g.lgb_env_at_mOA)}')
        print(f'  m 包络@LGB原生OA − LGB 原生：Δba {tci(g.m_env_at_lgbOA-g.ba_lgb)}  (nan=m 达不到该 OA: {int(R[(R.k==k)&(R.model==model)].m_env_at_lgbOA.isna().sum())} 条)')
        print(f'  m 包络@LGB(τ=.5)OA − LGB(τ=.5)：Δba {tci(g.m_env_at_lgb05OA-g.lgb_ba_tau05)}')
        print(f'  同 τ=0.5：m − LGB Δba {tci(g.m_ba_tau05-g.lgb_ba_tau05)}  ΔOA {tci(g.m_oa_tau05-g.lgb_oa_tau05)}')
