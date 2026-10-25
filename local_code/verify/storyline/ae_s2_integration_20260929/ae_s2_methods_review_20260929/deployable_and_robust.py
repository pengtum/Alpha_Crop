# 只读：由 step0 B2 已存 τ 曲线（curves_B2.csv，LightGBM，τ∈[-2,2]）计算
# (1) 可部署点 τ=0.5 与 τ=1.0 的 AE−S2（BA、OA），Z0 与 RT200；
# (2) 同 OA（W4_min）AE−S2 与 ΔFA 交互的稳健统计：11 国符号翻转置换 p、对称截尾均值（去最高最低各 1）。
import os; os.environ["OMP_NUM_THREADS"]="2"
import itertools, numpy as np, pandas as pd
from scipy import stats
C=['AT','BE','DE','EE','LT','LV','PT','RO','SE','SI','SK']; C9=[c for c in C if c not in ('RO','PT')]
cur=pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/step0/partB2/curves_B2.csv')
def tsum(v):
    v=pd.Series(v).dropna(); n=len(v); m=v.mean(); se=v.std(ddof=1)/np.sqrt(n); h=stats.t.ppf(.975,n-1)*se
    return f"{m:+.2f} [{m-h:+.2f}, {m+h:+.2f}] >0 {int((v>0).sum())}/{n}"
def signflip_p(v):
    v=np.asarray(pd.Series(v).dropna()); n=len(v); obs=abs(v.mean())
    cnt=0; tot=0
    for s in itertools.product([1,-1],repeat=n):
        tot+=1; cnt+= abs((v*np.array(s)).mean())>=obs-1e-12
    return cnt/tot
def trim(v):
    v=np.sort(np.asarray(pd.Series(v).dropna())); return v[1:-1].mean()
out=[]
for arm in ['Z0','RT200']:
    g=cur[cur.arm==arm]
    for tau in [0.0,0.5,1.0]:
        x=g[np.isclose(g.tau,tau)].pivot_table(index=['target','unit'],columns='clf',values=['oa','ba_ge30'])
        d=pd.DataFrame({'dba':x[('ba_ge30','ae')]-x[('ba_ge30','s2')],'doa':x[('oa','ae')]-x[('oa','s2')]}).groupby('target').mean().reindex(C)
        out.append(f"{arm} τ={tau}: ΔBA 11国 {tsum(d.dba)} | 9国 {tsum(d.loc[C9,'dba'])}；ΔOA 11国 {tsum(d.doa)} | 9国 {tsum(d.loc[C9,'doa'])}")
        if tau==0.5: out.append("   逐国 ΔBA τ=0.5: "+str(d.dba.round(2).to_dict()))
u=pd.read_csv('/private/tmp/ae_s2_evidence/matched_oa_units.csv').groupby('target').W4_min_d.mean().reindex(C)
out.append(f"\n同OA W4_min Z0 AE−S2：t {tsum(u)}；符号翻转精确 p(11国)={signflip_p(u):.4f}；截尾均值(去最高最低)={trim(u):+.2f}；中位数={u.median():+.2f}")
# ΔFA(RT200) 交互的 11 国逐国值：由 fa_rows_B2 重算
fr=pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/step0/partB2/fa_rows_B2.csv')
out.append("fa_rows_B2 列："+str(fr.columns.tolist()[:20]))
txt="\n".join(out); print(txt)
open('/private/tmp/ae_s2_methods_review_20260929/deployable_and_robust_out.txt','w').write(txt+"\n")
