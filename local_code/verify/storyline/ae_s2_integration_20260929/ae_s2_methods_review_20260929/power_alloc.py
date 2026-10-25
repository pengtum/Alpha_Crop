# 只读功效计算（不读数据、不训练）：确认运行中 AE–S2 假设与标签族的 α 分配代价
# SD 取自已有报告（复用测试集，偏乐观）；另按 SD×1.25 做悲观情形。
import os; os.environ["OMP_NUM_THREADS"]="2"
import numpy as np
from scipy import stats
def pow_onesided(delta, sd, n, alpha, null=0.0):
    df=n-1; se=sd/np.sqrt(n); tcrit=stats.t.ppf(1-alpha,df); ncp=(delta-null)/se
    return 1-stats.nct.cdf(tcrit,df,ncp)
def mde80(sd,n,alpha_two):
    df=n-1; return (stats.t.ppf(1-alpha_two/2,df)+stats.t.ppf(.8,df))*sd/np.sqrt(n)
def tost_power(delta,sd,n,margin,alpha=0.05,nsim=200000,seed=1):
    rng=np.random.default_rng(seed); df=n-1
    m=rng.normal(delta,sd/np.sqrt(n),nsim); s=sd*np.sqrt(rng.chisquare(df,nsim)/df); se=s/np.sqrt(n)
    h=stats.t.ppf(1-alpha,df)*se
    return np.mean((m-h>-margin)&(m+h<margin))
out=[]
out.append("# HA1 / H_R1 / H-AE1：同 OA ΔBA（AE−S2）>0 或 >1，单侧 t，国家为单位")
for n,sd in [(11,3.73),(9,3.12)]:
    for infl in [1.0,1.25]:
        s=sd*infl
        for d in [5.25 if n==11 else 6.01, 4.0, 3.0, 2.5]:
            row=[f"n={n} SD={s:.2f} δ={d:.2f}"]
            for a in [0.025,0.0125,0.005]:
                row.append(f"P(>0|α={a})={pow_onesided(d,s,n,a):.2f}")
            row.append(f"P(>1|α=.025)={pow_onesided(d,s,n,0.025,null=1.0):.2f}")
            out.append("  "+"；".join(row))
out.append("\n# 标签族主假设（P1 类，9 国 SD 1.10；ΔFA 量级）MDE80 随 α 的变化（双侧）")
for a in [0.05,0.045,0.04,0.025]:
    out.append(f"  α={a}: MDE80={mde80(1.10,9,a):.2f}（相对 α=.05 ×{mde80(1.10,9,a)/mde80(1.10,9,0.05):.3f}）")
out.append("\n# H_R2 类交互（ΔFA(RT200) AE−S2）TOST 功效，9 国 SD 1.05（×1 与 ×1.3），11 国 SD 2.93")
for n,sd in [(9,1.05),(9,1.05*1.3),(11,2.93)]:
    for margin in [0.5,1.0,2.0]:
        r=[f"n={n} SD={sd:.2f} 界±{margin}"]
        for d in [0.0,0.3,0.59]:
            r.append(f"真值{d:+.2f}: {tost_power(d,sd,n,margin):.2f}")
        out.append("  "+"；".join(r))
out.append("\n# 界值相对效应量：9 国 ΔFA(RT200) AE +2.40 / S2 +1.81；A_200 +2.20 / +1.97（summary_B2 / ae_s2_step0B2_paired_report）")
for m in [0.5,1.0,2.0]:
    out.append(f"  ±{m} pp ≈ S2 标签价值 ΔFA(RT200) 的 {m/1.81*100:.0f}%，A_200 的 {m/1.97*100:.0f}%")
txt="\n".join(out); print(txt)
open("/private/tmp/ae_s2_methods_review_20260929/power_alloc_out.txt","w").write(txt+"\n")
