# 只读：由已有汇总量/逐国值计算 (1) "未改变"类 AE−S2 量能通过 TOST 的最小界值；
# (2) 确认运行中 AE−S2 假设的 MDE80 与 TOST 功效（国家为推断单位，单样本 t）。
# 不训练；线程 ≤2。C 级描述，SD 来自复用测试集，不含换新测试集的变异（偏乐观）。
import os
os.environ["OMP_NUM_THREADS"]="2"; os.environ["OPENBLAS_NUM_THREADS"]="2"
import numpy as np, pandas as pd
from scipy import stats, optimize
E="/private/tmp/ae_s2_evidence"

def from_ci(mean, lo, hi, n):
    df=n-1; t975=stats.t.ppf(.975,df); se=(hi-lo)/(2*t975); sd=se*np.sqrt(n)
    t95=stats.t.ppf(.95,df)
    lo90, hi90 = mean-t95*se, mean+t95*se
    return dict(mean=mean, sd=sd, se=se, lo90=lo90, hi90=hi90, min_margin=max(abs(lo90),abs(hi90)))

def from_vals(v):
    v=np.asarray(v,float); n=len(v); m=v.mean(); se=v.std(ddof=1)/np.sqrt(n)
    t975=stats.t.ppf(.975,n-1)
    return from_ci(m, m-t975*se, m+t975*se, n)

def mde80(sd,n,alpha=.05,power=.8):
    df=n-1; tc=stats.t.ppf(1-alpha/2,df)
    f=lambda d: (1-stats.nct.cdf(tc,df,d/(sd/np.sqrt(n)))+stats.nct.cdf(-tc,df,d/(sd/np.sqrt(n))))-power
    return optimize.brentq(f,1e-6,8*sd/np.sqrt(n))

def tost_power(sd,n,margin,true=0.0,sims=40000,seed=42):
    rng=np.random.default_rng(seed); df=n-1; t95=stats.t.ppf(.95,df)
    x=rng.normal(true,sd,(sims,n)); m=x.mean(1); se=x.std(1,ddof=1)/np.sqrt(n)
    return float(np.mean((m-t95*se>-margin)&(m+t95*se<margin)))

out=[]
def row(name,d,grade,src): out.append((name,d,grade,src))

# ---------- (1) 未改变类：最小 TOST 界值 ----------
cd=pd.read_csv(f"{E}/country_dfeat_4learners.csv").set_index("target")
for L in ["lgb","lr","mlp","rf"]:
    ch=(cd[f"ba_ge30_k400_{L}"]-cd[f"ba_ge30_k0_{L}"])
    row(f"Δ_feat BA 的 k0→400 变化（τ=0，{L}，11 国）",from_vals(ch.values),"C","country_dfeat_4learners.csv")
    row(f"Δ_feat BA 的 k0→400 变化（τ=0，{L}，9 国）",from_vals(ch.drop(["RO","PT"]).values),"C","同上")
z0 ={'AT':4.5,'BE':6.06,'DE':12.19,'EE':5.88,'LT':4.73,'LV':6.25,'PT':-2.19,'RO':5.85,'SE':9.15,'SI':1.29,'SK':4.06}
rt200={'AT':5.19,'BE':6.26,'DE':13.11,'EE':8.39,'LT':4.78,'LV':6.11,'PT':0.16,'RO':4.81,'SE':7.98,'SI':1.43,'SK':5.95}
rt400={'AT':4.2,'BE':5.62,'DE':12.19,'EE':9.06,'LT':5.02,'LV':6.51,'PT':0.99,'RO':4.16,'SE':6.84,'SI':1.98,'SK':6.48}
s=pd.Series(rt400)-pd.Series(z0); s2=pd.Series(rt200)-pd.Series(z0)
row("同 OA 的 AE−S2 BA：RT400 − Z0（11 国）",from_vals(s.values),"C","matched_oa_by_arm_report.txt 逐国")
row("同 OA 的 AE−S2 BA：RT400 − Z0（9 国）",from_vals(s.drop(["RO","PT"]).values),"C","同上")
row("同 OA 的 AE−S2 BA：RT200 − Z0（9 国）",from_vals(s2.drop(["RO","PT"]).values),"C","同上")
for nm,m,lo,hi,n in [("B2 A_200 AE−S2（9 国）",0.23,-0.85,1.31,9),("B2 R_200 AE−S2（9 国）",0.37,-0.41,1.14,9),
                     ("B2 ΔFA(RT200) AE−S2（9 国）",0.59,-0.21,1.40,9),("B2 ΔFA(RT200) AE−S2（11 国）",-0.57,-2.54,1.40,11),
                     ("D1 Δhead BA AE−S2 k=200（11 国）",0.34,-1.03,1.72,11),("D1 决策层可挽回 BA AE−S2（11 国）",0.04,-1.42,1.51,11),
                     ("检验3 域残差 AE−S2 OA（11 国，预注册）",0.0037,-1.2647,1.2722,11),("检验3 域残差 AE−S2 BA（11 国，预注册）",0.7292,-1.8039,3.2624,11),
                     ("检验3 跨境 BA 总损失 AE−S2（11 国）",-0.83,-3.10,1.43,11)]:
    row(nm,from_ci(m,lo,hi,n),"C/B","各报告汇总行")

# ---------- (2) 确认运行 AE−S2 假设的功效 ----------
pw=[]
def prow(name,d,n,margins=(1.0,2.0)):
    r=dict(name=name,n=n,mean_obs=round(d["mean"],2),sd=round(d["sd"],2),mde80=round(mde80(d["sd"],n),2))
    for mg in margins: r[f"TOST±{mg}功效(真值0)"]=round(tost_power(d["sd"],n,mg),2)
    pw.append(r)
prow("同 OA ΔBA AE−S2，Z0（k=0）9 国",from_ci(6.01,3.61,8.41,9),9)
prow("同 OA ΔBA AE−S2，Z0（k=0）11 国",from_ci(5.25,2.75,7.76,11),11)
prow("同 OA ΔBA AE−S2，RT200 9 国",from_ci(6.58,4.14,9.01,9),9)
prow("τ=0 Δ_feat BA（LGB k=0）9 国",from_ci(3.29,0.42,6.16,9),9)
prow("τ=0 Δ_feat BA（LGB k=0）11 国",from_ci(2.42,-0.36,5.20,11),11)
prow("平移检验：同 OA AE−S2 的 RT200−Z0，9 国",from_vals(s2.drop(["RO","PT"]).values),9)
prow("平移检验：同 OA AE−S2 的 RT400−Z0，9 国",from_vals(s.drop(["RO","PT"]).values),9)
prow("B2 ΔFA(RT200) AE−S2（标签价值不变）9 国",from_ci(0.59,-0.21,1.40,9),9)
prow("B2 A_200 AE−S2（先验信息不变）9 国",from_ci(0.23,-0.85,1.31,9),9)
prow("D1 Δhead OA 代价 AE−S2 k=200，11 国",from_ci(1.64,1.20,2.09,11),11)
prow("EM 先验 OA 增益 AE−S2（检验3 随机划分）11 国",from_ci(-4.26,-7.49,-1.03,11),11)
prow("只用预测通路 AE−S2 指导（G）11 国",from_ci(0.340,0.037,0.642,11),11,margins=(0.3,))

pd.set_option("display.width",250); pd.set_option("display.max_colwidth",60)
with open("/private/tmp/ae_s2_rse_design_20260929/ae_s2_bounds_power_out.txt","w") as f:
    f.write("# (1) '未改变'类 AE−S2 量：90% CI 与能通过 TOST 的最小界值（pp）\n")
    for nm,d,g,src in out:
        f.write(f"  {nm}: 均值 {d['mean']:+.2f}，90% CI [{d['lo90']:+.2f}, {d['hi90']:+.2f}]，最小界值 {d['min_margin']:.2f}，SD {d['sd']:.2f}  [{g}; {src}]\n")
    f.write("\n# (2) 确认运行 AE−S2 假设：MDE80（双侧 α=0.05）与 TOST 功效（90% CI 落在 ±界值内，真值 0，模拟 40000 次）\n")
    f.write("#   SD 取自复用测试集的国家值，不含换新测试集的变异，偏乐观；观测均值是 C 级点估计，可能有赢家诅咒。\n")
    f.write(pd.DataFrame(pw).to_string(index=False)+"\n")
print(open("/private/tmp/ae_s2_rse_design_20260929/ae_s2_bounds_power_out.txt").read())
