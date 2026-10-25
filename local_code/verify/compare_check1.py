import numpy as np, pandas as pd
from scipy import stats
R = "/Users/perng_mini/AlphaCrops_local/results/"; V = "/Users/perng_mini/AlphaCrops_local/verify/"
orig = pd.read_csv(R + "check1_per_target.csv").rename(columns={"target": "t", "descriptor": "metric"})
mine = pd.read_csv(V + "redo_per_target_spearman.csv")
reg = pd.read_csv(V + "redo_top1_regret.csv")
m = orig.merge(mine, on=["feature", "t", "metric"], suffixes=("_o", "_m")).merge(reg, on=["t", "feature", "metric"])
assert len(m) == 231
m["drho"] = m.rho_o - m.rho_m
print("ρ 差异格（|Δ|>1e-9 或 NaN）：")
print(m[(m.drho.abs() > 1e-9) | m.rho_m.isna()][["feature", "t", "metric", "rho_o", "rho_m"]].to_string())
m["dhit"] = m.top1 - m.hit; m["dreg"] = m.regret_x - 100 * m.regret_y
print("\ntop1/regret 差异格：")
print(m[(m.dhit.abs() > 1e-9) | (m.dreg.abs() > 1e-6)][["feature", "t", "metric", "top1", "hit", "regret_x", "regret_y", "n_tied_min"]].assign(regret_y=lambda q: 100*q.regret_y).round(3).to_string())

# 用原脚本的 NaN→0 复现报告的 c2st_ae−c2st_s2
avg0 = orig.groupby(["metric", "t"]).rho.mean().unstack()
avgN = mine.groupby(["metric", "t"]).rho.mean().unstack()
for a, b in [("c2st_ae", "c2st_s2"), ("mmd_ae", "climate_diff"), ("mmd_ae", "c2st_ae")]:
    for lab, A in [("NaN→0", avg0), ("剔除NaN", avgN)]:
        d = (A.loc[a] - A.loc[b]).dropna()
        print(f"{a}−{b} [{lab}]: Δ={d.mean():+.3f} wins={int((d>0).sum())}/{len(d)} p={stats.wilcoxon(d).pvalue:.4f}")

# 报告第 2-3 行
p = pd.read_csv(R + "pairwise_v2.csv")
print("\n国内均值", (p[p.source == p.target].groupby("feature").ba.mean() * 100).round(2).to_dict())
print("跨国均值", (p[p.source != p.target].groupby("feature").ba.mean() * 100).round(2).to_dict())
d = pd.read_csv(R + "domain_distance.csv")
print(d.drop(columns=["c1", "c2"]).corr(method="spearman").round(2).loc[["mmd_ae"]].to_string())

# Wilcoxon 零差处理敏感性（主比较）
dd = (avgN.loc["mmd_ae"] - avgN.loc["mmd_s2"])
print("\n主比较逐国 Δρ:", dd.round(4).to_dict())
for zm in ["wilcox", "pratt", "zsplit"]:
    print(zm, round(stats.wilcoxon(dd, zero_method=zm).pvalue, 4))
print("exact(剔除零差):", round(stats.wilcoxon(dd[dd != 0], method="exact").pvalue, 4))
# 留一目标国
print("\n留一目标国（AE-MMD − S2-MMD, AE-MMD − geo）:")
for t in dd.index:
    k = dd.drop(t); g = (avgN.loc["mmd_ae"] - avgN.loc["geo_km"]).drop(t)
    print(f"  去掉 {t}: Δ(AE−S2)={k.mean():+.3f} p={stats.wilcoxon(k).pvalue:.3f} | Δ(AE−geo)={g.mean():+.3f} p={stats.wilcoxon(g).pvalue:.3f}")
