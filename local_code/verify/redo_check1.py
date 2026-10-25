# 独立重做检验 1 统计分析（不参考 analyze_check1.py）
import numpy as np, pandas as pd
from scipy import stats
import itertools, sys

R = "/Users/perng_mini/AlphaCrops_local/results/"
p = pd.read_csv(R + "pairwise_v2.csv")
d = pd.read_csv(R + "domain_distance.csv")
METRICS = ["mmd_ae", "mmd_s2", "geo_km", "climate_diff", "c2st_ae", "c2st_s2", "class_js"]
FEATS = ["ae", "s2", "fusion"]

# 对称展开
d2 = pd.concat([d.rename(columns={"c1": "s", "c2": "t"}),
                d.rename(columns={"c1": "t", "c2": "s"})], ignore_index=True)
assert not d2.duplicated(["s", "t"]).any() and len(d2) == 110

dom = p[p.source == p.target].set_index(["target", "feature"]).ba.rename("ba_dom")
x = p[p.source != p.target].rename(columns={"source": "s", "target": "t"})
x = x.merge(d2, on=["s", "t"], how="left", validate="many_to_one")
assert x[METRICS].notna().all().all() and len(x) == 330
x = x.join(dom, on=["t", "feature"])
x["loss"] = x.ba_dom - x.ba             # 迁移损失（越小越好）
x["rel"] = x.ba / x.ba_dom              # 相对保留率
COUNTRIES = sorted(x.t.unique())

def corr(a, b, method):
    if np.unique(a).size < 2 or np.unique(b).size < 2:
        return np.nan
    if method == "spearman":
        return stats.spearmanr(a, b).statistic
    return stats.kendalltau(a, b).statistic  # tau-b

def per_target(yvar="ba", method="spearman", sign=+1):
    """sign=+1: y 越大越好 (ba, rel)；sign=-1: y 越小越好 (loss)"""
    rows = []
    for (t, f), g in x.groupby(["t", "feature"]):
        for m in METRICS:
            rows.append(dict(t=t, feature=f, metric=m,
                             rho=corr(-g[m].values, sign * g[yvar].values, method)))
    return pd.DataFrame(rows)

def summarize(pt, label):
    print(f"\n==== {label} ====")
    tab = pt.pivot_table(index="metric", columns="feature", values="rho", aggfunc="mean").reindex(METRICS)[FEATS]
    # 每个目标国先对三种特征平均（nanmean），再对目标国平均
    ptm = pt.groupby(["t", "metric"]).rho.mean().unstack()  # pandas mean skipna
    tab["mean3(T-first)"] = ptm.mean().reindex(METRICS)
    tab["n_T_valid"] = ptm.notna().sum().reindex(METRICS)
    tab["n_nan_cells"] = pt.groupby("metric").rho.apply(lambda s: s.isna().sum()).reindex(METRICS)
    print(tab.round(3).to_string())
    return ptm

def paired(ptm, a, b, label, seed=0):
    dd = (ptm[a] - ptm[b]).dropna()
    wins = int((dd > 0).sum()); losses = int((dd < 0).sum()); ties = int((dd == 0).sum())
    w = stats.wilcoxon(dd.values) if len(dd) > 0 else None
    rng = np.random.default_rng(seed)
    bs = [dd.values[rng.integers(0, len(dd), len(dd))].mean() for _ in range(5000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"{label}: {a} - {b}: Δρ={dd.mean():+.3f} (median {dd.median():+.3f}), "
          f"wins/losses/ties={wins}/{losses}/{ties} (n={len(dd)}), Wilcoxon p={w.pvalue:.4f}, "
          f"boot95CI=[{lo:+.3f},{hi:+.3f}]")
    return dd

def paired_by_feature(pt, a, b):
    for f in FEATS:
        q = pt[pt.feature == f].pivot(index="t", columns="metric", values="rho")
        dd = (q[a] - q[b]).dropna()
        w = stats.wilcoxon(dd.values)
        print(f"   [{f}] {a}-{b}: Δρ={dd.mean():+.3f} wins={int((dd>0).sum())}/{len(dd)} p={w.pvalue:.4f}")

def top1_regret(yvar="ba", sign=+1):
    rows = []
    for (t, f), g in x.groupby(["t", "feature"]):
        yy = sign * g[yvar].values
        best = yy.max()
        for m in METRICS:
            dist = g[m].values
            mins = np.flatnonzero(dist == dist.min())
            # 距离并列时：取并列源国 y 的平均（随机选的期望），命中取期望
            hit = np.mean([yy[i] == best for i in mins])
            regret = best - yy[mins].mean()
            rows.append(dict(t=t, feature=f, metric=m, hit=hit, regret=regret,
                             n_tied_min=len(mins)))
    r = pd.DataFrame(rows)
    return r

def fe_partial_r2(yvar="ba"):
    out = {}
    for f in FEATS:
        g = x[x.feature == f].copy()
        for m in METRICS:
            # 在目标国内去均值 = 目标国 FE
            yd = g[yvar] - g.groupby("t")[yvar].transform("mean")
            xd = g[m] - g.groupby("t")[m].transform("mean")
            if (xd ** 2).sum() == 0:
                out[(f, m)] = np.nan; continue
            beta = (xd * yd).sum() / (xd ** 2).sum()
            ssr_full = ((yd - beta * xd) ** 2).sum()
            ssr_red = (yd ** 2).sum()
            out[(f, m)] = (1 - ssr_full / ssr_red, beta)
    return out

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    # ---------- 主分析 ----------
    pt = per_target("ba", "spearman")
    pt.to_csv("/Users/perng_mini/AlphaCrops_local/verify/redo_per_target_spearman.csv", index=False)
    ptm = summarize(pt, "Spearman ρ(−dist, BA)")
    print("\n逐目标国（3 特征平均）ρ：")
    print(ptm[METRICS].round(3).to_string())
    print()
    paired(ptm, "mmd_ae", "mmd_s2", "主比较")
    paired_by_feature(pt, "mmd_ae", "mmd_s2")
    paired(ptm, "mmd_ae", "geo_km", "AE-MMD vs 地理")
    paired_by_feature(pt, "mmd_ae", "geo_km")
    paired(ptm, "mmd_ae", "climate_diff", "AE-MMD vs 气候(非NaN目标国)")
    paired(ptm, "mmd_ae", "c2st_ae", "AE-MMD vs AE-C2ST")
    paired(ptm, "mmd_s2", "geo_km", "S2-MMD vs 地理")
    paired(ptm, "class_js", "mmd_ae", "类别JS vs AE-MMD")

    # ---------- top-1 / regret ----------
    r = top1_regret("ba")
    r.to_csv("/Users/perng_mini/AlphaCrops_local/verify/redo_top1_regret.csv", index=False)
    print("\n==== top-1 命中率（11 目标国 × 3 特征 = 33 格，并列取期望）与 regret（BA 点） ====")
    agg = r.groupby(["metric", "feature"]).agg(hit=("hit", "sum"), regret=("regret", "mean")).unstack()
    agg[("hit", "mean3")] = agg["hit"].mean(axis=1)
    agg[("regret", "mean3")] = agg["regret"].mean(axis=1)
    print(agg.reindex(METRICS).round(4).to_string())
    print("并列最小距离出现情况：")
    print(r[r.n_tied_min > 1].groupby("metric").n_tied_min.describe())
    # 随机选源国的基准
    rand = x.groupby(["t", "feature"]).ba.apply(lambda s: s.max() - s.mean()).groupby("feature").mean()
    print("随机选源国的期望 regret：", rand.round(4).to_dict(), "均值", round(rand.mean(), 4),
          " 随机 top-1 期望命中 =", 33 / 10, "/33")
    # regret 的配对比较 AE vs S2
    rr = r.pivot_table(index=["t", "feature"], columns="metric", values="regret")
    dreg = (rr["mmd_s2"] - rr["mmd_ae"]).groupby("t").mean()
    print("regret(S2-MMD)-regret(AE-MMD) 按目标国均值:", round(dreg.mean(), 4),
          "Wilcoxon p=", round(stats.wilcoxon(dreg[dreg != 0]).pvalue if (dreg != 0).sum() > 0 else np.nan, 4),
          "非零目标国数", int((dreg != 0).sum()))

    # ---------- FE 偏 R² ----------
    print("\n==== BA ~ dist + 目标国 FE 的偏 R²（β 符号：负=距离越大 BA 越低） ====")
    fe = fe_partial_r2("ba")
    tab = pd.DataFrame({f: {m: fe[(f, m)][0] if isinstance(fe[(f, m)], tuple) else np.nan for m in METRICS} for f in FEATS})
    tabb = pd.DataFrame({f: {m: fe[(f, m)][1] if isinstance(fe[(f, m)], tuple) else np.nan for m in METRICS} for f in FEATS})
    tab["mean3"] = tab.mean(axis=1)
    print(tab.round(3).to_string()); print("beta:"); print(tabb.to_string())

    # ---------- 稳健性 1：迁移损失 ----------
    print("\n\n######## 稳健性 1：y = 迁移损失 BA(T→T) − BA(S→T)（越小越好） ########")
    ptL = per_target("loss", "spearman", sign=-1)
    ptLm = summarize(ptL, "Spearman ρ(−dist, −loss)")
    print("与主分析逐格最大差异:", np.nanmax(np.abs(ptL.rho.values - pt.rho.values)))
    paired(ptLm, "mmd_ae", "mmd_s2", "主比较(loss)")
    paired(ptLm, "mmd_ae", "geo_km", "AE vs geo(loss)")
    ptR = per_target("rel", "spearman", sign=+1)
    print("相对保留率 BA(S→T)/BA(T→T) 与主分析逐格最大差异:", np.nanmax(np.abs(ptR.rho.values - pt.rho.values)))
    feL = fe_partial_r2("loss"); feR = fe_partial_r2("rel")
    print("FE 偏R²（loss）mmd_ae/mmd_s2/geo:", [round(np.mean([feL[(f, m)][0] for f in FEATS]), 3) for m in ["mmd_ae", "mmd_s2", "geo_km"]])
    print("FE 偏R²（rel）mmd_ae/mmd_s2/geo:", [round(np.mean([feR[(f, m)][0] for f in FEATS]), 3) for m in ["mmd_ae", "mmd_s2", "geo_km"]])
    # 真正会受影响的：跨目标国合并（pooled）相关
    print("\n合并 110 对（不分目标国）的 Spearman（3 特征平均）：")
    for yv, sg in [("ba", 1), ("loss", -1), ("rel", 1)]:
        vals = {m: np.mean([stats.spearmanr(-x[x.feature == f][m], sg * x[x.feature == f][yv]).statistic for f in FEATS]) for m in METRICS}
        print(f"  y={yv:5s}", {k: round(v, 3) for k, v in vals.items()})

    # ---------- 稳健性 2：Kendall τ ----------
    print("\n\n######## 稳健性 2：Kendall τ-b ########")
    ptK = per_target("ba", "kendall")
    ptK.to_csv("/Users/perng_mini/AlphaCrops_local/verify/redo_per_target_kendall.csv", index=False)
    ptKm = summarize(ptK, "Kendall τ(−dist, BA)")
    paired(ptKm, "mmd_ae", "mmd_s2", "主比较(τ)")
    paired_by_feature(ptK, "mmd_ae", "mmd_s2")
    paired(ptKm, "mmd_ae", "geo_km", "AE vs geo(τ)")
    paired(ptKm, "mmd_ae", "climate_diff", "AE vs climate(τ)")
