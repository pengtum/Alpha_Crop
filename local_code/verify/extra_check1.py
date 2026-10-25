import numpy as np, pandas as pd
from scipy import stats
import sys; sys.path.insert(0, "/Users/perng_mini/AlphaCrops_local/verify")
from redo_check1 import x, METRICS, FEATS, p
# PT / RO 细节
for t in ["PT", "RO"]:
    g = x[(x.t == t) & (x.feature == "ae")].set_index("s")[["ba", "mmd_ae", "mmd_s2", "geo_km", "class_js"]]
    g = g.assign(r_ae=g.mmd_ae.rank(), r_s2=g.mmd_s2.rank(), r_geo=g.geo_km.rank(), r_ba=(-g.ba).rank())
    print(t); print(g.sort_values("mmd_ae").round(3).to_string())
# 源国质量混杂：BA(S→T)/BA(S→S)
src = p[p.source == p.target].set_index(["source", "feature"]).ba.rename("ba_src")
y = x.join(src, on=["s", "feature"]); y["rel_src"] = y.ba / y.ba_src
rows = []
for (t, f), g in y.groupby(["t", "feature"]):
    for m in METRICS:
        a = -g[m].values
        rows.append(dict(t=t, f=f, m=m, rho=np.nan if np.unique(a).size < 2 else stats.spearmanr(a, g.rel_src).statistic))
q = pd.DataFrame(rows).groupby(["t", "m"]).rho.mean().unstack()
print("\n额外：y=BA(S→T)/BA(S→S)（按源国国内精度归一）的 mean ρ:", q.mean().round(3).to_dict())
dd = (q.mmd_ae - q.mmd_s2); print("  AE−S2 Δ=%.3f wins=%d/11 p=%.3f" % (dd.mean(), (dd > 0).sum(), stats.wilcoxon(dd).pvalue))
dg = (q.mmd_ae - q.geo_km); print("  AE−geo Δ=%.3f wins=%d/11 p=%.3f" % (dg.mean(), (dg > 0).sum(), stats.wilcoxon(dg).pvalue))
