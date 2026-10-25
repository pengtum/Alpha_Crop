# 独立重算：通路国家值、通路间配对对比、相关、稳健性。只读结果文件。
import glob, itertools
import numpy as np, pandas as pd
from scipy import stats
OUT = "/Users/perng_mini/AlphaCrops_local/results/guidance_G"
fs = sorted(glob.glob(OUT + "/rows_t*_*.csv"))
R = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
print("files", len(fs), "rows", len(R), "conds", sorted(R.cond.unique()))
print("tier-3 files:", [f for f in fs if "rows_t3" in f])
print("conds with AEg/S2g:", [c for c in R.cond.unique() if 'g_' in str(c)])
GEOM = ['lpm2_nogeo', 'kcenter', 'typiclust', 'kherding']; PRED = ['margin', 'predstrat_srs']; HYB = ['gist_margin', 'badge_c']
K = ['target', 'split', 'rep', 'method', 'clf', 'k']
def dlt(metric='ba_ge30'):
    a = R[R.cond == 'AE'].set_index(K)[metric]; b = R[R.cond == 'S2'].set_index(K)[metric]
    return (a - b).dropna().rename('d').reset_index()
def cv(d, methods, reps=None):
    x = d[d.method.isin(methods)]
    if reps is not None: x = x[x.rep.isin(reps)]
    return x.groupby(['target', 'method', 'clf', 'k']).d.mean().groupby('target').mean()
def tt(v, label):
    v = np.asarray(v, float); n = len(v); m = v.mean(); se = v.std(ddof=1)/np.sqrt(n)
    q = stats.t.ppf(.975, n-1); p = 2*stats.t.sf(abs(m/se), n-1)
    # 精确符号翻转置换检验
    sf = np.array(list(itertools.product([-1, 1], repeat=n)))
    pm = (np.abs((sf*v).mean(1)) >= abs(m) - 1e-12).mean()
    wp = stats.wilcoxon(v).pvalue
    print(f"{label:38s} mean {m:+.3f} 95%CI [{m-q*se:+.3f},{m+q*se:+.3f}] t-p={p:.3f} perm-p={pm:.3f} wilcoxon-p={wp:.3f} >0 {int((v>0).sum())}/{n}")
    return p
d = dlt()
# 覆盖检查
for fam, ms in (('geom', GEOM), ('pred', PRED), ('hyb', HYB)):
    x = d[d.method.isin(ms)]
    print(fam, 'cells', len(x), 'units', x.groupby(['target','split','rep']).ngroups, 'reps', sorted(x.rep.unique()))
vg = cv(d, GEOM); vg0 = cv(d, GEOM, [0]); vp = cv(d, PRED, [0]); vh = cv(d, HYB, [0])
T = pd.DataFrame(dict(geom_all=vg, geom_r0=vg0, pred=vp, hyb=vh)); print(T.round(3).to_string()); print(T.mean().round(3).to_string())
print("\n--- 各通路（国家值 t，df=10）")
ps = {}
for c in T: ps[c] = tt(T[c], c)
# Holm 于三条正式通路（geom 全部重复、pred、hyb）
pv = np.array([ps['geom_all'], ps['pred'], ps['hyb']]); o = np.argsort(pv); adj = np.empty(3)
run = 0
for i, j in enumerate(o): run = max(run, min(1, (3-i)*pv[j])); adj[j] = run
print("Holm 调整 p (geom, pred, hyb):", adj.round(3), " Bonferroni pred:", min(1, 3*ps['pred']).__round__(3))
print("\n--- 通路间配对对比（国家值）")
tt(T.pred - T.geom_r0, 'pred - geom(rep0)')
tt(T.pred - T.geom_all, 'pred - geom(all reps)')
tt(T.pred - T.hyb, 'pred - hyb')
tt(T.pred - (T.geom_r0 + T.hyb)/2, 'pred - mean(geom_r0, hyb)')
tt(T.hyb - T.geom_r0, 'hyb - geom(rep0)')
print("\n--- 相关")
for a, b in (('pred', 'geom_all'), ('pred', 'geom_r0'), ('pred', 'hyb'), ('hyb', 'geom_r0')):
    s = stats.spearmanr(T[a], T[b]); pr = stats.pearsonr(T[a], T[b])
    print(f"{a}~{b}: spearman {s.statistic:+.3f} (p={s.pvalue:.3f}); pearson {pr.statistic:+.3f} (p={pr.pvalue:.3f})")
print("\n--- pred 通路：留一国、去 RO、去 RO+SE")
loo = {c: T.pred.drop(c).mean() for c in T.index}
print('LOO means', {k: round(v, 3) for k, v in loo.items()})
for drop in (['RO'], ['RO', 'SE'], ['PT', 'BE', 'SI'], ['PT', 'SE', 'SK', 'LV']):
    tt(T.pred.drop(drop), 'pred drop ' + '+'.join(drop))
    tt((T.pred - T.geom_r0).drop(drop), 'pred-geom_r0 drop ' + '+'.join(drop))
print("\n--- pred 通路内逐方法")
for m in PRED: tt(cv(d, [m], [0]), 'pred only ' + m)
for m in HYB: tt(cv(d, [m], [0]), 'hyb only ' + m)
print("\n--- pred 通路逐分类器 / k")
x = d[d.method.isin(PRED) & (d.rep == 0)]
for clf in ('s2', 'ae'): tt(x[x.clf == clf].groupby(['target','method','k']).d.mean().groupby('target').mean(), f'pred | clf {clf}')
for k in (50, 200): tt(x[x.k == k].groupby(['target','method','clf']).d.mean().groupby('target').mean(), f'pred | k {k}')
print("\n--- pred 通路其他指标")
for met in ('oa', 'ba_P2', 'ba_P1', 'ba_far', 'ba_i200', 'ba_i250', 'f1_ge30'):
    dm = dlt(met); tt(cv(dm, PRED, [0]), f'pred {met}'); tt(cv(dm, PRED, [0]) - cv(dm, GEOM, [0]), f'pred-geom_r0 {met}')
T.to_csv("/Users/perng_mini/AlphaCrops_local/verify/G_audit/pathway_x/path_country_values.csv")
