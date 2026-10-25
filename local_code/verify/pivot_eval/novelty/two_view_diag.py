# 轻量诊断：AE 与 S2 两个外国模型（k=0）在目标国候选池/测试集上的误差相关性与互补性
# 只读 results/guidance_G/stage0/*.npz 与 data/cache_guidance_G/y.npy；不改任何文件
import os, glob, numpy as np, pandas as pd
os.environ.setdefault('OMP_NUM_THREADS', '1')
R = '/Users/perng_mini/AlphaCrops_local'
y_all = np.load(f'{R}/data/cache_guidance_G/y.npy')
rows = []
def ba_ge30(y, p):
    cls, cnt = np.unique(y, return_counts=True)
    cls = cls[cnt >= 30]
    return 100 * np.mean([(p[y == c] == c).mean() for c in cls])
def auroc(score, pos):
    from scipy.stats import rankdata
    r = rankdata(score); n1 = pos.sum(); n0 = len(pos) - n1
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
for f in sorted(glob.glob(f'{R}/results/guidance_G/stage0/*_s*.npz')):
    tgt, sp = os.path.basename(f)[:-4].split('_s')
    z = np.load(f)
    cand, test = z['cand'], z['test']; nc = len(cand)
    Pa, Ps = z['P_T_ae'].astype(np.float64), z['P_T_s2'].astype(np.float64)
    for part, sl, idx in (('pool', slice(0, nc), cand), ('test', slice(nc, None), test)):
        y = y_all[idx]; pa, ps = Pa[sl], Ps[sl]; pe = (pa + ps) / 2
        ya, ys, ye = pa.argmax(1), ps.argmax(1), pe.argmax(1)
        wa, ws = ya != y, ys != y
        agree = ya == ys
        conf = np.minimum(pa.max(1), ps.max(1))
        hi = agree & (conf >= np.quantile(conf[agree], 0.5)) if agree.any() else agree
        phi = np.corrcoef(wa, ws)[0, 1]
        # 自信且一致的伪标签：类别覆盖
        cls_true = np.unique(y); cls_hi = np.unique(ya[hi])
        marg_a = 1 - np.sort(pa, 1)[:, -1] + np.sort(pa, 1)[:, -2]
        rows.append(dict(target=tgt, split=int(sp), part=part, n=len(y),
            acc_ae=100 * (1 - wa.mean()), acc_s2=100 * (1 - ws.mean()), acc_ens=100 * (ye == y).mean(),
            ba_ae=ba_ge30(y, ya), ba_s2=ba_ge30(y, ys), ba_ens=ba_ge30(y, ye),
            agree=100 * agree.mean(), acc_agree=100 * (ya[agree] == y[agree]).mean(),
            err_ae_in_disagree=100 * wa[~agree].mean(), both_wrong=100 * (wa & ws).mean(),
            p_s2wrong_given_aewrong=100 * ws[wa].mean(), phi_err=phi,
            pseudo_frac=100 * hi.mean(), pseudo_acc=100 * (ya[hi] == y[hi]).mean(),
            pseudo_ncls=len(cls_hi), true_ncls=len(cls_true),
            auroc_disagree_err_ae=auroc((~agree).astype(float) + 1e-6 * marg_a, wa) if part == 'pool' else np.nan,
            auroc_margin_err_ae=auroc(marg_a, wa) if part == 'pool' else np.nan))
df = pd.DataFrame(rows)
out = '/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/novelty/two_view_diag.csv'
df.to_csv(out, index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
g = df.groupby(['part', 'target']).mean(numeric_only=True).drop(columns='split').round(2)
print(g.to_string())
print('\n=== 11 国等权均值 ===')
print(df.groupby('part').mean(numeric_only=True).drop(columns='split').round(3).T.to_string())
t = df[df.part == 'test'].groupby('target').mean(numeric_only=True)
d = t.ba_ens - t[['ba_ae', 'ba_s2']].max(1)
print('\nens - max(ae,s2) ba_ge30 (test, 国家均值):', d.round(2).to_dict(), 'mean', round(d.mean(), 3), 'pos', (d > 0).sum())
