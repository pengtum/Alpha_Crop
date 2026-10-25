"""补充：逐国数值、留一国稳健性、predstrat 相关的"分层质量"（期望选中构成）与 Δ 的关系。"""
import sys, os
LE = "/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion"
sys.path.insert(0, LE); os.chdir(LE)
import numpy as np, pandas as pd
from scipy import stats
import analyze_G as A
import guidance_G as G
from guidance_G import pool_positions, PRED
from samplers.designs import alloc, predstrat_weights

OUT = "/Users/perng_mini/AlphaCrops_local/results/guidance_G"
V = "/Users/perng_mini/AlphaCrops_local/verify/G_audit/verify_pred_quality"
X = pd.read_csv(f'{V}/q1_merge.csv').set_index(['target', 'split'])
Xc = X.groupby('target').mean()
print('逐国（两划分平均）：')
print(Xc[['d_pred', 'margin', 'predstrat_srs', 'nll_pool', 'ece_pool', 'auroc_margin', 'acc_pool', 'ba_pool_big',
          'nll_oof', 'ece_oof']].round(3).to_string())

print('\n留一国（国家层 n=10）Pearson r 范围：')
for c in ['nll_pool', 'ece_pool', 'prior_L1_argmax', 'auroc_margin', 'nll_oof', 'ece_oof']:
    rs = [stats.pearsonr(Xc.drop(t)[c], Xc.drop(t)['d_pred'])[0] for t in Xc.index]
    rhos = [stats.spearmanr(Xc.drop(t)[c], Xc.drop(t)['d_pred'])[0] for t in Xc.index]
    print(f'  d_pred~{c:16s} r∈[{min(rs):+.2f},{max(rs):+.2f}]  rho∈[{min(rhos):+.2f},{max(rhos):+.2f}]')

# 相关矩阵：质量指标之间（是否是同一因素的两面）
print('\n单元层 Δ质量指标之间的 Pearson 相关：')
print(X[['nll_pool', 'ece_pool', 'auroc_margin', 'acc_pool', 'nll_oof', 'ece_oof']].corr().round(2).to_string())

# predstrat 相关：期望选中构成的"真类别覆盖"与"分层纯度"
D = G.GData()
R = A.load_rows(OUT)
rows = []
for tgt in sorted(R.target.unique()):
    for split in (0, 1):
        S = np.load(f'{OUT}/stage0/{tgt}_s{split}.npz')
        pp = pool_positions(S, split, 0)
        y = D.y[S['allT'][pp]]
        yT = D.y[S['test']]
        big = np.flatnonzero(np.bincount(yT, minlength=10) >= 30)
        for f in ('ae', 's2'):
            P = S[f'P_T_{f}'][pp].astype(float)
            h = P.argmax(1)
            Nh = np.bincount(h, minlength=10)
            M = np.zeros((10, 10))
            for j in range(10):
                if Nh[j]:
                    M[j] = np.bincount(y[h == j], minlength=10) / Nh[j]
            for k in (50, 200):
                nh = alloc(Nh, predstrat_weights(P, h, 10, 'sqrt'), k, 2)
                ec = (nh[:, None] * M).sum(0) / k
                rows.append(dict(target=tgt, split=split, feat=f, k=k,
                                 purity=float((nh / k * M.max(1)).sum()),
                                 big_min_exp=float((ec[big] * k).min()),
                                 big_ge2=float(((ec[big] * k) >= 2).mean())))
E = pd.DataFrame(rows)
W = E.pivot_table(index=['target', 'split', 'k'], columns='feat', values=['purity', 'big_min_exp', 'big_ge2'])
De = pd.DataFrame({c: W[(c, 'ae')] - W[(c, 's2')] for c in ['purity', 'big_min_exp', 'big_ge2']})
d = A.deltas(R, 'AE', 'S2', ['predstrat_srs'], reps=[0]).groupby(['target', 'split', 'k']).d.mean()
Z = De.join(d)
print('\npredstrat_srs 期望构成指标 AE−S2（国家值 t）与和 Δba 的相关（n=44 单元×k）：')
for c in De:
    s = A.t_summary(De[c].groupby('target').mean())
    r, p = stats.spearmanr(Z[c], Z.d)
    print(f'  {c:12s} {A.fmt(s, "")}   rho(Δ{c},Δba)={r:+.2f} p={p:.3f}')
