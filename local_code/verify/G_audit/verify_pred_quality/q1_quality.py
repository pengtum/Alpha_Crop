"""独立核实：外国模型（AE vs S2 指导）预测质量 与 预测通路 Δ 的关系；ORACLE margin 对照。
只读结果文件，输出到本目录。"""
import sys, os
LE = "/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion"
sys.path.insert(0, LE); os.chdir(LE)
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import analyze_G as A
import guidance_G as G
from guidance_G import pool_positions, PRED

OUT = "/Users/perng_mini/AlphaCrops_local/results/guidance_G"
V = "/Users/perng_mini/AlphaCrops_local/verify/G_audit/verify_pred_quality"
D = G.GData()
R = A.load_rows(OUT)


def ece_top(P, y, nb=15):
    conf = P.max(1); pred = P.argmax(1); corr = (pred == y).astype(float)
    b = np.minimum((conf * nb).astype(int), nb - 1)
    e = 0.0
    for i in range(nb):
        m = b == i
        if m.any():
            e += m.mean() * abs(conf[m].mean() - corr[m].mean())
    return e


def metrics(P, y, big, pi_pool):
    P = P.astype(np.float64)
    P = P / P.sum(1, keepdims=True)
    yh = P.argmax(1)
    s = np.sort(P, 1); unc = 1 - (s[:, -1] - s[:, -2])
    wrong = yh != y
    rec = np.array([(yh[y == c] == c).mean() if (y == c).any() else np.nan for c in range(10)])
    onehot = np.eye(10)[y]
    return dict(
        nll_pool=float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-300, None)).mean()),
        auroc_margin=float(roc_auc_score(wrong, unc)),
        prior_L1_argmax=float(np.abs(np.bincount(yh, minlength=10) / len(y) - pi_pool).sum()),
        prior_L1_mean=float(np.abs(P.mean(0) - pi_pool).sum()),
        ece_pool=ece_top(P, y),
        acc_pool=float((~wrong).mean()),
        ba_pool_big=float(np.nanmean(rec[big])),
        brier_pool=float(((P - onehot) ** 2).sum(1).mean()),
    )


rows = []
for tgt in sorted(R.target.unique()):
    for split in (0, 1):
        S = np.load(f'{OUT}/stage0/{tgt}_s{split}.npz')
        pp = pool_positions(S, split, 0)
        y = D.y[S['allT'][pp]]
        yT = D.y[S['test']]
        big = np.flatnonzero(np.bincount(yT, minlength=10) >= 30)
        pi_pool = np.bincount(y, minlength=10) / len(y)
        for f in ('ae', 's2'):
            m = metrics(S[f'P_T_{f}'][pp], y, big, pi_pool)
            m.update(target=tgt, split=split, feat=f, ece_oof=float(S[f'ece_{f}']), nll_oof=float(S[f'nll_{f}']),
                     temp=float(S[f'temp_{f}']))
            rows.append(m)
        O = np.load(f'{OUT}/oracle/{tgt}_s{split}.npz')
        assert O['P_T'].shape[0] == len(pp)
        m = metrics(O['P_T'], y, big, pi_pool)
        m.update(target=tgt, split=split, feat='oracle')
        rows.append(m)
Q = pd.DataFrame(rows)
Q.to_csv(f'{V}/q1_quality.csv', index=False)
cols = ['nll_pool', 'auroc_margin', 'prior_L1_argmax', 'prior_L1_mean', 'ece_pool', 'acc_pool', 'ba_pool_big',
        'brier_pool', 'ece_oof', 'nll_oof', 'temp']
print('质量水平（22 单元平均）：')
print(Q.groupby('feat')[cols].mean().round(4).to_string())

# 与官方 guidance_quality_G.csv 的一致性
Qo = pd.read_csv(f'{OUT}/guidance_quality_G.csv').set_index(['target', 'split', 'feat'])
Qm = Q[Q.feat != 'oracle'].set_index(['target', 'split', 'feat'])
for c in ['nll_pool', 'auroc_margin', 'prior_L1_argmax', 'ece_oof', 'nll_oof', 'temp']:
    print(f'  与官方 CSV 最大绝对差 {c}: {float((Qm[c] - Qo[c].reindex(Qm.index)).abs().max()):.2e}')

# 越好方向：+1 表示数值越大越好
better = dict(nll_pool=-1, auroc_margin=+1, prior_L1_argmax=-1, prior_L1_mean=-1, ece_pool=-1, acc_pool=+1,
              ba_pool_big=+1, brier_pool=-1, ece_oof=-1, nll_oof=-1)
W = Q[Q.feat.isin(['ae', 's2'])].pivot_table(index=['target', 'split'], columns='feat', values=list(better))
Dq = pd.DataFrame({c: W[(c, 'ae')] - W[(c, 's2')] for c in better})
print('\nAE−S2 质量差（国家值 t，n=11）；"AE 更好的国家数"按越好方向计：')
for c in better:
    v = Dq[c].groupby('target').mean()
    s = A.t_summary(v)
    nb = int(((v * better[c]) > 0).sum())
    print(f'  {c:16s} {s["mean"]:+.4f} [{s["lo95"]:+.4f},{s["hi95"]:+.4f}] p={s["p"]:.3f}  AE 更好 {nb}/11  '
          f'({"AE 更好" if s["mean"] * better[c] > 0 else "S2 更好"})')

# 预测通路 Δ
d = A.deltas(R, 'AE', 'S2', PRED, reps=[0])
print('\n预测通路国家值（复现官方）：', A.fmt(A.t_summary(A.country_values(d))))
du = d.groupby(['target', 'split']).d.mean().rename('d_pred')
dm = d.groupby(['target', 'split', 'method']).d.mean().unstack()
X = Dq.join(du).join(dm)
X.to_csv(f'{V}/q1_merge.csv')
print('\n相关（Δ质量 × Δba）。"支持"= 相关方向与"AE 质量越相对更好→AE 指导越好"一致：')
for lev, XX in (('单元 n=22', X), ('国家 n=11', X.groupby('target').mean())):
    print(f' [{lev}]')
    for yv in ['d_pred', 'margin', 'predstrat_srs']:
        for c in better:
            r, p = stats.pearsonr(XX[c], XX[yv]); rho, prho = stats.spearmanr(XX[c], XX[yv])
            sup = '支持' if r * better[c] > 0 else '相反'
            sup2 = '支持' if rho * better[c] > 0 else '相反'
            print(f'   {yv:14s}~{c:16s} r={r:+.2f} (p={p:.3f},{sup})  rho={rho:+.2f} (p={prho:.3f},{sup2})')

# ORACLE margin 对照
for a, b in (('ORACLE', 'S2'), ('ORACLE', 'AE'), ('AE', 'S2')):
    dd = A.deltas(R, a, b, ['margin'], reps=[0])
    print(f'margin {a}−{b}：', A.fmt(A.t_summary(A.country_values(dd))), ' 格子数', len(dd))
for a, b in (('ORACLE', 'S2'), ('ORACLE', 'AE')):
    for m in ('lpm2_nogeo', 'typiclust'):
        dd = A.deltas(R, a, b, [m], reps=[0])
        print(f'{m} {a}−{b}：', A.fmt(A.t_summary(A.country_values(dd))))
