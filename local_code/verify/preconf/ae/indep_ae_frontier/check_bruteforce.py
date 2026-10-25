"""
对 indep_ae_frontier.py 的 D(W4, 锚点 min) 做一次不依赖 step0B.envelope 的暴力复算（纯 Python 逐段循环找 OA = o 的交点）。
输入：partB2/curves_B2.csv（LightGBM Z0、RT200）与本目录 curves_lr_mlp_k0.csv（LR、MLP k=0）；输出只打印并写 check_bruteforce.txt。
"""
import os
import sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
B2 = '/Users/perng_mini/AlphaCrops_local/verify/step0/partB2/curves_B2.csv'
EPS = 1e-9


def ba_at(oa, ba, o):
    """单条曲线：所有相邻 τ 段中包含 o 的段，线性插值 BA；平段（|ΔOA|<EPS）取两端 BA 较大者；取最大；不可达 → None。"""
    best = None
    for j in range(len(oa) - 1):
        o0, o1, b0, b1 = oa[j], oa[j + 1], ba[j], ba[j + 1]
        if not (min(o0, o1) - EPS <= o <= max(o0, o1) + EPS):
            continue
        if abs(o1 - o0) < EPS:
            v = max(b0, b1)
        else:
            lam = min(max((o - o0) / (o1 - o0), 0.0), 1.0)
            v = b0 + lam * (b1 - b0)
        if best is None or v > best:
            best = v
    return best


def d_w4(ca, cs, anchor):
    diffs, n = [], 0
    for q in range(41):
        o = anchor + round(-4.0 + 0.1 * q, 1)
        n += 1
        a, s = ba_at(*ca, o), ba_at(*cs, o)
        if a is not None and s is not None:
            diffs.append(a - s)
    return (np.mean(diffs) if len(diffs) / n >= 0.5 else np.nan), len(diffs) / n


U = pd.read_csv(os.path.join(HERE, 'ae1_units.csv'))
cb = pd.read_csv(B2, float_precision='round_trip')
cm = pd.read_csv(os.path.join(HERE, 'curves_lr_mlp_k0.csv'), float_precision='round_trip')
out = []
for _, r in U.iterrows():
    if r.learner == 'lgbm':
        g = cb[(cb.unit == r.unit) & (cb.arm == r.arm)].sort_values('tau')
        z = cb[(cb.unit == r.unit) & (cb.arm == 'Z0') & (np.isclose(cb.tau, 0))]
        anchor = min(z[z.clf == 'ae'].oa.iloc[0], z[z.clf == 's2'].oa.iloc[0])
    else:
        g = cm[(cm.target == r.target) & (cm.split == r.split) & (cm.learner == r.learner)].sort_values('tau')
        z = g[np.isclose(g.tau, 0)]
        anchor = min(z[z.clf == 'ae'].oa.iloc[0], z[z.clf == 's2'].oa.iloc[0])
    ca = tuple(g[g.clf == 'ae'][c].to_numpy() for c in ('oa', 'ba_ge30'))
    cs = tuple(g[g.clf == 's2'][c].to_numpy() for c in ('oa', 'ba_ge30'))
    d, cov = d_w4(ca, cs, anchor)
    out.append(dict(unit=r.unit, learner=r.learner, arm=r.arm, d_bf=d, d_main=r.D_W4_min, cov_bf=cov, cov_main=r.cov_W4_min))
O = pd.DataFrame(out)
lines = [f'暴力复算 D(W4,min)：{len(O)} 行；max|Δ D| {np.nanmax(np.abs(O.d_bf - O.d_main)):.2e}；'
         f'max|Δ 共同可达率| {np.abs(O.cov_bf - O.cov_main).max():.2e}；缺失一致 {int((O.d_bf.isna() == O.d_main.isna()).sum())}/{len(O)}']
for (l, a), g in O.groupby(['learner', 'arm']):
    lines.append(f'  {l:5s} {a:6s} n={len(g)} max|Δ D| {np.nanmax(np.abs(g.d_bf - g.d_main)):.2e}')
print('\n'.join(lines))
open(os.path.join(HERE, 'check_bruteforce.txt'), 'w').write('\n'.join(lines) + '\n')
