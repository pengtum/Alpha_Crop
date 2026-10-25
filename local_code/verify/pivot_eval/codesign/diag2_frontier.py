"""诊断 2：BA–OA 前沿。对每个学习器，用 k 个 SRS 标签的 Jeffreys 先验 π̂ 做后验调节 logit + τ·log π̂，
τ ∈ [-1, 2]，得到 (OA(τ), BA(τ)) 曲线；先在 44 个单元上按 τ 取平均，再比较各学习器的平均前沿。"""
import sys, time, hashlib
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/codesign')
from headlib import *  # noqa

y_all = np.load(f'{CACHE}/y.npy')
TAUS = np.round(np.arange(-1.0, 2.01, 0.1), 2)
HEADS = {'bias_l0.01_a1': ('bias', 0.01, 1.0), 'bias_l0.1_a1': ('bias', 0.1, 1.0),
         'feat_l0.1_a1': ('feat', 0.1, 1.0), 'mat_l1_a1': ('mat', 1.0, 1.0), 'feat_l0.03_a1': ('feat', 0.03, 1.0)}
rows = []
t0 = time.time()
def sweep(L, yT, big, lpi, meta):
    for tau in TAUS:
        oa, ba = scores(yT, (L + tau * lpi).argmax(1), big)
        rows.append(dict(**meta, tau=tau, oa=oa, ba=ba))
for tgt in COUNTRIES:
    for split in (0, 1):
        S = load_stage0(tgt, split)
        n_cand = len(S['cand']); test = S['test']; yT = y_all[test]
        big = np.flatnonzero(np.bincount(yT, minlength=C) >= MIN_N)
        for rep in (0, 1):
            unit = f'{tgt}_s{split}_r{rep}'
            pp = pool_positions(S, split, rep)
            for f in ('ae', 's2'):
                Zall = logp(S[f'P_T_{f}']); Zp, Zt = Zall[pp], Zall[n_cand:]
                Gf = S[f'G_{f}'].astype(np.float64); Gf = Gf / Gf[:, :1].std()
                Xp, Xt = Gf[pp][:, :16], Gf[n_cand:][:, :16]
                for k in (50, 200):
                    sel_pos, sel = load_sel(unit, 'none', 'srs', k)
                    ys = y_all[sel]
                    lpi = np.log((np.bincount(ys, minlength=C) + 0.5) / (k + 0.5 * C))
                    meta = dict(unit=unit, tgt=tgt, feat=f, k=k)
                    sweep(Zt, yT, big, lpi, dict(meta, learner='k0'))
                    h = hashlib.md5(np.sort(np.asarray(sel, dtype=np.int64)).tobytes()).hexdigest()[:16]
                    z = np.load(f'{GOUT}/probs/{unit}_{f}_k{k}_{h}.npz')
                    for it in ('i300', 'i200'):
                        sweep(logp(z[it]), yT, big, lpi, dict(meta, learner=f'lgb_{it}'))
                    for name, (kind, lam, alpha) in HEADS.items():
                        use_x = kind == 'feat'
                        pr = fit_head(Zp[sel_pos], ys, kind, Xp[sel_pos] if use_x else None, lam, alpha)
                        sweep(logp(pr(Zt, Xt if use_x else None)), yT, big, lpi, dict(meta, learner=name))
        print(tgt, split, f'{time.time() - t0:.0f}s', flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/diag2_frontier_rows.csv', index=False)
