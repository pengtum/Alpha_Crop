"""诊断 4：LightGBM 重训（表示/构成）与目标头（决策层均衡校准）能否互补？
学习器族：L = β·log p_LGB + (1−β)·log p_head + τ·log π̂；以及把在外国模型上学到的头偏置 γ·b 直接加到 LGB 上。
选样：SRS 与 AE_predstrat_srs（两者的 LGB 概率都已缓存）。"""
import sys, time, hashlib
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/codesign')
from headlib import *  # noqa

y_all = np.load(f'{CACHE}/y.npy')
TAUS = np.round(np.arange(-1.0, 1.51, 0.1), 2)
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
                    for cm in ('none_srs', 'AE_predstrat_srs'):
                        ld = load_sel(unit, *cm.split('_', 1), k)
                        if ld is None or rep != 0:
                            continue  # predstrat_srs 只在重复 0 上有；两种选样都只用重复 0，保证配对
                        sel_pos, sel = ld
                        ys = y_all[sel]
                        lpi = np.log((np.bincount(ys, minlength=C) + 0.5) / (k + 0.5 * C))
                        h = hashlib.md5(np.sort(np.asarray(sel, dtype=np.int64)).tobytes()).hexdigest()[:16]
                        Lg = logp(np.load(f'{GOUT}/probs/{unit}_{f}_k{k}_{h}.npz')['i300'])
                        Lg = Lg - Lg.mean(1, keepdims=True)
                        meta = dict(unit=unit, tgt=tgt, feat=f, k=k, sel=cm)
                        sweep(Lg, yT, big, lpi, dict(meta, learner='lgb'))
                        for hn, (kind, lam) in {'bias': ('bias', 0.01), 'feat03': ('feat', 0.03)}.items():
                            ux = kind == 'feat'
                            pr = fit_head(Zp[sel_pos], ys, kind, Xp[sel_pos] if ux else None, lam, 1.0)
                            Lh = logp(pr(Zt, Xt if ux else None)); Lh = Lh - Lh.mean(1, keepdims=True)
                            for beta in (0.25, 0.5, 0.75):
                                sweep(beta * Lg + (1 - beta) * Lh, yT, big, lpi, dict(meta, learner=f'ens_{hn}_b{beta}'))
                            if kind == 'bias':
                                b = pr.theta[1:1 + C] * (np.bincount(ys, minlength=C) > 0)
                                for gam in (0.5, 1.0):
                                    sweep(Lg + gam * b, yT, big, lpi, dict(meta, learner=f'lgb+bias_g{gam}'))
                            sweep(Lh, yT, big, lpi, dict(meta, learner=hn))
        print(tgt, split, f'{time.time() - t0:.0f}s', flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/diag4_ens_rows.csv', index=False)
