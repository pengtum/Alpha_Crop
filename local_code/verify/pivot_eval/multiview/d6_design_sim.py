# D6：在"两视图校准层"这种标签用法下，比较选样设计的期望效果（每单元 20 个种子，配对种子）。
# 设计：srs、predstrat_ae、predstrat_s2、xvstrat(λ=0)、xvstrat_dis(λ=1)、oracle_sqrt（神谕，参照）。
from common_mv import *
import pandas as pd, stacker as st, xvstrat as xv
from samplers.designs import alloc
KS = (50, 100, 200); NS = 20
rows = []
def pi_of(h, n_h, N_h):
    return (n_h / np.maximum(N_h, 1))[h]
for t in COUNTRIES:
    for s in (0, 1):
        z = np.load(f'{R}/results/guidance_G/stage0/{t}_s{s}.npz')
        nc = len(z['cand']); pp = pool_positions(nc, s, 0)
        Pa, Ps = z['P_T_ae'].astype(np.float64), z['P_T_s2'].astype(np.float64)
        yT = Y[z['test']]; big = np.flatnonzero(np.bincount(yT, minlength=10) >= 30)
        yP = Y[z['allT'][pp]]
        LaP, LsP, LaT, LsT = st._ll(Pa[pp]), st._ll(Ps[pp]), st._ll(Pa[nc:]), st._ll(Ps[nc:])
        PaP, PsP = Pa[pp], Ps[pp]
        N = len(pp)
        # 各设计的层与 π
        hx, Hx, disx = xv.xv_strata(PaP, PsP)
        strat = {'predstrat_ae': (PaP.argmax(1), 10, None), 'predstrat_s2': (PsP.argmax(1), 10, None),
                 'xvstrat': (hx, Hx, 0.0), 'xvstrat_dis': (hx, Hx, 1.0), 'oracle_sqrt': (yP, 10, None)}
        for k in KS:
            for seed in range(NS):
                rng = np.random.default_rng(10_000 + 97 * seed + k + 1000 * s)
                designs = {'srs': (np.sort(rng.choice(N, k, replace=False)), None)}
                for nm, (h, H, lam) in strat.items():
                    N_h = np.bincount(h, minlength=H)
                    w = np.sqrt(N_h) * (1.0 + (lam or 0.0) * (disx if lam is not None else 0))
                    n_h = alloc(N_h, w, k, 2)
                    r2 = np.random.default_rng(20_000 + 97 * seed + k + 1000 * s + {"predstrat_ae": 1, "predstrat_s2": 2, "xvstrat": 3, "xvstrat_dis": 4, "oracle_sqrt": 5}[nm] * 131)
                    sel = np.sort(np.concatenate([r2.choice(np.flatnonzero(h == g), n_h[g], replace=False)
                                                  for g in range(H) if n_h[g] > 0]))
                    designs[nm] = (sel, pi_of(h, n_h, N_h)[sel])
                for nm, (sel, pi) in designs.items():
                    ys = yP[sel]
                    variants = [('plain', None)] + ([('ht', 1 / pi)] if pi is not None else [])
                    for vn, sw in variants:
                        for views in (('both', 'ae') if vn == 'plain' else ('both',)):
                            th = st.fit(LaP[sel], LsP[sel], ys, mode=0.5, gw=5.0, gb=1.0, views=views, sw=sw)
                            oa, ba = st.scores(yT, st.predict(th, LaT, LsT), big)
                            rows.append(dict(target=t, split=s, k=k, seed=seed, design=nm, fit=vn, views=views,
                                             oa=oa, ba=ba, n_cls=len(np.unique(ys)),
                                             frac_dis=float((PaP[sel].argmax(1) != PsP[sel].argmax(1)).mean())))
        print(t, s, flush=True)
pd.DataFrame(rows).to_csv('d6_design_rows.csv', index=False)
