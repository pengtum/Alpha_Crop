"""诊断 5：头驱动的多轮选样（HRPS）是否比一次性 predstrat_srs 更好（在目标头学习器下评估；不重训 LightGBM）。
HRPS-R：第 1 轮 predstrat_srs(k/R)（外国 P_T 的 ŷ 分层，sqrt 分配，m_min=2，层内 SRS）；
之后每轮用当前全部标签拟合目标头，得到候选池后验 Q，在未选池上按 argmax Q 分层、sqrt 分配、层内 SRS 选 k/R。
直接调用项目 samplers.designs.select_predstrat_srs（只读导入）。"""
import sys, time, dataclasses
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/codesign')
from headlib import *  # noqa
sys.path.insert(0, '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion')
from samplers import base as B, designs as DS  # noqa

y_all = np.load(f'{CACHE}/y.npy')
HEADS = {'bias': ('bias', 0.01), 'feat03': ('feat', 0.03)}


def ctx_of(P):
    return B.Ctx(N=len(P), C=C, T_ae=None, T_s2=None, T_xy=None, T_nuts3=None, F_ae=None, F_s2=None, F_xy=None,
                 F_y=None, F_country=None, P_T=np.asarray(P, dtype=np.float64))


def predstrat(P, k, seed):
    return np.asarray(DS.select_predstrat_srs(ctx_of(P), k, seed), dtype=np.int64)


def hrps(Pp, Zp, Xp, y_pool, k, R, seed, head, deficit=False):
    kind, lam = HEADS[head]
    ks = [k // R + (1 if i < k % R else 0) for i in range(R)]
    S = predstrat(Pp, ks[0], seed)
    for r in range(1, R):
        ux = kind == 'feat'
        pr = fit_head(Zp[S], y_pool[S], kind, Xp[S] if ux else None, lam, 1.0)
        Q = pr(Zp, Xp if ux else None)
        rest = np.setdiff1d(np.arange(len(Pp)), S)
        loc = predstrat(Q[rest], ks[r], seed + 7919 * r)
        S = np.sort(np.r_[S, rest[loc]])
    return S


rows = []
t0 = time.time()
for tgt in COUNTRIES:
    for split in (0, 1):
        St = load_stage0(tgt, split)
        n_cand = len(St['cand']); test = St['test']; yT = y_all[test]
        big = np.flatnonzero(np.bincount(yT, minlength=C) >= MIN_N)
        for rep in (0, 1):
            unit = f'{tgt}_s{split}_r{rep}'
            pp = pool_positions(St, split, rep)
            y_pool = y_all[St['allT'][pp]]
            seed = 1000 * split + rep
            for f in ('ae', 's2'):
                Zall = logp(St[f'P_T_{f}']); Zp, Zt = Zall[pp], Zall[n_cand:]
                Gf = St[f'G_{f}'].astype(np.float64); Gf = Gf / Gf[:, :1].std()
                Xp, Xt = Gf[pp][:, :16], Gf[n_cand:][:, :16]
                Pp = St[f'P_T_{f}'][pp]  # 注意：这里指导与分类器同源（f）；G 的 predstrat_srs 是 AE/S2 两种指导
                for k in (50, 200):
                    sels = {'srs': load_sel(unit, 'none', 'srs', k)[0], 'predstrat': predstrat(Pp, k, seed)}
                    for R in (2, 4):
                        for hn in HEADS:
                            sels[f'hrps{R}_{hn}'] = hrps(Pp, Zp, Xp, y_pool, k, R, seed, hn)
                    for sn, sp in sels.items():
                        ys = y_pool[sp]
                        for hn, (kind, lam) in HEADS.items():
                            if sn.startswith('hrps') and not sn.endswith(hn):
                                continue  # 选样与评估用同一个头（协同设计）
                            ux = kind == 'feat'
                            pr = fit_head(Zp[sp], ys, kind, Xp[sp] if ux else None, lam, 1.0)
                            oa, ba = scores(yT, pr(Zt, Xt if ux else None).argmax(1), big)
                            rows.append(dict(unit=unit, tgt=tgt, feat=f, k=k, sel=sn.replace(f'_{hn}', ''), learner=hn,
                                             ba=ba, oa=oa, n_cls=len(np.unique(ys)),
                                             n_big_cls=int(np.isin(big, ys).sum()), n_big=len(big)))
        print(tgt, split, f'{time.time() - t0:.0f}s', flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/diag5_adaptive_rows.csv', index=False)
