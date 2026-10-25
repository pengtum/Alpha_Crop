"""诊断 1：同一批 SRS 标签（G 的 none_srs 选样），冻结外国模型 + 目标头 vs G 的 LightGBM 重训（r=50, 300 轮）。
问：目标头能否让 k 个随机标签带来更大的 ba_ge30 增益（通道 c）？只做轻量计算（2 线程）。"""
import sys, time, glob
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/codesign')
from headlib import *  # noqa

y_all = np.load(f'{CACHE}/y.npy')
G = pd.concat([pd.read_csv(f) for f in glob.glob(f'{GOUT}/rows_t1_*.csv')])
HEADS = [('bias', 0.01, 1.0), ('bias', 0.1, 1.0), ('bias', 0.01, 0.5), ('bcts', 0.01, 1.0),
         ('mat', 0.1, 1.0), ('mat', 1.0, 1.0), ('feat', 0.1, 1.0), ('feat', 1.0, 1.0), ('bias', 0.01, 0.0)]
rows = []
t0 = time.time()
for tgt in COUNTRIES:
    for split in (0, 1):
        S = load_stage0(tgt, split)
        n_cand = len(S['cand'])
        test = S['test']
        yT = y_all[test]
        big = np.flatnonzero(np.bincount(yT, minlength=C) >= MIN_N)
        for rep in (0, 1):
            unit = f'{tgt}_s{split}_r{rep}'
            pp = pool_positions(S, split, rep)
            idx_pool = S['allT'][pp]
            for f in ('ae', 's2'):
                Zall = logp(S[f'P_T_{f}'])
                Zp, Zt = Zall[pp], Zall[n_cand:]
                Gf = S[f'G_{f}'].astype(np.float64)
                Gf = Gf / Gf[:, :1].std()
                Xp, Xt = Gf[pp][:, :16], Gf[n_cand:][:, :16]
                oa0, ba0 = scores(yT, Zt.argmax(1), big)
                g0 = G[(G.target == tgt) & (G.split == split) & (G.rep == rep) & (G.clf == f) & (G.k == 0)]
                rows.append(dict(unit=unit, tgt=tgt, split=split, rep=rep, feat=f, k=0, head='k0', oa=oa0, ba=ba0,
                                 ba_G_k0=g0.ba_ge30.iloc[0] if len(g0) else np.nan))
                for k in (50, 100, 200, 400):
                    s = load_sel(unit, 'none', 'srs', k)
                    sel_pos, sel = s
                    assert np.array_equal(idx_pool[sel_pos], sel)
                    ys = y_all[sel]
                    g = G[(G.target == tgt) & (G.split == split) & (G.rep == rep) & (G.clf == f) & (G.k == k)
                          & (G.method == 'srs')]
                    rows.append(dict(unit=unit, tgt=tgt, split=split, rep=rep, feat=f, k=k, head='lgb_r50',
                                     oa=g.oa.iloc[0], ba=g.ba_ge30.iloc[0]))
                    rows.append(dict(unit=unit, tgt=tgt, split=split, rep=rep, feat=f, k=k, head='lgb_r50_i200',
                                     oa=g.oa_i200.iloc[0], ba=g.ba_i200.iloc[0]))
                    for kind, lam, alpha in HEADS:
                        pr = fit_head(Zp[sel_pos], ys, kind, Xp[sel_pos] if kind == 'feat' else None, lam, alpha)
                        P = pr(Zt, Xt if kind == 'feat' else None)
                        oa, ba = scores(yT, P.argmax(1), big)
                        rows.append(dict(unit=unit, tgt=tgt, split=split, rep=rep, feat=f, k=k,
                                         head=f'{kind}_l{lam}_a{alpha}', oa=oa, ba=ba))
                    P = fit_gbft(Zp[sel_pos], ys, Xp[sel_pos], Zt, Xt)
                    oa, ba = scores(yT, P.argmax(1), big)
                    rows.append(dict(unit=unit, tgt=tgt, split=split, rep=rep, feat=f, k=k, head='gbft', oa=oa, ba=ba))
        print(tgt, split, f'{time.time() - t0:.0f}s', flush=True)
R = pd.DataFrame(rows)
R.to_csv(f'{OUT}/diag1_srs_rows.csv', index=False)
