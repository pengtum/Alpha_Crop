# D5（冒烟，4 个单元，只作参考）：两视图校准后验给候选池打伪标签，再与外国池 + 200 个 SRS 标签一起重训 LightGBM（AE 特征）。
# 与 G 同一训练设置（300 轮、r=50、外国频率类别权重、同一种子）；第一个单元额外复现 G 的 SRS 重训以核对设置。
from common_mv import *
import pandas as pd, time, lightgbm as lgb, stacker as st
X = np.load(f'{R}/data/cache_guidance_G/X_ae.npy', mmap_mode='r')
def params(seed):
    return dict(objective='multiclass', num_class=10, learning_rate=0.05, num_leaves=63, min_data_in_leaf=20,
                bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, max_bin=63, force_col_wise=True,
                num_threads=2, seed=seed, deterministic=True, verbose=-1)
def cw_from(yF):
    n = np.bincount(yF, minlength=10).astype(float)
    return np.where(n > 0, len(yF) / (np.count_nonzero(n) * np.maximum(n, 1)), 1.0)
G = pd.read_pickle('/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/headroom_map/G_all.pkl')
rows = []
UNITS = [('AT', 0, 0), ('PT', 0, 0), ('SE', 0, 0), ('BE', 0, 0)]
for i, (t, s, r) in enumerate(UNITS):
    z = np.load(f'{R}/results/guidance_G/stage0/{t}_s{s}.npz')
    nc = len(z['cand']); pp = pool_positions(nc, s, r); allT = z['allT']
    fr, test = z['foreign'], z['test']; yF, yT = Y[fr], Y[test]
    big = np.flatnonzero(np.bincount(yT, minlength=10) >= 30)
    Pa, Ps = z['P_T_ae'].astype(np.float64), z['P_T_s2'].astype(np.float64)
    sp = np.load(f'{R}/results/guidance_G/sel/{t}_s{s}_r{r}_none_srs_k200.npz')['sel_pos']
    idx_pool = allT[pp]; sel = idx_pool[sp]; ys = Y[sel]
    th = st.fit(st._ll(Pa[pp][sp]), st._ll(Ps[pp][sp]), ys, mode=0.5, gw=5.0, gb=1.0, views='both')
    q = st.predict(th, st._ll(Pa[pp]), st._ll(Ps[pp]))
    oa_c, ba_c = st.scores(yT, st.predict(th, st._ll(Pa[nc:]), st._ll(Ps[nc:])), big)
    rest = np.setdiff1d(np.arange(len(pp)), sp)
    yh, qm = q[rest].argmax(1), q[rest].max(1)
    keep = qm >= 0.6
    pl_idx, pl_y, pl_q = idx_pool[rest][keep], yh[keep], qm[keep]
    ytrue = Y[pl_idx]
    info = dict(n_pl=int(keep.sum()), pl_acc=100 * (pl_y == ytrue).mean(), pl_ncls=len(np.unique(pl_y)),
                true_ncls=len(np.unique(Y[idx_pool])))
    cw = cw_from(yF)
    seed = 1000 * s + r
    arms = [('PL', True)] + ([('srs_repro', False)] if i == 0 else [])
    for arm, use_pl in arms:
        t0 = time.time()
        Xtr = [X[fr], X[sel]]; ytr = [yF, ys]; wtr = [cw[yF], 50.0 * cw[ys]]
        if use_pl:
            Xtr.append(X[pl_idx]); ytr.append(pl_y); wtr.append(cw[pl_y] * pl_q)
        m = lgb.train(params(seed), lgb.Dataset(np.vstack(Xtr), np.concatenate(ytr), weight=np.concatenate(wtr)),
                      num_boost_round=300)
        P = m.predict(X[test])
        oa, ba = st.scores(yT, P, big)
        rows.append(dict(target=t, split=s, rep=r, arm=arm, oa=oa, ba=ba, t=time.time() - t0, **info))
        print(rows[-1], flush=True)
    g = G[(G.target == t) & (G.split == s) & (G.rep == r) & (G.clf == 'ae') & (G.tier == 't1')]
    for nm, gg in (('G_k0', g[g.method == 'foreign_only']), ('G_srs200', g[(g.method == 'srs') & (g.k == 200)])):
        rows.append(dict(target=t, split=s, rep=r, arm=nm, oa=gg.oa.values[0], ba=gg.ba_ge30.values[0]))
    rows.append(dict(target=t, split=s, rep=r, arm='cal_both_a0.5', oa=oa_c, ba=ba_c))
pd.DataFrame(rows).to_csv('d5_expand_smoke.csv', index=False)
print(pd.DataFrame(rows).pivot_table(index='target', columns='arm', values=['ba', 'oa']).round(2).to_string())
