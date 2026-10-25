"""
补充核查：
(i)  AE / S2 的 PC1 是什么（与 log_area、stdDev 组均值、mean 组均值的相关）；S2 的 stdDev 组在前 d 个 PC 中的方差份额。
(ii) 重尾对几何类方法的影响：BE、SE、PT 三国仿真 3 万候选池，比较 AE 与 S2 空间下 kcenter / gist_maxmin / gist_margin
     所选样本落在"离池中心最远 1%"中的比例、所选样本面积；GIST 选中的候选类型与 minpair/d_max。
     u 用同一个轻量外国模型（S2 原始特征，50 棵树）的 margin，两种空间共用，只隔离几何。
(iii) 稳健预处理（z 截断到 ±5；分位数正态化）对 S2 d_max/中位距离的影响。
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

CODE = "/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion"
sys.path.insert(0, CODE)
import samplers.base as B  # noqa: E402
import samplers.designs, samplers.uncertainty, samplers.coverage  # noqa: E402,F401
from samplers.uncertainty import _diam_blocked, gist_core, GistMetric  # noqa: E402

COUNTRIES = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']
P = '/Users/perng_mini/AlphaCrops_local/data/dataset_v2.parquet'


def load():
    import pyarrow.parquet as pq
    names = pq.ParquetFile(P).schema_arrow.names
    ae = [c for c in names if c.startswith('ae_')]
    s2 = [c for c in names if c.startswith('s2_')]
    df = pd.read_parquet(P, columns=['country', 'label', 'log_area', 'has_ae', 'has_s2'] + ae + s2)
    df = df[df.has_ae & df.has_s2].reset_index(drop=True)
    return df, ae, s2


def scaled(T, d):
    A = T[:, :d].astype(np.float64)
    return A / np.sqrt(A.var(0).sum() / d)


def dmax_med(S, r):
    i, j = r.integers(0, len(S), 20000), r.integers(0, len(S), 20000)
    med = np.median(np.linalg.norm(S[i] - S[j], axis=1))
    return _diam_blocked(S, np.einsum('ij,ij->i', S, S), 1024)[0] / med


def main():
    df, ae, s2 = load()
    ctry = df.country.to_numpy()
    y = pd.factorize(df.label)[0]
    la = df.log_area.to_numpy()
    X = {'ae': df[ae].to_numpy(np.float32), 's2': df[s2].to_numpy(np.float32)}
    cols = {'ae': ae, 's2': s2}
    r = np.random.default_rng(0)
    idx = r.choice(len(df), 20_000, replace=False)
    print('== (i) PC1 的含义（全局 2 万）==')
    for f in ('ae', 's2'):
        Z = X[f][idx].astype(np.float64)
        Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
        p = PCA(32, random_state=0).fit(Zs)
        T = p.transform(Zs)
        is_std = np.array([c.endswith('stdDev') for c in cols[f]])
        std_mean = Zs[:, is_std].mean(1)
        mean_mean = Zs[:, ~is_std].mean(1)
        for j in range(3):
            print(f'  {f} PC{j + 1}: EVR {p.explained_variance_ratio_[j]:.3f}  corr(log_area) {np.corrcoef(T[:, j], la[idx])[0, 1]:+.2f}'
                  f'  corr(stdDev 组均值) {np.corrcoef(T[:, j], std_mean)[0, 1]:+.2f}'
                  f'  corr(mean 组均值) {np.corrcoef(T[:, j], mean_mean)[0, 1]:+.2f}')
        sh = ((p.components_ ** 2) * p.explained_variance_[:, None])
        for d in (8, 16, 32):
            s = sh[:d].sum(0)
            print(f'  {f} 前 {d} 个 PC 中 stdDev 组的方差份额 {s[is_std].sum() / s.sum():.3f}（stdDev 列占比 {is_std.mean():.2f}）')

    print('\n== (ii) 重尾对几何类方法的影响 ==')
    import lightgbm as lgb
    rows = []
    for tgt in ('BE', 'SE', 'PT'):
        rr = np.random.default_rng(1000 + COUNTRIES.index(tgt))
        foreign = np.concatenate([rr.choice(np.flatnonzero(ctry == c), 5000, replace=False)
                                  for c in COUNTRIES if c != tgt])
        allT = np.flatnonzero(ctry == tgt)
        pool = np.sort(rr.choice(allT, min(30_000, len(allT)), replace=False))
        pool_pos = np.searchsorted(allT, pool)
        m = lgb.train(dict(objective='multiclass', num_class=10, num_leaves=31, learning_rate=0.1, num_threads=2,
                           verbose=-1, seed=0, deterministic=True, force_col_wise=True),
                      lgb.Dataset(X['s2'][foreign], y[foreign]), num_boost_round=50)
        Pt = m.predict(X['s2'][pool])
        T = {}
        for f in ('ae', 's2'):
            Z = np.vstack([X[f][foreign], X[f][allT]]).astype(np.float64)
            mu, sd = Z.mean(0), Z.std(0) + 1e-9
            p = PCA(32, random_state=0).fit((Z - mu) / sd)
            T[f] = p.transform((X[f][pool].astype(np.float64) - mu) / sd).astype(np.float32)
        for f in ('ae', 's2'):
            ctx = B.Ctx(N=len(pool), C=10, T_ae=T[f], T_s2=None, T_xy=None, T_nuts3=None, F_ae=None, F_s2=None,
                        F_xy=None, F_y=y[foreign], F_country=None, P_T=Pt)
            S = scaled(T[f], 32)
            dc = np.linalg.norm(S - np.median(S, 0), axis=1)
            q99 = np.quantile(dc, 0.99)
            for name in ('kcenter', 'gist_maxmin', 'gist_margin', 'kherding', 'fscs'):
                for k in (50, 200):
                    sel = B.REGISTRY[name](ctx, k, 7)
                    rows.append(dict(target=tgt, space=f, method=name, k=k,
                                     frac_top1pct=float((dc[sel] > q99).mean()),
                                     dlogarea=float(np.median(la[pool][sel]) - np.median(la[pool]))))
            M = GistMetric(S)
            dm = M.diameter()
            for alpha in (0.0, 0.9):
                for k in (50, 200):
                    from samplers.base import unc_margin
                    _, info = gist_core(unc_margin(Pt), M, k, alpha=alpha, return_info=True, diameter=dm)
                    mp = M.min_pair(_)
                    rows.append(dict(target=tgt, space=f, method=f'gist_a{alpha}_info', k=k, kind=info['kind'],
                                     d_over_dmax=info['d'] / info['d_max'], minpair_over_med=mp / np.median(
                                         np.linalg.norm(S[rr.integers(0, len(S), 20000)] - S[rr.integers(0, len(S), 20000)], axis=1))))
        print(tgt, 'done', flush=True)
    R = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', 200)
    print(R.round(3).to_string(index=False))
    R.to_csv('extra_geom.csv', index=False)

    print('\n== (iii) 稳健预处理对 S2 重尾的影响（BE 仿真，3 万池）==')
    from sklearn.preprocessing import QuantileTransformer
    tgt = 'BE'
    rr = np.random.default_rng(1000 + COUNTRIES.index(tgt))
    foreign = np.concatenate([rr.choice(np.flatnonzero(ctry == c), 5000, replace=False) for c in COUNTRIES if c != tgt])
    allT = np.flatnonzero(ctry == tgt)
    pool = np.sort(rr.choice(allT, 30_000, replace=False))
    for f in ('ae', 's2'):
        Z = np.vstack([X[f][foreign], X[f][allT]]).astype(np.float64)
        Zp = X[f][pool].astype(np.float64)
        mu, sd = Z.mean(0), Z.std(0) + 1e-9
        out = {}
        for tag, fwd in (('zscore', lambda A: (A - mu) / sd),
                         ('z_clip5', lambda A: np.clip((A - mu) / sd, -5, 5))):
            p = PCA(32, random_state=0).fit(fwd(Z))
            out[tag] = dmax_med(scaled(p.transform(fwd(Zp)), 32), rr)
        qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal', subsample=200_000, random_state=0).fit(Z)
        p = PCA(32, random_state=0).fit(qt.transform(Z))
        out['quantile_normal'] = dmax_med(scaled(p.transform(qt.transform(Zp)), 32), rr)
        print(f, {k: round(v, 2) for k, v in out.items()})


if __name__ == '__main__':
    main()
