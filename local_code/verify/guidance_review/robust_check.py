"""
重尾的影响范围与修复效果（BE、SE 仿真 3 万池；u/P 用同一个轻量外国模型：S2 原始特征 50 棵树）。
空间：ae_z（现行）、s2_z（现行）、s2_clip5（标准化后截断到 ±5 再 PCA）、ae_clip5。
指标：所选样本落在"S2 标准化空间离中位点最远 1%"（S2 各变体共用此定义）或"AE 空间最远 1%"（AE 各变体）的比例；
GIST 选中的候选类型。
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pca_check import load, part_scale, COUNTRIES  # noqa: E402
import samplers.base as B  # noqa: E402
import samplers.designs, samplers.uncertainty, samplers.coverage  # noqa: E402,F401
from samplers.uncertainty import gist_core, GistMetric  # noqa: E402
from samplers.base import unc_margin  # noqa: E402

METHODS = [('kcenter', {}), ('gist_maxmin', {}), ('gist_margin', {}), ('badge', {}), ('typiclust', {}),
           ('lpm2', {'geo_w': 0.0}), ('predstrat', {'geo_w': 0.0}), ('fass_quota', {}), ('fscs', {}), ('kherding', {})]


def main():
    import lightgbm as lgb
    ctry, y, X, _, _ = load()
    rows = []
    for tgt in ('BE', 'SE'):
        r = np.random.default_rng(1000 + COUNTRIES.index(tgt))
        foreign = np.concatenate([r.choice(np.flatnonzero(ctry == c), 5000, replace=False)
                                  for c in COUNTRIES if c != tgt])
        allT = np.flatnonzero(ctry == tgt)
        pool = np.sort(r.choice(allT, min(30_000, len(allT)), replace=False))
        m = lgb.train(dict(objective='multiclass', num_class=10, num_leaves=31, learning_rate=0.1, num_threads=2,
                           verbose=-1, seed=0, deterministic=True, force_col_wise=True),
                      lgb.Dataset(X['s2'][foreign], y[foreign]), num_boost_round=50)
        Pt = m.predict(X['s2'][pool])
        T, ref = {}, {}
        for f in ('ae', 's2'):
            Z = np.vstack([X[f][foreign], X[f][allT]]).astype(np.float64)
            mu, sd = Z.mean(0), Z.std(0) + 1e-9
            for tag, clip in (('z', None), ('clip5', 5.0)):
                fz = (lambda A: (A - mu) / sd) if clip is None else (lambda A: np.clip((A - mu) / sd, -clip, clip))
                p = PCA(32, random_state=0).fit(fz(Z))
                T[f'{f}_{tag}'] = p.transform(fz(X[f][pool].astype(np.float64))).astype(np.float32)
            S = part_scale(T[f'{f}_z'], 32)
            dc = np.linalg.norm(S - np.median(S, 0), axis=1)
            ref[f] = dc > np.quantile(dc, 0.99)
        for sp, Tg in T.items():
            out = ref[sp[:2]]
            ctx = B.Ctx(N=len(pool), C=10, T_ae=Tg, T_s2=None, T_xy=None, T_nuts3=None, F_ae=None, F_s2=None,
                        F_xy=None, F_y=y[foreign], F_country=None, P_T=Pt)
            for name, hp in METHODS:
                for k in (50, 200):
                    ctx.meta = {}
                    sel = B.REGISTRY[name](ctx, k, 7, **hp)
                    rows.append(dict(target=tgt, space=sp, method=name, k=k, frac_out=float(out[sel].mean())))
            M = GistMetric(part_scale(Tg, 32))
            dm = M.diameter()
            for k in (50, 200):
                _, info = gist_core(unc_margin(Pt), M, k, alpha=0.9, return_info=True, diameter=dm)
                rows.append(dict(target=tgt, space=sp, method='gist_margin_kind', k=k, kind=info['kind'],
                                 d_over_dmax=info['d'] / info['d_max']))
        print(tgt, 'done', flush=True)
    R = pd.DataFrame(rows)
    R.to_csv('robust_check.csv', index=False)
    pd.set_option('display.width', 250)
    a = R[R.frac_out.notna()].pivot_table(index=['method', 'k'], columns=['target', 'space'], values='frac_out')
    print(a.round(2).to_string())
    print(R[R.method == 'gist_margin_kind'][['target', 'space', 'k', 'kind', 'd_over_dmax']].to_string(index=False))


if __name__ == '__main__':
    main()
