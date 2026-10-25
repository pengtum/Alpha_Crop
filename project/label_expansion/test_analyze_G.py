"""
T20：analyze_G 在已知 μ、γ、τ 的合成行上能还原估计量；REML τ² 与闭式解 / 网格搜索一致；判定表与门控的边界情况。
运行：python test_analyze_G.py
"""
import sys

import numpy as np
import pandas as pd

import analyze_G as A
from guidance_G import E1, PRED, HYB, COUNTRIES_G


def synth_rows(mu=0.3, gamma=0.2, tau=0.3, sigma=1.0, seed=0):
    """ba = 单元基线 + 方法效应 + [AE]·(μ + u_c + γ·s_clf) + 噪声；s_clf = +1（AE 分类器）/ −1（S2 分类器）。"""
    r = np.random.default_rng(seed)
    rows = []
    for c in COUNTRIES_G:
        u_c = r.normal(0, tau)
        for s in (0, 1):
            for rep in (0, 1):
                base = 60 + r.normal(0, 3)
                for clf in ('s2', 'ae'):
                    for k in (0, 50, 100, 200, 400):
                        rows.append(dict(target=c, split=s, rep=rep, cond='none',
                                         method='foreign_only' if k == 0 else 'srs', clf=clf, k=k,
                                         ba_ge30=base + 4 * np.log1p(k / 50) + r.normal(0, sigma)))
                    for k in (50, 200):
                        for m in E1 + (PRED + HYB if rep == 0 else []):
                            me = r.normal(0, 0.5)
                            for cond in ('AE', 'S2'):
                                eff = (mu + u_c + gamma * (1 if clf == 'ae' else -1)) if cond == 'AE' else 0.0
                                rows.append(dict(target=c, split=s, rep=rep, cond=cond, method=m, clf=clf, k=k,
                                                 ba_ge30=base + 4 * np.log1p(k / 50) + me + eff + r.normal(0, sigma)))
    R = pd.DataFrame(rows)
    for m in A.METRICS:
        if m != 'ba_ge30':
            R[m] = R['ba_ge30']
    return R


def test_recovery():
    th, ga, cover, taus = [], [], [], []
    for s in range(200):
        R = synth_rows(seed=s)
        d = A.deltas(R, 'AE', 'S2', E1)
        v = A.country_values(d)
        st = A.t_summary(v)
        th.append(st['mean'])
        cover.append(st['lo95'] < 0.3 < st['hi95'])
        g = d.pivot_table(index=A.UNIT + ['method', 'k'], columns='clf', values='d')
        ga.append(((g['ae'] - g['s2']) / 2).mean())
        taus.append(A.prediction_interval(v.to_numpy(), np.full(len(v), A.within_var(d)))['tau'])
    th, ga = np.array(th), np.array(ga)
    assert abs(th.mean() - 0.3) < 0.03, th.mean()
    assert abs(ga.mean() - 0.2) < 0.01, ga.mean()
    assert 0.92 <= np.mean(cover) <= 0.98, np.mean(cover)
    assert 0.2 < np.mean(taus) < 0.4, np.mean(taus)
    print(f'还原：θ̂ 均值 {th.mean():.3f}（真 0.3），γ̂ {ga.mean():.3f}（真 0.2），95% CI 覆盖 {np.mean(cover):.3f}，'
          f'τ̂ 均值 {np.mean(taus):.3f}（真 0.3）')


def test_reml():
    r = np.random.default_rng(1)
    y = r.normal(0, 0.5, 11)
    v = np.full(11, 0.04)
    assert abs(A.reml_tau2(y, v) - max(0, y.var(ddof=1) - 0.04)) < 1e-6
    v2 = r.uniform(0.01, 0.2, 11)
    grid = np.linspace(0, 2, 20001)

    def nll(t2):
        w = 1 / (v2 + t2)
        mu = (w * y).sum() / w.sum()
        return 0.5 * (np.log(v2 + t2).sum() + np.log(w.sum()) + (w * (y - mu) ** 2).sum())
    best = grid[np.argmin([nll(t) for t in grid])]
    assert abs(A.reml_tau2(y, v2) - best) < 2e-4, (A.reml_tau2(y, v2), best)
    # τ² = 0 的边界
    assert A.reml_tau2(np.zeros(11) + r.normal(0, 0.01, 11), np.full(11, 1.0)) == 0.0
    # 等方差时 HKSJ SE = t 检验 SE，PI 按公式
    pi = A.prediction_interval(y, v)
    assert abs(pi['se_hksj'] - y.std(ddof=1) / np.sqrt(11)) < 1e-9
    from scipy import stats
    h = stats.t.ppf(0.9, 9) * np.sqrt(pi['tau'] ** 2 + pi['se_hksj'] ** 2)
    assert abs(pi['pi_hi'] - (y.mean() + h)) < 1e-9
    print('REML 与 PI：通过')


def test_verdict_and_misc():
    base = dict(n=11, mean=0, se=0.1, p=0.5, n_pos=5)
    assert A.verdict(dict(base, lo95=0.05, hi95=0.5, lo90=0.1, hi90=0.45)) == 'AE 更好'
    assert '小于 0.3' in A.verdict(dict(base, lo95=0.01, hi95=0.3, lo90=0.05, hi90=0.25))
    assert A.verdict(dict(base, lo95=-0.25, hi95=0.25, lo90=-0.2, hi90=0.2)).startswith('等价')
    assert A.verdict(dict(base, lo95=-0.6, hi95=-0.1, lo90=-0.5, hi90=-0.15)).startswith('S2 更好')
    assert A.verdict(dict(base, lo95=-0.5, hi95=0.5, lo90=-0.4, hi90=0.4)).startswith('不确定')
    # 标签当量：线性曲线上可解析
    ks = np.array([0, 50, 100, 200, 400.])
    f = 60 + ks / 100
    assert abs(A.label_equiv(ks, f, 50, 0.3) - 30) < 1e-9
    assert abs(A.label_equiv(ks, f, 200, -0.5) + 50) < 1e-9
    assert A.label_equiv(ks, f, 200, 5) == np.inf
    # SRS 与 foreign_only 不进入任何家族；E1 与 FAMILIES['geom'] 一致
    from guidance_G import FAMILIES
    assert FAMILIES['geom'] == E1 and all('srs' not in v for v in FAMILIES.values())
    # WCB 在 μ=0 时 p 约均匀
    r = np.random.default_rng(3)
    ps = [A.webb_wcb(r.normal(0, 1, 11), B=999, seed=i) for i in range(200)]
    assert 0.02 < np.mean(np.array(ps) < 0.05) < 0.10, np.mean(np.array(ps) < 0.05)
    print('判定表、标签当量、WCB：通过')


def test_filters():
    """层级 3 的融合分类器与 k=1000 行不会混入 θ_E1。"""
    R = synth_rows(seed=3)
    base = A.t_summary(A.country_values(A.deltas(R, 'AE', 'S2', E1)))['mean']
    extra = R[R.cond.isin(['AE', 'S2']) & R.method.isin(E1)].copy()
    fus = extra.assign(clf='fus')
    fus.loc[fus.cond == 'AE', 'ba_ge30'] += 2
    k1k = extra.assign(k=1000)
    k1k.loc[k1k.cond == 'AE', 'ba_ge30'] += 2
    R2 = pd.concat([R, fus, k1k])
    assert A.t_summary(A.country_values(A.deltas(R2, 'AE', 'S2', E1)))['mean'] == base
    d = A.vs_srs(R2, 'AE', E1)
    assert set(d.clf) == {'s2', 'ae'} and set(d.k) == {50, 200}
    print('层级 3 行过滤：通过')


if __name__ == '__main__':
    test_filters()
    test_reml()
    test_verdict_and_misc()
    test_recovery()
    print('T20 全部通过')
    sys.exit(0)
