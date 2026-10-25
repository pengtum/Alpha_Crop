"""
samplers/uncertainty.py（M4 margin/熵/多轮、M6 GIST 与 k-center、M16 k-DPP、M7 BADGE）的单元测试。

运行：python samplers/tests/test_uncertainty_diversity.py    （也可用 pytest）
环境变量：SKIP_SLOW=1 跳过 N=30000、k=1000 的耗时测试。
交叉核对的参考实现：gist-select 0.1.0（导入名 gist）、dppy 0.3.3、scikit-activeml 1.0.0；未安装时相应测试跳过。
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '2')
os.environ.setdefault('LGBM_THREADS', '1')

import itertools  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(os.path.dirname(HERE))
if CODE not in sys.path:
    sys.path.insert(0, CODE)

import numpy as np  # noqa: E402
from scipy.spatial.distance import cdist, pdist, squareform  # noqa: E402

from samplers.base import REGISTRY, X_of, make_toy_ctx, unc_ent, unc_margin  # noqa: E402
from samplers import uncertainty as U  # noqa: E402

ALL = ['margin', 'entropy', 'margin_prior', 'margin_r2', 'margin_r4', 'gist_margin', 'gist_maxmin', 'gist_geo',
       'kcenter', 'kcenter_foreign', 'badge', 'badge_ae_s2', 'kdpp', 'kdpp_div']
RANDOM = ['kcenter', 'badge', 'badge_ae_s2', 'kdpp', 'kdpp_div']
# 单元测试里多轮版用轻量训练配置（机制与默认配置相同），耗时测试用默认配置
FAST_HP = {'margin_r2': dict(n_trees=20, calib_folds=2), 'margin_r4': dict(n_trees=20, calib_folds=2)}

_CTX = {}


def toy(N=3000, n_F=5000, seed=0):
    key = (N, n_F, seed)
    if key not in _CTX:
        _CTX[key] = make_toy_ctx(N=N, n_F=n_F, seed=seed)
    return _CTX[key]


def run(name, ctx, k, seed, **hp):
    return REGISTRY[name](ctx, k, seed, **{**FAST_HP.get(name, {}), **hp})


def _have(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


class Skip(Exception):
    pass


def skip(msg):
    try:
        import pytest
        pytest.skip(msg)
    except ImportError:
        raise Skip(msg)


# ---------------------------------------------------------------------
# 通用契约
# ---------------------------------------------------------------------

def test_registered():
    for nm in ALL:
        assert nm in REGISTRY, nm
    assert U.USES_ORACLE == {'margin_r2', 'margin_r4'}
    assert U.DETERMINISTIC.isdisjoint(RANDOM)


def test_contract_len_unique_sorted():
    ctx = toy()
    for nm in ALL:
        for k in (50, 200):
            idx = run(nm, ctx, k, 0)
            assert isinstance(idx, np.ndarray) and idx.dtype == np.int64, nm
            assert len(idx) == k and len(np.unique(idx)) == k, (nm, k)
            assert np.all(np.diff(idx) > 0) and idx.min() >= 0 and idx.max() < ctx.N, nm


def test_reproducible_same_seed():
    ctx = toy()
    for nm in ALL:
        a = run(nm, ctx, 60, 7)
        b = run(nm, ctx, 60, 7)
        assert np.array_equal(a, b), nm


def test_seed_dependence():
    """随机方法不同种子结果不同；确定性方法与种子无关。"""
    ctx = toy()
    for nm in RANDOM:
        a, b = run(nm, ctx, 100, 0), run(nm, ctx, 100, 1)
        assert not np.array_equal(a, b), nm
    for nm in U.DETERMINISTIC:
        a, b = run(nm, ctx, 100, 0), run(nm, ctx, 100, 1)
        assert np.array_equal(a, b), nm


def test_no_meta_written():
    """这些方法都不是概率设计（没有闭式 π），不应写 ctx.meta。"""
    ctx = make_toy_ctx(N=1500, n_F=3000, seed=3)
    for nm in ALL:
        run(nm, ctx, 50, 0)
    assert all(nm not in ctx.meta for nm in ALL), ctx.meta.keys()


def test_no_label_access():
    """删掉 _true_y、把 oracle 换成抛异常的函数：非 oracle 方法照常运行；oracle 方法只经 oracle 读标签。"""
    ctx = make_toy_ctx(N=2000, n_F=3000, seed=5)
    y_true = ctx._true_y
    del ctx._true_y

    class Leak(Exception):
        pass

    def boom(idx):
        raise Leak('selector read labels')

    ctx.oracle = boom
    for nm in ALL:
        if nm in U.USES_ORACLE:
            try:
                run(nm, ctx, 50, 0)
            except Leak:
                pass
            else:
                raise AssertionError(f'{nm} 应当经 oracle 读标签')
        else:
            idx = run(nm, ctx, 50, 0)
            assert len(idx) == 50
    ctx.oracle = None
    for nm in U.USES_ORACLE:
        try:
            run(nm, ctx, 50, 0)
        except RuntimeError:
            pass
        else:
            raise AssertionError(nm)
    # oracle 方法：只揭示最终入选、且属于前几轮的下标；调用次数 = rounds − 1
    calls = []

    def oracle(idx):
        calls.append(np.asarray(idx).copy())
        return y_true[np.asarray(idx)]

    ctx.oracle = oracle
    idx = run('margin_r4', ctx, 50, 0)
    assert len(calls) == 3 and [len(c) for c in calls] == [12, 12, 12]
    revealed = np.concatenate(calls)
    assert len(np.unique(revealed)) == 36 and np.isin(revealed, idx).all()


# ---------------------------------------------------------------------
# M4
# ---------------------------------------------------------------------

def test_margin_entropy_topk():
    ctx = toy()
    for nm, f in (('margin', unc_margin), ('entropy', unc_ent)):
        s = f(ctx.P_T)
        for k in (50, 200):
            idx = run(nm, ctx, k, 0)
            thr = np.sort(s)[::-1][k - 1]
            assert np.all(s[idx] >= thr - 1e-12), nm
            assert set(idx) == set(np.argsort(-s)[:k]), nm


def _fixed_proba_clf(P):
    """scikit-activeml 分类器包装：X 的第 0 列是行号，predict_proba 返回 P 对应行（不训练）。"""
    from sklearn.base import BaseEstimator, ClassifierMixin
    from skactiveml.classifier import SklearnClassifier

    class Lookup(ClassifierMixin, BaseEstimator):
        def __init__(self, P=None):
            self.P = P

        def fit(self, X, y, sample_weight=None):
            self.classes_ = np.arange(self.P.shape[1])
            return self

        def predict_proba(self, X):
            return self.P[np.asarray(X)[:, 0].astype(int)]

        def predict(self, X):
            return self.predict_proba(X).argmax(1)

    C = P.shape[1]
    clf = SklearnClassifier(Lookup(P), classes=np.arange(C), missing_label=-1)
    Xi = np.arange(len(P), dtype=float)[:, None]
    y0 = np.full(len(P), -1)
    y0[:C] = np.arange(C)
    clf.fit(Xi, y0)
    return clf, Xi


def test_margin_vs_skactiveml():
    """与 scikit-activeml 1.0.0 UncertaintySampling（margin_sampling / entropy）的批量 top-k 比对：集合应完全相同。"""
    if not _have('skactiveml'):
        skip('scikit-activeml 未安装')
    from skactiveml.pool import UncertaintySampling
    ctx = toy()
    clf, Xi = _fixed_proba_clf(np.asarray(ctx.P_T, dtype=np.float64))
    y = np.full(ctx.N, -1)
    for nm, meth in (('margin', 'margin_sampling'), ('entropy', 'entropy')):
        for k in (50, 200):
            q = UncertaintySampling(method=meth, random_state=0, missing_label=-1).query(
                Xi, y, clf, fit_clf=False, batch_size=k)
            assert set(np.asarray(q)) == set(run(nm, ctx, k, 0)), (nm, k)


def test_prior_adj_em():
    ctx = toy()
    p_true = np.bincount(ctx._true_y, minlength=ctx.C) / ctx.N
    pF = U._resolve_p_train(ctx, 'foreign')
    p_hat = U.em_prior(ctx.P_T, pF)
    assert abs(p_hat.sum() - 1) < 1e-9 and (p_hat >= 0).all()
    # 零标签 EM 应比外国先验更接近目标真实先验（玩具数据外国先验近似均匀，目标先验偏斜）
    assert np.abs(p_hat - p_true).sum() < np.abs(pF - p_true).sum()
    Q = U.prior_adjust(ctx.P_T, p_hat, pF)
    assert np.allclose(Q.sum(1), 1)
    a, b = run('margin', ctx, 200, 0), run('margin_prior', ctx, 200, 0)
    assert not np.array_equal(a, b)
    assert set(b) == set(np.argsort(-unc_margin(Q))[:200])


def test_multiround_mechanics():
    """第一轮 = 单轮 margin 的前 k_1 个；自定义 fit_predict 钩子被调用且其输出决定后续轮。"""
    ctx = toy()
    k = 100
    first = set(run('margin', ctx, 50, 0))
    got = {}

    def hook(c, S, yS, sd):
        got['S'] = np.asarray(S).copy()
        got['y'] = np.asarray(yS).copy()
        P = np.asarray(c.P_T)[::-1].copy()          # 故意换成另一套概率
        return P, np.full(c.C, 1.0 / c.C)

    idx = REGISTRY['margin_r2'](ctx, k, 0, fit_predict=hook)
    assert set(got['S']) == first and np.array_equal(got['y'], ctx._true_y[got['S']])
    u2 = unc_margin(np.asarray(ctx.P_T)[::-1])
    u2[list(first)] = -np.inf
    assert set(idx) == first | set(np.argsort(-u2)[:50])
    # 内置 LightGBM 钩子：输出是合法的概率，温度在界内
    S = np.sort(np.asarray(list(first)))
    P, p_tr, info = U.fit_predict_lgbm(ctx, S, ctx._true_y[S], 0, n_trees=20, calib_folds=2, return_info=True)
    assert P.shape == (ctx.N, ctx.C) and np.allclose(P.sum(1), 1) and (P >= 0).all()
    assert abs(p_tr.sum() - 1) < 1e-9 and 0.05 <= info['T'] <= 20
    # 重训后模型在目标池上的精度应不低于外国模型（加了 50 个目标标签）
    acc0 = (np.asarray(ctx.P_T).argmax(1) == ctx._true_y).mean()
    acc1 = (P.argmax(1) == ctx._true_y).mean()
    print(f'    [多轮] 外国模型目标精度 {acc0:.3f}，外国 ∪ 50 个目标标签重训（PCA 特征、20 棵树）{acc1:.3f}，T={info["T"]:.3f}')


def test_temperature_fit_recovers():
    """温度拟合：用 T=2.5 生成标签，应能近似恢复。"""
    r = np.random.default_rng(0)
    z = r.normal(scale=3, size=(20000, 6))
    P = U._softmax(z / 2.5)
    y = (P.cumsum(1) > r.random((20000, 1))).argmax(1)
    T = U.fit_temperature(z, y)
    assert abs(T - 2.5) < 0.15, T


# ---------------------------------------------------------------------
# M6 GIST / k-center
# ---------------------------------------------------------------------

def gist_f(u, D, S, k, alpha, d_max):
    S = list(S)
    mp = 0.0 if len(S) < 2 else min(D[i, j] for i, j in itertools.combinations(S, 2))
    return alpha * u[S].sum() / k + (1 - alpha) * mp / d_max


def exhaustive_opt(u, D, k, alpha, d_max):
    """k ≤ 4 的精确最优值 (OPT_{=k}, OPT_{≤k})。枚举最小距离对 (i, j)，在与 i、j 距离都 ≥ d_ij 的点中
    找 k−2 个两两距离 ≥ d_ij、u 之和最大的点；按上界降序剪枝。"""
    assert 2 <= k <= 4
    n = len(u)
    iu, ju = np.triu_indices(n, 1)
    dij = D[iu, ju]
    top = np.sort(u)[::-1][:k - 2].sum()
    ub = (1 - alpha) * dij / d_max + alpha / k * (u[iu] + u[ju] + top)
    best_eq, best_le = -np.inf, alpha * u.max() / k
    for t in np.argsort(-ub, kind='stable'):
        if ub[t] <= best_eq:
            break
        i, j, d = iu[t], ju[t], dij[t]
        base = (1 - alpha) * d / d_max + alpha / k * (u[i] + u[j])
        m = (D[i] >= d) & (D[j] >= d)
        m[[i, j]] = False
        C = np.flatnonzero(m)
        b1 = u[C].max() if len(C) else None
        b2 = None
        if k == 4 and len(C) >= 2:
            ok = np.triu(D[np.ix_(C, C)] >= d, 1)
            if ok.any():
                b2 = (u[C][:, None] + u[C][None, :])[ok].max()
        if k == 2:
            v_eq = v_le = base
        elif k == 3:
            v_eq = base + alpha / k * b1 if b1 is not None else -np.inf
            v_le = max(base, v_eq)
        else:
            v_eq = base + alpha / k * b2 if b2 is not None else -np.inf
            v_le = max(v_eq, base + alpha / k * b1 if b1 is not None else base)
        best_eq, best_le = max(best_eq, v_eq), max(best_le, v_le)
    # 剪枝以 best_eq 为界：剩余对的 v_le ≤ ub ≤ best_eq ≤ best_le，故 OPT_{≤k} 也精确
    return best_eq, best_le


def test_exhaustive_opt_matches_bruteforce():
    r = np.random.default_rng(1)
    for alpha in (0.0, 0.5, 0.9):
        for k in (3, 4):
            X = r.normal(size=(13, 3))
            u = r.random(13)
            D = squareform(pdist(X))
            dm = D.max()
            eq = max(gist_f(u, D, S, k, alpha, dm) for S in itertools.combinations(range(13), k))
            le = max(gist_f(u, D, S, k, alpha, dm) for s in range(1, k + 1)
                     for S in itertools.combinations(range(13), s))
            e2, l2 = exhaustive_opt(u, D, k, alpha, dm)
            assert abs(eq - e2) < 1e-12 and abs(le - l2) < 1e-12, (alpha, k, eq, e2, le, l2)


def test_gist_vs_exhaustive_n300():
    """n=300、k=4：GIST（不补足）目标值 ≥ (2/3 − ε)·OPT_{≤k}；补足版与 OPT_{=k} 比较。"""
    r = np.random.default_rng(2)
    X = r.normal(size=(300, 5))
    u = r.random(300)
    D = squareform(pdist(X))
    M = U.GistMetric(X)
    dmax, a, b = M.diameter()
    assert abs(dmax - D.max()) < 1e-9
    k, eps = 4, 0.05
    for alpha in (0.0, 0.5, 0.8, 0.9, 0.95):
        opt_eq, opt_le = exhaustive_opt(u, D, k, alpha, dmax)
        S0, info0 = U.gist_core(u, M, k, alpha=alpha, eps=eps, pad=False, return_info=True)
        S1, info1 = U.gist_core(u, M, k, alpha=alpha, eps=eps, pad=True, return_info=True)
        f0 = gist_f(u, D, S0, k, alpha, dmax)
        f1 = gist_f(u, D, S1, k, alpha, dmax)
        assert abs(f0 - info0['f']) < 1e-9 and abs(f1 - info1['f']) < 1e-9
        assert f0 >= (2 / 3 - eps) * opt_le - 1e-12, (alpha, f0, opt_le)
        assert f1 <= opt_eq + 1e-12 and len(S1) == k
        print(f'    [GIST 穷举] α={alpha:.2f}: 不补足 f/OPT≤k = {f0 / opt_le:.4f}；补足 f/OPT=k = {f1 / opt_eq:.4f}')


def test_gist_vs_gist_select():
    """与 gist-select 0.1.0（kclaka/gist-select）比对：同一直径、λ = (1−α)k/(α·d_max) 时，
    不补足的 GIST 应选出相同集合、目标值相同（换算 f = α/k · (Σu + λ·div)）。"""
    try:
        from gist import EuclideanDistance, LinearUtility
        from gist import gist as gist_fn
        from gist import GISTResult  # noqa: F401  确认导入的是 gist-select 而不是同名旧包
    except Exception:
        skip('gist-select 未安装或导入名 gist 被旧包占用')
    r = np.random.default_rng(3)
    X = r.normal(size=(300, 6))
    u = r.random(300)
    M = U.GistMetric(X)
    dmax, a, b = M.diameter()
    for k in (4, 20, 60):
        for alpha in (0.5, 0.8, 0.9, 0.95):
            S, info = U.gist_core(u, M, k, alpha=alpha, eps=0.05, pad=False, return_info=True)
            lam = (1 - alpha) * k / (alpha * dmax)
            res = gist_fn(points=X, utility=LinearUtility(u), distance=EuclideanDistance(), k=k, lam=lam,
                          eps=0.05, seed=0, diameter=(dmax, a, b))
            theirs = res.objective_value * alpha / k
            assert abs(info['f'] - theirs) < 1e-9, (k, alpha, info['f'], theirs)
            assert set(S) == set(res.indices.tolist()), (k, alpha)
    print('    [GIST vs gist-select] n=300，k∈{4,20,60}，α∈{0.5,0.8,0.9,0.95}：集合与目标值全部一致')


def test_gist_properties_toy():
    ctx = toy()
    u = unc_margin(ctx.P_T)
    X = X_of(ctx, 'ae', 32)
    M = U.GistMetric(X)
    for k in (50, 200):
        S, info = U.gist_core(u, M, k, alpha=0.9, eps=0.05, return_info=True)
        top = np.argsort(-u, kind='stable')[:k]
        f_top = 0.9 * u[top].sum() / k + 0.1 * M.min_pair(top) / info['d_max']
        assert info['f'] >= f_top - 1e-12
        assert info['n_cands'] == 2 + 76
        # 每个阈值候选（不补足）的最小成对距离 ≥ 阈值
        for d in U.gist_thresholds(info['d_max'], 0.05)[::15]:
            T = U._greedy_threshold(np.argsort(-u, kind='stable').tolist(), M, k, d * d)
            assert len(T) < 2 or M.min_pair(T) >= d * (1 - 1e-9)
        assert np.array_equal(np.sort(S), REGISTRY['gist_margin'](ctx, k, 0))
    # α = 1 退化为 top-k margin
    S = U.gist_core(u, M, 50, alpha=1.0)
    assert set(S) == set(np.argsort(-u, kind='stable')[:50])
    # gist_margin 的最小间距应大于 top-k margin
    gm = REGISTRY['gist_margin'](ctx, 200, 0)
    assert M.min_pair(gm) >= M.min_pair(np.argsort(-u)[:200])


def test_gist_geo_metric():
    r = np.random.default_rng(4)
    X = r.normal(size=(700, 4))
    xy = r.uniform(0, 100, size=(700, 2))
    sA, sG = 0.7, 5.0
    M = U.GistMetric(X, xy=xy, s_A=sA, s_G=sG)
    Dfull = np.maximum(cdist(X, X) / sA, cdist(xy, xy) / sG)
    assert np.allclose(np.sqrt(M.sq_to_all(17)), Dfull[17], atol=1e-9)
    dmax, a, b = M.diameter()
    assert abs(dmax - Dfull.max()) < 1e-9 and abs(Dfull[a, b] - dmax) < 1e-9
    S = r.choice(700, 30, replace=False)
    assert abs(M.min_pair(S) - Dfull[np.ix_(S, S)][np.triu_indices(30, 1)].min()) < 1e-12
    # knn_scale：与暴力计算一致（N ≤ n_query 时精确）
    m = 7
    s_true = np.median(np.sort(cdist(X, X), axis=1)[:, m])
    assert abs(U.knn_scale(X, m) - s_true) < 1e-9
    # 玩具 ctx：top-k 使 Σu 最大，而 f(GIST) ≥ f(top-k)，故 GIST-geo 在 geo 度量下的最小间距 ≥ top-k 的
    ctx = toy()
    k = 100
    g = REGISTRY['gist_geo'](ctx, k, 0)
    u = unc_margin(ctx.P_T)
    t = np.argsort(-u, kind='stable')[:k]
    Mg = U._gist_metric(ctx, k, metric='geo', s_G=5.0)
    assert Mg.min_pair(g) >= Mg.min_pair(t)
    assert Mg.s_A > 0 and Mg.s_G == 5.0


def test_kcenter_properties():
    ctx = toy()
    X = X_of(ctx, 'ae', 32)
    k = 100
    S = REGISTRY['kcenter'](ctx, k, 0)
    D = cdist(X, X[S])
    cover = D.min(1).max()
    # Gonzalez 性质：最终覆盖半径 ≤ 已选点之间的最小距离
    assert cover <= pdist(X[S]).min() + 1e-9
    rr = np.random.default_rng(0)
    rand_cover = np.median([cdist(X, X[rr.choice(ctx.N, k, replace=False)]).min(1).max() for _ in range(5)])
    assert cover < rand_cover
    # kcenter_foreign：覆盖半径（相对 S ∪ 外国池）≤ 已选点两两距离及其到外国池的距离
    from samplers.base import X_of_foreign
    XF = X_of_foreign(ctx, 'ae', 32)
    S2 = REGISTRY['kcenter_foreign'](ctx, k, 0)
    dF = np.sqrt(U.nearest_foreign_sq(X, XF))
    assert np.allclose(dF, cdist(X, XF).min(1), atol=1e-3)
    cover2 = np.minimum(cdist(X, X[S2]).min(1), dF).max()
    assert cover2 <= min(pdist(X[S2]).min(), dF[S2].min()) + 1e-3
    assert int(np.argmax(dF)) in set(S2)


def test_kcenter_foreign_vs_skactiveml_coreset():
    """与 scikit-activeml CoreSet（k-center 贪心，已标样本作初始中心）比对：外国池作已标样本时选择应相同。"""
    if not _have('skactiveml'):
        skip('scikit-activeml 未安装')
    from skactiveml.pool import CoreSet
    ctx = make_toy_ctx(N=400, n_F=600, seed=9)
    X = X_of(ctx, 'ae', 32)
    from samplers.base import X_of_foreign
    XF = X_of_foreign(ctx, 'ae', 32)
    Xall = np.vstack([X, XF])
    y = np.r_[np.full(len(X), -1), np.asarray(ctx.F_y)]
    q = CoreSet(missing_label=-1, random_state=0).query(Xall, y, batch_size=30)
    ours = REGISTRY['kcenter_foreign'](ctx, 30, 0)
    assert set(np.asarray(q).tolist()) == set(ours.tolist())


# ---------------------------------------------------------------------
# M7 BADGE
# ---------------------------------------------------------------------

def test_badge_distance_identity():
    r = np.random.default_rng(5)
    P = r.dirichlet(np.ones(10), size=50)
    h = r.normal(size=(50, 32))
    a = P.copy()
    a[np.arange(50), P.argmax(1)] -= 1
    G = (a[:, :, None] * h[:, None, :]).reshape(50, -1)      # 显式 Kronecker 向量
    D2 = squareform(pdist(G)) ** 2
    g2 = (a * a).sum(1) * (h * h).sum(1)
    D2b = g2[:, None] + g2[None, :] - 2 * (a @ a.T) * (h @ h.T)
    assert np.allclose(D2, D2b, atol=1e-9)
    S = U.badge_core(P, h, 5, rng=np.random.default_rng(0))
    assert S[0] == int(np.argmax((G * G).sum(1)))


def test_badge_vs_skactiveml():
    """与 scikit-activeml 1.0.0 Badge 比对：其 D² 初值为 ‖g‖²（原点也当作中心），用 origin_center=True
    并复现其随机数流（random_state_ = RandomState((RandomState(s).randint(1, 2³¹)·(n_unlabeled+1)) mod 2³¹)），
    选择序列应完全相同。"""
    if not _have('skactiveml'):
        skip('scikit-activeml 未安装')
    from sklearn.base import BaseEstimator, ClassifierMixin
    from skactiveml.classifier import SklearnClassifier
    from skactiveml.pool import Badge
    r = np.random.default_rng(6)
    n, d, C, k = 300, 6, 5, 25
    X = r.normal(size=(n, d))
    B = r.normal(size=(d, C))

    class Fixed(ClassifierMixin, BaseEstimator):
        def __init__(self, B=None):
            self.B = B

        def fit(self, X, y, sample_weight=None):
            self.classes_ = np.arange(self.B.shape[1])
            return self

        def predict_proba(self, X):
            return U._softmax(np.asarray(X) @ self.B)

    clf = SklearnClassifier(Fixed(B), classes=np.arange(C), missing_label=-1)
    y0 = np.full(n, -1)
    y0[:C] = np.arange(C)
    clf.fit(X, y0)
    P = U._softmax(X @ B)
    for s in (0, 1, 2):
        q = Badge(random_state=s, missing_label=-1).query(X, np.full(n, -1), clf, fit_clf=False, batch_size=k)
        rs0 = np.random.RandomState(s)
        rs = np.random.RandomState((rs0.randint(1, 2 ** 31) * (n + 1)) % (2 ** 31))
        ours = U.badge_core(P, X, k, origin_center=True,
                            chooser=lambda p: int(rs.choice(len(p), 1, replace=False, p=p)[0]))
        assert list(ours) == [int(i) for i in q], (s, list(ours)[:8], list(q)[:8])
    print('    [BADGE vs scikit-activeml] 3 个种子、k=25：选择序列完全一致（origin_center=True）')


def test_badge_sanity_toy():
    ctx = toy()
    u = unc_margin(ctx.P_T)
    rr = np.random.default_rng(0)
    for nm in ('badge', 'badge_ae_s2'):
        vals = [u[run(nm, ctx, 100, s)].mean() for s in range(3)]
        rnd = [u[rr.choice(ctx.N, 100, replace=False)].mean() for _ in range(3)]
        assert np.mean(vals) > np.mean(rnd), nm


# ---------------------------------------------------------------------
# M16 k-DPP
# ---------------------------------------------------------------------

def _kdpp_exact_probs(L, k):
    subs = list(itertools.combinations(range(len(L)), k))
    w = np.array([np.linalg.det(L[np.ix_(S, S)]) for S in subs])
    return subs, w / w.sum()


def test_kdpp_exact_distribution():
    """小规模：经验子集频率与 det(L_S)/Σ 比较（两种引擎都测），总变差 < 2 × 抽样噪声的期望值。"""
    r = np.random.default_rng(7)
    for Dd, n, k in ((5, 7, 3), (8, 6, 2)):     # 前者走对偶核（D<n），后者走原始核（D≥n）
        Phi = r.normal(size=(Dd, n)) * r.uniform(0.3, 1.5, size=n)
        subs, p = _kdpp_exact_probs(Phi.T @ Phi, k)
        pos = {S: i for i, S in enumerate(subs)}
        engines = ['stable'] + (['dppy'] if _have('dppy') else [])
        for eng in engines:
            rs = np.random.RandomState(0)
            cnt = np.zeros(len(subs))
            n_draw = 20000 if eng == 'stable' else 6000
            for _ in range(n_draw):
                cnt[pos[tuple(sorted(U.kdpp_sample(Phi, k, rs, engine=eng).tolist()))]] += 1
            tv = 0.5 * np.abs(cnt / n_draw - p).sum()
            tv_noise = 0.5 * np.sqrt(2 / np.pi) * np.sqrt(p * (1 - p) / n_draw).sum()   # 精确采样下 TV 的期望量级
            assert tv < 2.0 * tv_noise, (eng, Dd, n, k, tv, tv_noise)
            from scipy.stats import chisquare
            m = p * n_draw >= 5
            pv = chisquare(cnt[m], p[m] / p[m].sum() * cnt[m].sum()).pvalue
            assert pv > 1e-3, (eng, Dd, n, k, pv)
            print(f'    [k-DPP 精确分布] engine={eng} D={Dd} n={n} k={k}：总变差 {tv:.4f}'
                  f'（{n_draw} 次；精确采样的期望噪声 {tv_noise:.4f}；卡方 p={pv:.3f}）')


def test_kdpp_stable_equals_dppy():
    """同一 RandomState 下，本实现（对数域阶段 1 + dppy GS）与 dppy.sample_exact_k_dpp 给出相同样本。"""
    if not _have('dppy'):
        skip('dppy 未安装')
    r = np.random.default_rng(8)
    Phi = r.normal(size=(30, 200)) * r.uniform(0.2, 1.0, size=200)
    for s in range(5):
        for k in (5, 15):
            a = U.kdpp_sample(Phi, k, np.random.RandomState(s), engine='stable')
            b = U.kdpp_sample(Phi, k, np.random.RandomState(s), engine='dppy')
            assert sorted(a.tolist()) == sorted(b.tolist()), (s, k)


def test_kdpp_log_phase1_no_overflow():
    """k 较大时 dppy 的初等对称多项式递推会上溢；对数域版本应仍可用且秩检查正确。"""
    r = np.random.default_rng(9)
    lam = np.exp(r.normal(-6, 4, size=2000))
    sel = U._kdpp_phase1_log(lam, 1000, np.random.RandomState(0))
    assert len(np.unique(sel)) == 1000
    try:
        U._kdpp_phase1_log(np.r_[np.ones(3), np.zeros(10)], 5, np.random.RandomState(0))
    except RuntimeError:
        pass
    else:
        raise AssertionError('秩不足时应报错')


def test_kdpp_sanity_toy():
    """k-DPP 样本在 AE16 空间里比 SRS 更分散（平均最近邻距离更大）；γ>0 时 margin 更高。"""
    ctx = toy()
    X = X_of(ctx, 'ae', 16)
    u = unc_margin(ctx.P_T)
    rr = np.random.default_rng(0)

    def mnn(S):
        D = squareform(pdist(X[S]))
        np.fill_diagonal(D, np.inf)
        return D.min(1).mean()

    k = 100
    dpp = [mnn(run('kdpp_div', ctx, k, s)) for s in range(5)]
    srs = [mnn(rr.choice(ctx.N, k, replace=False)) for _ in range(5)]
    assert np.mean(dpp) > np.mean(srs), (np.mean(dpp), np.mean(srs))
    uq = np.mean([u[run('kdpp', ctx, k, s, gamma=1.0)].mean() for s in range(5)])
    u0 = np.mean([u[run('kdpp_div', ctx, k, s)].mean() for s in range(5)])
    assert uq > u0
    print(f'    [k-DPP] 平均最近邻距离 k-DPP(γ=0) {np.mean(dpp):.3f} vs SRS {np.mean(srs):.3f}；'
          f'所选 margin 均值 γ=1 {uq:.3f} vs γ=0 {u0:.3f}')


# ---------------------------------------------------------------------
# 耗时：N=30000、k=1000
# ---------------------------------------------------------------------

def test_runtime_N30k_k1000():
    if os.environ.get('SKIP_SLOW'):
        skip('SKIP_SLOW=1')
    ctx = make_toy_ctx(N=30000, n_F=50000, seed=11)
    times = {}
    for nm in ALL:
        t = time.time()
        idx = REGISTRY[nm](ctx, 1000, 0)     # 默认超参数（多轮版为完整训练配置）
        times[nm] = time.time() - t
        assert len(idx) == 1000 and len(np.unique(idx)) == 1000
        print(f'    [耗时 N=30000 k=1000] {nm:16s} {times[nm]:7.1f} s', flush=True)
    slow = {nm: t for nm, t in times.items() if t > 180}
    assert not slow, slow
    test_runtime_N30k_k1000.times = times


# ---------------------------------------------------------------------

def _main():
    tests = [(nm, fn) for nm, fn in sorted(globals().items(), key=lambda kv: getattr(kv[1], '__code__', None)
                                               and kv[1].__code__.co_firstlineno or 0)
             if nm.startswith('test_') and callable(fn)]
    n_pass = n_fail = n_skip = 0
    for nm, fn in tests:
        t = time.time()
        try:
            fn()
        except Skip as e:
            n_skip += 1
            print(f'SKIP {nm}: {e}')
            continue
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            import traceback
            traceback.print_exc()
            print(f'FAIL {nm} ({time.time() - t:.1f}s): {e!r}')
            continue
        n_pass += 1
        print(f'PASS {nm} ({time.time() - t:.1f}s)', flush=True)
    print(f'\n{n_pass} passed, {n_fail} failed, {n_skip} skipped')
    return n_fail == 0


if __name__ == '__main__':
    sys.exit(0 if _main() else 1)
