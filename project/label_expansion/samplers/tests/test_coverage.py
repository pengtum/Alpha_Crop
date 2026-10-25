"""
samplers/coverage.py（M10、M11、M12、M14、M15、M17、M20）的单元测试。

运行：
    OMP_NUM_THREADS=2 python samplers/tests/test_coverage.py      # 直接运行
    OMP_NUM_THREADS=2 uv run --no-project --python <venv>/bin/python --with pytest python -m pytest samplers/tests/test_coverage.py -q
    （项目虚拟环境没有装 pytest，用 uv 临时叠加，不改动虚拟环境）

与参考实现的交叉核对：
- scikit-activeml 1.0.0：Clue、TypiClust、MaxHerding、ProbCover（未安装时跳过）；
- coreax 1.0.0 KernelHerding：依赖 jax，没有装进项目虚拟环境；若设置环境变量 COREAX_PYTHON
  指向装有 coreax 的解释器，则通过子进程比对，否则跳过。
"""
import os
import sys
import time
import json
import tempfile
import subprocess
import unittest

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '2')

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(os.path.dirname(HERE))
if CODE not in sys.path:
    sys.path.insert(0, CODE)

from samplers.base import make_toy_ctx, REGISTRY, X_of, X_of_foreign, unc_margin, Ctx  # noqa: E402
from samplers import coverage as cv  # noqa: E402

NAMES = ['clue', 'clue_aes2', 'uherding', 'maxherding', 'uherding_foreign', 'typiclust', 'typiclust_strat',
         'fass_quota', 'fass_quota_prior', 'fass_quota_equal', 'fass_quota_prior_est', 'probcover', 'probcover_km',
         'kherding', 'kherding_foreign', 'fscs', 'fscs_geo']
# 随机性明显的方法：不同种子应给出不同样本（其余方法在给定数据下接近确定性：
# FASS 只在 s_c 的随机点对上有随机性，kernel herding（默认精确均值项）只在 σ 的随机点对上有随机性）
RANDOM_NAMES = ['clue', 'clue_aes2', 'typiclust', 'typiclust_strat', 'probcover', 'probcover_km', 'fscs', 'fscs_geo']
# UHerding 族本身是确定性贪心，随机性只来自候选子采样与 σ 的估计；N = 3000 时候选即全池，
# 所以在 N = 12000（候选 7000 < N）上检验不同种子
HERDING_NAMES = ['uherding', 'maxherding', 'uherding_foreign']
P_HAT = np.array([0.35, 0.2, 0.1, 0.1, 0.08, 0.06, 0.04, 0.03, 0.02, 0.02])


def _hp(name):
    return {'p_hat': P_HAT} if name == 'fass_quota_prior_est' else {}


def _run(name, ctx, k, seed=0, **hp):
    return REGISTRY[name](ctx, k, seed, **{**_hp(name), **hp})


def _skip(msg):
    raise unittest.SkipTest(msg)


def _jacc(a, b):
    a, b = set(map(int, a)), set(map(int, b))
    return len(a & b) / len(a | b)


def _sub_ctx(N, seed=0, n_F=2000):
    return make_toy_ctx(N=N, n_F=n_F, seed=seed)


# ---------------------------------------------------------------------------
# 通用契约
# ---------------------------------------------------------------------------

def test_registered():
    for n in NAMES:
        assert n in REGISTRY, n
        assert REGISTRY[n].__module__.endswith('coverage')


def test_contract_len_unique_sorted():
    ctx = make_toy_ctx()
    for n in NAMES:
        for k in (50, 200):
            S = _run(n, ctx, k)
            assert S.dtype == np.int64 and S.ndim == 1 and len(S) == k, (n, k)
            assert len(np.unique(S)) == k, (n, k)
            assert np.all(np.diff(S) > 0), (n, k)
            assert S.min() >= 0 and S.max() < ctx.N
    # 非概率设计：不写 ctx.meta（没有 pi）
    for n in NAMES:
        assert n not in ctx.meta or 'pi' not in ctx.meta[n], n


def test_reproducible_same_seed():
    ctx = make_toy_ctx()
    for n in NAMES:
        a = _run(n, ctx, 100, seed=7)
        b = _run(n, ctx, 100, seed=7)
        assert np.array_equal(a, b), n


def test_different_seeds_differ():
    ctx = make_toy_ctx()
    for n in RANDOM_NAMES:
        a = _run(n, ctx, 100, seed=0)
        b = _run(n, ctx, 100, seed=1)
        assert not np.array_equal(a, b), n
    ctx = make_toy_ctx(N=12000, n_F=3000)
    for n in HERDING_NAMES:
        a = _run(n, ctx, 100, seed=0)
        b = _run(n, ctx, 100, seed=1)
        assert not np.array_equal(a, b), n


def test_no_label_access():
    """把 oracle 换成抛异常的函数、删掉 _true_y 后仍能运行（本组方法都不允许用标签）。"""
    ctx = make_toy_ctx()

    def boom(idx):
        raise RuntimeError('选样函数不得调用 oracle')

    ctx.oracle = boom
    del ctx._true_y
    for n in NAMES:
        S = _run(n, ctx, 50)
        assert len(S) == 50


# ---------------------------------------------------------------------------
# M10 CLUE
# ---------------------------------------------------------------------------

def test_clue_uncertainty_weighting():
    ctx = make_toy_ctx()
    for n in ('clue', 'clue_aes2'):
        _run(n, ctx, 200)
        dg = cv.DIAG[n]
        assert dg['mean_H_sel'] > dg['mean_H_pool'], dg
    # 温度越高，熵越平，权重越接近均匀（T 只影响权重，不应报错）
    for T in (0.5, 2.0):
        assert len(REGISTRY['clue'](ctx, 50, 0, T=T)) == 50


def test_clue_vs_skactiveml():
    try:
        from skactiveml.pool import Clue
        from skactiveml.base import SkactivemlClassifier
        from sklearn.cluster import KMeans
    except Exception as e:  # pragma: no cover
        _skip(f'scikit-activeml 不可用：{e}')

    class FixedProba(SkactivemlClassifier):
        """直接返回给定概率的分类器（Clue 需要 estimator 接口）。"""

        def __init__(self, P=None, classes=None, missing_label=np.nan, cost_matrix=None, random_state=None):
            super().__init__(classes=classes, missing_label=missing_label, cost_matrix=cost_matrix,
                             random_state=random_state)
            self.P = P

        def fit(self, X, y, sample_weight=None):
            return self

        def predict_proba(self, X):
            assert len(X) == len(self.P)
            return self.P

    ctx = _sub_ctx(800)
    X = X_of(ctx, 'ae', 32)
    for k, seed in ((20, 0), (40, 3)):
        ours = REGISTRY['clue'](ctx, k, seed)
        est = FixedProba(P=ctx.P_T, classes=np.arange(ctx.C))
        q = Clue(method='entropy', cluster_algo=KMeans,
                 cluster_algo_dict={'n_init': 3, 'max_iter': 100, 'random_state': seed}, random_state=seed)
        ref = q.query(X, np.full(ctx.N, np.nan), estimator=est, fit_estimator=False, batch_size=k)
        assert np.array_equal(np.sort(ref), ours), (k, _jacc(ref, ours))


# ---------------------------------------------------------------------------
# M11 UHerding / MaxHerding
# ---------------------------------------------------------------------------

def _uc_dense(X, U, S, sigma):
    """精确目标 UC(S) = (1/N)Σ_n U_n max_{s∈S} k(x_n, x_s)。"""
    sq = (X ** 2).sum(1)
    D2 = np.maximum(sq[:, None] + sq[None, S] - 2 * X @ X[S].T, 0)
    return float((U * np.exp(-D2 / sigma ** 2).max(1)).sum() / len(X))


def _naive_herding_greedy(X, U, k, sigma):
    """稠密朴素贪心（参考实现）：每步选使加权覆盖增益最大的点。"""
    sq = (X ** 2).sum(1)
    K = np.exp(-np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0) / sigma ** 2)
    cov = np.zeros(len(X))
    chosen = np.zeros(len(X), bool)
    S = []
    for _ in range(k):
        g = (U[None, :] * np.maximum(K - cov[None, :], 0)).sum(1)
        g[chosen] = -np.inf
        c = int(np.argmax(g))
        S.append(c)
        chosen[c] = True
        cov = np.maximum(cov, K[c])
    return np.sort(S)


def test_uherding_lazy_equals_naive_dense():
    ctx = _sub_ctx(600)
    X = X_of(ctx, 'ae', 32)
    U = unc_margin(ctx.P_T)
    sigma = 2.0
    for variant, UU in (('uherding', U), ('maxherding', np.ones(ctx.N))):
        ours = cv._uherding(ctx, 30, 0, 't_' + variant, variant=variant, sigma=sigma, n_rand=ctx.N, n_top=0,
                            M_nn=ctx.N, kernel_floor=0.0)
        ref = _naive_herding_greedy(X, UU, 30, sigma)
        assert np.array_equal(ours, ref), (variant, _jacc(ours, ref))


def test_maxherding_vs_skactiveml():
    try:
        from skactiveml.pool import MaxHerding
    except Exception as e:  # pragma: no cover
        _skip(f'scikit-activeml 不可用：{e}')
    ctx = _sub_ctx(600)
    X = X_of(ctx, 'ae', 32)
    sigma = 2.0
    ours = cv._uherding(ctx, 30, 0, 't_mh', variant='maxherding', sigma=sigma, n_rand=ctx.N, n_top=0,
                        M_nn=ctx.N, kernel_floor=0.0)
    q = MaxHerding(normalize_samples=False, metric='rbf', metric_dict={'gamma': 1.0 / sigma ** 2}, random_state=0)
    ref = np.sort(q.query(X, np.full(ctx.N, np.nan), batch_size=30))
    one = np.ones(ctx.N)
    print(f'    MaxHerding: 与 skactiveml 的 Jaccard = {_jacc(ours, ref):.3f}；'
          f'UC 我们 {_uc_dense(X, one, ours, sigma):.6f} vs skactiveml {_uc_dense(X, one, ref, sigma):.6f}')
    assert np.array_equal(ours, ref)


def test_uherding_default_beats_random_and_prefers_uncertain():
    ctx = make_toy_ctx()
    X = X_of(ctx, 'ae', 32)
    U = unc_margin(ctx.P_T)
    rng = np.random.default_rng(0)
    for n, UU in (('uherding', U), ('maxherding', np.ones(ctx.N))):
        S = _run(n, ctx, 100)
        sigma = cv.DIAG[n]['sigma']
        val = _uc_dense(X, UU, S, sigma)
        rnd = [_uc_dense(X, UU, rng.choice(ctx.N, 100, replace=False), sigma) for _ in range(20)]
        assert val > max(rnd), (n, val, max(rnd))
    S = _run('uherding', ctx, 100)
    assert U[S].mean() > U.mean()


def test_uherding_foreign_avoids_foreign_covered():
    """条件于外国池：让外国池只覆盖目标池的一半（真实类 0–2 所在的团块），
    条件变体应把名额转向外国池没有覆盖的区域。（用 _true_y 构造测试数据，选样函数本身不读标签。）"""
    ctx = make_toy_ctx()
    y = ctx._true_y
    rng = np.random.default_rng(0)
    half = np.isin(y, [0, 1, 2])
    src = rng.choice(np.flatnonzero(half), 3000)
    ctx.F_ae = (ctx.T_ae[src] + rng.normal(scale=0.3, size=(3000, 32))).astype(np.float32)
    a = _run('uherding', ctx, 100)
    b = _run('uherding_foreign', ctx, 100)
    c = REGISTRY['uherding_foreign'](ctx, 100, 0, nu=0.5)
    fa, fb, fc = half[a].mean(), half[b].mean(), half[c].mean()
    print(f'    UHerding 落在外国已覆盖区域的比例：无条件 {fa:.2f}，ν=1 {fb:.2f}，ν=0.5 {fc:.2f}（池中 {half.mean():.2f}）')
    assert fb < fa - 0.1 and fb <= fc + 1e-12


# ---------------------------------------------------------------------------
# M12 TypiClust
# ---------------------------------------------------------------------------

def test_typiclust_vs_skactiveml():
    try:
        from skactiveml.pool import TypiClust
        from sklearn.cluster import KMeans
    except Exception as e:  # pragma: no cover
        _skip(f'scikit-activeml 不可用：{e}')
    ctx = _sub_ctx(1500)
    X = X_of(ctx, 'ae', 32)
    for k, seed in ((20, 0), (50, 5)):
        q = TypiClust(k=20, cluster_algo=KMeans, cluster_algo_dict={'random_state': seed}, random_state=seed)
        ref = np.sort(q.query(X, np.full(ctx.N, np.nan), batch_size=k))
        # skactiveml 不丢弃小簇：min_cluster = 1 时两者应完全一致
        ours1 = REGISTRY['typiclust'](ctx, k, seed, min_cluster=1)
        assert np.array_equal(ours1, ref), (k, _jacc(ours1, ref))
        # 默认 min_cluster = 5：差异只来自 < 5 个点的簇（每个小簇的那 1 个名额改给最大的有效簇）
        ours = REGISTRY['typiclust'](ctx, k, seed)
        lab = KMeans(n_clusters=k, random_state=seed).fit_predict(X)
        small = np.flatnonzero(np.bincount(lab, minlength=k) < 5)
        only_ref = np.setdiff1d(ref, ours)
        assert len(only_ref) == len(small) and set(lab[only_ref]) == set(small), (only_ref, small)
        print(f'    TypiClust k={k}: min_cluster=1 与 skactiveml 完全一致；默认版差 {len(small)} 个点（小簇）')


def test_typiclust_picks_typical():
    """所选点的 20-近邻平均距离应小于池均值（偏向高密度）。"""
    from sklearn.neighbors import NearestNeighbors
    ctx = make_toy_ctx()
    X = X_of(ctx, 'ae', 32)
    dk = NearestNeighbors(n_neighbors=21).fit(X).kneighbors(X)[0][:, 1:].mean(1)
    for n in ('typiclust', 'typiclust_strat'):
        S = _run(n, ctx, 100)
        assert dk[S].mean() < dk.mean(), n


def test_typiclust_strat_allocation():
    for N, ks in ((3000, (50, 200, 1000)),):
        ctx = make_toy_ctx(N=N)
        yhat = ctx.P_T.argmax(1)
        N_h = np.bincount(yhat, minlength=ctx.C)
        for k in ks:
            S = _run('typiclust_strat', ctx, k)
            want = cv.alloc(N_h, np.sqrt(N_h), k, 2)
            got = np.bincount(yhat[S], minlength=ctx.C)
            assert np.array_equal(got, want), (k, got, want)
            assert np.all(want[N_h > 0] >= np.minimum(N_h[N_h > 0], 2))


# ---------------------------------------------------------------------------
# alloc（M3 规格）与 M14 FASS-quota
# ---------------------------------------------------------------------------

def test_alloc_hand_examples():
    assert cv.alloc([100, 100, 100], [1, 1, 1], 10, 1).tolist() == [4, 3, 3]
    # 截断 + 溢出再分配：第 1 轮 [1,1,1]+[6,1,0] → 截断为 [2,2,1]；第 2 轮把剩余 5 个按 1:1 分给后两层 → [2,5,3]
    assert cv.alloc([2, 100, 100], [10, 1, 1], 10, 1).tolist() == [2, 5, 3]
    assert cv.alloc([5, 5, 0], [1, 1, 1], 7, 2).tolist() == [4, 3, 0]                 # 空层不分配
    assert cv.alloc([100, 100, 100, 100], [9, 4, 1, 0], 18, 1).tolist() == [10, 5, 2, 1]
    # m_min=2 放不下时退回 m_min=1
    n = cv.alloc([100] * 10, np.ones(10), 12, 2)
    assert n.sum() == 12 and n.min() >= 1
    # 权重 0 的层只拿保底
    assert cv.alloc([50, 50, 50], [0, 1, 1], 20, 1).tolist() == [1, 10, 9]
    # 非空层多于 k：m_min 退到 0，全部按权重的最大余数法分配
    assert cv.alloc([5, 5, 5, 5], [1, 4, 3, 2], 2, 1).tolist() == [0, 1, 1, 0]


def test_alloc_matches_designs_m3():
    """规格要求 M14 的 alloc“与 M3 的 alloc 相同”：与 designs.py 中 M3 的实现逐一比对（该文件不存在时跳过）。"""
    try:
        from samplers.designs import alloc as alloc_m3
    except Exception as e:  # pragma: no cover
        _skip(f'designs.py 的 alloc 不可用：{e}')
    rng = np.random.default_rng(1)
    n_cmp = 0
    for _ in range(2000):
        H = int(rng.integers(1, 12))
        N_h = rng.integers(0, 60, H) * (rng.random(H) > 0.15)
        if N_h.sum() == 0:
            continue
        k = int(rng.integers(1, N_h.sum() + 1))
        w = rng.random(H) * (rng.random(H) > 0.1)
        for m_min in (1, 2):
            assert np.array_equal(cv.alloc(N_h, w, k, m_min), alloc_m3(N_h, w, k, m_min)), (N_h, w, k, m_min)
            n_cmp += 1
    print(f'    alloc 与 designs.alloc（M3）在 {n_cmp} 个随机例子上完全一致')


def test_alloc_random_properties():
    rng = np.random.default_rng(0)
    for _ in range(500):
        H = int(rng.integers(2, 12))
        N_h = rng.integers(0, 200, H)
        if N_h.sum() < 5:
            continue
        k = int(rng.integers(1, N_h.sum() + 1))
        w = rng.random(H) * (rng.random(H) > 0.1)
        m_min = int(rng.integers(1, 4))
        n = cv.alloc(N_h, w, k, m_min)
        assert n.sum() == k
        assert np.all(n <= N_h) and np.all(n >= 0)
        assert np.all(n[N_h == 0] == 0)
        ne = N_h > 0
        if ne.sum() * m_min <= k:
            assert np.all(n[ne] >= np.minimum(N_h[ne], m_min))
        # 无截断时：扣除保底后的名额应为最大余数法结果（与 rest·w/Σw 相差 < 1）
        base = np.where(ne, np.minimum(N_h, m_min), 0)
        if base.sum() <= k and np.all(n < N_h) and w[ne].sum() > 0 and np.all(w[ne] > 0):
            rest = k - base.sum()
            q = rest * np.where(ne, w, 0) / np.where(ne, w, 0).sum()
            assert np.all(np.abs((n - base) - q) < 1 + 1e-9)


def _fass_expected_quota(ctx, k, quota, p_hat=None, beta=None):
    """在测试里独立重算 M14 的候选集与配额。"""
    N, C = ctx.N, ctx.C
    beta = min(N, max(3000, 10 * k)) if beta is None else beta
    u = unc_margin(ctx.P_T)
    cand = np.argsort(-u, kind='stable')[:beta]
    cap = np.bincount(ctx.P_T[cand].argmax(1), minlength=C)
    pool = np.bincount(ctx.P_T.argmax(1), minlength=C).astype(float)
    w = {'sqrt': np.sqrt(pool), 'prior': pool, 'equal': np.ones(C), 'prior_est': p_hat}[quota]
    return cand, cap, cv.alloc(cap, w, k, 1)


def test_fass_quota_satisfied_toy():
    """核心：各类实际入选数（按外国预测类）必须精确等于配额；所选点都在 margin 最不确定的 β 个候选内。"""
    ctx = make_toy_ctx()
    yhat = ctx.P_T.argmax(1)
    for name, quota in (('fass_quota', 'sqrt'), ('fass_quota_prior', 'prior'), ('fass_quota_equal', 'equal'),
                        ('fass_quota_prior_est', 'prior_est')):
        for k in (50, 200, 1000):
            S = _run(name, ctx, k)
            cand, cap, want = _fass_expected_quota(ctx, k, quota, p_hat=P_HAT)
            got = np.bincount(yhat[S], minlength=ctx.C)
            assert np.array_equal(got, want), (name, k, got, want)
            assert np.array_equal(cv.DIAG[name]['n_c'], want)
            assert np.array_equal(cv.DIAG[name]['realized'], want)
            assert np.all(np.isin(S, cand)), (name, k)
            assert np.all(want <= cap) and want.sum() == k
            assert np.all(want[cap > 0] >= 1)        # m_min = 1


def test_fass_quota_shapes_composition():
    """配额确实改变类别构成：equal 比 prior 更均匀；prior 的构成≈外国预测类比例（受容量截断）。"""
    ctx = make_toy_ctx(N=30000, n_F=5000)
    yhat = ctx.P_T.argmax(1)
    k = 200
    c_prior = np.bincount(yhat[_run('fass_quota_prior', ctx, k)], minlength=ctx.C)
    c_sqrt = np.bincount(yhat[_run('fass_quota', ctx, k)], minlength=ctx.C)
    c_eq = np.bincount(yhat[_run('fass_quota_equal', ctx, k)], minlength=ctx.C)

    def ent(c):
        p = c / c.sum()
        p = p[p > 0]
        return -(p * np.log(p)).sum()

    assert ent(c_prior) < ent(c_sqrt) < ent(c_eq) + 1e-12, (c_prior, c_sqrt, c_eq)
    pool = np.bincount(yhat, minlength=ctx.C) / ctx.N
    cap = cv.DIAG['fass_quota_equal']['cap']
    free = cv.fass_quota_plan(ctx, k, 'prior')['n_c'] < cap
    # 未被容量截断的类之间，扣除保底 1 个后的名额应与外国预测类比例成正比（每轮最大余数最多差 1）
    R = (c_prior[free] - 1).sum()
    expect = R * pool[free] / pool[free].sum()
    assert np.all(np.abs((c_prior[free] - 1) - expect) <= 2), (c_prior, expect)


def test_fass_quota_capacity_overflow():
    """容量截断：用很小的 β 让小类的候选数少于等额配额，检查截断与溢出再分配后仍精确满足。"""
    ctx = make_toy_ctx()
    yhat = ctx.P_T.argmax(1)
    k, beta = 200, 400
    S = REGISTRY['fass_quota_equal'](ctx, k, 0, beta=beta)
    cand, cap, want = _fass_expected_quota(ctx, k, 'equal', beta=beta)
    got = np.bincount(yhat[S], minlength=ctx.C)
    assert np.array_equal(got, want), (got, want, cap)
    capped = cap < k // ctx.C
    assert capped.any(), cap                       # 确实出现了截断
    assert np.array_equal(want[capped], cap[capped])
    assert np.all(want[~capped] >= k // ctx.C)     # 溢出量分给了未满的类
    assert np.all(np.isin(S, cand))


def test_fass_quota_edge_cases():
    """β = k 时只能取全部候选（配额 = 容量）；p̂ 中权重为 0 的类只拿保底 1 个。"""
    ctx = make_toy_ctx()
    yhat = ctx.P_T.argmax(1)
    k = 150
    S = REGISTRY['fass_quota'](ctx, k, 0, beta=k)
    cand, cap, want = _fass_expected_quota(ctx, k, 'sqrt', beta=k)
    assert np.array_equal(np.sort(cand), S) and np.array_equal(want, cap)
    p0 = P_HAT.copy()
    p0[[0, 3]] = 0.0
    S = REGISTRY['fass_quota_prior_est'](ctx, 200, 0, p_hat=p0)
    got = np.bincount(yhat[S], minlength=ctx.C)
    cap = cv.DIAG['fass_quota_prior_est']['cap']
    assert np.all(got[[0, 3]] == np.minimum(cap[[0, 3]], 1)), got
    _, _, want = _fass_expected_quota(ctx, 200, 'prior_est', p_hat=p0)
    assert np.array_equal(got, want)


def test_fass_quota_real_scale():
    ctx = make_toy_ctx(N=30000, n_F=5000)
    yhat = ctx.P_T.argmax(1)
    for name, quota in (('fass_quota', 'sqrt'), ('fass_quota_equal', 'equal'), ('fass_quota_prior_est', 'prior_est')):
        for k in (50, 200, 1000):
            S = _run(name, ctx, k)
            _, _, want = _fass_expected_quota(ctx, k, quota, p_hat=P_HAT)
            assert np.array_equal(np.bincount(yhat[S], minlength=ctx.C), want), (name, k)


def test_fass_prior_est_sources():
    ctx = make_toy_ctx()
    a = REGISTRY['fass_quota_prior_est'](ctx, 100, 0, p_hat=P_HAT)
    ctx.meta['p_hat_T'] = P_HAT
    b = REGISTRY['fass_quota_prior_est'](ctx, 100, 0)
    assert np.array_equal(a, b)
    del ctx.meta['p_hat_T']
    try:
        REGISTRY['fass_quota_prior_est'](ctx, 100, 0)
    except ValueError:
        pass
    else:
        raise AssertionError('缺少 p_hat 时应报错')


def _fl_value(X, S, s_c):
    sq = (X ** 2).sum(1)
    W = np.exp(-np.maximum(sq[:, None] + sq[None, S] - 2 * X @ X[S].T, 0) / (2 * s_c ** 2))
    return float(W.max(1).sum())


def _naive_fl_greedy(X, n_sel, s_c):
    """规格伪代码的朴素稠密贪心。"""
    sq = (X ** 2).sum(1)
    W = np.exp(-np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0) / (2 * s_c ** 2))
    cov = np.zeros(len(X))
    S = []
    for _ in range(n_sel):
        g = np.maximum(W - cov[:, None], 0).sum(0)
        g[S] = -np.inf
        j = int(np.argmax(g))
        S.append(j)
        cov = np.maximum(cov, W[:, j])
    return S


def test_fass_lazy_equals_naive_and_sparse_close():
    rng = np.random.default_rng(0)
    ctx = make_toy_ctx()
    X = X_of(ctx, 'ae', 32)
    Xc = X[rng.choice(ctx.N, 1200, replace=False)]
    s_c = float(np.median(cv._random_pair_dists(Xc, 2000, np.random.default_rng(1))))
    for n_sel in (5, 40):
        ours, _ = cv._fl_greedy_class(Xc, n_sel, None, s_c=s_c)
        ref = _naive_fl_greedy(Xc, n_sel, s_c)
        assert list(ours) == ref, (n_sel, list(ours)[:10], ref[:10])
        # 稀疏 50-近邻分支（#U_c > 5,000 时启用）：目标值应接近稠密贪心
        sp, _ = cv._fl_greedy_class(Xc, n_sel, None, s_c=s_c, dense_max=0, knn=50)
        assert len(np.unique(sp)) == n_sel
        v_d, v_s = _fl_value(Xc, ours, s_c), _fl_value(Xc, sp, s_c)
        rnd = np.mean([_fl_value(Xc, rng.choice(len(Xc), n_sel, replace=False), s_c) for _ in range(10)])
        print(f'    FASS 类内设施选址 n_sel={n_sel}: 稠密 {v_d:.1f}，稀疏 50-NN {v_s:.1f}，随机 {rnd:.1f}')
        assert v_s > rnd and v_s >= 0.9 * v_d


def test_fass_sparse_branch_quota():
    """强制走稀疏分支（dense_max=0）时配额同样精确满足。"""
    ctx = make_toy_ctx()
    yhat = ctx.P_T.argmax(1)
    S = REGISTRY['fass_quota'](ctx, 200, 0, dense_max=0)
    _, _, want = _fass_expected_quota(ctx, 200, 'sqrt')
    assert np.array_equal(np.bincount(yhat[S], minlength=ctx.C), want)


def test_fass_prefers_uncertain():
    ctx = make_toy_ctx(N=30000, n_F=5000)
    u = unc_margin(ctx.P_T)
    S = _run('fass_quota', ctx, 200)
    assert u[S].mean() > np.quantile(u, 0.8)


# ---------------------------------------------------------------------------
# M15 ProbCover
# ---------------------------------------------------------------------------

def _naive_probcover(X, delta, k):
    sq = (X ** 2).sum(1)
    A = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0)) <= delta
    np.fill_diagonal(A, True)
    covered = np.zeros(len(X), bool)
    chosen = np.zeros(len(X), bool)
    S = []
    for _ in range(k):
        deg = (A & ~covered[None, :]).sum(1)
        deg[chosen] = -1
        c = int(np.argmax(deg))
        if deg[c] <= 0:
            break
        S.append(c)
        chosen[c] = True
        covered |= A[c]
    return S, covered.mean()


def test_probcover_greedy_equals_naive():
    ctx = _sub_ctx(700)
    X = X_of(ctx, 'ae', 32)
    delta = 2.3
    ours = REGISTRY['probcover'](ctx, 40, 0, delta=delta)
    ref, cov_ref = _naive_probcover(X, delta, 40)
    assert cv.DIAG['probcover']['n_fill'] == 0
    assert np.array_equal(ours, np.sort(ref))
    assert abs(cv.DIAG['probcover']['coverage'] - cov_ref) < 1e-12


def test_probcover_vs_skactiveml_fixed_delta():
    try:
        from skactiveml.pool import ProbCover
    except Exception as e:  # pragma: no cover
        _skip(f'scikit-activeml 不可用：{e}')
    ctx = _sub_ctx(700)
    X = X_of(ctx, 'ae', 32)
    delta = 2.3
    ours = REGISTRY['probcover'](ctx, 40, 0, delta=delta)
    q = ProbCover(deltas=[delta], alpha=0.95, random_state=0)
    ref = np.sort(q.query(X, np.full(ctx.N, np.nan), batch_size=40))
    sq = (X ** 2).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))
    c_ours, c_ref = (D[:, ours] <= delta).any(1).mean(), (D[:, ref] <= delta).any(1).mean()
    print(f'    ProbCover δ={delta}: 覆盖率 我们 {c_ours:.4f} vs skactiveml {c_ref:.4f}，Jaccard {_jacc(ours, ref):.3f}'
          '（skactiveml 对并列出度随机取，差异来自并列）')
    assert abs(c_ours - c_ref) <= 0.02


def _blob_ctx(seed=0):
    """8 个大小不等、相距较远的团块（团内点对约占 17%）；伪标签可被恢复，纯度在网格中间由 1 降到 α 以下。"""
    r = np.random.default_rng(seed)
    sizes = [150, 120, 100, 80, 60, 40, 30, 20]
    C = 8
    cen = r.normal(size=(C, 32)) * 1.2
    y = np.repeat(np.arange(C), sizes)
    T = cen[y] + r.normal(size=(len(y), 32)) * 0.35
    N = len(y)
    P = np.full((N, C), 0.05)
    P[np.arange(N), y] = 0.85
    ctx = Ctx(N=N, C=C, T_ae=T.astype(np.float32), T_s2=T.astype(np.float32), T_xy=r.uniform(0, 100, (N, 2)),
              T_nuts3=np.zeros(N, int), F_ae=T[:50].astype(np.float32), F_s2=T[:50].astype(np.float32),
              F_xy=np.zeros((50, 2)), F_y=y[:50], F_country=np.zeros(50, int), P_T=P)
    return ctx


def test_probcover_delta_selection_vs_skactiveml():
    try:
        from skactiveml.pool import ProbCover
        from sklearn.cluster import KMeans
    except Exception as e:  # pragma: no cover
        _skip(f'scikit-activeml 不可用：{e}')
    ctx = _blob_ctx()
    X = X_of(ctx, 'ae', 32)
    for n in ('probcover', 'probcover_km'):
        REGISTRY[n](ctx, 20, 0)
        dg = cv.DIAG[n]
        assert dg['purity_met'], dg['purity']
        assert 0 < dg['g'] < len(dg['grid']) - 1, dg['purity']   # δ* 落在网格中间，检验有意义
        if n == 'probcover_km':
            q = ProbCover(n_classes=ctx.C, deltas=dg['grid'], alpha=0.95, cluster_algo=KMeans,
                          cluster_algo_dict={'random_state': 0}, random_state=0)
            q.query(X, np.full(ctx.N, np.nan), batch_size=5)
            print(f"    ProbCover δ*: 我们 {dg['delta']:.4f} vs skactiveml {q.delta_max_:.4f}")
            assert abs(q.delta_max_ - dg['delta']) < 1e-9


def test_probcover_coverage_beats_random():
    ctx = make_toy_ctx()
    X = X_of(ctx, 'ae', 32)
    S = _run('probcover', ctx, 100)
    delta = cv.DIAG['probcover']['delta']
    from sklearn.neighbors import NearestNeighbors

    def cov(idx):   # δ 球覆盖率：离最近入选点 ≤ δ 的点所占比例
        return float((NearestNeighbors(n_neighbors=1).fit(X[idx]).kneighbors(X)[0][:, 0] <= delta).mean())

    rng = np.random.default_rng(0)
    rnd = max(cov(rng.choice(ctx.N, 100, replace=False)) for _ in range(20))
    assert cov(S) > rnd, (cov(S), rnd)


# ---------------------------------------------------------------------------
# M17 Kernel herding
# ---------------------------------------------------------------------------

def _mmd2(X, S, sigma):
    """精确 MMD²(目标池经验分布, S 的均匀经验分布)，核 exp(−‖Δ‖²/σ²)，省略与 S 无关的常数项。"""
    sq = (X ** 2).sum(1)
    KS = np.exp(-np.maximum(sq[:, None] + sq[None, S] - 2 * X @ X[S].T, 0) / sigma ** 2)
    return float(KS[S].mean() - 2 * KS.mean())


def _naive_herding(X, k, sigma):
    sq = (X ** 2).sum(1)
    K = np.exp(-np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0) / sigma ** 2)
    mu = K.mean(1)
    acc = np.zeros(len(X))
    chosen = np.zeros(len(X), bool)
    S = []
    for t in range(k):
        s = mu - acc / (t + 1)
        s[chosen] = -np.inf
        i = int(np.argmax(s))
        S.append(i)
        chosen[i] = True
        acc += K[i]
    return S


def test_kherding_exact_equals_naive():
    ctx = _sub_ctx(800)
    X = X_of(ctx, 'ae', 32)
    ours = REGISTRY['kherding'](ctx, 40, 0, mean_mode='exact', sigma=5.0)
    ref = _naive_herding(X, 40, 5.0)
    assert np.array_equal(ours, np.sort(ref))


def test_kherding_beats_random_and_rff_diagnostic():
    """默认（精确均值项）的 MMD 应远低于随机样本；同时记录规格 RFF 版（D=2048）的表现（这是改默认的依据）。"""
    ctx = make_toy_ctx()
    X = X_of(ctx, 'ae', 32)
    b = _run('kherding', ctx, 100)
    sigma = cv.DIAG['kherding']['sigma']
    a = REGISTRY['kherding'](ctx, 100, 0, mean_mode='rff')
    rng = np.random.default_rng(0)
    const = cv._exact_mean_kernel(X, X, sigma).mean()
    rnd = np.array([_mmd2(X, rng.choice(ctx.N, 100, replace=False), sigma) for _ in range(50)]) + const
    m_rff, m_ex = _mmd2(X, a, sigma) + const, _mmd2(X, b, sigma) + const
    print(f'    herding MMD²：精确均值 {m_ex:.5f}，RFF(D=2048) {m_rff:.5f}，随机中位数 {np.median(rnd):.5f}、'
          f'最好 {rnd.min():.5f}；RFF 与精确版的 Jaccard {_jacc(a, b):.3f}')
    assert m_ex < rnd.min() and m_ex < 0.3 * np.median(rnd)
    # RFF 均值项本身近似良好（相关 > 0.98），问题出在与精确 acc 项混用
    Wr = np.random.default_rng(1).normal(0, np.sqrt(2) / sigma, size=(32, 2048))
    br = np.random.default_rng(2).uniform(0, 2 * np.pi, 2048)
    assert np.corrcoef(cv._rff_mean_kernel(X, X, Wr, br), cv._exact_mean_kernel(X, X, sigma))[0, 1] > 0.98


def test_kherding_foreign_targets_underrepresented():
    """条件于外国池：让外国池只覆盖目标池中真实类 0–2 的团块（占 71%）。
    m_F = k 时所选样本在匹配 2μ_T − μ_F，落在外国已覆盖区域的比例应约为 2·0.71 − 1 ≈ 0.42；
    m_F = 5k 时 6μ_T − 5μ_F 在该区域为负，比例应接近 0。（_true_y 只用于构造测试数据。）"""
    ctx = make_toy_ctx()
    y = ctx._true_y
    rng = np.random.default_rng(0)
    half = np.isin(y, [0, 1, 2])
    src = rng.choice(np.flatnonzero(half), 3000)
    ctx.F_ae = (ctx.T_ae[src] + rng.normal(scale=0.3, size=(3000, 32))).astype(np.float32)
    a = _run('kherding', ctx, 100)
    b = _run('kherding_foreign', ctx, 100)
    c = REGISTRY['kherding_foreign'](ctx, 100, 0, m_F_mult=5)
    fa, fb, fc = half[a].mean(), half[b].mean(), half[c].mean()
    print(f'    herding 落在外国已覆盖区域的比例：目标版 {fa:.2f}，m_F=k {fb:.2f}，m_F=5k {fc:.2f}（池中 {half.mean():.2f}）')
    assert abs(fa - half.mean()) < 0.1
    assert fb < fa - 0.15 and fc < fb
    assert abs(fb - (2 * half.mean() - 1)) < 0.15


_COREAX_SCRIPT = r'''
import sys, json, numpy as np
import jax
jax.config.update('jax_enable_x64', True)
from coreax import Data
from coreax.kernels import SquaredExponentialKernel
from coreax.solvers import KernelHerding
X = np.load(sys.argv[1]); k = int(sys.argv[2]); sigma = float(sys.argv[3])
kern = SquaredExponentialKernel(length_scale=sigma / np.sqrt(2.0))   # exp(-|d|^2/(2 l^2)) = exp(-|d|^2/sigma^2)
cs, _ = KernelHerding(coreset_size=k, kernel=kern, unique=True).reduce(Data(X))
print(json.dumps(np.asarray(cs.unweighted_indices).astype(int).tolist()))
'''


def test_kherding_vs_coreax():
    py = os.environ.get('COREAX_PYTHON')
    if not py or not os.path.exists(py):
        _skip('未设置 COREAX_PYTHON（coreax 依赖 jax，未装入项目虚拟环境）')
    ctx = _sub_ctx(800)
    X = X_of(ctx, 'ae', 32)
    k, sigma = 40, 5.0
    ours = REGISTRY['kherding'](ctx, k, 0, mean_mode='exact', sigma=sigma)
    with tempfile.TemporaryDirectory() as td:
        np.save(os.path.join(td, 'X.npy'), X)
        sp = os.path.join(td, 'run.py')
        open(sp, 'w').write(_COREAX_SCRIPT)
        out = subprocess.run([py, sp, os.path.join(td, 'X.npy'), str(k), str(sigma)], capture_output=True,
                             text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    ref = np.sort(json.loads(out.stdout.strip().splitlines()[-1]))
    print(f'    kernel herding 与 coreax KernelHerding 的 Jaccard = {_jacc(ours, ref):.3f}')
    assert np.array_equal(ours, ref)


# ---------------------------------------------------------------------------
# M20 FSCS
# ---------------------------------------------------------------------------

def test_fscs_nearest_to_centroids_and_msssd():
    from sklearn.cluster import KMeans
    ctx = make_toy_ctx()
    X = X_of(ctx, 'ae', 32)
    k, seed = 100, 3
    S = REGISTRY['fscs'](ctx, k, seed)
    km = KMeans(n_clusters=k, n_init=3, max_iter=300, random_state=seed).fit(X)
    # 逐簇心复核：所选点 = 该簇心按距离排序后第一个未被前面簇心占用的点
    taken = set()
    for mu in km.cluster_centers_:
        order = np.argsort(((X - mu) ** 2).sum(1), kind='stable')
        j = next(int(i) for i in order if int(i) not in taken)
        taken.add(j)
    assert set(map(int, S)) == taken

    def msssd(idx):
        sq = (X ** 2).sum(1)
        return float(np.maximum(sq[:, None] + sq[None, idx] - 2 * X @ X[idx].T, 0).min(1).mean())

    rng = np.random.default_rng(0)
    assert msssd(S) < min(msssd(rng.choice(ctx.N, k, replace=False)) for _ in range(20))


def test_fscs_geo_spreads_in_space():
    ctx = make_toy_ctx()
    from scipy.spatial import cKDTree
    a = _run('fscs', ctx, 100)
    b = _run('fscs_geo', ctx, 100)

    def geo_msssd(idx):
        return float((cKDTree(ctx.T_xy[idx]).query(ctx.T_xy)[0] ** 2).mean())

    assert geo_msssd(b) < geo_msssd(a)


# ---------------------------------------------------------------------------
# 耗时：N = 30,000、k = 1,000、d = 32（外国池 5 万）
# ---------------------------------------------------------------------------

TIMINGS = {}


def test_timing_N30000_k1000():
    ctx = make_toy_ctx(N=30000, n_F=50000)
    for n in NAMES:
        t = time.time()
        S = _run(n, ctx, 1000)
        dt = time.time() - t
        TIMINGS[n] = dt
        assert len(S) == 1000
        assert dt < 180, (n, dt)
    print('    耗时（秒）：' + '，'.join(f'{n} {t:.1f}' for n, t in TIMINGS.items()))


# ---------------------------------------------------------------------------

def _main():
    tests = [(n, f) for n, f in globals().items() if n.startswith('test_') and callable(f)]
    n_ok = n_fail = n_skip = 0
    for name, f in tests:
        t = time.time()
        try:
            f()
        except unittest.SkipTest as e:
            n_skip += 1
            print(f'SKIP {name}: {e}')
            continue
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            import traceback
            traceback.print_exc()
            print(f'FAIL {name}: {e!r}')
            continue
        n_ok += 1
        print(f'ok   {name} ({time.time() - t:.1f}s)')
    print(f'\n通过 {n_ok}，失败 {n_fail}，跳过 {n_skip}')
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(_main())
