"""
samplers/designs.py 的单元测试（M1、M2、M3、M5、M9、M19 与公共工具 lpm2、alloc、先验估计）。

pytest 风格；也可直接运行：
    OMP_NUM_THREADS=2 python samplers/tests/test_designs.py          # 全部（约 5 分钟）
    DESIGNS_FAST=1 python samplers/tests/test_designs.py             # 缩小蒙特卡洛次数（约 1 分钟）
R 交叉核对需要 Rscript 和 BalancedSampling（R_LIBS 指向其所在库）；clhs 交叉核对需要 PyPI clhs 1.0.2。
缺少时对应测试跳过。

交叉核对结果（2026-09-24，make_toy_ctx(N=2000, seed=3)，X = X_of(ctx,'ae',16,geo_w=2)，各 300 次）：
- LPM2 所选样本在 X 中的平均最近邻距离
    k=50 ：R BalancedSampling 2.1.1 lpm2 = 2.8343 ± 0.0056，本实现 = 2.8383 ± 0.0057，SRS = 2.4527
    k=200：R = 2.0116 ± 0.0014，本实现 = 2.0123 ± 0.0016，SRS = 1.8531
    k=100（600 次）：R = 2.3552 ± 0.0020，本实现 = 2.3548 ± 0.0023，SRS = 2.1025
  入样频率的均值与标准差与 R 一致（k=50：均为 0.0250 / 0.0090，二项标准差 0.0090）。
- Cube（平衡 [π, P_T[:, :9]]）的平衡误差 ‖Σ_S x/π − Σ_U x‖/‖Σ_U x‖，中位数
    k=50 ：R cube = 5.42e-2，本实现 = 5.36e-2，SRS = 2.27e-1
    k=200：R cube = 1.22e-2，本实现 = 1.24e-2，SRS = 1.13e-1
  分层版（按 NUTS3 编码分层）：k=50 R cubestratified = 9.60e-2，本实现 = 8.36e-2；k=200 R = 2.42e-2，本实现 = 2.07e-2。
  各层样本量与 N_h·k/N 的最大偏差（平均）：R 0.83 / 本实现 0.82（k=50）。
  R cubestratified 的总样本量在 46–55（k=50）之间波动；本实现固定为 k。
- cLHS（12 个变量，10000 次迭代，3 个种子）目标值 O（越小越好）：
    k=50 ：PyPI clhs 1.0.2 = 241.6 / 233.8 / 232.1（4–9 秒），本实现 = 215.9 / 206.5 / 208.4（0.3–1.2 秒），随机 ≈ 460
    k=200：clhs 1.0.2 = 1011 / 1029 / 1028（约 5 秒），本实现 = 917 / 913 / 894（约 0.4 秒），随机 ≈ 1700
  clhs 1.0.2 在 numpy 2.x 下无法直接运行（resample_worst 的单元素数组赋值报错），核对时用
  designs._clhs_pkg_resample_worst 逐行保留原逻辑、只修正标量转换。
"""
import os
import sys
import time
import shutil
import subprocess
import tempfile
import unittest

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '2')

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(os.path.dirname(HERE))
if CODE not in sys.path:
    sys.path.insert(0, CODE)

import numpy as np
from scipy.spatial import cKDTree

from samplers.base import make_toy_ctx, REGISTRY, X_of, unc_margin
from samplers import designs as D

FAST = os.environ.get('DESIGNS_FAST') == '1'
METHODS = D.DESIGN_METHODS


def _skip(msg):
    raise unittest.SkipTest(msg)


def _no_label_ctx(N=3000, seed=0):
    """把神谕换成抛异常的函数、删掉 _true_y 的玩具上下文。"""
    ctx = make_toy_ctx(N=N, seed=seed)

    def _forbidden(idx):
        raise AssertionError('选样函数不得调用 ctx.oracle')
    ctx.oracle = _forbidden
    del ctx._true_y
    return ctx


# ---------------------------------------------------------------------
# 通用接口
# ---------------------------------------------------------------------

def test_registered():
    for m in METHODS:
        assert m in REGISTRY, m


def test_length_unique_sorted():
    ctx = make_toy_ctx(N=3000, seed=0)
    for m in METHODS:
        for k in (50, 200):
            idx = REGISTRY[m](ctx, k, 7)
            assert idx.dtype == np.int64 and idx.ndim == 1 and len(idx) == k, (m, k)
            assert len(np.unique(idx)) == k and (np.diff(idx) > 0).all(), (m, k)
            assert idx.min() >= 0 and idx.max() < ctx.N


def test_reproducible_same_seed():
    ctx = make_toy_ctx(N=3000, seed=0)
    for m in METHODS:
        a = REGISTRY[m](ctx, 100, 123)
        b = REGISTRY[m](ctx, 100, 123)
        assert np.array_equal(a, b), m


def test_different_seeds_differ():
    ctx = make_toy_ctx(N=3000, seed=0)
    for m in METHODS:           # 本组全部方法都是随机的
        a = REGISTRY[m](ctx, 100, 1)
        b = REGISTRY[m](ctx, 100, 2)
        assert not np.array_equal(a, b), m


def test_no_label_access():
    """oracle 抛异常、_true_y 删除后全部方法仍能运行（本组没有允许用神谕的方法）。"""
    ctx = _no_label_ctx()
    for m in METHODS:
        for k in (50, 200):
            idx = REGISTRY[m](ctx, k, 0)
            assert len(idx) == k


def test_probability_designs_meta():
    ctx = make_toy_ctx(N=3000, seed=1)
    for m in D.PROB_DESIGNS:
        for k in (50, 200):
            idx = REGISTRY[m](ctx, k, 3)
            meta = ctx.meta[m]
            pi = meta['pi']
            assert pi.shape == (ctx.N,), m
            assert (pi >= 0).all() and (pi <= 1).all(), m
            assert abs(pi.sum() - k) < 1e-6, (m, pi.sum())
            assert (pi[idx] > 0).all(), m
            assert np.array_equal(np.sort(meta['prior_idx']), idx), m
            assert meta['estimator'] in ('ht', 'olofsson', 'aipw'), m
    # 非概率方法不写 meta
    ctx2 = make_toy_ctx(N=3000, seed=1)
    for m in ('pmht_gumbel', 'clhs'):
        REGISTRY[m](ctx2, 50, 0)
        assert m not in ctx2.meta


# ---------------------------------------------------------------------
# alloc
# ---------------------------------------------------------------------

def test_alloc_basic():
    N_h = np.array([1000, 500, 300, 100, 50, 30, 10, 5, 3, 0])
    for w in (N_h, np.sqrt(N_h), (N_h > 0).astype(float)):
        for k in (20, 50, 200, 1000, N_h.sum()):
            for m_min in (0, 2):
                n = D.alloc(N_h, w, k, m_min)
                assert n.sum() == k and (n <= N_h).all() and (n >= 0).all(), (k, m_min, n)
                assert n[-1] == 0
                if m_min == 2 and k >= 2 * (N_h > 0).sum():
                    assert (n[N_h > 0] >= np.minimum(N_h[N_h > 0], 2)).all()
    # 比例分配（m_min=0）与 k·N_h/N 的差不超过 1
    n = D.alloc(N_h, N_h, 200, 0)
    assert np.abs(n - 200 * N_h / N_h.sum()).max() < 1
    # 溢出再分配：等权重，小层截断后余量给大层
    assert D.alloc([5, 100, 1000], [1, 1, 1], 300, 0).tolist() == [5, 100, 195]
    # m_min × 非空层数 > k 时降为 1
    n = D.alloc(np.full(30, 100), np.ones(30), 50, 2)
    assert n.sum() == 50 and n.min() == 1
    # 非空层数 > k 时降为 0
    n = D.alloc(np.full(30, 100), np.ones(30), 20, 2)
    assert n.sum() == 20 and n.max() == 1
    # 权重全 0 时平均分配
    n = D.alloc([10, 10, 10], [0, 0, 0], 9, 0)
    assert n.tolist() == [3, 3, 3]


# ---------------------------------------------------------------------
# LPM2
# ---------------------------------------------------------------------

def _inclusion_check(pi, X, reps, label):
    N = len(pi)
    cnt = np.zeros(N)
    k = int(round(pi.sum()))
    for s in range(reps):
        idx = D.lpm2(pi, X, np.random.default_rng(10_000 + s))
        assert len(idx) == k and len(np.unique(idx)) == k
        cnt[idx] += 1
    f = cnt / reps
    m = (pi > 0) & (pi < 1)
    z = (f[m] - pi[m]) / np.sqrt(pi[m] * (1 - pi[m]) / reps)
    frac3 = np.mean(np.abs(z) > 3)
    print(f'    [{label}] reps={reps}: |z|>3 的比例 {frac3:.4f}（期望 0.0027），mean z² = {np.mean(z ** 2):.3f}，'
          f'max|z| = {np.abs(z).max():.2f}')
    assert frac3 < 0.01, frac3
    assert 0.8 < np.mean(z ** 2) < 1.2, np.mean(z ** 2)
    assert np.all(f[pi == 1] == 1) and np.all(f[pi == 0] == 0)


def test_lpm2_inclusion_probabilities():
    """2,000 点玩具数据上抽 2,000 次（FAST 模式 400 次）：入样频率与 π 之差在 3 个标准误内。"""
    reps = 400 if FAST else 2000
    ctx = make_toy_ctx(N=2000, seed=5)
    X = X_of(ctx, 'ae', 16, geo_w=2.0)
    _inclusion_check(np.full(2000, 100 / 2000), X, reps, '等概率 k=100')
    # 不等概率（M5 的 π），并加入几个 π=0 与 π=1 的单元
    pi = D.pmht_pi(ctx.P_T, 150, beta=2.0, tau=0.3)
    pi[:3] = 0.0
    pi[3:6] = 1.0
    pi[6:] *= (150 - 3) / pi[6:].sum()
    assert abs(pi.sum() - 150) < 1e-9 and pi.max() <= 1
    _inclusion_check(pi, X, reps, '不等概率 k=150')


def test_lpm2_edge_cases():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    # 非整数和：样本量为 floor 或 ceil
    sizes = {len(D.lpm2(np.full(50, 0.105), X, np.random.default_rng(s))) for s in range(200)}
    assert sizes <= {5, 6} and len(sizes) == 2
    # 全 0 / 全 1
    assert len(D.lpm2(np.zeros(50), X, rng)) == 0
    assert len(D.lpm2(np.ones(50), X, rng)) == 50
    # 一维 X、重复点
    Xd = np.repeat(np.arange(10.0), 5)
    idx = D.lpm2(np.full(50, 0.2), Xd, np.random.default_rng(1))
    assert len(idx) == 10
    # q=1（每次只查 1 个近邻，需要翻倍）
    idx = D.lpm2(np.full(50, 0.2), X, np.random.default_rng(2), q=1)
    assert len(idx) == 10


def test_lpm2_spread():
    """LPM2 样本在 X 中比 SRS 更分散（平均最近邻距离更大），且同时覆盖更多 NUTS3 格。"""
    ctx = make_toy_ctx(N=3000, seed=2)
    X = X_of(ctx, 'ae', 16, geo_w=2.0)

    def mnn(idx):
        d, _ = cKDTree(X[idx]).query(X[idx], k=2)
        return d[:, 1].mean()
    lp = np.mean([mnn(REGISTRY['lpm2'](ctx, 100, s)) for s in range(20)])
    sr = np.mean([mnn(REGISTRY['srs'](ctx, 100, s)) for s in range(20)])
    print(f'    平均最近邻距离 LPM2={lp:.3f}，SRS={sr:.3f}')
    assert lp > 1.05 * sr


# ---------------------------------------------------------------------
# M1
# ---------------------------------------------------------------------

def test_srs_nuts3_allocation():
    ctx = make_toy_ctx(N=3000, seed=0)
    codes, N_h = D._codes(ctx.T_nuts3)
    for k in (50, 200):
        idx = REGISTRY['srs_nuts3'](ctx, k, 0)
        n_obs = np.bincount(codes[idx], minlength=N_h.size)
        assert np.array_equal(n_obs, D.alloc(N_h, N_h, k, 0))
        meta = ctx.meta['srs_nuts3']
        assert meta['scheme'] == 'stratified'
        assert np.allclose(meta['pi'], (meta['n_h'] / N_h)[codes])
    # 层数（25）> k：系统抽样，每层样本数为 floor 或 ceil(N_h k/N)（按 NUTS3 排序后的隐含比例分层），π = k/N
    k = 20
    for s in range(20):
        idx = REGISTRY['srs_nuts3'](ctx, k, s)
        assert ctx.meta['srs_nuts3']['scheme'] == 'systematic'
        n_obs = np.bincount(codes[idx], minlength=N_h.size)
        e = N_h * k / ctx.N
        # 系统抽样的层样本量在 floor(e)..ceil(e)+1 之间（层边界不对齐步长时可能多 1）
        assert (n_obs >= np.floor(e) - 1).all() and (n_obs <= np.ceil(e) + 1).all()
    assert np.allclose(ctx.meta['srs_nuts3']['pi'], k / ctx.N)


def test_srs_nuts3_string_codes():
    """T_nuts3 为字符串时也能运行（NUTS2 前缀相同的层相邻）。"""
    ctx = make_toy_ctx(N=3000, seed=0)
    ctx.T_nuts3 = np.array(['BE2%02d' % v for v in ctx.T_nuts3])
    idx = REGISTRY['srs_nuts3'](ctx, 50, 0)
    assert len(idx) == 50
    idx = REGISTRY['cube_nuts2'](ctx, 50, 0)
    assert ctx.meta['cube_nuts2']['strata_source'] == 'T_nuts3[:4]'


# ---------------------------------------------------------------------
# M3
# ---------------------------------------------------------------------

def test_predstrat_allocation_and_pi():
    ctx = make_toy_ctx(N=3000, seed=0)
    for name, rule, strata in (('predstrat', 'sqrt', 'yhat'), ('predstrat_prop', 'prop', 'yhat'),
                               ('predstrat_equal', 'equal', 'yhat'), ('predstrat_neyman', 'neyman', 'yhat'),
                               ('predstrat_srs', 'sqrt', 'yhat'), ('predstrat_conf', 'sqrt', 'yhat_conf')):
        for k in (50, 200, 1000):
            idx = REGISTRY[name](ctx, k, 0)
            meta = ctx.meta[name]
            used = 'yhat' if (strata == 'yhat_conf' and k < 200) else strata
            assert meta['strata_used'] == used
            h, H = D.predstrat_strata(ctx.P_T, used)
            N_h = np.bincount(h, minlength=H)
            n_exp = D.alloc(N_h, D.predstrat_weights(ctx.P_T, h, H, rule), k, 2)
            n_obs = np.bincount(h[idx], minlength=H)
            assert np.array_equal(n_obs, n_exp), (name, k, n_obs, n_exp)
            assert np.allclose(meta['pi'], (n_exp / np.maximum(N_h, 1))[h])
            assert (n_obs[N_h > 0] >= np.minimum(2, N_h[N_h > 0])).all()
    # sqrt / equal 比 prop 更偏向小预测类
    h, H = D.predstrat_strata(ctx.P_T, 'yhat')
    N_h = np.bincount(h, minlength=H)
    small = np.argsort(N_h)[:3]
    n_prop = np.bincount(h[REGISTRY['predstrat_prop'](ctx, 200, 0)], minlength=H)[small].sum()
    n_sqrt = np.bincount(h[REGISTRY['predstrat'](ctx, 200, 0)], minlength=H)[small].sum()
    n_eq = np.bincount(h[REGISTRY['predstrat_equal'](ctx, 200, 0)], minlength=H)[small].sum()
    assert n_prop < n_sqrt < n_eq


def test_predstrat_conf_strata():
    ctx = make_toy_ctx(N=3000, seed=0)
    h, H = D.predstrat_strata(ctx.P_T, 'yhat_conf')
    assert H == 2 * ctx.C
    yhat = ctx.P_T.argmax(1)
    assert np.array_equal(h // 2, yhat)
    # 每个预测类内高/低置信两半大致各半（中位数划分）
    for c in np.unique(yhat):
        nh = np.bincount(h[yhat == c] % 2, minlength=2)
        assert abs(nh[0] - nh[1]) <= 1 + 0.01 * nh.sum()


# ---------------------------------------------------------------------
# M5
# ---------------------------------------------------------------------

def test_pmht_pi_and_selection():
    ctx = make_toy_ctx(N=3000, seed=0)
    N = ctx.N
    s = unc_margin(ctx.P_T)
    for beta, tau in ((1, 0.3), (2, 0.3), (4, 0.3), (1, 0.2), (1, 0.5)):
        for k in (50, 200, 1000):
            pi = D.pmht_pi(ctx.P_T, k, beta, tau)
            assert abs(pi.sum() - k) < 1e-6 and pi.max() <= 1 and pi.min() >= tau * k / N * (1 - 1e-9)
            # π 随不确定性单调不减
            o = np.argsort(s)
            assert (np.diff(pi[o]) >= -1e-12).all()
    # 截断：β=4、k=1000 时有 π=1 的单元，和仍为 k
    pi = D.pmht_pi(ctx.P_T, 1000, 4.0, 0.3)
    assert (pi == 1).sum() > 0 and abs(pi.sum() - 1000) < 1e-6
    # 所选样本的平均 margin 高于总体，β 越大越高
    ms = {m: np.mean([s[REGISTRY[m](ctx, 200, r)].mean() for r in range(5)]) for m in ('srs', 'pmht', 'pmht_b4')}
    print(f'    样本平均不确定性：SRS={ms["srs"]:.3f}，PM-HT β=1={ms["pmht"]:.3f}，β=4={ms["pmht_b4"]:.3f}')
    assert ms['srs'] < ms['pmht'] < ms['pmht_b4']
    # meta 的 π 与 pmht_pi 一致
    REGISTRY['pmht_t05'](ctx, 200, 0)
    assert np.allclose(ctx.meta['pmht_t05']['pi'], D.pmht_pi(ctx.P_T, 200, 1.0, 0.5))
    # Gumbel 消融也偏向高不确定性
    g = np.mean([s[REGISTRY['pmht_gumbel'](ctx, 200, r)].mean() for r in range(5)])
    assert g > ms['srs']


# ---------------------------------------------------------------------
# M9 Cube
# ---------------------------------------------------------------------

def test_cube_balance():
    """HT 估计的辅助变量总量与真值的相对误差：cube 远小于 SRS；π 列（样本量）精确平衡。"""
    ctx = make_toy_ctx(N=3000, seed=0)
    for k in (50, 200):
        pi0, Xbal = D.cube_xbal(ctx, k, 'phat')
        e_cube, e_srs, rel_cube, rel_srs = [], [], [], []
        for s in range(20):
            idx = REGISTRY['cube'](ctx, k, s)
            e, rel = D.balance_error(Xbal, pi0, idx)
            assert abs(e - ctx.meta['cube']['balance_err']) < 1e-12
            assert rel[0] < 1e-9                          # Σ_S π/π = k
            e_cube.append(e)
            rel_cube.append(rel)
            e2, rel2 = D.balance_error(Xbal, pi0, REGISTRY['srs'](ctx, k, s))
            e_srs.append(e2)
            rel_srs.append(rel2)
        mc, ms = np.median(e_cube), np.median(e_srs)
        print(f'    k={k}: 平衡误差中位数 cube={mc:.2e}，SRS={ms:.2e}；逐变量相对误差均值 cube={np.mean(rel_cube):.2e}，'
              f'SRS={np.mean(rel_srs):.2e}')
        assert mc < 0.4 * ms
    # phat+ae3：AE 前 3 个主成分也被平衡
    k = 200
    pi0, Xbal = D.cube_xbal(ctx, k, 'phat+ae3')
    ae_c = np.mean([D.balance_error(Xbal, pi0, REGISTRY['cube_ae3'](ctx, k, s))[1][-3:] for s in range(10)])
    ae_s = np.mean([D.balance_error(Xbal, pi0, REGISTRY['srs'](ctx, k, s))[1][-3:] for s in range(10)])
    print(f'    cube_ae3：AE 前 3 维总量的相对误差 cube={ae_c:.2e}，SRS={ae_s:.2e}')
    assert ae_c < 0.5 * ae_s


def test_cube_inclusion_probabilities():
    """cube 保持一阶包含概率（含不等概率）。"""
    reps = 300 if FAST else 1000
    rng0 = np.random.default_rng(0)
    N, k = 300, 30
    P = rng0.dirichlet(np.ones(4), size=N)
    for pi0 in (np.full(N, k / N), None):
        if pi0 is None:
            w = rng0.uniform(0.5, 2, N)
            pi0 = np.clip(k * w / w.sum(), 0, 1)
            pi0 *= k / pi0.sum()
        Xbal = np.column_stack([pi0, P[:, :3]])
        cnt = np.zeros(N)
        for s in range(reps):
            pi = D.cube_sample(pi0, Xbal, np.random.default_rng(s))
            assert set(np.unique(pi)) <= {0.0, 1.0} and pi.sum() == k
            cnt += pi
        f = cnt / reps
        z = (f - pi0) / np.sqrt(pi0 * (1 - pi0) / reps)
        print(f'    cube reps={reps}: |z|>3 的比例 {np.mean(np.abs(z) > 3):.4f}，mean z² = {np.mean(z ** 2):.3f}')
        assert np.mean(np.abs(z) > 3) < 0.02 and 0.75 < np.mean(z ** 2) < 1.25


def test_cube_stratified():
    ctx = make_toy_ctx(N=3000, seed=0)
    codes, N_h = D._codes(ctx.T_nuts3)
    for k in (50, 200):
        dev_s, dev_u = [], []
        for s in range(10):
            idx = REGISTRY['cube_nuts2'](ctx, k, s)       # 整数编码：退回 NUTS3 分层
            assert 'NUTS3' in ctx.meta['cube_nuts2']['strata_source']
            dev_s.append(np.abs(np.bincount(codes[idx], minlength=N_h.size) - N_h * k / ctx.N).max())
            idx = REGISTRY['cube'](ctx, k, s)
            dev_u.append(np.abs(np.bincount(codes[idx], minlength=N_h.size) - N_h * k / ctx.N).max())
        print(f'    k={k}: max_h|n_h − N_h k/N| 分层 cube={np.mean(dev_s):.2f}，不分层={np.mean(dev_u):.2f}')
        assert np.mean(dev_s) < 1.5 and np.mean(dev_s) < np.mean(dev_u)


# ---------------------------------------------------------------------
# M19 cLHS
# ---------------------------------------------------------------------

def test_clhs_objective():
    ctx = make_toy_ctx(N=3000, seed=0)
    Z = D.clhs_matrix(ctx)
    assert Z.shape == (3000, 12)
    for k in (50, 200):
        idx, info = D.clhs_anneal(Z, k, np.random.default_rng(0), max_iterations=10000)
        full = D.clhs_objective(Z, idx)
        assert abs(full[0] - info['obj']) < 1e-6 and abs(full[1] - info['O1']) < 1e-9   # 增量更新与从头计算一致
        rnd = np.mean([D.clhs_objective(Z, np.random.default_rng(s).choice(3000, k, replace=False))[0]
                       for s in range(5)])
        print(f'    k={k}: cLHS O={full[0]:.1f}（O1={full[1]:.0f}，O3={full[3]:.2f}），随机样本 O≈{rnd:.1f}')
        assert full[0] < 0.7 * rnd
    # 带类别变量（NUTS 编码）的 O2
    idx, info = D.clhs_anneal(Z, 100, np.random.default_rng(1), max_iterations=3000, cat=ctx.T_nuts3)
    full = D.clhs_objective(Z, idx, cat=ctx.T_nuts3)
    assert abs(full[0] - info['obj']) < 1e-6
    idx2 = REGISTRY['clhs'](ctx, 100, 1, cat='nuts2', max_iterations=3000)
    assert np.array_equal(idx, idx2)


# ---------------------------------------------------------------------
# 先验估计
# ---------------------------------------------------------------------

def test_prior_estimators_formulas():
    # Olofsson：手算例子
    y = np.array([0, 0, 1, 1, 1, 2])
    h = np.array([0, 0, 0, 1, 1, 1])
    N_h = np.array([60, 40])
    p, se = D.olofsson_prior(y, h, N_h, C=3)
    q = np.array([[2 / 3, 1 / 3, 0], [0, 2 / 3, 1 / 3]])
    W = np.array([0.6, 0.4])
    assert np.allclose(p, W @ q)
    assert np.allclose(se, np.sqrt((W[:, None] ** 2 * q * (1 - q) / 2).sum(0)))
    # 分层设计下 Hájek 与 Olofsson 点估计相同
    pi = (np.array([3, 3]) / N_h)[h]
    assert np.allclose(D.ht_prior(y, pi, C=3)[0], p)
    assert np.allclose(D.ht_prior(y, pi, C=3, N=100)[0], p)
    # 等概率：Hájek = 样本比例
    assert np.allclose(D.ht_prior(y, np.full(6, 0.1), C=3)[0], [2 / 6, 3 / 6, 1 / 6])
    # AIPW：P_T 完全正确时残差为 0，θ̂ = 全池均值
    rng = np.random.default_rng(0)
    yT = rng.integers(0, 3, 100)
    P = np.eye(3)[yT]
    sel = np.arange(0, 100, 10)
    th, se = D.aipw_prior(yT[sel], np.full(10, 0.1), P, sel)
    assert np.allclose(th, P.mean(0)) and np.allclose(se, 0)
    # 等概率时 AIPW = PPI 均值估计 mean_U f + mean_S (y − f)
    P2 = rng.dirichlet(np.ones(3), 100)
    th, _ = D.aipw_prior(yT[sel], np.full(10, 0.1), P2, sel, clip=False)
    assert np.allclose(th, P2.mean(0) + (np.eye(3)[yT[sel]] - P2[sel]).mean(0))


def test_prior_estimators_unbiased():
    """蒙特卡洛：M3 + Olofsson、M5 + AIPW、LPM2 + HT 的先验估计对真实先验无偏（误差 < 4 个蒙特卡洛标准误）。"""
    reps = 60 if FAST else 200
    ctx = make_toy_ctx(N=3000, seed=4)
    y = ctx._true_y
    truth = np.bincount(y, minlength=ctx.C) / ctx.N
    k = 200
    est = {'predstrat+olofsson': [], 'pmht+aipw': [], 'pmht_b4+aipw': [], 'lpm2+ht': [], 'cube+ht': []}
    ses = {key: [] for key in est}
    for r in range(reps):
        idx = REGISTRY['predstrat'](ctx, k, r)
        m = ctx.meta['predstrat']
        p, se = D.olofsson_prior(y[idx], m['strata'][idx], m['N_h'], C=ctx.C)
        est['predstrat+olofsson'].append(p)
        ses['predstrat+olofsson'].append(se)
        for name in ('pmht', 'pmht_b4'):
            idx = REGISTRY[name](ctx, k, r)
            p, se = D.aipw_prior(y[idx], ctx.meta[name]['pi'][idx], ctx.P_T, idx, clip=False)
            est[name + '+aipw'].append(p)
            ses[name + '+aipw'].append(se)
        for name in ('lpm2', 'cube'):
            idx = REGISTRY[name](ctx, k, r)
            p, se = D.ht_prior(y[idx], ctx.meta[name]['pi'][idx], C=ctx.C, N=ctx.N)
            est[name + '+ht'].append(p)
            ses[name + '+ht'].append(se)
    for key, v in est.items():
        v = np.array(v)
        mc_se = v.std(0, ddof=1) / np.sqrt(reps)
        z = (v.mean(0) - truth) / np.maximum(mc_se, 1e-12)
        emp_sd = v.std(0, ddof=1)
        print(f'    {key:20s}: max|偏差/MC标准误| = {np.abs(z).max():.2f}；经验 SD 之和 = {emp_sd.sum():.4f}，'
              f'平均 SE 估计之和 = {np.mean(ses[key], 0).sum():.4f}')
        assert np.abs(z).max() < 4.0, (key, z)


# ---------------------------------------------------------------------
# 耗时
# ---------------------------------------------------------------------

def test_timing_N30000_k1000():
    ctx = make_toy_ctx(N=30000, n_F=5000, seed=0)
    times = {}
    for m in METHODS:
        t = time.time()
        idx = REGISTRY[m](ctx, 1000, 0)
        times[m] = time.time() - t
        assert len(idx) == 1000
        assert times[m] < 180, (m, times[m])
    print('    N=30000, k=1000 耗时（秒）：' + '，'.join(f'{m} {t:.2f}' for m, t in times.items()))


# ---------------------------------------------------------------------
# 交叉核对（依赖外部实现，缺少时跳过）
# ---------------------------------------------------------------------

def _r_has_balancedsampling():
    if shutil.which('Rscript') is None:
        return False
    r = subprocess.run(['Rscript', '-e', 'suppressMessages(library(BalancedSampling))'], capture_output=True)
    return r.returncode == 0


def test_crosscheck_R_balancedsampling():
    """与 R BalancedSampling::lpm2 / cube 比较（平均最近邻距离、平衡误差）。"""
    if not _r_has_balancedsampling():
        _skip('没有 Rscript 或 BalancedSampling（设置 R_LIBS 指向其所在库后可运行）')
    reps = 50 if FAST else 150
    ctx = make_toy_ctx(N=2000, seed=3)
    N, k = ctx.N, 100
    X = X_of(ctx, 'ae', 16, geo_w=2.0)
    pi = np.full(N, k / N)
    pi0, Xbal = D.cube_xbal(ctx, k, 'phat')
    with tempfile.TemporaryDirectory() as td:
        np.savetxt(os.path.join(td, 'X.csv'), X, delimiter=',')
        np.savetxt(os.path.join(td, 'Xb.csv'), Xbal, delimiter=',')
        code = f'''suppressMessages(library(BalancedSampling))
X <- as.matrix(read.csv("{td}/X.csv", header=FALSE)); Xb <- as.matrix(read.csv("{td}/Xb.csv", header=FALSE))
pi <- rep({k}/{N}, {N}); set.seed(1)
a <- t(sapply(1:{reps}, function(r) sort(lpm2(pi, X))))
b <- t(sapply(1:{reps}, function(r) sort(cube(Xb[,1], Xb))))
write.table(a, "{td}/a.csv", sep=",", row.names=FALSE, col.names=FALSE)
write.table(b, "{td}/b.csv", sep=",", row.names=FALSE, col.names=FALSE)'''
        with open(os.path.join(td, 'x.R'), 'w') as f:
            f.write(code)
        subprocess.run(['Rscript', os.path.join(td, 'x.R')], check=True, capture_output=True)
        Ra = np.loadtxt(os.path.join(td, 'a.csv'), delimiter=',').astype(int) - 1
        Rb = np.loadtxt(os.path.join(td, 'b.csv'), delimiter=',').astype(int) - 1

    def mnn(idx):
        d, _ = cKDTree(X[idx]).query(X[idx], k=2)
        return d[:, 1].mean()
    nn_r = np.array([mnn(s) for s in Ra])
    nn_m = np.array([mnn(D.lpm2(pi, X, np.random.default_rng(s))) for s in range(reps)])
    se = np.sqrt(nn_r.var() / reps + nn_m.var() / reps)
    print(f'    LPM2 平均最近邻距离：R={nn_r.mean():.4f}，本实现={nn_m.mean():.4f}（差/SE={(nn_m.mean() - nn_r.mean()) / se:.2f}）')
    assert abs(nn_m.mean() - nn_r.mean()) < 4 * se
    be_r = np.array([D.balance_error(Xbal, pi0, s)[0] for s in Rb])
    be_m = np.array([D.balance_error(Xbal, pi0, np.flatnonzero(D.cube_sample(pi0, Xbal, np.random.default_rng(s)) > .5))[0]
                     for s in range(reps)])
    print(f'    Cube 平衡误差中位数：R={np.median(be_r):.3e}，本实现={np.median(be_m):.3e}')
    assert np.median(be_m) < 1.25 * np.median(be_r)


def test_crosscheck_clhs_package():
    """与 PyPI clhs 1.0.2 比较目标值（同迭代次数下本实现不应更差）。"""
    try:
        import clhs  # noqa: F401
    except ImportError:
        _skip('未安装 PyPI clhs')
    ctx = make_toy_ctx(N=2000, seed=3)
    Z = D.clhs_matrix(ctx)
    k, it = 50, (2000 if FAST else 5000)
    o_pkg, o_own = [], []
    for s in range(2):
        o_pkg.append(D.clhs_objective(Z, D.run_clhs_package(Z, k, s, max_iterations=it))[0])
        o_own.append(D.clhs_objective(Z, D.clhs_anneal(Z, k, np.random.default_rng(s), max_iterations=it)[0])[0])
    print(f'    cLHS 目标值（k={k}，{it} 次迭代）：clhs 1.0.2 = {np.round(o_pkg, 1)}，本实现 = {np.round(o_own, 1)}')
    assert np.mean(o_own) <= 1.05 * np.mean(o_pkg)
    # 包的调用不改变全局随机状态
    st = np.random.get_state()[1].copy()
    D.run_clhs_package(Z, 20, 0, max_iterations=50)
    assert np.array_equal(np.random.get_state()[1], st)


# ---------------------------------------------------------------------

def _main():
    tests = [(n, f) for n, f in globals().items() if n.startswith('test_') and callable(f)]
    npass = nfail = nskip = 0
    t_all = time.time()
    for n, f in tests:
        t = time.time()
        try:
            f()
            npass += 1
            print(f'PASS {n} ({time.time() - t:.1f}s)', flush=True)
        except unittest.SkipTest as e:
            nskip += 1
            print(f'SKIP {n}: {e}', flush=True)
        except Exception:
            nfail += 1
            import traceback
            traceback.print_exc()
            print(f'FAIL {n} ({time.time() - t:.1f}s)', flush=True)
    print(f'\n{npass} passed, {nfail} failed, {nskip} skipped in {time.time() - t_all:.0f}s')
    return nfail


if __name__ == '__main__':
    sys.exit(1 if _main() else 0)
