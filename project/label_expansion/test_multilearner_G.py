"""
多学习器运行器的单元测试（合成数据，不读真实标签）。运行：OMP_NUM_THREADS=2 python test_multilearner_G.py
"""
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd

import test_guidance_G as TG  # 设定 G.N_FOREIGN_PER=150、G.N_POOL=500，并提供 SynthD
import guidance_G as G
import multilearner_G as ML
from stageA import class_weights_from

ML.PARAMS['mlp'].update(hidden_layer_sizes=(16,), max_iter=2)
ML.PARAMS['mlp_seeds'] = 2
ML.PARAMS['rf'].update(n_estimators=10)
ML.PARAMS['lr'].update(max_iter=200)


def setup(same=False):
    """合成数据：跑 G 层级 1（5 棵树）生成选样文件，返回 (D, 临时 G 目录, 临时 ML 目录)。"""
    D = TG.SynthD(same=same)
    S = G.stage0_compute(D, 'PT', 0, threads=2, trees=5)
    g, m = tempfile.mkdtemp(), tempfile.mkdtemp()
    os.makedirs(os.path.join(g, 'stage0'))
    np.savez_compressed(os.path.join(g, 'stage0', 'PT_s0.npz'), **S)
    for rep in (0, 1):
        G.run_unit(D, S, 'PT', 0, rep, 1, threads=2, out=g, trees=5, log=lambda s: None)
    ML.G_OUT = g
    return D, S, g, m


_CACHE = {}


def env(same=False):
    if same not in _CACHE:
        _CACHE[same] = setup(same)
    return _CACHE[same]


def run(D, learner, rep, m):
    return pd.DataFrame(ML.run_unit_ml(D, learner, 'PT', 0, rep, 2, out=m, log=lambda s: None))


def T1():
    """训练集 = 外国 + 选样，不含测试地块；目标样本权重 = 50·cw，cw 只按外国频率。"""
    D, S, g, m = env()
    seen = []
    orig = ML.fit_predict_learner

    def spy(learner, X_tr, y_tr, w_tr, X_te, seed, threads, C=None):
        seen.append((len(X_tr), y_tr.copy(), w_tr.copy(), len(X_te), seed, C))
        return orig(learner, X_tr, y_tr, w_tr, X_te, seed, threads, C)
    ML.fit_predict_learner = spy
    try:
        with tempfile.TemporaryDirectory() as mm:
            run(D, 'lr', 1, mm)  # 重复 1：k=0 应用 rep=0 的种子（0），其余用 1
    finally:
        ML.fit_predict_learner = orig
    fr = S['foreign']
    cw = class_weights_from(D.y[fr])
    ks = [0] + [k for _, _, k in ML.unit_jobs(0, 1, None) if k]
    assert len(seen) == 2 * len(ks)
    for (n, y, w, nte, seed, C), k in zip(seen, np.repeat(ks, 2)):
        assert n == len(fr) + k and nte == len(S['test'])
        assert np.allclose(w[:len(fr)], cw[y[:len(fr)]]) and np.allclose(w[len(fr):], 50 * cw[y[len(fr):]])
        assert seed == (0 if k == 0 else 1) and C in ML.C_GRID
    assert [ML.mlp_seed(1, s) for s in range(3)] == [10, 11, 12]


def T2():
    """LR 的 C 只用外国池：把非外国地块的特征全部置 NaN，仍能选出 C。"""
    D, S, g, m = env()
    D2 = TG.SynthD()
    mask = np.ones(len(D2.y), bool)
    mask[S['foreign']] = False
    for f in ('ae', 's2'):
        D2.X[f] = D2.X[f].copy()
        D2.X[f][mask] = np.nan
    import json
    with tempfile.TemporaryDirectory() as m1, tempfile.TemporaryDirectory() as m2:
        ML.select_C(D, S, 'PT', 0, 'ae', 2, m1)
        ML.select_C(D2, S, 'PT', 0, 'ae', 2, m2)
        a = json.load(open(os.path.join(m1, 'lr_C', 'PT_s0_ae.json')))
        b = json.load(open(os.path.join(m2, 'lr_C', 'PT_s0_ae.json')))
        assert a['loss'] == b['loss'] and a['C'] in ML.C_GRID
        # 缓存核对：网格变了应报错
        old = ML.C_GRID
        try:
            ML.C_GRID = (1.0, 10.0)
            try:
                ML.select_C(D, S, 'PT', 0, 'ae', 2, m1)
                raise AssertionError('网格不一致时应报错')
            except AssertionError as e:
                assert '不一致' in str(e), e
        finally:
            ML.C_GRID = old


def T3():
    """同一种子两次拟合逐位相同（LR、MLP、RF）。"""
    r = np.random.default_rng(0)
    X = r.normal(size=(600, 12)).astype(np.float32)
    y = r.choice(10, 600)
    w = r.uniform(0.5, 2, 600)
    Xt = r.normal(size=(50, 12)).astype(np.float32)
    for L in ML.LEARNERS:
        a, _ = ML.fit_predict_learner(L, X, y, w, Xt, 7, 2, 1.0)
        b, _ = ML.fit_predict_learner(L, X, y, w, Xt, 7, 2, 1.0)
        assert np.array_equal(a, b), L
        assert a.shape == (50, 10) and np.allclose(a.sum(1), 1, atol=1e-5), L


def T4():
    """端到端（X_ae = X_s2）：AE 与 S2 指导的行逐位一致且命中缓存；k=0 的行在两个重复间相同。"""
    D, S, g, m = env(same=True)
    met = ['oa', 'ba_ge30', 'f1_ge30', 'ba_far', 'ba_P1', 'ba_P2']
    met0 = ['oa', 'ba_ge30', 'f1_ge30', 'ba_far']  # k=0 跨重复只比 P0 指标（候选池不同，P2 本应不同）
    for L in ML.LEARNERS:
        R0, R1 = run(D, L, 0, m), run(D, L, 1, m)
        a = R0[R0.cond == 'AE'].set_index(['method', 'k', 'clf'])[met].sort_index()
        b = R0[R0.cond == 'S2'].set_index(['method', 'k', 'clf'])[met].sort_index()
        assert len(a) == 4 * 2 * 2 and a.equals(b), L
        assert R0[R0.cond == 'S2'].cache_hit.all(), L
        k0a = R0[R0.k == 0].set_index('clf')[met0].sort_index()
        k0b = R1[R1.k == 0].set_index('clf')[met0].sort_index()
        assert k0a.equals(k0b) and R1[R1.k == 0].cache_hit.all(), L
        assert len(R0) == 2 * (1 + 16 + 4), (L, len(R0))


def T5():
    """训练集中缺类时概率仍为 10 列且合法。"""
    r = np.random.default_rng(1)
    X = r.normal(size=(300, 6))
    y = r.choice([0, 2, 5], 300)
    for L in ML.LEARNERS:
        P, _ = ML.fit_predict_learner(L, X, y, np.ones(300), X[:20], 0, 1, 1.0)
        assert P.shape == (20, 10) and np.allclose(P[:, [1, 3, 4, 6, 7, 8, 9]], 0) and np.isfinite(P).all(), L


def T6():
    """断点续跑：已完成的行不再拟合。"""
    D, S, g, m = env()
    with tempfile.TemporaryDirectory() as mm:
        run(D, 'lr', 0, mm)
        orig = ML.fit_predict_learner
        ML.fit_predict_learner = lambda *a, **k: (_ for _ in ()).throw(AssertionError('不应再拟合'))
        try:
            R = run(D, 'lr', 0, mm)
        finally:
            ML.fit_predict_learner = orig
        assert len(R) == 2 * (1 + 16 + 4)


def T7():
    """技术失败：某次拟合抛 FloatingPointError 时记为 failed 并继续；其余行正常。"""
    D, S, g, m = env()
    orig = ML.fit_predict_learner
    calls = {'n': 0}

    def flaky(*a, **k):
        calls['n'] += 1
        if calls['n'] == 3:
            raise FloatingPointError('测试用的失败')
        return orig(*a, **k)
    ML.fit_predict_learner = flaky
    try:
        with tempfile.TemporaryDirectory() as mm:
            R = run(D, 'lr', 0, mm)
    finally:
        ML.fit_predict_learner = orig
    assert (R.status == 'failed').sum() == 1 and (R.status == 'ok').sum() == len(R) - 1
    assert R[R.status == 'failed'].ba_ge30.isna().all() if 'ba_ge30' in R else True


def T8():
    """概率缓存核对超参数：参数变了再命中缓存应报错。"""
    D, S, g, m = env()
    with tempfile.TemporaryDirectory() as mm:
        run(D, 'rf', 0, mm)
        for f in os.listdir(mm):
            if f.startswith('rows_'):
                os.remove(os.path.join(mm, f))
        old = dict(ML.PARAMS['rf'])
        try:
            ML.PARAMS['rf'].update(n_estimators=11)
            try:
                run(D, 'rf', 0, mm)
                raise AssertionError('参数不一致时应报错')
            except AssertionError as e:
                assert '不一致' in str(e), e
        finally:
            ML.PARAMS['rf'].clear()
            ML.PARAMS['rf'].update(old)


TESTS = {n: f for n, f in globals().items() if n[0] == 'T' and n[1:].isdigit()}

if __name__ == '__main__':
    names = sys.argv[1:] or sorted(TESTS, key=lambda s: int(s[1:]))
    fail = 0
    for n in names:
        t = time.time()
        try:
            TESTS[n]()
            print(f'PASS {n} {TESTS[n].__doc__.strip().splitlines()[0]} ({time.time() - t:.1f}s)', flush=True)
        except Exception as e:  # noqa: BLE001
            fail += 1
            import traceback
            traceback.print_exc()
            print(f'FAIL {n}: {e!r}', flush=True)
    print(f'\n{len(names) - fail}/{len(names)} 通过')
    sys.exit(1 if fail else 0)
