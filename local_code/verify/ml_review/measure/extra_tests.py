"""(3) 额外合成测试（不读真实标签）。运行：OMP_NUM_THREADS=2 python extra_tests.py <L>"""
import os
import sys
import tempfile
import time

L = sys.argv[1]
sys.path.insert(0, L)
os.chdir(L)
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import test_multilearner_G as TM  # noqa: E402  （把 ML.PARAMS 调小，并设 G.N_FOREIGN_PER=150、G.N_POOL=500）
import test_guidance_G as TG  # noqa: E402
import guidance_G as G  # noqa: E402
import multilearner_G as ML  # noqa: E402
from stageA import class_weights_from  # noqa: E402

TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp_extra')
os.makedirs(TMP, exist_ok=True)


def mk(same=False, drop_foreign=None, drop_all=None):
    """合成数据 + G 层级 1 选样。drop_foreign：外国国家里这一类改标为 0（外国池缺这一类，目标国仍有）；
    drop_all：所有国家里这一类改标为 1（完全缺类）。"""
    D = TG.SynthD(same=same)
    if drop_foreign is not None:
        D.y[(D.ctry != 'PT') & (D.y == drop_foreign)] = 0
    if drop_all is not None:
        D.y[D.y == drop_all] = 1
    S = G.stage0_compute(D, 'PT', 0, threads=2, trees=5)
    g = tempfile.mkdtemp(dir=TMP)
    os.makedirs(os.path.join(g, 'stage0'))
    np.savez_compressed(os.path.join(g, 'stage0', 'PT_s0.npz'), **S)
    for rep in (0, 1):
        G.run_unit(D, S, 'PT', 0, rep, 1, threads=2, out=g, trees=5, log=lambda s: None)
    return D, S, g


def X1():
    """训练集缺类（外国池缺 9 类、目标国有；7 类全缺）：三种学习器端到端不崩溃，概率 10 列合法；k=0 时第 9、7 列恒为 0；
    选样含 9 类时第 9 列非零；cw[9] = 1。"""
    D, S, g = mk(drop_foreign=9, drop_all=7)
    ML.G_OUT = g
    fr = S['foreign']
    assert (D.y[fr] == 9).sum() == 0 and (D.y == 7).sum() == 0 and (D.y[S['test']] == 9).sum() > 0
    assert class_weights_from(D.y[fr])[9] == 1.0
    for learner in ML.LEARNERS:
        m = tempfile.mkdtemp(dir=TMP)
        R = pd.DataFrame(ML.run_unit_ml(D, learner, 'PT', 0, 0, 2, out=m, log=lambda s: None))
        assert len(R) == 42, len(R)
        for f in os.listdir(os.path.join(m, 'probs', learner)):
            P = np.load(os.path.join(m, 'probs', learner, f))['P']
            assert P.shape[1] == 10 and np.isfinite(P).all() and np.allclose(P.sum(1), 1, atol=1e-4), (learner, f)
            assert np.all(P[:, 7] == 0), (learner, f)
            if '_k0' in f:
                assert np.all(P[:, 9] == 0), (learner, f)
        has9 = R[(R.k > 0)].copy()
        # 选样里含 9 类的某个拟合，第 9 列应非零
        n9 = 0
        for _, r in has9.iterrows():
            z = np.load(ML.sel_path('PT', 0, 0, r.cond, r.method, r.k))
            if (D.y[z['sel']] == 9).any():
                P = np.load(os.path.join(m, 'probs', learner, f'PT_s0_r0_{r.clf}_k{r.k}_{r.sel_hash}.npz'))['P']
                assert P[:, 9].max() > 0, (learner, r.cond, r.method, r.k)
                n9 += 1
        assert n9 > 0
        assert R[['ba_ge30', 'oa', 'ba_P1', 'ba_P2']].notna().all().all(), learner


def X2():
    """k=0 共享缓存与运行顺序无关：先跑 rep1 再跑 rep0（两个新目录），k=0 概率文件逐位相同；rep1 复用 rep0 的缓存。"""
    D, S, g, _m = TM.env()
    ML.G_OUT = g
    for learner in ML.LEARNERS:
        a, b = tempfile.mkdtemp(dir=TMP), tempfile.mkdtemp(dir=TMP)
        ra0 = pd.DataFrame(ML.run_unit_ml(D, learner, 'PT', 0, 0, 2, out=a, log=lambda s: None))
        ra1 = pd.DataFrame(ML.run_unit_ml(D, learner, 'PT', 0, 1, 2, out=a, log=lambda s: None))
        rb1 = pd.DataFrame(ML.run_unit_ml(D, learner, 'PT', 0, 1, 2, out=b, log=lambda s: None))
        assert ra1[ra1.k == 0].cache_hit.all() and not rb1[rb1.k == 0].cache_hit.any()
        for clf in ('ae', 's2'):
            Pa = np.load(os.path.join(a, 'probs', learner, f'PT_s0_{clf}_k0.npz'))['P']
            Pb = np.load(os.path.join(b, 'probs', learner, f'PT_s0_{clf}_k0.npz'))['P']
            assert np.array_equal(Pa, Pb), (learner, clf)
        met = ['oa', 'ba_ge30', 'ba_P1', 'ba_P2', 'ba_far']
        x = ra1[ra1.k == 0].set_index('clf')[met].sort_index()
        y = rb1[rb1.k == 0].set_index('clf')[met].sort_index()
        assert x.equals(y), learner
        # k=0 的 P2 用各自重复的 pi_pool：两个重复的 pi_pool 不同，但 P0 指标相同
        assert ra0[ra0.k == 0].set_index('clf')[['oa', 'ba_ge30']].sort_index().equals(x[['oa', 'ba_ge30']])


def X3():
    """C 选择对非外国行不敏感：非外国行的特征换成任意有限值、标签随机重排，损失字典逐位相同；对外国行的扰动则会改变损失。"""
    D, S, g, _m = TM.env()
    fr = S['foreign']
    r = np.random.default_rng(3)
    D2 = TG.SynthD()
    mask = np.ones(len(D2.y), bool)
    mask[fr] = False
    D2.y = D2.y.copy()
    D2.y[mask] = r.permutation(D2.y[mask])
    for f in ('ae', 's2'):
        D2.X[f] = D2.X[f].copy()
        D2.X[f][mask] = r.normal(size=(mask.sum(), D2.X[f].shape[1])) * 100
    import json
    for f in ('ae', 's2'):
        o1, o2 = tempfile.mkdtemp(dir=TMP), tempfile.mkdtemp(dir=TMP)
        c1 = ML.select_C(TG.SynthD(), S, 'PT', 0, f, 2, o1)
        c2 = ML.select_C(D2, S, 'PT', 0, f, 2, o2)
        l1 = json.load(open(os.path.join(o1, 'lr_C', f'PT_s0_{f}.json')))
        l2 = json.load(open(os.path.join(o2, 'lr_C', f'PT_s0_{f}.json')))
        assert c1 == c2 and l1 == l2, f
    # 反向对照：扰动外国行应改变损失（确认测试有检出力）
    D3 = TG.SynthD()
    D3.X['ae'] = D3.X['ae'].copy()
    D3.X['ae'][fr[:50]] += 5
    o3 = tempfile.mkdtemp(dir=TMP)
    ML.select_C(D3, S, 'PT', 0, 'ae', 2, o3)
    l3 = json.load(open(os.path.join(o3, 'lr_C', 'PT_s0_ae.json')))
    o1 = tempfile.mkdtemp(dir=TMP)
    ML.select_C(TG.SynthD(), S, 'PT', 0, 'ae', 2, o1)
    l1 = json.load(open(os.path.join(o1, 'lr_C', 'PT_s0_ae.json')))
    assert l1['loss'] != l3['loss']


def X4():
    """C 选择的加权对数损失与独立实现（sklearn.metrics.log_loss，sample_weight = cw[y]，按组拼接袋外概率）一致。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    import json
    D, S, g, _m = TM.env()
    fr = S['foreign']
    y = D.y[fr]
    X = D.X['ae'][fr].astype(np.float64)
    w = class_weights_from(y)[y]
    grp = G.fold_groups(D, 'PT', 0, fr)
    o = tempfile.mkdtemp(dir=TMP)
    ML.select_C(D, S, 'PT', 0, 'ae', 2, o)
    got = json.load(open(os.path.join(o, 'lr_C', 'PT_s0_ae.json')))['loss']
    for C in ML.C_GRID:
        P = np.zeros((len(y), 10))
        for gg in np.unique(grp):
            tr, te = grp != gg, grp == gg
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            m = LogisticRegression(C=C, max_iter=ML.PARAMS['lr']['max_iter'], tol=1e-4).fit(
                (X[tr] - mu) / sd, y[tr], sample_weight=w[tr] / w[tr].mean())
            P[np.ix_(te, m.classes_)] = m.predict_proba((X[te] - mu) / sd)
        ref = log_loss(y, np.clip(P, 1e-12, None) / np.clip(P, 1e-12, None).sum(1, keepdims=True), sample_weight=w,
                       labels=list(range(10)))
        assert abs(ref - got[str(C)]) < 1e-6 * max(1, ref), (C, ref, got[str(C)])


def X5():
    """测试集行之间互不影响（标准化只用训练集）：改动 X_te 的后半部分，前半部分的概率逐位不变（LR、MLP、RF）。"""
    r = np.random.default_rng(2)
    X = r.normal(size=(800, 12))
    y = r.choice(10, 800)
    w = r.uniform(0.5, 3, 800)
    Xt = r.normal(size=(60, 12))
    Xt2 = Xt.copy()
    Xt2[30:] = r.normal(size=(30, 12)) * 50 + 7
    for Lr in ML.LEARNERS:
        a, _ = ML.fit_predict_learner(Lr, X, y, w, Xt, 3, 1, 0.1)
        b, _ = ML.fit_predict_learner(Lr, X, y, w, Xt2, 3, 1, 0.1)
        assert np.array_equal(a[:30], b[:30]), Lr


def X6():
    """线程数对结果的影响（复现性披露）：RF 与 MLP 在 threads=1 与 4 下是否逐位相同；LR 同。只报告，不断言。"""
    r = np.random.default_rng(4)
    X = r.normal(size=(3000, 64))
    y = r.choice(10, 3000)
    w = r.uniform(0.5, 3, 3000)
    Xt = r.normal(size=(200, 64))
    out = {}
    for Lr in ML.LEARNERS:
        a, _ = ML.fit_predict_learner(Lr, X, y, w, Xt, 3, 1, 0.1)
        b, _ = ML.fit_predict_learner(Lr, X, y, w, Xt, 3, 4, 0.1)
        out[Lr] = (bool(np.array_equal(a, b)), float(np.abs(a - b).max()))
    print('   threads 1 vs 4 逐位相同 / 最大差：', out)


def X7():
    """sel 文件缺失、或选样与测试集重叠时应报错（防御性）：伪造一个与测试集重叠的选样文件。"""
    D, S, g, _m = TM.env()
    g2 = tempfile.mkdtemp(dir=TMP)
    import shutil
    shutil.copytree(os.path.join(g, 'stage0'), os.path.join(g2, 'stage0'))
    shutil.copytree(os.path.join(g, 'sel'), os.path.join(g2, 'sel'))
    p = os.path.join(g2, 'sel', 'PT_s0_r0_AE_kcenter_k50.npz')
    z = dict(np.load(p))
    z['sel'] = z['sel'].copy()
    z['sel'][0] = S['test'][0]
    np.savez_compressed(p, **z)
    ML.G_OUT = g2
    try:
        ML.run_unit_ml(D, 'lr', 'PT', 0, 0, 2, out=tempfile.mkdtemp(dir=TMP), log=lambda s: None)
        raise RuntimeError('应当报错')
    except AssertionError:
        pass
    finally:
        ML.G_OUT = g


TESTS = {n: f for n, f in globals().items() if n[0] == 'X' and n[1:].isdigit()}
if __name__ == '__main__':
    names = sys.argv[2:] or sorted(TESTS, key=lambda s: int(s[1:]))
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
