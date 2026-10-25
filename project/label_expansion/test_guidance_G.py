"""
G 运行器的单元测试（方案 ANALYSIS_PLAN_guidance.md；编号同 GUIDANCE_DESIGN_REVIEW.md 的实现待办）。
全部在合成数据上运行，不读真实标签。运行：
    OMP_NUM_THREADS=2 python test_guidance_G.py            # 全部
    OMP_NUM_THREADS=2 python test_guidance_G.py T10 T19    # 只跑指定的
"""
import ast
import inspect
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd

import guidance_G as G
import samplers.base as B

G.N_FOREIGN_PER = 150
G.N_POOL = 500
TREES = 5
COUNTRIES = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']


class SynthD:
    """11 国合成数据：类别中心 + 国家平移 + 噪声；same=True 时 S2 特征与 AE 完全相同。"""
    def __init__(self, n_per=900, same=False, seed=0):
        r = np.random.default_rng(seed)
        C = 10
        centers = r.normal(size=(C, 128)) * 1.5
        ys, ctry, lon, lat, Xa = [], [], [], [], []
        for j, c in enumerate(COUNTRIES):
            p = r.dirichlet(np.ones(C) * 2)
            y = r.choice(C, n_per, p=p)
            ys.append(y)
            ctry += [c] * n_per
            lon.append(j * 2.0 + r.uniform(0, 1.5, n_per))
            lat.append(45 + r.uniform(0, 1.5, n_per))
            Xa.append(centers[y] + r.normal(scale=0.3, size=128) + r.normal(size=(n_per, 128)))
        self.y = np.concatenate(ys)
        self.ctry = np.array(ctry)
        self.lon, self.lat = np.concatenate(lon), np.concatenate(lat)
        self.area = r.uniform(0.1, 5, len(self.y))
        Xa = np.vstack(Xa).astype(np.float32)
        Xs = Xa.copy() if same else np.hstack([Xa[:, ::-1] ** 3 / 10, r.normal(size=(len(Xa), 40))]).astype(np.float32)
        self.X = {'ae': Xa, 's2': Xs}
        self.countries = COUNTRIES
        self.nuts3 = np.array(['NA'] * len(self.y))
        mk = lambda p, n: [f'{p}_{i // 2:02d}_{"mean" if i % 2 == 0 else "stdDev"}' for i in range(n)]  # noqa: E731
        self.cols = {'ae': mk('ae', 128), 's2': mk('s2', Xs.shape[1])}

    def fusion(self):
        if 'fus' not in self.X:
            self.X['fus'] = np.hstack([self.X['ae'], self.X['s2']])
        return self.X['fus']


_CACHE = {}


def synth(same=False):
    if same not in _CACHE:
        D = SynthD(same=same)
        S = G.stage0_compute(D, 'PT', 0, threads=2, trees=TREES)
        _CACHE[same] = (D, S)
    return _CACHE[same]


def sel(D, S, cond, m, k=50, seed=0, oracle=None):
    return G.select_one(D, S, G.pool_positions(S, 0, 0), cond, m, k, seed, oracle)[0]


def fake_oracle(S, n):
    r = np.random.default_rng(5)
    return dict(G=r.normal(size=(n, 10)).astype(np.float32), P_T=r.dirichlet(np.ones(10), n))


ALL_M = G.E1 + G.PRED + G.HYB


# ------------------------------------------------------------------ T1–T9（原型的 9 项）
def T1():
    """无关字段为 None 时全部 G 方法、全部条件都能运行，且输出合法。"""
    D, S = synth()
    n = len(G.pool_positions(S, 0, 0))
    orc = fake_oracle(S, n)
    for cond, ms in (('AE', ALL_M), ('S2', ALL_M), ('XY', ['lpm2_nogeo', 'kcenter']),
                     ('S2XY', ['lpm2_nogeo', 'kcenter']), ('ORACLE', ['lpm2_nogeo', 'typiclust', 'margin']),
                     ('AEg_S2p', G.HYB), ('S2g_AEp', G.HYB), ('RAWZ_AE', G.E1), ('NATIVE_S2', ['lpm2_nogeo', 'typiclust']),
                     ('AE', ['kcenter_foreign']), ('none', ['srs'])):
        for m in ms:
            s = sel(D, S, cond, m, 50, 0, orc)
            assert len(s) == 50 and len(np.unique(s)) == 50, (cond, m)


def T2():
    """S2 指导设为与 AE 相同时，各对条件的选样逐位一致。"""
    D, S = synth()
    S2 = dict(S, G_s2=S['G_ae'], P_T_s2=S['P_T_ae'], GF_s2=S['GF_ae'])
    for m in ALL_M + ['kcenter_foreign']:
        assert np.array_equal(sel(D, S2, 'AE', m), sel(D, S2, 'S2', m)), m
    for m in G.HYB:
        assert np.array_equal(sel(D, S2, 'AEg_S2p', m), sel(D, S2, 'S2g_AEp', m)), m


def T3():
    """交换 AE 与 S2 的指导数据后，两条件的选样对调。"""
    D, S = synth()
    Sw = dict(S, G_ae=S['G_s2'], G_s2=S['G_ae'], P_T_ae=S['P_T_s2'], P_T_s2=S['P_T_ae'],
              GF_ae=S['GF_s2'], GF_s2=S['GF_ae'])
    for m in ALL_M:
        assert np.array_equal(sel(D, Sw, 'AE', m), sel(D, S, 'S2', m)), m
        assert np.array_equal(sel(D, Sw, 'S2', m), sel(D, S, 'AE', m)), m


def T4():
    """几何组对预测（P_T）的替换不变。"""
    D, S = synth()
    r = np.random.default_rng(1)
    Sp = dict(S, P_T_ae=r.dirichlet(np.ones(10), len(S['P_T_ae'])).astype(np.float32))
    for m in G.E1:
        assert np.array_equal(sel(D, Sp, 'AE', m), sel(D, S, 'AE', m)), m


def T5():
    """gist_maxmin 实际读 P_T（替换后选样改变），因此不在任何 G 家族中。"""
    D, S = synth()
    r = np.random.default_rng(1)
    Sp = dict(S, P_T_ae=r.dirichlet(np.ones(10), len(S['P_T_ae'])).astype(np.float32))
    pp = G.pool_positions(S, 0, 0)
    a = B.REGISTRY['gist_maxmin'](G.build_ctx_guided(D, Sp, pp, 'AE'), 50, 0, d=32)
    b = B.REGISTRY['gist_maxmin'](G.build_ctx_guided(D, S, pp, 'AE'), 50, 0, d=32)
    assert not np.array_equal(a, b)
    assert all('gist_maxmin' not in v for v in G.FAMILIES.values())


def T6():
    """预测组对几何的替换不变。"""
    D, S = synth()
    r = np.random.default_rng(2)
    Sg = dict(S, G_ae=r.normal(size=S['G_ae'].shape).astype(np.float32))
    for m in G.PRED:
        assert np.array_equal(sel(D, Sg, 'AE', m), sel(D, S, 'AE', m)), m


def T7():
    """签名里有 geo_w 的方法都显式收到 0；ctx.T_xy=None，geo_w>0 时直接报错。"""
    G.check_geo_w()
    old = G.G_HP['lpm2_nogeo']
    try:
        G.G_HP['lpm2_nogeo'] = {'d': 16, 'geo_w': 2.0}
        try:
            G.check_geo_w()
            raise AssertionError('check_geo_w 应当失败')
        except AssertionError as e:
            if 'check_geo_w' in str(e):
                raise
        D, S = synth()
        try:
            sel(D, S, 'AE', 'lpm2_nogeo')
            raise AssertionError('geo_w=2 应当因 T_xy=None 报错')
        except (TypeError, AttributeError):
            pass
    finally:
        G.G_HP['lpm2_nogeo'] = old


def T8():
    """XY 条件对坐标的整体平移不变（坐标先居中）。"""
    D, S = synth()
    D2 = SynthD()
    D2.lon = D.lon + 0.0  # 经度整体平移只改变 x 的常数项
    D2.lon[:] += 0.7
    for m in ('lpm2_nogeo', 'kcenter'):
        assert np.array_equal(sel(D, S, 'XY', m), sel(D2, S, 'XY', m)), m


def T9():
    """阶段 0：特征相同时，AE 与 S2 的 P_T、温度、袋外概率、几何逐位一致，分组共用。"""
    D, S = synth(same=True)
    for a, b in (('P_T_ae', 'P_T_s2'), ('P_F_oof_ae', 'P_F_oof_s2'), ('G_ae', 'G_s2'), ('GF_ae', 'GF_s2'),
                 ('RZ_ae', 'RZ_s2')):
        assert np.array_equal(S[a], S[b]), a
    assert S['temp_ae'] == S['temp_s2']
    assert len(np.unique(S['groups'])) == G.N_GROUPS


# ------------------------------------------------------------------ T10–T20
def _run(D, S, tier, out, rep=0):
    return pd.DataFrame(G.run_unit(D, S, 'PT', 0, rep, tier, threads=2, out=out, trees=TREES, log=lambda s: None))


def T10():
    """端到端（合成数据，5 棵树，X_ae = X_s2）：AE 与 S2 条件的指标行逐位一致。"""
    D, S = synth(same=True)
    with tempfile.TemporaryDirectory() as out:
        R = _run(D, S, 1, out)
    met = ['oa', 'ba_ge30', 'f1_ge30', 'oa_P1', 'ba_P1', 'oa_P2', 'ba_P2', 'ba_far', 'sel_hash']
    a = R[R.cond == 'AE'].set_index(['method', 'k', 'clf'])[met].sort_index()
    b = R[R.cond == 'S2'].set_index(['method', 'k', 'clf'])[met].sort_index()
    assert len(a) == 8 * 2 * 2 and a.equals(b)
    assert R[R.cond == 'S2'].cache_hit.all()


def T11():
    """SRS 与指导无关：只以 cond='none' 出现，且不在任何家族中。"""
    D, S = synth()
    with tempfile.TemporaryDirectory() as out:
        R = _run(D, S, 1, out, rep=1)
    assert set(R[R.method == 'srs'].cond) == {'none'}
    assert all('srs' not in v for v in G.FAMILIES.values())
    assert set(R[R.cond == 'AE'].method) == set(G.E1)  # 重复 1 只跑 E1
    assert sorted(R[R.method == 'srs'].k.unique()) == [50, 100, 200, 400]


def T12():
    """预处理函数没有按特征名的分支：源码里没有字符串常量；打乱列序后两两距离不变。"""
    for fn in (G.preprocess_guidance, G.preprocess_rawz):
        tree = ast.parse(inspect.getsource(fn))
        strs = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        assert all(s.startswith(('稳健', '阶段')) for s in strs), strs  # 只允许文档字符串
    r = np.random.default_rng(3)
    Z = r.normal(size=(400, 20)) * r.uniform(0.1, 5, 20) + r.standard_t(3, size=(400, 20))
    perm = r.permutation(20)
    A, Bm = G.preprocess_guidance(Z, n_comp=8), G.preprocess_guidance(Z[:, perm], n_comp=8)
    da = ((A[:, None] - A[None]) ** 2).sum(-1)
    db = ((Bm[:, None] - Bm[None]) ** 2).sum(-1)
    assert np.allclose(da, db, rtol=1e-3, atol=1e-3)


def T13():
    """把条件改名后选样不变（种子与条件名无关）。"""
    D, S = synth()
    G.COND['ZZ_renamed'] = G.COND['AE']
    try:
        for m in ALL_M:
            assert np.array_equal(sel(D, S, 'ZZ_renamed', m), sel(D, S, 'AE', m)), m
    finally:
        del G.COND['ZZ_renamed']


def T14():
    """S2XY：X_of 输出 d+2 列，坐标列方差份额 = 2·geo_w²/(d + 2·geo_w²)。"""
    D, S = synth()
    pp = G.pool_positions(S, 0, 0)
    for d in (16, 32):
        ctx = G.build_ctx_guided(D, S, pp, 'S2XY', d)
        X = B.X_of(ctx, 'ae', d + 2)
        assert X.shape[1] == d + 2
        v = X.var(0)
        share = v[-2:].sum() / v.sum()
        gw = G.GEO_W_S2XY
        assert abs(share - 2 * gw ** 2 / (d + 2 * gw ** 2)) < 1e-9, (d, share)


def T15():
    """badge_c 对 T 的整体平移不变。"""
    D, S = synth()
    S2 = dict(S, G_ae=S['G_ae'] + 50.0)
    assert np.array_equal(sel(D, S, 'AE', 'badge_c', 200), sel(D, S2, 'AE', 'badge_c', 200))


def T16():
    """来源标记：把另一来源的数组全部置 NaN 后，ctx 中没有 NaN（AE 条件不含 S2 数组，反之亦然；XY 只含坐标）。"""
    D, S = synth()
    pp = G.pool_positions(S, 0, 0)
    nan_like = lambda a: np.full_like(a, np.nan, dtype=np.float32)  # noqa: E731
    fl = lambda k: np.ndim(S[k]) > 0 and np.asarray(S[k]).dtype.kind == 'f'  # noqa: E731
    s2_keys = [k for k in S if (k.endswith('_s2') or k.startswith('P0test_s2')) and fl(k)]
    ae_keys = [k for k in S if (k.endswith('_ae') or k.startswith('P0test_ae')) and fl(k)]
    no_s2 = dict(S, **{k: nan_like(S[k]) for k in s2_keys})
    no_ae = dict(S, **{k: nan_like(S[k]) for k in ae_keys})
    no_both = dict(no_s2, **{k: nan_like(S[k]) for k in ae_keys})

    def arrays(ctx):
        return [v for v in (ctx.T_ae, ctx.P_T, ctx.F_ae) if v is not None]
    for Sx, cond in ((no_s2, 'AE'), (no_ae, 'S2'), (no_both, 'XY')):
        for fw in (False, True):
            ctx = G.build_ctx_guided(D, Sx, pp, cond, 32, with_foreign=fw)
            assert all(np.isfinite(a).all() for a in arrays(ctx)), cond
    ctx = G.build_ctx_guided(D, no_both, pp, 'XY', 32)
    assert ctx.T_ae.shape[1] == 2 and ctx.P_T is None
    for f in ('T_s2', 'T_xy', 'T_nuts3', 'F_s2', 'F_xy', 'F_country', 'P_T_loco', 'dom', 'P_F_oof'):
        assert getattr(ctx, f) is None, f


def T17():
    """ORACLE 交叉拟合：改动地块 i 的标签，i 自己的袋外概率与 LDA 分数不变；测试集地块不进入候选池。"""
    D, S = synth()
    pp = G.pool_positions(S, 0, 0)
    idx = S['allT'][pp]
    assert not np.intersect1d(idx, S['test']).size
    o1 = G.oracle_compute(D, idx, threads=2, trees=TREES)
    D2 = SynthD()
    i = 17
    D2.y = D.y.copy()
    D2.y[idx[i]] = (D.y[idx[i]] + 3) % 10
    o2 = G.oracle_compute(D2, idx, threads=2, trees=TREES)
    assert np.array_equal(o1['P_oof'][i], o2['P_oof'][i])
    assert np.allclose(o1['G'][i], o2['G'][i])
    assert not np.array_equal(o1['P_oof'], o2['P_oof'])  # 其他折确实用到了这个标签


def T18():
    """E_far 掩码在单元内的所有条件与方法间完全相同（n_far 恒定）。"""
    D, S = synth()
    with tempfile.TemporaryDirectory() as out:
        R = pd.concat([_run(D, S, 1, out), _run(D, S, 2, out)])
    assert R.n_far.nunique() == 1 and R.n_far.iloc[0] == S['efar'].sum() > 0


def T19():
    """训练缓存命中时指标相同；num_iteration=j 的概率与单独训练 j 棵树的结果一致。"""
    D, S = synth()
    with tempfile.TemporaryDirectory() as out:
        R1 = _run(D, S, 1, out)
        for f in os.listdir(out):
            if f.startswith('rows_'):
                os.remove(os.path.join(out, f))
        R2 = _run(D, S, 1, out)
    real = R2.method != 'foreign_only'
    assert R2[real].cache_hit.all()
    cols = ['oa', 'ba_ge30', 'ba_P1', 'ba_P2', 'ba_far']
    assert R1.set_index(['cond', 'method', 'k', 'clf'])[cols].sort_index().equals(
        R2.set_index(['cond', 'method', 'k', 'clf'])[cols].sort_index())
    old = G.ITERS
    try:
        G.ITERS = [3, 6]
        X, y = D.X['s2'][:2000], D.y[:2000]
        w = np.ones(len(y))
        P6 = G.train_predict(X, y, w, X[:300], 2, 7, trees=6)
        P3 = G.train_predict(X, y, w, X[:300], 2, 7, trees=3)
        assert np.allclose(P6[3], P3[3], atol=1e-6)
    finally:
        G.ITERS = old


def T21():
    """（附加）阶段 0 的分组：每组恰好 10/N_GROUPS 个外国国家；k=0 行与候选池无关、两个重复相同。"""
    D, S = synth()
    fc = D.ctry[S['foreign']]
    for g in np.unique(S['groups']):
        assert len(np.unique(fc[S['groups'] == g])) == 10 // G.N_GROUPS
    with tempfile.TemporaryDirectory() as out:
        a = _run(D, S, 1, out, rep=0)
        b = _run(D, S, 1, out, rep=1)
    fa = a[a.method == 'foreign_only'].set_index('clf')[['oa', 'ba_ge30']]
    fb = b[b.method == 'foreign_only'].set_index('clf')[['oa', 'ba_ge30']]
    assert fa.equals(fb)


def T22():
    """rev1：修正后的 ORACLE 几何不再编码折号（缺席类的列被删掉），稀有类的列被删掉，其余列 z 截断。"""
    from sklearn.model_selection import KFold
    r = np.random.default_rng(4)
    n = 3000
    y = r.choice(10, n, p=np.r_[[0.3, 0.2, 0.15, 0.1, 0.1, 0.1, 0.0493], [0.0005, 0.0001, 0.0001]])
    G = r.normal(size=(n, 10))
    fold = np.zeros(n, int)
    for f, (_, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(G)):
        fold[te] = f
        G[te, 8] = -40 - 5 * f  # 模拟缺席类：折内常数，只编码折号
        G[te, 7] *= 300           # 模拟稀有类：尺度极大
    G2 = __import__('guidance_G').oracle_geometry_v2(G, y)
    assert G2.shape[1] == int((np.bincount(y, minlength=10) >= 30).sum()) == 7
    assert np.abs(G2).max() <= 5 + 1e-6 and np.allclose(G2.std(0), 1, atol=0.05)
    share = [np.var([G2[fold == f, j].mean() for f in range(5)]) / G2[:, j].var() for j in range(G2.shape[1])]
    assert max(share) < 0.05, share


def T23():
    """rev1：--conds 过滤只跑指定条件；层级 4 只含 ORACLE2 的两个几何方法。"""
    D, S = synth()
    J4 = G.jobs_for(4, 0, 0)
    assert {j[0] for j in J4} == {'ORACLE2'} and {j[1] for j in J4} == {'lpm2_nogeo', 'typiclust'}
    assert G.jobs_for(4, 0, 1) == []
    with tempfile.TemporaryDirectory() as out:
        R = pd.DataFrame(G.run_unit(D, S, 'PT', 0, 0, 3, threads=2, out=out, trees=TREES, log=lambda s: None,
                                    conds={'RAWZ_AE', 'NATIVE_S2'}))
    assert set(R.cond) == {'RAWZ_AE', 'NATIVE_S2'} and set(R.k) == {50}


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
