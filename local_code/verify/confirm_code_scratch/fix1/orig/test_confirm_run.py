"""
confirm_split.py 与 confirm_run.py 的单元测试（合成数据；除 T13 核对真实环境的哈希外，不读真实数据）。
运行：/Users/perng_mini/AlphaCrops_local/.venv/bin/python -B test_confirm_run.py [T1 T2 ...]
"""
import os
import sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True
import copy  # noqa: E402
import hashlib  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import confirm_split as CS  # noqa: E402
import confirm_run as CR  # noqa: E402
G = CS.G
ML = CR.ML
from stageA import class_weights_from  # noqa: E402

# 合成数据的规模（运行时覆盖模块常数，不改文件）
CS.N_POOL = 2000
CS.N_FOREIGN_PER = 300
G.N_FOREIGN_PER = 300
CS.MIN_TEST = 100
TGT = 'BE'   # 阶段 A 开发国：U 含 5 个旧划分；左右两国的条带与之相邻（外国池剔除 ≤1 km 会生效）


class SynthC:
    """11 国合成数据：国家 j 占经度 [2j, 2j+2)、纬度 [50, 51.5) 的条带（相邻国家接壤）；每国复制 3 个地块的坐标与面积（去重用）。"""

    def __init__(self, n_per=6000, seed=0):
        r = np.random.default_rng(seed)
        lon, lat, y, ctry = [], [], [], []
        for j, c in enumerate(CS.COUNTRIES):
            lon.append(2.0 * j + r.uniform(0, 2.0, n_per))
            lat.append(50 + r.uniform(0, 1.5, n_per))
            y.append(r.choice(10, n_per, p=r.dirichlet(np.ones(10))))
            ctry += [c] * n_per
        self.lon, self.lat, self.y = np.concatenate(lon), np.concatenate(lat), np.concatenate(y)
        self.ctry = np.array(ctry)
        self.area = r.uniform(0.1, 5, len(self.y))
        for j in range(len(CS.COUNTRIES)):
            i = j * n_per
            for a in (self.lon, self.lat, self.area):
                a[i + 10:i + 13] = a[i:i + 3]
        self.X = {'ae': r.normal(size=(len(self.y), 16)).astype(np.float32),
                  's2': r.normal(size=(len(self.y), 20)).astype(np.float32)}
        self.countries = list(CS.COUNTRIES)


_C = {}


def env():
    """合成数据上的 BE 两个单元（无标签构建）；缓存。"""
    if 'u' not in _C:
        D = SynthC()
        C, units = CS.build_all(CS.LabelFree(D), TGT, stored=lambda kind, tgt, s: None)
        _C.update(D=D, C=C, u=units)
    return _C['D'], _C['C'], _C['u']


def raises(fn, text):
    try:
        fn()
    except (AssertionError, RuntimeError) as e:
        assert text in str(e), f'报错信息不含 "{text}"：{e}'
        return
    raise AssertionError(f'应报错（{text}）但没有')


# ---------------------------------------------------------------- 选样的纯函数
def T1():
    """BA 分配：手算小例子、平局取小编号、容量、全满报错；与逐步 Python 重算一致。"""
    # γ = (0.5, 0.3, 0.2)，n⁰ = (10, 0, 0)，容量 (100, 100, 1)。f(m) = (m+1)^−½ − (m+2)^−½：
    # 第 1 步 γf：0.5·f(10)=0.00642，0.3·f(0)=0.08787，0.2·f(0)=0.05858 → 组 1；第 2 步 0.3·f(1)=0.03893 < 0.05858 → 组 2（此后满）；
    # 第 3、4 步组 1（0.03893、0.02320 > 0.00642）；组 1 到 m=7 时 0.3·f(7)=0.00608 < 0.00642 → 第 9 步给组 0。
    seq = CS.ba_sequence([0.5, 0.3, 0.2], [10, 0, 0], [100, 100, 1], 9)
    assert seq.tolist() == [1, 2, 1, 1, 1, 1, 1, 1, 0], seq
    assert CS.ba_sequence([0.25] * 4, [0] * 4, [9] * 4, 4).tolist() == [0, 1, 2, 3]     # 平局 → 编号小的组
    assert CS.ba_sequence([0.9, 0.1], [0, 0], [1, 5], 3).tolist() == [0, 1, 1]          # 组 0 满后不再给
    raises(lambda: CS.ba_sequence([0.5, 0.5], [0, 0], [1, 0], 2), '满容量')
    r = np.random.default_rng(3)
    for _ in range(20):
        g = r.dirichlet(np.ones(8))
        n0 = r.integers(0, 30, 8)
        cap = r.integers(0, 80, 8)
        k = int(min(cap.sum(), 150))
        seq = CS.ba_sequence(g, n0, cap, k)
        n = [0] * 8
        for i in range(k):                     # 独立重算（纯 Python）
            best, bg = -1.0, -1
            for j in range(8):
                if n[j] >= cap[j]:
                    continue
                m = n0[j] + n[j]
                v = g[j] * ((m + 1) ** -0.5 - (m + 2) ** -0.5)
                if v > best:
                    best, bg = v, j
            n[bg] += 1
            assert seq[i] == bg, (i, seq[i], bg)


def T2():
    """E–S 顺序：首抽频率 ∝ 权重（蒙特卡洛）；同一种子结果相同。"""
    w = np.array([1.0, 2.0, 3.0, 4.0])
    r = np.random.default_rng(0)
    first = np.bincount([CS.es_order(r, w)[0] for _ in range(40000)], minlength=4) / 40000
    assert np.abs(first - w / w.sum()).max() < 0.01, first
    assert np.array_equal(CS.es_order(np.random.default_rng(5), w), CS.es_order(np.random.default_rng(5), w))


def T3():
    """种子：= md5(方案字符串) % 2**31，都在 [0, 2**31)；同一单元的各种子串互不相同；各 k/臂/特征共用抽样的拟合种子。"""
    D, C, units = env()
    sp, sel = units[0]
    u = sp['unit']
    strs = [f'confirm_v1_halves_{TGT}', f'confirm_v1_pool_{u}', f'confirm_v1_foreign_{u}', f'confirm_v1_stage0_{u}',
            f'confirm_v1_tgt_{u}'] + [f'confirm_v1_{p}_{u}_r{r}' for p in ('srs', 'S0', 'fit') for r in range(3)]
    seeds = [CS.seed_of(s) for s in strs]
    assert all(0 <= s < 2 ** 31 for s in seeds) and len(set(seeds)) == len(seeds)
    assert seeds[0] == int(hashlib.md5(strs[0].encode()).hexdigest(), 16) % 2 ** 31
    for r in range(3):
        ss = {CR.train_rows(a, r, sp, sel)[2] for a in CR.ARMS if a not in ('Z0', 'TGT')}
        assert ss == {f'confirm_v1_fit_{u}_r{r}'}, ss
    assert CR.train_rows('Z0', 0, sp, sel)[2] == f'confirm_v1_stage0_{u}'
    assert CR.train_rows('TGT', 0, sp, sel)[2] == f'confirm_v1_tgt_{u}'


# ---------------------------------------------------------------- 划分与选样（合成数据）
def T4():
    """划分：两半互斥且并为储备；储备与 U 不交；测试集 = 半区格内的去重地块；候选、外国池到测试 >1 km（外国剔除生效）；池 ⊂ 候选。"""
    D, C, units = env()
    CS.check_halves(len(C['cells']), C['U'], C['halves'])
    assert len(C['old_test']) > 0 and len(C['U']) > 0
    (s1, _), (s2, _) = units
    assert not np.intersect1d(s1['test'], s2['test']).size
    assert not np.intersect1d(s1['test'], C['old_test']).size
    for sp, sel in units:
        t, c = sp['test'], sp['cand']
        assert CS.min_km_indep(D, t, c) > 1 and CS.min_km_indep(D, t, sp['foreign']) > 1
        near = CS.Nearest(D.lon[t], D.lat[t])
        rest = np.setdiff1d(C['t_idx'], t)
        assert np.array_equal(c, rest[near.km(D.lon[rest], D.lat[rest]) > 1.0])
        assert np.array_equal(sp['pool'], np.sort(sp['pool'])) and np.isin(sp['pool'], c).all()
        assert len(sp['pool']) == min(CS.N_POOL, len(c))
        assert len(sp['foreign']) == 10 * CS.N_FOREIGN_PER and not (D.ctry[sp['foreign']] == TGT).any()
        print(f'    {sp["unit"]}: test {len(t)} cand {len(c)} 外国剔除 {sp["n_foreign_excl"].tolist()} '
              f'S0 区数 {[len(sel[f"S0reg_r{r}"]) for r in range(3)]}')
    assert sum(int(sp['n_foreign_excl'].sum()) for sp, _ in units) > 0, '合成数据应有 ≤1 km 的外国地块被剔除'
    dup = np.flatnonzero(D.ctry == TGT)[10:13]
    assert not np.isin(dup, C['t_idx']).any(), '重复坐标的地块应被去重'


def T5():
    """选样：π_r 是池的排列；S0 = 10 格 × 10、格合格且在所选区；S0 ∩ S1 = ∅；k1=100 是 300 的前缀；RT-k 是 π_r 前缀；
    SRS/EXT 按 π_r 顺序；BA 的样本与分配一致；check_selection 通过。"""
    D, C, units = env()
    for sp, sel in units:
        CS.check_selection(sp, sel)
        pool = sp['pool']
        for r in range(3):
            S0 = sel[f'S0_r{r}']
            for s in CS.STRATS:
                a, b = sel[f'S1_{s}_k100_r{r}'], sel[f'S1_{s}_k300_r{r}']
                assert np.array_equal(a, b[:100]) and not np.isin(b, S0).any()
            prev = set()
            for k in CS.K_A:
                cur = set(CR.train_rows(f'RT{k}', r, sp, sel)[1].tolist())
                assert prev <= cur and len(cur) == k
                prev = cur
            pr = sp['cand_reg'][np.searchsorted(sp['cand'], sel[f'S1_EXT_k300_r{r}'])]
            assert np.isin(pr, sel[f'S0reg_r{r}']).all()
            assert sel[f'BA_n_k300_r{r}'].sum() == 300 and (sel[f'BA_n_k100_r{r}'] <= sel[f'BA_n_k300_r{r}']).all()
            # SRS：按 π_r 顺序跳过 S0
            order = pool[sel[f'pi_r{r}']]
            assert np.array_equal(sel[f'S1_SRS_k300_r{r}'], order[~np.isin(order, S0)][:300])


def T6():
    """S0 追加区的规则（构造的框）：每区恰有 10 个合格格 → 2 区只有 20 < 30，必须追加到 3 区；格、每格 10 个都满足。"""
    cell_cnt = np.tile(np.r_[np.full(10, 20), [5, 5]], 8)
    cell_reg = np.repeat(np.arange(8), 12)
    cand_cell = np.repeat(np.arange(96), cell_cnt)
    sp = dict(unit='X_F1', cell_cnt=cell_cnt, cell_reg=cell_reg, cand_cell=cand_cell, cand_reg=cell_reg[cand_cell])
    for s in range(20):
        pos, regs, cells = CS.draw_S0(sp, np.random.default_rng(s))
        assert len(regs) == 3 and len(np.unique(regs)) == 3
        assert (cell_cnt[cells] >= 10).all() and np.isin(cell_reg[cells], regs).all() and len(np.unique(cells)) == 10
        cc = np.bincount(cand_cell[pos], minlength=96)
        assert (cc[cells] == 10).all() and cc.sum() == 100 and len(np.unique(pos)) == 100
    cell_cnt2 = np.tile(np.r_[np.full(15, 20)], 8)   # 每区 15 个合格格 → 2 区即 30，不追加
    cell_reg2 = np.repeat(np.arange(8), 15)
    cc2 = np.repeat(np.arange(120), cell_cnt2)
    sp2 = dict(unit='X_F1', cell_cnt=cell_cnt2, cell_reg=cell_reg2, cand_cell=cc2, cand_reg=cell_reg2[cc2])
    assert len(CS.draw_S0(sp2, np.random.default_rng(0))[1]) == 2


def T7():
    """断言 4：y、X 的代理一访问就报错；无标签构建不报错；freeze_big 在代理上报错。"""
    D, C, units = env()
    Dl = CS.LabelFree(D)
    raises(lambda: Dl.y[0], '断言 4')
    raises(lambda: np.asarray(Dl.y), '断言 4')
    raises(lambda: len(Dl.y), '断言 4')
    raises(lambda: Dl.X['ae'], '断言 4')
    raises(lambda: Dl.y.shape, '断言 4')
    raises(lambda: CS.freeze_big(Dl, units[0][0]), '断言 4')
    fb = CS.freeze_big(D, units[0][0])
    cnt = np.bincount(D.y[units[0][0]['test']], minlength=10)
    assert fb['big'].tolist() == np.flatnonzero(cnt >= 30).tolist() and fb['n_test'] == len(units[0][0]['test'])


def T8():
    """断言 1、2、3、5、6、7 在构造的数据上会触发；正常数据上 check_unit 通过。"""
    D, C, units = env()
    sp, sel = units[0]
    CS.check_unit(D, sp, sel, C['old_test'])
    # 1
    h = C['halves']
    raises(lambda: CS.check_halves(len(C['cells']), C['U'], {1: h[1], 2: np.r_[h[2], h[1][:1]]}), '断言 1')
    raises(lambda: CS.check_halves(len(C['cells']), C['U'], {1: np.r_[h[1], C['U'][:1]], 2: h[2]}), '断言 1')
    # 2
    raises(lambda: CS.check_unit(D, sp, sel, np.r_[C['old_test'], sp['test'][:1]]), '断言 2')
    # 3
    raises(lambda: CS.build_country(CS.LabelFree(D), TGT, stored=lambda k, t, s: G.make_split(D, t, s)['test'][1:]),
           '断言 3')
    CS.build_country(CS.LabelFree(D), TGT, stored=lambda k, t, s: G.make_split(D, t, s)['test'])   # 与重算相同 → 通过
    # 5：把缓冲带内（≤1 km）的一个地块放进候选集
    t = sp['test']
    rest = np.setdiff1d(C['t_idx'], np.r_[t, sp['cand']])
    near = rest[CS.Nearest(D.lon[t], D.lat[t]).km(D.lon[rest], D.lat[rest]) <= 1.0]
    assert len(near) > 0
    bad = dict(sp, cand=np.sort(np.r_[sp['cand'], near[:1]]))
    bad['pool_pos'] = np.searchsorted(bad['cand'], sp['pool'])
    raises(lambda: CS.check_unit(D, bad, sel, C['old_test']), '最小大圆距离')
    raises(lambda: CS.check_unit(D, dict(sp, cand=np.r_[sp['cand'], t[:1]]), sel, C['old_test']), '断言 5')
    s2 = copy.deepcopy(sel)
    s2['S1_SRS_k300_r1'][5] = s2['S0_r1'][0]
    s2['S1_SRS_k100_r1'][5] = s2['S0_r1'][0]
    raises(lambda: CS.check_selection(sp, s2), 'S0 ∩ S1')
    s3 = copy.deepcopy(sel)
    s3['S1_EXT_k100_r0'] = s3['S1_EXT_k100_r0'][::-1].copy()
    raises(lambda: CS.check_selection(sp, s3), '前缀')
    s4 = copy.deepcopy(sel)
    s4['pi_r2'][0] = s4['pi_r2'][1]
    raises(lambda: CS.check_selection(sp, s4), '排列')
    s5 = copy.deepcopy(sel)
    s5['BA_n_k100_r0'] = s5['BA_n_k100_r0'] + np.r_[1, -1, 0, 0, 0, 0, 0, 0]
    raises(lambda: CS.check_selection(sp, s5), 'BA')
    # 6：候选中复制一个测试地块的 AE 特征行
    D6 = copy.copy(D)
    D6.X = {k: v.copy() for k, v in D.X.items()}
    D6.X['ae'][sp['cand'][7]] = D6.X['ae'][t[3]]
    raises(lambda: CS.check_unit(D6, sp, sel, C['old_test']), '断言 6')
    assert CS.rows_isin(np.c_[D.lon[t[:3]], D.lat[t[:3]]], np.c_[D.lon[t[2:5]], D.lat[t[2:5]]]).tolist() == \
        [False, False, True]
    # 7
    raises(lambda: CS.check_same(sp, dict(sp, pool=sp['pool'][::-1]), 'x'), '断言 7')


def T9():
    """断言 7（正例）：同一段代码从头构建两次，全部数组逐位相同；存盘再读，内容哈希不变；已存在的不同内容拒绝覆盖。"""
    D, C, units = env()
    _, again = CS.build_all(CS.LabelFree(D), TGT, stored=lambda kind, tgt, s: None)
    for (a, b), (c, d) in zip(units, again):
        CS.check_same(a, c, 'sp')
        CS.check_same(b, d, 'sel')
    root = tempfile.mkdtemp()
    try:
        sp, sel = units[0]
        sp = dict(sp, **CS.freeze_big(D, sp))
        CS.write_unit(root, sp, sel)
        sp2, sel2 = CR.load_unit(root, sp['unit'])
        assert CS.content_hash(sp2) == CS.content_hash(sp) and CS.content_hash(sel2) == CS.content_hash(sel)
        CS.write_unit(root, sp, sel)                        # 内容相同：通过
        raises(lambda: CS.write_unit(root, dict(sp, lat0=1.0), sel), '已存在且内容不同')
    finally:
        shutil.rmtree(root)


def T10():
    """几何：haversine 已知值；Nearest 与暴力最近邻一致。"""
    assert abs(CS.haversine_km(0.0, 0.0, 0.0, 1.0) - 2 * np.pi * CS.R_EARTH / 360) < 1e-9
    r = np.random.default_rng(1)
    lo, la = r.uniform(5, 6, 300), r.uniform(50, 51, 300)
    qo, qa = r.uniform(5, 6, 200), r.uniform(50, 51, 200)
    d = CS.Nearest(lo, la).km(qo, qa)
    brute = np.array([CS.haversine_km(qo[i], qa[i], lo, la).min() for i in range(200)])
    assert np.abs(d - brute).max() < 1e-9


# ---------------------------------------------------------------- 运行器
def T11():
    """训练集构成与权重：Z0 = 外国；RT-k = 池 π_r 前 k；TGT = 候选池、池频率权重；B 臂 = S0 ∪ S1；目标权重 = 50·cw；
    每单元 LightGBM 70 次、RF 8 次，臂名 ⊂ 接口的 13 个。"""
    D, C, units = env()
    sp, sel = units[1]
    fr, pool = sp['foreign'], sp['pool']
    cw = class_weights_from(D.y[fr])
    fi, ti, _ = CR.train_rows('Z0', 0, sp, sel)
    assert np.array_equal(fi, fr) and len(ti) == 0
    fi, ti, _ = CR.train_rows('RT200', 1, sp, sel)
    assert np.array_equal(ti, pool[sel['pi_r1'][:200]]) and np.array_equal(fi, fr)
    fi, ti, _ = CR.train_rows('TGT', 0, sp, sel)
    assert len(fi) == 0 and np.array_equal(ti, pool)
    w = CR.sample_weights('TGT', D.y[fi], D.y[ti], cw)
    assert np.allclose(w, class_weights_from(D.y[pool])[D.y[pool]])
    fi, ti, _ = CR.train_rows('B_S0', 2, sp, sel)
    assert np.array_equal(ti, sel['S0_r2'])
    for s in CS.STRATS:
        for k1 in CS.K1:
            fi, ti, _ = CR.train_rows(f'B_{s}{k1}', 2, sp, sel)
            assert np.array_equal(ti, np.r_[sel['S0_r2'], sel[f'S1_{s}_k{k1}_r2']]) and len(ti) == 100 + k1
            w = CR.sample_weights(f'B_{s}{k1}', D.y[fi], D.y[ti], cw)
            assert np.allclose(w[:len(fi)], cw[D.y[fi]]) and np.allclose(w[len(fi):], 50 * cw[D.y[ti]])
    P = CR.fit_plan('lgb')
    assert len(P) == 70 and len(set(P)) == 70 and len(CR.fit_plan('rf')) == 8
    assert {a for a, _, _ in P} == set(CR.ARMS) and len(CR.ARMS) == 13
    assert {a for a, _, _ in CR.fit_plan('rf')} == {'Z0', 'RT200'}
    raises(lambda: CR.train_rows('B_XYZ100', 0, sp, sel), '未知的臂')


def _root_with(units, D):
    root = tempfile.mkdtemp()
    for sp, sel in units:
        CS.write_unit(root, dict(sp, **CS.freeze_big(D, sp)), sel)
    return root


def T12():
    """LightGBM（2 轮）端到端：70 个概率文件（P_test float32、P_pool float16、行和为 1）；runs 只有元数据列；
    续跑不重复拟合；删掉一个文件只补那一个；哈希不符拒绝续跑；种子相同则概率逐位相同。"""
    D, C, units = env()
    root = _root_with(units[:1], D)
    try:
        u = units[0][0]['unit']
        n = CR.run_unit(D, root, u, 'lgb', 2, 2, 'code0', log=lambda s: None)
        assert n == 70, n
        R_ = CR.read_runs(os.path.join(root, 'runs', f'{u}.csv'))
        assert list(R_.columns) == CR.RUN_COLS and (R_.status == 'ok').all() and len(R_) == 70
        assert not {'oa', 'ba', 'ba_ge30', 'fa', 'acc'} & set(c.lower() for c in R_.columns)
        sp, sel = CR.load_unit(root, u)
        for arm, f, r in CR.fit_plan('lgb'):
            z = np.load(CR.probs_path(root, u, 'lgb', arm, f, r))
            assert z['P_test'].dtype == np.float32 and z['P_test'].shape == (len(sp['test']), 10)
            assert z['P_pool'].dtype == np.float16 and z['P_pool'].shape == (len(sp['pool']), 10)
            assert np.abs(z['P_test'].sum(1) - 1).max() < 1e-4
        x = R_.set_index(['arm', 'feat', 'r'])
        assert x.loc[('RT200', 'ae', 1), 'sel_hash'] == G.sel_hash(sp['pool'][sel['pi_r1'][:200]])
        assert x.loc[('Z0', 's2', 0), 'k'] == 0 and x.loc[('TGT', 'ae', 0), 'k'] == len(sp['pool'])
        assert x.loc[('B_BA300', 's2', 2), 'k'] == 400 and x.loc[('B_S0', 'ae', 0), 'k'] == 100
        assert x.loc[('RT50', 'ae', 0), 'seed'] == x.loc[('B_EXT100', 's2', 0), 'seed'] == \
            CS.seed_of(f'confirm_v1_fit_{u}_r0')
        assert CR.run_unit(D, root, u, 'lgb', 2, 2, 'code0', log=lambda s: None) == 0
        p = CR.probs_path(root, u, 'lgb', 'RT100', 's2', 2)
        P_old = np.load(p)['P_test']
        os.remove(p)
        assert CR.run_unit(D, root, u, 'lgb', 2, 2, 'code0', log=lambda s: None) == 1
        assert np.array_equal(np.load(p)['P_test'], P_old)
        assert len(CR.read_runs(os.path.join(root, 'runs', f'{u}.csv'))) == 70
        raises(lambda: CR.run_unit(D, root, u, 'lgb', 2, 2, 'code1', log=lambda s: None), '哈希与当前不同')
    finally:
        shutil.rmtree(root)


def T13():
    """失败处理：同一种子重试一次；两次都失败记 failed、不写概率文件，续跑时不再重试；RF（3 棵）8 次拟合。"""
    D, C, units = env()
    root = _root_with(units[1:], D)
    try:
        u = units[1][0]['unit']
        calls = []

        def flaky(learner, X_tr, y_tr, w_tr, X_te, seed, threads, trees):
            calls.append(seed)
            if len(calls) == 1:
                raise MemoryError('模拟')
            if len(calls) in (4, 5):
                raise ValueError('模拟')
            return CR.fit_predict(learner, X_tr, y_tr, w_tr, X_te, seed, threads, trees)
        ML.PARAMS['rf'] = dict(CR.RF_PARAMS, n_estimators=3)
        n = CR.run_unit(D, root, u, 'rf', 2, 3, 'c', log=lambda s: None, fitter=flaky)
        assert n == 8 and calls[0] == calls[1], calls[:2]
        R_ = CR.read_runs(os.path.join(root, 'runs', f'{u}.csv')).set_index(['arm', 'feat', 'r'])
        assert R_.loc[('Z0', 'ae', 0), 'status'] == 'ok' and R_.loc[('Z0', 'ae', 0), 'attempts'] == 2
        assert R_.loc[('RT200', 'ae', 0), 'status'] == 'failed' and R_.loc[('RT200', 'ae', 0), 'attempts'] == 2
        assert 'ValueError' in R_.loc[('RT200', 'ae', 0), 'error']
        assert not os.path.exists(CR.probs_path(root, u, 'rf', 'RT200', 'ae', 0))
        assert (R_.status == 'ok').sum() == 7
        assert CR.run_unit(D, root, u, 'rf', 2, 3, 'c', log=lambda s: None) == 0
        ML.PARAMS['rf'] = dict(CR.RF_PARAMS)
        raises(lambda: CR.fit_predict('rf', np.zeros((5, 3)), np.arange(5), np.ones(5), np.zeros((2, 3)), 0, 1, 3),
               'RF 参数不符')
    finally:
        ML.PARAMS['rf'] = dict(CR.RF_PARAMS)
        shutil.rmtree(root)


def T14():
    """环境核对（真实文件）：冻结模块哈希、source stamp、库版本通过；篡改登记值会报错；dry-run 的国家数与输出目录受限。"""
    CS.check_env()
    k = next(iter(CS.FROZEN))
    old = CS.FROZEN[k]
    try:
        CS.FROZEN[k] = '0' * 64
        raises(CS.check_env, '冻结模块已变')
    finally:
        CS.FROZEN[k] = old
    argv = sys.argv
    try:
        sys.argv = ['x', '--dry-run', '--countries', 'AT,BE,DE', '--root', os.path.join(CS.SCRATCH, 'dryrun', 't')]
        raises(CS.main, '≤2 个国家')
        sys.argv = ['x', '--dry-run', '--countries', 'AT', '--root', os.path.join(CS.LOCAL, 'results', 'x')]
        raises(CS.main, 'dryrun 下')
    finally:
        sys.argv = argv


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
