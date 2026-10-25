"""
analyze_confirm.py 的单元测试与已知答案测试（方案 CONFIRM_PLAN_v5.md §7 第 2 步）。
全部用合成数据（合成标签与概率），不读取任何正式划分或新格子；端到端测试写到
/Users/perng_mini/AlphaCrops_local/verify/confirm_code_scratch/test_synth/。

用法：/Users/perng_mini/AlphaCrops_local/.venv/bin/python -B test_analyze_confirm.py
"""
import os
import sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True
import hashlib
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_confirm as A  # noqa: E402

SYNTH = '/Users/perng_mini/AlphaCrops_local/verify/confirm_code_scratch/test_synth'
GUARD = '/Users/perng_mini/AlphaCrops_local/verify/confirm_code_scratch/test_guard'
PRIOR = np.array([0.30, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05, 0.03, 0.015, 0.005])


def synth_labels(n, seed):
    return np.random.default_rng(seed).choice(A.C, size=n, p=PRIOR)


def synth_P(yT, strength, seed):
    """合成概率：正确类加 strength，另加一个固定的类偏置（制造 OA 与 BA 的交换），float32。"""
    rng = np.random.default_rng(seed)
    Z = rng.normal(0.0, 1.0, (len(yT), A.C)) + 1.5 * np.log(PRIOR)[None]
    Z[np.arange(len(yT)), yT] += strength
    P = np.exp(Z - Z.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    return P.astype(np.float32)


def strength_of(lrn, f, arm):
    s = {'Z0': 1.0, 'TGT': 2.5, 'B_S0': 1.15}.get(arm)
    if s is None and arm.startswith('RT'):
        s = 1.0 + int(arm[2:]) / 300
    if s is None:  # B_{策略}{k1}
        k1 = int(arm[-3:])
        s = 1.15 + k1 / 300 + {'SRS': 0.05, 'EXT': -0.05, 'BA': 0.1}[arm[2:-3]]
    return s + (0.3 if f == 'ae' else 0.0) + (0.1 if lrn == 'rf' else 0.0)


def synth_unit_P(yT, seed, mode='formal', R=3):
    P = {}
    for lrn, arms in A.arms_of(mode).items():
        for arm in arms:
            for f in A.FEATS:
                for r in A.draws_of(arm, R):
                    P[(lrn, f, arm, r)] = synth_P(yT, strength_of(lrn, f, arm),
                                                  A.G.stable_hash(f"{seed}_{lrn}_{f}_{arm}_{r}") % 2 ** 31)
    return P


def rows_df(unit, yT, P, R=3):
    big = np.flatnonzero(np.bincount(yT, minlength=A.C) >= 30)
    return pd.DataFrame(A.unit_rows(unit, A.unit_curves(yT, big, P), 'formal', R))


class KnownAnswer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.yT = synth_labels(3000, 1)
        cls.P = synth_unit_P(cls.yT, 7)

    def test_rt_replaced_by_z0_gives_zero_dFA(self):
        P = dict(self.P)
        for (lrn, f, arm, r) in list(P):
            if arm.startswith('RT'):
                P[(lrn, f, arm, r)] = P[(lrn, f, 'Z0', 0)].copy()
        U = rows_df('AT_F1', self.yT, P)
        d = U[(U.level == 'diff') & U.name.str.match(r'dFA_RT\d+$')]
        self.assertEqual(len(d), (4 * 3 + 3) * 2)  # LightGBM 4 档 × 3 次 + RF RT200 × 3 次，× 2 特征
        self.assertTrue((d.value == 0.0).all(), d.value.describe())
        # 对照：原合成数据下 ΔFA(RT200) 不为 0
        U0 = rows_df('AT_F1', self.yT, self.P)
        self.assertTrue((U0[(U0.level == 'diff') & (U0.name == 'dFA_RT200')].value.abs() > 0).all())

    def test_s2_replaced_by_ae_gives_zero_D(self):
        P = dict(self.P)
        for (lrn, f, arm, r) in list(P):
            if f == 's2':
                P[(lrn, f, arm, r)] = P[(lrn, 'ae', arm, r)].copy()
        U = rows_df('AT_F1', self.yT, P)
        D = U[U.level == 'D']
        self.assertEqual(len(D), 8)
        self.assertTrue((D.value == 0.0).all(), D.value.tolist())
        t0 = U[(U.level == 't0') & (U.feat == 'ae-s2')]
        self.assertTrue((t0.value == 0.0).all())

    def test_swap_ae_s2_flips_D(self):
        P = {(lrn, {'ae': 's2', 's2': 'ae'}[f], arm, r): v for (lrn, f, arm, r), v in self.P.items()}
        U0, U1 = rows_df('AT_F1', self.yT, self.P), rows_df('AT_F1', self.yT, P)
        k = ['learner', 'r', 'name']
        D0 = U0[U0.level == 'D'].set_index(k).value
        D1 = U1[U1.level == 'D'].set_index(k).value.reindex(D0.index)
        self.assertTrue((D0.abs() > 0).all())
        self.assertTrue(np.array_equal(D1.to_numpy(), -D0.to_numpy()), (D0.tolist(), D1.tolist()))
        # 同一特征内的两臂之差：交换后 ae 行等于原来的 s2 行
        a0 = U0[(U0.level == 'diff') & (U0.feat == 's2')].set_index(['learner', 'r', 'name']).value
        a1 = U1[(U1.level == 'diff') & (U1.feat == 'ae')].set_index(['learner', 'r', 'name']).value.reindex(a0.index)
        self.assertTrue(np.allclose(a0.to_numpy(), a1.to_numpy(), equal_nan=True, rtol=0, atol=0))

    def test_row_counts(self):
        U = rows_df('AT_F1', self.yT, self.P)
        self.assertEqual(U.groupby('level').size().to_dict(), {'arm': 78, 'diff': 188, 'D': 8, 't0': 14})
        self.assertEqual(A.expected_counts('formal', 3), {'arm': 78, 'diff': 188, 'D': 8, 't0': 14})
        self.assertEqual(A.expected_counts('retro', 2), {'arm': 24, 'diff': 20, 'D': 4, 't0': 10})

    def test_missing_arm_gives_nan_not_crash(self):
        P = dict(self.P)
        del P[('lgb', 'ae', 'RT200', 1)]
        U = rows_df('AT_F1', self.yT, P)
        x = U[(U.name == 'dFA_RT200') & (U.learner == 'lgb') & (U.feat == 'ae')].set_index('r').value
        self.assertTrue(np.isnan(x[1]) and np.isfinite(x[0]) and np.isfinite(x[2]))
        self.assertEqual(len(U), 288)


class Stats(unittest.TestCase):
    def test_holm(self):
        H = A.holm
        r = H({'H1': 0.01, 'H2': 0.02})
        self.assertTrue(r['H1']['reject'] and r['H2']['reject'])
        self.assertEqual((r['H1']['alpha_step'], r['H2']['alpha_step']), (0.0125, 0.025))
        r = H({'H1': 0.013, 'H2': 0.001})       # H2 先在 0.0125 下通过，H1 再在 0.025 下通过
        self.assertEqual((r['H2']['step'], r['H1']['step']), (1, 2))
        self.assertTrue(r['H1']['reject'] and r['H2']['reject'])
        r = H({'H1': 0.02, 'H2': 0.03})         # 较小者 0.02 > 0.0125：都不拒绝
        self.assertFalse(r['H1']['reject'] or r['H2']['reject'])
        r = H({'H1': 0.001, 'H2': 0.03})
        self.assertTrue(r['H1']['reject'] and not r['H2']['reject'])
        r = H({'H1': 0.0125, 'H2': 0.025})      # 边界取 ≤
        self.assertTrue(r['H1']['reject'] and r['H2']['reject'])
        r = H({'H1': np.nan, 'H2': 0.01})       # 不可判定按 p = 1
        self.assertTrue(r['H2']['reject'] and not r['H1']['reject'] and r['H2']['alpha_step'] == 0.0125)
        r = H({'H1': 0.02, 'H2': 0.0126})       # 0.0126 > 0.0125：停止，0.02 也不拒绝
        self.assertFalse(r['H1']['reject'] or r['H2']['reject'])

    def test_tsum_matches_scipy(self):
        v = np.array([2.1, 0.4, 3.3, -0.2, 1.7, 2.9, 0.8, 1.1, 2.2])
        s = A.tsum(pd.Series(v))
        tt = stats.ttest_1samp(v, 0.0, alternative='greater')
        self.assertAlmostEqual(s['t'], tt.statistic, places=12)
        self.assertAlmostEqual(s['p_one'], tt.pvalue, places=12)
        ci = stats.t.interval(0.975, len(v) - 1, loc=v.mean(), scale=stats.sem(v))
        self.assertAlmostEqual(s['lo975'], ci[0], places=12)
        self.assertAlmostEqual(s['hi975'], ci[1], places=12)
        self.assertEqual(s['n_pos'], 8)
        self.assertEqual(A.tsum(pd.Series([np.nan, 1.0]))['n'], 1)

    def test_pav(self):
        self.assertTrue(np.allclose(A.pav([1, 3, 2, 4]), [1, 2.5, 2.5, 4]))
        self.assertTrue(np.allclose(A.pav([3, 2, 1]), [2, 2, 2]))
        self.assertTrue(np.allclose(A.pav([0, 1, 2]), [0, 1, 2]))
        self.assertTrue(np.allclose(A.pav([0, 2, 1, 0.5, 3]), [0, 7 / 6, 7 / 6, 7 / 6, 3]))

    def test_interp_and_k_eq(self):
        kn = A.KN_R
        fv = A.pav([0.0, 1.0, 1.0, 2.0, 3.0])
        self.assertEqual(A.k_eq(1.0, kn, fv), 50.0)       # 平段取最小的 k
        self.assertEqual(A.k_eq(1.5, kn, fv), 150.0)      # 100 与 200 之间线性插值
        self.assertEqual(A.k_eq(2.5, kn, fv), 300.0)
        self.assertEqual(A.k_eq(3.0, kn, fv), 400.0)
        self.assertEqual(A.k_eq(3.01, kn, fv), np.inf)    # "> 400"
        self.assertEqual(A.k_eq(-0.1, kn, fv), -np.inf)   # "< 0"
        self.assertEqual((A.fmt_k(np.inf), A.fmt_k(-np.inf)), ('> 400', '< 0'))
        self.assertEqual(A.k_eq(0.0, kn, fv), 0.0)
        self.assertEqual(A.interp_noextrap(150, kn, fv), 1.5)
        self.assertTrue(np.isnan(A.interp_noextrap(401, kn, fv)))
        # PAV 后的例子：原始 [0, 2, 1, 3, 4] → [0, 1.5, 1.5, 3, 4]
        fv = A.pav([0, 2, 1, 3, 4])
        self.assertEqual(A.k_eq(1.5, kn, fv), 50.0)
        self.assertEqual(A.k_eq(2.25, kn, fv), 150.0)

    def test_x3_point(self):
        m = {'dFA_RT50': 1.0, 'dFA_RT100': 2.0, 'dFA_RT200': 3.0, 'dFA_RT400': 4.0,
             'dFA_B_S0': 1.0, 'dFA_B_EXT100': 1.5, 'dFA_B_EXT300': 2.0,
             'dFA_B_SRS100': 3.0, 'dFA_B_SRS300': 5.0, 'dFA_B_BA100': 2.5, 'dFA_B_BA300': 4.0}
        out, fr, fc = A.x3_point(m)
        self.assertEqual(out[('keq', 'B_S0')], 50.0)
        self.assertEqual(out[('rate', 'B_S0')], 0.5)
        self.assertEqual(out[('keq', 'B_SRS100')], 200.0)
        self.assertEqual(out[('keq', 'B_SRS300')], np.inf)
        self.assertEqual(out[('keq', 'B_BA100')], 150.0)
        self.assertEqual(out[('conc_diff', 200)], 1.5 - 3.0)
        self.assertEqual(out[('conc_keq', 400)], 100.0)
        self.assertEqual(out[('conc_rate', 400)], 0.25)

    def test_q_interval(self):
        x = np.arange(2000, dtype=float)
        self.assertEqual(A.q_interval(x), (99.0, 1899.0))
        x[-300:] = np.inf
        self.assertEqual(A.q_interval(x)[1], np.inf)

    def test_b1_reading(self):
        self.assertEqual(A.b1_reading(-0.5, 0.8, 0.1), '补采位置的影响小于 1 pp')
        self.assertEqual(A.b1_reading(0.2, 0.8, 0.5), '补采位置的影响小于 1 pp')   # 按顺序先判 (−1, +1)
        self.assertTrue(A.b1_reading(0.5, 2.0, 1.2).startswith('方向为正'))
        self.assertTrue(A.b1_reading(-3.0, -1.2, -2.0).startswith('方向为负'))
        self.assertEqual(A.b1_reading(-1.5, 2.0, 0.2), '未能区分，区间为 [-1.50, 2.00]')

    def test_country_table_hierarchy(self):
        # 国家值 = 对 r 平均 → 对 F1、F2 平均（跳过缺失）；both 任一缺失则缺失
        rec = []
        vals = {('AT', 1, 'ae'): [1.0, 2.0, 3.0], ('AT', 2, 'ae'): [10.0, np.nan, np.nan],
                ('AT', 1, 's2'): [0.0, 0.0, 0.0], ('AT', 2, 's2'): [4.0, 4.0, 4.0],
                ('BE', 1, 'ae'): [np.nan] * 3, ('BE', 2, 'ae'): [np.nan] * 3,
                ('BE', 1, 's2'): [1.0, 1.0, 1.0], ('BE', 2, 's2'): [3.0, 3.0, 3.0]}
        for (c, h, f), v in vals.items():
            for r, x in enumerate(v):
                rec.append(dict(level='diff', learner='lgb', country=c, half=h, unit=f'{c}_F{h}', r=r, feat=f,
                                name='q', value=x))
        CT = A.country_table(pd.DataFrame(rec))
        cv = A.cv_lookup(CT)
        self.assertEqual(cv[('lgb', 'q', 'ae')]['AT'], (2.0 + 10.0) / 2)
        self.assertEqual(cv[('lgb', 'q', 's2')]['AT'], 2.0)
        self.assertEqual(cv[('lgb', 'q', 'both')]['AT'], (6.0 + 2.0) / 2)
        self.assertEqual(cv[('lgb', 'q', 'inter')]['AT'], 6.0 - 2.0)
        self.assertTrue(np.isnan(cv[('lgb', 'q', 'ae')]['BE']) and np.isnan(cv[('lgb', 'q', 'both')]['BE']))
        self.assertEqual(cv[('lgb', 'q', 's2')]['BE'], 2.0)


class EndToEnd(unittest.TestCase):
    """合成的 22 个单元写成统一接口 → analyze（正式模式）→ 结构检查通过；另测剔除与接口校验。"""

    @classmethod
    def setUpClass(cls):
        shutil.rmtree(SYNTH, ignore_errors=True)
        units = A.expected_units()
        n_per = {u: 3000 for u in units}
        n_per['SI_F2'] = 2500                      # 不可行单元（n_test < 3000），应被剔除
        cls.y = np.concatenate([synth_labels(n_per[u], i) for i, u in enumerate(units)])
        start = np.cumsum([0] + [n_per[u] for u in units])
        for sub in ('splits', 'selection', 'runs', 'probs'):
            os.makedirs(os.path.join(SYNTH, sub))
        empty = np.zeros((0, A.C), np.float16)
        foreign = 2 * 10 ** 7 + np.arange(50)
        code, man = A.runner_code_hash(), []
        for i, u in enumerate(units):
            test = np.arange(start[i], start[i + 1])
            yT = cls.y[test]
            # 合成的候选池与选样清单（下标只用于哈希，不对应标签）；目标训练下标按方案 §2 在此处独立写出
            rng = np.random.default_rng(i)
            pool = 10 ** 7 + 1000 * i + np.arange(600)
            sel = {}
            for r in range(3):
                sel[f'pi_r{r}'] = rng.permutation(600)
                sel[f'S0_r{r}'] = 3 * 10 ** 7 + 1000 * i + 100 * r + np.arange(100)
                for s in A.STRATS:
                    s1 = pool[rng.permutation(600)[:300]]
                    sel[f'S1_{s}_k300_r{r}'], sel[f'S1_{s}_k100_r{r}'] = s1, s1[:100]
            big = np.flatnonzero(np.bincount(yT, minlength=A.C) >= 30)
            np.savez(os.path.join(SYNTH, 'splits', f'{u}.npz'), test=test, pool=pool, foreign=foreign, big=big,
                     feasible=bool(len(test) >= 3000 and len(big) >= 2))
            np.savez(os.path.join(SYNTH, 'selection', f'{u}.npz'), **sel)
            hs = A.content_hash(A.npz(os.path.join(SYNTH, 'splits', f'{u}.npz')))
            hl = A.content_hash(A.npz(os.path.join(SYNTH, 'selection', f'{u}.npz')))
            man.append(dict(unit=u, splits_hash=hs, selection_hash=hl))

            def targets(arm, r):
                if arm == 'Z0':
                    return np.zeros(0, np.int64)
                if arm == 'TGT':
                    return pool
                if arm.startswith('RT'):
                    return pool[sel[f'pi_r{r}'][:int(arm[2:])]]
                if arm == 'B_S0':
                    return sel[f'S0_r{r}']
                return np.r_[sel[f'S0_r{r}'], sel[f'S1_{arm[2:-3]}_k{arm[-3:]}_r{r}']]
            os.makedirs(os.path.join(SYNTH, 'probs', u))
            runs = []
            for (lrn, f, arm, r), Pt in synth_unit_P(yT, i).items():
                np.savez(os.path.join(SYNTH, 'probs', u, f'{lrn}_{arm}_{f}_r{r}.npz'), P_test=Pt, P_pool=empty)
                ti = np.asarray(targets(arm, r), np.int64)
                tr = np.r_[foreign if arm != 'TGT' else [], ti].astype(np.int64)
                runs.append(dict(learner=lrn, arm=arm, feat=f, r=r, k=len(ti),
                                 sel_hash=A.G.sel_hash(ti) if len(ti) else 'k0', seed=1, n_train=len(tr), t_fit=0.0,
                                 status='ok', code_hash=code, threads=2, splits_hash=hs, selection_hash=hl,
                                 train_sha=hashlib.sha256(tr.tobytes()).hexdigest()[:16]))
            pd.DataFrame(runs)[A.RUN_COLS + A.FORMAL_COLS].to_csv(os.path.join(SYNTH, 'runs', f'{u}.csv'), index=False)
        pd.DataFrame(man).to_csv(os.path.join(SYNTH, 'split_manifest.csv'), index=False)
        cls.res = A.analyze(SYNTH, cls.y, 'formal', 3, log=lambda s: None)

    def check(self, unit, strict=True):
        return A.check_unit(SYNTH, unit, self.y, 'formal', 3, strict=strict)

    def edit_runs(self, unit, fn):
        """改一个单元的 runs（fn(DataFrame) → DataFrame），返回恢复函数。"""
        p = os.path.join(SYNTH, 'runs', f'{unit}.csv')
        orig = pd.read_csv(p, dtype={c: str for c in A.HASH_COLS})
        fn(orig.copy()).to_csv(p, index=False)
        return lambda: orig.to_csv(p, index=False)

    def test_structure(self):
        lines, bad = A.structure_check(self.res, 'formal', 3)
        self.assertEqual(bad, 0, '\n'.join(lines))
        self.assertEqual(list(self.res['excluded']), ['SI_F2'])
        self.assertEqual(self.res['unit_rows'].unit.nunique(), 21)
        F = self.res['fill_table']
        self.assertEqual(set(zip(F.sentence, F.variant, F.placeholder)), set(A.FILL_KEYS))
        self.assertEqual(len(self.res['tests']), 2)
        self.assertEqual(sorted(self.res['x2_position'].config), sorted(A.B_ARMS))

    def test_summary_renders(self):
        lines = A.summary_lines(self.res, 'formal', 3)
        self.assertTrue(any('Holm' in s for s in lines) and any('⟦ρ⟧' in s for s in lines))
        # 第 6 条：读法不是"可区分"的集中样本交换率不打印点估计
        X3 = self.res['x3_equiv']
        for kind in ('conc_keq', 'conc_rate'):
            for _, r in X3[X3.kind == kind].iterrows():
                ln = [s for s in lines if s.startswith(f'  {kind:10s} {str(r["item"]):9s}')]
                self.assertEqual(len(ln), 1)
                self.assertEqual('不报' in ln[0], r.reading != A.X3_DISTINCT, ln[0])

    def test_bad_runs_column_rejected(self):
        restore = self.edit_runs('AT_F1', lambda x: x.assign(oa=1.0))
        try:
            with self.assertRaises(AssertionError):
                self.check('AT_F1')
        finally:
            restore()

        def other_sel(x):
            x.loc[(x.arm == 'RT200') & (x.feat == 'ae') & (x.r == 0), 'sel_hash'] = 'different'
            return x
        restore = self.edit_runs('AT_F1', other_sel)
        try:
            with self.assertRaises(AssertionError):   # AE 与 S2 的选样不同
                self.check('AT_F1')
        finally:
            restore()

    def test_sel_manifest_mismatch_rejected(self):
        # §7 第 6 步：runs 的 sel_hash / train_sha / n_train 与选样清单不一致时拒绝
        self.check('DE_F2')                                              # 原样通过
        p = os.path.join(SYNTH, 'runs', 'DE_F2.csv')
        orig = pd.read_csv(p, dtype={c: str for c in A.HASH_COLS})
        m = (orig.learner == 'lgb') & (orig.arm == 'B_EXT100') & (orig.feat == 's2') & (orig.r == 1)
        try:
            for col, val in (('sel_hash', orig.loc[(orig.arm == 'B_SRS100') & (orig.r == 1), 'sel_hash'].iloc[0]),
                             ('train_sha', '0' * 16), ('n_train', 1)):
                orig.assign(**{col: orig[col].where(~m, val)}).to_csv(p, index=False)
                with self.assertRaises(AssertionError, msg=col):
                    self.check('DE_F2')
        finally:
            orig.to_csv(p, index=False)

    def test_failed_fit_counted(self):
        def fail(x):
            x.loc[(x.learner == 'lgb') & (x.arm == 'B_BA300') & (x.r == 2), 'status'] = 'failed'
            return x
        restore = self.edit_runs('BE_F1', fail)
        try:
            d, _ = self.check('BE_F1')
            self.assertEqual(sum(v == 'failed' for v in d['status'].values()), 2)
            self.assertNotIn(('lgb', 'ae', 'B_BA300', 2), A.load_probs(SYNTH, d))
        finally:
            restore()

    def test_missing_rf_rows_rejected(self):
        # 审查第 1 条：删去 RF 行 → 正式分析拒绝运行；结构检查计为问题（不再报"通过"）
        restore = self.edit_runs('LT_F2', lambda x: x[x.learner != 'rf'])
        try:
            with self.assertRaises(AssertionError) as cm:
                self.check('LT_F2')
            self.assertIn('缺 8 次计划拟合', str(cm.exception))
            with self.assertRaises(AssertionError):
                A.analyze(SYNTH, self.y, 'formal', 3, log=lambda s: None, units=['LT_F2'])
            res = A.analyze(SYNTH, self.y, 'formal', 3, log=lambda s: None, units=['LT_F2'], strict=False)
            lines, bad = A.structure_check(res, 'formal', 3)
            self.assertGreaterEqual(bad, 8, '\n'.join(lines))
            self.assertIn('结构检查：', lines[-1])
            self.assertNotIn('通过', lines[-1])
        finally:
            restore()

    def test_plan_keys_and_status(self):
        # 方案外的键、status 非 ok/failed：即使结构检查（strict=False）也拒绝
        def extra(x):
            y = x[(x.learner == 'rf') & (x.arm == 'Z0')].copy()
            y['arm'] = 'TGT'
            return pd.concat([x, y])
        for fn in (extra, lambda x: x.assign(status=x.status.where(x.arm != 'RT50', 'running'))):
            restore = self.edit_runs('EE_F1', fn)
            try:
                with self.assertRaises(AssertionError):
                    self.check('EE_F1', strict=False)
            finally:
                restore()

    def test_hashes_threads_code_tied(self):
        # 审查第 3、5 条：划分 / 选样内容哈希、清单、runs、线程、code_hash、可行性彼此挂钩
        u = 'SK_F1'
        self.check(u)
        for fn in (lambda x: x.assign(splits_hash='0' * 64), lambda x: x.assign(selection_hash='0' * 64),
                   lambda x: x.assign(threads=1)):
            restore = self.edit_runs(u, fn)
            try:
                with self.assertRaises(AssertionError):
                    self.check(u)
            finally:
                restore()
        restore = self.edit_runs(u, lambda x: x.assign(code_hash='0123456789abcdef'))
        try:
            self.check(u)                                               # 单元内一致，但与当前运行器代码不同
            with self.assertRaises(AssertionError):
                A.analyze(SYNTH, self.y, 'formal', 3, log=lambda s: None, units=[u])
        finally:
            restore()
        mp = os.path.join(SYNTH, 'split_manifest.csv')
        man = pd.read_csv(mp, dtype=str)
        try:
            man.assign(splits_hash=man.splits_hash.where(man.unit != u, '0' * 64)).to_csv(mp, index=False)
            with self.assertRaises(AssertionError):
                self.check(u)
        finally:
            man.to_csv(mp, index=False)
        sp = os.path.join(SYNTH, 'splits', f'{u}.npz')
        S = A.npz(sp)
        try:
            np.savez(sp, **dict(S, feasible=False))                     # 可行性与重算不一致（清单同步改，单测这一条）
            man.assign(splits_hash=man.splits_hash.where(man.unit != u, A.content_hash(A.npz(sp)))).to_csv(mp, index=False)
            with self.assertRaises(AssertionError) as cm:
                self.check(u)
            self.assertIn('可行性', str(cm.exception))
        finally:
            np.savez(sp, **S)
            man.to_csv(mp, index=False)
        self.check(u)

    def test_missing_split_or_manifest_unit_rejected(self):
        sp = os.path.join(SYNTH, 'splits', 'LV_F1.npz')
        os.rename(sp, sp + '.bak')
        try:
            with self.assertRaises(AssertionError):
                self.check('LV_F1')
        finally:
            os.rename(sp + '.bak', sp)
        mp = os.path.join(SYNTH, 'split_manifest.csv')
        man = pd.read_csv(mp, dtype=str)
        try:
            man[man.unit != 'RO_F2'].to_csv(mp, index=False)             # 清单少一个单元：全量正式分析拒绝
            with self.assertRaises(AssertionError):
                A.analyze(SYNTH, self.y, 'formal', 3, log=lambda s: None)
        finally:
            man.to_csv(mp, index=False)


class Hashing(unittest.TestCase):
    def test_same_algorithm_as_runner(self):
        import confirm_split as CS
        rng = np.random.default_rng(0)
        d = dict(a=rng.integers(0, 9, 20), b=rng.random((3, 4)).astype(np.float32), unit='AT_F1', lat0=50.1,
                 feasible=True, e=np.zeros(0, np.int64))
        self.assertEqual(A.content_hash(d), CS.content_hash(d))
        self.assertEqual(A.runner_code_hash(), CS.code_hash())
        with tempfile.TemporaryDirectory() as t:
            np.savez(os.path.join(t, 'x.npz'), **d)
            self.assertEqual(A.content_hash(A.npz(os.path.join(t, 'x.npz'))), CS.content_hash(d))


class DeleteGuards(unittest.TestCase):
    """审查第 2 条：--retro / --overwrite 只在 scratch 之下、只删带标记的目录（只用临时目录测试，不触碰真实结果）。"""

    def test_assert_scratch(self):
        for bad in ('/Users/perng_mini/AlphaCrops_local/results/guidance_G', A.FORMAL_ROOT, A.SCRATCH,
                    A.SCRATCH + '_x/retro', '/Users/perng_mini/AlphaCrops_local/results/stageA'):
            with self.assertRaises(AssertionError, msg=bad):
                A.assert_scratch(bad)
        A.assert_scratch(os.path.join(A.SCRATCH, 'retro_x'))

    def test_build_retro_refuses(self):
        outside = tempfile.mkdtemp()
        shutil.rmtree(GUARD, ignore_errors=True)
        try:
            for root in (outside, GUARD):                             # scratch 外；scratch 内但无标记
                os.makedirs(os.path.join(root, 'probs'), exist_ok=True)
                f = os.path.join(root, 'probs', 'keep.txt')
                open(f, 'w').write('x')
                with self.assertRaises(AssertionError):
                    A.build_retro(root, None)
                self.assertTrue(os.path.exists(f))
        finally:
            shutil.rmtree(outside)
            shutil.rmtree(GUARD, ignore_errors=True)

    def test_clear_own(self):
        shutil.rmtree(GUARD, ignore_errors=True)
        out = os.path.join(GUARD, 'analysis')
        os.makedirs(out)
        open(os.path.join(out, 'x.csv'), 'w').write('x')
        try:
            with self.assertRaises(AssertionError):
                A.clear_own(out, A.ANALYSIS_MARK)
            self.assertTrue(os.path.exists(os.path.join(out, 'x.csv')))
            open(os.path.join(out, A.ANALYSIS_MARK), 'w').write('')
            A.clear_own(out, A.ANALYSIS_MARK)
            self.assertFalse(os.path.exists(out))
            A.clear_own(out, A.ANALYSIS_MARK)                          # 不存在：直接返回
        finally:
            shutil.rmtree(GUARD, ignore_errors=True)


class X3Reading(unittest.TestCase):
    """审查第 6、14 条：不可区分或不可判定时不报交换率；交换率超界写 "> 400/n" / "< 0"；X2、X3 不足 7 国不可判定。"""

    @staticmethod
    def cvt(conc, n_have=9, seed=0):
        """9 国国家值：各量为给定值（标量则加 SD 0.05 的噪声；数组为逐国值），第 n_have 国之后记缺失。"""
        rng = np.random.default_rng(seed)
        base = {'dFA_RT50': 1.0, 'dFA_RT100': 2.0, 'dFA_RT200': 3.0, 'dFA_RT400': 4.0, 'dFA_B_SRS100': 3.0,
                'dFA_B_SRS300': 4.2, 'dFA_B_BA100': 2.9, 'dFA_B_BA300': 4.5}
        out = {}
        for n, v in {**base, **conc}.items():
            x = np.array(v, dtype=float) + (rng.normal(0, 0.05, len(A.NINE)) if np.ndim(v) == 0 else 0.0)
            x[n_have:] = np.nan
            out[('lgb', n, 'both')] = pd.Series(x, index=list(A.NINE))
        return out

    def rows(self, X3, kind, n):
        return X3[(X3.kind == kind) & (X3.item == n)].iloc[0]

    def test_distinct_reports_rate(self):
        _, X3, _ = A.x_blocks(self.cvt({'dFA_B_S0': 0.5, 'dFA_B_EXT100': 1.0, 'dFA_B_EXT300': 1.5}))
        d, r = self.rows(X3, 'conc_diff', 100), self.rows(X3, 'conc_rate', 100)
        self.assertEqual(d.reading, A.X3_DISTINCT)
        self.assertTrue(np.isfinite(r.point) and r.text == f'{r.point:.2f}' and abs(r.point - 0.25) < 0.05)
        self.assertTrue(bool(X3.decidable.iloc[0]))

    def test_indistinct_hides_rate(self):
        rng = np.random.default_rng(1)
        noisy = {'dFA_B_S0': 2.0 + rng.normal(0, 5, 9), 'dFA_B_EXT100': 3.0 + rng.normal(0, 5, 9),
                 'dFA_B_EXT300': 4.0 + rng.normal(0, 5, 9)}
        _, X3, _ = A.x_blocks(self.cvt(noisy))
        for n in (100, 200, 400):
            self.assertTrue(self.rows(X3, 'conc_diff', n).reading.startswith('集中与分散样本的价值未能区分'))
            for kind in ('conc_keq', 'conc_rate'):
                r = self.rows(X3, kind, n)
                self.assertEqual(r.text, '不报')
                self.assertTrue(np.isnan(r.point) and np.isnan(r.lo90) and np.isnan(r.hi90))

    def test_undecidable_under_7(self):
        x2, X3, _ = A.x_blocks(self.cvt({'dFA_B_S0': 0.5, 'dFA_B_EXT100': 1.0, 'dFA_B_EXT300': 1.5}, n_have=6))
        self.assertEqual(int(X3.n_countries.iloc[0]), 6)
        self.assertFalse(bool(X3.decidable.iloc[0]) or bool(x2.decidable.iloc[0]))
        self.assertTrue(self.rows(X3, 'conc_diff', 100).reading.startswith('不可判定'))
        self.assertEqual(self.rows(X3, 'conc_rate', 100).text, '不报')

    def test_fmt_rate(self):
        self.assertEqual((A.fmt_rate(np.inf, 200), A.fmt_rate(-np.inf, 200), A.fmt_rate(0.5, 200)),
                         ('> 2.00', '< 0', '0.50'))
        self.assertEqual(A.x3_fmt('rate', 'B_SRS300', np.inf), '> 1.00')


if __name__ == '__main__':
    unittest.main(verbosity=2)
