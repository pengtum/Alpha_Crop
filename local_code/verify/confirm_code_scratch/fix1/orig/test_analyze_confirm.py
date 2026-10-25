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
import unittest

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_confirm as A  # noqa: E402

SYNTH = '/Users/perng_mini/AlphaCrops_local/verify/confirm_code_scratch/test_synth'
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
            np.savez(os.path.join(SYNTH, 'splits', f'{u}.npz'), test=test, pool=pool, foreign=foreign,
                     big=np.flatnonzero(np.bincount(yT, minlength=A.C) >= 30))
            np.savez(os.path.join(SYNTH, 'selection', f'{u}.npz'), **sel)

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
                                 status='ok', code_hash='synthetic',
                                 train_sha=hashlib.sha256(tr.tobytes()).hexdigest()[:16]))
            pd.DataFrame(runs)[A.RUN_COLS + ['train_sha']].to_csv(os.path.join(SYNTH, 'runs', f'{u}.csv'), index=False)
        cls.res = A.analyze(SYNTH, cls.y, 'formal', 3, log=lambda s: None)

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

    def test_bad_runs_column_rejected(self):
        p = os.path.join(SYNTH, 'runs', 'AT_F1.csv')
        orig = pd.read_csv(p, dtype={'sel_hash': str})
        try:
            orig.assign(oa=1.0).to_csv(p, index=False)
            with self.assertRaises(AssertionError):
                A.load_unit(SYNTH, 'AT_F1', self.y, 'formal', 3)
            x = orig.copy()
            x.loc[(x.arm == 'RT200') & (x.feat == 'ae') & (x.r == 0), 'sel_hash'] = 'different'
            x.to_csv(p, index=False)
            with self.assertRaises(AssertionError):   # AE 与 S2 的选样不同
                A.load_unit(SYNTH, 'AT_F1', self.y, 'formal', 3)
        finally:
            orig.to_csv(p, index=False)

    def test_sel_manifest_mismatch_rejected(self):
        # §7 第 6 步：runs 的 sel_hash / train_sha / n_train 与选样清单不一致时拒绝
        p = os.path.join(SYNTH, 'runs', 'DE_F2.csv')
        orig = pd.read_csv(p, dtype={'sel_hash': str, 'train_sha': str})
        try:
            A.load_unit(SYNTH, 'DE_F2', self.y, 'formal', 3)            # 原样通过
            m = (orig.learner == 'lgb') & (orig.arm == 'B_EXT100') & (orig.feat == 's2') & (orig.r == 1)
            for col, val in (('sel_hash', orig.loc[(orig.arm == 'B_SRS100') & (orig.r == 1), 'sel_hash'].iloc[0]),
                             ('train_sha', '0' * 16), ('n_train', 1)):
                orig.assign(**{col: orig[col].where(~m, val)}).to_csv(p, index=False)
                with self.assertRaises(AssertionError, msg=col):
                    A.load_unit(SYNTH, 'DE_F2', self.y, 'formal', 3)
        finally:
            orig.to_csv(p, index=False)

    def test_failed_fit_counted(self):
        p = os.path.join(SYNTH, 'runs', 'BE_F1.csv')
        orig = pd.read_csv(p, dtype={'sel_hash': str})
        try:
            x = orig.copy()
            x.loc[(x.learner == 'lgb') & (x.arm == 'B_BA300') & (x.r == 2), 'status'] = 'failed'
            x.to_csv(p, index=False)
            d, _ = A.load_unit(SYNTH, 'BE_F1', self.y, 'formal', 3)
            self.assertEqual(sum(v == 'failed' for v in d['status'].values()), 2)
            self.assertNotIn(('lgb', 'ae', 'B_BA300', 2), d['P'])
        finally:
            orig.to_csv(p, index=False)


if __name__ == '__main__':
    unittest.main(verbosity=2)
