"""
G 实验对称性单元测试（原型）。运行：
  OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 python test_guidance_symmetry.py
"""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import guidance_proto as G  # noqa: E402
import samplers.base as B  # noqa: E402
import samplers.designs, samplers.uncertainty, samplers.coverage  # noqa: E402,F401

ALL = ['srs'] + G.FAMILY['geom'] + G.FAMILY['pred'] + G.FAMILY['both']
KS, SEEDS = (50, 200), (3, 11)


def toy_stage0(N=3000, nF=4000, seed=0):
    t = B.make_toy_ctx(N=N, n_F=nF, seed=seed)
    r = np.random.default_rng(seed + 1)
    P2 = r.dirichlet(np.ones(10) * 0.3, size=N)
    S = dict(T_ae=t.T_ae, F_ae=t.F_ae, T_s2=t.T_s2, F_s2=t.F_s2, P_T_ae=t.P_T, P_T_s2=P2)
    return S, t.F_y, t.T_xy


def run(S, cond, F_y, xy, name, k, seed, pool_pos=None):
    pool_pos = np.arange(len(S['T_ae'])) if pool_pos is None else pool_pos
    ctx = G.build_ctx_guided(S, pool_pos, cond, F_y, xy_pool=xy)
    return B.REGISTRY[name](ctx, k, seed, **G.G_HP.get(name, {}))


class T(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.S, cls.Fy, cls.xy = toy_stage0()

    def test_1_unused_fields_none(self):
        """与指导无关的字段置 None 后全部方法仍能运行 → 没有隐藏读取 T_s2/T_xy/T_nuts3/P_T_loco/dom 等。"""
        for name in ALL:
            for cond in ('AE', 'S2', 'AEg_S2p', 'S2g_AEp'):
                run(self.S, cond, self.Fy, self.xy, name, 50, 3)

    def test_2_identical_guidance_bitwise(self):
        """把 S2 的指导数据设成与 AE 相同 → 各条件的选样逐位一致。"""
        S = dict(self.S)
        S['T_s2'], S['F_s2'], S['P_T_s2'] = S['T_ae'].copy(), S['F_ae'].copy(), S['P_T_ae'].copy()
        for name in ALL:
            for k in KS:
                for seed in SEEDS:
                    ref = run(S, 'AE', self.Fy, self.xy, name, k, seed)
                    for cond in ('S2', 'AEg_S2p', 'S2g_AEp'):
                        np.testing.assert_array_equal(ref, run(S, cond, self.Fy, self.xy, name, k, seed),
                                                      err_msg=f'{name} k={k} seed={seed} {cond}')

    def test_3_label_swap(self):
        """交换 AE/S2 两套数据后，'AE' 条件的结果 == 原数据 'S2' 条件的结果（排除按名字硬编码）。"""
        Sw = dict(T_ae=self.S['T_s2'], F_ae=self.S['F_s2'], T_s2=self.S['T_ae'], F_s2=self.S['F_ae'],
                  P_T_ae=self.S['P_T_s2'], P_T_s2=self.S['P_T_ae'])
        for name in ALL:
            for k in KS:
                np.testing.assert_array_equal(run(Sw, 'AE', self.Fy, self.xy, name, k, 3),
                                              run(self.S, 'S2', self.Fy, self.xy, name, k, 3), err_msg=name)

    def test_4_geom_family_ignores_P(self):
        """纯几何组对 P_T 的替换不变（gist_maxmin 已移出该组）。"""
        for name in G.FAMILY['geom']:
            for k in KS:
                np.testing.assert_array_equal(run(self.S, 'AE', self.Fy, self.xy, name, k, 3),
                                              run(self.S, 'AEg_S2p', self.Fy, self.xy, name, k, 3), err_msg=name)

    def test_4b_gist_maxmin_reads_P(self):
        """记录：gist_maxmin 在 α=0 时仍依赖 P_T（所以不能当纯几何方法）。"""
        a = run(self.S, 'AE', self.Fy, self.xy, 'gist_maxmin', 50, 3)
        b = run(self.S, 'AEg_S2p', self.Fy, self.xy, 'gist_maxmin', 50, 3)
        self.assertFalse(np.array_equal(a, b))

    def test_5_pred_family_ignores_geometry(self):
        for name in G.FAMILY['pred']:
            for k in KS:
                np.testing.assert_array_equal(run(self.S, 'AE', self.Fy, self.xy, name, k, 3),
                                              run(self.S, 'S2g_AEp', self.Fy, self.xy, name, k, 3), err_msg=name)

    def test_6_geo_w_zero(self):
        self.assertEqual(G.check_geo_w(ALL), [])

    def test_7_xy_offset_invariance(self):
        """XY 参照：坐标平移 5000 km 不改变结果（build_ctx_guided 内部已居中）。"""
        for name in G.FAMILY['geom']:
            a = run(self.S, 'XY', self.Fy, self.xy, name, 50, 3)
            b = run(self.S, 'XY', self.Fy, self.xy + 5000.0, name, 50, 3)
            np.testing.assert_array_equal(a, b, err_msg=name)

    def test_8_stage0_symmetry(self):
        """阶段 0：若 X_ae 与 X_s2 完全相同，则 P_T_ae、P_T_s2、温度与袋外概率逐位一致；分组只依赖 (tgt, split)。"""
        CODE = G.CODE
        sys.path.insert(0, CODE)
        os.environ.setdefault('STAGEA_OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '_stageA_out'))
        import stageA
        r = np.random.default_rng(0)
        cs = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']
        ctry = np.repeat(cs, 300)
        y = r.integers(0, 10, len(ctry))
        X = r.normal(size=(len(ctry), 20)).astype(np.float32) + y[:, None] * 0.3
        tgt = 'SI'
        foreign = np.flatnonzero(ctry != tgt)
        allT = np.flatnonzero(ctry == tgt)
        others = [c for c in cs if c != tgt]
        g1 = G.fold_groups(others, tgt, 0)
        g2 = G.fold_groups(others, tgt, 0)
        self.assertEqual(g1, g2)
        outs = [G.guidance_models(Xf, y, ctry, foreign, allT, g1, stageA.fit_predict, stageA.class_weights_from,
                                  threads=1, trees=5) for Xf in (X, X.copy())]
        np.testing.assert_array_equal(outs[0][0], outs[1][0])
        self.assertEqual(outs[0][1], outs[1][1])
        np.testing.assert_array_equal(outs[0][2], outs[1][2])


if __name__ == '__main__':
    unittest.main(verbosity=2)
