"""indep_confirm.py 的已知答案测试（开发用，scratch）。"""
import os
import sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True
import numpy as np

sys.path.insert(0, '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion/confirm')
import indep_confirm as I  # noqa: E402

# 1) 包络手算
e = I.envelope(np.array([10., 20., 30.]), np.array([50., 40., 60.]), np.array([15., 25., 30., 35., 20., 9.9999999999]))
assert np.allclose(e[:5], [45, 50, 60, np.nan, 40], equal_nan=True) and e[5] == 50, e
e = I.envelope(np.array([10., 30., 20.]), np.array([50., 70., 90.]), np.array([25.]))
assert e[0] == 80, e
e = I.envelope(np.array([10., 10., 20.]), np.array([30., 50., 40.]), np.array([10., 15.]))
assert e[0] == 50 and e[1] == 45, e
print('1 包络手算：通过')

# 2) 与 step0B.envelope 在随机曲线上对照（只在测试中导入，不进 indep_confirm.py）
sys.path.insert(0, '/Users/perng_mini/AlphaCrops_local/verify/step0/partB')
cwd = os.getcwd()
import step0B as B  # noqa: E402
os.chdir(cwd)
rng = np.random.default_rng(0)
worst, nanbad = 0.0, 0
for it in range(3000):
    n = 81
    oa = np.round(rng.uniform(60, 80, n), rng.integers(0, 3))          # 取整制造平段与重复点
    ba = rng.uniform(30, 60, n)
    if it % 3 == 0:
        oa = np.sort(oa)
    grid = oa[rng.integers(0, n)] + I.OFFS if it % 2 else rng.uniform(55, 85) + I.OFFS
    a = I.envelope(oa, ba, grid)
    b = B.envelope([(oa, ba)], grid)[0]
    nanbad += int((np.isnan(a) != np.isnan(b)).sum())
    f = np.isfinite(a) & np.isfinite(b)
    if f.any():
        worst = max(worst, float(np.abs(a[f] - b[f]).max()))
assert nanbad == 0 and worst < 1e-9, (nanbad, worst)
print(f'2 随机曲线 3000 条与 step0B.envelope 对照：NaN 不一致 0，最大 |Δ| {worst:.2e}，通过')

# 3) Holm 手算
H = I.holm
assert H({'H1': 0.01, 'H2': 0.02}) == {'H1': (0.0125, True), 'H2': (0.025, True)}
assert H({'H1': 0.013, 'H2': 0.02}) == {'H1': (0.0125, False), 'H2': (0.025, False)}
assert H({'H1': 0.001, 'H2': 0.03}) == {'H1': (0.0125, True), 'H2': (0.025, False)}
assert H({'H1': 0.03, 'H2': 0.001}) == {'H2': (0.0125, True), 'H1': (0.025, False)}
assert H({'H1': np.nan, 'H2': 0.01}) == {'H2': (0.0125, True), 'H1': (0.025, False)}
assert H({'H1': 0.0125, 'H2': 0.025}) == {'H1': (0.0125, True), 'H2': (0.025, True)}
print('3 Holm 手算：通过')

# 4) 在一个真实回溯单元（AT_s0）上：RT 换成 Z0 → ΔFA = 0；S2 换成 AE → D = 0；交换 AE 与 S2 → D 变号
sys.path.insert(0, I.L)
from guidance_G import GData  # noqa: E402
y = np.asarray(GData().y)
meta, yT, big, load = next(I.retro_units(y, lambda s: None))
reps = (0, 1)


def val(rows, name, feat=None):
    return np.array([r['value'] for r in rows if r['name'] == name and (feat is None or r['feat'] == feat)])


base = I.run_unit(meta, yT, big, load, reps, False)
rt_z0 = I.run_unit(meta, yT, big, lambda l, a, f, r: load(l, 'Z0', f, 0), reps, False)
assert all(np.all(val(rt_z0, f'dFA_RT{k}') == 0) for k in I.KS) and np.all(val(rt_z0, 'rf_dFA_RT200') == 0)
same = I.run_unit(meta, yT, big, lambda l, a, f, r: load(l, a, 'ae', r), reps, False)
assert val(same, 'D_Z0')[0] == 0
swap = I.run_unit(meta, yT, big, lambda l, a, f, r: load(l, a, {'ae': 's2', 's2': 'ae'}[f], r), reps, False)
d0, d1 = val(base, 'D_Z0')[0], val(swap, 'D_Z0')[0]
assert abs(d0 + d1) < 1e-12 and d0 != 0, (d0, d1)
print(f'4 RT := Z0 → ΔFA = 0；S2 := AE → D = 0；交换 → D 变号（{d0:+.6f} / {d1:+.6f}）：通过')

# 5) τ = 0 的 OA / BA 与 guidance_G._scores 一致
import guidance_G as G  # noqa: E402
for arm in ('Z0', 'RT200'):
    P = load('lgb', arm, 'ae', 0)
    oa, ba = I.tau_curve(P, yT, big)
    o2, b2 = G._scores(yT, P.argmax(1), big)[:2]
    assert abs(oa[I.I_T0] - o2) < 1e-9 and abs(ba[I.I_T0] - b2) < 1e-9
print('5 τ = 0 的 OA / ba_ge30 与 guidance_G._scores 一致：通过')

# 6) B1 读法
S = dict(decidable=True, mean=0.3, lo95=-0.5, hi95=0.9)
assert I.read_b1(S) == '补采位置的影响小于 1 pp'
assert I.read_b1(dict(S, mean=1.5, lo95=0.4, hi95=2.6)).startswith('正向')
assert I.read_b1(dict(S, mean=-1.5, lo95=-2.6, hi95=-0.4)).startswith('负向')
assert I.read_b1(dict(S, mean=0.5, lo95=-0.4, hi95=1.4)).startswith('未能区分')
assert I.read_b1(dict(S, decidable=False)) == '不可判定'
print('6 B1 读法：通过')
