"""partB_indep.py 的单元测试：上包络插值、可达率、FA 缺失规则、指标与 guidance_G._scores 一致。"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import partB_indep as M  # noqa: E402


def naive_env(O, B, taus, o):
    best, bt = -np.inf, np.nan
    for i in range(len(O) - 1):
        lo, hi = min(O[i], O[i + 1]), max(O[i], O[i + 1])
        if lo - 1e-9 <= o <= hi + 1e-9:
            if O[i] == O[i + 1]:
                v, t = max(B[i], B[i + 1]), taus[i] if B[i] >= B[i + 1] else taus[i + 1]
            else:
                w = (o - O[i]) / (O[i + 1] - O[i])
                v, t = B[i] + w * (B[i + 1] - B[i]), taus[i] + w * (taus[i + 1] - taus[i])
            if v > best + 1e-12:
                best, bt = v, t
    return (np.nan, np.nan) if best == -np.inf else (best, bt)


rng = np.random.default_rng(0)
# 1) 随机非单调折线：与朴素实现逐点一致
for trial in range(200):
    n = rng.integers(3, 30)
    taus = np.sort(rng.uniform(-1, 2, n))
    O = rng.uniform(70, 80, n)
    B = rng.uniform(40, 60, n)
    if trial % 5 == 0:
        O[1] = O[2]  # 水平段
    grid = M.window(rng.uniform(72, 80))
    env, tau_o = M.envelope(O, B, taus, grid)
    for j, o in enumerate(grid):
        v, t = naive_env(O, B, taus, o)
        assert (np.isnan(v) and np.isnan(env[j])) or abs(v - env[j]) < 1e-9, (trial, j, v, env[j])
        assert (np.isnan(t) and np.isnan(tau_o[j])) or abs(t - tau_o[j]) < 1e-9
# 2) 单调直线：OA = 80 − 10τ，BA = 40 + 10τ → 在 o 处 BA = 120 − o
taus = np.round(np.arange(-0.5, 2.0001, 0.05), 2)
O, B = 80 - 10 * taus, 40 + 10 * taus
grid = M.window(78.0)
env, tau_o = M.envelope(O, B, taus, grid)
assert np.allclose(env, 120 - grid) and np.allclose(tau_o, (80 - grid) / 10)
fa = M.fa_of(env)
assert fa['cov'] == 1.0 and not fa['partial'] and abs(fa['FA'] - np.mean(120 - grid)) < 1e-9
# 3) 部分可达：曲线 OA 只覆盖 [76, 90] → 窗口 [74, 78] 中 21/41 可达
O2 = np.linspace(90, 76, 20)
env, _ = M.envelope(O2, np.linspace(30, 50, 20), np.arange(20.), grid)
fa = M.fa_of(env)
assert abs(fa['cov'] - 21 / 41) < 1e-12 and fa['partial'] and np.isfinite(fa['FA'])
O3 = np.linspace(90, 76.5, 20)
fa = M.fa_of(M.envelope(O3, np.linspace(30, 50, 20), np.arange(20.), grid)[0])
assert fa['cov'] < 0.5 and np.isnan(fa['FA']) and np.isfinite(fa['fa_raw'])
# 4) 两条曲线取上包络
e1, _ = M.envelope(O, B, taus, grid)
e2, _ = M.envelope(O, B + 1.0, taus, grid)
e, t, cid = M.envelope_multi([(O, B, taus), (O, B + 1.0, taus)], grid)
assert np.allclose(e, e2) and (cid == 1).all()
# 5) 窗口端点
g = M.window(77.123456789)
assert g[-1] == 77.123456789 and abs(g[0] - (77.123456789 - 4)) < 1e-12 and len(g) == 41
# 6) oa_ba 与 guidance_G._scores 一致
for _ in range(50):
    n = 5000
    y = rng.integers(0, 10, n)
    ph = np.where(rng.random(n) < 0.6, y, rng.integers(0, 10, n))
    big = np.flatnonzero(np.bincount(y, minlength=10) >= 30)
    o1, b1 = M.oa_ba(y, ph, big)
    o2, b2, _, _ = M.G._scores(y, ph, big)
    assert abs(o1 - o2) < 1e-9 and abs(b1 - b2) < 1e-9
print('全部测试通过')
