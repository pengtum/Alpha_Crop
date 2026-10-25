"""NC 正式实验的主度量 Δ_V（方案 v6 第 3 节）：稳健版"同 OA 下的 BA"差，单位 pp。

输入曲线 c = (OA, BA)，即 analyze_confirm.unit_curves 对每个臂给出的 81 个 τ 顶点（τ = −2…2，步长 0.05，τ = 0 为第 40 个），
单位为 %。路径 = 这 81 个顶点按 τ 顺序相连的折线。只依赖 numpy，不 import 参考实现（design_calc/metrics.py 只在单元测试中用于比对）。
"""
import numpy as np

EPS = 1e-9
N_TAU = 81
I0 = 40                                          # τ = 0 的顶点下标
OFFS = np.round(-4 + 0.1 * np.arange(41), 1)     # 格点偏移 −4.0, −3.9, …, 0.0（j = 0…40）


def _check(c):
    oa, ba = (np.asarray(v, dtype=float) for v in c)
    assert oa.shape == ba.shape == (N_TAU,), (oa.shape, ba.shape)
    assert np.all(np.isfinite(oa)) and np.all(np.isfinite(ba))
    return oa, ba


def ba_plus(oa, ba, grid):
    """BA⁺(o)：路径上 OA ≥ o − EPS 的所有点中的最高 BA，对 grid 中每个 o 计算。

    每条线段上 BA 关于 OA 线性，所以线段被 OA ≥ o − EPS 截下的部分，其最大 BA 只在端点取得：
    要么是 OA ≥ o − EPS 的顶点，要么是一端在内、一端在外时 OA 恰为 o 的穿越点。不可达记 −inf。
    """
    x0, x1, y0, y1 = oa[:-1], oa[1:], ba[:-1], ba[1:]
    g = np.asarray(grid, dtype=float)[:, None]
    in0, in1 = x0 >= g - EPS, x1 >= g - EPS                       # (格点, 80 条线段)
    cross = in0 != in1                                            # 穿越：外端 OA < o − EPS ≤ 内端 OA，故 x1 ≠ x0
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.clip((g - x0) / (x1 - x0), 0.0, 1.0)
    yc = np.where(cross, y0 + t * (y1 - y0), -np.inf)
    yv = np.maximum(np.where(in0, y0, -np.inf), np.where(in1, y1, -np.inf))
    return np.maximum(yv, yc).max(1)


def delta_v(cA, cB):
    """Δ_V(A → B)：模型 B 相对参照 A，在 41 个格点上 BA⁺_B − BA⁺_A 的平均。

    锚点 m = min(OA_A(τ=0), OA_B(τ=0))；格点 o_j = m + OFFS[j]。格点均 ≤ 两模型 τ = 0 的 OA，故两条包络处处可达，无覆盖规则。
    """
    oa_a, ba_a = _check(cA)
    oa_b, ba_b = _check(cB)
    grid = min(oa_a[I0], oa_b[I0]) + OFFS
    d = ba_plus(oa_b, ba_b, grid) - ba_plus(oa_a, ba_a, grid)
    assert np.all(np.isfinite(d))
    return float(d.mean())
