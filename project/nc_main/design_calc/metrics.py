"""候选指标（两条 τ 曲线 A = 基线、B = 处理之间的差，单位 pp）。曲线 = 81 个 τ 顶点，相邻顶点线性插值。"""
import numpy as np

EPS = 1e-9
OFFS = np.round(np.arange(-4.0, 0.0 + 1e-9, 0.1), 1)
I0 = 40
MIN_COV = 0.5


def env_exact(x, y, grid):
    """step0B.envelope 的同一规则：在路径上找 x = g 的点（线性插值；平段取两端 y 较大者），取 y 最大；不可达为 NaN。"""
    x0, x1, y0, y1 = x[:-1], x[1:], y[:-1], y[1:]
    lo, hi = np.minimum(x0, x1) - EPS, np.maximum(x0, x1) + EPS
    d = x1 - x0
    flat = np.abs(d) < EPS
    inside = (grid[:, None] >= lo[None]) & (grid[:, None] <= hi[None])
    lm = np.clip((grid[:, None] - x0[None]) / np.where(flat, 1.0, d)[None], 0.0, 1.0)
    lm = np.where(flat[None], (y1 > y0).astype(float)[None], lm)
    val = np.where(inside, y0[None] + lm * (y1 - y0)[None], -np.inf)
    b = val.max(1)
    b[~np.isfinite(b)] = np.nan
    return b


def env_fd(x, y, grid):
    """自由处置（Pareto）包络：路径上 x ≥ g 的所有点中 y 的最大值；g 高于路径最大 x 时为 NaN。"""
    e = env_exact(x, y, grid)
    m = x[None, :] >= grid[:, None] - EPS
    v = np.where(m, y[None, :], -np.inf).max(1)
    v = np.fmax(v, np.where(np.isfinite(e), e, -np.inf))
    v[~np.isfinite(v)] = np.nan
    return v


def paired(ea, eb):
    ok = np.isfinite(ea) & np.isfinite(eb)
    return float(np.mean(ea[ok] - eb[ok])) if ok.mean() >= MIN_COV else np.nan


def diag_gap(oa, ba, pts):
    """对点 a：max_{p ∈ 路径} min(p_OA − a_OA, p_BA − a_BA)，即路径的可达区域沿 (1,1) 方向超出 a 的量（可为负）。"""
    pts = np.asarray(pts, float)
    o0, o1, b0, b1 = oa[:-1], oa[1:], ba[:-1], ba[1:]
    ao, ab = pts[:, 0:1], pts[:, 1:2]
    f0 = np.minimum(o0[None] - ao, b0[None] - ab)
    f1 = np.minimum(o1[None] - ao, b1[None] - ab)
    den = (o1 - o0) - (b1 - b0)
    with np.errstate(divide='ignore', invalid='ignore'):
        lam = ((b0[None] - ab) - (o0[None] - ao)) / den[None]
    ok = np.isfinite(lam) & (lam > 0) & (lam < 1)
    lam = np.where(ok, lam, 0.0)
    fc = np.where(ok, (o0[None] + lam * (o1 - o0)[None]) - ao, -np.inf)
    return np.max(np.maximum(np.maximum(f0, f1), fc), 1)


def ref_points(ca, anc):
    g = anc + OFFS
    b = env_fd(ca['oa'], ca['ba'], g)
    ok = np.isfinite(b)
    return np.c_[g[ok], b[ok]]


def all_metrics(cA, cB, ref):
    """cA、cB：dict(oa, ba, f1, kappa)（81 点）。ref ∈ {'A', 'B'}：注册口径的锚点模型（Δ_L：A = Z0；Δ_R：OA★ 较小者）。"""
    out = {}
    sA, sB = cA['oa'][I0], cB['oa'][I0]
    cR = cA if ref == 'A' else cB
    anc = cR['oa'][I0]
    g = anc + OFFS
    # 0 注册口径
    out['reg_BA@OA'] = paired(env_exact(cB['oa'], cB['ba'], g), env_exact(cA['oa'], cA['ba'], g))
    # τ = 0 读数
    out['t0_BA'] = cB['ba'][I0] - cA['ba'][I0]
    out['t0_OA'] = cB['oa'][I0] - cA['oa'][I0]
    out['t0_bal'] = 0.5 * (out['t0_BA'] + out['t0_OA'])
    out['t0_F1'] = cB['f1'][I0] - cA['f1'][I0]
    out['t0_kappa'] = cB['kappa'][I0] - cA['kappa'][I0]
    # 1 自由处置包络的同 OA 下 BA（同窗口）
    out['fd_BA@OA'] = paired(env_fd(cB['oa'], cB['ba'], g), env_fd(cA['oa'], cA['ba'], g))
    # 2 对偶：同 BA 下 OA，窗口 BA★ + [0, 4]（与注册窗口在前沿同一侧）
    bst = cR['ba'][I0]
    gb = bst + (OFFS - OFFS[0])   # BA★ + [0, 4]
    out['fd_OA@BA'] = paired(env_fd(cB['ba'], cB['oa'], gb), env_fd(cA['ba'], cA['oa'], gb))
    out['two_way'] = 0.5 * (out['fd_BA@OA'] + out['fd_OA@BA']) if np.isfinite(out['fd_BA@OA']) and np.isfinite(out['fd_OA@BA']) else np.nan
    # 3 对角（等量）差：在锚点模型窗口内的前沿点上
    P = ref_points(cR, anc)
    if len(P) >= MIN_COV * len(OFFS):
        out['diag_ref'] = float(np.mean(diag_gap(cB['oa'], cB['ba'], P) - diag_gap(cA['oa'], cA['ba'], P)))
    else:
        out['diag_ref'] = np.nan
    PA, PB = ref_points(cA, sA), ref_points(cB, sB)
    if len(PA) >= MIN_COV * len(OFFS) and len(PB) >= MIN_COV * len(OFFS):
        dA = np.mean(diag_gap(cB['oa'], cB['ba'], PA) - diag_gap(cA['oa'], cA['ba'], PA))
        dB = np.mean(diag_gap(cB['oa'], cB['ba'], PB) - diag_gap(cA['oa'], cA['ba'], PB))
        out['diag_sym'] = float(0.5 * (dA + dB))
        out['diag_A'] = float(dA)
        out['diag_B'] = float(dB)
    else:
        out['diag_sym'] = out['diag_A'] = out['diag_B'] = np.nan
    # 对角差只在锚点（τ=0 点）一点上
    a0 = np.array([[cR['oa'][I0], env_fd(cR['oa'], cR['ba'], np.array([cR['oa'][I0]]))[0]]])
    out['diag_pt'] = float(diag_gap(cB['oa'], cB['ba'], a0)[0] - diag_gap(cA['oa'], cA['ba'], a0)[0])
    # 4 τ 族上的最优值（支撑函数）
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        sc = lambda c: np.max(w * c['oa'] + (1 - w) * c['ba'])  # noqa: E731
        out[f'max_w{w:g}'] = float(sc(cB) - sc(cA))
    out['max_F1'] = float(np.max(cB['f1']) - np.max(cA['f1']))
    out['max_kappa'] = float(np.max(cB['kappa']) - np.max(cA['kappa']))
    # 5 窗口锚定在两模型 OA★ 较大者 / 中点（精确包络与自由处置包络）
    for nm, a in (('maxanc', max(sA, sB)), ('midanc', 0.5 * (sA + sB))):
        gg = a + OFFS
        out[f'reg_{nm}'] = paired(env_exact(cB['oa'], cB['ba'], gg), env_exact(cA['oa'], cA['ba'], gg))
        out[f'fd_{nm}'] = paired(env_fd(cB['oa'], cB['ba'], gg), env_fd(cA['oa'], cA['ba'], gg))
    # 7 最小锚点（两模型 τ=0 OA 的较小者；BA 对偶用两模型 τ=0 BA 的较小者）：覆盖率恒为 1，交换 A、B 时取负
    m = min(sA, sB)
    gm = m + OFFS
    out['V_min'] = paired(env_fd(cB['oa'], cB['ba'], gm), env_fd(cA['oa'], cA['ba'], gm))
    mb = min(cA['ba'][I0], cB['ba'][I0])
    gbm = mb + OFFS
    out['H_min'] = paired(env_fd(cB['ba'], cB['oa'], gbm), env_fd(cA['ba'], cA['oa'], gbm))
    out['TW_min'] = 0.5 * (out['V_min'] + out['H_min'])
    # 对角差在最小锚点模型的窗口点上
    cM = cA if sA <= sB else cB
    Pm = ref_points(cM, m)
    out['diag_min'] = float(np.mean(diag_gap(cB['oa'], cB['ba'], Pm) - diag_gap(cA['oa'], cA['ba'], Pm)))
    # 6 前沿积分：自由处置包络在共同 OA 区间上的平均差
    lo = max(cA['oa'].min(), cB['oa'].min())
    hi = min(cA['oa'].max(), cB['oa'].max())
    if hi - lo >= 0.1:
        gg = np.arange(lo, hi + 1e-9, 0.1)
        out['frontier_int'] = paired(env_fd(cB['oa'], cB['ba'], gg), env_fd(cA['oa'], cA['ba'], gg))
        out['frontier_width'] = hi - lo
    else:
        out['frontier_int'] = np.nan
        out['frontier_width'] = hi - lo
    return out


def xfit_bal(cA_h, cB_h, w=0.5):
    """交叉拟合：在一半上选 τ（使 w·OA + (1−w)·BA 最大），在另一半上读；两个方向平均。cX_h = {'h1': c, 'h2': c}。"""
    def one(ch):
        v = []
        for s, t in (('h1', 'h2'), ('h2', 'h1')):
            j = int(np.argmax(w * ch[s]['oa'] + (1 - w) * ch[s]['ba']))
            v.append(w * ch[t]['oa'][j] + (1 - w) * ch[t]['ba'][j])
        return np.mean(v)

    def ins(ch):
        return np.mean([np.max(w * ch[s]['oa'] + (1 - w) * ch[s]['ba']) for s in ('h1', 'h2')])
    return one(cB_h) - one(cA_h), ins(cB_h) - ins(cA_h), one(cA_h) - ins(cA_h)


def xfit_w(cA_h, cB_h, ws=(0.0, 0.25, 0.5, 0.75, 1.0)):
    out = {}
    for w in ws:
        out[f'xmax_w{w:g}'] = float(xfit_bal(cA_h, cB_h, w)[0])
    return out
