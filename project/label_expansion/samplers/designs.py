"""
概率设计与统计设计类选样方法，以及其他组共用的抽样与先验估计工具。

规格：sampling_shortlist.json 中 M1、M2、M3、M5、M9、M19 条目；SAMPLING_SURVEY_2026-09-24.md 第 3 节。

公共工具（其他组 `from samplers.designs import lpm2, alloc`）：
- lpm2(pi, X, rng)：局部枢轴法 LPM2（Grafström, Lundström & Schelin 2012），严格保持一阶包含概率 pi，
  并在 X 空间中分散。cKDTree 查最近的未决单元，已决单元超过 25% 时重建树。
- alloc(N_h, w_h, k, m_min)：按权重的分层样本量分配（最大余数法取整，超出层大小的截断后再分配）。
- 先验估计：olofsson_prior（Olofsson et al. 2014 式 9–10）、ht_prior（Hájek / Horvitz–Thompson）、
  aipw_prior（差分 / AIPW，Zrnic & Candès 2024）。它们都返回 (p̂, SE)。

登记的方法（注册名 → 规格）：
- M1  'srs'（简单随机）、'srs_nuts3'（NUTS3 比例分层；非空层数 > k 时改为按 NUTS3 排序的系统抽样）
- M2  'lpm2'（AE16 + 地理 geo_w=2）、'lpm2_nogeo'（geo_w=0）、'lpm2_aes2'（AE16+S2 16 + 地理）
- M3  'predstrat'（ŷ 分层、sqrt 分配、层内 LPM2）、'predstrat_prop'、'predstrat_equal'、'predstrat_neyman'、
      'predstrat_conf'（ŷ × 置信度二分，H≤20，仅 k≥200，k<200 时退回 ŷ 分层）、'predstrat_srs'（层内 SRS）
- M5  'pmht'（β=1, τ=0.3, 不等概率 LPM2）、'pmht_b2'、'pmht_b4'、'pmht_t02'、'pmht_t05'、
      'pmht_gumbel'（Kirsch 的 Gumbel-top-k 消融；没有 π，不写 meta）
- M9  'cube'（平衡 [π, P_T[:, :C−1]]）、'cube_ae3'（再加 T_ae 前 3 个主成分）、'cube_nuts2'（按 NUTS2 分层的 cube）
- M19 'clhs'（条件拉丁超立方，12 个连续变量，自写模拟退火，与 PyPI clhs 1.0.2 同一目标函数与退火日程）

所有概率设计写 ctx.meta[注册名] = {'pi', 'prior_idx', 'estimator', ...}；'estimator' 提示应使用的先验估计：
'ht'（等概率或一般 π，用 ht_prior）、'olofsson'（分层，meta 里另有 'strata' 与 'N_h'，用 olofsson_prior）、
'aipw'（M5，用 aipw_prior）。
选样函数都不读取目标池标签，也不调用 ctx.oracle。

ctx 需求：只用必需字段 T_ae、T_s2、T_xy、T_nuts3、P_T（不需要 P_T_loco、dom、P_F_oof、oracle、外国池）。
NUTS2（'cube_nuts2'、'clhs' 的 cat='nuts2'）优先取可选属性 ctx.T_nuts2；否则 T_nuts3 为 NUTS 字符串
（如 'BE211'）时取前 4 个字符；T_nuts3 为整数编码时无法还原 NUTS2，退回 NUTS3 并记在 meta['strata_source']。
'srs_nuts3' 的系统抽样分支按 T_nuts3 取值排序，T_nuts3 为 NUTS 字符串（或按字典序编的整数码）时
同一 NUTS2 的层相邻，即规格中的 (NUTS2, NUTS3, 随机键) 排序。
"""
import math
import warnings

import numpy as np
from scipy.spatial import cKDTree

from .base import register, X_of, unc_margin, finalize

EPS = 1e-9


# =====================================================================
# 公共工具：LPM2
# =====================================================================

def lpm2(pi, X, rng, eps=EPS, q=32, rebuild_frac=0.25):
    """局部枢轴法 LPM2（Grafström, Lundström & Schelin 2012, Biometrics）。

    参数
    ----
    pi  : (N,) 一阶包含概率，Σpi 为整数时恰好抽到 Σpi 个单元。
    X   : (N, d) 辅助空间坐标（欧氏距离）。
    rng : np.random.Generator。
    eps : pi ≤ eps 视为 0，pi ≥ 1−eps 视为 1。
    q   : 每次近邻查询的邻居数；q 个邻居全已定案时 q 翻倍。
    rebuild_frac : 自上次建树以来定案单元数超过树大小的该比例时，用剩余活动单元重建 cKDTree。

    算法：每步从活动单元（0<pi<1）中均匀随机取 i，找其最近的活动邻居 j，按
      s = pi_i + pi_j < 1：以概率 pi_j/s 令 (pi_i, pi_j) = (0, s)，否则 (s, 0)；
      s ≥ 1：以概率 (1−pi_j)/(2−s) 令 (pi_i, pi_j) = (1, s−1)，否则 (s−1, 1)。
    每步至少让一个单元定案，所以至多 N−1 步；每步的期望增量为 0，因此一阶包含概率保持不变。
    与 Olofsson, Grafström & Lundström 式 (2) 一致。

    返回：升序的入样下标（相对于输入数组）。
    """
    pi = np.asarray(pi, dtype=np.float64).copy()
    N = pi.shape[0]
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[0] != N:
        raise ValueError('X 与 pi 的行数不一致')
    if (pi < -eps).any() or (pi > 1 + eps).any():
        raise ValueError('pi 必须在 [0, 1] 内')
    pi[pi <= eps] = 0.0
    pi[pi >= 1 - eps] = 1.0

    act_idx = np.flatnonzero((pi > 0) & (pi < 1))
    n_act = act_idx.size
    if n_act > 0:
        # 循环内用 Python 列表做标量读写（比 numpy 标量索引快数倍）
        pl = pi.tolist()
        active = [False] * N
        for v in act_idx.tolist():
            active[v] = True
        # 活动单元池：交换删除，O(1) 均匀随机取样
        pool = act_idx.tolist()
        pos = {v: t for t, v in enumerate(pool)}
        size = n_act
        # 每步用两个均匀数（选 i、做决定）；步数 ≤ n_act − 1
        U = rng.random((n_act + 1, 2)).tolist()
        tree_ids = act_idx.tolist()
        tree = cKDTree(X[act_idx])
        n_settled = 0
        it = 0
        while size > 1:
            u0, u = U[it]
            i = pool[int(u0 * size)]
            xi = X[i]
            kq = min(q, len(tree_ids))
            j = -1
            while j < 0:
                _, nn = tree.query(xi, k=kq)
                for c in np.atleast_1d(nn).tolist():
                    cid = tree_ids[c]
                    if cid != i and active[cid]:
                        j = cid
                        break
                else:
                    # 树里包含所有活动单元，size>1 时 kq 等于树大小必能找到；这里只需翻倍
                    kq = min(2 * kq, len(tree_ids))
            pi_i, pi_j = pl[i], pl[j]
            s = pi_i + pi_j
            if s < 1:
                if u < pi_j / s:
                    pl[i], pl[j] = 0.0, s
                else:
                    pl[i], pl[j] = s, 0.0
            else:
                if u < (1 - pi_j) / (2 - s):
                    pl[i], pl[j] = 1.0, s - 1
                else:
                    pl[i], pl[j] = s - 1, 1.0
            it += 1
            for v in (i, j):
                pv = pl[v]
                if pv <= eps or pv >= 1 - eps:
                    pl[v] = 0.0 if pv <= eps else 1.0
                    active[v] = False
                    p_ = pos.pop(v)
                    last = pool[size - 1]
                    if last != v:
                        pool[p_] = last
                        pos[last] = p_
                    size -= 1
                    n_settled += 1
            if size > 1 and n_settled > rebuild_frac * len(tree_ids):
                tree_ids = pool[:size]
                tree = cKDTree(X[np.asarray(tree_ids, dtype=np.int64)])
                n_settled = 0
        if size == 1:
            # Σpi 为整数时最后一个单元的 pi 已经是 0 或 1（数值误差内）；否则以概率 pi 取整
            v = pool[0]
            pl[v] = 1.0 if U[it][0] < pl[v] else 0.0
        pi = np.asarray(pl, dtype=np.float64)
    return np.flatnonzero(pi >= 1 - eps).astype(np.int64)


# =====================================================================
# 公共工具：分层样本量分配
# =====================================================================

def alloc(N_h, w_h, k, m_min=0):
    """分层样本量分配（规格 M3）。

    N_h   : (H,) 各层大小（可以含空层）。
    w_h   : (H,) 分配权重（prop: N_h；sqrt: √N_h；equal: 1；neyman: N_h·S_h），也可为标量。
    k     : 总样本量，须 ≤ ΣN_h。
    m_min : 每个非空层的最少样本量（不超过层大小）；若 m_min × 非空层数 > k，改为 m_min=1，仍超则 0。

    步骤：n = min(N_h, m_min)（仅非空层）；然后反复把剩余量 rest = k − Σn 按 w_h 比例分给 n < N_h 的层
    （最大余数法取整，同分按层序），超过 N_h 的截断，溢出量进入下一轮，直到 Σn = k。
    权重在可分配的层上全为 0 时，改为平均分配。
    """
    N_h = np.asarray(N_h, dtype=np.int64)
    w = np.broadcast_to(np.asarray(w_h, dtype=np.float64), N_h.shape).copy()
    w[~np.isfinite(w)] = 0.0
    w = np.maximum(w, 0.0)
    k = int(k)
    if k > N_h.sum():
        raise ValueError(f'k={k} 超过总体大小 {N_h.sum()}')
    nonempty = N_h > 0
    n = None
    for m in sorted({int(m_min), 1, 0}, reverse=True):
        if m > m_min:
            continue
        n = np.where(nonempty, np.minimum(N_h, m), 0).astype(np.int64)
        if n.sum() <= k:
            break
    while True:
        rest = k - int(n.sum())
        if rest == 0:
            break
        open_ = n < N_h
        ww = np.where(open_, w, 0.0)
        if ww.sum() <= 0:
            ww = open_.astype(np.float64)
        share = rest * ww / ww.sum()
        add = np.floor(share).astype(np.int64)
        r = rest - int(add.sum())
        if r > 0:
            frac = share - add
            frac[~open_] = -1.0
            top = np.argsort(-frac, kind='stable')[:r]
            add[top] += 1
        n = np.minimum(n + add, N_h)
    return n


# =====================================================================
# 先验估计（选样之外使用；这里只提供函数）
# =====================================================================

def _onehot(y, C):
    y = np.asarray(y, dtype=np.int64)
    Y = np.zeros((y.size, C))
    Y[np.arange(y.size), y] = 1.0
    return Y


def olofsson_prior(y_sel, strata_sel, N_h, C=10, return_q=False):
    """分层估计（Olofsson et al. 2014, RSE, 式 9–10），层为外国模型预测类（或 NUTS3 等）。

    y_sel      : (n,) 所选样本的真实标签（0..C−1）。
    strata_sel : (n,) 所选样本所在层的编码（0..H−1）。
    N_h        : (H,) 各层在候选池中的大小。
    返回 (p̂, SE)，均为 (C,)：
      p̂_c = Σ_h W_h q_hc，q_hc = n_hc/n_h，W_h = N_h/N；
      SE_c = sqrt(Σ_h W_h² q_hc(1 − q_hc)/(n_h − 1))（无有限总体校正，与原文一致）。
    n_h = 1 的层方差项记为 0（此时 q_hc ∈ {0,1}）；n_h = 0 的非空层无法估计，
    在其余层上重新归一化 W_h 并发出警告（此时估计有偏）。
    return_q=True 时另返回 q (H, C)，即目标国混淆 P(y=c | 层=h) 的估计。
    """
    y = np.asarray(y_sel, dtype=np.int64)
    h = np.asarray(strata_sel, dtype=np.int64)
    N_h = np.asarray(N_h, dtype=np.float64)
    H = N_h.size
    n_h = np.bincount(h, minlength=H).astype(np.float64)
    n_hc = np.zeros((H, C))
    np.add.at(n_hc, (h, y), 1.0)
    ok = n_h > 0
    W = N_h / N_h.sum()
    if (N_h[~ok] > 0).any():
        warnings.warn('有非空层没有样本，已在其余层上重新归一化层权重（估计有偏）')
        W = np.where(ok, W, 0.0)
        W = W / W.sum()
    q = n_hc / np.maximum(n_h, 1.0)[:, None]
    p_hat = (W[:, None] * q).sum(0)
    denom = np.maximum(n_h - 1.0, 1.0)[:, None]
    var = ((W ** 2)[:, None] * q * (1 - q) / denom).sum(0)
    se = np.sqrt(var)
    if return_q:
        return p_hat, se, q
    return p_hat, se


def ht_prior(y_sel, pi_sel, C=10, N=None):
    """Horvitz–Thompson / Hájek 类先验估计。

    默认（N=None）返回 Hájek 比估计 p̂_c = Σ_S 1[y=c]/π / Σ_S 1/π（和为 1）；
    给定 N 时返回 HT 估计 p̂_c = Σ_S 1[y=c]/π / N（设计无偏，但和不一定为 1）。
    SE 用有放回（Hansen–Hurwitz）近似：对 LPM / cube / 分层这些分散设计偏保守。
    等概率设计下 Hájek 即样本类比例；分层设计 π = n_h/N_h 时点估计与 olofsson_prior 相同。
    返回 (p̂, SE)。
    """
    Y = _onehot(y_sel, C)
    w = 1.0 / np.asarray(pi_sel, dtype=np.float64)
    n = w.size
    if N is None:
        tw = w.sum()
        p_hat = (w[:, None] * Y).sum(0) / tw
        z = w[:, None] * (Y - p_hat)
        var = n / max(n - 1, 1) * (z ** 2).sum(0) / tw ** 2
    else:
        t = (w[:, None] * Y).sum(0)
        p_hat = t / N
        z = n * w[:, None] * Y                       # 每个样本对总量的 HH 估计
        var = ((z - t) ** 2).sum(0) / (n * max(n - 1, 1)) / N ** 2
    return p_hat, np.sqrt(var)


def aipw_prior(y_sel, pi_sel, P_T, sel_idx, clip=True):
    """差分 / AIPW 先验估计（规格 M5；Zrnic & Candès 2024；Breidt & Opsomer 2017）。

    θ̂_c = (1/N)·[Σ_{i∈U} P_T[i,c] + Σ_{i∈S} (1[y_i=c] − P_T[i,c])/π_i]，
    clip=True（默认）时截断到 ≥0 后归一化；clip=False 返回未截断的设计无偏估计。
    SE 用残差 HT 总量的有放回近似（截断前）。等概率 π=k/N 时即 PPI 均值估计
    mean_U f + mean_S (y − f)。返回 (θ̂, SE)。
    """
    P_T = np.asarray(P_T, dtype=np.float64)
    N, C = P_T.shape
    sel_idx = np.asarray(sel_idx, dtype=np.int64)
    pi_sel = np.asarray(pi_sel, dtype=np.float64)
    Y = _onehot(y_sel, C)
    e = (Y - P_T[sel_idx]) / pi_sel[:, None]          # (n, C)
    theta = (P_T.sum(0) + e.sum(0)) / N
    n = e.shape[0]
    t = e.sum(0)
    var = ((n * e - t) ** 2).sum(0) / (n * max(n - 1, 1)) / N ** 2
    if not clip:
        return theta, np.sqrt(var)
    th = np.clip(theta, 0.0, None)
    th = th / th.sum() if th.sum() > 0 else np.full(C, 1.0 / C)
    return th, np.sqrt(var)


# =====================================================================
# 小工具
# =====================================================================

def _codes(v):
    """把任意编码（字符串或整数）映射为 0..H−1（按取值排序，字符串即字典序）。返回 (codes, N_h)。"""
    _, codes = np.unique(np.asarray(v), return_inverse=True)
    codes = codes.ravel().astype(np.int64)
    return codes, np.bincount(codes)


def _nuts2_of(ctx):
    """NUTS2 编码。优先用 ctx.T_nuts2（若存在）；T_nuts3 为字符串时取前 4 个字符；
    否则（整数编码，无法还原 NUTS2）退回 NUTS3，并在返回的说明中注明。"""
    v = getattr(ctx, 'T_nuts2', None)
    if v is not None:
        return np.asarray(v), 'T_nuts2'
    a = np.asarray(ctx.T_nuts3)
    if a.dtype.kind in 'OUS':
        return np.array([str(x)[:4] for x in a]), 'T_nuts3[:4]'
    return a, 'T_nuts3（整数编码，无法取 NUTS2 前缀，改用 NUTS3）'


def _group(codes, H):
    """按层编码分组，返回每层成员下标（层内保持原顺序）。"""
    order = np.argsort(codes, kind='stable')
    bounds = np.searchsorted(codes[order], np.arange(H + 1))
    return [order[bounds[h]:bounds[h + 1]] for h in range(H)]


# =====================================================================
# M1 简单随机抽样（含 NUTS3 比例分层）
# =====================================================================

def _m1(ctx, k, seed, variant='srs', _name='srs'):
    rng = np.random.default_rng(seed)
    N = ctx.N
    if variant == 'srs':
        idx = rng.choice(N, size=k, replace=False)
        ctx.meta[_name] = {'pi': np.full(N, k / N), 'prior_idx': np.sort(idx), 'estimator': 'ht'}
        return finalize(idx, k, N)
    if variant != 'nuts3_prop':
        raise ValueError(variant)
    codes, N_h = _codes(ctx.T_nuts3)
    H = N_h.size
    if (N_h > 0).sum() > k:
        # 层数多于样本量：按 (NUTS3, 随机键) 排序后系统抽样（NUTS3 编码按字典序排列，同一 NUTS2 的层相邻，
        # 等价于按 (NUTS2, NUTS3, 随机键) 排序），每个单元的包含概率恰为 k/N
        key = rng.random(N)
        order = np.lexsort((key, codes))
        step = N / k
        u = rng.uniform(0.0, step)
        posn = np.minimum(np.floor(u + step * np.arange(k)).astype(np.int64), N - 1)
        idx = order[posn]
        ctx.meta[_name] = {'pi': np.full(N, k / N), 'prior_idx': np.sort(idx), 'estimator': 'ht',
                           'scheme': 'systematic'}
        return finalize(idx, k, N)
    # 规格为 m_min=0，但会让小层分到 0 个样本、π=0，使先验估计有偏（集成审查指出）；阶段 A 改为 m_min=1
    n_h = alloc(N_h, N_h, k, m_min=1)
    groups = _group(codes, H)
    idx = np.concatenate([rng.choice(groups[h], size=n_h[h], replace=False) for h in range(H) if n_h[h] > 0])
    pi = (n_h / np.maximum(N_h, 1))[codes]
    ctx.meta[_name] = {'pi': pi, 'prior_idx': np.sort(idx), 'estimator': 'olofsson', 'scheme': 'stratified',
                       'strata': codes, 'N_h': N_h, 'n_h': n_h}
    return finalize(idx, k, N)


@register('srs')
def select_srs(ctx, k, seed, variant='srs'):
    """M1 简单随机抽样。meta pi = k/N；先验用样本类比例（ht_prior）。"""
    return _m1(ctx, k, seed, variant=variant, _name='srs')


@register('srs_nuts3')
def select_srs_nuts3(ctx, k, seed, variant='nuts3_prop'):
    """M1 变体：NUTS3 比例分层（alloc(N_h, N_h, k, 0)，层内 SRS）；非空层数 > k 时改为系统抽样。
    分层时先验用 olofsson_prior(y, meta['strata'][idx], meta['N_h'])。"""
    return _m1(ctx, k, seed, variant=variant, _name='srs_nuts3')


# =====================================================================
# M2 LPM2（AE + 地理分散的等概率样本）
# =====================================================================

def _m2(ctx, k, seed, space='ae', d=16, geo_w=2.0, eps=EPS, q=32, _name='lpm2'):
    rng = np.random.default_rng(seed)
    N = ctx.N
    X = X_of(ctx, space=space, d=d, geo_w=geo_w)
    pi = np.full(N, k / N)
    idx = lpm2(pi, X, rng, eps=eps, q=q)
    ctx.meta[_name] = {'pi': pi, 'prior_idx': idx.copy(), 'estimator': 'ht'}
    return finalize(idx, k, N)


@register('lpm2')
def select_lpm2(ctx, k, seed, space='ae', d=16, geo_w=2.0, eps=EPS, q=32):
    """M2 默认：X = X_of(ctx, 'ae', 16, geo_w=2)，pi = k/N。"""
    return _m2(ctx, k, seed, space=space, d=d, geo_w=geo_w, eps=eps, q=q, _name='lpm2')


@register('lpm2_nogeo')
def select_lpm2_nogeo(ctx, k, seed, space='ae', d=16, geo_w=0.0, eps=EPS, q=32):
    """M2 变体：只在 AE 空间分散（geo_w=0）。"""
    return _m2(ctx, k, seed, space=space, d=d, geo_w=geo_w, eps=eps, q=q, _name='lpm2_nogeo')


@register('lpm2_aes2')
def select_lpm2_aes2(ctx, k, seed, space='ae+s2', d=16, geo_w=2.0, eps=EPS, q=32):
    """M2 变体：AE16 + S2 16 + 地理。"""
    return _m2(ctx, k, seed, space=space, d=d, geo_w=geo_w, eps=eps, q=q, _name='lpm2_aes2')


# =====================================================================
# M3 PredStrat：外国预测类分层 + 层内 LPM2 + Olofsson 先验
# =====================================================================

def predstrat_strata(P_T, strata='yhat'):
    """M3 的层编码：'yhat' → ŷ（H=C）；'yhat_conf' → 2ŷ + [max p > 该预测类的中位数]（H=2C）。"""
    P = np.asarray(P_T)
    C = P.shape[1]
    yhat = P.argmax(1)
    if strata == 'yhat':
        return yhat.astype(np.int64), C
    if strata == 'yhat_conf':
        pmax = P.max(1)
        med = np.full(C, np.inf)
        for c in np.unique(yhat):
            med[c] = np.median(pmax[yhat == c])
        conf = (pmax > med[yhat]).astype(np.int64)
        return 2 * yhat.astype(np.int64) + conf, 2 * C
    raise ValueError(strata)


def predstrat_weights(P_T, h, H, rule):
    """M3 的分配权重 w_h。"""
    N_h = np.bincount(h, minlength=H).astype(np.float64)
    if rule == 'prop':
        return N_h
    if rule == 'sqrt':
        return np.sqrt(N_h)
    if rule == 'equal':
        return (N_h > 0).astype(np.float64)
    if rule == 'neyman':
        P = np.asarray(P_T, dtype=np.float64)
        v = (P * (1 - P)).sum(1)                      # Σ_c p(1−p)
        S_h = np.sqrt(np.bincount(h, weights=v, minlength=H) / np.maximum(N_h, 1))
        return N_h * S_h
    raise ValueError(rule)


_alloc = alloc  # _m3 的超参数名 'alloc' 会遮蔽函数名


def _m3(ctx, k, seed, strata='yhat', alloc='sqrt', m_min=2, within='lpm2', space='ae', d=16, geo_w=2.0,
        _name='predstrat'):
    rng = np.random.default_rng(seed)
    N = ctx.N
    strata_used = strata
    if strata == 'yhat_conf' and k < 200:
        # 规格：yhat_conf 仅用于 k ≥ 200；小预算时退回 ŷ 分层并在 meta 中记录
        strata_used = 'yhat'
    h, H = predstrat_strata(ctx.P_T, strata_used)
    N_h = np.bincount(h, minlength=H)
    w = predstrat_weights(ctx.P_T, h, H, alloc)
    n_h = _alloc(N_h, w, k, m_min)
    groups = _group(h, H)
    X = X_of(ctx, space=space, d=d, geo_w=geo_w) if within == 'lpm2' else None
    parts = []
    for g in range(H):
        if n_h[g] == 0:
            continue
        mem = groups[g]
        if n_h[g] == N_h[g]:
            parts.append(mem)
        elif within == 'lpm2':
            parts.append(mem[lpm2(np.full(mem.size, n_h[g] / N_h[g]), X[mem], rng)])
        elif within == 'srs':
            parts.append(rng.choice(mem, size=n_h[g], replace=False))
        else:
            raise ValueError(within)
    idx = np.concatenate(parts)
    pi = (n_h / np.maximum(N_h, 1))[h]
    ctx.meta[_name] = {'pi': pi, 'prior_idx': np.sort(idx), 'estimator': 'olofsson', 'strata': h,
                       'N_h': N_h, 'n_h': n_h, 'strata_used': strata_used, 'alloc': alloc}
    return finalize(idx, k, N)


@register('predstrat')
def select_predstrat(ctx, k, seed, strata='yhat', alloc='sqrt', m_min=2, within='lpm2', space='ae', d=16,
                     geo_w=2.0):
    """M3 默认：ŷ 分层，sqrt 分配，m_min=2，层内 LPM2（X = X_of(ctx,'ae',16,geo_w=2)）。
    meta pi_i = n_h/N_h，strata = ŷ；先验用 olofsson_prior(y, meta['strata'][idx], meta['N_h'])。"""
    return _m3(ctx, k, seed, strata, alloc, m_min, within, space, d, geo_w, _name='predstrat')


@register('predstrat_prop')
def select_predstrat_prop(ctx, k, seed, strata='yhat', alloc='prop', m_min=2, within='lpm2', space='ae', d=16,
                          geo_w=2.0):
    """M3 变体：比例分配（≈按预测构成平衡的随机样本）。"""
    return _m3(ctx, k, seed, strata, alloc, m_min, within, space, d, geo_w, _name='predstrat_prop')


@register('predstrat_equal')
def select_predstrat_equal(ctx, k, seed, strata='yhat', alloc='equal', m_min=2, within='lpm2', space='ae', d=16,
                           geo_w=2.0):
    """M3 变体：等量分配。"""
    return _m3(ctx, k, seed, strata, alloc, m_min, within, space, d, geo_w, _name='predstrat_equal')


@register('predstrat_neyman')
def select_predstrat_neyman(ctx, k, seed, strata='yhat', alloc='neyman', m_min=2, within='lpm2', space='ae', d=16,
                            geo_w=2.0):
    """M3 变体：Neyman 分配，S_h = sqrt(mean_{i∈h} Σ_c p_ic(1−p_ic))。"""
    return _m3(ctx, k, seed, strata, alloc, m_min, within, space, d, geo_w, _name='predstrat_neyman')


@register('predstrat_conf')
def select_predstrat_conf(ctx, k, seed, strata='yhat_conf', alloc='sqrt', m_min=2, within='lpm2', space='ae', d=16,
                          geo_w=2.0):
    """M3 变体：ŷ × 置信度二分（H≤20）；规格限 k≥200，k<200 时退回 ŷ 分层（meta['strata_used'] 记录）。"""
    return _m3(ctx, k, seed, strata, alloc, m_min, within, space, d, geo_w, _name='predstrat_conf')


@register('predstrat_srs')
def select_predstrat_srs(ctx, k, seed, strata='yhat', alloc='sqrt', m_min=2, within='srs', space='ae', d=16,
                         geo_w=2.0):
    """M3 变体：层内 SRS（代替 LPM2）。"""
    return _m3(ctx, k, seed, strata, alloc, m_min, within, space, d, geo_w, _name='predstrat_srs')


# =====================================================================
# M5 PM-HT：概率化 margin + 不等概率 LPM2 + AIPW 先验
# =====================================================================

def pmht_pi(P_T, k, beta=1.0, tau=0.3):
    """M5 的包含概率：π ∝ (1−τ)·s^β/mean(s^β) + τ（乘 k/N），超过 1 的截到 1 并把余量按比例分给其他单元。"""
    P = np.asarray(P_T, dtype=np.float64)
    N = P.shape[0]
    s = unc_margin(P) + 1e-6
    q = s ** beta
    q = q / q.mean()
    pi = (k / N) * ((1 - tau) * q + tau)
    while (pi > 1).any():
        over = pi >= 1
        pi[over] = 1.0
        pi[~over] *= (k - over.sum()) / pi[~over].sum()
    return np.minimum(pi, 1.0)


def _m5(ctx, k, seed, beta=1.0, tau=0.3, design='lpm2', space='ae', d=16, geo_w=2.0, _name='pmht'):
    rng = np.random.default_rng(seed)
    N = ctx.N
    if design == 'gumbel':
        # Kirsch et al. power 采集：key = log s + Gumbel/β，取最大的 k 个；没有已知 π，不写 meta
        s = unc_margin(ctx.P_T) + 1e-6
        key = np.log(s) + rng.gumbel(size=N) / beta
        idx = np.argpartition(-key, k - 1)[:k]
        return finalize(idx, k, N)
    if design != 'lpm2':
        raise ValueError(design)
    pi = pmht_pi(ctx.P_T, k, beta, tau)
    X = X_of(ctx, space=space, d=d, geo_w=geo_w)
    idx = lpm2(pi, X, rng)
    ctx.meta[_name] = {'pi': pi, 'prior_idx': idx.copy(), 'estimator': 'aipw', 'beta': beta, 'tau': tau}
    return finalize(idx, k, N)


@register('pmht')
def select_pmht(ctx, k, seed, beta=1.0, tau=0.3, design='lpm2', space='ae', d=16, geo_w=2.0):
    """M5 默认：β=1，τ=0.3，不等概率 LPM2（X = X_of(ctx,'ae',16,geo_w=2)）。
    先验用 aipw_prior(y, meta['pi'][idx], ctx.P_T, idx)；训练 (b) 版权重 ∝ 1/π。"""
    return _m5(ctx, k, seed, beta, tau, design, space, d, geo_w, _name='pmht')


@register('pmht_b2')
def select_pmht_b2(ctx, k, seed, beta=2.0, tau=0.3, design='lpm2', space='ae', d=16, geo_w=2.0):
    """M5 变体：β=2。"""
    return _m5(ctx, k, seed, beta, tau, design, space, d, geo_w, _name='pmht_b2')


@register('pmht_b4')
def select_pmht_b4(ctx, k, seed, beta=4.0, tau=0.3, design='lpm2', space='ae', d=16, geo_w=2.0):
    """M5 变体：β=4。"""
    return _m5(ctx, k, seed, beta, tau, design, space, d, geo_w, _name='pmht_b4')


@register('pmht_t02')
def select_pmht_t02(ctx, k, seed, beta=1.0, tau=0.2, design='lpm2', space='ae', d=16, geo_w=2.0):
    """M5 变体：τ=0.2。"""
    return _m5(ctx, k, seed, beta, tau, design, space, d, geo_w, _name='pmht_t02')


@register('pmht_t05')
def select_pmht_t05(ctx, k, seed, beta=1.0, tau=0.5, design='lpm2', space='ae', d=16, geo_w=2.0):
    """M5 变体：τ=0.5。"""
    return _m5(ctx, k, seed, beta, tau, design, space, d, geo_w, _name='pmht_t05')


@register('pmht_gumbel')
def select_pmht_gumbel(ctx, k, seed, beta=1.0, tau=0.3, design='gumbel', space='ae', d=16, geo_w=2.0):
    """M5 消融：Gumbel-top-k（Kirsch 原版），没有已知 π，不写 meta。"""
    return _m5(ctx, k, seed, beta, tau, design, space, d, geo_w, _name='pmht_gumbel')


# =====================================================================
# M9 Cube 平衡抽样（numpy 实现：快速飞行阶段 + 逐个丢弃平衡变量的着陆阶段）
# =====================================================================

def _cube_flight(pi, A, units, rng, eps=EPS):
    """Cube 的快速飞行阶段（Deville & Tillé 2004；Chauvet & Tillé 2006 的逐批实现）。

    pi    : (N,) 当前包含概率，原地修改。
    A     : (N, p) 平衡矩阵，第 i 行为 x_i/π_i（π 为初始包含概率）。
    units : 参与的单元（按处理顺序）。
    每步取 p+1 个未决单元 B，在 A[B]ᵀ 的核中取方向 u，沿 ±u 随机游走到某个单元的 π 到达 0 或 1
    （概率使 E[Δπ]=0），把定案单元换成顺序中的下一个。候选用尽后，只要 A[B]ᵀ 的核非平凡就继续。
    返回飞行结束时仍未定案的单元（保持顺序）。
    """
    units = np.asarray(units, dtype=np.int64)
    units = units[(pi[units] > eps) & (pi[units] < 1 - eps)]
    p = A.shape[1]
    m = units.size
    if m == 0:
        return units
    U = rng.random(m + 1)
    B = list(units[:p + 1])
    nxt = min(p + 1, m)
    it = 0
    while len(B) > 0:
        Bi = np.asarray(B, dtype=np.int64)
        M = A[Bi].T                                    # (p, b)
        _, sv, Vt = np.linalg.svd(M, full_matrices=True)
        b = Bi.size
        if b <= p:
            # 候选用尽：只有 A[B]ᵀ 列秩不满时才有可行方向
            if sv.size and sv[-1] > 1e-10 * max(sv[0], 1.0):
                break
        u = Vt[-1].copy()
        umax = np.abs(u).max()
        u[np.abs(u) < 1e-12 * umax] = 0.0
        pb = pi[Bi]
        pos_, neg_ = u > 0, u < 0
        # λ1：沿 +u 的最大步长；λ2：沿 −u 的最大步长
        r1 = np.full(b, np.inf)
        r1[pos_] = (1 - pb[pos_]) / u[pos_]
        r1[neg_] = pb[neg_] / -u[neg_]
        r2 = np.full(b, np.inf)
        r2[pos_] = pb[pos_] / u[pos_]
        r2[neg_] = (1 - pb[neg_]) / -u[neg_]
        j1, j2 = int(np.argmin(r1)), int(np.argmin(r2))
        l1, l2 = r1[j1], r2[j2]
        if U[min(it, m)] < l2 / (l1 + l2):
            pb = pb + l1 * u
            pb[j1] = 1.0 if u[j1] > 0 else 0.0
        else:
            pb = pb - l2 * u
            pb[j2] = 0.0 if u[j2] > 0 else 1.0
        it += 1
        pb[pb <= eps] = 0.0
        pb[pb >= 1 - eps] = 1.0
        pb = np.clip(pb, 0.0, 1.0)
        pi[Bi] = pb
        B = [v for v, pv in zip(B, pb) if 0.0 < pv < 1.0]
        while len(B) < p + 1 and nxt < m:
            v = units[nxt]
            nxt += 1
            if eps < pi[v] < 1 - eps:
                B.append(v)
    return np.asarray(B, dtype=np.int64)


def _cube_land(pi, A, units, rng, eps=EPS):
    """着陆阶段：从最后一列起逐个丢弃平衡变量继续飞行（Chauvet & Tillé 2006 的变量删除法），
    最后只剩第 0 列（常数列 = 固定样本量）时即枢轴法。Σπ 非整数时剩下的单元以概率 π 取整。"""
    rem = np.asarray(units, dtype=np.int64)
    for p2 in range(A.shape[1] - 1, 0, -1):
        if rem.size <= 1:
            break
        rem = _cube_flight(pi, A[:, :p2], rem, rng, eps)
    for v in rem:
        if eps < pi[v] < 1 - eps:
            pi[v] = 1.0 if rng.random() < pi[v] else 0.0
    return pi


def cube_sample(pi0, Xbal, rng, strata=None, eps=EPS):
    """Cube 平衡抽样（numpy 版，替代 R BalancedSampling::cube / cubestratified）。

    pi0    : (N,) 包含概率；Xbal : (N, p) 平衡变量，第 0 列应为 pi0（保证固定样本量）。
    strata : None 或 (N,) 整数层编码。分层版（近似 Chauvet 2009）：先在各层内单独飞行，
             再把各层剩余单元合在一起，以 [常数, 层指示, 其余平衡变量] 飞行，最后按
             “先丢平衡变量、再丢层指示、最后只留常数列”着陆。
    返回 (N,) 取值 0/1 的最终 π（即入样指示）。
    """
    pi0 = np.asarray(pi0, dtype=np.float64)
    N = pi0.size
    A = np.asarray(Xbal, dtype=np.float64) / np.where(pi0 > 0, pi0, 1.0)[:, None]
    pi = pi0.copy()
    pi[pi <= eps] = 0.0
    pi[pi >= 1 - eps] = 1.0
    if strata is None:
        rem = _cube_flight(pi, A, rng.permutation(N), rng, eps)
        _cube_land(pi, A, rem, rng, eps)
        return pi
    codes, _ = _codes(strata)
    H = codes.max() + 1
    groups = _group(codes, H)
    rems = [_cube_flight(pi, A, rng.permutation(g), rng, eps) for g in groups if g.size]
    rem = np.concatenate(rems) if rems else np.zeros(0, dtype=np.int64)
    rem = rem[rng.permutation(rem.size)]
    hs = np.unique(codes[rem])
    ind = (codes[:, None] == hs[None, 1:]).astype(np.float64)       # 去掉一个层指示以免与常数列共线
    A2 = np.hstack([A[:, :1], ind, A[:, 1:]])
    rem = _cube_flight(pi, A2, rem, rng, eps)
    _cube_land(pi, A2, rem, rng, eps)
    return pi


def cube_xbal(ctx, k, xbal='phat'):
    """M9 的平衡变量：'phat' = [π, P_T[:, :C−1]]；'phat+ae3' 再加 T_ae 前 3 个主成分。"""
    N, C = ctx.N, ctx.C
    pi0 = np.full(N, k / N)
    cols = [pi0[:, None], np.asarray(ctx.P_T, dtype=np.float64)[:, :C - 1]]
    if xbal == 'phat+ae3':
        cols.append(np.asarray(ctx.T_ae, dtype=np.float64)[:, :3])
    elif xbal != 'phat':
        raise ValueError(xbal)
    return pi0, np.hstack(cols)


def balance_error(Xbal, pi0, idx):
    """平衡误差 ‖Σ_S x/π − Σ_U x‖ / ‖Σ_U x‖，以及逐变量的相对误差。"""
    tot = Xbal.sum(0)
    ht = (Xbal[idx] / pi0[idx][:, None]).sum(0)
    rel = np.abs(ht - tot) / np.maximum(np.abs(tot), 1e-12)
    return float(np.linalg.norm(ht - tot) / np.linalg.norm(tot)), rel


def _m9(ctx, k, seed, xbal='phat', stratify=None, _name='cube'):
    rng = np.random.default_rng(seed)
    N = ctx.N
    pi0, Xbal = cube_xbal(ctx, k, xbal)
    strata, note = None, None
    if stratify == 'nuts2':
        strata, note = _nuts2_of(ctx)
    elif stratify is not None:
        raise ValueError(stratify)
    pi = cube_sample(pi0, Xbal, rng, strata=strata)
    idx = np.flatnonzero(pi > 0.5)
    err, rel = balance_error(Xbal, pi0, idx)
    ctx.meta[_name] = {'pi': pi0, 'prior_idx': idx.copy(), 'estimator': 'ht', 'balance_err': err,
                       'balance_err_per_var': rel, 'strata_source': note}
    return finalize(idx, k, N)


@register('cube')
def select_cube(ctx, k, seed, xbal='phat', stratify=None):
    """M9 默认：等概率 π=k/N，平衡 [π, P_T[:, :C−1]]。meta 记录 balance_err。先验用 ht_prior 或 aipw_prior。"""
    return _m9(ctx, k, seed, xbal, stratify, _name='cube')


@register('cube_ae3')
def select_cube_ae3(ctx, k, seed, xbal='phat+ae3', stratify=None):
    """M9 变体：再平衡 T_ae 前 3 个主成分。"""
    return _m9(ctx, k, seed, xbal, stratify, _name='cube_ae3')


@register('cube_nuts2')
def select_cube_nuts2(ctx, k, seed, xbal='phat', stratify='nuts2'):
    """M9 变体：按 NUTS2 分层的 cube（NUTS2 取自 T_nuts2 或 T_nuts3 字符串前缀；整数编码时退回 NUTS3）。"""
    return _m9(ctx, k, seed, xbal, stratify, _name='cube_nuts2')


# =====================================================================
# M19 cLHS 条件拉丁超立方抽样
# =====================================================================

def clhs_matrix(ctx):
    """M19 的 12 个连续变量：T_ae 前 6 个、T_s2 前 4 个主成分、x_km、y_km。"""
    return np.hstack([np.asarray(ctx.T_ae, dtype=np.float64)[:, :6], np.asarray(ctx.T_s2, dtype=np.float64)[:, :4],
                      np.asarray(ctx.T_xy, dtype=np.float64)]).astype(np.float64)


def clhs_objective(Z, sel, cat=None, weights=None):
    """cLHS 目标值 (O, O1, O2, O3)（Minasny & McBratney 2006；与 clhs 1.0.2 同义，O2 对全部类别求和）。"""
    Z = np.asarray(Z, dtype=np.float64)
    k = len(sel)
    edges = np.quantile(Z, np.linspace(0, 1, k + 1), axis=0)
    p = Z.shape[1]
    eta = np.zeros((k, p), dtype=np.int64)
    for j in range(p):
        b = np.clip(np.searchsorted(edges[1:-1, j], Z[sel, j], side='right'), 0, k - 1)
        np.add.at(eta[:, j], b, 1)
    O1 = float(np.abs(eta - 1).sum())
    O3 = float(np.abs(np.corrcoef(Z, rowvar=False) - np.corrcoef(Z[sel], rowvar=False)).sum())
    O2 = 0.0
    if cat is not None:
        cc, _ = _codes(cat)
        kappa = np.bincount(cc) / cc.size
        O2 = float(np.abs(np.bincount(cc[sel], minlength=kappa.size) / k - kappa).sum())
    w = np.ones(3) if weights is None else np.asarray(weights, dtype=np.float64)
    return float(w[0] * O1 + w[1] * O2 + w[2] * O3), O1, O2, O3


def clhs_anneal(Z, k, rng, max_iterations=10000, initial_temp=1.0, temp_decrease=0.95, cycle_length=10, p=0.5,
                weights=None, cat=None):
    """cLHS 的模拟退火（numpy 自写，增量更新目标值）。

    目标 O = w1·O1 + w2·O2 + w3·O3：
      O1 = Σ_分位层 Σ_变量 |η − 1|（每个变量的 k 个等频分位层中各应有 1 个样本）；
      O2 = Σ_类别 |n_j/k − κ_j|（cat 给定时）；O3 = Σ|corr(全体) − corr(样本)|。
    每步以概率 p 随机换出一个样本，否则换出“最差”样本（其所在各分位层的 η−1 之和最大，同分随机），
    换入一个随机的未选单元；按 Metropolis 准则 exp(−ΔO/T) 接受；每 cycle_length 步 T ← T·temp_decrease
    （含第 0 步，与 clhs 1.0.2 相同）。返回最终状态（与 clhs 1.0.2 相同，不是历史最优）。

    与 clhs 1.0.2 的差别：
    - 1.0.2 的“换出最差”把分位层下标当作样本位置使用（resample_worst 中 inverse_indices 的取值被当成位置），
      实际换出的并不是最差分位层中的样本；这里按 Minasny & McBratney 的原意换出落在最拥挤分位层的样本。
    - O2 对全部类别求和（1.0.2 用 pandas 对齐，样本中缺失的类别被当作 NaN 跳过）。
    - 用 rng（np.random.Generator），不改全局随机状态；目标值增量更新（O(p²)/步），比 1.0.2 快两个数量级。
    """
    Z = np.asarray(Z, dtype=np.float64)
    Z = Z - Z.mean(0)
    N, P = Z.shape
    if k >= N:
        raise ValueError('k 必须小于 N')
    w = np.ones(3) if weights is None else np.asarray(weights, dtype=np.float64)
    edges = np.quantile(Z, np.linspace(0, 1, k + 1), axis=0)
    Bn = np.empty((N, P), dtype=np.int64)
    for j in range(P):
        Bn[:, j] = np.clip(np.searchsorted(edges[1:-1, j], Z[:, j], side='right'), 0, k - 1)
    C_data = np.corrcoef(Z, rowvar=False)
    ar = np.arange(P)

    if cat is not None:
        cc, cnt_all = _codes(cat)
        kappa = cnt_all / N
    else:
        cc, kappa = None, None

    sel = rng.choice(N, size=k, replace=False)
    in_s = np.zeros(N, dtype=bool)
    in_s[sel] = True
    rem = np.flatnonzero(~in_s)

    eta = np.zeros((k, P), dtype=np.int64)
    np.add.at(eta, (Bn[sel], np.broadcast_to(ar, (k, P))), 1)
    S1 = Z[sel].sum(0)
    S2 = Z[sel].T @ Z[sel]
    cnt = np.bincount(cc[sel], minlength=kappa.size).astype(np.float64) if cc is not None else None

    def corr_of(s1, s2):
        mu = s1 / k
        cov = (s2 - k * np.outer(mu, mu)) / (k - 1)
        sd = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
        return cov / np.outer(sd, sd)

    O1 = float(np.abs(eta - 1).sum())
    O3 = float(np.abs(C_data - corr_of(S1, S2)).sum())
    O2 = float(np.abs(cnt / k - kappa).sum()) if cc is not None else 0.0
    obj = w[0] * O1 + w[1] * O2 + w[2] * O3
    temp = float(initial_temp)

    for it in range(int(max_iterations)):
        if rng.random() < p:
            a = int(rng.integers(k))
        else:
            contrib = (eta[Bn[sel], ar] - 1).sum(1)
            worst = np.flatnonzero(contrib == contrib.max())
            a = int(worst[rng.integers(worst.size)])
        b = int(rng.integers(rem.size))
        out_u, in_u = sel[a], rem[b]
        bo, bi = Bn[out_u], Bn[in_u]
        # O1 增量：只影响 bo、bi 两组分位格
        old = np.abs(eta[bo, ar] - 1).sum() + np.abs(eta[bi, ar] - 1).sum()
        eta[bo, ar] -= 1
        eta[bi, ar] += 1
        new = np.abs(eta[bo, ar] - 1).sum() + np.abs(eta[bi, ar] - 1).sum()
        # bo == bi 的变量在 old/new 中各被数了两次，但两次相同且变化为 0，差值不受影响
        dO1 = float(new - old)
        zo, zi = Z[out_u], Z[in_u]
        S1n = S1 - zo + zi
        S2n = S2 - np.outer(zo, zo) + np.outer(zi, zi)
        O3n = float(np.abs(C_data - corr_of(S1n, S2n)).sum())
        if cc is not None:
            cnt[cc[out_u]] -= 1
            cnt[cc[in_u]] += 1
            O2n = float(np.abs(cnt / k - kappa).sum())
        else:
            O2n = 0.0
        objn = w[0] * (O1 + dO1) + w[1] * O2n + w[2] * O3n
        delta = objn - obj
        if delta <= 0 or rng.random() < math.exp(max(-delta / temp, -745.0)):
            sel[a], rem[b] = in_u, out_u
            O1 += dO1
            O2, O3, obj = O2n, O3n, objn
            S1, S2 = S1n, S2n
        else:
            eta[bo, ar] += 1
            eta[bi, ar] -= 1
            if cc is not None:
                cnt[cc[out_u]] += 1
                cnt[cc[in_u]] -= 1
        if it % cycle_length == 0:
            temp *= temp_decrease
        if (it + 1) % 2000 == 0:
            # 定期从头重算样本的一、二阶矩，防止增量更新的舍入误差累积
            S1 = Z[sel].sum(0)
            S2 = Z[sel].T @ Z[sel]
    return np.sort(sel), {'obj': obj, 'O1': O1, 'O2': O2, 'O3': O3}


def _clhs_pkg_resample_worst(continuous_objective_values, sampled_indices, remaining_indices, include=None):
    """clhs 1.0.2 的 resample_worst，逻辑逐行保留，只把单元素数组转为标量。

    原函数在“最差值唯一”时把 1 元素数组赋给标量位置，numpy ≥ 2 下抛
    ValueError('setting an array element with a sequence')，所以 PyPI clhs 1.0.2 在本环境中无法直接运行。
    注意原逻辑本身有误：inverse_indices[mask] 取到的是“唯一值个数 − 1”这个数，而不是最差样本的位置，
    所以“换出最差”实际总是换出样本数组中第 (唯一值个数 − 1) 个位置的样本。这里保持原样，只供交叉核对。
    """
    new_remaining_indices = remaining_indices.copy()
    new_sampled_indices = sampled_indices.copy()
    if include is None:
        include = np.array([], dtype=int)
    include_ = np.unique(include)
    idx_not_included = np.arange(new_sampled_indices.size)[
        np.isin(new_sampled_indices, include_, assume_unique=True, invert=True)]
    unique_obj_vals, inverse_indices = np.unique(continuous_objective_values[idx_not_included], return_inverse=True)
    idx_worst = idx_not_included[inverse_indices[(inverse_indices == unique_obj_vals.size - 1)]]
    if idx_worst.size > 1:
        idx_worst = np.random.choice(idx_worst)
    idx_worst = int(np.ravel(idx_worst)[0])
    worst_sampled_index = new_sampled_indices[idx_not_included[idx_worst]]
    idx_added = np.random.choice(new_remaining_indices.size)
    added_remaining_index = new_remaining_indices[idx_added]
    new_sampled_indices[idx_worst] = added_remaining_index
    new_remaining_indices[idx_added] = worst_sampled_index
    return new_sampled_indices, new_remaining_indices


def run_clhs_package(Z, k, seed, cat=None, **kw):
    """调用 PyPI clhs 1.0.2（只供交叉核对）：临时替换其 resample_worst 为 _clhs_pkg_resample_worst，
    并在调用前后保存、恢复 numpy 全局随机状态（该包用 np.random.seed）。返回升序下标。"""
    import importlib
    import pandas as pd
    mod = importlib.import_module('clhs.clhs')     # 子模块；包顶层的 clhs 名字是函数
    df = pd.DataFrame(np.asarray(Z, dtype=np.float64))
    if cat is not None:
        df['cat'] = _codes(cat)[0]                  # 非 float 列被 clhs 1.0.2 识别为类别变量
    state = np.random.get_state()
    orig = mod.resample_worst
    mod.resample_worst = _clhs_pkg_resample_worst
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')         # 该包退火时 np.exp 溢出的 RuntimeWarning
            res = mod.clhs(df, k, random_state=int(seed), progress=False, **kw)
    finally:
        mod.resample_worst = orig
        np.random.set_state(state)
    return np.sort(np.asarray(res['sample_indices']).astype(np.int64))


@register('clhs')
def select_clhs(ctx, k, seed, max_iterations=None, initial_temp=1.0, temp_decrease=0.95, cycle_length=10, p=0.5,
                weights=None, cat=None, impl='own'):
    """M19 cLHS：12 个连续变量（T_ae 前 6、T_s2 前 4、x_km、y_km）。
    max_iterations 默认 10000（k≤200）/ 20000（k>200）；initial_temp=1，temp_decrease=0.95，cycle_length=10，p=0.5，
    weights=None（三项权重均为 1）；cat=None 或 'nuts2'（把 NUTS2 作为类别变量，进入 O2）。
    impl='own'（默认，自写增量版）或 'clhs'（调用 PyPI clhs 1.0.2；它用全局 np.random，这里调用前后保存并恢复
    全局随机状态；k=1000 时很慢）。非概率设计，不写 meta。
    """
    N = ctx.N
    if max_iterations is None:
        max_iterations = 10000 if k <= 200 else 20000
    Z = clhs_matrix(ctx)
    cat_v = None
    if cat == 'nuts2':
        cat_v = _nuts2_of(ctx)[0]
    elif cat is not None:
        raise ValueError(cat)
    if impl == 'own':
        idx, _ = clhs_anneal(Z, k, np.random.default_rng(seed), max_iterations=max_iterations,
                             initial_temp=initial_temp, temp_decrease=temp_decrease, cycle_length=cycle_length,
                             p=p, weights=weights, cat=cat_v)
    elif impl == 'clhs':
        idx = run_clhs_package(Z, k, seed, cat=cat_v, max_iterations=max_iterations, initial_temp=initial_temp,
                               temp_decrease=temp_decrease, cycle_length=cycle_length, p=p, weights=weights)
    else:
        raise ValueError(impl)
    return finalize(idx, k, N)


# 本文件登记的方法名（测试与报告用）
DESIGN_METHODS = ['srs', 'srs_nuts3', 'lpm2', 'lpm2_nogeo', 'lpm2_aes2', 'predstrat', 'predstrat_prop',
                  'predstrat_equal', 'predstrat_neyman', 'predstrat_conf', 'predstrat_srs', 'pmht', 'pmht_b2',
                  'pmht_b4', 'pmht_t02', 'pmht_t05', 'pmht_gumbel', 'cube', 'cube_ae3', 'cube_nuts2', 'clhs']
# 概率设计（写 meta，含 pi）
PROB_DESIGNS = [m for m in DESIGN_METHODS if m not in ('pmht_gumbel', 'clhs')]
