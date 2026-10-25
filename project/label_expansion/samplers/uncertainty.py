"""
不确定性与"不确定性 + 多样性"类选样方法（SAMPLING_SURVEY_2026-09-24.md 第 3 节；规格见 sampling_shortlist.json）：

- M4  Margin 不确定性采样（外国模型校准概率 P_T 上的 margin / 熵；可选零标签 EM 先验校正；多轮版）。
      注册名 'margin'（单轮 margin，默认）、'entropy'（单轮归一化熵）、'margin_prior'（prior_adj=True）、
      'margin_r2'、'margin_r4'（多轮，需要 ctx.oracle：每轮揭示所选标签，外国 ∪ 已标目标样本重训轻量 LightGBM，
      外国分组留国（LOCO）温度校准后重算分数）。
- M6  GIST（Fahrbach et al., NeurIPS 2025）：线性效用 u = margin + max-min 多样化，阈值贪心独立集。
      注册名 'gist_margin'（α=0.9）、'gist_maxmin'（α=0）、'gist_geo'（d = max(‖ΔA‖/s_A, ‖Δxy‖/s_G)）、
      'kcenter'（Gonzalez 最远点贪心，随机首点）、'kcenter_foreign'（mind 初始化为到最近外国点的距离）。
- M16 k-DPP：L_ij = q_i φ_iᵀφ_j q_j（φ 为 RFF，q = margin^γ）。注册名 'kdpp'（γ=0.5）、'kdpp_div'（γ=0）。
      k ≥ 1000 时按规格把候选子采样到 1 万。
- M7  BADGE 树模型代理：幻觉梯度 g = (p − e_ŷ)⊗h 上的 k-means++ 播种。注册名 'badge'（h = AE32）、
      'badge_ae_s2'（h = AE16 + S2_16）。

以上方法都不是概率设计（没有闭式 π），因此都不写 ctx.meta。
只有 'margin_r2'、'margin_r4' 调用 ctx.oracle（第 6.2 节第 11 条：仅 M4 多轮版、M8 与神谕允许）；其余方法不读标签。
M7 规格中的"两轮版"需要 oracle，不在允许名单内，因此不登记；本文件提供 badge_core(init_idx=...) 供 M8 等复用。

本文件新增的公共函数（base.py 未提供）：
- em_prior(P, p_train)：Saerens–Latinne–Decaestecker 零标签 EM 先验估计；prior_adjust(P, p_hat, p_train)。
- fit_predict_lgbm(ctx, S, y_S, seed, ...)：外国 ∪ 已标目标样本的轻量 LightGBM + 分组留国温度校准，
  返回 (目标池校准概率, 训练集加权类别比例)。实验主程序可用 fit_predict 超参数注入正式训练函数。
- gist_core / GistMetric：GIST 主体（任意度量），供 M8 'hybrid_gist' 等复用。
- badge_core：BADGE 主体（可指定已选点作为初始中心）。
- kdpp_sample(Phi, k, random_state)：L = PhiᵀPhi 的精确 k-DPP 采样（阶段 1 对数域自写，阶段 2 用 dppy 的 GS）。
"""
import contextlib
import io
import math
import os
import warnings

import numpy as np
from scipy.spatial.distance import pdist

from .base import X_of, X_of_foreign, finalize, register, unc_ent, unc_margin

# 需要 ctx.oracle 的注册名（其余方法可在没有 oracle、没有 _true_y 的 ctx 上运行）
USES_ORACLE = frozenset({'margin_r2', 'margin_r4'})
# 与 seed 无关的确定性方法（同一 ctx 下不同 seed 结果相同；平局打破噪声只影响完全相等的分数）
DETERMINISTIC = frozenset({'margin', 'entropy', 'margin_prior', 'gist_margin', 'gist_maxmin', 'gist_geo',
                           'kcenter_foreign'})


def _threads(n_jobs=None):
    """LightGBM 线程数：显式参数 > 环境变量 LGBM_THREADS > 1。
    机器满载时 2 线程（OpenMP 自旋等待）实测比 1 线程慢约 3 倍，故默认 1。"""
    if n_jobs is not None:
        return int(n_jobs)
    return int(os.environ.get('LGBM_THREADS', 1))


# =====================================================================
# M4 Margin / 熵 不确定性采样
# =====================================================================

def _resolve_p_train(ctx, p_train):
    """外国模型的训练先验 p_F。'foreign'（默认，规格）：外国池类比例；'uniform'：外国模型若用了类别平衡权重，
    其隐含训练先验为均匀；也可直接给 (C,) 数组。"""
    C = ctx.C
    if isinstance(p_train, str):
        if p_train == 'foreign':
            p = np.bincount(np.asarray(ctx.F_y, dtype=np.int64), minlength=C)[:C].astype(np.float64)
        elif p_train == 'uniform':
            p = np.ones(C)
        else:
            raise ValueError(p_train)
    else:
        p = np.asarray(p_train, dtype=np.float64)
    return p / p.sum()


def em_prior(P, p_train, max_iter=1000, tol=1e-10):
    """Saerens, Latinne & Decaestecker (2002) 零标签 EM：估计目标先验 p̂_T。
    P 应是已校准（温度或 BCTS）的外国模型概率，p_train 是其训练先验。不读任何标签。"""
    P = np.asarray(P, dtype=np.float64)
    pt = np.clip(np.asarray(p_train, dtype=np.float64), 1e-12, None)
    p = pt / pt.sum()
    for _ in range(max_iter):
        A = P * (p / pt)
        A /= np.clip(A.sum(1, keepdims=True), 1e-300, None)
        p_new = A.mean(0)
        done = np.abs(p_new - p).sum() < tol
        p = p_new
        if done:
            break
    return p


def prior_adjust(P, p_hat, p_train):
    """P' ∝ P · p̂_T / p_train，再按行归一化（Saerens 式先验校正）。"""
    ratio = np.clip(np.asarray(p_hat, float), 1e-12, None) / np.clip(np.asarray(p_train, float), 1e-12, None)
    Q = np.asarray(P, dtype=np.float64) * ratio[None, :]
    return Q / np.clip(Q.sum(1, keepdims=True), 1e-300, None)


_SCORES = {'margin': unc_margin, 'entropy': unc_ent}


def _feat(ctx, feat):
    """重训用的特征：ctx 中的 PCA 特征（树模型对缩放不敏感，直接用原值）。"""
    parts = {'ae': ('ae',), 's2': ('s2',), 'ae+s2': ('ae', 's2')}[feat]
    F = np.hstack([np.asarray(getattr(ctx, f'F_{p}'), dtype=np.float32) for p in parts])
    T = np.hstack([np.asarray(getattr(ctx, f'T_{p}'), dtype=np.float32) for p in parts])
    return F, T


def _raw_full(model, X, C):
    """LightGBM 的原始分数，展开到全部 C 类；训练集中缺失的类记为 −inf（概率 0）。"""
    z = model.predict(X, raw_score=True)
    if z.ndim == 1:  # 只有两类时 LightGBM 给出类 1 的对数几率
        z = np.c_[np.zeros_like(z), z]
    out = np.full((len(X), C), -np.inf)
    out[:, np.asarray(model.classes_, dtype=np.int64)] = z
    return out


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(z, y, bounds=(0.05, 20.0)):
    """在 (logits, 标签) 上按负对数似然拟合单一温度 T（p = softmax(z/T)）。真实类 logit 为 −inf 的样本丢弃。"""
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp
    y = np.asarray(y, dtype=np.int64)
    zy = z[np.arange(len(y)), y]
    ok = np.isfinite(zy)
    z, zy = z[ok], zy[ok]

    def nll(lt):
        a = math.exp(-lt)
        return float(np.mean(logsumexp(z * a, axis=1) - zy * a))

    res = minimize_scalar(nll, bounds=(math.log(bounds[0]), math.log(bounds[1])), method='bounded',
                          options={'xatol': 1e-4})
    return float(math.exp(res.x))


def fit_predict_lgbm(ctx, S, y_S, seed, feat='ae+s2', r=20.0, class_weight='foreign', n_trees=100,
                     num_leaves=31, learning_rate=0.1, calib='ts', calib_folds=5, n_jobs=None, return_info=False):
    """外国池 ∪ 已标目标样本 S 训练轻量 LightGBM，返回 (目标池校准概率 (N, C), 训练集加权类别比例 (C,))。

    - 样本权重：外国 1、目标 r；类别权重只按外国池类频率计算，外国与目标同样应用（第 5.4 节）。
    - 校准（calib='ts'）：外国池按国家分成 calib_folds 组做"分组留国"，每组用（其余外国 ∪ S）重训同配置模型、
      在留出国上得到袋外 logits，拟合单一温度 T。calib_folds=None 或 ≥ 国家数时为严格 LOCO（每国一组）。
      默认 5 组（每组约 2 国）：严格 LOCO 每轮要多训 10 个模型，rounds=4 时超过 3 分钟预算。
    - 特征：ctx 的 PCA 特征（默认 AE32 + S2_32）；ctx 不含原始特征，正式训练函数请用 fit_predict 超参数注入。
    """
    import lightgbm as lgb
    C = ctx.C
    XF, XT = _feat(ctx, feat)
    yF = np.asarray(ctx.F_y, dtype=np.int64)
    S = np.asarray(S, dtype=np.int64)
    y_S = np.asarray(y_S, dtype=np.int64)
    X = np.vstack([XF, XT[S]])
    y = np.r_[yF, y_S]
    nF = len(yF)
    w = np.r_[np.ones(nF), np.full(len(S), float(r))]
    if class_weight == 'foreign':
        cnt = np.bincount(yF, minlength=C)[:C].astype(np.float64)
        cw = np.where(cnt > 0, nF / (C * np.maximum(cnt, 1.0)), 1.0)
        w = w * cw[y]
    elif class_weight is not None:
        raise ValueError(class_weight)
    params = dict(n_estimators=n_trees, learning_rate=learning_rate, num_leaves=num_leaves, max_depth=-1,
                  min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  random_state=int(seed) % (2 ** 31), n_jobs=_threads(n_jobs), verbose=-1,
                  deterministic=True, force_col_wise=True)

    def fit(rows):
        m = lgb.LGBMClassifier(**params)
        m.fit(X[rows], y[rows], sample_weight=w[rows])
        return m

    model = fit(np.arange(len(y)))
    zT = _raw_full(model, XT, C)
    T = 1.0
    if calib == 'ts':
        ctry = np.asarray(ctx.F_country)
        countries = np.unique(ctry)
        if calib_folds is None or calib_folds >= len(countries):
            groups = [[c] for c in countries]
        else:
            perm = np.random.default_rng(seed).permutation(countries)
            groups = [list(g) for g in np.array_split(perm, calib_folds)]
        z_oof = np.full((nF, C), -np.inf)
        tgt_rows = nF + np.arange(len(S))
        for g in groups:
            hold = np.isin(ctry, g)
            m = fit(np.r_[np.flatnonzero(~hold), tgt_rows])
            z_oof[hold] = _raw_full(m, XF[hold], C)
        T = fit_temperature(z_oof, yF)
    elif calib not in (None, 'none'):
        raise ValueError(calib)
    P = _softmax(zT / T)
    p_tr = np.bincount(y, weights=w, minlength=C)[:C]
    p_tr = p_tr / p_tr.sum()
    if return_info:
        return P, p_tr, {'T': T}
    return P, p_tr


def _uncertainty_select(ctx, k, seed, score='margin', prior_adj=False, rounds=1, p_train='foreign',
                        fit_predict=None, **fit_kw):
    """M4 主体。单轮：对 u = score(P) 取前 k（1e-12 量级随机噪声只用于打破平局）。
    多轮：k_t = k // rounds（余数给最后一轮）；每轮在未选样本中取分数最高的 k_t 个，
    非最后一轮时经 ctx.oracle 揭示本轮标签，用 fit_predict(ctx, S, y_S, seed) 重训并重算分数。"""
    if score not in _SCORES:
        raise ValueError(score)
    sc = _SCORES[score]
    N = ctx.N
    rng = np.random.default_rng(seed)
    noise = 1e-12 * rng.random(N)
    P = np.asarray(ctx.P_T, dtype=np.float64)
    pt = _resolve_p_train(ctx, p_train)
    if prior_adj:
        P = prior_adjust(P, em_prior(P, pt), pt)
    rounds = int(rounds)
    if rounds <= 1:
        idx = np.argsort(-(sc(P) + noise), kind='stable')[:k]
        return finalize(idx, k, N)

    # ---- 多轮版：需要 oracle ----
    if ctx.oracle is None:
        raise RuntimeError('M4 多轮版（rounds > 1）需要 ctx.oracle 揭示每轮所选标签')
    rounds = min(rounds, k)
    base = k // rounds
    sizes = [base] * (rounds - 1) + [k - base * (rounds - 1)]
    if fit_predict is None:
        def fit_predict(c, S_, yS_, sd):
            return fit_predict_lgbm(c, S_, yS_, sd, **fit_kw)
    elif fit_kw:
        raise ValueError(f'给了 fit_predict 时不接受训练超参数：{sorted(fit_kw)}')
    taken = np.zeros(N, dtype=bool)
    S = np.zeros(0, dtype=np.int64)
    y_S = np.zeros(0, dtype=np.int64)
    for t, kt in enumerate(sizes):
        s = sc(P) + noise
        s[taken] = -np.inf
        new = np.argsort(-s, kind='stable')[:kt]
        taken[new] = True
        S = np.r_[S, new]
        if t < rounds - 1:
            y_S = np.r_[y_S, np.asarray(ctx.oracle(new), dtype=np.int64)]
            out = fit_predict(ctx, S, y_S, int(seed) * 1000 + t + 1)
            P, pt_new = (out if isinstance(out, tuple) else (out, None))[:2]
            P = np.asarray(P, dtype=np.float64)
            if prior_adj:
                if pt_new is None:
                    raise ValueError('prior_adj=True 时 fit_predict 需返回 (P, p_train)')
                P = prior_adjust(P, em_prior(P, pt_new), pt_new)
    return finalize(S, k, N)


@register('margin')
def select_margin(ctx, k, seed, score='margin', prior_adj=False, rounds=1, p_train='foreign',
                  fit_predict=None, **fit_kw):
    """M4 单轮 margin（默认）。score ∈ {'margin','entropy'}；prior_adj：零标签 EM 先验校正后再算分数；
    rounds > 1 为多轮版（需要 ctx.oracle），fit_kw 传给 fit_predict_lgbm。"""
    return _uncertainty_select(ctx, k, seed, score=score, prior_adj=prior_adj, rounds=rounds, p_train=p_train,
                               fit_predict=fit_predict, **fit_kw)


@register('entropy')
def select_entropy(ctx, k, seed, prior_adj=False, rounds=1, p_train='foreign', fit_predict=None, **fit_kw):
    """M4 单轮归一化熵。"""
    return _uncertainty_select(ctx, k, seed, score='entropy', prior_adj=prior_adj, rounds=rounds,
                               p_train=p_train, fit_predict=fit_predict, **fit_kw)


@register('margin_prior')
def select_margin_prior(ctx, k, seed, score='margin', p_train='foreign', rounds=1, fit_predict=None, **fit_kw):
    """M4 prior_adj=True：P ∝ P_T·p̂_T/p_F（p̂_T 为零标签 EM 估计）后算 margin。"""
    return _uncertainty_select(ctx, k, seed, score=score, prior_adj=True, rounds=rounds, p_train=p_train,
                               fit_predict=fit_predict, **fit_kw)


@register('margin_r2')
def select_margin_r2(ctx, k, seed, score='margin', prior_adj=False, rounds=2, p_train='foreign',
                     fit_predict=None, **fit_kw):
    """M4 多轮版（2 轮）。需要 ctx.oracle。"""
    return _uncertainty_select(ctx, k, seed, score=score, prior_adj=prior_adj, rounds=rounds, p_train=p_train,
                               fit_predict=fit_predict, **fit_kw)


@register('margin_r4')
def select_margin_r4(ctx, k, seed, score='margin', prior_adj=False, rounds=4, p_train='foreign',
                     fit_predict=None, **fit_kw):
    """M4 多轮版（4 轮）。需要 ctx.oracle。"""
    return _uncertainty_select(ctx, k, seed, score=score, prior_adj=prior_adj, rounds=rounds, p_train=p_train,
                               fit_predict=fit_predict, **fit_kw)


# =====================================================================
# M6 GIST 与 k-center
# =====================================================================

class GistMetric:
    """GIST 用的度量。'euclid'：X 上的欧氏距离；'geo'：d = max(‖ΔA‖/s_A, ‖Δxy‖/s_G)（两个度量取最大仍是度量）。
    内部统一用平方距离比较阈值；集合的最小成对距离与直径用 float64 精确计算。"""

    def __init__(self, X, xy=None, s_A=1.0, s_G=1.0):
        self.X = np.ascontiguousarray(X, dtype=np.float64)
        self.sq = np.einsum('ij,ij->i', self.X, self.X)
        self.N = len(self.X)
        self.xy = None if xy is None else np.ascontiguousarray(xy, dtype=np.float64)
        self.s_A = float(s_A)
        self.s_G = float(s_G)

    def sq_to_all(self, v):
        """点 v 到全部点的平方距离 (N,)。"""
        d = self.X @ self.X[v]
        d *= -2.0
        d += self.sq
        d += self.sq[v]
        np.maximum(d, 0.0, out=d)
        if self.xy is not None:
            d *= 1.0 / self.s_A ** 2
            g = self.xy - self.xy[v]
            g = np.einsum('ij,ij->i', g, g) * (1.0 / self.s_G ** 2)
            np.maximum(d, g, out=d)
        return d

    def min_pair(self, S):
        """集合 S 的最小成对距离；#S < 2 时为 0（规格约定）。"""
        S = np.asarray(S, dtype=np.int64)
        if len(S) < 2:
            return 0.0
        dA = pdist(self.X[S])
        if self.xy is None:
            return float(dA.min())
        return float(np.maximum(dA / self.s_A, pdist(self.xy[S]) / self.s_G).min())

    def pair(self, i, j):
        dA = float(np.linalg.norm(self.X[i] - self.X[j]))
        if self.xy is None:
            return dA
        return max(dA / self.s_A, float(np.linalg.norm(self.xy[i] - self.xy[j])) / self.s_G)

    def diameter(self, block=512):
        """精确直径 (d_max, a, b)。欧氏部分分块计算（块大小 512 行以控制内存，规格建议 4096）；
        geo 度量的直径 = max(AE 直径/s_A, 地理直径/s_G)。"""
        dA, a, b = _diam_blocked(self.X, self.sq, block)
        if self.xy is None:
            return dA, a, b
        dG, ag, bg = _diam_2d(self.xy)
        if dA / self.s_A >= dG / self.s_G:
            return dA / self.s_A, a, b
        return dG / self.s_G, ag, bg


def _diam_blocked(X, sq, block=512):
    N = len(X)
    best, a, b = -1.0, 0, min(1, N - 1)
    for s in range(0, N, block):
        e = min(N, s + block)
        G = X[s:e] @ X[s:].T
        G *= -2.0
        G += sq[s:e, None]
        G += sq[None, s:]
        j = int(np.argmax(G))
        r, c = divmod(j, G.shape[1])
        if G[r, c] > best:
            best, a, b = float(G[r, c]), s + r, s + c
    return float(np.linalg.norm(X[a] - X[b])), int(a), int(b)


def _diam_2d(xy):
    """二维点集的精确直径：直径端点必在凸包顶点上，只在顶点间穷举。"""
    cand = np.arange(len(xy))
    if len(xy) >= 3:
        try:
            from scipy.spatial import ConvexHull
            cand = ConvexHull(xy).vertices
        except Exception:  # 共线等退化情形：退回全部点
            pass
    P = xy[cand]
    if len(P) > 4000:   # 只在退化且点很多时发生
        d, a, b = _diam_blocked(P, np.einsum('ij,ij->i', P, P))
    else:
        D = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        a, b = np.unravel_index(int(np.argmax(D)), D.shape)
    ia, ib = int(cand[a]), int(cand[b])
    return float(np.linalg.norm(xy[ia] - xy[ib])), ia, ib


def _greedy_threshold(order_list, metric, k, d2):
    """GreedyIndependentSet（线性效用 = 按 u 降序扫描）：到已选点的最小距离 ≥ d 才接受。"""
    mind = np.full(metric.N, np.inf)
    S = []
    for v in order_list:
        if mind[v] >= d2:
            S.append(v)
            if len(S) == k:
                break
            np.minimum(mind, metric.sq_to_all(v), out=mind)
    return S


def _pad(S, order, k, N):
    """#S < k 时按 order 补入未选点（补入点不受阈值约束）。"""
    S = np.asarray(S, dtype=np.int64)
    if len(S) >= k:
        return S[:k]
    used = np.zeros(N, dtype=bool)
    used[S] = True
    rest = order[~used[order]][:k - len(S)]
    return np.r_[S, rest]


def gist_thresholds(d_max, eps):
    """D = {(1+ε)^i · ε·d_max/2 : (1+ε)^i ≤ 2/ε}（与 gist-select 相同的乘法递推，ε=0.05 时 76 个）。"""
    out, mult = [], 1.0
    while mult <= 2.0 / eps:
        out.append(eps * d_max / 2.0 * mult)
        mult *= 1.0 + eps
    return out


def gist_core(u, metric, k, alpha=0.9, eps=0.05, pad=True, diameter=None, return_info=False):
    """GIST（线性效用）。f(S) = α·Σ_{i∈S} u_i / k + (1−α)·minpair(S)/d_max，#S<2 时 minpair=0。
    候选：纯 top-k（阈值 0）、直径对、每个阈值 d 的贪心独立集；pad=True 时各候选按 u 降序补足到 k 后再算 f。
    返回 f 最大的候选（平局取先出现者）。"""
    u = np.asarray(u, dtype=np.float64)
    N = len(u)
    k = int(min(k, N))
    order = np.argsort(-u, kind='stable')
    order_list = order.tolist()
    d_max, a, b = metric.diameter() if diameter is None else diameter
    d_max = float(d_max)

    def f(S):
        mp = metric.min_pair(S)
        return alpha * u[S].sum() / k + ((1.0 - alpha) * mp / d_max if d_max > 0 else 0.0)

    fin = (lambda S: _pad(S, order, k, N)) if pad else (lambda S: np.asarray(S, dtype=np.int64))
    cands = [('topk', 0.0, k, order[:k].copy())]       # (类型, 阈值 d, 补足前的点数, 集合)
    if k >= 2 and d_max > 0:
        cands.append(('diam', d_max, 2, fin([a, b])))
        for d in gist_thresholds(d_max, eps):
            S = _greedy_threshold(order_list, metric, k, d * d)
            cands.append(('thr', d, len(S), fin(S)))
    vals = np.array([f(c[3]) for c in cands])
    best = int(np.argmax(vals))
    idx = np.asarray(cands[best][3], dtype=np.int64)
    if return_info:
        return idx, {'f': float(vals[best]), 'kind': cands[best][0], 'd': cands[best][1], 'd_max': d_max,
                     'n_greedy': cands[best][2], 'diam_pair': (int(a), int(b)), 'n_cands': len(cands),
                     'f_all': vals, 'greedy_sizes': [c[2] for c in cands]}
    return idx


def knn_scale(X, m, n_query=4000, block=256):
    """目标池第 m 近邻距离（不含自身）的中位数。N 较大时对 n_query 个查询点（固定种子子样本，与 seed 无关）计算，
    被查询集合仍是全池；中位数对子采样不敏感。"""
    X = np.asarray(X, dtype=np.float64)
    N = len(X)
    m = int(min(max(m, 1), N - 1))
    q = np.arange(N) if N <= n_query else np.sort(np.random.default_rng(12345).choice(N, n_query, replace=False))
    sq = np.einsum('ij,ij->i', X, X)
    out = np.empty(len(q))
    for s in range(0, len(q), block):
        qi = q[s:s + block]
        D = X[qi] @ X.T
        D *= -2.0
        D += sq[qi, None]
        D += sq[None, :]
        np.maximum(D, 0.0, out=D)
        out[s:s + block] = np.sqrt(np.partition(D, m, axis=1)[:, m])   # 位置 0 是自身（距离 0）
    return float(np.median(out))


def _gist_metric(ctx, k, metric='euclid', s_G=5.0, space='ae', d=32):
    X = X_of(ctx, space, d)
    if metric == 'euclid':
        return GistMetric(X)
    if metric == 'geo':
        s_A = knn_scale(X, math.ceil(ctx.N / k))
        return GistMetric(X, xy=np.asarray(ctx.T_xy, dtype=np.float64), s_A=s_A, s_G=s_G)
    raise ValueError(metric)


def _gist(ctx, k, seed, alpha=0.9, eps=0.05, metric='euclid', s_G=5.0, space='ae', d=32, pad=True):
    u = unc_margin(np.asarray(ctx.P_T, dtype=np.float64))
    M = _gist_metric(ctx, k, metric=metric, s_G=s_G, space=space, d=d)
    idx = gist_core(u, M, k, alpha=alpha, eps=eps, pad=pad)
    return finalize(idx, k, ctx.N)


@register('gist_margin')
def select_gist_margin(ctx, k, seed, alpha=0.9, eps=0.05, metric='euclid', s_G=5.0, space='ae', d=32):
    """M6 GIST-margin：u = unc_margin(P_T)，X = X_of(ctx,'ae',32)，α=0.9（论文值），ε=0.05。确定性。"""
    return _gist(ctx, k, seed, alpha=alpha, eps=eps, metric=metric, s_G=s_G, space=space, d=d)


@register('gist_maxmin')
def select_gist_maxmin(ctx, k, seed, alpha=0.0, eps=0.05, metric='euclid', s_G=5.0, space='ae', d=32):
    """M6 消融 α=0：纯 max-min（u 只决定贪心扫描顺序）。确定性。"""
    return _gist(ctx, k, seed, alpha=alpha, eps=eps, metric=metric, s_G=s_G, space=space, d=d)


@register('gist_geo')
def select_gist_geo(ctx, k, seed, alpha=0.9, eps=0.05, s_G=5.0, space='ae', d=32):
    """M6 GIST-geo：d = max(‖ΔA‖/s_A, ‖Δxy‖/s_G)，s_A = 目标池第 ceil(N/k) 近邻距离中位数，s_G ∈ {1, 5} km（默认 5）。"""
    return _gist(ctx, k, seed, alpha=alpha, eps=eps, metric='geo', s_G=s_G, space=space, d=d)


def kcenter_core(metric, k, first=None, init_sq=None):
    """Gonzalez 最远点贪心（k-center 2-近似）。first：首点；init_sq：已有中心（如外国池）给出的初始平方距离。"""
    N = metric.N
    S = []
    if init_sq is None:
        S.append(int(first))
        mind = metric.sq_to_all(S[0]).copy()
        mind[S[0]] = 0.0
    else:
        mind = np.asarray(init_sq, dtype=np.float64).copy()
    chosen = np.zeros(N, dtype=bool)
    chosen[S] = True
    while len(S) < k:
        v = int(np.argmax(mind))
        if chosen[v] or mind[v] <= 0:   # 退化：剩余点都与已选点重合
            v = int(np.flatnonzero(~chosen)[0])
        S.append(v)
        chosen[v] = True
        np.minimum(mind, metric.sq_to_all(v), out=mind)
        mind[v] = 0.0
    return np.asarray(S, dtype=np.int64)


def nearest_foreign_sq(XT, XF, block=512):
    """每个目标点到最近外国点的平方距离（分块 BLAS，float32 计算以节省时间，结果转 float64）。"""
    XT = np.asarray(XT, dtype=np.float32)
    XF = np.asarray(XF, dtype=np.float32)
    sqF = np.einsum('ij,ij->i', XF, XF)
    sqT = np.einsum('ij,ij->i', XT, XT)
    out = np.empty(len(XT))
    for s in range(0, len(XT), block):
        G = XT[s:s + block] @ XF.T
        G *= -2.0
        G += sqF[None, :]
        out[s:s + block] = G.min(1) + sqT[s:s + block]
    return np.maximum(out, 0.0)


@register('kcenter')
def select_kcenter(ctx, k, seed, space='ae', d=32):
    """M6 消融 'kcenter'：Sener & Savarese (2018) 的 k-center 贪心；首点随机（seed），之后确定性。"""
    rng = np.random.default_rng(seed)
    M = GistMetric(X_of(ctx, space, d))
    idx = kcenter_core(M, k, first=int(rng.integers(ctx.N)))
    return finalize(idx, k, ctx.N)


@register('kcenter_foreign')
def select_kcenter_foreign(ctx, k, seed, space='ae', d=32):
    """M6 消融 'kcenter_foreign'：mind 初始化为到最近外国点的距离（同一缩放的 X_of_foreign），
    等价于试点的覆盖缺口方法。与 seed 无关（确定性）。"""
    X = X_of(ctx, space, d)
    XF = X_of_foreign(ctx, space, d)
    M = GistMetric(X)
    idx = kcenter_core(M, k, init_sq=nearest_foreign_sq(X, XF))
    return finalize(idx, k, ctx.N)


# =====================================================================
# M7 BADGE 树模型代理
# =====================================================================

def badge_core(P, h, k, rng=None, origin_center=False, chooser=None, init_idx=None):
    """在幻觉梯度 g_x = (p(x) − e_ŷ)⊗h(x) 上做 k-means++ 播种（Ash et al., ICLR 2020）。

    ‖g_i − g_j‖² = ‖a_i‖²‖h_i‖² + ‖a_j‖²‖h_j‖² − 2(a_i·a_j)(h_i·h_j)，无需构造 C·d 维向量。
    - 首点取 argmax‖g‖²（与 scikit-activeml 一致），之后按到最近已选点的距离平方成比例抽样。
    - origin_center=True：把原点也当作一个中心（scikit-activeml 1.0.0 Badge 的实际行为：D² 初值为 ‖g‖²），
      只用于交叉核对；默认 False 与原论文代码及规格伪代码一致。
    - init_idx：已选点（如 M8 第一批），作为初始中心，此时不再取 argmax 首点；返回值不含 init_idx。
    - chooser(p) -> 下标：默认 rng.choice(N, p=p)。
    """
    P = np.asarray(P, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    N = len(P)
    a = P.copy()
    a[np.arange(N), P.argmax(1)] -= 1.0
    na = np.einsum('ij,ij->i', a, a)
    nh = np.einsum('ij,ij->i', h, h)
    g2 = na * nh

    def d2(i):
        return np.maximum(g2 + g2[i] - 2.0 * (a @ a[i]) * (h @ h[i]), 0.0)

    if chooser is None:
        if rng is None:
            raise ValueError('需要 rng 或 chooser')
        chooser = lambda p: int(rng.choice(len(p), p=p))  # noqa: E731
    chosen = np.zeros(N, dtype=bool)
    S = []
    if init_idx is not None and len(init_idx) > 0:
        init_idx = np.asarray(init_idx, dtype=np.int64)
        chosen[init_idx] = True
        mind = g2.copy() if origin_center else np.full(N, np.inf)
        for i in init_idx:
            np.minimum(mind, d2(int(i)), out=mind)
        mind[chosen] = 0.0
    else:
        first = int(np.argmax(g2))
        S.append(first)
        chosen[first] = True
        mind = d2(first)
        if origin_center:
            np.minimum(mind, g2, out=mind)
        mind[first] = 0.0
    while len(S) < k:
        tot = mind.sum()
        if not np.isfinite(tot) or tot <= 0:   # 退化：剩余点都与中心重合，在未选点中均匀抽
            rest = np.flatnonzero(~chosen)
            p = np.zeros(N)
            p[rest] = 1.0 / len(rest)
        else:
            p = mind / tot
        i = int(chooser(p))
        if chosen[i]:   # 只可能出现在浮点误差下；退回未选点中概率最大者
            i = int(np.flatnonzero(~chosen)[np.argmax(mind[~chosen])])
        S.append(i)
        chosen[i] = True
        np.minimum(mind, d2(i), out=mind)
        mind[i] = 0.0
    return np.asarray(S, dtype=np.int64)


def _badge_h(ctx, space):
    if space == 'ae32':
        return X_of(ctx, 'ae', 32)
    if space == 'ae+s2':
        return X_of(ctx, 'ae+s2', 16)
    raise ValueError(space)


@register('badge')
def select_badge(ctx, k, seed, space='ae32'):
    """M7 BADGE 代理：P = P_T，h = X_of(ctx,'ae',32)（space='ae+s2' 时 AE16 + S2_16）。"""
    rng = np.random.default_rng(seed)
    idx = badge_core(ctx.P_T, _badge_h(ctx, space), k, rng=rng)
    return finalize(idx, k, ctx.N)


@register('badge_ae_s2')
def select_badge_ae_s2(ctx, k, seed, space='ae+s2'):
    """M7 变体：h = AE16 + S2_16。"""
    return select_badge(ctx, k, seed, space=space)


# =====================================================================
# M16 k-DPP
# =====================================================================

def _kdpp_phase1_log(lam, k, rs):
    """k-DPP 阶段 1（Kulesza & Taskar 2012 算法 8）：按初等对称多项式选特征向量。
    与 dppy.exact_sampling.k_dpp_eig_vecs_selector 逐步相同（同样的 rs.rand() 消耗顺序），
    但在对数域递推 e_j，避免 k 较大时 e_j 上溢（dppy 在 k=1000、D=2000 时溢出为 inf/nan）。"""
    lam = np.asarray(lam, dtype=np.float64)
    D = lam.size
    with np.errstate(divide='ignore'):
        ll = np.log(lam)
    logE = np.full((k + 1, D + 1), -np.inf)
    logE[0, :] = 0.0
    for n in range(1, D + 1):
        logE[1:, n] = np.logaddexp(logE[1:, n - 1], ll[n - 1] + logE[:-1, n - 1])
    sel = np.zeros(k, dtype=np.int64)
    kk = k
    for n in range(D, 0, -1):
        num = ll[n - 1] + logE[kk - 1, n - 1]      # log(λ_n · e_{kk−1}(λ_1..λ_{n−1}))
        # num 有限时 logE[kk, n] ≥ num 也有限；num = −inf（零特征值或不可能状态）时概率为 0
        p = 0.0 if num == -np.inf else math.exp(min(0.0, num - logE[kk, n]))
        if rs.rand() < p:
            kk -= 1
            sel[kk] = n - 1
            if kk == 0:
                break
    if kk != 0:
        raise RuntimeError('k-DPP 阶段 1 未选满 k 个特征向量（秩不足）')
    return sel


def _proj_gs_fallback(V, k, rs):
    """投影 DPP 的 Gram–Schmidt 链式采样（与 dppy.proj_dpp_sampler_eig_GS 相同，只是每步把概率重新归一化）。
    只在 dppy 不可用或其概率和校验因浮点误差失败时使用。"""
    N = V.shape[0]
    avail = np.ones(N, dtype=bool)
    c = np.zeros((N, k))
    norms = np.einsum('ij,ij->i', V, V)
    out = np.zeros(k, dtype=np.int64)
    ground = np.arange(N)
    for it in range(k):
        p = np.abs(norms[avail])
        j = int(rs.choice(ground[avail], p=p / p.sum()))
        out[it] = j
        if it == k - 1:
            break
        avail[j] = False
        c[avail, it] = (V[avail] @ V[j] - c[avail, :it] @ c[j, :it]) / math.sqrt(norms[j])
        norms[avail] -= c[avail, it] ** 2
    return out


def kdpp_sample(Phi, k, random_state, engine='stable'):
    """从 k-DPP（L = PhiᵀPhi，Phi 形状 (D, n)）精确采样 k 个下标。

    engine='stable'（默认）：对偶核 PhiPhiᵀ（D<n 时）或原始核的特征分解 → 对数域阶段 1 → 选中的特征向量
      映射回原始空间并 QR 重新正交化（投影 DPP 只依赖张成的子空间，分布不变）→ 阶段 2 调用 dppy 的
      proj_dpp_sampler_eig_GS。与 dppy 的 sample_exact_k_dpp 在同一 RandomState 下给出相同样本（测试核对）。
    engine='dppy'：直接调用 dppy FiniteDPP('likelihood', L_gram_factor=Phi).sample_exact_k_dpp(size=k, mode='GS')；
      k 较大时其初等对称多项式会上溢，只用于小规模核对。
    """
    rs = random_state if isinstance(random_state, np.random.RandomState) else np.random.RandomState(random_state)
    Phi = np.asarray(Phi, dtype=np.float64)
    D, n = Phi.shape
    if engine == 'dppy':
        from dppy.finite_dpps import FiniteDPP
        with contextlib.redirect_stdout(io.StringIO()):
            dpp = FiniteDPP('likelihood', **{'L_gram_factor': Phi})
            s = dpp.sample_exact_k_dpp(size=k, mode='GS', random_state=rs)
        return np.asarray(s, dtype=np.int64)
    if engine != 'stable':
        raise ValueError(engine)
    if D < n:
        lam, W = np.linalg.eigh(Phi @ Phi.T)
    else:
        lam, W = np.linalg.eigh(Phi.T @ Phi)
    lam = np.clip(lam, 0.0, None)
    tol = lam.max() * n * np.finfo(float).eps       # 与 dppy 相同的数值秩判据
    rank = int(np.count_nonzero(lam > tol))
    if k > rank:
        raise ValueError(f'k={k} 大于 L 的数值秩 {rank}；增大 D 或减小 k')
    lam = np.where(lam > tol, lam, 0.0)
    sel = _kdpp_phase1_log(lam, k, rs)
    if D < n:
        V = Phi.T @ (W[:, sel] / np.sqrt(lam[sel]))
    else:
        V = W[:, sel]
    V, _ = np.linalg.qr(V)
    try:
        from dppy.exact_sampling import proj_dpp_sampler_eig_GS
    except ImportError:
        warnings.warn('dppy 不可用，阶段 2 改用本文件的 Gram–Schmidt 实现')
        return _proj_gs_fallback(V, k, rs)
    state = rs.get_state()
    try:
        return np.asarray(proj_dpp_sampler_eig_GS(V, size=k, random_state=rs), dtype=np.int64)
    except ValueError:   # 概率和校验因浮点误差失败：从同一随机状态重来，改用逐步归一化的版本
        rs.set_state(state)
        return _proj_gs_fallback(V, k, rs)


def rff_features(X, D, sigma, rng):
    """高斯核 exp(−‖x−y‖²/(2σ²)) 的随机傅里叶特征，返回 (n, D)。"""
    W = rng.normal(0.0, 1.0 / sigma, size=(X.shape[1], D))
    b = rng.uniform(0.0, 2 * np.pi, D)
    return np.sqrt(2.0 / D) * np.cos(X @ W + b)


def _kdpp(ctx, k, seed, gamma=0.5, d=16, D=None, max_cand=10000, subsample_min_k=1000, n_sigma_pairs=5000,
          engine='stable'):
    rng = np.random.default_rng(seed)
    N = ctx.N
    X = X_of(ctx, 'ae', d)
    # 规格："k = 1000 时先把候选子采样到 1 万（D = 2,000）"，否则 N·k² 的 GS 采样与 N·D² 的对偶核超出时间预算
    if k >= subsample_min_k and N > max_cand:
        cand = np.sort(rng.choice(N, max_cand, replace=False))
    else:
        cand = np.arange(N)
    Xc = X[cand]
    n = len(cand)
    if gamma > 0:
        q = unc_margin(np.asarray(ctx.P_T, dtype=np.float64)[cand]) ** gamma
    else:
        q = np.ones(n)
    i = rng.integers(0, n, n_sigma_pairs)
    j = rng.integers(0, n, n_sigma_pairs)
    keep = i != j
    sigma = float(np.median(np.linalg.norm(Xc[i[keep]] - Xc[j[keep]], axis=1)))
    D = int(D) if D is not None else max(256, 2 * k)
    Phi = (rff_features(Xc, D, sigma, rng) * q[:, None]).T       # (D, n)，L = PhiᵀPhi
    loc = kdpp_sample(Phi, k, np.random.RandomState(seed), engine=engine)
    return finalize(cand[loc], k, N)


@register('kdpp')
def select_kdpp(ctx, k, seed, gamma=0.5, d=16, D=None, max_cand=10000, subsample_min_k=1000,
                n_sigma_pairs=5000, engine='stable'):
    """M16 k-DPP：X = X_of(ctx,'ae',16)，q = margin^γ（γ=0.5），RFF 维数 D = max(256, 2k)，
    σ = 5,000 对随机距离的中位数。k ≥ 1000 且 N > 1 万时候选均匀子采样到 1 万。没有闭式 π，不写 meta。"""
    return _kdpp(ctx, k, seed, gamma=gamma, d=d, D=D, max_cand=max_cand, subsample_min_k=subsample_min_k,
                 n_sigma_pairs=n_sigma_pairs, engine=engine)


@register('kdpp_div')
def select_kdpp_div(ctx, k, seed, gamma=0.0, d=16, D=None, max_cand=10000, subsample_min_k=1000,
                    n_sigma_pairs=5000, engine='stable'):
    """M16 γ=0：纯多样性 k-DPP（不需要外国模型）。"""
    return _kdpp(ctx, k, seed, gamma=gamma, d=d, D=D, max_cand=max_cand, subsample_min_k=subsample_min_k,
                 n_sigma_pairs=n_sigma_pairs, engine=engine)
