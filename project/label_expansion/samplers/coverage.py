"""
覆盖 / 代表性 / 不确定性加权覆盖族的选样方法（sampling_shortlist.json 中的 M10、M11、M12、M14、M15、M17、M20）。

注册名（REGISTRY 键）：
- M10 CLUE：'clue'（AE32，默认）、'clue_aes2'（AE32 + S2 32）
- M11 UHerding：'uherding'（U = margin，默认）、'maxherding'（U ≡ 1）、'uherding_foreign'（条件于外国池）
- M12 TypiClust：'typiclust'（plain，默认）、'typiclust_strat'（按外国预测类分层）
- M14 FASS-quota：'fass_quota'（quota='sqrt'，默认）、'fass_quota_prior'、'fass_quota_equal'、'fass_quota_prior_est'
- M15 ProbCover：'probcover'（伪标签 = 外国预测类，默认）、'probcover_km'（伪标签 = KMeans(C) 簇，原文做法）
- M17 Kernel herding：'kherding'（匹配目标池，默认）、'kherding_foreign'（条件于外国池）
- M20 FSCS：'fscs'（AE32）、'fscs_geo'（AE32 + 2 倍权重的标准化坐标）

约定：
- 这些方法都不是概率设计，不写 ctx.meta[name]；诊断信息（σ、δ*、配额等）写入本模块的 DIAG[name]，不占用 ctx.meta。
- 都不读取目标池标签，也不调用 ctx.oracle。
- 本文件额外定义的公共函数：alloc（按 M3 规格的最大余数分配，含溢出再分配；M12 'strat' 与 M14 共用）、
  fass_quota_plan（只计算 M14 的候选集与各类配额，不选样，供检查“类别构成”用）。
"""
import heapq
import math

import numpy as np
from scipy.spatial import cKDTree
from scipy import sparse
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from .base import register, X_of, X_of_foreign, unc_margin, finalize

DIAG = {}   # 各方法最近一次调用的诊断信息（非概率设计，不写 ctx.meta）


# ---------------------------------------------------------------------------
# 公共小工具
# ---------------------------------------------------------------------------

def _random_pair_dists(X, n_pairs, rng):
    """X 中 n_pairs 个随机点对（i ≠ j）的欧氏距离。"""
    n = len(X)
    if n < 2:
        return np.zeros(1)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n - 1, n_pairs)
    j[j >= i] += 1
    return np.sqrt(((X[i] - X[j]) ** 2).sum(1))


def _sqdist_to(X, sqn, i):
    """X 中所有点到 X[i] 的平方距离（非负截断）。"""
    return np.maximum(sqn + sqn[i] - 2.0 * (X @ X[i]), 0.0)


def _nearest_unselected(X, centers, q=10):
    """按簇心顺序，为每个簇心取距离最近且尚未入选的点（cKDTree，先查 q 个，不够再扩大）。"""
    N = len(X)
    tree = cKDTree(X)
    qq = min(q, N)
    _, NB = tree.query(centers, k=qq)
    NB = NB.reshape(len(centers), -1)
    taken = np.zeros(N, bool)
    S = []
    for r, mu in enumerate(centers):
        nb = NB[r]
        free = nb[~taken[nb]]
        m = qq
        while len(free) == 0 and m < N:        # 前 q 个都已入选时扩大查询
            m = min(N, m * 4)
            _, nb = tree.query(mu, k=m)
            nb = np.atleast_1d(nb)
            free = nb[~taken[nb]]
        j = int(free[0])
        taken[j] = True
        S.append(j)
    return np.asarray(S, dtype=np.int64)


def _lazy_greedy(n_sel, init_gain, gain_fn, commit_fn):
    """单调子模函数的懒惰贪心（Minoux）。
    堆中存各元素增益上界；弹出堆顶并重算，若仍 ≥ 次大上界则选中，否则放回。
    返回所选元素（按入选顺序）与各自的实际增益。"""
    heap = [(-float(g), int(j)) for j, g in enumerate(init_gain)]
    heapq.heapify(heap)
    S, G = [], []
    while len(S) < n_sel and heap:
        _, j = heapq.heappop(heap)
        g = float(gain_fn(j))
        if not heap or g >= -heap[0][0]:
            S.append(j)
            G.append(g)
            commit_fn(j)
        else:
            heapq.heappush(heap, (-g, j))
    return S, G


def alloc(N_h, w_h, k, m_min=1):
    """按 M3 规格分配各层名额（最大余数法，含截断与溢出再分配）。

    - 先给每个非空层 min(N_h, m_min) 个；若总数超过 k，令 m_min = 1 重来；仍超过（非空层多于 k）则 m_min = 0，
      全部按权重分配（与 designs.py 中 M3 的 alloc 一致，测试里逐一比对）。
    - 循环：rest = k − Σn；在 n < N_h 的层上按 w_h 比例用最大余数法分配 rest；
      超过 N_h 的截断，溢出量在下一轮分给其余未满层，直到无溢出。
    - 未满层的权重全为 0 时在这些层上均分（补充处理）。
    """
    N_h = np.asarray(N_h, dtype=np.int64)
    w = np.nan_to_num(np.asarray(w_h, dtype=np.float64), nan=0.0)
    w = np.maximum(w, 0.0)
    k = int(k)
    if N_h.sum() < k:
        raise ValueError(f'各层容量之和 {N_h.sum()} 小于 k={k}')
    nonempty = N_h > 0
    for m in (int(m_min), 1, 0):
        if m > m_min:
            continue
        n = np.where(nonempty, np.minimum(N_h, m), 0).astype(np.int64)
        if n.sum() <= k:
            break
    while True:
        rest = k - int(n.sum())
        if rest <= 0:
            break
        active = n < N_h
        wa = np.where(active, w, 0.0)
        if wa.sum() <= 0:
            wa = active.astype(np.float64)
        q = rest * wa / wa.sum()
        add = np.floor(q).astype(np.int64)
        r = rest - int(add.sum())
        if r > 0:
            frac = np.where(active, q - add, -1.0)
            add[np.argsort(-frac, kind='stable')[:r]] += 1   # 余数最大的 r 个层各加 1，并列按下标
        n = np.minimum(n + add, N_h)                          # 截断；溢出在下一轮再分
    return n


def _softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


# ---------------------------------------------------------------------------
# M10 CLUE 树模型版：预测熵加权 k-means
# ---------------------------------------------------------------------------

def _clue(ctx, k, seed, name, T=1.0, space='ae', d=32, n_init=None, max_iter=100, minibatch=False):
    X = X_of(ctx, space, d)
    PT = _softmax(np.log(ctx.P_T + 1e-12) / T)              # 温度作用在 log P_T 上
    H = -(PT * np.log(PT + 1e-12)).sum(1) + 1e-6
    if n_init is None:
        n_init = 3 if k <= 200 else 1
    if minibatch:   # 规格允许 k = 1000 时改用 MiniBatchKMeans（默认不用：实测 KMeans 在 N=3 万、k=1000 时只需数秒）
        km = MiniBatchKMeans(n_clusters=k, batch_size=4096, n_init=n_init, max_iter=max_iter,
                             random_state=int(seed)).fit(X, sample_weight=H)
    else:
        km = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=int(seed)).fit(X, sample_weight=H)
    S = _nearest_unselected(X, km.cluster_centers_, q=10)
    DIAG[name] = {'T': T, 'space': space, 'n_iter': int(km.n_iter_), 'mean_H_sel': float(H[S].mean()),
                  'mean_H_pool': float(H.mean())}
    return finalize(S, k, ctx.N)


@register('clue')
def select_clue(ctx, k, seed, T=1.0, d=32, n_init=None, max_iter=100, minibatch=False):
    """M10 CLUE（AE32）。T ∈ {0.5, 1, 2}；n_init 默认 k ≤ 200 取 3，否则 1。"""
    return _clue(ctx, k, seed, 'clue', T=T, space='ae', d=d, n_init=n_init, max_iter=max_iter, minibatch=minibatch)


@register('clue_aes2')
def select_clue_aes2(ctx, k, seed, T=1.0, d=32, n_init=None, max_iter=100, minibatch=False):
    """M10 CLUE 变体：特征为 AE32 与 S2 32 拼接。"""
    return _clue(ctx, k, seed, 'clue_aes2', T=T, space='ae+s2', d=d, n_init=n_init, max_iter=max_iter,
                 minibatch=minibatch)


# ---------------------------------------------------------------------------
# M11 UHerding / MaxHerding / 条件于外国池：稀疏近邻 + 懒惰贪心的加权设施选址
# ---------------------------------------------------------------------------

def _uherding(ctx, k, seed, name, variant='uherding', space='ae', d=32, nu=1.0, sigma=None,
              n_rand=None, n_top=None, M_nn=None, kernel_floor=1e-3, n_sigma_probe=2000):
    rng = np.random.default_rng(seed)
    N = ctx.N
    X = X_of(ctx, space, d)
    m = int(math.ceil(N / k))
    if M_nn is None:
        M_nn = min(N, max(200, 3 * m))
    M_nn = int(min(M_nn, N))
    nn = NearestNeighbors(n_neighbors=M_nn).fit(X)
    # σ：2,000 个随机点到第 m 近邻距离的中位数（第 0 个近邻是自身，故取第 m 列）
    if sigma is None:
        probe = rng.choice(N, min(n_sigma_probe, N), replace=False)
        mm = min(m, N - 1)
        dm = nn.kneighbors(X[probe], n_neighbors=mm + 1)[0][:, mm]
        sigma = float(np.median(dm))
        if sigma <= 0:
            sigma = float(np.median(_random_pair_dists(X, 5000, rng))) or 1.0
    # 不确定性权重
    U = np.ones(N) if variant == 'maxherding' else unc_margin(ctx.P_T).astype(np.float64)
    # 候选集：随机 n_rand ∪ U 最高的 n_top（k > 200 时 8,000 ∪ 3,000）
    if n_rand is None:
        n_rand = 5000 if k <= 200 else 8000
    if n_top is None:
        n_top = 2000 if k <= 200 else 3000
    if variant == 'maxherding':
        # U ≡ 1 时“U 最高”全部并列，按下标取会引入偏差；改为同样总数的随机候选（我们的处理）
        cand = rng.choice(N, min(N, n_rand + n_top), replace=False)
    else:
        rnd = rng.choice(N, min(N, n_rand), replace=False)
        top = np.argsort(-U, kind='stable')[:min(N, n_top)]
        cand = np.union1d(rnd, top)
    if len(cand) < k:   # 候选不足 k 个时随机补足
        rest = np.setdiff1d(np.arange(N), cand)
        cand = np.union1d(cand, rng.choice(rest, k - len(cand), replace=False))
    cand = np.sort(cand)
    dist, NB = nn.kneighbors(X[cand], n_neighbors=M_nn)
    Kv = np.exp(-(dist ** 2) / sigma ** 2)
    Kv[Kv < kernel_floor] = 0.0                               # 丢弃 < 1e-3 的项
    NB = NB.astype(np.int64)
    del dist
    # 初始覆盖
    cov = np.zeros(N)
    if variant == 'cond_foreign':
        F = X_of_foreign(ctx, space, d)
        # 核随距离单调递减，max over 10 个外国近邻 = 最近外国点的核值
        dF = NearestNeighbors(n_neighbors=1).fit(F).kneighbors(X, n_neighbors=1)[0][:, 0]
        cov = nu * np.exp(-(dF ** 2) / sigma ** 2)
    cov0 = cov.copy()
    UN = U[NB]                                                # (n_cand, M_nn)

    def gain_fn(c):
        return (UN[c] * np.maximum(Kv[c] - cov[NB[c]], 0.0)).sum() / N

    def commit_fn(c):
        nb = NB[c]
        cov[nb] = np.maximum(cov[nb], Kv[c])

    init = np.empty(len(cand))
    for s in range(0, len(cand), 1024):   # 分块计算初始增益，控制临时数组内存
        init[s:s + 1024] = (UN[s:s + 1024] * np.maximum(Kv[s:s + 1024] - cov[NB[s:s + 1024]], 0.0)).sum(1) / N
    sel, gains = _lazy_greedy(k, init, gain_fn, commit_fn)
    S = cand[np.asarray(sel, dtype=np.int64)]
    DIAG[name] = {'variant': variant, 'sigma': sigma, 'M_nn': M_nn, 'n_cand': int(len(cand)), 'nu': nu,
                  'objective_gain': float(np.sum(gains)),
                  'UC_trunc': float((U * cov).sum() / N), 'UC_foreign_init': float((U * cov0).sum() / N)}
    return finalize(S, k, N)


@register('uherding')
def select_uherding(ctx, k, seed, sigma=None, n_rand=None, n_top=None, M_nn=None, d=32):
    """M11 UHerding：U = unc_margin(P_T)，核 exp(−‖Δ‖²/σ²)。"""
    return _uherding(ctx, k, seed, 'uherding', variant='uherding', sigma=sigma, n_rand=n_rand, n_top=n_top,
                     M_nn=M_nn, d=d)


@register('maxherding')
def select_maxherding(ctx, k, seed, sigma=None, n_rand=None, n_top=None, M_nn=None, d=32):
    """M11 变体 MaxHerding：U ≡ 1（广义覆盖）。"""
    return _uherding(ctx, k, seed, 'maxherding', variant='maxherding', sigma=sigma, n_rand=n_rand, n_top=n_top,
                     M_nn=M_nn, d=d)


@register('uherding_foreign')
def select_uherding_foreign(ctx, k, seed, nu=1.0, sigma=None, n_rand=None, n_top=None, M_nn=None, d=32):
    """M11 变体 cond_foreign：覆盖初值 = ν·max_f k(x_n, x_f)（外国池同一空间），ν ∈ {0.5, 1}。"""
    return _uherding(ctx, k, seed, 'uherding_foreign', variant='cond_foreign', nu=nu, sigma=sigma, n_rand=n_rand,
                     n_top=n_top, M_nn=M_nn, d=d)


# ---------------------------------------------------------------------------
# M12 TypiClust
# ---------------------------------------------------------------------------

def _typicality(Xc, K_nn=20):
    """簇内典型性：1 /（m 个近邻的平均距离），m = min(K_nn, n − 1)；与 scikit-activeml 相同，
    取 m + 1 个近邻（含自身，距离 0）的距离和再除以 m，对重复点稳健。"""
    n = len(Xc)
    if n == 1:
        return np.ones(1)
    m = min(K_nn, n - 1)
    if n <= 2000:
        sq = (Xc ** 2).sum(1)
        D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * Xc @ Xc.T, 0))
        np.fill_diagonal(D, 0.0)
        Dm = np.partition(D, m, axis=1)[:, :m + 1]
    else:
        Dm = NearestNeighbors(n_neighbors=m + 1).fit(Xc).kneighbors(Xc)[0]
    return 1.0 / (Dm.sum(1) / m + 1e-7)


def _typiclust_core(X, k, seed, K_nn=20, max_clusters=None, min_cluster=5):
    """返回 X 内的 k 个下标（相对 X）。"""
    n = len(X)
    Kc = int(min(k, max_clusters if max_clusters is not None else k, n))
    if Kc <= 1:
        labels = np.zeros(n, dtype=np.int64)
        Kc = 1
    elif Kc <= 50:   # K ≤ 50 用 KMeans，否则用 MiniBatchKMeans，与论文一致
        labels = KMeans(n_clusters=Kc, random_state=int(seed)).fit_predict(X)
    else:
        labels = MiniBatchKMeans(n_clusters=Kc, random_state=int(seed)).fit_predict(X)
    sizes = np.bincount(labels, minlength=Kc)
    valid = sizes >= min_cluster
    remaining = sizes.copy()
    cnt = np.zeros(Kc, dtype=np.int64)
    order = np.argsort(labels, kind='stable')
    starts = np.r_[0, np.cumsum(sizes)]
    members = [order[starts[c]:starts[c + 1]] for c in range(Kc)]
    taken = np.zeros(n, bool)
    S = []
    while len(S) < k:
        elig = valid & (remaining > 0)
        if not elig.any():             # 有效簇（≥ 5 个点）用尽后才允许小簇（我们的补充）
            elig = remaining > 0
            if not elig.any():
                break
        cmin = cnt[elig].min()
        cc = np.flatnonzero(elig & (cnt == cmin))
        c = cc[np.argmax(sizes[cc])]   # cnt 最小的簇中样本最多者；并列取下标小者
        idx = members[c][~taken[members[c]]]
        typ = _typicality(X[idx], K_nn)
        i = int(idx[np.argmax(typ)])
        S.append(i)
        taken[i] = True
        cnt[c] += 1
        remaining[c] -= 1
    return np.asarray(S, dtype=np.int64), {'Kc': Kc, 'n_valid': int(valid.sum())}


@register('typiclust')
def select_typiclust(ctx, k, seed, K_nn=20, max_clusters=None, min_cluster=5, d=32):
    """M12 TypiClust（plain）：max_clusters 默认 = k（L = ∅）。"""
    X = X_of(ctx, 'ae', d)
    S, info = _typiclust_core(X, k, seed, K_nn=K_nn, max_clusters=max_clusters, min_cluster=min_cluster)
    DIAG['typiclust'] = info
    return finalize(S, k, ctx.N)


@register('typiclust_strat')
def select_typiclust_strat(ctx, k, seed, K_nn=20, min_cluster=5, m_min=2, weights='sqrt', d=32):
    """M12 TypiClust 'strat'：在每个外国预测类内分别运行，名额用 M3 的 alloc('sqrt')（m_min = 2，同 M3 默认）。"""
    X = X_of(ctx, 'ae', d)
    yhat = ctx.P_T.argmax(1)
    N_h = np.bincount(yhat, minlength=ctx.C)
    w = {'sqrt': np.sqrt(N_h), 'prop': N_h.astype(float), 'equal': (N_h > 0).astype(float)}[weights]
    n_h = alloc(N_h, w, k, m_min)
    seeds = np.random.default_rng(seed).integers(0, 2 ** 31 - 1, size=ctx.C)
    S = []
    for h in range(ctx.C):
        if n_h[h] == 0:
            continue
        idx_h = np.flatnonzero(yhat == h)
        s, _ = _typiclust_core(X[idx_h], int(n_h[h]), int(seeds[h]), K_nn=K_nn, min_cluster=min_cluster)
        S.append(idx_h[s])
    DIAG['typiclust_strat'] = {'n_h': n_h, 'N_h': N_h}
    return finalize(np.concatenate(S), k, ctx.N)


# ---------------------------------------------------------------------------
# M14 FASS-quota：不确定性过滤 + 按预测类设施选址 + 类别配额
# ---------------------------------------------------------------------------

def fass_quota_plan(ctx, k, quota='sqrt', beta=None, m_min=1, p_hat=None, quota_base='pool'):
    """只计算 M14 的候选集、假设标签与各类配额（不选样）。

    quota：'sqrt'(默认) | 'prior' | 'equal' | 'prior_est'
    quota_base：权重所基于的类计数。'pool'(默认) = 外国模型在整个目标池上的预测类计数
      （'prior' 即“外国模型预测类比例”，'sqrt' 与 M3 的 w_h = sqrt(N_h) 一致）；'cand' = 候选集内的计数。
    容量（alloc 的 N_h）始终是各类候选数 #U_c。
    'prior_est' 需要 p_hat（长度 C）；未给出时取 ctx.meta['p_hat_T']，都没有则报错。
    """
    N, C = ctx.N, ctx.C
    if beta is None:
        beta = min(N, max(3000, 10 * k))
    beta = int(min(N, beta))
    u = unc_margin(ctx.P_T)
    cand = np.argsort(-u, kind='stable')[:beta]
    yh = ctx.P_T[cand].argmax(1)
    cap = np.bincount(yh, minlength=C)
    if quota_base == 'pool':
        base = np.bincount(ctx.P_T.argmax(1), minlength=C).astype(np.float64)
    elif quota_base == 'cand':
        base = cap.astype(np.float64)
    else:
        raise ValueError(quota_base)
    if quota == 'sqrt':
        w = np.sqrt(base)
    elif quota == 'prior':
        w = base
    elif quota == 'equal':
        w = np.ones(C)
    elif quota == 'prior_est':
        if p_hat is None:
            p_hat = ctx.meta.get('p_hat_T') if isinstance(ctx.meta, dict) else None
        if p_hat is None:
            raise ValueError("quota='prior_est' 需要 p_hat（M3/M8 的目标先验估计），可用 p_hat=... 或 ctx.meta['p_hat_T'] 提供")
        w = np.asarray(p_hat, dtype=np.float64)
        assert w.shape == (C,)
    else:
        raise ValueError(quota)
    n_c = alloc(cap, w, k, m_min)
    return {'cand': cand, 'yh': yh, 'cap': cap, 'w': w, 'n_c': n_c, 'beta': beta}


def _fl_greedy_class(Xc, n_sel, rng, dense_max=5000, knn=50, n_pairs=2000, s_c=None):
    """类内设施选址贪心：f(S) = Σ_i max_{s∈S} W(i, s)，W = exp(−‖Δ‖²/(2 s_c²))。
    #U_c ≤ dense_max 用稠密矩阵，否则用 knn-近邻稀疏矩阵（i ∈ NN_knn(j) 时 W_ij 非零）。
    用懒惰贪心求与朴素贪心相同的解（增益不增，懒惰贪心合法）。返回相对下标与 s_c。"""
    n = len(Xc)
    if n_sel >= n:
        return np.arange(n), float('nan')   # 名额 ≥ 候选数：全取（alloc 保证不会超过）
    if s_c is None:
        s_c = float(np.median(_random_pair_dists(Xc, n_pairs, rng)))
    if not s_c > 0:
        s_c = 1.0
    cov = np.zeros(n)
    if n <= dense_max:
        sq = (Xc ** 2).sum(1)
        W = np.exp(-np.maximum(sq[:, None] + sq[None, :] - 2 * Xc @ Xc.T, 0) / (2 * s_c ** 2))
        np.fill_diagonal(W, 1.0)
        init = W.sum(0)

        def gain_fn(j):
            return np.maximum(W[j] - cov, 0.0).sum()     # W 对称，第 j 行即第 j 列

        def commit_fn(j):
            np.maximum(cov, W[j], out=cov)
    else:
        kk = min(knn, n)
        dist, NB = NearestNeighbors(n_neighbors=kk).fit(Xc).kneighbors(Xc)
        Kv = np.exp(-(dist ** 2) / (2 * s_c ** 2))
        init = Kv.sum(1)

        def gain_fn(j):
            return np.maximum(Kv[j] - cov[NB[j]], 0.0).sum()

        def commit_fn(j):
            nb = NB[j]
            cov[nb] = np.maximum(cov[nb], Kv[j])
    sel, _ = _lazy_greedy(n_sel, init, gain_fn, commit_fn)
    return np.asarray(sel, dtype=np.int64), s_c


def _fass_quota(ctx, k, seed, name, quota='sqrt', beta=None, m_min=1, p_hat=None, quota_base='pool',
                d=32, dense_max=5000, knn=50):
    rng = np.random.default_rng(seed)
    X = X_of(ctx, 'ae', d)
    plan = fass_quota_plan(ctx, k, quota=quota, beta=beta, m_min=m_min, p_hat=p_hat, quota_base=quota_base)
    cand, yh, n_c = plan['cand'], plan['yh'], plan['n_c']
    S, s_cs = [], np.full(ctx.C, np.nan)
    for c in range(ctx.C):
        if n_c[c] == 0:
            continue
        Uc = cand[yh == c]
        sel, s_cs[c] = _fl_greedy_class(X[Uc], int(n_c[c]), rng, dense_max=dense_max, knn=knn)
        S.append(Uc[sel])
    S = np.concatenate(S)
    DIAG[name] = {**{kk: plan[kk] for kk in ('cap', 'w', 'n_c', 'beta')}, 'quota': quota, 's_c': s_cs,
                  'realized': np.bincount(ctx.P_T[S].argmax(1), minlength=ctx.C)}
    return finalize(S, k, ctx.N)


@register('fass_quota')
def select_fass_quota(ctx, k, seed, quota='sqrt', beta=None, m_min=1, p_hat=None, quota_base='pool', d=32,
                      dense_max=5000, knn=50):
    """M14 FASS-quota（默认 quota='sqrt'）。β 默认 min(N, max(3000, 10k))。"""
    return _fass_quota(ctx, k, seed, 'fass_quota', quota=quota, beta=beta, m_min=m_min, p_hat=p_hat,
                       quota_base=quota_base, d=d, dense_max=dense_max, knn=knn)


@register('fass_quota_prior')
def select_fass_quota_prior(ctx, k, seed, beta=None, m_min=1, quota_base='pool', d=32, dense_max=5000, knn=50):
    """M14 变体：配额按外国模型在目标池上的预测类比例。"""
    return _fass_quota(ctx, k, seed, 'fass_quota_prior', quota='prior', beta=beta, m_min=m_min,
                       quota_base=quota_base, d=d, dense_max=dense_max, knn=knn)


@register('fass_quota_equal')
def select_fass_quota_equal(ctx, k, seed, beta=None, m_min=1, d=32, dense_max=5000, knn=50):
    """M14 变体：各类等额配额（受各类候选数截断，溢出再分配）。"""
    return _fass_quota(ctx, k, seed, 'fass_quota_equal', quota='equal', beta=beta, m_min=m_min, d=d,
                       dense_max=dense_max, knn=knn)


@register('fass_quota_prior_est')
def select_fass_quota_prior_est(ctx, k, seed, p_hat=None, beta=None, m_min=1, d=32, dense_max=5000, knn=50):
    """M14 变体：配额按 M3/M8 估计的目标先验 p̂_T（p_hat=... 或 ctx.meta['p_hat_T']）。"""
    return _fass_quota(ctx, k, seed, 'fass_quota_prior_est', quota='prior_est', beta=beta, m_min=m_min,
                       p_hat=p_hat, d=d, dense_max=dense_max, knn=knn)


# ---------------------------------------------------------------------------
# M15 ProbCover
# ---------------------------------------------------------------------------

def _center_stats(X, pl, centers, grid, chunk=256):
    """对每个中心 c：r_c = 到最近的异伪标签点的距离（球在 δ 下纯 ⇔ r_c > δ），
    cnt[c, g] = #{j : ‖x_c − x_j‖ ≤ grid[g]}（含自身）。与在各 δ 上逐个做 radius_neighbors 等价。"""
    N = len(X)
    G = len(grid)
    sq = (X ** 2).sum(1)
    r = np.empty(len(centers))
    cnt = np.empty((len(centers), G), dtype=np.int64)
    for s in range(0, len(centers), chunk):
        cc = centers[s:s + chunk]
        m = len(cc)
        D = np.sqrt(np.maximum(sq[cc, None] + sq[None, :] - 2 * X[cc] @ X.T, 0))
        D[np.arange(m), cc] = 0.0
        r[s:s + m] = np.where(pl[None, :] != pl[cc, None], D, np.inf).min(1)
        g = np.searchsorted(grid, D, side='left')             # D ≤ grid[g'] ⇔ g' ≥ g
        h = np.bincount((np.arange(m)[:, None] * (G + 1) + g).ravel(), minlength=m * (G + 1)).reshape(m, G + 1)
        cnt[s:s + m] = np.cumsum(h, axis=1)[:, :G]
    return r, cnt


def _radius_graph(X, delta, chunk=4096):
    """δ 球邻接（CSR，行 = 球心，列 = 被覆盖点，含自身）；按块构建以控制内存。"""
    N = len(X)
    nn = NearestNeighbors(radius=delta).fit(X)
    lens, inds = [], []
    for s in range(0, N, chunk):
        res = nn.radius_neighbors(X[s:s + chunk], return_distance=False)
        lens.append(np.fromiter((len(a) for a in res), dtype=np.int64, count=len(res)))
        inds.append(np.concatenate(res).astype(np.int32))
    lens = np.concatenate(lens)
    indptr = np.r_[0, np.cumsum(lens)]
    indices = np.concatenate(inds)
    A = sparse.csr_matrix((np.ones(len(indices), dtype=np.int8), indices, indptr), shape=(N, N))
    return A


def _probcover_greedy(A, k, rng):
    """最大覆盖贪心：每次选出度（未覆盖的被覆盖点数）最大的点，删去新覆盖点的所有入边。
    出度为 0 时剩余名额从未选点中随机补足（我们的处理）。返回所选下标与是否随机补足的个数。"""
    N = A.shape[0]
    Acsc = A.tocsc()
    covered = np.zeros(N, bool)
    chosen = np.zeros(N, bool)
    deg = np.diff(A.indptr).astype(np.int64)
    S = []
    n_fill = 0
    for t in range(k):
        dd = np.where(chosen, -1, deg)
        c = int(np.argmax(dd))
        if dd[c] <= 0:
            rest = np.flatnonzero(~chosen)
            fill = rng.choice(rest, k - len(S), replace=False)
            n_fill = len(fill)
            S.extend(int(x) for x in fill)
            break
        S.append(c)
        chosen[c] = True
        row = A.indices[A.indptr[c]:A.indptr[c + 1]]
        new = row[~covered[row]]
        covered[new] = True
        if len(new):
            ins = Acsc[:, new].indices                        # 覆盖了这些新点的所有球心
            deg -= np.bincount(ins, minlength=N)
    return np.asarray(S, dtype=np.int64), n_fill, covered


def _probcover(ctx, k, seed, name, pseudo='pred', alpha=0.95, d=32, n_pairs=5000, n_centers=3000,
               pct_lo=1, pct_hi=30, deg_cap=3000, delta=None):
    rng = np.random.default_rng(seed)
    N = ctx.N
    X = X_of(ctx, 'ae', d)
    if pseudo == 'pred':
        pl = ctx.P_T.argmax(1)
    elif pseudo == 'kmeans':
        pl = KMeans(n_clusters=ctx.C, random_state=int(seed)).fit_predict(X)
    else:
        raise ValueError(pseudo)
    dp = _random_pair_dists(X, n_pairs, rng)
    grid = np.percentile(dp, np.arange(pct_lo, pct_hi + 1))
    centers = rng.choice(N, min(n_centers, N), replace=False)
    r, cnt = _center_stats(X, pl, centers, grid)
    purity = (r[:, None] > grid[None, :]).mean(0)
    meandeg = cnt.mean(0)
    info = {'grid': grid, 'purity': purity, 'meandeg': meandeg}
    if delta is None:
        ok = np.flatnonzero(purity >= alpha)
        g = int(ok.max()) if len(ok) else 0                   # 无 δ 达到 α 时取网格最小值（记入 DIAG）
        info['purity_met'] = bool(len(ok))
        while g > 0 and meandeg[g] > deg_cap:                 # 平均度超过上限：继续减小 δ
            g -= 1
        delta = float(grid[g])
        if meandeg[g] > deg_cap:                              # 网格最小值仍超限：取使 N·F(δ) ≈ deg_cap 的随机对距离分位数
            delta = float(min(delta, np.quantile(dp, deg_cap / N)))
        info['g'] = g
    A = _radius_graph(X, delta)
    S, n_fill, covered = _probcover_greedy(A, k, rng)
    info.update({'delta': delta, 'n_fill': n_fill, 'coverage': float(covered.mean()),
                 'avg_deg': float(A.nnz / N)})
    DIAG[name] = info
    return finalize(S, k, N)


@register('probcover')
def select_probcover(ctx, k, seed, alpha=0.95, deg_cap=3000, delta=None, d=32):
    """M15 ProbCover，伪标签 = P_T.argmax（我们的默认）。δ* = max{δ : purity ≥ α}，δ 网格 = 随机对距离第 1–30 百分位。"""
    return _probcover(ctx, k, seed, 'probcover', pseudo='pred', alpha=alpha, deg_cap=deg_cap, delta=delta, d=d)


@register('probcover_km')
def select_probcover_km(ctx, k, seed, alpha=0.95, deg_cap=3000, delta=None, d=32):
    """M15 变体：伪标签 = KMeans(C) 簇标签（原文做法）。"""
    return _probcover(ctx, k, seed, 'probcover_km', pseudo='kmeans', alpha=alpha, deg_cap=deg_cap, delta=delta, d=d)


# ---------------------------------------------------------------------------
# M17 Kernel herding（MMD 分布匹配到目标池）
# ---------------------------------------------------------------------------

def _rff_mean_kernel(Xq, Xp, Wr, br, chunk=4096):
    """≈ (1/n_p) Σ_j k(x_q, x_p,j)：先求 φ(Xp) 的均值，再与 φ(Xq) 做内积（分块，控制内存）。"""
    D = Wr.shape[1]
    c = np.sqrt(2.0 / D)
    mp = np.zeros(D)
    for s in range(0, len(Xp), chunk):
        mp += (c * np.cos(Xp[s:s + chunk] @ Wr + br)).sum(0)
    mp /= len(Xp)
    out = np.empty(len(Xq))
    for s in range(0, len(Xq), chunk):
        out[s:s + chunk] = (c * np.cos(Xq[s:s + chunk] @ Wr + br)) @ mp
    return out


def _exact_mean_kernel(Xq, Xp, sigma, chunk=None):
    """精确的 (1/n_p) Σ_j exp(−‖x_q − x_p,j‖²/σ²)（O(n_q·n_p·d)，分块使每块约 2×10^7 个元素）。"""
    if chunk is None:
        chunk = max(64, int(2e7 // max(1, len(Xp))))
    sq_p = (Xp ** 2).sum(1)
    out = np.empty(len(Xq))
    for s in range(0, len(Xq), chunk):
        Q = Xq[s:s + chunk]
        D2 = np.maximum((Q ** 2).sum(1)[:, None] + sq_p[None, :] - 2 * Q @ Xp.T, 0)
        out[s:s + chunk] = np.exp(-D2 / sigma ** 2).mean(1)
    return out


def _kherding(ctx, k, seed, name, variant='target', d=32, D_rff=2048, m_F_mult=1, sigma=None, n_pairs=5000,
              mean_mode='exact'):
    """mean_mode：'exact'(默认) 精确计算均值项 (1/n)Σ_j k(x, x_j)，分块 O(N·(N+n_F)·d)，N=3 万时约 5–10 s；
    'rff' 按规格用 D_rff 维随机傅里叶特征近似。
    偏离规格的原因（见测试 test_kherding_exact_beats_random_rff_does_not）：D_rff = 2048 时 RFF 均值项的误差
    标准差约 0.01（均值项本身的标准差约 0.1，相关 0.995），但与精确的 acc 项混用后，herding 会去拟合这个误差函数，
    所选样本的 MMD 反而比多数随机样本还差；精确均值项的代价只有数秒，因此默认用精确版。"""
    rng = np.random.default_rng(seed)
    N = ctx.N
    X = X_of(ctx, 'ae', d)
    if sigma is None:
        sigma = float(np.median(_random_pair_dists(X, n_pairs, rng)))
    # 核 k(x, x') = exp(−‖x − x'‖²/σ²)。对应的 RFF 频率 ω ~ N(0, (2/σ²) I)，即标准差 √2/σ；
    # 这样均值项与 acc 项用的是同一个核（M16 伪代码的 1/σ 对应 exp(−‖Δ‖²/(2σ²))，此处按本核换算）。
    # 无论 mean_mode 取什么都抽取 RFF 参数，保证随机数流一致。
    Wr = rng.normal(0.0, np.sqrt(2.0) / sigma, size=(X.shape[1], D_rff))
    br = rng.uniform(0.0, 2 * np.pi, D_rff)
    if mean_mode == 'rff':
        muT = _rff_mean_kernel(X, X, Wr, br)
    elif mean_mode == 'exact':
        muT = _exact_mean_kernel(X, X, sigma)
    else:
        raise ValueError(mean_mode)
    muF = None
    m_F = m_F_mult * k
    if variant == 'cond_foreign':
        F = X_of_foreign(ctx, 'ae', d)
        muF = _rff_mean_kernel(X, F, Wr, br) if mean_mode == 'rff' else _exact_mean_kernel(X, F, sigma)
    sqn = (X ** 2).sum(1)
    acc = np.zeros(N)
    chosen = np.zeros(N, bool)
    S = []
    for T in range(k):
        if variant == 'cond_foreign':
            score = muT - (m_F * muF + acc) / (m_F + T + 1)
        else:
            score = muT - acc / (T + 1)
        score[chosen] = -np.inf
        i = int(np.argmax(score))
        S.append(i)
        chosen[i] = True
        acc += np.exp(-_sqdist_to(X, sqn, i) / sigma ** 2)
    DIAG[name] = {'sigma': sigma, 'D_rff': D_rff, 'variant': variant, 'm_F': m_F if variant == 'cond_foreign' else None}
    return finalize(S, k, N)


@register('kherding')
def select_kherding(ctx, k, seed, D_rff=2048, sigma=None, mean_mode='exact', d=32):
    """M17 Kernel herding，p = 目标池经验分布；σ = 5,000 对随机距离的中位数；均值项默认精确计算（mean_mode='rff' 为规格的 RFF 版）。"""
    return _kherding(ctx, k, seed, 'kherding', variant='target', D_rff=D_rff, sigma=sigma, mean_mode=mean_mode, d=d)


@register('kherding_foreign')
def select_kherding_foreign(ctx, k, seed, m_F_mult=1, D_rff=2048, sigma=None, mean_mode='exact', d=32):
    """M17 变体 cond_foreign：把外国池当作已有的 m_F 个“样本质量”，m_F = m_F_mult·k，m_F_mult ∈ {1, 5}。"""
    return _kherding(ctx, k, seed, 'kherding_foreign', variant='cond_foreign', m_F_mult=m_F_mult, D_rff=D_rff,
                     sigma=sigma, mean_mode=mean_mode, d=d)


# ---------------------------------------------------------------------------
# M20 FSCS：k-means 代表性
# ---------------------------------------------------------------------------

def _fscs(ctx, k, seed, name, geo_w=0.0, d=32, n_init=None, max_iter=300):
    X = X_of(ctx, 'ae', d, geo_w=geo_w)
    if n_init is None:
        n_init = 3 if k <= 200 else 1
    km = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=int(seed)).fit(X)
    S = _nearest_unselected(X, km.cluster_centers_, q=10)
    DIAG[name] = {'n_iter': int(km.n_iter_), 'inertia': float(km.inertia_)}
    return finalize(S, k, ctx.N)


@register('fscs')
def select_fscs(ctx, k, seed, d=32, n_init=None, max_iter=300):
    """M20 FSCS：KMeans(k) 后取离各簇心最近的未选点。n_init 默认 k ≤ 200 取 3，否则 1。"""
    return _fscs(ctx, k, seed, 'fscs', geo_w=0.0, d=d, n_init=n_init, max_iter=max_iter)


@register('fscs_geo')
def select_fscs_geo(ctx, k, seed, geo_w=2.0, d=32, n_init=None, max_iter=300):
    """M20 变体：特征拼接 geo_w = 2 的标准化坐标。"""
    return _fscs(ctx, k, seed, 'fscs_geo', geo_w=geo_w, d=d, n_init=n_init, max_iter=max_iter)
