"""
迁移感知与信息最优的选样方法（SAMPLING_SURVEY_2026-09-24.md 第 3 节；规格见 sampling_shortlist.json）：

- M13 TQS-lite：留一国委员会 margin + 委员会分歧（JS）+ 高斯域性（域判别器分位秩），候选 3k 内按 Q 随机化选择。
      注册名 'tqs'（terms='u+c+d'）、'tqs_uc'（'u+c'）、'tqs_u'（'u'，确定性 top-k）。
- M18 BAIT 近似：多项逻辑"最后一层" Fisher 的 A-最优贪心，外国池 Fisher 初始化，Woodbury 低秩更新。
      注册名 'bait'（w=1）、'bait_wk10'（w=k/10）、'bait_wk'（w=k）、'bait_backward'（先前向 2k 再后向删到 k）。
- M8 Hybrid：第一批 ρ·k 用 M3 概率设计（PredStrat）→ oracle 取标签 → Olofsson 先验估计 →
      外国 ∪ S1 重训 LightGBM → Saerens 式先验校正 → 第二批用 BADGE（或 margin / GIST）在未选池上选 k−k0 个。
      注册名 'hybrid'（stage2='badge'）、'hybrid_margin'、'hybrid_gist'。这是唯一允许调用 ctx.oracle 的方法组。

公共约定见 base.py：select(ctx, k, seed, **hp) -> 升序 int64 下标；选样不读目标池标签（Hybrid 只经 ctx.oracle 读第一批）。
依赖：Hybrid 第一批直接调用 samplers.designs 的 M3（select_predstrat）与 olofsson_prior；
第二批直接调用 samplers.uncertainty 的 M7 select_badge / M4 select_margin / M6 select_gist_margin
（在限制到未选池、P_T 换成校正后 P1 的 ctx 副本上调用），重训用 uncertainty.fit_predict_lgbm，先验校正用 prior_adjust。
本文件新增的公共函数（base.py 未提供）：tqs_scores（TQS 各分量）、fisher_mean / bait_B / bait_V / bait_setup /
bait_objective（BAIT 的 Fisher 矩阵与 A-最优目标）、stage1_predstrat / stage2_select（Hybrid 两批）。
"""
import dataclasses

import numpy as np
from scipy.stats import rankdata

from .base import X_of, X_of_foreign, finalize, register, unc_margin


# =====================================================================
# M13 TQS-lite
# =====================================================================

def tqs_scores(ctx, mu=0.5, sigma=0.25, terms='u+c+d'):
    """返回 TQS 的总分 Q 及三个分量（dict）。只用 P_T_loco 与 dom，不读标签。"""
    if ctx.P_T_loco is None:
        raise ValueError('TQS 需要 ctx.P_T_loco（留一国委员会概率，形状 (M, N, C)）')
    P = np.asarray(ctx.P_T_loco, dtype=np.float64)
    M, N, C = P.shape
    assert N == ctx.N
    # Q_u：委员会平均 margin 不确定性
    Q_u = np.mean([unc_margin(P[m]) for m in range(M)], axis=0)
    # Q_c：委员会分歧 = 各成员到平均分布的 KL 的均值（广义 JS 散度），取分位秩
    pbar = P.mean(0)
    Pc = np.clip(P, 1e-12, None)
    js = (P * (np.log(Pc) - np.log(np.clip(pbar, 1e-12, None))[None])).sum(-1).mean(0)
    Q_c = rankdata(js) / N
    parts = {'u': Q_u, 'c': Q_c, 'js': js}
    Q = np.zeros(N)
    use = terms.split('+')
    for t in use:
        if t not in ('u', 'c', 'd'):
            raise ValueError(terms)
    if 'd' in use:
        if ctx.dom is None:
            raise ValueError("terms 含 'd' 时需要 ctx.dom（域判别器 P(目标|x)）")
        dom = np.clip(np.asarray(ctx.dom, dtype=np.float64), 1e-6, 1 - 1e-6)
        z = np.log(dom) - np.log1p(-dom)                      # logit
        rr = rankdata(z) / N                                  # 分位秩：跨国判别器输出常接近 1，原始尺度无法区分
        parts['d'] = np.exp(-(rr - mu) ** 2 / (2 * sigma ** 2))
        parts['rr'] = rr
    for t in use:
        Q = Q + parts[t]
    return Q, parts


def _tqs(ctx, k, seed, mu=0.5, sigma=0.25, terms='u+c+d', cand_mult=3, randomize=None):
    rng = np.random.default_rng(seed)
    Q, _ = tqs_scores(ctx, mu=mu, sigma=sigma, terms=terms)
    if randomize is None:
        randomize = terms != 'u'           # 规格：terms='u' 为确定性 top-k 的消融
    order = np.argsort(-(Q + 1e-12 * rng.random(ctx.N)), kind='stable')
    if not randomize:
        return finalize(order[:k], k, ctx.N)
    b = min(ctx.N, int(cand_mult) * k)     # b′ = min(N, 3k)
    cand = order[:b]
    p = Q[cand] / Q[cand].sum()
    idx = rng.choice(cand, size=k, replace=False, p=p)
    return finalize(idx, k, ctx.N)


@register('tqs')
def select_tqs(ctx, k, seed, mu=0.5, sigma=0.25, terms='u+c+d', cand_mult=3, randomize=None):
    """M13 TQS-lite 默认版：Q = Q_u + Q_c + Q_d，在 Q 最高的 min(N,3k) 个候选中按 Q 成比例无放回抽 k 个。"""
    return _tqs(ctx, k, seed, mu=mu, sigma=sigma, terms=terms, cand_mult=cand_mult, randomize=randomize)


@register('tqs_uc')
def select_tqs_uc(ctx, k, seed, mu=0.5, sigma=0.25, cand_mult=3, randomize=None):
    """M13 消融：去掉域性项（Q = Q_u + Q_c），仍做随机化选择。"""
    return _tqs(ctx, k, seed, mu=mu, sigma=sigma, terms='u+c', cand_mult=cand_mult, randomize=randomize)


@register('tqs_u')
def select_tqs_u(ctx, k, seed, cand_mult=3, randomize=None):
    """M13 消融：只用委员会平均 margin，确定性 top-k。"""
    return _tqs(ctx, k, seed, terms='u', cand_mult=cand_mult, randomize=randomize)


# =====================================================================
# M18 BAIT 近似
# =====================================================================
# 记号：p_i ∈ R^C（外国模型概率），h_i ∈ R^d（AE PCA 前 d 维），D = C·d。
# 单点 Fisher I(x_i) = A_i ⊗ h_i h_iᵀ，A_i = diag(p_i) − p_i p_iᵀ = B_i B_iᵀ，
# B_i = (I − p_i 1ᵀ)·diag(sqrt(p_i))，即 B_i[:, c] = sqrt(p_c)(e_c − p_i)，
# 与 JordanAsh/badge 的 get_exp_grad_embedding 第 c 行（假想标签 c 的梯度乘 sqrt(p_c)）一致。
# V_i = B_i ⊗ h_i ∈ R^{D×C}，行序为 (类 a, 特征 j)。
# 目标：min_S tr((M_0 + Σ_{i∈S} V_i V_iᵀ)^{-1} I_U)，M_0 = λI + w·I_F。
# 增益 gain_i = tr(M^{-1} I_U) − tr((M + V_i V_iᵀ)^{-1} I_U) = tr((I_C + R^M_i)^{-1} R^G_i)，
# 其中 R^M_i = V_iᵀ M^{-1} V_i，R^G_i = V_iᵀ G V_i，G = M^{-1} I_U M^{-1}。
# 实现上对每个候选维护 R^M_i、R^G_i（C×C），每选一个点做秩 C 的增量更新（推导见 _BaitState.update），
# 每步代价 O(n_cand·C²·(C+d))，而非规格伪代码中每步重算 H_i^Q 的 O(n_cand·C²·d²)；结果相同（测试中与直接计算比对）。
# 每 refresh_every 步直接求逆重算一次，消除数值漂移。
# 候选（规格）：k ≤ 200 用全池；k > 200 取 P_T margin 最高的 1 万个（N=3 万、k=1000 时约 40 s，否则会超过 3 分钟）。

def bait_B(P):
    """B[n, a, c] = (δ_ac − p_a)·sqrt(p_c)，满足 B Bᵀ = diag(p) − p pᵀ。"""
    P = np.asarray(P, dtype=np.float64)
    n, C = P.shape
    sq = np.sqrt(np.clip(P, 0, None))
    B = -P[:, :, None] * sq[:, None, :]
    B[:, np.arange(C), np.arange(C)] += sq
    return B


def bait_V(B_i, h_i):
    """单点的 V_i = B_i ⊗ h_i，形状 (D, C)，行序 (类 a, 特征 j)。"""
    C = B_i.shape[0]
    d = h_i.shape[0]
    return (B_i[:, None, :] * h_i[None, :, None]).reshape(C * d, C)


def fisher_mean(P, H, chunk=8192):
    """(1/n) Σ_i A_i ⊗ h_i h_iᵀ，按 (C,d,C,d) 排列后 reshape 为 (D, D)。等价于 einsum('nab,nj,nk->ajbk')。"""
    P = np.asarray(P, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    n, C = P.shape
    d = H.shape[1]
    acc = np.zeros((C * C, d * d))
    for s in range(0, n, chunk):
        p = P[s:s + chunk]
        h = H[s:s + chunk]
        A = -p[:, :, None] * p[:, None, :]
        A[:, np.arange(C), np.arange(C)] += p
        acc += A.reshape(len(p), C * C).T @ (h[:, :, None] * h[:, None, :]).reshape(len(p), d * d)
    I = (acc / n).reshape(C, C, d, d).transpose(0, 2, 1, 3).reshape(C * d, C * d)
    return 0.5 * (I + I.T)


def _VtQV(B, H, Q, chunk=2048):
    """对每个候选 n 计算 V_nᵀ Q V_n（C×C）。Q 为 (D, D) 对称阵。"""
    n, C, _ = B.shape
    d = H.shape[1]
    Qr = Q.reshape(C, d, C, d).transpose(1, 0, 2, 3).reshape(d, C * C * d)   # [j, (a, b, k)]
    out = np.empty((n, C, C))
    for s in range(0, n, chunk):
        h = H[s:s + chunk]
        T = (h @ Qr).reshape(len(h), C * C, d)                   # Σ_j h_j Q[a,j,b,k]
        HQ = np.matmul(T, h[:, :, None]).reshape(len(h), C, C)   # H^Q[a,b] = hᵀ Q[a,:,b,:] h
        b = B[s:s + chunk]
        out[s:s + chunk] = b.transpose(0, 2, 1) @ HQ @ b         # R = Bᵀ H^Q B
    return 0.5 * (out + out.transpose(0, 2, 1))


def _VtX(B, H, X):
    """对每个候选 n 计算 V_nᵀ X，X 为 (D, m)；返回 (n, C, m)。"""
    n, C, _ = B.shape
    d = H.shape[1]
    m = X.shape[1]
    Xr = X.reshape(C, d, m).transpose(1, 0, 2).reshape(d, C * m)
    Z = (H @ Xr).reshape(n, C, m)                                # Z[n, a, :] = Σ_j h_j X[(a,j), :]
    return np.matmul(B.transpose(0, 2, 1), Z)                    # Σ_a B[n,a,c] Z[n,a,:]


class _BaitState:
    """全局量 M、M^{-1}、G = M^{-1} I_U M^{-1}，以及一组行（候选）的 R^M、R^G。"""

    def __init__(self, M, I_U, B, H):
        self.M = M.copy()
        self.I_U = I_U
        self.B, self.H = B, H
        self.C = B.shape[1]
        self.refresh()

    def refresh(self):
        """直接求逆并重算所有候选的 R^M、R^G（消除增量更新的数值漂移）。"""
        self.Minv = np.linalg.inv(self.M)
        self.Minv = 0.5 * (self.Minv + self.Minv.T)
        self.G = self.Minv @ self.I_U @ self.Minv
        self.G = 0.5 * (self.G + self.G.T)
        self.RM = _VtQV(self.B, self.H, self.Minv)
        self.RG = _VtQV(self.B, self.H, self.G)

    def gain_add(self):
        """加入第 i 个候选后 tr(M^{-1} I_U) 的下降量。"""
        S = np.eye(self.C)[None] + self.RM
        return np.trace(np.linalg.solve(S, self.RG), axis1=1, axis2=2)

    def loss_remove(self):
        """删去第 i 个（已在 M 中的）点后 tr(M^{-1} I_U) 的上升量。"""
        S = np.eye(self.C)[None] - self.RM
        return np.trace(np.linalg.solve(S, self.RG), axis1=1, axis2=2)

    def update(self, v, sign):
        """M ← M + sign·v vᵀ（sign=+1 加入，−1 删除），Woodbury 更新 M^{-1}、G 与各候选的 R^M、R^G。

        记 U = M^{-1} v，Pm = G v，R^M_* = vᵀU，R^G_* = vᵀPm，W = sign·(I + sign·R^M_*)^{-1}，则
          M^{-1}_new = M^{-1} − U W Uᵀ
          G_new      = G − U W Pmᵀ − Pm W Uᵀ + U (W R^G_* W) Uᵀ
          R^M_new[n] = R^M[n] − Y_u W Y_uᵀ
          R^G_new[n] = R^G[n] − Y_u W Y_pᵀ − Y_p W Y_uᵀ + Y_u (W R^G_* W) Y_uᵀ，Y_u = V_nᵀU，Y_p = V_nᵀPm。
        """
        C = self.C
        U = self.Minv @ v
        Pm = self.G @ v
        RMs = v.T @ U
        RGs = v.T @ Pm
        W = sign * np.linalg.inv(np.eye(C) + sign * RMs)
        W = 0.5 * (W + W.T)
        K = W @ RGs @ W
        self.M += sign * (v @ v.T)
        self.Minv -= U @ W @ U.T
        UWP = U @ W @ Pm.T
        self.G += -UWP - UWP.T + U @ K @ U.T
        Y = _VtX(self.B, self.H, np.hstack([U, Pm]))
        Yu, Yp = Y[:, :, :C], Y[:, :, C:]
        n = len(Yu)
        YuW = (Yu.reshape(n * C, C) @ W).reshape(n, C, C)
        YuK = (Yu.reshape(n * C, C) @ K).reshape(n, C, C)
        YuT = Yu.transpose(0, 2, 1)
        self.RM -= YuW @ YuT
        X1 = YuW @ Yp.transpose(0, 2, 1)
        self.RG -= X1 + X1.transpose(0, 2, 1) - YuK @ YuT

    def objective(self):
        return float(np.trace(self.Minv @ self.I_U))


def bait_setup(ctx, d=16, w=1.0, lam_rel=1e-2):
    """构造 BAIT 的 h、B、I_U、I_F 与 M_0 = λI + w·I_F。"""
    if ctx.P_F_oof is None:
        raise ValueError('BAIT 需要 ctx.P_F_oof（外国池袋外概率）来构造外国 Fisher 初始化')
    h = X_of(ctx, 'ae', d)
    h_F = X_of_foreign(ctx, 'ae', d)
    P = np.asarray(ctx.P_T, dtype=np.float64)
    I_U = fisher_mean(P, h)
    I_F = fisher_mean(ctx.P_F_oof, h_F)
    D = I_F.shape[0]
    lam = lam_rel * np.trace(I_F) / D
    M0 = lam * np.eye(D) + w * I_F
    return dict(h=h, P=P, I_U=I_U, I_F=I_F, M0=M0, lam=lam, D=D)


def bait_objective(ctx, idx, d=16, w=1.0, lam_rel=1e-2, setup=None):
    """tr((M_0 + Σ_{i∈idx} I(x_i))^{-1} I_U)，用于检验与比较。"""
    s = setup or bait_setup(ctx, d=d, w=w, lam_rel=lam_rel)
    idx = np.asarray(idx)
    B = bait_B(s['P'][idx])
    M = s['M0'].copy()
    for b, h in zip(B, s['h'][idx]):
        v = bait_V(b, h)
        M += v @ v.T
    return float(np.trace(np.linalg.solve(M, s['I_U'])))


def _bait(ctx, k, seed, d=16, w='1', lam_rel=1e-2, n_cand='auto', backward=False, refresh_every=250,
          return_state=False):
    # w 可为数值，或字符串 '1' / 'k/10' / 'k'（按预算缩放外国 Fisher）
    if isinstance(w, str):
        w = {'1': 1.0, 'k/10': k / 10.0, 'k': float(k)}[w]
    s = bait_setup(ctx, d=d, w=float(w), lam_rel=lam_rel)
    N = ctx.N
    # 候选：k ≤ 200 用全池；否则取 margin 最高的 1 万个（规格；k=1000 时保证 3 分钟内）
    if n_cand == 'auto':
        n_cand = N if k <= 200 else min(N, 10000)
    n_cand = int(min(N, max(n_cand, k)))
    if n_cand < N:
        rng = np.random.default_rng(seed)
        u = unc_margin(s['P'])
        cand = np.sort(np.argsort(-(u + 1e-12 * rng.random(N)), kind='stable')[:n_cand])
    else:
        cand = np.arange(N)
    B = bait_B(s['P'][cand])
    H = s['h'][cand]
    st = _BaitState(s['M0'], s['I_U'], B, H)
    n_fwd = min(n_cand, 2 * k) if backward else k
    chosen = np.zeros(n_cand, dtype=bool)
    sel = []
    for t in range(n_fwd):
        if t > 0 and refresh_every and t % refresh_every == 0:
            st.refresh()
        g = st.gain_add()
        g[chosen] = -np.inf
        i = int(np.argmax(g))
        chosen[i] = True
        sel.append(i)
        st.update(bait_V(B[i], H[i]), +1)
    if backward and n_fwd > k:
        # 后向：在已选集合上逐个删去"删去后目标上升最少"的点（JordanAsh/badge 的 backward pruning）
        sel = list(sel)
        sub = np.array(sel)
        st2 = _BaitState(st.M, s['I_U'], B[sub], H[sub])
        alive = np.ones(len(sub), dtype=bool)
        for t in range(n_fwd - k):
            loss = st2.loss_remove()
            loss[~alive] = np.inf
            j = int(np.argmin(loss))
            alive[j] = False
            st2.update(bait_V(B[sub[j]], H[sub[j]]), -1)
        sel = sub[alive].tolist()
        st = st2
    idx = cand[np.array(sel, dtype=np.int64)]
    out = finalize(idx, k, N)
    if return_state:
        return out, st, s
    return out


@register('bait')
def select_bait(ctx, k, seed, d=16, w='1', lam_rel=1e-2, n_cand='auto', backward=False, refresh_every=250):
    """M18 BAIT 近似（默认 w=1，前向贪心）。确定性方法：seed 只在候选截断时打破并列。"""
    return _bait(ctx, k, seed, d=d, w=w, lam_rel=lam_rel, n_cand=n_cand, backward=backward,
                 refresh_every=refresh_every)


@register('bait_wk10')
def select_bait_wk10(ctx, k, seed, d=16, lam_rel=1e-2, n_cand='auto', backward=False, refresh_every=250):
    """M18 变体：外国 Fisher 缩放 w = k/10。"""
    return _bait(ctx, k, seed, d=d, w='k/10', lam_rel=lam_rel, n_cand=n_cand, backward=backward,
                 refresh_every=refresh_every)


@register('bait_wk')
def select_bait_wk(ctx, k, seed, d=16, lam_rel=1e-2, n_cand='auto', backward=False, refresh_every=250):
    """M18 变体：外国 Fisher 缩放 w = k。"""
    return _bait(ctx, k, seed, d=d, w='k', lam_rel=lam_rel, n_cand=n_cand, backward=backward,
                 refresh_every=refresh_every)


@register('bait_backward')
def select_bait_backward(ctx, k, seed, d=16, w='1', lam_rel=1e-2, n_cand='auto', refresh_every=250):
    """M18 变体：原文做法，先前向选 2k 个，再后向删到 k 个。"""
    return _bait(ctx, k, seed, d=d, w=w, lam_rel=lam_rel, n_cand=n_cand, backward=True,
                 refresh_every=refresh_every)


# =====================================================================
# M8 Hybrid
# =====================================================================
# 两批标注的真实流程：
#   S1 = M3 PredStrat(ctx, k0)（概率设计，π_i = n_h/N_h）→ y1 = oracle(S1) → p̂ = Olofsson(S1, y1)
#   → (P1, p_tr) = fit_predict(外国 ∪ S1)（目标权重 r，类别权重只按外国频率）
#   → P1 ← P1·p̂/p_tr 再行归一化 → S2 = stage2 选择器（在未选池、以 P1 为概率）选 k − k0 个。
# 第一批与第二批的选择器都直接调用其他组的正式实现：samplers.designs（M3、Olofsson）、
# samplers.uncertainty（M7 BADGE、M4 margin、M6 GIST、训练钩子 fit_predict_lgbm、prior_adjust）。

def stage1_predstrat(ctx, k0, seed):
    """第一批：M3 PredStrat 默认参数（ŷ 分层、alloc='sqrt'、m_min=2、层内 LPM2）。
    在去掉 oracle、meta 独立的 ctx 副本上调用：保证第一批选样不读标签，也不改写 ctx.meta['predstrat']。
    返回 (S1, M3 的 meta：含 pi、strata、N_h、n_h)。"""
    from . import designs
    c1 = dataclasses.replace(ctx, oracle=None, meta={})
    S1 = designs.select_predstrat(c1, k0, seed)
    return np.asarray(S1, dtype=np.int64), c1.meta['predstrat']


def _subctx(ctx, rest, P_new):
    """把 ctx 限制到下标 rest，并以 P_new[rest] 作为 P_T；去掉 oracle（第二批不读标签）。
    X_of 的整体缩放在子池上重算，只差一个标量，BADGE / margin / GIST 的选择都与缩放无关。"""
    return dataclasses.replace(
        ctx, N=len(rest), T_ae=ctx.T_ae[rest], T_s2=ctx.T_s2[rest], T_xy=ctx.T_xy[rest],
        T_nuts3=ctx.T_nuts3[rest], P_T=P_new[rest],
        P_T_loco=None if ctx.P_T_loco is None else np.asarray(ctx.P_T_loco)[:, rest],
        dom=None if ctx.dom is None else np.asarray(ctx.dom)[rest], oracle=None, meta={})


def stage2_select(ctx, rest, P1, k2, seed, stage2='badge'):
    """第二批：在未选池 rest 上、以校正后的 P1 为概率选 k2 个，返回全池下标。
    'badge' = M7 默认（h = AE32），'margin' = M4 单轮，'gist' = M6 GIST-margin（α=0.9）。"""
    if k2 <= 0:
        return np.zeros(0, dtype=np.int64)
    from . import uncertainty as U
    fn = {'badge': U.select_badge, 'margin': U.select_margin, 'gist': U.select_gist_margin}[stage2]
    loc = np.asarray(fn(_subctx(ctx, rest, P1), k2, seed), dtype=np.int64)
    return rest[loc]


def _hybrid(ctx, k, seed, rho=0.5, stage2='badge', r=20.0, fit_predict=None, fit_kw=None, name='hybrid'):
    if ctx.oracle is None:
        raise ValueError('Hybrid 需要 ctx.oracle 揭示第一批标签')
    N, C = ctx.N, ctx.C
    k0 = int(min(k, max(10, round(rho * k))))
    # 第一批与 M3 用同一 seed（ρ=1 时结果与 'predstrat' 相同）；训练与第二批用由 seed 派生的独立种子
    seed_fit, seed2 = [int(x) % (2 ** 31) for x in np.random.SeedSequence([int(seed), 8]).generate_state(2)]
    # 第一批：概率设计（不读标签）
    S1, m3 = stage1_predstrat(ctx, k0, seed)
    # 揭示第一批标签（唯一一次调用 oracle）
    y1 = np.asarray(ctx.oracle(S1), dtype=np.int64)
    # Olofsson 先验估计（式 9、10；层 = 外国预测类 ŷ）
    from .designs import olofsson_prior
    p_hat, p_se, q = olofsson_prior(y1, m3['strata'][S1], m3['N_h'], C=C, return_q=True)
    meta = dict(pi=np.asarray(m3['pi'], dtype=np.float64), prior_idx=S1, estimator='olofsson',
                strata=m3['strata'], N_h=m3['N_h'], n_h=m3['n_h'], k0=k0, rho=rho, p_hat=p_hat, p_hat_se=p_se,
                confusion_q=q, stage2=stage2)
    S2 = np.zeros(0, dtype=np.int64)
    if k0 < k:
        from . import uncertainty as U
        # 外国 ∪ S1 重训（默认：uncertainty.fit_predict_lgbm，与 M4 多轮版同一训练钩子，含分组留国温度校准）
        if fit_predict is None:
            P1, p_tr = U.fit_predict_lgbm(ctx, S1, y1, seed_fit, r=r, **(fit_kw or {}))[:2]
        else:
            P1, p_tr = fit_predict(ctx, S1, y1, seed_fit)[:2]
        P1 = np.asarray(P1, dtype=np.float64)
        p_tr = np.asarray(p_tr, dtype=np.float64)
        # Saerens / Kluger 式先验校正：q ∝ p·p̂/p_tr
        P1c = U.prior_adjust(P1, p_hat, p_tr)
        rest = np.setdiff1d(np.arange(N), S1)
        S2 = stage2_select(ctx, rest, P1c, k - k0, seed2, stage2=stage2)
        meta.update(p_tr=p_tr)
    meta['S2'] = np.sort(S2)
    ctx.meta[name] = meta
    return finalize(np.r_[S1, S2], k, N)


_HYBRID_DOC = """
    r：目标样本权重，传给默认训练钩子（第 5.4 节的 r(k) 由主程序在开发国上用 SRS 选定后传入；默认 20 与
       uncertainty.fit_predict_lgbm 一致，只是占位）。
    fit_predict(ctx, S, y_S, seed) -> (P (N, C), p_tr (C,))：可注入主程序的正式训练函数（此时 r、fit_kw 不用，
       由注入函数自行处理）；缺省用 uncertainty.fit_predict_lgbm(ctx, S1, y1, seed, r=r, **fit_kw)。
    fit_kw：传给默认训练钩子的参数（如 n_trees、calib、calib_folds）。
    meta[注册名]：pi（只对第一批定义，π_i = n_h/N_h，和为 k0）、prior_idx = S1、estimator='olofsson'、
       strata / N_h / n_h（第一批的分层信息）、p_hat、p_hat_se（S1 的设计估计先验，报告 P1 档时使用）、
       confusion_q（目标国混淆 P(y|ŷ) 的估计）、p_tr、S2。
"""


@register('hybrid')
def select_hybrid(ctx, k, seed, rho=0.5, stage2='badge', r=20.0, fit_predict=None, fit_kw=None):
    return _hybrid(ctx, k, seed, rho=rho, stage2=stage2, r=r, fit_predict=fit_predict, fit_kw=fit_kw,
                   name='hybrid')


select_hybrid.__doc__ = 'M8 Hybrid 默认版：ρ=0.5，第二批 BADGE（M7 默认）。唯一调用 ctx.oracle（一次，只揭示 S1）。' + _HYBRID_DOC


@register('hybrid_margin')
def select_hybrid_margin(ctx, k, seed, rho=0.5, r=20.0, fit_predict=None, fit_kw=None):
    return _hybrid(ctx, k, seed, rho=rho, stage2='margin', r=r, fit_predict=fit_predict, fit_kw=fit_kw,
                   name='hybrid_margin')


select_hybrid_margin.__doc__ = 'M8 变体：第二批用 M4 单轮 margin。' + _HYBRID_DOC


@register('hybrid_gist')
def select_hybrid_gist(ctx, k, seed, rho=0.5, r=20.0, fit_predict=None, fit_kw=None):
    return _hybrid(ctx, k, seed, rho=rho, stage2='gist', r=r, fit_predict=fit_predict, fit_kw=fit_kw,
                   name='hybrid_gist')


select_hybrid_gist.__doc__ = 'M8 变体：第二批用 M6 GIST-margin（α=0.9）。' + _HYBRID_DOC
