"""
samplers/transfer.py 的单元测试（M13 TQS-lite、M18 BAIT 近似、M8 Hybrid）。

pytest 风格；也可直接运行：
    OMP_NUM_THREADS=2 LGBM_THREADS=2 python samplers/tests/test_transfer.py
环境变量 SKIP_SLOW=1 跳过 N=30000、k=1000 的耗时测试。

交叉核对：
- BAIT：与 JordanAsh/badge（query_strategies/bait_sampling.py 的 select，及 strategy.py 的 get_exp_grad_embedding）
  逐式移植的稠密 numpy 版比对所选下标序列（前向与前向+后向）；Woodbury 增量更新与直接求逆比对。
  原仓库是 PyTorch + CUDA 代码、不是 PyPI 包，因此移植核心公式而非安装。
- TQS：thuml/Transferable-Query-Selection 的 code/active.py 只有 random 与 uncertainty 两种选择
  （README 标注 "Waiting for code update"），没有可比对的 TQS 实现；只做公式级检查。
- Hybrid：无外部参考实现；第一批直接用 samplers.designs 的 M3（检查 ρ=1 时与 'predstrat' 完全一致），
  第二批直接用 samplers.uncertainty 的 M7 / M4 / M6（检查与在子池上直接调用的结果一致）。
"""
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # label_expansion/

from samplers.base import REGISTRY, make_toy_ctx, unc_margin, X_of  # noqa: E402
from samplers import transfer as TR  # noqa: E402

RANDOM_METHODS = ['tqs', 'tqs_uc', 'hybrid', 'hybrid_margin', 'hybrid_gist']
DETERMINISTIC = ['tqs_u', 'bait', 'bait_wk10', 'bait_wk', 'bait_backward']
# 单元测试里用更轻的重训（默认训练钩子为 uncertainty.fit_predict_lgbm：100 棵树 + 5 组留国温度校准）
FAST_LGBM = {'fit_kw': {'n_trees': 20, 'calib_folds': 2}}


def toy(N=3000, n_F=5000, seed=0):
    """make_toy_ctx 加上假的 P_T_loco、dom、P_F_oof。"""
    ctx = make_toy_ctx(N=N, n_F=n_F, seed=seed)
    r = np.random.default_rng(seed + 100)
    lp = np.log(ctx.P_T)
    loco = []
    for m in range(10):
        # 各留一国成员：共同的对数概率 + 成员特有的类偏置 + 个体噪声
        z = lp + r.normal(scale=0.3, size=ctx.C)[None] + r.normal(scale=0.4, size=lp.shape)
        z = np.exp(z - z.max(1, keepdims=True))
        loco.append(z / z.sum(1, keepdims=True))
    ctx.P_T_loco = np.stack(loco)
    # 域判别器：离外国池均值越远越像目标，多数输出接近 1（模拟跨国判别器）
    dist = np.linalg.norm(ctx.T_ae[:, :8] - ctx.F_ae[:, :8].mean(0), axis=1)
    ctx.dom = 1 / (1 + np.exp(-(3 + 0.5 * (dist - np.median(dist)) + r.normal(scale=0.5, size=N))))
    # 外国池袋外概率：真实类上偏高的带噪 softmax
    zF = 2.5 * np.eye(ctx.C)[ctx.F_y] + r.normal(size=(n_F, ctx.C))
    zF = np.exp(zF - zF.max(1, keepdims=True))
    ctx.P_F_oof = zF / zF.sum(1, keepdims=True)
    return ctx


def hp_for(name):
    return FAST_LGBM if name.startswith('hybrid') else {}


# ---------------------------------------------------------------------
# 通用：长度、无重复、可复现、种子差异、不读标签
# ---------------------------------------------------------------------

def test_length_unique_sorted():
    ctx = toy()
    for name in RANDOM_METHODS + DETERMINISTIC:
        for k in (50, 200):
            idx = REGISTRY[name](ctx, k, 0, **hp_for(name))
            assert idx.dtype == np.int64 and len(idx) == k, (name, k)
            assert len(np.unique(idx)) == k and (np.diff(idx) > 0).all(), name
            assert idx.min() >= 0 and idx.max() < ctx.N


def test_reproducible_same_seed():
    ctx = toy()
    for name in RANDOM_METHODS + DETERMINISTIC:
        a = REGISTRY[name](ctx, 50, 3, **hp_for(name))
        b = REGISTRY[name](ctx, 50, 3, **hp_for(name))
        assert np.array_equal(a, b), name


def test_different_seeds():
    ctx = toy()
    for name in RANDOM_METHODS:
        a = REGISTRY[name](ctx, 50, 0, **hp_for(name))
        b = REGISTRY[name](ctx, 50, 1, **hp_for(name))
        assert not np.array_equal(a, b), name
    # 确定性方法（BAIT 贪心、TQS 'u' 的 top-k）：不同种子结果相同
    for name in DETERMINISTIC:
        a = REGISTRY[name](ctx, 50, 0)
        b = REGISTRY[name](ctx, 50, 1)
        assert np.array_equal(a, b), name


def _raise(*a, **kw):
    raise RuntimeError('选样函数读取了标签')


def test_no_label_access():
    ctx = toy()
    ref = {name: REGISTRY[name](ctx, 50, 0) for name in DETERMINISTIC + ['tqs', 'tqs_uc']}
    ctx.oracle = _raise
    del ctx._true_y
    for name in DETERMINISTIC + ['tqs', 'tqs_uc']:
        assert np.array_equal(REGISTRY[name](ctx, 50, 0), ref[name]), name
    # Hybrid 是允许用 oracle 的方法：oracle 抛异常时它必须失败（说明 oracle 是唯一的标签通道）……
    try:
        REGISTRY['hybrid'](ctx, 50, 0, **FAST_LGBM)
        raise AssertionError('hybrid 未调用 oracle')
    except RuntimeError as e:
        assert '标签' in str(e)
    # ……且只调用一次、只揭示第一批
    ctx2 = toy()
    yT = ctx2._true_y
    del ctx2._true_y
    calls = []

    def oracle(idx):
        calls.append(np.asarray(idx).copy())
        return yT[np.asarray(idx)]
    ctx2.oracle = oracle
    idx = REGISTRY['hybrid'](ctx2, 50, 0, **FAST_LGBM)
    m = ctx2.meta['hybrid']
    assert len(calls) == 1 and len(calls[0]) == m['k0'] == 25
    assert np.array_equal(np.sort(calls[0]), m['prior_idx'])
    assert np.isin(m['prior_idx'], idx).all()


def test_missing_optional_fields():
    """缺可选字段时给出明确错误；Hybrid 不需要 P_T_loco / dom / P_F_oof。"""
    ctx = make_toy_ctx()
    for name, field in (('tqs', 'P_T_loco'), ('tqs_u', 'P_T_loco'), ('bait', 'P_F_oof')):
        try:
            REGISTRY[name](ctx, 20, 0)
            raise AssertionError(name)
        except ValueError as e:
            assert field in str(e), (name, e)
    c2 = toy()
    c2.dom = None
    assert len(REGISTRY['tqs_uc'](c2, 20, 0)) == 20     # 不含域性项时不需要 dom
    try:
        REGISTRY['tqs'](c2, 20, 0)
        raise AssertionError('tqs 缺 dom 应报错')
    except ValueError as e:
        assert 'dom' in str(e)
    assert len(REGISTRY['hybrid'](ctx, 30, 0, **FAST_LGBM)) == 30


# ---------------------------------------------------------------------
# M13 TQS-lite
# ---------------------------------------------------------------------

def test_tqs_u_is_committee_margin_topk():
    ctx = toy()
    k = 100
    Qu = np.mean([unc_margin(P) for P in ctx.P_T_loco], axis=0)
    idx = REGISTRY['tqs_u'](ctx, k, 0)
    assert np.array_equal(idx, np.sort(np.argsort(-Qu, kind='stable')[:k]))


def test_tqs_scores_and_candidates():
    ctx = toy()
    k = 100
    Q, parts = TR.tqs_scores(ctx)
    # 分量范围
    assert (parts['u'] >= 0).all() and (parts['u'] <= 1).all()
    assert (parts['c'] > 0).all() and (parts['c'] <= 1).all()
    assert (parts['d'] > 0).all() and (parts['d'] <= 1).all()
    # 委员会完全一致时 JS = 0；成员差异越大 JS 越大
    c0 = toy()
    c0.P_T_loco = np.repeat(c0.P_T[None], 10, 0)
    _, p0 = TR.tqs_scores(c0)
    assert np.allclose(p0['js'], 0, atol=1e-12)
    assert (parts['js'] >= -1e-12).all() and parts['js'].mean() > 1e-3
    # 所选点都在 Q 最高的 3k 个候选中
    top = set(np.argsort(-Q, kind='stable')[:3 * k])
    for s in range(5):
        assert set(REGISTRY['tqs'](ctx, k, s)) <= top
    # 高斯域性把样本拉向判别器分位秩 μ 附近：|rr − μ| 的均值小于不含域性项的版本
    rr = parts['rr']
    dev_d = np.mean([np.abs(rr[REGISTRY['tqs'](ctx, k, s)] - 0.5).mean() for s in range(10)])
    dev_nod = np.mean([np.abs(rr[REGISTRY['tqs_uc'](ctx, k, s)] - 0.5).mean() for s in range(10)])
    assert dev_d < dev_nod, (dev_d, dev_nod)
    # μ 改变时所选样本的域性分位中位数随之移动
    lo = np.median(rr[REGISTRY['tqs'](ctx, k, 0, mu=0.3)])
    hi = np.median(rr[REGISTRY['tqs'](ctx, k, 0, mu=0.7)])
    assert lo < hi, (lo, hi)


def test_tqs_inclusion_increases_with_Q():
    """随机化选择：候选内入样频率与 Q 正相关（按 Q 成比例的无放回抽样）。"""
    ctx = toy(N=600, n_F=800)
    k = 30
    Q, _ = TR.tqs_scores(ctx)
    cand = np.argsort(-Q, kind='stable')[:3 * k]
    freq = np.zeros(ctx.N)
    for s in range(400):
        freq[REGISTRY['tqs'](ctx, k, s)] += 1
    assert freq[np.setdiff1d(np.arange(ctx.N), cand)].sum() == 0
    # 候选内 Q 的范围较窄（约 2.2–2.7），按 Q 成比例抽样的入样频率差异温和；实测相关约 0.48
    assert np.corrcoef(Q[cand], freq[cand])[0, 1] > 0.3
    hi = Q[cand] >= np.median(Q[cand])
    assert freq[cand][hi].mean() > freq[cand][~hi].mean()


# ---------------------------------------------------------------------
# M18 BAIT
# ---------------------------------------------------------------------

def test_bait_fisher_building_blocks():
    r = np.random.default_rng(0)
    n, C, d = 40, 5, 3
    z = r.normal(size=(n, C))
    P = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    H = r.normal(size=(n, d))
    B = TR.bait_B(P)
    A = np.einsum('na,ab->nab', P, np.eye(C)) - P[:, :, None] * P[:, None, :]
    assert np.allclose(B @ B.transpose(0, 2, 1), A)
    # 单点 Fisher：V Vᵀ = A ⊗ h hᵀ（行序 (a, j)）
    V = TR.bait_V(B[0], H[0])
    assert np.allclose(V @ V.T, np.kron(A[0], np.outer(H[0], H[0])))
    # V 的第 c 列 = sqrt(p_c)(e_c − p)⊗h，即 JordanAsh/badge get_exp_grad_embedding 的第 c 行
    c = 2
    emb = np.concatenate([H[0] * ((1 - P[0, c]) if a == c else -P[0, a]) for a in range(C)]) * np.sqrt(P[0, c])
    assert np.allclose(V[:, c], emb)
    # fisher_mean 与规格中的 einsum 写法一致
    ref = np.einsum('nab,nj,nk->ajbk', A, H, H).reshape(C * d, C * d) / n
    assert np.allclose(TR.fisher_mean(P, H, chunk=7), ref)
    # _VtQV、_VtX 与显式构造的 V 一致
    Q = r.normal(size=(C * d, C * d))
    Q = Q @ Q.T
    X = r.normal(size=(C * d, 4))
    R = TR._VtQV(B, H, Q, chunk=9)
    Y = TR._VtX(B, H, X)
    for i in (0, 7, 39):
        Vi = TR.bait_V(B[i], H[i])
        assert np.allclose(R[i], Vi.T @ Q @ Vi)
        assert np.allclose(Y[i], Vi.T @ X)


def test_bait_woodbury_matches_direct():
    """增量（Woodbury）更新后的 M^{-1}、G、各候选 R^M、R^G 与直接求逆重算一致；增益与暴力计算一致。"""
    ctx = toy(N=800, n_F=1500)
    k = 40
    idx, st, s = TR._bait(ctx, k, 0, refresh_every=0, return_state=True)   # 全程只用增量更新
    assert np.allclose(st.Minv, np.linalg.inv(st.M), rtol=1e-8, atol=1e-10 * np.abs(st.Minv).max())
    Minv = np.linalg.inv(st.M)
    G = Minv @ s['I_U'] @ Minv
    assert np.allclose(st.G, G, atol=1e-9 * np.abs(G).max())
    RM = TR._VtQV(st.B, st.H, Minv)
    RG = TR._VtQV(st.B, st.H, G)
    assert np.allclose(st.RM, RM, atol=1e-9 * np.abs(RM).max())
    assert np.allclose(st.RG, RG, atol=1e-9 * np.abs(RG).max())
    # 增益 = tr(M^{-1} I_U) − tr((M + V Vᵀ)^{-1} I_U)，与暴力逐点求逆一致
    g = st.gain_add()
    base = np.trace(Minv @ s['I_U'])
    for i in (0, 11, 123, 799):
        V = TR.bait_V(st.B[i], st.H[i])
        brute = base - np.trace(np.linalg.solve(st.M + V @ V.T, s['I_U']))
        assert np.isclose(g[i], brute, rtol=1e-7, atol=1e-12), (g[i], brute)
    # 删除的 Woodbury（后向步）：tr((M − V Vᵀ)^{-1} I_U) − tr(M^{-1} I_U)
    sub = st.B[idx[:5]], st.H[idx[:5]]
    st2 = TR._BaitState(st.M, s['I_U'], *sub)
    loss = st2.loss_remove()
    for j in range(5):
        V = TR.bait_V(sub[0][j], sub[1][j])
        brute = np.trace(np.linalg.solve(st.M - V @ V.T, s['I_U'])) - base
        assert np.isclose(loss[j], brute, rtol=1e-7), (loss[j], brute)
    # 目标值：返回的状态与 bait_objective 从头计算一致
    assert np.isclose(st.objective(), TR.bait_objective(ctx, idx), rtol=1e-8)


def _ref_bait_select(Xemb, K, fisher, iterates, lamb, nLabeled, over_sample=2):
    """JordanAsh/badge query_strategies/bait_sampling.py 中 select() 的逐式 numpy 移植（torch/cuda → numpy）。
    Xemb: (n, rank=C, dim=D) 期望梯度嵌入；fisher: 池平均 Fisher；iterates: 已标注平均 Fisher。"""
    indsAll = []
    dim = Xemb.shape[-1]
    rank = Xemb.shape[-2]
    currentInv = np.linalg.inv(lamb * np.eye(dim) + iterates * nLabeled / (nLabeled + K))
    X = Xemb * np.sqrt(K / (nLabeled + K))
    for i in range(int(over_sample * K)):
        innerInv = np.linalg.inv(np.eye(rank) + X @ currentInv @ X.transpose(0, 2, 1))
        traceEst = np.diagonal(X @ currentInv @ fisher @ currentInv @ X.transpose(0, 2, 1) @ innerInv,
                               axis1=-2, axis2=-1).sum(-1)
        for j in np.argsort(traceEst)[::-1]:
            if j not in indsAll:
                ind = j
                break
        indsAll.append(ind)
        xt_ = X[ind][None]
        innerInv = np.linalg.inv(np.eye(rank) + xt_ @ currentInv @ xt_.transpose(0, 2, 1))
        currentInv = (currentInv - currentInv @ xt_.transpose(0, 2, 1) @ innerInv @ xt_ @ currentInv)[0]
    for i in range(len(indsAll) - K):
        xt_ = X[indsAll]
        innerInv = np.linalg.inv(-1 * np.eye(rank) + xt_ @ currentInv @ xt_.transpose(0, 2, 1))
        traceEst = np.diagonal(xt_ @ currentInv @ fisher @ currentInv @ xt_.transpose(0, 2, 1) @ innerInv,
                               axis1=-2, axis2=-1).sum(-1)
        delInd = int(np.argmin(-1 * traceEst))
        xt_ = X[indsAll[delInd]][None]
        innerInv = np.linalg.inv(-1 * np.eye(rank) + xt_ @ currentInv @ xt_.transpose(0, 2, 1))
        currentInv = (currentInv - currentInv @ xt_.transpose(0, 2, 1) @ innerInv @ xt_ @ currentInv)[0]
        del indsAll[delInd]
    return indsAll


def test_bait_matches_reference_port():
    """与原作者代码的移植版比对。参数换算：原代码 M' = lamb·I + nL/(nL+K)·I_L + K/(nL+K)·Σ I(x)
    = K/(nL+K)·[λI + w·I_F + Σ I(x)]，取 nL = w·K、lamb = λ·K/(nL+K) 时与本实现只差一个正比例，argmax/argmin 相同。"""
    ctx = toy(N=300, n_F=600, seed=5)
    d = 4
    for w, K in ((1.0, 12), (5.0, 8)):
        s = TR.bait_setup(ctx, d=d, w=w)
        B = TR.bait_B(s['P'])
        Xemb = np.stack([TR.bait_V(B[i], s['h'][i]).T for i in range(ctx.N)])    # (n, C, D)
        nL = w * K
        lamb = s['lam'] * K / (nL + K)
        # 前向（over_sample=1）
        ref = _ref_bait_select(Xemb, K, s['I_U'], s['I_F'], lamb, nL, over_sample=1)
        ours, st, _ = TR._bait(ctx, K, 0, d=d, w=w, n_cand=ctx.N, return_state=True)
        assert np.array_equal(np.sort(ref), ours), (w, K)
        # 前向 2K + 后向删到 K（原文默认）
        ref2 = _ref_bait_select(Xemb, K, s['I_U'], s['I_F'], lamb, nL, over_sample=2)
        ours2 = TR._bait(ctx, K, 0, d=d, w=w, n_cand=ctx.N, backward=True)
        assert np.array_equal(np.sort(ref2), ours2), (w, K)


def test_bait_objective_beats_baselines():
    """BAIT 的 A-最优目标 tr(M_S^{-1} I_U) 应低于随机样本与 margin top-k。"""
    ctx = toy()
    k = 50
    s = TR.bait_setup(ctx)
    ob = TR.bait_objective(ctx, REGISTRY['bait'](ctx, k, 0), setup=s)
    rng = np.random.default_rng(0)
    orand = np.mean([TR.bait_objective(ctx, rng.choice(ctx.N, k, replace=False), setup=s) for _ in range(5)])
    omarg = TR.bait_objective(ctx, np.argsort(-unc_margin(ctx.P_T))[:k], setup=s)
    o0 = TR.bait_objective(ctx, np.zeros(0, dtype=int), setup=s)
    assert ob < orand < o0 and ob < omarg, (ob, orand, omarg, o0)
    # w 越大，新点相对外国 Fisher 的作用越小：目标的相对下降越少
    drops = []
    for name in ('bait', 'bait_wk10', 'bait_wk'):
        w = {'bait': 1.0, 'bait_wk10': k / 10, 'bait_wk': float(k)}[name]
        sw = TR.bait_setup(ctx, w=w)
        drops.append(1 - TR.bait_objective(ctx, REGISTRY[name](ctx, k, 0), setup=sw)
                     / TR.bait_objective(ctx, np.zeros(0, dtype=int), setup=sw))
    assert drops[0] > drops[1] > drops[2], drops


def test_bait_candidates_large_k():
    """k > 200 时候选只取 margin 最高的 1 万个（这里用 n_cand 缩小验证同一逻辑）。"""
    ctx = toy()
    idx = REGISTRY['bait'](ctx, 60, 0, n_cand=500)
    top = np.argsort(-unc_margin(ctx.P_T), kind='stable')[:500]
    assert np.isin(idx, top).all()


# ---------------------------------------------------------------------
# M8 Hybrid
# ---------------------------------------------------------------------

def test_hybrid_design_meta():
    ctx = toy()
    from samplers.designs import alloc
    for k, rho in ((50, 0.5), (200, 0.3), (200, 0.5)):
        idx = REGISTRY['hybrid'](ctx, k, 0, rho=rho, **FAST_LGBM)
        m = ctx.meta['hybrid']
        k0 = max(10, round(rho * k))
        pi = m['pi']
        assert m['k0'] == k0 and len(m['prior_idx']) == k0
        assert pi.shape == (ctx.N,) and (pi >= 0).all() and (pi <= 1).all()
        assert abs(pi.sum() - k0) < 1e-6, (pi.sum(), k0)
        assert (pi[m['prior_idx']] > 0).all()
        S1, S2 = m['prior_idx'], m['S2']
        assert len(np.intersect1d(S1, S2)) == 0 and len(S2) == k - k0
        assert np.array_equal(np.sort(np.r_[S1, S2]), idx)
        # 第一批按外国预测类分层：各层样本数 = sqrt 分配（m_min=2），π_i = n_h/N_h
        yhat = ctx.P_T.argmax(1)
        N_h = np.bincount(yhat, minlength=ctx.C)
        n_h = alloc(N_h, np.sqrt(N_h), k0, 2)
        assert np.array_equal(np.bincount(yhat[S1], minlength=ctx.C), n_h)
        assert np.allclose(pi, (n_h / np.maximum(N_h, 1))[yhat])
        assert 'predstrat' not in ctx.meta   # 第一批在副本上调用，不污染 M3 的 meta


def test_hybrid_rho1_equals_predstrat():
    """ρ=1 即 M3：不训练、不做第二批，结果与同一 seed 的 'predstrat' 相同。"""
    ctx = toy()
    from samplers import designs  # noqa: F401  （登记 predstrat）
    a = REGISTRY['hybrid'](ctx, 100, 7, rho=1.0, fit_predict=_raise)
    b = REGISTRY['predstrat'](ctx, 100, 7)
    assert np.array_equal(a, b)
    assert len(ctx.meta['hybrid']['S2']) == 0


def test_hybrid_prior_estimate_and_correction():
    ctx = toy()
    yT = ctx._true_y
    true_prior = np.bincount(yT, minlength=ctx.C) / ctx.N
    # Olofsson 估计：大样本时各类都在 4 个标准误以内
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        REGISTRY['hybrid'](ctx, 600, 0, rho=1.0)
    m = ctx.meta['hybrid']
    z = np.abs(m['p_hat'] - true_prior) / np.maximum(m['p_hat_se'], 1e-9)
    assert (z < 4).all(), z
    assert abs(m['p_hat'].sum() - 1) < 1e-9
    # 先验校正（uncertainty.prior_adjust）：p̂ = p_tr 时不变；行和为 1；按 p̂/p_tr 改变列比例
    from samplers.uncertainty import prior_adjust
    P = ctx.P_T
    assert np.allclose(prior_adjust(P, np.full(10, .1), np.full(10, .1)), P)
    p_hat = np.linspace(1, 2, 10)
    p_hat /= p_hat.sum()
    Qc = prior_adjust(P, p_hat, np.full(10, .1))
    assert np.allclose(Qc.sum(1), 1)
    assert np.allclose((Qc[:, 9] / Qc[:, 0]) / (P[:, 9] / P[:, 0]), p_hat[9] / p_hat[0])


def test_hybrid_stage2_uses_corrected_probs():
    """注入 fit_predict：第二批应恰为在未选池上、以先验校正后的概率调用 M4 margin / M7 BADGE / M6 GIST 的结果。"""
    from samplers import uncertainty as U
    ctx = toy()
    r = np.random.default_rng(3)
    Pfit = ctx.P_T * r.uniform(0.5, 1.5, size=ctx.P_T.shape)
    Pfit /= Pfit.sum(1, keepdims=True)
    p_tr = np.full(ctx.C, 1 / ctx.C)
    seen = {}

    def fit_predict(c, S, y, sd):
        seen['n'] = len(S)
        assert np.array_equal(y, c._true_y[S]) if hasattr(c, '_true_y') else True
        return Pfit, p_tr
    k = 120
    idx = REGISTRY['hybrid_margin'](ctx, k, 0, fit_predict=fit_predict)
    m = ctx.meta['hybrid_margin']
    assert seen == {'n': 60}
    Pc = U.prior_adjust(Pfit, m['p_hat'], p_tr)
    rest = np.setdiff1d(np.arange(ctx.N), m['prior_idx'])
    u = unc_margin(Pc[rest])
    expect = np.sort(rest[np.argsort(-u, kind='stable')[:k - 60]])
    assert np.array_equal(m['S2'], expect)
    assert np.array_equal(idx, np.sort(np.r_[m['prior_idx'], expect]))
    # BADGE / GIST：与在同一子池 ctx 上直接调用 uncertainty 的实现一致
    for name, fn in (('hybrid', U.select_badge), ('hybrid_gist', U.select_gist_margin)):
        REGISTRY[name](ctx, k, 0, fit_predict=fit_predict)
        m = ctx.meta[name]
        rest = np.setdiff1d(np.arange(ctx.N), m['prior_idx'])
        seed2 = int(np.random.SeedSequence([0, 8]).generate_state(2)[1]) % (2 ** 31)
        sub = TR._subctx(ctx, rest, U.prior_adjust(Pfit, m['p_hat'], p_tr))
        assert np.array_equal(m['S2'], np.sort(rest[fn(sub, k - 60, seed2)])), name


def test_hybrid_default_fit():
    """默认训练钩子（uncertainty.fit_predict_lgbm）确实被调用：p_tr 是外国 ∪ S1 的加权类别比例（外国类别平衡权重下近似均匀）。"""
    ctx = toy()
    REGISTRY['hybrid'](ctx, 100, 0, **FAST_LGBM)
    m = ctx.meta['hybrid']
    assert m['p_tr'].shape == (ctx.C,) and abs(m['p_tr'].sum() - 1) < 1e-9
    assert np.abs(m['p_tr'] - 0.1).max() < 0.05, m['p_tr']


# ---------------------------------------------------------------------
# 耗时：N=30000、k=1000（另测 BAIT k=200 的全池情形）
# ---------------------------------------------------------------------

def test_runtime_N30k():
    if os.environ.get('SKIP_SLOW'):
        print('  （SKIP_SLOW：跳过）')
        return
    ctx = toy(N=30000, n_F=50000)
    # bait_wk10 / bait_wk 与 bait 同代价，hybrid_margin 比 hybrid 便宜，不重复计时
    for name, k in (('tqs', 1000), ('tqs_u', 1000), ('bait', 50), ('bait', 200), ('bait', 1000),
                    ('bait_backward', 1000), ('hybrid', 1000), ('hybrid_gist', 1000)):
        t0 = time.time()
        idx = REGISTRY[name](ctx, k, 0)
        dt = time.time() - t0
        print(f'  {name:8s} k={k:5d}: {dt:6.1f} s')
        assert len(idx) == k
        assert dt < 180, (name, k, dt)


if __name__ == '__main__':
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_') and callable(f)]
    n_ok = 0
    for n, f in tests:
        t0 = time.time()
        try:
            f()
            n_ok += 1
            print(f'PASS {n} ({time.time() - t0:.1f}s)', flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f'FAIL {n}: {e!r}', flush=True)
    print(f'{n_ok}/{len(tests)} passed')
    sys.exit(0 if n_ok == len(tests) else 1)
