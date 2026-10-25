"""检验 3 的先验修正工具函数（全部在 float64 上计算）。来源：方法审查 agent 编写并通过单元测试的参考实现（verify/check3_review/prior_shift_ref.py）。

约定：p 为 (n, C) 概率矩阵，列 = 10 个全局类别；训练时没见过的类，该列恒为 0。
trained = 训练集中出现的类（bool, C）；pi_train = 模型"内含"的训练先验（类别平衡 → trained 上均匀）。
"""
import numpy as np


def as64(p):
    p = np.asarray(p, dtype=np.float64)
    s = p.sum(1, keepdims=True)
    assert np.all(np.abs(s - 1) < 1e-4), '行和偏离 1（float32 存储误差应 < 1e-6）'
    return p / s                                   # 消除 float32 舍入


def trained_mask(p, atol=0.0):
    """未训练类的列恒为精确 0（full_proba 的构造），据此推断训练类别集合。"""
    return p.max(0) > atol


def uniform_prior(trained):
    return trained / trained.sum()


def reweight(p, pi_new, pi_train, fallback='keep'):
    """q(y|x) ∝ p(y|x)·pi_new(y)/pi_train(y)；pi_train=0 的类 q=0。
    行和为 0（所有有概率的类都被 pi_new 置零）时：fallback='keep' 保留原 p；返回 (q, 回退行数)。"""
    w = np.divide(pi_new, pi_train, out=np.zeros_like(pi_new, dtype=np.float64), where=pi_train > 0)
    q = p * w
    s = q.sum(1, keepdims=True)
    bad = (s[:, 0] <= 0) | ~np.isfinite(s[:, 0])
    q = np.where(bad[:, None], p, q / np.where(bad[:, None], 1.0, s))
    return q, int(bad.sum())


def label_set(p, present, pi_train):
    """LS：T 中不存在的类置零后重新归一化（等价于 pi_new = 1[present] 的 reweight）。"""
    return reweight(p, present.astype(np.float64), pi_train)


def em_prior(p, pi_train, max_iter=200, tol=1e-6, floor=0.0):
    """Saerens, Latinne & Decaestecker (2002) EM，无标签估计目标先验。
    返回 pi_hat, n_iter, converged, loglik 轨迹（应单调不减）。
    - 只在 pi_train>0 的类上迭代；其余类恒为 0（避免 0/0）。
    - 初值 = pi_train（Saerens 原文）；收敛判据 max|Δπ| < tol。
    - 目标中不存在的类，π̂ 以几何速度趋于 0 但不会精确为 0；200 次迭代内不会下溢到 subnormal，
      但任何 log(π̂) 都要先 clip；floor>0 只在需要取 log 的场景使用。"""
    k = pi_train > 0
    P = p[:, k]
    pt = pi_train[k]
    pi = pt.copy()
    ll = []
    for it in range(1, max_iter + 1):
        num = P * (pi / pt)
        s = num.sum(1, keepdims=True)
        ll.append(float(np.log(np.maximum(s[:, 0], 1e-300)).sum()))
        new = (num / np.maximum(s, 1e-300)).mean(0)
        if floor > 0:
            new = np.maximum(new, floor); new /= new.sum()
        done = np.max(np.abs(new - pi)) < tol
        pi = new
        if done:
            break
    out = np.zeros_like(pi_train, dtype=np.float64); out[k] = pi
    return out, it, bool(done), np.array(ll)


def laplace_prior(y_lab, C, alpha=1.0):
    """PK：k 个带标签样本的类别比例 + Laplace 平滑（对全部 C 类加 alpha）。"""
    n = np.bincount(y_lab, minlength=C).astype(np.float64)
    return (n + alpha) / (n.sum() + alpha * C)


def bbse_prior(p_src_val, y_src_val, w_src_val, p_tgt, trained, soft=False, eps_cond=1e-8):
    """Lipton, Wang & Smola (2018) BBSE。
    源域"带标签的留出集"（不能是训练集，也最好不是早停集）→ 混淆矩阵 Cm[i,j] = P_S(ŷ=i, y=j)，
    用与训练一致的样本权重（类别平衡）使 P_S(y) = pi_train；
    目标域无标签 → mu[i] = P_T(ŷ=i)；解 Cm·w = mu 得 w = π_T/π_S，π̂_T = w·π_S。
    soft=True 时用概率代替硬预测（软混淆矩阵，方差更小）。
    负值裁到 0 后重新归一化；报告条件数。"""
    k = np.flatnonzero(trained)
    Ps, Pt = p_src_val[:, k], p_tgt[:, k]
    ys = np.searchsorted(k, y_src_val)                    # 源标签映射到 trained 子空间（源标签必在 trained 内）
    ws = w_src_val / w_src_val.sum()
    if soft:
        Hs, Ht = Ps, Pt
    else:
        Hs = np.eye(len(k))[Ps.argmax(1)]; Ht = np.eye(len(k))[Pt.argmax(1)]
    Y = np.eye(len(k))[ys]
    Cm = (Hs * ws[:, None]).T @ Y                          # (pred i, true j)
    pi_S = Y.T @ ws                                        # = 源（加权）先验
    mu = Ht.mean(0)
    cond = np.linalg.cond(Cm)
    w, *_ = np.linalg.lstsq(Cm + eps_cond * np.eye(len(k)), mu, rcond=None)
    pi_hat = np.clip(w * pi_S, 0, None)
    pi_hat = pi_hat / pi_hat.sum()
    out = np.zeros(len(trained)); out[k] = pi_hat
    return out, cond, w


# ---------- 指标（与项目约定一致时替换为项目函数） ----------
def recall_per_class(y, yhat, C):
    return np.array([(yhat[y == c] == c).mean() if (y == c).any() else np.nan for c in range(C)])


def metrics(y, yhat, C, min_n=30):
    from sklearn.metrics import f1_score
    r = recall_per_class(y, yhat, C)
    n = np.bincount(y, minlength=C)
    present = np.flatnonzero(n > 0)
    return dict(oa=float((y == yhat).mean()),
                ba=float(np.nanmean(r)),
                ba_ge30=float(np.mean(r[n >= min_n])),
                f1_present=float(f1_score(y, yhat, labels=present, average='macro', zero_division=0)),
                f1_union=float(f1_score(y, yhat, average='macro', zero_division=0)))
