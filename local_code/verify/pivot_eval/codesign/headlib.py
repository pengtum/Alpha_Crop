"""学习器-选样协同设计诊断的公共函数（只读项目与结果文件；输出只写到本目录）。
思路：外国模型（阶段 0，k=0，已缓存的温度校准概率 P_T）冻结不动，只用 k 个目标标签拟合一个低维"目标头"。
这样不需要重训 LightGBM，11 国 × 2 划分 × 2 重复 × 全部 G 选样都能在几分钟内评估。"""
import os

for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
          'NUMEXPR_NUM_THREADS'):
    os.environ[v] = '2'

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

GOUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
CACHE = '/Users/perng_mini/AlphaCrops_local/data/cache_guidance_G'
OUT = '/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/codesign'
COUNTRIES = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']
C = 10
MIN_N = 30
N_POOL = 30_000


def pool_positions(S, split, rep):
    """与 guidance_G.pool_positions 逐位一致。"""
    n_cand = len(S['cand'])
    rng = np.random.default_rng(10_000 * split + 100 * rep + 7)
    return np.sort(rng.choice(n_cand, min(N_POOL, n_cand), replace=False))


def load_stage0(tgt, split):
    return dict(np.load(os.path.join(GOUT, 'stage0', f'{tgt}_s{split}.npz')))


def load_sel(unit, cond, method, k):
    p = os.path.join(GOUT, 'sel', f'{unit}_{cond}_{method}_k{k}.npz')
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return z['sel_pos'], z['sel']


def scores(yT, ph, big):
    rec = np.array([(ph[yT == c] == c).mean() if (yT == c).any() else np.nan for c in range(C)])
    return 100 * float((ph == yT).mean()), 100 * float(np.nanmean(rec[big]))


def logp(P):
    return np.log(np.clip(np.asarray(P, dtype=np.float64), 1e-9, None))


def _softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def sample_weights(y, alpha):
    """w_i ∝ n_{y_i}^{-alpha}；alpha=0 普通 CE（学目标先验，偏 OA），alpha=1 类别均衡（偏 BA）。"""
    n = np.bincount(y, minlength=C).astype(float)
    w = n[y] ** (-alpha)
    return w / w.sum()


def fit_head(Z, y, kind='bias', X=None, lam=0.01, alpha=1.0, freeze_absent=True):
    """目标头：logits = s·Z + b (+ Z W) (+ X V)，Z = 外国模型 log 概率（冻结偏移）。
    参数向 0 收缩（即向外国模型收缩）；加权 CE + lam·‖θ‖²。返回 predict(Z', X') -> 概率。
    kind: 'bias'（10 参）、'bcts'（温度+偏置，11 参）、'mat'（+10×10 残差矩阵）、'feat'（偏置 + X 的线性残差）。
    freeze_absent：标注集中没出现的类，其偏置与残差参数固定为 0。"""
    y = np.asarray(y)
    k = len(y)
    w = sample_weights(y, alpha)
    present = np.bincount(y, minlength=C) > 0
    act = present if freeze_absent else np.ones(C, bool)
    Y = np.zeros((k, C))
    Y[np.arange(k), y] = 1
    dX = 0 if X is None else X.shape[1]
    use_s = kind == 'bcts'
    use_W = kind == 'mat'
    use_V = kind == 'feat'
    n_b = C
    n_W = C * C if use_W else 0
    n_V = dX * C if use_V else 0
    n = 1 + n_b + n_W + n_V

    def unpack(th):
        a = th[0] if use_s else 0.0
        b = th[1:1 + n_b] * act
        W = th[1 + n_b:1 + n_b + n_W].reshape(C, C) * act[None, :] if use_W else None
        V = th[1 + n_b + n_W:].reshape(dX, C) * act[None, :] if use_V else None
        return a, b, W, V

    def logits(th, Zq, Xq):
        a, b, W, V = unpack(th)
        L = np.exp(a) * Zq + b
        if W is not None:
            L = L + Zq @ W
        if V is not None:
            L = L + Xq @ V
        return L

    def f(th):
        a, b, W, V = unpack(th)
        L = logits(th, Z, X)
        P = _softmax(L)
        loss = -(w * np.log(np.clip(P[np.arange(k), y], 1e-300, None))).sum()
        G = (P - Y) * w[:, None]  # dLoss/dL
        g = np.zeros(n)
        if use_s:
            g[0] = np.exp(a) * (G * Z).sum()
        g[1:1 + n_b] = G.sum(0) * act
        if use_W:
            g[1 + n_b:1 + n_b + n_W] = ((Z.T @ G) * act[None, :]).ravel()
        if use_V:
            g[1 + n_b + n_W:] = ((X.T @ G) * act[None, :]).ravel()
        loss += lam * (th @ th)
        g += 2 * lam * th
        return loss, g

    th0 = np.zeros(n)
    r = minimize(f, th0, jac=True, method='L-BFGS-B', options=dict(maxiter=500))
    th = r.x

    def predict(Zq, Xq=None):
        return _softmax(logits(th, Zq, Xq))
    predict.theta = th
    return predict


def fit_gbft(Z, y, X, Zt, Xt, alpha=1.0, rounds=60, leaves=4, lr=0.05, min_leaf=5, seed=0):
    """GBDT 微调：init_score = 外国 log 概率，只在 k 个目标样本上续训少量浅树。返回测试集概率。"""
    import lightgbm as lgb
    w = sample_weights(np.asarray(y), alpha) * len(y)
    pr = dict(objective='multiclass', num_class=C, learning_rate=lr, num_leaves=leaves, min_data_in_leaf=min_leaf,
              min_sum_hessian_in_leaf=1e-3, lambda_l2=1.0, num_threads=2, seed=seed, deterministic=True,
              force_col_wise=True, verbose=-1, max_bin=63, feature_fraction=0.8, bagging_fraction=1.0)
    ds = lgb.Dataset(X, y, weight=w, init_score=Z, free_raw_data=True)
    m = lgb.train(pr, ds, num_boost_round=rounds)
    raw = m.predict(Xt, raw_score=True)
    return _softmax(Zt + raw)
