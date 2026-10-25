# 两视图目标校准层（XV-Cal）：z_c = wa_c·log Pa_c + ws_c·log Ps_c + b_c，只用目标国 k 个标签拟合（30 个参数）。
import numpy as np
from scipy.optimize import minimize

def _ll(P):
    return np.log(np.clip(P, 1e-6, None))

def class_w(y, C=10, mode='bal', cap=20.0):
    n = np.bincount(y, minlength=C).astype(float)
    if mode == 'unw':
        return np.ones(C)
    a = 1.0 if mode == 'bal' else float(mode)   # 权重 ∝ n_c^(−a)：a=0 不加权，a=1 类别均衡
    w = np.where(n > 0, (np.maximum(n, 1) / len(y)) ** (-a), 0.0)
    w = w / (w[n > 0] * n[n > 0]).sum() * len(y)
    return np.minimum(w, cap)

def fit(La, Ls, y, mode='bal', gw=5.0, gb=1.0, views='both', C=10, sw=None):
    """La, Ls：log 概率 (n, C)。views: 'both' / 'ae' / 's2'。返回参数向量 (wa, ws, b)。"""
    if sw is None:
        om = class_w(y, C, mode)[y]
    else:  # 设计权重（1/π）：类别权重按 HT 加权的类别总量计算
        sw = np.asarray(sw, float)
        nhat = np.bincount(y, weights=sw, minlength=C)
        a = 1.0 if mode == 'bal' else (0.0 if mode == 'unw' else float(mode))
        cwv = np.where(nhat > 0, (np.maximum(nhat, 1e-9) / nhat.sum()) ** (-a), 0.0)
        om = sw * np.minimum(cwv / (cwv[nhat > 0] * nhat[nhat > 0] / nhat.sum()).sum(), 20.0)[y]
    om = om / om.sum()
    n = len(y)
    ma = 1.0 if views in ('both', 'ae') else 0.0
    ms = 1.0 if views in ('both', 's2') else 0.0
    w0 = 0.5 if views == 'both' else 1.0
    Y = np.eye(C)[y]
    def f(th):
        wa, ws, b = th[:C] * ma, th[C:2*C] * ms, th[2*C:]
        z = La * wa + Ls * ws + b
        z = z - z.max(1, keepdims=True)
        lse = np.log(np.exp(z).sum(1, keepdims=True))
        logp = z - lse
        p = np.exp(logp)
        loss = -(om * (Y * logp).sum(1)).sum()
        pen = (gw * (((th[:C] - w0) * ma) ** 2 + ((th[C:2*C] - w0) * ms) ** 2).sum() + gb * (b ** 2).sum()) / n
        G = (p - Y) * om[:, None]
        g = np.r_[(G * La).sum(0) * ma + 2 * gw * (th[:C] - w0) * ma / n,
                  (G * Ls).sum(0) * ms + 2 * gw * (th[C:2*C] - w0) * ms / n,
                  G.sum(0) + 2 * gb * b / n]
        return loss + pen, g
    th0 = np.r_[np.full(C, w0 * ma), np.full(C, w0 * ms), np.zeros(C)]
    r = minimize(f, th0, jac=True, method='L-BFGS-B')
    th = r.x.copy(); th[:C] *= ma; th[C:2*C] *= ms
    return th

def predict(th, La, Ls, C=10):
    z = La * th[:C] + Ls * th[C:2*C] + th[2*C:]
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)

def scores(yT, P, big):
    ph = P.argmax(1)
    rec = [(ph[yT == c] == c).mean() for c in big]
    return 100 * (ph == yT).mean(), 100 * np.mean(rec)
