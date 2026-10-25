"""欠/过自信对 PO、EM 的影响；温度缩放 / BCTS（Alexandari et al. 2020）在源留出集上校准后是否修复。"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, lightgbm as lgb
from scipy.optimize import minimize
from sklearn.utils.class_weight import compute_sample_weight
from prior_shift_ref import *
C, d = 10, 8
pi_T = np.array([38471, 10721, 1122, 2304, 2635, 2415, 139, 664, 1576, 7589], float); pi_T /= pi_T.sum()
pi_F = np.array([44972, 4569, 6186, 18940, 930, 320, 78, 3520, 1840, 8645], float); pi_F /= pi_F.sum()
def make(rng, n, pi, mu, shift=0.0):
    y = rng.choice(C, n, p=pi); return mu[y] + rng.normal(size=(n, d)) + shift, y
P = dict(n_estimators=500, learning_rate=0.05, num_leaves=63, min_child_samples=20, class_weight=None,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=2, random_state=42, verbose=-1)

def fit_bcts(p, y, w, bias=True):
    """在源留出集上最小化加权 NLL：q ∝ exp(log p / T + b)。bias=False 即温度缩放。只在 p>0 的列上。"""
    L = np.log(np.clip(p, 1e-300, None))
    def nll(th):
        T = np.exp(th[0]); b = np.r_[0.0, th[1:]] if bias else 0.0
        Z = L / T + b; Z -= Z.max(1, keepdims=True)
        lq = Z - np.log(np.exp(Z).sum(1, keepdims=True))
        return -(w * lq[np.arange(len(y)), y]).sum() / w.sum()
    th0 = np.zeros(1 + (p.shape[1] - 1 if bias else 0))
    th = minimize(nll, th0, method='L-BFGS-B').x
    T = np.exp(th[0]); b = np.r_[0.0, th[1:]] if bias else np.zeros(p.shape[1])
    def apply(pp):
        Z = np.log(np.clip(pp, 1e-300, None)) / T + b; Z -= Z.max(1, keepdims=True)
        q = np.exp(Z); return q / q.sum(1, keepdims=True)
    return apply, T

rng = np.random.default_rng(7)
mu = rng.normal(scale=1.0, size=(C, d))
Xs, ys = make(rng, 30_000, pi_F, mu); Xsv, ysv = make(rng, 3_000, pi_F, mu); Xsh, ysh = make(rng, 10_000, pi_F, mu)
Xp, yp = make(rng, 20_000, pi_T, mu); Xt, yt = make(rng, 20_000, pi_T, mu)
for lr, es in ((0.05, 50), (0.3, None)):
    kw = dict(eval_set=[(Xsv, ysv)], eval_sample_weight=[compute_sample_weight('balanced', ysv)],
              callbacks=[lgb.early_stopping(es, verbose=False)]) if es else {}
    m = lgb.LGBMClassifier(**{**P, 'learning_rate': lr}).fit(Xs, ys, sample_weight=compute_sample_weight('balanced', ys), **kw)
    f = lambda X: m.predict_proba(X)
    ph, pp, pt = f(Xsh), f(Xp), f(Xt)
    wsh = compute_sample_weight('balanced', ysh)
    piT = np.bincount(yp, minlength=C) / len(yp); u = np.full(C, 0.1)
    print(f'\n== lr={lr}, early_stop={es}, n_iter={m.booster_.current_iteration()} ==')
    for name, cal in (('未校准', None), ('温度缩放', False), ('BCTS', True)):
        if cal is None:
            g, T = (lambda z: z), 1.0
        else:
            g, T = fit_bcts(ph, ysh, wsh, bias=cal)
        imp = (g(ph) * wsh[:, None]).sum(0) / wsh.sum()        # 模型在源（加权）上"内含"的先验
        q0, qp = g(pt), g(pp)
        po = metrics(yt, reweight(q0, piT, u)[0].argmax(1), C)['oa']
        po_imp = metrics(yt, reweight(q0, piT, imp)[0].argmax(1), C)['oa']
        pe, it, conv, _ = em_prior(qp, u)
        em = metrics(yt, reweight(q0, pe, u)[0].argmax(1), C)['oa']
        pe2, it2, _, _ = em_prior(qp, imp)
        em2 = metrics(yt, reweight(q0, pe2, imp)[0].argmax(1), C)['oa']
        l0 = metrics(yt, q0.argmax(1), C)['oa']
        print(f'  {name:6s} T={T:5.2f} 内含先验范围[{imp.min():.3f},{imp.max():.3f}]  L0={l0:.4f}  PO(π_tr=均匀)={po:.4f}  '
              f'PO(π_tr=内含)={po_imp:.4f}  PEM(均匀)={em:.4f} |Δπ|₁={np.abs(pe-piT).sum():.3f}  '
              f'PEM(内含)={em2:.4f} |Δπ|₁={np.abs(pe2-piT).sum():.3f}')
    print(f'  （上限参照：真实后验下 PO 的 OA 见 test_prior_correction A 部分）')
