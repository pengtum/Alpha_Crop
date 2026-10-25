"""
检验 3 方法审查：合成数据小实验（不接触真实数据与 check3 结果）。

问题：
  (a) 类别平衡权重训练的 LightGBM，其隐含训练先验是否 ≈ 均匀；概率是否校准；
  (b) PO / EM / BBSE 在纯先验偏移、以及先验 + 条件偏移下能恢复多少；BCTS 校准的作用；
  (c) 2×2（条件 F/T × 决策先验 均匀/π_T）的路径依赖与交互项；
  (d) Laplace+1 平滑在 k=50 时的收缩效应。
设定仿照真实数据：10 类，源先验 = 外国池（AT 目标时的计数），目标先验 = SE / PT 型。
"""
import sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import lightgbm as lgb
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

C, D = 10, 12
SRC_COUNTS = np.array([49773, 5960, 6285, 20694, 1360, 668, 100, 3358, 2105, 9697], float)
TGT = {
    'SE': np.array([23773, 127, 44, 4384, 177, 92, 26, 187, 310, 1997], float),
    'PT': np.array([10448, 2261, 15150, 3047, 274, 10, 1, 1535, 1297, 106], float),
}
PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=2, verbose=-1)


def make_world(seed, spread=1.1):
    r = np.random.default_rng(seed)
    means = r.normal(0, spread, (C, D))
    dirs = r.normal(0, 1, (C, D)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return means, dirs


def sample(means, prior, n, r, shift=None):
    y = r.choice(C, n, p=prior / prior.sum())
    mu = means if shift is None else means + shift
    X = mu[y] + r.normal(0, 1, (n, D))
    return X.astype(np.float32), y


def bayes_post(X, means, prior):
    ll = np.stack([multivariate_normal(means[c], np.eye(D)).logpdf(X) for c in range(C)], 1)
    lp = ll + np.log(np.maximum(prior, 1e-300))
    return softmax(lp, 1)


def fit(X, y, w, seed):
    cnt = np.bincount(y, minlength=C)
    strat = np.where(cnt[y] >= 2, y, -1)
    if (strat == -1).sum() == 1: strat[strat == -1] = cnt.argmax()
    tr, va = train_test_split(np.arange(len(y)), test_size=0.1, random_state=seed, stratify=strat)
    m = lgb.LGBMClassifier(**PARAMS, random_state=seed)
    m.fit(X[tr], y[tr], sample_weight=None if w is None else w[tr], eval_set=[(X[va], y[va])],
          eval_sample_weight=None if w is None else [w[va]],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return m, va


def fp(m, X):
    p = np.zeros((len(X), C)); p[:, m.classes_] = m.predict_proba(X); return p


def po(p, pi_t, pi_tr):
    q = p * (pi_t / pi_tr)
    return q / q.sum(1, keepdims=True)


def em(p, pi_tr, iters=200, tol=1e-6):
    pi = pi_tr.copy()
    for _ in range(iters):
        q = po(p, pi, pi_tr); new = q.mean(0)
        if np.abs(new - pi).max() < tol:
            pi = new; break
        pi = new
    return pi


def bcts_fit(p, y, w):
    """bias-corrected temperature scaling：softmax(log p / T + b)，最小化加权 NLL。"""
    L = np.log(np.clip(p, 1e-12, 1))
    def nll(th):
        T = np.exp(th[0]); b = np.r_[0.0, th[1:]]
        z = L / T + b
        return -(w * (z[np.arange(len(y)), y] - logsumexp(z, 1))).sum() / w.sum()
    th = minimize(nll, np.zeros(C), method='L-BFGS-B').x
    return np.exp(th[0]), np.r_[0.0, th[1:]]


def bcts_apply(p, T, b):
    return softmax(np.log(np.clip(p, 1e-12, 1)) / T + b, 1)


def bbse(p_src, y_src, w_src, p_tgt, pi_src):
    yh = p_src.argmax(1)
    Cm = np.zeros((C, C))
    np.add.at(Cm, (yh, y_src), w_src); Cm /= w_src.sum()          # P_S(ŷ=i, y=j)
    mu = np.bincount(p_tgt.argmax(1), minlength=C) / len(p_tgt)
    wv = np.linalg.lstsq(Cm, mu, rcond=None)[0]
    pi = np.clip(wv * pi_src, 0, None)
    return pi / pi.sum()


def oa(p, y): return (p.argmax(1) == y).mean()


def ba(p, y):
    yh = p.argmax(1); return np.mean([(yh[y == c] == c).mean() for c in np.unique(y)])


def ece(p, y, bins=15):
    conf = p.max(1); acc = (p.argmax(1) == y)
    e = 0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        s = (conf > lo) & (conf <= lo + 1 / bins)
        if s.any(): e += s.mean() * abs(acc[s].mean() - conf[s].mean())
    return e


def run(seed, tname, delta, n_src=30000, n_pool=10000, n_test=10000):
    r = np.random.default_rng(1000 + seed)
    means, dirs = make_world(seed)
    shift = None if delta == 0 else delta * dirs
    piS = SRC_COUNTS / SRC_COUNTS.sum(); piT = TGT[tname] / TGT[tname].sum()
    Xs, ys = sample(means, piS, n_src, r)
    Xp, yp = sample(means, piT, n_pool, r, shift)
    Xt, yt = sample(means, piT, n_test, r, shift)
    U = np.ones(C) / C
    ws = compute_sample_weight('balanced', ys)
    loco, va = fit(Xs, ys, ws, seed)
    ps_va = fp(loco, Xs[va]); w_va = ws[va]
    implied = (w_va[:, None] * ps_va).sum(0) / w_va.sum()          # 源验证集（平衡加权）上的隐含训练先验
    Tc, bc = bcts_fit(ps_va, ys[va], w_va)
    pp, pt = fp(loco, Xp), fp(loco, Xt)
    ppc, ptc = bcts_apply(pp, Tc, bc), bcts_apply(pt, Tc, bc)
    res = {}
    res['L0'] = pt
    S = np.bincount(yp, minlength=C) > 0
    res['LS'] = (pt * S) / (pt * S).sum(1, keepdims=True)
    piT_pool = np.bincount(yp, minlength=C) / len(yp)          # 与真实方案一致：π_T 取 pool 的经验比例
    res['PO'] = po(pt, piT_pool, U)
    res['PO_implied'] = po(pt, piT_pool, implied)
    res['PO_bcts'] = po(ptc, piT_pool, U)
    pi_em = em(pp, U); res['PEM'] = po(pt, pi_em, U)
    pi_emc = em(ppc, U); res['PEM_bcts'] = po(ptc, pi_emc, U)
    pi_bb = bbse(ps_va, ys[va], w_va, pp, U); res['BBSE'] = po(pt, pi_bb, U)
    # k=50 Laplace+1
    rk = np.random.default_rng(seed)
    kk = rk.choice(n_pool, 50, replace=False)
    pik = (np.bincount(yp[kk], minlength=C) + 1) / (50 + C); res['PK50_lap1'] = po(pt, pik, U)
    pik5 = (np.bincount(yp[kk], minlength=C) + 0.5) / (50 + 0.5 * C); res['PK50_lap.5'] = po(pt, pik5, U)
    # 目标标签拟合 BCTS（"决策层修正的上界"：只调温度与 10 个偏置，用 pool 标签）
    To, bo = bcts_fit(pp, yp, np.ones(len(yp))); res['BIAS_ORACLE'] = bcts_apply(pt, To, bo)
    # WITHIN
    wp = compute_sample_weight('balanced', yp)
    w0, _ = fit(Xp, yp, wp, seed); pw = fp(w0, Xt)
    res['W0'] = pw; res['WPO'] = po(pw, piT_pool, U)  # 缺失类的 p=0，常数 π_train 不影响 argmax
    wn, _ = fit(Xp, yp, None, seed); res['Wnat'] = fp(wn, Xt)
    # 小样本 WITHIN（样本量混杂）
    sub = r.choice(n_pool, 3000, replace=False)
    w3, _ = fit(Xp[sub], yp[sub], compute_sample_weight('balanced', yp[sub]), seed)
    res['W0_n3k'] = fp(w3, Xt); res['WPO_n3k'] = po(res['W0_n3k'], piT_pool, U)
    # Bayes（真实生成模型，目标条件分布、目标先验）
    mu_t = means if shift is None else means + shift
    res['BAYES_T'] = bayes_post(Xt, mu_t, piT)
    out = {k: (oa(v, yt), ba(v, yt)) for k, v in res.items()}
    diag = dict(implied_max_dev=np.abs(implied - U).max(), implied_min=implied.min(), implied_max=implied.max(),
                T_bcts=Tc, ece_src=ece(bcts_apply(ps_va, 1, np.zeros(C)), ys[va]), ece_tgt=ece(pt, yt),
                ece_tgt_bcts=ece(ptc, yt), em_l1=np.abs(pi_em - piT).sum(), emc_l1=np.abs(pi_emc - piT).sum(),
                bb_l1=np.abs(pi_bb - piT).sum(), iters=loco.best_iteration_)
    return out, diag


if __name__ == '__main__':
    seeds = [0, 1]
    import pandas as pd
    rows, drows = [], []
    for tname in ['SE', 'PT']:
        for delta in [0.0, 1.0]:
            for s in seeds:
                out, diag = run(s, tname, delta)
                for k, (a, b) in out.items():
                    rows.append(dict(T=tname, delta=delta, seed=s, arm=k, OA=a, BA=b))
                drows.append(dict(T=tname, delta=delta, seed=s, **diag))
                print(tname, delta, s, 'done', flush=True)
    R = pd.DataFrame(rows); Dg = pd.DataFrame(drows)
    R.to_csv('/Users/perng_mini/AlphaCrops_local/verify/check3_review/synth_arms.csv', index=False)
    Dg.to_csv('/Users/perng_mini/AlphaCrops_local/verify/check3_review/synth_diag.csv', index=False)
    pd.set_option('display.width', 250)
    print((R.groupby(['T', 'delta', 'arm'])[['OA', 'BA']].mean() * 100).round(1).unstack(['T', 'delta']).to_string())
    print(Dg.groupby(['T', 'delta']).mean().round(3).to_string())
