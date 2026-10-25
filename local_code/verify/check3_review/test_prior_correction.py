"""第 4 项：类别平衡训练得到的 p，按 q ∝ p·π_T 修正后 OA 是否升、BA 是否降。并测试 EM / BBSE / PK / LOCO-PM。"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, lightgbm as lgb
from scipy.stats import multivariate_normal
from sklearn.utils.class_weight import compute_sample_weight
from prior_shift_ref import *

C, d = 10, 8
# 先验取自真实划分的量级（BE 的 pool 比例 / 外国 f_tr 比例），其中类 6 很稀有
pi_T = np.array([38471, 10721, 1122, 2304, 2635, 2415, 139, 664, 1576, 7589], float); pi_T /= pi_T.sum()
pi_F = np.array([44972, 4569, 6186, 18940, 930, 320, 78, 3520, 1840, 8645], float); pi_F /= pi_F.sum()

def make(rng, n, pi, mu, shift=0.0):
    y = rng.choice(C, n, p=pi)
    X = mu[y] + rng.normal(size=(n, d)) + shift
    return X, y

def oracle_p(X, mu, shift=0.0):
    L = np.stack([multivariate_normal(mu[c] + shift, np.eye(d)).logpdf(X) for c in range(C)], 1)
    L -= L.max(1, keepdims=True); p = np.exp(L); return p / p.sum(1, keepdims=True)

def show(tag, y, p):
    m = metrics(y, p.argmax(1), C)
    print(f'  {tag:34s} OA={m["oa"]:.4f}  BA={m["ba"]:.4f}  ba_ge30={m["ba_ge30"]:.4f}  F1={m["f1_present"]:.4f}')
    return m

print('== A. 解析后验（均匀先验下的 Bayes 后验），目标先验 = π_T，n_test=200k ==')
for rep in range(3):
    rng = np.random.default_rng(rep)
    mu = rng.normal(scale=1.0, size=(C, d))
    X, y = make(rng, 200_000, pi_T, mu)
    p = oracle_p(X, mu)
    a = show(f'rep{rep} L0 (均匀先验)', y, p)
    b = show(f'rep{rep} PO (q ∝ p·π_T)', y, reweight(p, pi_T, np.full(C, .1))[0])
    print(f'    ΔOA={100*(b["oa"]-a["oa"]):+.2f} pp  ΔBA={100*(b["ba"]-a["ba"]):+.2f} pp  Δba_ge30={100*(b["ba_ge30"]-a["ba_ge30"]):+.2f} pp')

print('\n== B. LightGBM：外国源（先验 π_F）+ 类别平衡权重；目标（先验 π_T，可加条件偏移）==')
P = dict(n_estimators=500, learning_rate=0.05, num_leaves=63, min_child_samples=20, class_weight=None,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=2, random_state=42, verbose=-1)
def fit(X, y, w, Xv, yv, wv):
    return lgb.LGBMClassifier(**P).fit(X, y, sample_weight=w, eval_set=[(Xv, yv)], eval_sample_weight=[wv],
                                       callbacks=[lgb.early_stopping(50, verbose=False)])
def fp(m, X):
    p = np.zeros((len(X), C)); p[:, m.classes_] = m.predict_proba(X); return p

for shift in (0.0, 0.5):
    rng = np.random.default_rng(7)
    mu = rng.normal(scale=1.0, size=(C, d))
    Xs, ys = make(rng, 30_000, pi_F, mu)                      # 外国池
    Xsv, ysv = make(rng, 3_000, pi_F, mu)                     # 早停集
    Xsh, ysh = make(rng, 10_000, pi_F, mu)                    # 源留出集（BBSE 用，不参与早停）
    Xp, yp = make(rng, 20_000, pi_T, mu, shift)               # 目标 pool（无标签用于 EM）
    Xt, yt = make(rng, 20_000, pi_T, mu, shift)               # 目标 test
    m = fit(Xs, ys, compute_sample_weight('balanced', ys), Xsv, ysv, compute_sample_weight('balanced', ysv))
    pp, pt = fp(m, Xp), fp(m, Xt)
    tr = trained_mask(pt); pi_tr = uniform_prior(tr)
    wsh = compute_sample_weight('balanced', ysh)
    ps_h = fp(m, Xsh)
    print(f' -- 条件偏移 shift={shift}（best_iter={m.best_iteration_}）; 源留出集上加权平均 p = '
          f'{np.round((ps_h * wsh[:, None]).sum(0) / wsh.sum(), 3)}（理想为均匀 0.1）')
    piT_pool = np.bincount(yp, minlength=C) / len(yp)
    r = {}
    r['L0'] = show('L0', yt, pt)
    r['PO'] = show('PO  (π_T = pool 真实比例)', yt, reweight(pt, piT_pool, pi_tr)[0])
    pi_em, it, conv, ll = em_prior(pp, pi_tr)
    r['PEM'] = show(f'PEM (EM {it} 次, 收敛={conv})', yt, reweight(pt, pi_em, pi_tr)[0])
    print(f'      EM loglik 单调不减: {bool(np.all(np.diff(ll) >= -1e-6))};  |π̂_EM − π_T|_1 = {np.abs(pi_em - piT_pool).sum():.3f}')
    for soft in (False, True):
        pb, cond, _ = bbse_prior(ps_h, ysh, wsh, pp, tr, soft=soft)
        r['BB'] = show(f'BBSE{"-soft" if soft else "-hard"} (cond={cond:.0f})', yt, reweight(pt, pb, pi_tr)[0])
        print(f'      |π̂_BBSE − π_T|_1 = {np.abs(pb - piT_pool).sum():.3f}')
    for k in (50, 200):
        accs = []
        for s in range(3):
            lab = np.random.default_rng(s).choice(len(yp), k, replace=False)
            accs.append(metrics(yt, reweight(pt, laplace_prior(yp[lab], C), pi_tr)[0].argmax(1), C)['oa'])
        print(f'  PK{k:<3d} OA (3 seeds) = {np.round(accs, 4)}')
    # LOCO-PM：训练时用 π_T/π_F 重要性权重
    piF_hat = np.bincount(ys, minlength=C) / len(ys)
    ratio = piT_pool / piF_hat
    mpm = fit(Xs, ys, ratio[ys], Xsv, ysv, ratio[ysv])
    r['PM'] = show('LOCO-PM (训练时校正)', yt, fp(mpm, Xt))
    agree = (fp(mpm, Xt).argmax(1) == reweight(pt, piT_pool, pi_tr)[0].argmax(1)).mean()
    print(f'      PM vs PO 预测一致率 {agree:.3f}；ΔOA(PO−L0)={100*(r["PO"]["oa"]-r["L0"]["oa"]):+.2f} pp, '
          f'ΔBA(PO−L0)={100*(r["PO"]["ba"]-r["L0"]["ba"]):+.2f} pp, Δba_ge30={100*(r["PO"]["ba_ge30"]-r["L0"]["ba_ge30"]):+.2f} pp')
