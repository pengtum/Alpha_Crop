"""替代模型试点的公共函数（只读项目代码与结果；输出只写到本目录）。
训练集构造与 guidance_G.run_unit 相同：外国 5 万 + k 个目标样本；w = cw[y]（外国）与 r*cw[y]（目标），cw 只按外国频率。"""
import os
import sys
import time

THREADS = 4
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[v] = str(THREADS)

PROJ = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import numpy as np  # noqa: E402
import guidance_G as G  # noqa: E402
from stageA import class_weights_from, MIN_N  # noqa: E402

GOUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
OUT = '/Users/perng_mini/AlphaCrops_local/verify/model_diag/altmodels'
R_MULT = 50.0


def load_unit(D, tgt, split=0):
    S = dict(np.load(os.path.join(GOUT, 'stage0', f'{tgt}_s{split}.npz')))
    fr, test = S['foreign'], S['test']
    yT = D.y[test]
    big = np.flatnonzero(np.bincount(yT, minlength=10) >= MIN_N)
    cw = class_weights_from(D.y[fr])
    return S, fr, test, yT, big, cw


def load_sel(tgt, cond, method, k, split=0, rep=0):
    return np.load(os.path.join(GOUT, 'sel', f'{tgt}_s{split}_r{rep}_{cond}_{method}_k{k}.npz'))['sel']


def train_set(D, feat, fr, sel, cw, r=R_MULT):
    X = D.X[feat]
    sel = np.asarray(sel, dtype=np.int64)
    X_tr = np.vstack([X[fr], X[sel]]).astype(np.float32)
    y_tr = np.r_[D.y[fr], D.y[sel]]
    w_tr = np.r_[cw[D.y[fr]], r * cw[D.y[sel]]]
    is_t = np.r_[np.zeros(len(fr), bool), np.ones(len(sel), bool)]
    return X_tr, y_tr, w_tr, is_t


def scores(yT, ph, big):
    oa, ba, f1, rec = G._scores(yT, ph, big)
    return oa, ba


# ---------------------------------------------------------------- 模型
def m_lgb(X_tr, y_tr, w_tr, X_te, seed=0, trees=300, **_):
    P = G.train_predict(X_tr, y_tr, w_tr, X_te, THREADS, seed, trees)[trees]
    return P.argmax(1)


def _std(X_tr, X_te):
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6
    return (X_tr - mu) / sd, (X_te - mu) / sd


def m_hgb(X_tr, y_tr, w_tr, X_te, **_):
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=150, max_leaf_nodes=63, min_samples_leaf=20,
                                       max_bins=63, early_stopping=False, random_state=0)
    m.fit(X_tr, y_tr, sample_weight=w_tr)
    return m.predict(X_te)


def m_rf(X_tr, y_tr, w_tr, X_te, n=150, **_):
    from sklearn.ensemble import RandomForestClassifier
    m = RandomForestClassifier(n_estimators=n, max_features='sqrt', min_samples_leaf=1, n_jobs=THREADS,
                               random_state=0)
    m.fit(X_tr, y_tr, sample_weight=w_tr)
    return m.predict(X_te)


def m_et(X_tr, y_tr, w_tr, X_te, n=300, **_):
    from sklearn.ensemble import ExtraTreesClassifier
    m = ExtraTreesClassifier(n_estimators=n, max_features='sqrt', min_samples_leaf=1, n_jobs=THREADS, random_state=0)
    m.fit(X_tr, y_tr, sample_weight=w_tr)
    return m.predict(X_te)


def m_lr(X_tr, y_tr, w_tr, X_te, C=1.0, **_):
    from sklearn.linear_model import LogisticRegression
    A, B_ = _std(X_tr, X_te)
    m = LogisticRegression(C=C, max_iter=1000, tol=1e-4)
    m.fit(A, y_tr, sample_weight=w_tr / w_tr.mean())
    return m.predict(B_)


def knn_votes(X_tr, y_tr, w_tr, X_te, kmax=30):
    """标准化后欧氏距离的 kmax 近邻；返回近邻下标（按距离排序）。"""
    from sklearn.neighbors import NearestNeighbors
    A, B_ = _std(X_tr, X_te)
    nn = NearestNeighbors(n_neighbors=kmax, algorithm='brute', n_jobs=THREADS).fit(A)
    _, idx = nn.kneighbors(B_)
    return idx


def knn_predict(idx, y_tr, w_tr, kk):
    """加权投票：每个近邻的票 = 其样本权重（外国 cw[y]，目标 r*cw[y]），与其他模型的 sample_weight 同义。"""
    I = idx[:, :kk]
    V = np.zeros((len(I), 10))
    np.add.at(V, (np.repeat(np.arange(len(I)), kk), y_tr[I].ravel()), w_tr[I].ravel())
    return V.argmax(1)


def m_mlp(X_tr, y_tr, w_tr, X_te, hidden=(256, 128), **_):
    from sklearn.neural_network import MLPClassifier
    A, B_ = _std(X_tr, X_te)
    m = MLPClassifier(hidden_layer_sizes=hidden, alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
                      early_stopping=True, validation_fraction=0.1, n_iter_no_change=8, max_iter=60,
                      random_state=0)
    m.fit(A, y_tr, sample_weight=w_tr / w_tr.mean())
    return m.predict(B_), m.n_iter_


def domain_weights(Xf, Xt, seed=0, clip=(0.05, 20.0)):
    """域判别器（外国=0，目标候选池=1），5 折交叉拟合 LightGBM 给出外国行的袋外 P(目标|x)；
    w = p/(1-p) * nF/nT，归一到均值 1，截断后再归一。返回 (w, auc, ess/n)。"""
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    X = np.vstack([Xf, Xt])
    yd = np.r_[np.zeros(len(Xf), int), np.ones(len(Xt), int)]
    p = np.zeros(len(yd))
    pr = dict(objective='binary', learning_rate=0.05, num_leaves=31, num_threads=THREADS, seed=seed,
              deterministic=True, force_col_wise=True, max_bin=63, verbose=-1, min_data_in_leaf=50)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, yd):
        m = lgb.train(pr, lgb.Dataset(X[tr], yd[tr]), num_boost_round=100)
        p[te] = m.predict(X[te])
    auc = roc_auc_score(yd, p)
    pf = np.clip(p[:len(Xf)], 1e-6, 1 - 1e-6)
    w = pf / (1 - pf) * len(Xf) / len(Xt)
    w = w / w.mean()
    w = np.clip(w, *clip)
    w = w / w.mean()
    ess = w.sum() ** 2 / (w ** 2).sum() / len(w)
    return w, float(auc), float(ess)


def timed(f, *a, **kw):
    t0 = time.time()
    out = f(*a, **kw)
    return out, time.time() - t0
