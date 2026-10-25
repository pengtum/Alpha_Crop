"""单次计时（只读项目；不读测试标签，不算任何精度）。
用法：python bench_one.py <learner> <feat> <k> <tgt> <threads> [split=0] [rep=0]
训练集与 guidance_G.run_unit 相同：外国 5 万 + k 个目标样本（k=200 用 G 已存的 SRS 选样），
w = cw[y]（外国）与 50*cw[y]（目标），cw 只按外国频率。输出一行 JSON：准备、拟合、预测耗时，内存峰值，概率合法性。"""
import os
import sys

learner, feat, k, tgt, T = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
split = int(sys.argv[6]) if len(sys.argv) > 6 else 0
rep = int(sys.argv[7]) if len(sys.argv) > 7 else 0
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
          'NUMEXPR_NUM_THREADS'):
    os.environ[v] = str(T)
PROJ = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, PROJ)
os.chdir(PROJ)
import json, pickle, resource, time, warnings  # noqa: E401,E402
import numpy as np  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402
import guidance_G as G  # noqa: E402
from stageA import class_weights_from  # noqa: E402

warnings.filterwarnings('ignore')
GOUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R_MULT = 50.0


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20  # macOS：字节 → MB


t0 = time.time()
D = G.GData()
S = np.load(os.path.join(GOUT, 'stage0', f'{tgt}_s{split}.npz'))
fr, test = S['foreign'], S['test']
cw = class_weights_from(D.y[fr])
sel = (np.load(os.path.join(GOUT, 'sel', f'{tgt}_s{split}_r{rep}_none_srs_k{k}.npz'))['sel'] if k else
       np.array([], np.int64))
X = D.X[feat]
X_tr = np.vstack([X[fr], X[sel]]).astype(np.float32)
y_tr = np.r_[D.y[fr], D.y[sel]]  # 训练标签（外国 + 已选目标样本），不含测试
w_tr = np.r_[cw[D.y[fr]], R_MULT * cw[D.y[sel]]]
X_te = np.asarray(X[test], dtype=np.float32)
t_prep = time.time() - t0
rss0 = rss()
info = {}


def std_fit(A):
    mu, sd = A.mean(0), A.std(0) + 1e-6
    return mu, sd


def full_proba(m, B):
    P = np.zeros((len(B), 10))
    P[:, m.classes_] = m.predict_proba(B)
    return P


def softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


with threadpool_limits(limits=T):
    t1 = time.time()
    model = None
    if learner == 'lgb':
        import lightgbm as lgb
        model = lgb.train(G.lgb_params(T, 0) if hasattr(G, 'lgb_params') else None,
                          lgb.Dataset(X_tr, y_tr, weight=w_tr), num_boost_round=300)
        t_fit = time.time() - t1; t2 = time.time()
        P = model.predict(X_te)
    elif learner in ('lr', 'lr_c01'):
        from sklearn.linear_model import LogisticRegression
        mu, sd = std_fit(X_tr)
        A, B = (X_tr - mu) / sd, (X_te - mu) / sd
        model = LogisticRegression(C=1.0 if learner == 'lr' else 0.01, max_iter=1000, tol=1e-4)
        model.fit(A, y_tr, sample_weight=w_tr / w_tr.mean())
        info['n_iter'] = int(np.max(model.n_iter_))
        t_fit = time.time() - t1; t2 = time.time()
        P = full_proba(model, B)
    elif learner == 'mlp_es':
        from sklearn.neural_network import MLPClassifier
        mu, sd = std_fit(X_tr)
        A, B = (X_tr - mu) / sd, (X_te - mu) / sd
        model = MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
                              early_stopping=True, validation_fraction=0.1, n_iter_no_change=8, max_iter=60,
                              random_state=0).fit(A, y_tr, sample_weight=w_tr / w_tr.mean())
        info['n_iter'] = int(model.n_iter_)
        t_fit = time.time() - t1; t2 = time.time()
        P = full_proba(model, B)
    elif learner == 'mlp_f3':
        # 不早停、固定 20 轮、3 个种子，概率平均（verify_altmodels/mlp_fair.py 预先指定的主 MLP：mlpF_ens 第 20 轮）
        from sklearn.neural_network import MLPClassifier
        mu, sd = std_fit(X_tr)
        A, B = (X_tr - mu) / sd, (X_te - mu) / sd
        ms = [MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
                            early_stopping=False, max_iter=20, n_iter_no_change=10 ** 9, tol=0.0,
                            random_state=s).fit(A, y_tr, sample_weight=w_tr / w_tr.mean()) for s in (0, 1, 2)]
        info['n_iter'] = [int(m.n_iter_) for m in ms]
        model = ms
        t_fit = time.time() - t1; t2 = time.time()
        P = sum(full_proba(m, B) for m in ms) / 3
    elif learner in ('rf', 'rf_msl5'):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=500, max_features='sqrt',
                                       min_samples_leaf=1 if learner == 'rf' else 5, n_jobs=T, random_state=0)
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        t_fit = time.time() - t1; t2 = time.time()
        P = full_proba(model, X_te)
    elif learner in ('et', 'et_msl5'):
        from sklearn.ensemble import ExtraTreesClassifier
        model = ExtraTreesClassifier(n_estimators=500, max_features='sqrt',
                                     min_samples_leaf=1 if learner == 'et' else 5, n_jobs=T, random_state=0)
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        t_fit = time.time() - t1; t2 = time.time()
        P = full_proba(model, X_te)
    elif learner == 'hgb':
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=63,
                                               min_samples_leaf=20, max_bins=63, l2_regularization=0.0,
                                               early_stopping=False, random_state=0)
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        info['n_iter'] = int(model.n_iter_)
        t_fit = time.time() - t1; t2 = time.time()
        P = full_proba(model, X_te)
    elif learner == 'knn':
        # 标准化后欧氏距离，brute；近邻按样本权重（外国 cw、目标 50·cw）加权投票，k=10 与 k=30 同一次搜索
        from sklearn.neighbors import NearestNeighbors
        mu, sd = std_fit(X_tr)
        A, B = (X_tr - mu) / sd, (X_te - mu) / sd
        model = NearestNeighbors(n_neighbors=30, algorithm='brute', n_jobs=T).fit(A)
        t_fit = time.time() - t1; t2 = time.time()
        _, idx = model.kneighbors(B)
        V = np.zeros((len(B), 10))
        I = idx[:, :10]
        np.add.at(V, (np.repeat(np.arange(len(I)), 10), y_tr[I].ravel()), w_tr[I].ravel())
        P = V / V.sum(1, keepdims=True)
    elif learner == 'knn_cached':
        # 缓存版：外国部分的 30 近邻按（国家, 划分, 特征）只算一次（标准化只用外国统计量），
        # 每个选样只需算测试到 k 个目标样本的距离，再与外国近邻合并。这里分别计时两步。
        from sklearn.neighbors import NearestNeighbors
        nF = len(fr)
        mu, sd = std_fit(X_tr[:nF])
        AF, B = (X_tr[:nF] - mu) / sd, (X_te - mu) / sd
        dF, iF = NearestNeighbors(n_neighbors=30, algorithm='brute', n_jobs=T).fit(AF).kneighbors(B)
        info['t_foreign_once'] = round(time.time() - t1, 3)
        t1 = time.time()
        AT = (X_tr[nF:] - mu) / sd
        if len(AT):
            from sklearn.metrics import pairwise_distances
            dT = pairwise_distances(B, AT, n_jobs=1)
            dd = np.hstack([dF, dT]); ii = np.hstack([iF, nF + np.arange(len(AT))[None, :].repeat(len(B), 0)])
            o = np.argsort(dd, 1, kind='stable')[:, :10]
            I = np.take_along_axis(ii, o, 1)
        else:
            I = iF[:, :10]
        V = np.zeros((len(B), 10))
        np.add.at(V, (np.repeat(np.arange(len(I)), 10), y_tr[I].ravel()), w_tr[I].ravel())
        P = V / V.sum(1, keepdims=True)
        t_fit = time.time() - t1; t2 = time.time()
    elif learner == 'linsvc':
        from sklearn.svm import LinearSVC
        mu, sd = std_fit(X_tr)
        A, B = (X_tr - mu) / sd, (X_te - mu) / sd
        model = LinearSVC(C=1.0, dual=False, tol=1e-4, max_iter=1000).fit(A, y_tr, sample_weight=w_tr / w_tr.mean())
        info['n_iter'] = int(model.n_iter_)
        t_fit = time.time() - t1; t2 = time.time()
        Z = np.full((len(B), 10), -1e3); Z[:, model.classes_] = model.decision_function(B)
        P = softmax(Z)  # 未校准，只用于 argmax；先验调整类指标不可用
    elif learner == 'sgd_mh':
        from sklearn.linear_model import SGDClassifier
        mu, sd = std_fit(X_tr)
        A, B = (X_tr - mu) / sd, (X_te - mu) / sd
        model = SGDClassifier(loss='modified_huber', alpha=1e-4, max_iter=50, tol=1e-4, n_jobs=T,
                              random_state=0).fit(A, y_tr, sample_weight=w_tr / w_tr.mean())
        info['n_iter'] = int(model.n_iter_)
        t_fit = time.time() - t1; t2 = time.time()
        P = full_proba(model, B)
    elif learner == 'lr_csel':
        # 在外国池上按"留国家组"选 C：G.fold_groups 的 5 组 × C ∈ {1e-3, 1e-2, 1e-1, 1, 10}；只计时，不看任何得分
        from sklearn.linear_model import LogisticRegression
        groups = G.fold_groups(D, tgt, split, fr)
        nF = len(fr)
        A0, y0, w0 = X_tr[:nF], y_tr[:nF], w_tr[:nF]
        n_fit = 0
        for g in np.unique(groups):
            tr_, te_ = groups != g, groups == g
            mu, sd = std_fit(A0[tr_])
            A, B = (A0[tr_] - mu) / sd, (A0[te_] - mu) / sd
            m = None
            for C in (1e-3, 1e-2, 1e-1, 1.0, 10.0):  # 从小到大热启动
                m = LogisticRegression(C=C, max_iter=1000, tol=1e-4, warm_start=True) if m is None else m.set_params(C=C)
                m.fit(A, y0[tr_], sample_weight=w0[tr_] / w0[tr_].mean())
                _ = m.predict_proba(B)
                n_fit += 1
        info['n_inner_fits'] = n_fit
        t_fit = time.time() - t1; t2 = time.time()
        P = np.full((len(X_te), 10), 0.1)
    else:
        raise ValueError(learner)
    t_pred = time.time() - t2

P = np.asarray(P, dtype=np.float64)
info['rss_after_pred_mb'] = round(rss())  # 拟合 + 预测的峰值（不含下面序列化模型的临时缓冲）
ok =bool(P.shape == (len(test), 10) and np.isfinite(P).all() and np.allclose(P.sum(1), 1, atol=1e-4))
try:
    msize = len(pickle.dumps(model, protocol=4)) / 2 ** 20 if model is not None else 0.0
except Exception:
    msize = float('nan')
print(json.dumps(dict(learner=learner, feat=feat, k=k, tgt=tgt, split=split, rep=rep, threads=T,
                      n_train=len(y_tr), n_test=len(test), t_prep=round(t_prep, 2), t_fit=round(t_fit, 2),
                      t_pred=round(t_pred, 2), t_total=round(t_fit + t_pred, 2), rss_before_fit_mb=round(rss0),
                      rss_peak_mb=round(rss()), model_mb=round(msize, 1), proba_ok=ok, **info)), flush=True)
