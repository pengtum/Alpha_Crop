"""第 1 部分：11 国 × 划分 0 × 重复 0；k ∈ {0, 200}（k=200 为 G 的 SRS 选样）；特征 ae、s2。
LightGBM 参照值取自 G 的 rows CSV（SI 两种特征已按同一代码重训逐位复现）。结果逐行追加到 part1_rows.csv，可断点续跑。"""
import sys; sys.path.insert(0, '/Users/perng_mini/AlphaCrops_local/verify/model_diag/altmodels')
from altlib import *
from altlib import _std
import pandas as pd
import warnings; warnings.filterwarnings('ignore')

D = G.GData()
path = os.path.join(OUT, 'part1_rows.csv')
rows = pd.read_csv(path).to_dict('records') if os.path.exists(path) else []
done = {(r['target'], r['feat'], r['k'], r['model']) for r in rows}
countries = sys.argv[1].split(',') if len(sys.argv) > 1 and sys.argv[1] != 'all' else D.countries
# 时间预算（AT 之后）：跳过的 (模型, k)；环境变量 SKIP='hgb:*,iw_lgb:0,rf:0,et:0'
SKIP = {(a, None if b == '*' else int(b)) for a, b in (x.split(':') for x in os.environ.get('SKIP', '').split(',') if x)}


def add(tgt, feat, k, model, ph, t, yT, big, **extra):
    oa, ba = scores(yT, ph, big)
    rows.append(dict(target=tgt, feat=feat, k=k, model=model, ba_ge30=ba, oa=oa, t_fit=round(t, 2), **extra))
    pd.DataFrame(rows).to_csv(path + '.tmp', index=False)
    os.replace(path + '.tmp', path)
    print(f'{tgt} {feat} k{k} {model:10s} ba {ba:6.2f} oa {oa:6.2f} {t:6.1f}s {extra if extra else ""}', flush=True)


def train_ba(model_pred, y, big_tr):
    return scores(y, model_pred, big_tr)[1]


for tgt in countries:
    S, fr, test, yT, big, cw = load_unit(D, tgt)
    pool_idx = S['allT'][G.pool_positions(S, 0, 0)]
    sel200 = load_sel(tgt, 'none', 'srs', 200)
    for feat in ('ae', 's2'):
        X_te = np.asarray(D.X[feat][test], dtype=np.float32)
        iw = None
        for k in (0, 200):
            sel = sel200 if k else np.array([], np.int64)
            X_tr, y_tr, w_tr, is_t = train_set(D, feat, fr, sel, cw)
            big_tr = np.flatnonzero(np.bincount(y_tr[~is_t], minlength=10) >= 30)

            def need(m):
                if (m, k) in SKIP or (m, None) in SKIP:
                    return False
                return (tgt, feat, k, m) not in done
            if need('lr'):
                from sklearn.linear_model import LogisticRegression
                t0 = time.time()
                A, B_ = _std(X_tr, X_te)
                m = LogisticRegression(C=1.0, max_iter=1000).fit(A, y_tr, sample_weight=w_tr / w_tr.mean())
                ph = m.predict(B_); t = time.time() - t0
                add(tgt, feat, k, 'lr', ph, t, yT, big, train_ba_foreign=train_ba(m.predict(A[~is_t]), y_tr[~is_t], big_tr),
                    n_iter=int(np.max(m.n_iter_)))
            for name, hid in (('mlp2', (256, 128)), ('mlp3', (512, 256, 128))):
                if need(name):
                    from sklearn.neural_network import MLPClassifier
                    t0 = time.time()
                    A, B_ = _std(X_tr, X_te)
                    m = MLPClassifier(hidden_layer_sizes=hid, alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
                                      early_stopping=True, validation_fraction=0.1, n_iter_no_change=8, max_iter=60,
                                      random_state=0).fit(A, y_tr, sample_weight=w_tr / w_tr.mean())
                    ph = m.predict(B_); t = time.time() - t0
                    add(tgt, feat, k, name, ph, t, yT, big, train_ba_foreign=train_ba(m.predict(A[~is_t]), y_tr[~is_t], big_tr),
                        n_iter=int(m.n_iter_))
            if need('hgb'):
                from sklearn.ensemble import HistGradientBoostingClassifier
                t0 = time.time()
                m = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=150, max_leaf_nodes=63, min_samples_leaf=20,
                                                   max_bins=63, early_stopping=False, random_state=0).fit(X_tr, y_tr, sample_weight=w_tr)
                ph = m.predict(X_te); t = time.time() - t0
                add(tgt, feat, k, 'hgb', ph, t, yT, big, train_ba_foreign=train_ba(m.predict(X_tr[~is_t]), y_tr[~is_t], big_tr))
            if need('rf'):
                ph, t = timed(m_rf, X_tr, y_tr, w_tr, X_te, n=150)
                add(tgt, feat, k, 'rf', ph, t, yT, big)
            if need('et'):
                ph, t = timed(m_et, X_tr, y_tr, w_tr, X_te, n=300)
                add(tgt, feat, k, 'et', ph, t, yT, big)
            if need('knn10') or need('knn30'):
                idx, t = timed(knn_votes, X_tr, y_tr, w_tr, X_te, kmax=30)
                for kk in (10, 30):
                    add(tgt, feat, k, f'knn{kk}', knn_predict(idx, y_tr, w_tr, kk), t, yT, big)
            # 目标感知 1：外国样本按域判别器重要性加权（判别器：外国 5 万 vs 目标候选池 3 万，无标签，交叉拟合）
            if need('iw_lgb') or need('iw_lr'):
                if iw is None:
                    (w_dom, auc, ess), t_dom = timed(domain_weights, np.asarray(D.X[feat][fr], np.float32),
                                                     np.asarray(D.X[feat][pool_idx], np.float32))
                    iw = (w_dom, auc, ess, t_dom)
                w_dom, auc, ess, t_dom = iw
                w_iw = w_tr.copy(); w_iw[~is_t] *= w_dom
                if need('iw_lgb'):
                    ph, t = timed(m_lgb, X_tr, y_tr, w_iw, X_te, seed=0)
                    add(tgt, feat, k, 'iw_lgb', ph, t + t_dom, yT, big, dom_auc=auc, iw_ess=ess, wmax=float(w_dom.max()))
                if need('iw_lr'):
                    ph, t = timed(m_lr, X_tr, y_tr, w_iw, X_te)
                    add(tgt, feat, k, 'iw_lr', ph, t + t_dom, yT, big, dom_auc=auc, iw_ess=ess)
            # 目标感知 2：只用 k=200 个目标样本（无外国），类别权重仍按外国频率
            if k == 200:
                Xs, ys, ws = X_tr[is_t], y_tr[is_t], cw[y_tr[is_t]]
                n_cls = len(np.unique(ys))
                if need('tonly_lgb'):
                    ph, t = timed(m_lgb, Xs, ys, ws, X_te, seed=0)
                    add(tgt, feat, k, 'tonly_lgb', ph, t, yT, big, n_cls_sel=n_cls)
                if need('tonly_lr'):
                    ph, t = timed(m_lr, Xs, ys, ws, X_te)
                    add(tgt, feat, k, 'tonly_lr', ph, t, yT, big, n_cls_sel=n_cls)
                if need('tonly_rf'):
                    ph, t = timed(m_rf, Xs, ys, ws, X_te, n=300)
                    add(tgt, feat, k, 'tonly_rf', ph, t, yT, big, n_cls_sel=n_cls)
                if need('tonly_knn10'):
                    idx, t = timed(knn_votes, Xs, ys, ws, X_te, kmax=10)
                    add(tgt, feat, k, 'tonly_knn10', knn_predict(idx, ys, ws, 10), t, yT, big, n_cls_sel=n_cls)
print('PART1 DONE', flush=True)
