import sys; sys.path.insert(0, '/Users/perng_mini/AlphaCrops_local/verify/model_diag/altmodels')
from altlib import *
import pandas as pd
D = G.GData()
tgt = sys.argv[1] if len(sys.argv) > 1 else 'SI'
S, fr, test, yT, big, cw = load_unit(D, tgt)
sel = load_sel(tgt, 'none', 'srs', 200)
rows = pd.read_csv(f'{GOUT}/rows_t1_{tgt}_s0_r0.csv')
for feat in ('ae', 's2'):
    X_tr, y_tr, w_tr, is_t = train_set(D, feat, fr, sel, cw)
    X_te = np.asarray(D.X[feat][test], dtype=np.float32)
    ref = rows[(rows.method == 'srs') & (rows.k == 200) & (rows.clf == feat)].iloc[0]
    print(feat, 'csv lgb', round(ref.ba_ge30, 3), round(ref.oa, 3), flush=True)
    for name, f in (('lgb', m_lgb), ('lr', m_lr), ('et', m_et), ('hgb', m_hgb), ('rf', m_rf)):
        ph, t = timed(f, X_tr, y_tr, w_tr, X_te)
        print(feat, name, [round(v, 3) for v in scores(yT, ph, big)], f'{t:.1f}s', flush=True)
    (idx, t) = timed(knn_votes, X_tr, y_tr, w_tr, X_te)
    for kk in (10, 30):
        print(feat, f'knn{kk}', [round(v, 3) for v in scores(yT, knn_predict(idx, y_tr, w_tr, kk), big)], f'{t:.1f}s', flush=True)
    (ph, it), t = timed(m_mlp, X_tr, y_tr, w_tr, X_te)
    print(feat, 'mlp', [round(v, 3) for v in scores(yT, ph, big)], f'{t:.1f}s iters {it}', flush=True)
