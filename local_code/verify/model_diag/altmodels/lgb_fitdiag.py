"""LightGBM（G 配置，4 线程）拟合诊断：k=200 SRS，训练集外国部分的 ba（拟合程度）、目标测试 ba（与 CSV 比对复现）、耗时。
同时对照 LR、MLP2 的训练 ba 已在 part1_rows.csv。"""
import sys; sys.path.insert(0, '/Users/perng_mini/AlphaCrops_local/verify/model_diag/altmodels')
from altlib import *
import pandas as pd, lightgbm as lgb
D = G.GData()
out = []
ref = pd.read_csv(os.path.join(OUT, 'lgb_ref_s0r0.csv'))
for tgt in sys.argv[1].split(','):
    S, fr, test, yT, big, cw = load_unit(D, tgt)
    sel = load_sel(tgt, 'none', 'srs', 200)
    for feat in ('ae', 's2'):
        X_tr, y_tr, w_tr, is_t = train_set(D, feat, fr, sel, cw)
        X_te = np.asarray(D.X[feat][test], np.float32)
        t0 = time.time()
        m = lgb.train(G.lgb_params(THREADS, 0), lgb.Dataset(X_tr, y_tr, weight=w_tr), num_boost_round=300)
        t = time.time() - t0
        big_tr = np.flatnonzero(np.bincount(y_tr[~is_t], minlength=10) >= 30)
        row = dict(target=tgt, feat=feat, t_fit_4thr=t)
        for it in (50, 100, 200, 300):
            row[f'train_ba_foreign_i{it}'] = scores(y_tr[~is_t], m.predict(X_tr[~is_t], num_iteration=it).argmax(1), big_tr)[1]
            row[f'train_acc_target_i{it}'] = 100 * float((m.predict(X_tr[is_t], num_iteration=it).argmax(1) == y_tr[is_t]).mean())
            row[f'test_ba_i{it}'] = scores(yT, m.predict(X_te, num_iteration=it).argmax(1), big)[1]
        r = ref[(ref.target == tgt) & (ref.clf == feat) & (ref.k == 200)].iloc[0]
        row['csv_ba'] = r.ba_ge30
        out.append(row)
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        pd.DataFrame(out).to_csv(os.path.join(OUT, f'lgb_fitdiag_{sys.argv[1].replace(",", "_")}.csv'), index=False)
