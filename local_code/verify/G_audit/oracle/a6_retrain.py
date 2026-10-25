"""审计 6：用修正几何 V2 的 ORACLE 选样（a5 生成）重新训练并评估，复算 H_V2。
先做一次复现检查：用存档的 ORACLE 选样按同一设置训练，与 rows 的 ba_ge30 比较。
输出只写到 verify 目录。用法：python a6_retrain.py check | run"""
import glob, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import guidance_G as G
from stageA import class_weights_from, MIN_N

OUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
AUD = '/Users/perng_mini/AlphaCrops_local/verify/G_audit/oracle'
THREADS = int(os.environ.get('THR', 6))
D = G.GData()
R = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT, 'rows_t2_*.csv'))], ignore_index=True)


def one(tgt, split, sel_pos, clf, k, S):
    pp = G.pool_positions(S, split, 0)
    idx_pool = S['allT'][pp]
    fr, test = S['foreign'], S['test']
    yT = D.y[test]
    big = np.flatnonzero(np.bincount(yT, minlength=10) >= MIN_N)
    cw = class_weights_from(D.y[fr])
    sel = idx_pool[sel_pos]
    ys = D.y[sel]
    X = D.X[clf]
    w_tr = np.r_[cw[D.y[fr]], G.R_MULT * cw[ys]]
    P_it = G.train_predict(np.vstack([X[fr], X[sel]]), np.r_[D.y[fr], ys], w_tr, X[test], THREADS, 1000 * split, 300)
    P = P_it[300].astype(np.float64)
    P /= P.sum(1, keepdims=True)
    oa, ba, f1, _ = G._scores(yT, P.argmax(1), big)
    return oa, ba


mode = sys.argv[1]
if mode == 'check':
    tgt, split, m, k, clf = 'SI', 0, 'lpm2_nogeo', 50, 's2'
    S = dict(np.load(os.path.join(OUT, 'stage0', f'{tgt}_s{split}.npz')))
    z = np.load(os.path.join(OUT, 'sel', f'{tgt}_s{split}_r0_ORACLE_{m}_k{k}.npz'))
    t0 = time.time()
    oa, ba = one(tgt, split, z['sel_pos'], clf, k, S)
    rr = R[(R.target == tgt) & (R.split == split) & (R.cond == 'ORACLE') & (R.method == m) & (R.k == k) & (R.clf == clf)]
    print(f'复训 oa {oa:.4f} ba {ba:.4f} ；rows oa {rr.oa.iloc[0]:.4f} ba {rr.ba_ge30.iloc[0]:.4f} ；{time.time() - t0:.0f}s')
else:
    path = os.path.join(AUD, f"a6_retrain_V2_{os.environ.get('TAG', 'all')}.csv")
    rows = pd.read_csv(path).to_dict('records') if os.path.exists(path) else []
    done = {(r['tgt'], r['split'], r['method'], r['k'], r['clf']) for r in rows}
    countries = os.environ.get('CTRY', ','.join(G.COUNTRIES_G)).split(',')
    for tgt in countries:
        for split in (0, 1):
            S = dict(np.load(os.path.join(OUT, 'stage0', f'{tgt}_s{split}.npz')))
            for m in ('lpm2_nogeo', 'typiclust'):
                for k in (50, 200):
                    sp = np.load(os.path.join(AUD, 'sel_fix', f'{tgt}_s{split}_V2_{m}_k{k}.npz'))['sel_pos']
                    for clf in ('s2', 'ae'):
                        if (tgt, split, m, k, clf) in done:
                            continue
                        t0 = time.time()
                        oa, ba = one(tgt, split, sp, clf, k, S)
                        rows.append(dict(tgt=tgt, split=split, method=m, k=k, clf=clf, oa=oa, ba_ge30=ba))
                        pd.DataFrame(rows).to_csv(path + '.tmp', index=False)
                        os.replace(path + '.tmp', path)
                        print(tgt, split, m, k, clf, f'{ba:.3f}', f'{time.time() - t0:.0f}s', flush=True)
