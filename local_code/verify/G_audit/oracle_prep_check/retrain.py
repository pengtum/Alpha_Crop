"""用"按方案 z 截断"的 ORACLE 选样（resel.py 生成，sel_fixed/）重新训练并评估 ba_ge30。
用法：python retrain.py check        # 用存档 ORACLE 选样复现一行 rows
      python retrain.py run CLF WORKERS THREADS"""
import glob, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import guidance_G as G
from stageA import class_weights_from, MIN_N
OUT = G.OUT
AUD = '/Users/perng_mini/AlphaCrops_local/verify/G_audit/oracle_prep_check'
_D = None
def D_():
    global _D
    if _D is None: _D = G.GData()
    return _D

def one(tgt, split, sel_pos, clf, k, threads):
    D = D_()
    S = dict(np.load(os.path.join(OUT, 'stage0', f'{tgt}_s{split}.npz')))
    pp = G.pool_positions(S, split, 0); idx_pool = S['allT'][pp]
    fr, test = S['foreign'], S['test']
    yT = D.y[test]; big = np.flatnonzero(np.bincount(yT, minlength=10) >= MIN_N)
    cw = class_weights_from(D.y[fr])
    sel = idx_pool[sel_pos]; ys = D.y[sel]
    X = D.X[clf]
    w_tr = np.r_[cw[D.y[fr]], G.R_MULT * cw[ys]]
    P_it = G.train_predict(np.vstack([X[fr], X[sel]]), np.r_[D.y[fr], ys], w_tr, X[test], threads, 1000 * split, 300)
    P = P_it[300].astype(np.float64); P /= P.sum(1, keepdims=True)
    oa, ba, f1, _ = G._scores(yT, P.argmax(1), big)
    return ba, oa, G.sel_hash(sel)

def task(a):
    tgt, split, m, k, clf, threads = a
    p = os.path.join(AUD, 'retrain_rows', f'{tgt}_s{split}_{m}_k{k}_{clf}.csv')
    if os.path.exists(p): return pd.read_csv(p).iloc[0].to_dict()
    sel_pos = np.load(f'{AUD}/sel_fixed/{tgt}_s{split}_r0_ORACLEfix_{m}_k{k}.npz')['sel_pos']
    t0 = time.time()
    ba, oa, h = one(tgt, split, sel_pos, clf, k, threads)
    r = dict(target=tgt, split=split, rep=0, method=m, k=k, clf=clf, ba_fix=ba, oa_fix=oa, sel_hash_fix=h, t=time.time() - t0)
    pd.DataFrame([r]).to_csv(p + '.tmp', index=False); os.replace(p + '.tmp', p)
    print(r, flush=True)
    return r

if __name__ == '__main__':
    mode = sys.argv[1]
    os.makedirs(os.path.join(AUD, 'retrain_rows'), exist_ok=True)
    if mode == 'check':
        tgt, split, m, k, clf = 'SI', 0, 'typiclust', 50, 's2'
        R = pd.read_csv(os.path.join(OUT, f'rows_t2_{tgt}_s{split}_r0.csv'))
        row = R[(R.cond == 'ORACLE') & (R.method == m) & (R.k == k) & (R.clf == clf)].iloc[0]
        sp = np.load(os.path.join(OUT, 'sel', f'{tgt}_s{split}_r0_ORACLE_{m}_k{k}.npz'))['sel_pos']
        t0 = time.time(); ba, oa, h = one(tgt, split, sp, clf, k, 4)
        print('repro ba', ba, 'rows', row.ba_ge30, 'oa', oa, row.oa, 'hash', h, row.sel_hash, f'{time.time()-t0:.0f}s')
    else:
        clf, W, T = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
        D = G.GData()
        J = [(t, s, m, k, clf, T) for t in D.countries for s in (0, 1) for m in ('typiclust', 'lpm2_nogeo') for k in (50, 200)]
        from multiprocessing import get_context
        with get_context('spawn').Pool(W) as pool:
            res = pool.map(task, J, chunksize=1)
        pd.DataFrame(res).to_csv(os.path.join(AUD, f'retrain_{clf}.csv'), index=False)
