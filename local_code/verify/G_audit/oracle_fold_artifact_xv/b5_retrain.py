"""复训（独立实现，与 run_unit 相同设置：外国池 + 所选 k 个，R_MULT 权重，300 棵，种子 1000*split）。
用法：python b5_retrain.py check            # 存档 ORACLE 选样复训一行，逐位对照 rows 与 probs 缓存
      python b5_retrain.py run TAG SPEC...   # SPEC = tgt:split:var:method:k:clf，var ∈ V2/F1/PF（本目录 sel/）
输出只写本目录。"""
import glob, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import guidance_G as G
from stageA import class_weights_from, MIN_N

OUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
W = '/Users/perng_mini/AlphaCrops_local/verify/G_audit/oracle_fold_artifact_xv'
THR = int(os.environ.get('THR', 3))
D = G.GData()


def train_eval(tgt, split, sel_pos, clf):
    S = dict(np.load(os.path.join(OUT, 'stage0', f'{tgt}_s{split}.npz')))
    pp = G.pool_positions(S, split, 0)
    idx_pool = S['allT'][pp]
    fr, test = S['foreign'], S['test']
    yT = D.y[test]
    big = np.flatnonzero(np.bincount(yT, minlength=10) >= MIN_N)
    cw = class_weights_from(D.y[fr])
    sel = idx_pool[sel_pos]
    ys = D.y[sel]
    X = D.X[clf]
    w = np.r_[cw[D.y[fr]], G.R_MULT * cw[ys]]
    P_it = G.train_predict(np.vstack([X[fr], X[sel]]), np.r_[D.y[fr], ys], w, X[test], THR, 1000 * split, 300)
    P = P_it[300].astype(np.float64)
    P /= P.sum(1, keepdims=True)
    oa, ba, _, _ = G._scores(yT, P.argmax(1), big)
    return ba, oa, G.sel_hash(sel), P_it


if sys.argv[1] == 'check':
    tgt, split, m, k, clf = 'PT', 0, 'typiclust', 200, 's2'
    z = np.load(os.path.join(OUT, 'sel', f'{tgt}_s{split}_r0_ORACLE_{m}_k{k}.npz'))
    t0 = time.time()
    ba, oa, h, P_it = train_eval(tgt, split, z['sel_pos'], clf)
    R = pd.read_csv(os.path.join(OUT, f'rows_t2_{tgt}_s{split}_r0.csv'))
    r = R[(R.cond == 'ORACLE') & (R.method == m) & (R.k == k) & (R.clf == clf)].iloc[0]
    cache = np.load(os.path.join(OUT, 'probs', f'{tgt}_s{split}_r0_{clf}_k{k}_{h}.npz'))
    print(f'复训 ba {ba!r} oa {oa!r} hash {h}；rows ba {r.ba_ge30!r} oa {r.oa!r} hash {r.sel_hash}；'
          f'probs 与缓存逐位相同 {np.array_equal(P_it[300], cache["i300"])}；{time.time() - t0:.0f}s')
else:
    tag = sys.argv[2]
    path = f'{W}/b5_{tag}.csv'
    rows = pd.read_csv(path).to_dict('records') if os.path.exists(path) else []
    for spec in sys.argv[3:]:
        tgt, split, var, m, k, clf = spec.split(':')
        split, k = int(split), int(k)
        sp = np.load(f'{W}/sel/{tgt}_s{split}_{var}_{m}_k{k}.npz')['sel_pos']
        t0 = time.time()
        ba, oa, h, _ = train_eval(tgt, split, sp, clf)
        rows.append(dict(tgt=tgt, split=split, var=var, method=m, k=k, clf=clf, ba_ge30=ba, oa=oa, sel_hash=h))
        pd.DataFrame(rows).to_csv(path, index=False)
        print(spec, f'ba {ba:.3f}', f'{time.time() - t0:.0f}s', flush=True)
