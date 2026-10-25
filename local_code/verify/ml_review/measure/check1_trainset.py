"""(1) 独立重建 G 的 LightGBM 训练集，与 multilearner_G.run_unit_ml 实际传给学习器的参数逐位比较。
不训练任何新学习器：fit_predict_learner 被替换为记录参数 + 返回均匀概率；G.evaluate 被替换为记录参数 + 返回 {}，
因此不计算任何测试精度。输出只写到本目录下的 scratch_out/。"""
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCR = os.path.join(HERE, 'scratch_out')
shutil.rmtree(SCR, ignore_errors=True)
os.makedirs(SCR)
os.environ['ML_OUT'] = SCR
L = sys.argv[1]
sys.path.insert(0, L)
os.chdir(L)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import guidance_G as G  # noqa: E402
import multilearner_G as ML  # noqa: E402
from stageA import class_weights_from  # noqa: E402

GOUT = G.OUT
D = G.GData()
UNITS = [('AT', 0, 0), ('RO', 1, 0), ('LV', 0, 1)]


def dig(a):
    a = np.ascontiguousarray(a)
    return (str(a.dtype), a.shape, hashlib.sha1(a.tobytes()).hexdigest())


# ---------- 独立重建（照抄 guidance_G.run_unit 的逻辑，不调用 ML 的任何函数）
def expected(tgt, split, rep):
    S = dict(np.load(os.path.join(GOUT, 'stage0', f'{tgt}_s{split}.npz')))
    n_cand = len(S['cand'])
    rng = np.random.default_rng(10_000 * split + 100 * rep + 7)
    pool_pos = np.sort(rng.choice(n_cand, min(30_000, n_cand), replace=False))
    assert np.array_equal(pool_pos, G.pool_positions(S, split, rep))
    idx_pool = S['allT'][pool_pos]
    fr, test = S['foreign'], S['test']
    assert np.array_equal(S['allT'][n_cand:], test)
    yF = D.y[fr]
    cw = class_weights_from(yF)
    yT = D.y[test]
    big = np.flatnonzero(np.bincount(yT, minlength=10) >= 30)
    pi_pool = np.bincount(D.y[idx_pool], minlength=10) / len(idx_pool)
    E = {}
    jobs = [('none', 'foreign_only', 0)] + [(c, m, k) for c in ('AE', 'S2') for m in G.E1 for k in (50, 200)] + \
           [('none', 'srs', k) for k in (50, 100, 200, 400)]
    for cond, method, k in jobs:
        if k == 0:
            sel = np.array([], dtype=np.int64)
            pi_sel = np.array([])
        else:
            z = np.load(os.path.join(GOUT, 'sel', f'{tgt}_s{split}_r{rep}_{cond}_{method}_k{k}.npz'))
            sel = idx_pool[z['sel_pos']]  # G 训练时用的就是 idx_pool[sel_pos]
            assert np.array_equal(sel, z['sel']), (cond, method, k)
            pi_sel = z['pi_sel']
        ys = D.y[sel]
        w = np.r_[cw[yF], 50.0 * cw[ys]]
        pi_train = np.bincount(np.r_[yF, ys], weights=w, minlength=10) / w.sum()
        pi_des = None
        if len(pi_sel):
            ipw = 1 / pi_sel
            pi_des = np.bincount(ys, weights=ipw, minlength=10) / ipw.sum()
        for clf in ('s2', 'ae'):
            X = D.X[clf]
            E[(cond, method, k, clf)] = dict(
                X_tr=dig(np.vstack([X[fr], X[sel]])), y_tr=dig(np.r_[yF, ys]), w_tr=dig(w), X_te=dig(X[test]),
                seed=1000 * split + (0 if k == 0 else rep), sel=sel, pi_train=pi_train, pi_des=pi_des,
                h=G.sel_hash(sel) if k else 'k0', n_tr=len(fr) + len(sel))
    return S, E, dict(yT=yT, big=big, efar=S['efar'].astype(bool), pi_pool=pi_pool, n_test=len(test))


# ---------- 截获
REC, EV = [], []


def spy(learner, X_tr, y_tr, w_tr, X_te, seed, threads, C=None):
    REC.append(dict(learner=learner, X_tr=dig(X_tr), y_tr=dig(y_tr), w_tr=dig(w_tr), X_te=dig(X_te), seed=seed,
                    threads=threads, C=C))
    return np.full((len(X_te), 10), 0.1, np.float32), {}


def ev(P_it, yT, big, efar, pi_train, pi_pool, pi_des):
    EV.append(dict(yT=dig(yT), big=big.copy(), efar=dig(efar), pi_train=pi_train.copy(), pi_pool=pi_pool.copy(),
                   pi_des=None if pi_des is None else pi_des.copy(), keys=list(P_it)))
    return {}


ML.fit_predict_learner = spy
G.evaluate = ev

fails = []
summary = []
for tgt, split, rep in UNITS:
    S, E, U = expected(tgt, split, rep)
    unit = f'{tgt}_s{split}_r{rep}'
    # G 的行：每个 (cond, method, k, clf) 的 sel_hash 与 G probs 文件的行数
    GR = pd.read_csv(os.path.join(GOUT, f'rows_t1_{unit}.csv'), usecols=['cond', 'method', 'k', 'clf', 'sel_hash'],
                     dtype={'sel_hash': str})
    for learner in ('lr', 'mlp', 'rf'):
        out = os.path.join(SCR, f'{unit}_{learner}')
        os.makedirs(os.path.join(out, 'lr_C'), exist_ok=True)
        dummyC = {'ae': 0.0123, 's2': 0.0456}
        for f, c in dummyC.items():  # 预置 C，避免在这里真的选 C
            json.dump(dict(C=c), open(os.path.join(out, 'lr_C', f'{tgt}_s{split}_{f}.json'), 'w'))
        REC.clear(), EV.clear()
        rows = ML.run_unit_ml(D, learner, tgt, split, rep, 1, out=out, log=lambda s: None)
        R = pd.DataFrame(rows)
        assert len(R) == 42, len(R)
        assert len(EV) == 42
        fit_rows = R[~R.cache_hit.astype(bool)]
        assert len(fit_rows) == len(REC), (len(fit_rows), len(REC))
        for (i, r), rec in zip(fit_rows.iterrows(), REC):
            e = E[(r.cond, r.method, r.k, r.clf)]
            for key in ('X_tr', 'y_tr', 'w_tr', 'X_te'):
                if rec[key] != e[key]:
                    fails.append((unit, learner, r.cond, r.method, r.k, r.clf, key, rec[key][:2], e[key][:2]))
            if rec['seed'] != e['seed']:
                fails.append((unit, learner, r.cond, r.method, r.k, r.clf, 'seed', rec['seed'], e['seed']))
            if learner == 'lr' and rec['C'] != dummyC[r.clf]:
                fails.append((unit, learner, r.cond, r.method, r.k, r.clf, 'C', rec['C'], dummyC[r.clf]))
            if learner != 'lr' and rec['C'] is not None:
                fails.append((unit, learner, 'C should be None', rec['C']))
        for (i, r), ev_ in zip(R.iterrows(), EV):
            e = E[(r.cond, r.method, r.k, r.clf)]
            if ev_['yT'] != dig(U['yT']) or not np.array_equal(ev_['big'], U['big']) or ev_['efar'] != dig(U['efar']):
                fails.append((unit, learner, r.cond, r.method, r.k, r.clf, 'yT/big/efar'))
            if not np.array_equal(ev_['pi_train'], e['pi_train']) or not np.array_equal(ev_['pi_pool'], U['pi_pool']):
                fails.append((unit, learner, r.cond, r.method, r.k, r.clf, 'pi_train/pi_pool'))
            if (ev_['pi_des'] is None) != (e['pi_des'] is None) or (
                    e['pi_des'] is not None and not np.array_equal(ev_['pi_des'], e['pi_des'])):
                fails.append((unit, learner, r.cond, r.method, r.k, r.clf, 'pi_des'))
            if r.sel_hash != e['h']:
                fails.append((unit, learner, r.cond, r.method, r.k, r.clf, 'sel_hash', r.sel_hash, e['h']))
            # 与 G 的行一致：同一 (cond, method, k, clf) 的 sel_hash
            g = GR[(GR.cond == r.cond) & (GR.method == r.method) & (GR.k == r.k) & (GR.clf == r.clf)]
            assert len(g) == 1, (r.cond, r.method, r.k, r.clf, len(g))
            gh = g.sel_hash.iloc[0]
            if r.k and gh != r.sel_hash:
                fails.append((unit, learner, r.cond, r.method, r.k, r.clf, 'G sel_hash', gh, r.sel_hash))
            # 测试集行数与 G 的 probs 一致
            if r.k:
                gp = np.load(os.path.join(GOUT, 'probs', f'{unit}_{r.clf}_k{r.k}_{gh}.npz'))
                shp = {kk: gp[kk].shape for kk in gp.files}
                if any(s != (U['n_test'], 10) for s in shp.values()):
                    fails.append((unit, r.cond, r.method, r.k, r.clf, 'G probs shape', shp))
            else:
                shp = S[f'P0test_{r.clf}_i300'].shape
                if shp != (U['n_test'], 10):
                    fails.append((unit, 'k0 G P0test shape', shp))
        summary.append(dict(unit=unit, learner=learner, rows=len(R), fits=len(REC), cache_hits=int(R.cache_hit.sum()),
                            n_foreign=len(S['foreign']), n_test=U['n_test'],
                            seeds=sorted({r['seed'] for r in REC}), Cs=sorted({str(r['C']) for r in REC}),
                            threads=sorted({r['threads'] for r in REC})))
        print(summary[-1], flush=True)

# 缓存命中的行：是否因为 AE 与 S2 选样相同（哈希相同）或 k=0 共用
print('fails:', len(fails))
for f in fails[:40]:
    print('  ', f)
json.dump(dict(fails=[list(map(str, f)) for f in fails], summary=summary), open(os.path.join(HERE, 'check1_result.json'), 'w'),
          ensure_ascii=False, indent=1, default=str)
