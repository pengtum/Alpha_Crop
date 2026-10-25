"""少量重训：验证 probs 文件确实由所述特征 + 所选样本训练得到（逐位或近似），以及 P0test 与外国全量模型一致。
在 label_expansion 目录下运行（只 import stageA.lgb_params 取参数；不写项目文件）。"""
import os
import sys
import time

import numpy as np
import lightgbm as lgb

sys.path.insert(0, os.getcwd())
from stageA import lgb_params  # noqa: E402

RES = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
CACHE = '/Users/perng_mini/AlphaCrops_local/data/cache_guidance_G'
y = np.load(os.path.join(CACHE, 'y.npy'))
X = {f: np.load(os.path.join(CACHE, f'X_{f}.npy'), mmap_mode='r') for f in ('ae', 's2')}


def cw_of(yF):
    n = np.bincount(yF, minlength=10).astype(float)
    return np.where(n > 0, len(yF) / ((n > 0).sum() * np.maximum(n, 1)), 1.0)


def train(Xtr, ytr, w, Xte, seed, threads=2):
    m = lgb.train(lgb_params(threads, seed), lgb.Dataset(Xtr, ytr, weight=w), num_boost_round=300)
    return {it: m.predict(Xte, num_iteration=it).astype(np.float32) for it in (200, 250, 300)}


import hashlib  # noqa: E402


def md5h(sel):
    return hashlib.md5(np.sort(np.asarray(sel, dtype=np.int64)).tobytes()).hexdigest()[:16]


cases = [('PT', 0, 0, 'AE', 'kcenter', 50), ('SE', 1, 1, 'S2', 'lpm2_nogeo', 200), ('RO', 0, 0, 'ORACLE', 'margin', 200)]
for tgt, s, r, cond, method, k in cases:
    S = np.load(os.path.join(RES, 'stage0', f'{tgt}_s{s}.npz'))
    fr, test = S['foreign'], S['test']
    sel = np.load(os.path.join(RES, 'sel', f'{tgt}_s{s}_r{r}_{cond}_{method}_k{k}.npz'))['sel']
    h = md5h(sel)
    yF, ys = y[fr], y[sel]
    cw = cw_of(yF)
    w = np.r_[cw[yF], 50.0 * cw[ys]]
    for clf in ('ae', 's2'):
        t0 = time.time()
        P = train(np.vstack([X[clf][fr], X[clf][sel]]), np.r_[yF, ys], w, X[clf][test], 1000 * s + r)
        z = np.load(os.path.join(RES, 'probs', f'{tgt}_s{s}_r{r}_{clf}_k{k}_{h}.npz'))
        other = 's2' if clf == 'ae' else 'ae'
        zo = np.load(os.path.join(RES, 'probs', f'{tgt}_s{s}_r{r}_{other}_k{k}_{h}.npz'))
        for it in (200, 300):
            d = np.abs(P[it] - z[f'i{it}'])
            print(f'{tgt} s{s} r{r} {cond} {method} k{k} clf={clf} i{it}: exact={np.array_equal(P[it], z[f"i{it}"])} '
                  f'maxdiff={d.max():.2e} argmax_agree={np.mean(P[it].argmax(1) == z[f"i{it}"].argmax(1)):.5f} | '
                  f'vs other-clf file maxdiff={np.abs(P[it] - zo[f"i{it}"]).max():.3f} ({time.time() - t0:.0f}s)',
                  flush=True)

# 外国全量模型 → P0test（float16 存储）
for tgt, s, clf in (('SI', 1, 'ae'), ('LV', 0, 's2')):
    S = np.load(os.path.join(RES, 'stage0', f'{tgt}_s{s}.npz'))
    fr, test = S['foreign'], S['test']
    yF = y[fr]
    cw = cw_of(yF)
    t0 = time.time()
    P = train(X[clf][fr], yF, cw[yF], X[clf][test], 0)
    for it in (200, 300):
        q = S[f'P0test_{clf}_i{it}'].astype(np.float32)
        d = np.abs(P[it] - q)
        print(f'P0test {tgt} s{s} {clf} i{it}: maxdiff={d.max():.2e} (float16 量化) argmax_agree='
              f'{np.mean(P[it].argmax(1) == q.argmax(1)):.6f} n_diff={int((P[it].argmax(1) != q.argmax(1)).sum())} '
              f'({time.time() - t0:.0f}s)', flush=True)
    # P_T_{clf} 的温度校准：apply_temp(full, temp) 的候选 + 测试
    allT = S['allT']
    m = lgb.train(lgb_params(2, 0), lgb.Dataset(X[clf][fr], yF, weight=cw[yF]), num_boost_round=300)
    Pf = m.predict(X[clf][allT])
    tt = float(S[f'temp_{clf}'])
    z = tt * np.log(np.clip(Pf, 1e-12, None))
    z -= z.max(1, keepdims=True)
    Q = np.exp(z)
    Q /= Q.sum(1, keepdims=True)
    print(f'P_T_{clf} {tgt} s{s}: maxdiff vs stored {np.abs(Q - S[f"P_T_{clf}"]).max():.2e}, temp {tt:.3f}', flush=True)
