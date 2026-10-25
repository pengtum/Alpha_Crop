"""
先验/标签偏移视角的轻量诊断（只读 G 的阶段 0、选样与缓存概率；不训练任何模型）。

A. 冻结外国模型 + 用 k 个目标标签只拟合决策层（温度 + 类偏置，Alexandari BCTS 形式），
   两种目标：
     - 'bal'：设计加权 + 类内 Hájek 归一（每个观测到的类总权重相同）→ 面向 BA 的头
     - 'nat'：只用设计权重（自然先验）→ 面向 OA 的头
   标签来自：G 实际的 SRS 选样（与 G 重训行逐单元配对）、G 的 predstrat_srs 选样（重复 0）、
   神谕配额 oracle_sqrt / oracle_equal（只作上界）。另给"全池标签"上界（BIAS_ORACLE 的 G 版）。
B. 对 G 已缓存的重训概率（SRS 与 predstrat_srs）做"去偏斜"：把有效训练先验 π_train(r=50) 改回均匀，
   看目标标签带来的先验偏斜是否压低了 BA。
输出：calib_rows.csv（逐单元）、calib_report.txt
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import glob
import hashlib
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

R = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
CACHE = '/Users/perng_mini/AlphaCrops_local/data/cache_guidance_G'
OUT = os.path.dirname(os.path.abspath(__file__))
C = 10
MIN_N = 30
N_POOL = 30_000
R_MULT = 50.0
COUNTRIES = 'AT,BE,DE,EE,LT,LV,PT,RO,SE,SI,SK'.split(',')
LAMS = [0.02, 0.1, 0.5]
LAM0 = 0.1  # 预先固定的主设置（未在测试集上调）

y_all = np.load(os.path.join(CACHE, 'y.npy'))


def pool_positions(n_cand, split, rep):
    rng = np.random.default_rng(10_000 * split + 100 * rep + 7)
    return np.sort(rng.choice(n_cand, min(N_POOL, n_cand), replace=False))


def class_weights_from(y_ref):
    n = np.bincount(y_ref, minlength=C).astype(float)
    return np.where(n > 0, len(y_ref) / (np.count_nonzero(n) * np.maximum(n, 1)), 1.0)


def scores(y, ph, big):
    rec = np.array([(ph[y == c] == c).mean() if (y == c).any() else np.nan for c in range(C)])
    return 100 * float((ph == y).mean()), 100 * float(np.nanmean(rec[big])), rec


def fit_bcts(Z, y, w, lam):
    """logits = s·Z + b；最小化 Σ w·NLL / Σ w + lam/2·||b||²。Z 为 log 概率 (n, C)。"""
    Y = np.eye(C)[y]
    w = w / w.sum()

    def f(th):
        s, b = th[0], th[1:]
        l = s * Z + b
        l = l - l.max(1, keepdims=True)
        q = np.exp(l)
        q /= q.sum(1, keepdims=True)
        nll = -(w * np.log(np.clip((q * Y).sum(1), 1e-300, None))).sum() + 0.5 * lam * (b ** 2).sum()
        g = (q - Y) * w[:, None]
        return nll, np.r_[(g * Z).sum(), g.sum(0) + lam * b]
    r = minimize(f, np.r_[1.0, np.zeros(C)], jac=True, method='L-BFGS-B',
                 bounds=[(1e-3, 50)] + [(-30, 30)] * C)
    return r.x[0], r.x[1:]


def weights(y, pi, mode):
    ipw = 1.0 / pi
    if mode == 'nat':
        return ipw
    tot = np.bincount(y, weights=ipw, minlength=C)
    return ipw / tot[y]


def oracle_pick(y, k, rule, rng):
    cls, cnt = np.unique(y, return_counts=True)
    want = {'sqrt': np.sqrt(cnt), 'equal': np.ones(len(cls)), 'prop': cnt.astype(float)}[rule]
    alloc = np.minimum(np.floor(want / want.sum() * k).astype(int), cnt)
    while alloc.sum() < k:
        room = np.flatnonzero(alloc < cnt)
        alloc[room[np.argmax((want / want.sum() * k - alloc)[room])]] += 1
    pick = np.concatenate([rng.choice(np.flatnonzero(y == c), a, replace=False) for c, a in zip(cls, alloc) if a > 0])
    pi = np.zeros(len(y))
    for c, a in zip(cls, alloc):
        pi[y == c] = a / (y == c).sum()
    return np.sort(pick), pi


def reweight(P, new, old):
    w = np.divide(new, old, out=np.zeros_like(new), where=old > 0)
    q = P * w
    return q / q.sum(1, keepdims=True)


def sel_hash(sel):
    return hashlib.md5(np.sort(np.asarray(sel, dtype=np.int64)).tobytes()).hexdigest()[:16]


rows = []
for tgt in COUNTRIES:
    for split in (0, 1):
        S = np.load(os.path.join(R, 'stage0', f'{tgt}_s{split}.npz'))
        n_cand = len(S['cand'])
        allT, test, fr = S['allT'], S['test'], S['foreign']
        yT = y_all[test]
        big = np.flatnonzero(np.bincount(yT, minlength=C) >= MIN_N)
        cw = class_weights_from(y_all[fr])
        for rep in (0, 1):
            unit = f'{tgt}_s{split}_r{rep}'
            pool_pos = pool_positions(n_cand, split, rep)
            idx_pool = allT[pool_pos]
            yP = y_all[idx_pool]
            N = len(pool_pos)
            for clf in ('s2', 'ae'):
                PP = S[f'P_T_{clf}'][pool_pos].astype(np.float64)
                PT = S[f'P_T_{clf}'][n_cand:].astype(np.float64)
                ZP, ZT = np.log(np.clip(PP, 1e-12, None)), np.log(np.clip(PT, 1e-12, None))
                oa0, ba0, _ = scores(yT, PT.argmax(1), big)
                base = dict(target=tgt, split=split, rep=rep, unit=unit, clf=clf)
                rows.append(dict(base, design='none', k=0, head='k0', oa=oa0, ba=ba0))

                def calib_rows(design, k, sel_pos, pi_full, extra=None):
                    ys = yP[sel_pos]
                    pis = pi_full[sel_pos]
                    ipw = 1 / pis
                    pihat = np.bincount(ys, weights=ipw, minlength=C) / ipw.sum()
                    out = []
                    for lam in LAMS:
                        for mode in ('bal', 'nat'):
                            s, b = fit_bcts(ZP[sel_pos], ys, weights(ys, pis, mode), lam)
                            L = s * ZT + b
                            o, bb, _ = scores(yT, L.argmax(1), big)
                            out.append(dict(base, design=design, k=k, head=f'{mode}_l{lam}', oa=o, ba=bb,
                                            **(extra or {})))
                            if mode == 'bal' and lam == LAM0:
                                # BA 头 + 设计先验 → 另一个 OA 头（同一组偏置，加 log π̂）
                                L2 = L + np.log(np.clip(pihat, 1e-6, None))
                                o2, b2, _ = scores(yT, L2.argmax(1), big)
                                out.append(dict(base, design=design, k=k, head='bal+prior', oa=o2, ba=b2,
                                                **(extra or {})))
                    # 只做先验修正（P1 类：π_train 取均匀——外国模型按外国频率加权训练）
                    Q = reweight(PT, np.clip(pihat, 0, None), np.full(C, 1 / C))
                    o3, b3, _ = scores(yT, Q.argmax(1), big)
                    out.append(dict(base, design=design, k=k, head='prior_only', oa=o3, ba=b3, **(extra or {})))
                    return out

                for k in (50, 100, 200, 400):
                    z = np.load(os.path.join(R, 'sel', f'{unit}_none_srs_k{k}.npz'))
                    rows += calib_rows('srs_G', k, z['sel_pos'], np.full(N, k / N))
                    # B：G 重训模型去偏斜
                    sel = z['sel']
                    pp = os.path.join(R, 'probs', f'{unit}_{clf}_k{k}_{sel_hash(sel)}.npz')
                    if os.path.exists(pp):
                        Pr = np.load(pp)['i300'].astype(np.float64)
                        Pr /= Pr.sum(1, keepdims=True)
                        o, bb, _ = scores(yT, Pr.argmax(1), big)
                        rows.append(dict(base, design='srs_G', k=k, head='retrain', oa=o, ba=bb))
                        ys = y_all[sel]
                        w = np.r_[cw[y_all[fr]], R_MULT * cw[ys]]
                        pit = np.bincount(np.r_[y_all[fr], ys], weights=w, minlength=C) / w.sum()
                        o, bb, _ = scores(yT, reweight(Pr, np.full(C, 1 / C), pit).argmax(1), big)
                        rows.append(dict(base, design='srs_G', k=k, head='retrain_deskew', oa=o, ba=bb))
                        Pr2 = np.load(pp)['i200'].astype(np.float64)
                        o, bb, _ = scores(yT, Pr2.argmax(1), big)
                        rows.append(dict(base, design='srs_G', k=k, head='retrain_i200', oa=o, ba=bb))
                if rep == 0:
                    for g in ('AE', 'S2'):
                        for k in (50, 200):
                            z = np.load(os.path.join(R, 'sel', f'{unit}_{g}_predstrat_srs_k{k}.npz'))
                            pi_full = np.zeros(N)
                            pi_full[z['sel_pos']] = z['pi_sel']
                            rows += calib_rows(f'predstrat_{g}', k, z['sel_pos'], pi_full)
                            sel = z['sel']
                            pp = os.path.join(R, 'probs', f'{unit}_{clf}_k{k}_{sel_hash(sel)}.npz')
                            if os.path.exists(pp):
                                Pr = np.load(pp)['i300'].astype(np.float64)
                                Pr /= Pr.sum(1, keepdims=True)
                                o, bb, _ = scores(yT, Pr.argmax(1), big)
                                rows.append(dict(base, design=f'predstrat_{g}', k=k, head='retrain', oa=o, ba=bb))
                                ys = y_all[sel]
                                w = np.r_[cw[y_all[fr]], R_MULT * cw[ys]]
                                pit = np.bincount(np.r_[y_all[fr], ys], weights=w, minlength=C) / w.sum()
                                o, bb, _ = scores(yT, reweight(Pr, np.full(C, 1 / C), pit).argmax(1), big)
                                rows.append(dict(base, design=f'predstrat_{g}', k=k, head='retrain_deskew',
                                                 oa=o, ba=bb))
                    # 神谕配额（上界；3 个种子）
                    for rule in ('sqrt', 'equal'):
                        for k in (50, 200):
                            for sd in range(3):
                                pick, pi_full = oracle_pick(yP, k, rule, np.random.default_rng(777 + sd))
                                rows += calib_rows(f'oracle_{rule}', k, pick, np.where(pi_full > 0, pi_full, 1.0),
                                                   extra=dict(seed=sd))
                    # 全池标签（决策层上界，G 版 BIAS_ORACLE）
                    allpos = np.arange(N)
                    for mode in ('bal', 'nat'):
                        s, b = fit_bcts(ZP, yP, weights(yP, np.ones(N), mode), 1e-4)
                        o, bb, _ = scores(yT, (s * ZT + b).argmax(1), big)
                        rows.append(dict(base, design='allpool', k=N, head=f'{mode}_full', oa=o, ba=bb))
                    # 额外 SRS 抽样（与 G 选样独立；估计 SRS+校准本身的单元内方差）
                    for k in (50, 200):
                        for sd in range(5):
                            pick = np.sort(np.random.default_rng(5000 + sd).choice(N, k, replace=False))
                            rows += calib_rows('srs_extra', k, pick, np.full(N, k / N), extra=dict(seed=sd))
        print(tgt, split, flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'calib_rows.csv'), index=False)
print(df.shape)
