"""
D1 独立复核（不 import guidance_G / analyze_G / calib_diag；不训练分类器；线程 2）。
- 按 rev4 D1 的文字自行实现：候选池位置、D1 种子、类均衡权重、BCTS 头（s·log p + b，λ/2‖b‖²，边界同规格）、
  ba_ge30（测试集 ≥30 个地块的类）与 OA。
- Δhead：44 单元 × 2 特征 × 5 个 k × 5 重复 + 全池，逐行与 decision_curve_rows.csv 对比（指标、s、b）。
- Δretrain：直接由 G 的 sel/*.npz → sel_hash → probs/*.npz（i300）与 stage0 的 P0test_*_i300 重算，
  与 G rows 以及 decision_curve_units.csv 的 dret 对比。
- 自行聚合（国家值 → t 95% CI），与 summary 的主要数字对比。
- 敏感性：同一目标、更严格的 L-BFGS-B 收敛阈值，看优化器容差对 BA 的影响。
只读 results/guidance_G；不读 guidance_G_ml。
"""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[v] = '2'
import glob
import hashlib
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

R = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
assert 'guidance_G_ml' not in R
Y = np.load('/Users/perng_mini/AlphaCrops_local/data/cache_guidance_G/y.npy')
D1 = '/Users/perng_mini/AlphaCrops_local/verify/storyline/decision_curve'
OUT = os.path.dirname(os.path.abspath(__file__))
C = 10
KS = [50, 100, 200, 400, 1000]
KR = [50, 100, 200, 400]
CTRY = 'AT,BE,DE,EE,LT,LV,PT,RO,SE,SI,SK'.split(',')
TIGHT_UNITS = {'AT_s0_r0', 'SE_s1_r1', 'PT_s0_r1', 'LV_s1_r0', 'RO_s1_r0'}


def pool_pos(n_cand, split, rep):
    g = np.random.default_rng(10_000 * split + 100 * rep + 7)
    return np.sort(g.choice(n_cand, min(30_000, n_cand), replace=False))


def seed(t, s, rp, k, r):
    return int(hashlib.md5(f'D1_{t}_{s}_{rp}_{k}_{r}'.encode()).hexdigest()[:8], 16)


def fit(Z, yy, lam, tight=False):
    cls, inv, cnt = np.unique(yy, return_inverse=True, return_counts=True)
    w = 1.0 / (len(cls) * cnt[inv])  # 每个出现的类总权重 1/m，类内均匀；和为 1
    n = len(yy)
    ar = np.arange(n)

    def obj(th):
        s, b = th[0], th[1:]
        L = s * Z + b
        L = L - L.max(1, keepdims=True)
        lse = np.log(np.exp(L).sum(1))
        f = -(w * (L[ar, yy] - lse)).sum() + 0.5 * lam * (b @ b)
        P = np.exp(L - lse[:, None])
        P[ar, yy] -= 1.0
        P *= w[:, None]
        return f, np.r_[(P * Z).sum(), P.sum(0) + lam * b]
    opt = dict(ftol=1e-15, gtol=1e-11, maxiter=20000, maxfun=40000) if tight else {}
    r = minimize(obj, np.r_[1.0, np.zeros(C)], jac=True, method='L-BFGS-B',
                 bounds=[(1e-3, 50)] + [(-30, 30)] * C, options=opt)
    return r.x[0], r.x[1:], r.fun


def score(yT, ph, big):
    oa = 100 * np.mean(ph == yT)
    ba = 100 * np.mean([np.mean(ph[yT == c] == c) for c in big])
    return oa, ba


def sel_hash(sel):
    return hashlib.md5(np.sort(np.asarray(sel, dtype=np.int64)).tobytes()).hexdigest()[:16]


rows, ret, tight = [], [], []
for t in CTRY:
    for sp in (0, 1):
        S = np.load(os.path.join(R, 'stage0', f'{t}_s{sp}.npz'))
        cand, test, allT = S['cand'], S['test'], S['allT']
        nc = len(cand)
        assert np.array_equal(allT[:nc], cand) and np.array_equal(allT[nc:], test)
        assert len(np.intersect1d(cand, test)) == 0
        yT = Y[test]
        big = np.flatnonzero(np.bincount(yT, minlength=C) >= 30)
        for rp in (0, 1):
            unit = f'{t}_s{sp}_r{rp}'
            pp = pool_pos(nc, sp, rp)
            assert pp.max() < nc
            yP = Y[allT[pp]]
            N = len(pp)
            for f in ('ae', 's2'):
                ZP = np.log(np.clip(S[f'P_T_{f}'][pp].astype(np.float64), 1e-12, None))
                ZT = np.log(np.clip(S[f'P_T_{f}'][nc:].astype(np.float64), 1e-12, None))
                o0, b0 = score(yT, ZT.argmax(1), big)
                rows.append(dict(unit=unit, target=t, split=sp, rep=rp, clf=f, design='base_PT', k=0, r=-1,
                                 oa=o0, ba=b0))
                for k in KS:
                    for r in range(5):
                        sd = seed(t, sp, rp, k, r)
                        pick = np.sort(np.random.default_rng(sd).choice(N, k, replace=False))
                        s_, b_, fv = fit(ZP[pick], yP[pick], 0.1)
                        o, b = score(yT, (s_ * ZT + b_).argmax(1), big)
                        rows.append(dict(unit=unit, target=t, split=sp, rep=rp, clf=f, design='srs_D1', k=k, r=r,
                                         seed=sd, oa=o, ba=b, s=s_, fval=fv, **{f'b{c}': b_[c] for c in range(C)}))
                        if unit in TIGHT_UNITS and r < 2:
                            s2_, b2_, fv2 = fit(ZP[pick], yP[pick], 0.1, tight=True)
                            o2, bb2 = score(yT, (s2_ * ZT + b2_).argmax(1), big)
                            tight.append(dict(unit=unit, clf=f, k=k, r=r, oa=o, ba=b, oa_t=o2, ba_t=bb2,
                                              fval=fv, fval_t=fv2, ds=abs(s2_ - s_), db=np.abs(b2_ - b_).max()))
                s_, b_, fv = fit(ZP, yP, 0.1)
                o, b = score(yT, (s_ * ZT + b_).argmax(1), big)
                rows.append(dict(unit=unit, target=t, split=sp, rep=rp, clf=f, design='fullpool', k=N, r=-1,
                                 oa=o, ba=b, s=s_, fval=fv, **{f'b{c}': b_[c] for c in range(C)}))
                if unit in TIGHT_UNITS:
                    s2_, b2_, fv2 = fit(ZP, yP, 0.1, tight=True)
                    o2, bb2 = score(yT, (s2_ * ZT + b2_).argmax(1), big)
                    tight.append(dict(unit=unit, clf=f, k=N, r=-1, oa=o, ba=b, oa_t=o2, ba_t=bb2,
                                      fval=fv, fval_t=fv2, ds=abs(s2_ - s_), db=np.abs(b2_ - b_).max()))
                # Δretrain：直接由缓存概率重算
                P0 = S[f'P0test_{f}_i300'].astype(np.float64)
                P0 /= P0.sum(1, keepdims=True)
                of, bf = score(yT, P0.argmax(1), big)
                for k in KR:
                    z = np.load(os.path.join(R, 'sel', f'{unit}_none_srs_k{k}.npz'))
                    assert np.array_equal(z['sel'], allT[pp][z['sel_pos']])
                    h = sel_hash(z['sel'])
                    Pr = np.load(os.path.join(R, 'probs', f'{unit}_{f}_k{k}_{h}.npz'))['i300'].astype(np.float64)
                    Pr /= Pr.sum(1, keepdims=True)
                    o, b = score(yT, Pr.argmax(1), big)
                    ret.append(dict(unit=unit, target=t, split=sp, rep=rp, clf=f, k=k, oa_ret=o, ba_ret=b,
                                    oa_fo=of, ba_fo=bf, sel_hash=h))
        print(t, sp, flush=True)

M = pd.DataFrame(rows)
RT = pd.DataFrame(ret)
TT = pd.DataFrame(tight)
M.to_csv(os.path.join(OUT, 'recheck_rows.csv'), index=False)
RT.to_csv(os.path.join(OUT, 'recheck_retrain.csv'), index=False)
TT.to_csv(os.path.join(OUT, 'recheck_tight.csv'), index=False)
print('done', M.shape, RT.shape, TT.shape)
