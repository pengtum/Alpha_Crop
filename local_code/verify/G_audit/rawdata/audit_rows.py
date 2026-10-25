"""原始数据完整性审计（只读）：独立重算 rows 的指标、核对 sel/probs/stage0。
输出写到 /Users/perng_mini/AlphaCrops_local/verify/G_audit/rawdata/ 下。"""
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

RES = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
CACHE = '/Users/perng_mini/AlphaCrops_local/data/cache_guidance_G'
OUTD = '/Users/perng_mini/AlphaCrops_local/verify/G_audit/rawdata'
C = 10
MIN_N = 30
R_MULT = 50.0
N_POOL = 30_000

# ---------------- 数据（直接读 .npy，不经过项目代码）
y_all = np.load(os.path.join(CACHE, 'y.npy'))
ctry_all = np.load(os.path.join(CACHE, 'ctry.npy'))

# ---------------- 读全部 rows
files = sorted(glob.glob(os.path.join(RES, 'rows_t*_*.csv')))
dfs = []
for f in files:
    d = pd.read_csv(f)
    d['tier'] = int(os.path.basename(f).split('_')[1][1:])
    d['file'] = os.path.basename(f)
    dfs.append(d)
R = pd.concat(dfs, ignore_index=True)
print('rows files', len(files), 'rows', len(R))
key = ['cond', 'method', 'k', 'clf', 'target', 'split', 'rep']
print('duplicate keys', R.duplicated(key).sum())
print('NaN ba_ge30', R['ba_ge30'].isna().sum(), 'NaN oa', R['oa'].isna().sum(), 'NaN ba_P2', R['ba_P2'].isna().sum(),
      'NaN ba_far', R['ba_far'].isna().sum())
print(R.groupby(['tier', 'cond', 'method', 'k']).size().to_string())


# ---------------- 独立实现
def pool_pos_of(n_cand, split, rep):
    rng = np.random.default_rng(10_000 * split + 100 * rep + 7)
    return np.sort(rng.choice(n_cand, min(N_POOL, n_cand), replace=False))


def cw_of(yF):
    n = np.bincount(yF, minlength=C).astype(float)
    present = (n > 0).sum()
    return np.where(n > 0, len(yF) / (present * np.maximum(n, 1)), 1.0)


def pi_train_of(yF, ys, r):
    cw = cw_of(yF)
    num = np.zeros(C)
    for c in range(C):
        num[c] = cw[c] * ((yF == c).sum() + r * (ys == c).sum())
    return num / num.sum()


def recall_vec(y, ph):
    out = np.full(C, np.nan)
    for c in range(C):
        m = y == c
        if m.any():
            out[c] = np.mean(ph[m] == c)
    return out


def metr(y, ph, big, mask=None):
    if mask is not None:
        y, ph = y[mask], ph[mask]
    rec = recall_vec(y, ph)
    return 100 * np.mean(ph == y), 100 * np.nanmean(rec[big]), \
        100 * f1_score(y, ph, labels=big, average='macro', zero_division=0), rec


def saerens(P, pi_new, pi_tr):
    w = np.zeros(C)
    ok = pi_tr > 0
    w[ok] = pi_new[ok] / pi_tr[ok]
    Q = P * w
    s = Q.sum(1, keepdims=True)
    good = s[:, 0] > 0
    Q2 = P.copy()
    Q2[good] = Q[good] / s[good]
    return Q2


def em(P, pi_tr, it=200, tol=1e-6):
    k = pi_tr > 0
    Pk, pt = P[:, k], pi_tr[k]
    pi = pt.copy()
    for _ in range(it):
        num = Pk * (pi / pt)
        new = (num / np.maximum(num.sum(1, keepdims=True), 1e-300)).mean(0)
        done = np.max(np.abs(new - pi)) < tol
        pi = new
        if done:
            break
    out = np.zeros(C)
    out[k] = pi
    return out


def md5h(sel):
    return hashlib.md5(np.sort(np.asarray(sel, dtype=np.int64)).tobytes()).hexdigest()[:16]


# ---------------- 逐单元重算
recs = []
selrecs = []
probpairs = []
S_cache = {}
t0 = time.time()
for (tgt, split, rep), grp in R.groupby(['target', 'split', 'rep']):
    unit = f'{tgt}_s{split}_r{rep}'
    S = np.load(os.path.join(RES, 'stage0', f'{tgt}_s{split}.npz'))
    cand, test, allT, foreign = S['cand'], S['test'], S['allT'], S['foreign']
    efar = S['efar'].astype(bool)
    n_cand = len(cand)
    pp = pool_pos_of(n_cand, split, rep)
    idx_pool = allT[pp]
    yT = y_all[test]
    big = np.flatnonzero(np.bincount(yT, minlength=C) >= MIN_N)
    yF = y_all[foreign]
    pi_pool = np.bincount(y_all[idx_pool], minlength=C) / len(idx_pool)
    cand_set, test_set = set(cand.tolist()), set(test.tolist())
    pool_set = set(idx_pool.tolist())
    sel_by = {}
    for _, r in grp.iterrows():
        cond, method, k, clf = r['cond'], r['method'], int(r['k']), r['clf']
        ys = np.array([], int)
        pi_des = None
        if method == 'foreign_only':
            pre = f'P0test_{clf}_i'
            P_it = {int(kk[len(pre):]): S[kk].astype(np.float32) for kk in S.files if kk.startswith(pre)}
            pi_tr = pi_train_of(yF, ys, 0.0)
            prob_ok = None
        else:
            sp = os.path.join(RES, 'sel', f'{unit}_{cond}_{method}_k{k}.npz')
            z = np.load(sp)
            sel_pos, sel, pi_sel = z['sel_pos'], z['sel'], z['pi_sel']
            h = md5h(sel)
            if (cond, method, k) not in sel_by:
                sel_by[(cond, method, k)] = sel
                selrecs.append(dict(
                    unit=unit, target=tgt, split=split, rep=rep, cond=cond, method=method, k=k,
                    n=len(sel), n_unique=len(np.unique(sel)), pos_unique=len(np.unique(sel_pos)),
                    pos_in_range=bool((sel_pos >= 0).all() and (sel_pos < len(pp)).all()),
                    sel_eq_pool=bool(np.array_equal(idx_pool[sel_pos], sel)),
                    in_pool=all(s in pool_set for s in sel.tolist()),
                    in_cand=all(s in cand_set for s in sel.tolist()),
                    in_test=sum(s in test_set for s in sel.tolist()),
                    in_foreign=int(np.isin(sel, foreign).sum()),
                    ctry_ok=bool((ctry_all[sel] == tgt).all()),
                    hash_file=h, hash_row=r['sel_hash'], n_pi=len(pi_sel),
                    pi_ok=bool(len(pi_sel) == 0 or ((pi_sel > 0) & (pi_sel <= 1 + 1e-9)).all())))
            ys = y_all[sel]
            if len(pi_sel):
                ipw = 1 / pi_sel
                pi_des = np.bincount(ys, weights=ipw, minlength=C) / ipw.sum()
            pf = os.path.join(RES, 'probs', f'{unit}_{clf}_k{k}_{r["sel_hash"]}.npz')
            prob_ok = os.path.exists(pf)
            zz = np.load(pf)
            P_it = {int(kk[1:]): zz[kk].astype(np.float32) for kk in zz.files}
            pi_tr = pi_train_of(yF, ys, R_MULT)
        last = max(P_it)
        P = P_it[last].astype(np.float64)
        P /= P.sum(1, keepdims=True)
        ph = P.argmax(1)
        oa, ba, f1, rec = metr(yT, ph, big)
        out = dict(unit=unit, target=tgt, split=split, rep=rep, cond=cond, method=method, k=k, clf=clf,
                   prob_file=prob_ok, oa=oa, ba_ge30=ba, f1_ge30=f1)
        for c in range(C):
            out[f'rec{c}'] = rec[c]
        for it, Pi in P_it.items():
            o, b, f, _ = metr(yT, Pi.argmax(1), big)
            out[f'ba_i{it}'], out[f'oa_i{it}'] = b, o
        src = 'design'
        if pi_des is None:
            pi_des, src = em(P, pi_tr), 'em'
        for tag, pin in (('P1', pi_des), ('P2', pi_pool)):
            o, b, _, _ = metr(yT, saerens(P, pin, pi_tr).argmax(1), big)
            out[f'oa_{tag}'], out[f'ba_{tag}'] = o, b
        o, b, _, _ = metr(yT, ph, big, efar)
        out.update(oa_far=o, ba_far=b, n_far=int(efar.sum()), prior_src=src,
                   n_classes_sel=len(np.unique(ys)) if len(ys) else 0,
                   comp_L1=float(np.abs(np.bincount(ys, minlength=C) / k - pi_pool).sum()) if k > 0 else np.nan,
                   n_big=len(big), n_big_far_lt30=int(sum((yT[efar] == c).sum() < MIN_N for c in big)),
                   n_big_far0=int(sum((yT[efar] == c).sum() == 0 for c in big)))
        recs.append(out)
    # AE vs S2 Jaccard（同一 (单元, 方法, k)）
    for (cond, method, k), sel in sel_by.items():
        for other in ('S2', 'XY', 'S2XY', 'ORACLE'):
            if cond == 'AE' and (other, method, k) in sel_by:
                s2 = sel_by[(other, method, k)]
                a, b = set(sel.tolist()), set(s2.tolist())
                probpairs.append(dict(unit=unit, method=method, k=k, pair=f'AE-{other}',
                                      jacc=len(a & b) / len(a | b), inter=len(a & b)))
        if cond == 'S2' and ('S2XY', method, k) in sel_by:
            a, b = set(sel.tolist()), set(sel_by[('S2XY', method, k)].tolist())
            probpairs.append(dict(unit=unit, method=method, k=k, pair='S2-S2XY', jacc=len(a & b) / len(a | b),
                                  inter=len(a & b)))
    print(unit, len(grp), f'{time.time() - t0:.0f}s', flush=True)

M = pd.DataFrame(recs)
M.to_csv(os.path.join(OUTD, 'recomputed_rows.csv'), index=False)
pd.DataFrame(selrecs).to_csv(os.path.join(OUTD, 'sel_checks.csv'), index=False)
pd.DataFrame(probpairs).to_csv(os.path.join(OUTD, 'jaccard_pairs.csv'), index=False)

# ---------------- 比对
J = R.merge(M, on=['target', 'split', 'rep', 'cond', 'method', 'k', 'clf'], suffixes=('', '_re'), how='outer',
            indicator=True)
print('merge', J['_merge'].value_counts().to_dict())
cols = ['oa', 'ba_ge30', 'f1_ge30', 'ba_P1', 'oa_P1', 'ba_P2', 'oa_P2', 'ba_far', 'oa_far', 'ba_i200', 'ba_i250',
        'ba_i300', 'oa_i200', 'oa_i250', 'n_far', 'n_classes_sel', 'comp_L1'] + [f'rec{c}' for c in range(C)]
for c in cols:
    a, b = J[c].to_numpy(float), J[c + '_re'].to_numpy(float)
    both_nan = np.isnan(a) & np.isnan(b)
    d = np.where(both_nan, 0, np.abs(a - b))
    bad = ~(d < 1e-8)
    print(f'{c:14s} max|diff| {np.nanmax(d):.3g}  n_bad {bad.sum()}')
    if bad.any():
        print(J.loc[bad, ['file', 'cond', 'method', 'k', 'clf', c, c + '_re']].head(10).to_string())
print('prior_src mismatch', (J['prior_src'] != J['prior_src_re']).sum())
print('elapsed', time.time() - t0)
