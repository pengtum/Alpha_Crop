"""
F3b 的稳健性（只读；无训练）：
 (a) 重训模型换成第 200 轮（G 缓存里已有）：F3b 的增益是否与"少训练"重叠？
 (b) bal 头的设计权重开/关（predstrat 标签）；λ ∈ {0.02, 0.1, 0.5}
 (c) 逐国表（k=200，α=0.5）
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
import pandas as pd
from scipy import stats

__file__ = os.path.abspath(__file__)
exec(open(os.path.join(os.path.dirname(__file__), 'calib_diag.py')).read().split('rows = []')[0])


def lsm(L):
    L = L - L.max(1, keepdims=True)
    return L - np.log(np.exp(L).sum(1, keepdims=True))


rows = []
for tgt in COUNTRIES:
    for split in (0, 1):
        S = np.load(os.path.join(R, 'stage0', f'{tgt}_s{split}.npz'))
        n_cand = len(S['cand'])
        allT, test = S['allT'], S['test']
        yT = y_all[test]
        big = np.flatnonzero(np.bincount(yT, minlength=C) >= MIN_N)
        for rep in (0, 1):
            unit = f'{tgt}_s{split}_r{rep}'
            pool_pos = pool_positions(n_cand, split, rep)
            yP = y_all[allT[pool_pos]]
            N = len(pool_pos)
            designs = [('srs_G', k, f'{unit}_none_srs_k{k}.npz') for k in (50, 200)]
            if rep == 0:
                designs += [('predstrat_AE', k, f'{unit}_AE_predstrat_srs_k{k}.npz') for k in (50, 200)]
            for clf in ('s2', 'ae'):
                PP = S[f'P_T_{clf}'][pool_pos].astype(np.float64)
                PT = S[f'P_T_{clf}'][n_cand:].astype(np.float64)
                ZP, ZT = np.log(np.clip(PP, 1e-12, None)), np.log(np.clip(PT, 1e-12, None))
                base = dict(target=tgt, split=split, rep=rep, unit=unit, clf=clf)
                for des, k, fn in designs:
                    z = np.load(os.path.join(R, 'sel', fn))
                    sel_pos, sel = z['sel_pos'], z['sel']
                    pis = z['pi_sel'] if len(z['pi_sel']) else np.full(k, k / N)
                    ys = yP[sel_pos]
                    Z = np.load(os.path.join(R, 'probs', f'{unit}_{clf}_k{k}_{sel_hash(sel)}.npz'))
                    for it in ('i300', 'i200'):
                        Pr = Z[it].astype(np.float64)
                        Lr = np.log(np.clip(Pr / Pr.sum(1, keepdims=True), 1e-12, None))
                        o, bb, _ = scores(yT, Lr.argmax(1), big)
                        rows.append(dict(base, design=des, k=k, it=it, var='retrain', alpha=0.0, oa=o, ba=bb))
                        for lam in (0.02, 0.1, 0.5):
                            for dw in ((True, False) if des != 'srs_G' else (True,)):
                                if it == 'i200' and (lam != 0.1 or not dw):
                                    continue
                                w = weights(ys, pis if dw else np.ones(k), 'bal')
                                s_b, b_b = fit_bcts(ZP[sel_pos], ys, w, lam)
                                Le = lsm(s_b * ZT + b_b)
                                for a in (0.3, 0.4, 0.5):
                                    o, bb, _ = scores(yT, ((1 - a) * Lr + a * Le).argmax(1), big)
                                    rows.append(dict(base, design=des, k=k, it=it, var=f'F3b_l{lam}_dw{int(dw)}',
                                                     alpha=a, oa=o, ba=bb))
        print(tgt, split, flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'frontier3_rows.csv'), index=False)
key = ['unit', 'target', 'clf', 'k']


def summ(j):
    c = j.groupby('target')[['dba', 'doa']].mean()
    out = []
    for col in ('dba', 'doa'):
        v = c[col]
        h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        pos = int((v > 0).sum()) if col == 'dba' else int((v >= -0.3).sum())
        out.append(f'{col} {v.mean():+.2f} [{v.mean() - h:+.2f},{v.mean() + h:+.2f}] {pos}/11')
    return '  '.join(out)


def get(des, it, var, a, k):
    return df[(df.design == des) & (df.it == it) & (df['var'] == var) & (df.alpha == a) & (df.k == k)] \
        .set_index(key)[['ba', 'oa']]


def diff(x, y):
    return (x - y).dropna().rename(columns={'ba': 'dba', 'oa': 'doa'}).reset_index()


L = []
for des in ('srs_G', 'predstrat_AE'):
    for k in (50, 200):
        L.append(f'\n== {des} k={k} ==')
        r300 = get(des, 'i300', 'retrain', 0.0, k)
        r200 = get(des, 'i200', 'retrain', 0.0, k)
        L.append('  重训 i200 − 重训 i300               ' + summ(diff(r200, r300)))
        for a in (0.3, 0.4, 0.5):
            x = get(des, 'i200', 'F3b_l0.1_dw1', a, k)
            L.append(f'  F3b(i200) α={a} − 重训 i200        ' + summ(diff(x, r200)))
            L.append(f'  F3b(i200) α={a} − 重训 i300        ' + summ(diff(x, r300)))
        for lam in (0.02, 0.1, 0.5):
            for dw in ((1, 0) if des != 'srs_G' else (1,)):
                x = get(des, 'i300', f'F3b_l{lam}_dw{dw}', 0.5, k)
                L.append(f'  F3b λ={lam} 设计权重={dw} α=0.5 − 重训 i300  ' + summ(diff(x, r300)))

L.append('\n== 逐国（k=200，α=0.5，λ=0.1，i300）：ΔBA / ΔOA ==')
for des, ref_des in (('srs_G', 'srs_G'), ('predstrat_AE', 'predstrat_AE'), ('predstrat_AE', 'srs_G')):
    x = get(des, 'i300', 'F3b_l0.1_dw1', 0.5, 200)
    ref = get(ref_des, 'i300', 'retrain', 0.0, 200)
    j = diff(x, ref)
    if ref_des != des:
        j = j[j.unit.str.endswith('r0')]
    t = j.groupby('target')[['dba', 'doa']].mean().round(2).T
    L.append(f'F3b({des}) − 重训 {ref_des}\n' + t.to_string())
txt = '\n'.join(L)
open(os.path.join(OUT, 'frontier3_report.txt'), 'w').write(txt)
print(txt)
