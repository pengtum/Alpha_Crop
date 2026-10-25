"""重放 check3_supp.py 的 count_matched 抽样，核对 CM 训练/早停集。"""
import os
import sys

sys.dont_write_bytecode = True
CODE = ('/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/'
        'Research/Alpha_Crops/label_expansion')
sys.path.insert(0, CODE)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from common import COUNTRIES, load_dataset  # noqa: E402

D = '/Users/perng_mini/AlphaCrops_local/results/check3'
OUT = '/Users/perng_mini/AlphaCrops_local/verify/check3'
RS = 42


def strat_split(idx, y, test_size, seed):
    counts = np.bincount(y[idx], minlength=y.max() + 1)
    strat = np.where(counts[y[idx]] >= 2, y[idx], -1)
    if 0 < (strat == -1).sum() < 2:
        strat[strat == -1] = counts.argmax()
    return train_test_split(idx, test_size=test_size, random_state=seed, stratify=strat)


def count_matched(pref, extra, counts, y, rng):
    out, short = [], {}
    for c in np.flatnonzero(counts):
        a = pref[y[pref] == c]
        if len(a) >= counts[c]:
            out.append(rng.choice(a, counts[c], replace=False))
            continue
        b = extra[y[extra] == c]
        need = counts[c] - len(a)
        take = rng.choice(b, min(need, len(b)), replace=False)
        out += [a, take]
        short[int(c)] = dict(need=int(counts[c]), from_pref=int(len(a)), topup=int(len(take)),
                             missing=int(need - len(take)))
    return np.concatenate(out), short


df = load_dataset()
classes = sorted(df['label'].unique())
C = len(classes)
y = df['label'].map({c: i for i, c in enumerate(classes)}).to_numpy()
ctry = df['country'].to_numpy()
print('classes', classes, 'N', len(df))

rows = []
store = {}
for T in COUNTRIES:
    sp = np.load(os.path.join(D, f'{T}_split.npz'))
    cal = np.load(os.path.join(D, f'{T}_calib.npz'))
    foreign, pool, test = sp['foreign'], sp['pool'], sp['test']
    # 外国池重放
    rng0 = np.random.default_rng(RS)
    chk = np.concatenate([rng0.choice(np.flatnonzero(ctry == c), min(10_000, int((ctry == c).sum())), replace=False)
                          for c in COUNTRIES if c != T])
    assert np.array_equal(chk, foreign), T
    assert np.array_equal(y[pool], sp['y_pool']) and np.array_equal(y[test], sp['y_test'])
    f_tr, f_va = strat_split(foreign, y, 0.1, RS)
    p_tr, p_va = strat_split(pool, y, 0.1, RS)
    extra = np.setdiff1d(np.flatnonzero(ctry != T), foreign)
    rng = np.random.default_rng(RS + 1)
    calib = np.concatenate([rng.choice(extra[ctry[extra] == c], 1000, replace=False)
                            for c in COUNTRIES if c != T])
    ok_calib = np.array_equal(calib, cal['calib'])
    ok_fva = np.array_equal(f_va, cal['f_va'])
    ok_pva = np.array_equal(p_va, cal['p_va'])
    extra_cm = np.setdiff1d(extra, calib)
    n_ptr = np.bincount(y[p_tr], minlength=C)
    n_pva = np.bincount(y[p_va], minlength=C)
    cm_tr, s_tr = count_matched(f_tr, extra_cm, n_ptr, y, rng)
    cm_va, s_va = count_matched(f_va, np.setdiff1d(extra_cm, cm_tr), n_pva, y, rng)
    n_cmtr = np.bincount(y[cm_tr], minlength=C)
    n_cmva = np.bincount(y[cm_va], minlength=C)
    t_all = np.flatnonzero(ctry == T)
    r = dict(T=T, ok_calib=ok_calib, ok_fva=ok_fva, ok_pva=ok_pva,
             n_ptr=int(n_ptr.sum()), n_cmtr=len(cm_tr), n_pva=int(n_pva.sum()), n_cmva=len(cm_va),
             tr_count_equal=bool(np.array_equal(n_ptr, n_cmtr)),
             va_count_equal=bool(np.array_equal(n_pva, n_cmva)),
             tr_diff={classes[c]: int(n_ptr[c] - n_cmtr[c]) for c in range(C) if n_ptr[c] != n_cmtr[c]},
             va_diff={classes[c]: int(n_pva[c] - n_cmva[c]) for c in range(C) if n_pva[c] != n_cmva[c]},
             cm_extra_classes=[classes[c] for c in range(C) if n_cmtr[c] > 0 and n_ptr[c] == 0],
             tr_unique=len(np.unique(cm_tr)) == len(cm_tr), va_unique=len(np.unique(cm_va)) == len(cm_va),
             tr_va_overlap=len(np.intersect1d(cm_tr, cm_va)),
             any_T_in_cm=int((ctry[cm_tr] == T).sum() + (ctry[cm_va] == T).sum()),
             cm_in_Tall=len(np.intersect1d(np.concatenate([cm_tr, cm_va]), t_all)),
             cm_calib_overlap=len(np.intersect1d(np.concatenate([cm_tr, cm_va]), calib)),
             cm_va_in_ftr=len(np.intersect1d(cm_va, f_tr)),
             cm_tr_in_fva=len(np.intersect1d(cm_tr, f_va)),
             topup_tr=int(sum(v['topup'] for v in s_tr.values())),
             short_tr=s_tr, short_va=s_va,
             n_countries_tr=len(np.unique(ctry[cm_tr])))
    rows.append(r)
    store[T] = dict(cm_tr=cm_tr, cm_va=cm_va, p_tr=p_tr, p_va=p_va, f_tr=f_tr, f_va=f_va)
    print(T, {k: v for k, v in r.items() if k not in ('short_tr', 'short_va')})
    print('   short_tr', s_tr)
    print('   short_va', s_va)

np.savez(os.path.join(OUT, 'replay_cm_idx.npz'),
         **{f'{T}_{k}': v for T, d in store.items() for k, v in d.items()})
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'replay_cm.csv'), index=False)

# 与 supp_meta.csv 的 n_train / n_val 对照
meta = pd.read_csv(os.path.join(D, 'supp_meta.csv'))
bad = []
for r in rows:
    for mdl in ('cmb', 'cmn'):
        m = meta[(meta.target == r['T']) & (meta.model == mdl)]
        for _, mm in m.iterrows():
            if mm.n_train != r['n_cmtr'] or mm.n_val != r['n_cmva']:
                bad.append((r['T'], mm.feat, mdl, mm.n_train, r['n_cmtr'], mm.n_val, r['n_cmva']))
print('supp_meta n_train/n_val mismatches:', bad)
print(meta.groupby('model').size())
