import os, sys, pickle, itertools
CODE = "/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion"
sys.path.insert(0, CODE)
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
import common
from common import COUNTRIES, RANDOM_STATE
from pilot_target_selection import N_FOREIGN_PER_COUNTRY, TARGET_WEIGHT_SHARE
import pilot_betti as pb
V = '/Users/perng_mini/AlphaCrops_local/verify/taskA/'
L = '/Users/perng_mini/AlphaCrops_local/'
# ---- 主试点：逐字复现 pilot_target_selection.py 第 66-85 行（load_dataset 只读必要列）
df = pd.read_parquet(common.DATASET_V2, columns=['parcel_id', 'country', 'label', 'has_ae', 'has_s2'])
for r in ('ae', 's2'): df = df[df[f'has_{r}']]
df = df.reset_index(drop=True)
ctry = df['country'].to_numpy()
main = {}
for tgt in ['PT', 'SE', 'SK', 'LV']:
    rng = np.random.default_rng(RANDOM_STATE)
    foreign = np.concatenate([rng.choice(np.flatnonzero(ctry == c), N_FOREIGN_PER_COUNTRY, replace=False) for c in COUNTRIES if c != tgt])
    t_idx = np.flatnonzero(ctry == tgt)
    pool, test = train_test_split(t_idx, test_size=0.3, random_state=RANDOM_STATE)
    main[tgt] = (foreign, pool, test)
# merge 保序检查
nuts = pd.read_parquet(os.path.join(common.DATA_DIR, 'parcel_nuts.parquet'), columns=['parcel_id', 'nuts3'])
m = df.merge(nuts, on='parcel_id', how='left', validate='one_to_one')
print('merge preserves order:', (m.parcel_id.values == df.parcel_id.values).all(), 'nuts3 NA:', m.nuts3.isna().sum())
# 与后台真实主试点的 pool 大小对照
ps = pd.read_csv(L + 'results/pilot_selection.csv')
print('main pilot target_full budgets (=len(pool)):', ps[ps.strategy == 'target_full'].groupby('target').budget.first().to_dict())
print('replicated len(pool):', {t: len(v[1]) for t, v in main.items()}, 'len(test):', {t: len(v[2]) for t, v in main.items()})
for t in ['PT', 'SE']:
    for cf in ['ae', 's2']:
        f = L + f'results/pilot_probs_{t}_{cf}.npy'
        if os.path.exists(f): print(f, np.load(f, mmap_mode='r').shape)

def load(mode): return pickle.load(open(V + f'capture_{mode}.pkl', 'rb'))
cap = load('parcel'); sh = load('parcel_shuffled')
# ---- pb 内部的 train_test_split 调用
for (t_idx, pool, test, k), tgt in zip(cap['splits'], cap['targets']):
    F, P, T = main[tgt]
    print(tgt, 'split identical:', np.array_equal(pool, P) and np.array_equal(test, T), k)
# ---- 按循环顺序还原每次 fit 的 (tgt,b,seed,strat)
order = [(t, b, s, st) for t in cap['targets'] for b in cap['budgets'] for s in cap['seeds'] for st in cap['strategies']]
assert len(order) == len(cap['calls']) == len(sh['calls'])
bad = 0; sel_by = {}
for (tgt, b, s, st), c, c2, te in zip(order, cap['calls'], sh['calls'], cap['tests']):
    F, P, T = main[tgt]
    tr = c['tr']; nf = len(F)
    fo, sel = tr[:nf], tr[nf:]
    ok = dict(
        foreign_same=np.array_equal(fo, F),
        test_same=np.array_equal(te, T),
        n_eq_budget=len(sel) == b,
        unique=len(np.unique(sel)) == len(sel),
        in_pool=np.isin(sel, P).all(),
        not_in_test=not np.isin(sel, T).any(),
        w_foreign_1=np.all(c['w'][:nf] == 1),
        w_target=np.allclose(c['w'][nf:], TARGET_WEIGHT_SHARE * nf / len(sel)),
        tgt_w_share=np.isclose(c['w'][nf:].sum() / c['w'][:nf].sum(), 0.25),
        seed_passed=c['seed'] == s,
        classblind=np.array_equal(tr, c2['tr']),
    )
    if not all(ok.values()):
        bad += 1; print('FAIL', tgt, b, s, st, {k: v for k, v in ok.items() if not v})
    sel_by[(tgt, b, s, st)] = sel
print('parcel-mode calls checked:', len(order), 'failures:', bad)
# ---- admin_rep：选中地块在各 NUTS3 的数量是否等于 allocate_parcels 的名额
nuts3 = m['nuts3'].to_numpy()
for tgt in cap['targets']:
    P = main[tgt][1]
    g, lab = pd.factorize(nuts3[P]); cap_g = np.bincount(g)
    for b in cap['budgets']:
        n_g = pb.allocate_parcels(cap_g / len(g), cap_g, b)
        for s in cap['seeds']:
            sel = sel_by[(tgt, b, s, 'admin_rep')]
            got = pd.Series(nuts3[sel]).value_counts().reindex(lab, fill_value=0).to_numpy()
            if not np.array_equal(got, n_g): print('ALLOC MISMATCH', tgt, b, s)
    print(tgt, 'admin_rep groups', len(cap_g), 'b=50 alloc', dict(zip(lab, pb.allocate_parcels(cap_g / len(g), cap_g, 50))))
# seeds 之间的重叠（同一 b）
for tgt in ['PT']:
    for st in cap['strategies']:
        a, b_ = sel_by[(tgt, 1000, 0, st)], sel_by[(tgt, 1000, 1, st)]
        print(tgt, st, 'overlap seed0/seed1 at b=1000:', len(np.intersect1d(a, b_)))
