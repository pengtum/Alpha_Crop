"""只用标签重放 check3_decomposition.main() 的划分逻辑（不训练、不读 results/check3）。"""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
COUNTRIES = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']
RS = 42
df = pd.read_parquet('/Users/perng_mini/AlphaCrops_local/data/dataset_v2.parquet',
                     columns=['country','label','has_ae','has_s2','longitude','latitude','area_ha'])
df = df[df.has_ae & df.has_s2].reset_index(drop=True)
classes = sorted(df['label'].unique()); C = len(classes)
y = df['label'].map({c: i for i, c in enumerate(classes)}).to_numpy()
ctry = df['country'].to_numpy()

def strat_split(idx, y, test_size, seed):
    counts = np.bincount(y[idx], minlength=y.max() + 1)
    strat = np.where(counts[y[idx]] >= 2, y[idx], -1)
    return train_test_split(idx, test_size=test_size, random_state=seed, stratify=strat)

short = [c[:5] for c in classes]
print('classes', list(enumerate(classes)))
for tgt in COUNTRIES:
    rng = np.random.default_rng(RS)
    foreign = np.concatenate([rng.choice(np.flatnonzero(ctry == c), min(10_000, int((ctry == c).sum())), replace=False)
                              for c in COUNTRIES if c != tgt])
    t_idx = np.flatnonzero(ctry == tgt)
    n_raw = len(t_idx)
    t_idx = t_idx[~df.iloc[t_idx].duplicated(['longitude', 'latitude', 'area_ha']).to_numpy()]
    pool, test = train_test_split(t_idx, test_size=0.3, random_state=RS)
    msg = []
    try:
        f_tr, f_va = strat_split(foreign, y, 0.1, RS)
    except Exception as e:
        msg.append(f'foreign strat_split ERROR: {e}'); f_tr = f_va = None
    try:
        p_tr, p_va = strat_split(pool, y, 0.1, RS)
    except Exception as e:
        msg.append(f'pool strat_split ERROR: {type(e).__name__}: {e}'); p_tr = p_va = None
    bc = lambda a: np.bincount(y[a], minlength=C)
    print(f'\n== {tgt}: raw {n_raw}, dedup {len(t_idx)} (-{n_raw-len(t_idx)}), pool {len(pool)}, test {len(test)}, foreign {len(foreign)}')
    print('   pool ', bc(pool).tolist()); print('   test ', bc(test).tolist())
    if p_tr is not None:
        print('   p_tr ', bc(p_tr).tolist()); print('   p_va ', bc(p_va).tolist())
        unseen = set(np.unique(y[p_va])) - set(np.unique(y[p_tr]))
        if unseen: msg.append(f'p_va has classes unseen in p_tr: {unseen} -> LightGBM LabelEncoder error')
    if f_tr is not None:
        print('   f_tr ', bc(f_tr).tolist())
        unseen = set(np.unique(y[f_va])) - set(np.unique(y[f_tr]))
        if unseen: msg.append(f'f_va unseen in f_tr: {unseen}')
    pi_T = bc(pool) / len(pool); pi_F = bc(foreign) / len(foreign)
    ratio = np.divide(pi_T, pi_F, out=np.zeros(C), where=pi_F > 0)
    w = ratio[y[f_tr]]; w = w[w > 0]
    print('   ratio', np.round(ratio, 3).tolist(), f' max/min(pos) = {w.max()/w.min():.0f}, sum(w)/n_keep={w.mean():.3f}, n_keep={len(w)}')
    cls_T_only = [classes[c] for c in range(C) if pi_T[c] > 0 and pi_F[c] == 0]
    cls_test_not_pool = [classes[c] for c in range(C) if bc(test)[c] > 0 and bc(pool)[c] == 0]
    ge30 = [short[c] for c in range(C) if bc(test)[c] >= 30]
    print(f'   T-only classes {cls_T_only}; test∖pool {cls_test_not_pool}; ge30 classes ({len(ge30)}): {ge30}')
    # Kish effective sample size of loco_pm and loco weights per class
    for m in msg: print('   !!!', m)
