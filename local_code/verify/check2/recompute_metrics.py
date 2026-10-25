"""独立复算：从 preds/*.npy 与 dataset_v2 真实标签重算 ba / ba_ge30 / macro_f1，与 CSV 逐行对比。
不导入 label_expansion 下任何模块（common.load_dataset 的逻辑在此手写复现）。"""
import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

L = '/Users/perng_mini/AlphaCrops_local'
R = f'{L}/results'
P = f'{R}/preds'
OUTD = f'{L}/verify/check2'

df = pd.read_parquet(f'{L}/data/dataset_v2.parquet', columns=['parcel_id', 'country', 'longitude', 'latitude', 'area_ha', 'label', 'has_ae', 'has_s2'])
df = df[df.has_ae & df.has_s2].reset_index(drop=True)
classes = sorted(df.label.unique())
cmap = {c: i for i, c in enumerate(classes)}
y = df.label.map(cmap).to_numpy()
print('n rows', len(df), 'n classes', len(classes))
print(classes)


def my_metrics(yt, yp):
    present, cnt = np.unique(yt, return_counts=True)
    rec = np.array([np.mean(yp[yt == c] == c) for c in present])
    f1s = []
    for c in present:
        tp = np.sum((yp == c) & (yt == c)); fp = np.sum((yp == c) & (yt != c)); fn = np.sum((yp != c) & (yt == c))
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1s.append(0.0 if prec + r == 0 else 2 * prec * r / (prec + r))
    return dict(ba=rec.mean(), ba_ge30=rec[cnt >= 30].mean(), macro_f1=float(np.mean(f1s)))


# 独立复现划分，核对 test_idx
ctry = df.country.to_numpy()
COUNTRIES = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']
split_info = {}
for tgt in ['PT', 'SE', 'SK', 'LV']:
    t_idx = np.flatnonzero(ctry == tgt)
    dup = df.iloc[t_idx].duplicated(['longitude', 'latitude', 'area_ha']).to_numpy()
    t2 = t_idx[~dup]
    pool, test = train_test_split(t2, test_size=0.3, random_state=42)
    saved = np.load(f'{P}/{tgt}_test_idx.npy')
    same = np.array_equal(np.sort(saved), np.sort(test)) and np.array_equal(saved, test)
    yt = y[saved]
    pres, cnt = np.unique(yt, return_counts=True)
    split_info[tgt] = dict(n_all=len(t_idx), n_dup=int(dup.sum()), pool=len(pool), test=len(test), test_match=same,
                           n_cls_test=len(pres), n_cls_ge30=int((cnt >= 30).sum()),
                           small_classes={classes[c]: int(n) for c, n in zip(pres, cnt) if n < 30},
                           pool_test_overlap=len(np.intersect1d(pool, saved)))
    print(tgt, split_info[tgt])

sel = pd.read_csv(f'{R}/pilot_selection.csv')
bet = pd.read_csv(f'{R}/pilot_betti_parcel.csv')

rows = []
for _, r in sel.iterrows():
    fn = f'{P}/{r.target}_{r.strategy}_{r.budget}_{r.seed}_{r.clf_feat}.npy'
    rows.append(('sel', r, fn))
for _, r in bet.iterrows():
    fn = f'{P}/{r.target}_betti-{r.unit}-{r.strategy}_{r.budget}_{r.seed}_{r.clf_feat}.npy'
    rows.append(('betti', r, fn))

test_idx = {t: np.load(f'{P}/{t}_test_idx.npy') for t in ['PT', 'SE', 'SK', 'LV']}
out = []
used = set()
for src, r, fn in rows:
    pred = np.load(fn).astype(int)
    used.add(os.path.basename(fn))
    yt = y[test_idx[r.target]]
    assert len(pred) == len(yt), fn
    m = my_metrics(yt, pred)
    out.append(dict(src=src, target=r.target, strategy=r.strategy, budget=r.budget, seed=r.seed, clf_feat=r.clf_feat,
                    **{f'my_{k}': v for k, v in m.items()},
                    **{f'csv_{k}': r[k] for k in ['ba', 'ba_ge30', 'macro_f1']}))
o = pd.DataFrame(out)
for k in ['ba', 'ba_ge30', 'macro_f1']:
    o[f'd_{k}'] = (o[f'my_{k}'] - o[f'csv_{k}']).abs()
    print(k, 'max abs diff', o[f'd_{k}'].max(), 'n>1e-12', (o[f'd_{k}'] > 1e-12).sum())
print('rows compared', len(o))
allf = set(f for f in os.listdir(P) if not f.endswith('test_idx.npy'))
print('pred files not referenced by CSV:', sorted(allf - used))
o.to_csv(f'{OUTD}/recomputed_metrics.csv', index=False)
