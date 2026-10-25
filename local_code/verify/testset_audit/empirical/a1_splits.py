"""审计 1：重算 make_split，与 G 的 stage0 npz、阶段 A 的 stage0 npz 逐位比较。只读。"""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '4')
LE = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, LE)
import numpy as np
import guidance_G as G

R = '/Users/perng_mini/AlphaCrops_local/results'
OUTD = '/Users/perng_mini/AlphaCrops_local/verify/testset_audit/empirical'
D = G.GData()
print('GData ok; n =', len(D.y), 'countries', D.countries)
KEYS = ['test', 'cand', 'foreign', 'allT', 'efar', 'd_far', 'lat0']
res = []
splits = {}
for tgt in G.COUNTRIES_G:
    for s in (0, 1):
        t0 = time.time()
        sp = G.make_split(D, tgt, s)
        splits[(tgt, s)] = sp
        z = np.load(f'{R}/guidance_G/stage0/{tgt}_s{s}.npz')
        row = dict(tgt=tgt, split=s, n_test=len(sp['test']), n_cand=len(sp['cand']), n_foreign=len(sp['foreign']),
                   n_efar=int(sp['efar'].sum()))
        for k in KEYS:
            a, b = np.asarray(sp[k]), np.asarray(z[k])
            row[f'eq_{k}'] = bool(a.shape == b.shape and (np.array_equal(a, b) if k != 'd_far' else np.array_equal(a.astype(np.float32), b)))
        # 阶段 A
        pA = f'{R}/stageA/stage0/{tgt}_s{s}.npz'
        if os.path.exists(pA):
            zA = np.load(pA)
            for k in ('test', 'cand', 'foreign', 'allT', 'lat0'):
                row[f'A_eq_{k}'] = bool(np.array_equal(np.asarray(zA[k]), np.asarray(z[k])))
        res.append(row)
        print(row, f'{time.time()-t0:.1f}s', flush=True)
# 阶段 A 的 s2：用同一函数生成
for tgt in ('PT', 'BE', 'SI'):
    sp = G.make_split(D, tgt, 2)
    splits[(tgt, 2)] = sp
    zA = np.load(f'{R}/stageA/stage0/{tgt}_s2.npz')
    row = dict(tgt=tgt, split=2, n_test=len(sp['test']), n_cand=len(sp['cand']), n_foreign=len(sp['foreign']))
    for k in ('test', 'cand', 'foreign', 'allT', 'lat0'):
        row[f'A_eq_{k}'] = bool(np.array_equal(np.asarray(zA[k]), np.asarray(sp[k])))
    res.append(row)
    print('stageA s2', row, flush=True)
import pandas as pd
pd.DataFrame(res).to_csv(f'{OUTD}/a1_splits.csv', index=False)
# 缓存测试集索引供后续脚本使用
np.savez_compressed(f'{OUTD}/splits_cache.npz', **{f'{t}_s{s}_{k}': np.asarray(v[k]) for (t, s), v in splits.items()
                                                  for k in ('test', 'cand', 'foreign', 'efar', 'd_far')},
                    **{f'{t}_s{s}_lat0': np.float64(v['lat0']) for (t, s), v in splits.items()})
print('done')
