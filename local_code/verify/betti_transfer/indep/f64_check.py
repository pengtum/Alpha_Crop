# 稳健性：KMeans 输入改为 float64 时，主设定 k=200 的 TV_a 是否变化（只作复核附注）
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[v] = '2'
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, '/Users/perng_mini/AlphaCrops_local/verify/betti_transfer/indep')
import zipfile
import numpy as np, pandas as pd
import indep_diag as D
from threadpoolctl import threadpool_limits
D.SETTINGS = [('q95', 0.95, [200])]
out = []
with threadpool_limits(limits=2):
    for tgt in D.COUNTRIES:
        for split in D.SPLITS:
            with zipfile.ZipFile(os.path.join(D.STAGE0, f'{tgt}_s{split}.npz')) as z:
                cand = D.npz_full_1d(z, 'cand')
                geo = {f: (D.npz_first_rows(z, f'G_{f}', len(cand))[0], D.npz_first_rows(z, f'GF_{f}')[0]) for f in D.SPACES}
            for rep in D.REPS:
                pos = D.pool_positions({'cand': cand}, split, rep)
                for f in D.SPACES:
                    Gc, GF = geo[f]
                    D.nn_dist = lambda *a, **k: np.array([1.0])  # 跳过分离比
                    rows, _, _ = D.unit_space(Gc[pos].astype(np.float64), GF.astype(np.float64))
                    out.append(dict(target=tgt, split=split, rep=rep, space=f, tv_a64=rows[0]['tv_a'], mis=rows[0]['n_mislab']))
                    print(out[-1], flush=True)
O = pd.DataFrame(out)
R = pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/betti_transfer/indep/indep_rows.csv')
R = R[(R.thr == 'q95') & (R.k == 200)][['target', 'split', 'rep', 'space', 'tv_a']]
M = R.merge(O, on=['target', 'split', 'rep', 'space'])
M['d'] = M.tv_a64 - M.tv_a
print(M.groupby('space').agg(med32=('tv_a', 'median'), med64=('tv_a64', 'median'), n_diff=('d', lambda x: (x.abs() > 1e-12).sum()), maxabs=('d', lambda x: x.abs().max())))
print(M[M.d.abs() > 1e-12])
