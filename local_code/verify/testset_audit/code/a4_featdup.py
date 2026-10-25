"""审计 4：测试地块与候选池/外国池之间特征向量完全相同（AE 或 S2）的数量。只读。"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
import guidance_G as G
R = '/Users/perng_mini/AlphaCrops_local/results'
D = G.GData()
def rowhash(X, idx):
    A = np.ascontiguousarray(np.asarray(X[idx]))
    return pd.util.hash_array(A.view(np.dtype((np.void, A.dtype.itemsize * A.shape[1]))).ravel().astype(object)) if False else \
        np.array([hash(A[i].tobytes()) for i in range(len(A))])
for t in D.countries:
    out = [t]
    c3 = np.load(f'{R}/check3/{t}_split.npz')
    g = np.load(f'{R}/guidance_G/stage0/{t}_s0.npz')
    for f in ('ae', 's2'):
        ht = rowhash(D.X[f], c3['test']); hp = set(rowhash(D.X[f], c3['pool']).tolist())
        out.append(f'check3 {f}: test==pool feat {np.isin(ht, list(hp)).sum()}')
        ht = rowhash(D.X[f], g['test']); hc = set(rowhash(D.X[f], g['cand']).tolist()); hf = set(rowhash(D.X[f], g['foreign']).tolist())
        out.append(f'G s0 {f}: test==cand {np.isin(ht, list(hc)).sum()} test==foreign {np.isin(ht, list(hf)).sum()} dup-within-test {len(ht)-len(set(ht.tolist()))}')
    print(' | '.join(out), flush=True)
