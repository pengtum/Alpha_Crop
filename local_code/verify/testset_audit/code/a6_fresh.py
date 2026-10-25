"""审计 6：哪些地块/格子从未作为评估测试集；rep0 与 rep1 候选池的重叠。只读。"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
import guidance_G as G
from stageA import km_xy, CELL_KM
R = '/Users/perng_mini/AlphaCrops_local/results'
D = G.GData()
rows = []
for t in D.countries:
    g = {s: np.load(f'{R}/guidance_G/stage0/{t}_s{s}.npz') for s in (0, 1)}
    t_idx = np.flatnonzero(D.ctry == t)
    keep = ~pd.DataFrame({'a': D.lon[t_idx], 'b': D.lat[t_idx], 'c': D.area[t_idx]}).duplicated().to_numpy()
    t_idx = t_idx[keep]
    xy = km_xy(D.lon[t_idx], D.lat[t_idx], D.lat[t_idx].mean())
    cell = pd.factorize(pd.Series(np.floor(xy[:, 0] / CELL_KM).astype(int)).astype(str) + '_' +
                        pd.Series(np.floor(xy[:, 1] / CELL_KM).astype(int)).astype(str))[0]
    used = np.isin(t_idx, g[0]['test']) | np.isin(t_idx, g[1]['test'])
    if t in ('PT', 'BE', 'SI'):
        used |= np.isin(t_idx, np.load(f'{R}/stageA/stage0/{t}_s2.npz')['test'])
    used_cells = np.unique(cell[used])
    fresh_cell = ~np.isin(cell, used_cells)
    c3 = np.load(f'{R}/check3/{t}_split.npz')['test']
    sp = np.load(f'{R}/check3/{t}_split_sp.npz')['test_sp']
    r = dict(country=t, n=len(t_idx), n_cells=cell.max() + 1, cells_never_G_test=len(np.setdiff1d(np.arange(cell.max() + 1), used_cells)),
             parcels_in_fresh_cells=int(fresh_cell.sum()), frac_parcels_fresh=fresh_cell.mean(),
             fresh_and_not_check3_test=int((fresh_cell & ~np.isin(t_idx, c3) & ~np.isin(t_idx, sp)).sum()))
    for s in (0, 1):
        n_c = len(g[s]['cand'])
        p0, p1 = G.pool_positions(g[s], s, 0), G.pool_positions(g[s], s, 1)
        r[f'pool_overlap_r0r1_s{s}'] = len(np.intersect1d(p0, p1)) / len(p0)
    rows.append(r)
df = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(df.round(3).to_string())
df.to_csv('/Users/perng_mini/AlphaCrops_local/verify/testset_audit/code/a6_fresh.csv', index=False)
