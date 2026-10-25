"""审计 6：用同一函数 make_split 生成 s2、s3、s4（只算划分，不训练），与已用测试集的重合度。只读。"""
import os, sys, glob
LE = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, LE)
import numpy as np
import pandas as pd
import guidance_G as G
from stageA import km_xy, CELL_KM, TEST_FRAC

R = '/Users/perng_mini/AlphaCrops_local/results'
OUTD = '/Users/perng_mini/AlphaCrops_local/verify/testset_audit/empirical'
D = G.GData()
Y = np.asarray(D.y)


def cells_of(tgt):
    """与 make_split 相同的格子编号（与划分号无关）。"""
    t_idx = np.flatnonzero(D.ctry == tgt)
    keep = ~pd.DataFrame({'a': D.lon[t_idx], 'b': D.lat[t_idx], 'c': D.area[t_idx]}).duplicated().to_numpy()
    t_idx = t_idx[keep]
    xy = km_xy(D.lon[t_idx], D.lat[t_idx], D.lat[t_idx].mean())
    cell = pd.factorize(pd.Series(np.floor(xy[:, 0] / CELL_KM).astype(int)).astype(str) + '_' +
                        pd.Series(np.floor(xy[:, 1] / CELL_KM).astype(int)).astype(str))[0]
    return t_idx, cell


def jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


# 已用的测试集（按地块）
used_A_s2 = {t: set(np.load(f'{R}/stageA/stage0/{t}_s2.npz')['test']) for t in ('PT', 'BE', 'SI')}
chk3_test = {t: set(np.load(f'{R}/check3/{t}_split.npz')['test']) for t in G.COUNTRIES_G}
chk3_sp = {t: set(np.load(f'{R}/check3/{t}_split_sp.npz')['test_sp']) for t in G.COUNTRIES_G}
pilot_test = {t: set(np.load(f'{R}/preds/{t}_test_idx.npy')) for t in ('PT', 'SE', 'SK', 'LV')}
# 索引空间核对：check3 保存的 y_test 与 GData 的标签一致
idx_ok = all(np.array_equal(Y[np.load(f'{R}/check3/{t}_split.npz')['test']], np.load(f'{R}/check3/{t}_split.npz')['y_test'])
             for t in G.COUNTRIES_G)
print('check3 index space == GData:', idx_ok)
# G 中被选中过（标签进入训练）的地块
g_sel = {}
for f in glob.glob(f'{R}/guidance_G/sel/*.npz'):
    t = os.path.basename(f)[:2]
    g_sel.setdefault(t, set()).update(np.load(f)['sel'].tolist())

rows = []
for tgt in G.COUNTRIES_G:
    t_idx, cell = cells_of(tgt)
    cell_of = dict(zip(t_idx.tolist(), cell.tolist()))
    n_cells = cell.max() + 1
    sp = {s: G.make_split(D, tgt, s) for s in range(5)}
    tc = {s: set(np.unique([cell_of[i] for i in sp[s]['test']])) for s in sp}
    tp = {s: set(sp[s]['test'].tolist()) for s in sp}
    old_c = tc[0] | tc[1]
    old_p = tp[0] | tp[1]
    if tgt in used_A_s2:
        old_c_A = old_c | tc[2]
        old_p_A = old_p | used_A_s2[tgt]
        assert used_A_s2[tgt] == tp[2]
    else:
        old_c_A, old_p_A = old_c, old_p
    for s in (2, 3, 4):
        new_c, new_p = tc[s], tp[s]
        r = dict(tgt=tgt, split=s, n_cells=n_cells, n_test_cells=len(new_c), n_test=len(new_p),
                 jac_cells_s0=jac(new_c, tc[0]), jac_cells_s1=jac(new_c, tc[1]),
                 frac_cells_in_s0us1=len(new_c & old_c) / len(new_c),
                 frac_parcels_in_s0us1_test=len(new_p & old_p) / len(new_p),
                 frac_parcels_in_any_used_test_GA=len(new_p & old_p_A) / len(new_p),
                 frac_parcels_in_check3_test=len(new_p & chk3_test[tgt]) / len(new_p),
                 frac_parcels_in_check3_sp_test=len(new_p & chk3_sp[tgt]) / len(new_p),
                 frac_parcels_in_pilot_test=(len(new_p & pilot_test[tgt]) / len(new_p)) if tgt in pilot_test else np.nan,
                 frac_parcels_selected_in_G=len(new_p & g_sel.get(tgt, set())) / len(new_p),
                 frac_parcels_never_G_A_test=1 - len(new_p & old_p_A) / len(new_p))
        rows.append(r)
    # 从未当过 G/阶段 A 测试的格子与地块（可作为真正未看过的测试储备）
    fresh_c = set(range(n_cells)) - old_c_A
    fresh_p = set(t_idx.tolist()) - old_p_A
    rows.append(dict(tgt=tgt, split='never_tested_reservoir', n_cells=n_cells, n_test_cells=len(fresh_c),
                     n_test=len(fresh_p), frac_cells_in_s0us1=len(fresh_c) / n_cells,
                     frac_parcels_never_G_A_test=len(fresh_p) / len(t_idx),
                     frac_parcels_in_check3_test=len(fresh_p & chk3_test[tgt]) / max(1, len(fresh_p)),
                     frac_parcels_in_check3_sp_test=len(fresh_p & chk3_sp[tgt]) / max(1, len(fresh_p))))
    # 同一划分内 s0 vs s1 的重合（供参照）
    rows.append(dict(tgt=tgt, split='s1_vs_s0', jac_cells_s0=jac(tc[1], tc[0]),
                     frac_cells_in_s0us1=len(tc[1] & tc[0]) / len(tc[1]),
                     frac_parcels_in_s0us1_test=len(tp[1] & tp[0]) / len(tp[1])))
df = pd.DataFrame(rows)
df.to_csv(f'{OUTD}/a4_new_splits.csv', index=False)
pd.set_option('display.width', 250)
print(df.round(3).to_string())
new = df[df.split.isin([2, 3, 4])]
print('\n新划分（s2–s4，33 个）汇总：')
print(new[['jac_cells_s0', 'jac_cells_s1', 'frac_cells_in_s0us1', 'frac_parcels_in_s0us1_test',
           'frac_parcels_in_any_used_test_GA', 'frac_parcels_in_check3_test', 'frac_parcels_selected_in_G']].astype(float).describe().round(3).to_string())
print('理论：随机 30% 格子，与 s0∪s1 重合的期望 = 1 - 0.7^2 =', 1 - 0.7 ** 2, '；再加 s2（PT/BE/SI）=', 1 - 0.7 ** 3)
# s2 vs s3 重合
for tgt in G.COUNTRIES_G:
    pass
