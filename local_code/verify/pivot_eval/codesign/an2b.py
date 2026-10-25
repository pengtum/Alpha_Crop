"""在固定 OA 水平处比较各学习器平均前沿（上包络）的 BA。"""
import numpy as np, pandas as pd
R = pd.read_csv('diag2_frontier_rows.csv')
def env_at(c, x):
    c = c.sort_values('oa'); oa, ba = c.oa.to_numpy(), c.ba.to_numpy()
    e = np.maximum.accumulate(ba[::-1])[::-1]
    return float(np.interp(x, oa, e, left=np.nan, right=np.nan))
L = ['k0', 'lgb_i300', 'lgb_i200', 'bias_l0.1_a1', 'feat_l0.1_a1', 'feat_l0.03_a1', 'mat_l1_a1', 'bias_l0.01_a1']
for sub, mask in (('全部', R.tgt.notna()), ('去RO/PT', ~R.tgt.isin(['RO', 'PT']))):
    M = R[mask].groupby(['learner', 'feat', 'k', 'tau'])[['oa', 'ba']].mean().reset_index()
    for f in ('ae', 's2'):
        for k in (50, 200):
            m = M[(M.feat == f) & (M.k == k)]
            oa0 = m[(m.learner == 'k0') & (m.tau == 0)].oa.iloc[0]
            oaL = m[(m.learner == 'lgb_i300') & (m.tau == 0)].oa.iloc[0]
            lv = [oa0 - 3, oa0, (oa0 + oaL) / 2, oaL - 0.3]
            print(f'{sub} {f} k{k}  OA 水平: ' + ' '.join(f'{x:.1f}' for x in lv))
            for l in L:
                print(f'   {l:14s} ' + ' '.join(f'{env_at(m[m.learner == l], x):6.2f}' for x in lv))
