import numpy as np, pandas as pd
R = pd.read_csv('diag4_ens_rows.csv')
def env_at(c, x):
    c = c.sort_values('oa'); oa, ba = c.oa.to_numpy(), c.ba.to_numpy()
    e = np.maximum.accumulate(ba[::-1])[::-1]
    return float(np.interp(x, oa, e, left=np.nan, right=np.nan))
for sub, mask in (('全部', R.tgt.notna()), ('去RO/PT', ~R.tgt.isin(['RO', 'PT']))):
    M = R[mask].groupby(['sel', 'learner', 'feat', 'k', 'tau'])[['oa', 'ba']].mean().reset_index()
    for f in ('ae', 's2'):
        for k in (50, 200):
            m = M[(M.feat == f) & (M.k == k)]
            ref = m[(m.sel == 'none_srs') & (m.learner == 'lgb') & (m.tau == 0)].iloc[0]
            lv = [ref.oa - 4, ref.oa - 2, ref.oa - 1, ref.oa - 0.3, ref.oa]
            print(f'{sub} {f} k{k}: 基线 LGB+SRS τ=0 BA {ref.ba:.2f} OA {ref.oa:.2f}；列 = OA 水平 ' + ' '.join(f'{x:.1f}' for x in lv))
            for (s, l), c in m.groupby(['sel', 'learner']):
                t0 = c[c.tau == 0].iloc[0]
                print(f'   {s[:10]:10s} {l:16s} ' + ' '.join(f'{env_at(c, x) - ref.ba:+6.2f}' for x in lv)
                      + f' | τ=0: ΔBA {t0.ba - ref.ba:+.2f} ΔOA {t0.oa - ref.oa:+.2f}')
