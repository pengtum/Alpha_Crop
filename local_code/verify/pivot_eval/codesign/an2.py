import numpy as np, pandas as pd
pd.set_option('display.width', 250)
R = pd.read_csv('diag2_frontier_rows.csv')
def env(c):
    """平均前沿的上包络：按 OA 排序，BA 取累计最大（从右往左）。返回插值函数 BA(OA)。"""
    c = c.sort_values('oa')
    oa, ba = c.oa.to_numpy(), c.ba.to_numpy()
    ba_env = np.maximum.accumulate(ba[::-1])[::-1]
    return lambda x: np.interp(x, oa, ba_env, left=np.nan, right=np.nan), oa.min(), oa.max()
for sub, mask in (('全部 11 国', R.tgt.notna()), ('去掉 RO、PT', ~R.tgt.isin(['RO', 'PT']))):
    M = R[mask].groupby(['learner', 'feat', 'k', 'tau'])[['oa', 'ba']].mean().reset_index()
    print(f'===== {sub}')
    for f in ('ae', 's2'):
        for k in (50, 200):
            m = M[(M.feat == f) & (M.k == k)]
            k0 = m[(m.learner == 'k0') & (m.tau == 0)].iloc[0]
            ref = m[(m.learner == 'lgb_i300') & (m.tau == 0)].iloc[0]
            out = [f'[{f} k{k}] k0 BA {k0.ba:.2f} OA {k0.oa:.2f} | LGB r50 τ=0: BA {ref.ba:.2f} OA {ref.oa:.2f}']
            for L in m.learner.unique():
                fn, lo, hi = env(m[m.learner == L])
                b_at = fn(ref.oa)
                b_at2 = fn(ref.oa - 1.0)
                b0 = m[(m.learner == L) & (m.tau == 0)].iloc[0]
                # 该学习器能达到的最大 BA 以及此时 OA
                mx = m[m.learner == L].sort_values('ba').iloc[-1]
                out.append(f'  {L:15s} τ=0: BA {b0.ba:6.2f} OA {b0.oa:6.2f} | 在 LGB 的 OA 处 BA {b_at:6.2f} (Δ {b_at - ref.ba:+.2f})'
                           f' | OA 低 1pp 处 BA {b_at2:6.2f} | maxBA {mx.ba:.2f}@τ{mx.tau:+.1f} OA {mx.oa:.2f} | OA 范围 {lo:.1f}–{hi:.1f}')
            print('\n'.join(out))
