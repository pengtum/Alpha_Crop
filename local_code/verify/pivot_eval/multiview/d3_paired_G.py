# D3：与 G 完全配对——同一批 k 个标签（G 保存的选样），比较"重训 LightGBM（G 行）"与"冻结 k=0 两视图 + 目标校准层"。
from common_mv import *
import pandas as pd, glob, stacker as st
rows = []
cache = {}
files = sorted(glob.glob(f'{R}/results/guidance_G/sel/*.npz'))
for f in files:
    b = os.path.basename(f)[:-4]
    parts = b.split('_')
    t, s, r = parts[0], int(parts[1][1:]), int(parts[2][1:])
    k = int(parts[-1][1:])
    rest = parts[3:-1]
    if rest[0] in ('NATIVE', 'RAWZ'):
        cond, method = rest[0] + '_' + rest[1], '_'.join(rest[2:])
    else:
        cond, method = rest[0], '_'.join(rest[1:])
    if k not in (50, 100, 200, 400): continue
    key = (t, s, r)
    if key not in cache:
        z = np.load(f'{R}/results/guidance_G/stage0/{t}_s{s}.npz')
        nc = len(z['cand']); pp = pool_positions(nc, s, r)
        Pa, Ps = z['P_T_ae'].astype(np.float64), z['P_T_s2'].astype(np.float64)
        yT = Y[z['test']]
        cache.clear()
        cache[key] = dict(pp=pp, LaP=st._ll(Pa[pp]), LsP=st._ll(Ps[pp]), LaT=st._ll(Pa[nc:]), LsT=st._ll(Ps[nc:]),
                          yP=Y[z['allT'][pp]], yT=yT, big=np.flatnonzero(np.bincount(yT, minlength=10) >= 30))
    u = cache[key]
    sp = np.load(f)['sel_pos']
    ys = u['yP'][sp]
    for a in (0.0, 0.5, 1.0):
        for views in ('both', 'ae', 's2'):
            th = st.fit(u['LaP'][sp], u['LsP'][sp], ys, mode=a, gw=5.0, gb=1.0, views=views)
            oa, ba = st.scores(u['yT'], st.predict(th, u['LaT'], u['LsT']), u['big'])
            rows.append(dict(target=t, split=s, rep=r, cond=cond, method=method, k=k, alpha=a, views=views,
                             oa=oa, ba=ba, n_cls=len(np.unique(ys))))
df = pd.DataFrame(rows)
df.to_csv('d3_paired_rows.csv', index=False)
print(len(df))
