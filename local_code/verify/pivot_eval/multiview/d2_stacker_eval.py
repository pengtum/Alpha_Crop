# D2：两视图校准层在 k=0 外国模型之上能挽回多少 ba_ge30；与 G 中"用 k 个随机标签重训 LightGBM"比较。
from common_mv import *
import pandas as pd, stacker as st
from scipy import stats
rows = []
KS = (50, 100, 200, 400)
NDRAW = 10
for t in COUNTRIES:
    for s in (0, 1):
        z = np.load(f'{R}/results/guidance_G/stage0/{t}_s{s}.npz')
        nc = len(z['cand'])
        pp = pool_positions(nc, s, 0)
        yP = Y[z['allT'][pp]]; yT = Y[z['test']]
        big = np.flatnonzero(np.bincount(yT, minlength=10) >= 30)
        Pa, Ps = z['P_T_ae'].astype(np.float64), z['P_T_s2'].astype(np.float64)
        LaP, LsP = st._ll(Pa[pp]), st._ll(Ps[pp])
        LaT, LsT = st._ll(Pa[nc:]), st._ll(Ps[nc:])
        base = {}
        for nm, P in (('k0_ae', Pa[nc:]), ('k0_s2', Ps[nc:]), ('k0_avg', (Pa[nc:] + Ps[nc:]) / 2)):
            oa, ba = st.scores(yT, P, big); base[nm] = ba
            rows.append(dict(target=t, split=s, arm=nm, k=0, draw=-1, oa=oa, ba=ba))
        # 神谕：全池 3 万标签拟合（弱正则）
        for views in ('both', 'ae', 's2'):
            for mode in ('bal', 'unw'):
                th = st.fit(LaP, LsP, yP, mode=mode, gw=0.01, gb=0.01, views=views)
                oa, ba = st.scores(yT, st.predict(th, LaT, LsT), big)
                rows.append(dict(target=t, split=s, arm=f'orc_{views}_{mode}', k=len(yP), draw=-1, oa=oa, ba=ba))
        # 可实现：k 个 SRS 标签
        for k in KS:
            for d in range(NDRAW):
                rng = np.random.default_rng(1000 * s + 17 * d + k)
                sel = rng.choice(len(yP), k, replace=False)
                for views in ('both', 'ae', 's2'):
                    th = st.fit(LaP[sel], LsP[sel], yP[sel], mode='bal', gw=5.0, gb=1.0, views=views)
                    oa, ba = st.scores(yT, st.predict(th, LaT, LsT), big)
                    rows.append(dict(target=t, split=s, arm=f'cal_{views}', k=k, draw=d, oa=oa, ba=ba))
        print(t, s, 'done', flush=True)
df = pd.DataFrame(rows)
df.to_csv('d2_stacker_rows.csv', index=False)
