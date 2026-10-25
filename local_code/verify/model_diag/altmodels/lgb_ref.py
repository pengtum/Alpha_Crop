"""LightGBM 参照：从 G 的 rows CSV 取 s0 r0 的 foreign_only(k=0) 与 srs k=200；以及 E1×{AE,S2}×k=200。
另算容量诊断：第 200/250/300 轮的 ba；外国分组留出（阶段 0 P_F_oof）的 ba。"""
import sys; sys.path.insert(0, '/Users/perng_mini/AlphaCrops_local/verify/model_diag/altmodels')
from altlib import *
import pandas as pd
D = G.GData()
rs = []
for c in D.countries:
    r = pd.read_csv(f'{GOUT}/rows_t1_{c}_s0_r0.csv')
    rs.append(r)
R = pd.concat(rs)
ref = R[((R.method == 'foreign_only') & (R.k == 0)) | ((R.method == 'srs') & (R.k == 200) & (R.cond == 'none'))]
ref = ref[['target', 'clf', 'k', 'ba_ge30', 'oa', 'ba_i200', 'ba_i250', 'ba_i300', 'oa_i200', 'oa_i300', 't_train', 'ba_P2', 'oa_P2']]
ref.to_csv(os.path.join(OUT, 'lgb_ref_s0r0.csv'), index=False)
e1 = R[(R.k == 200) & R.method.isin(G.E1) & R.cond.isin(['AE', 'S2'])][['target', 'cond', 'method', 'clf', 'ba_ge30', 'oa', 'sel_hash']]
e1.to_csv(os.path.join(OUT, 'lgb_e1_k200_s0r0.csv'), index=False)
print(len(ref), len(e1))
# 轮数趋势
for k in (0, 200):
    x = ref[ref.k == k]
    d = x.ba_i300 - x.ba_i200
    print(f'k={k} ba_i300-ba_i200: mean {d.mean():+.2f}, #neg {int((d<0).sum())}/{len(d)}; by clf:', x.assign(d=d).groupby('clf').d.mean().round(2).to_dict())
    print(x.assign(d=d).pivot(index='target', columns='clf', values='d').round(2).T.to_string())
# 外国分组留出 ba（外国 10 国分 5 组，LightGBM 300 棵）
out = []
for c in D.countries:
    S = np.load(f'{GOUT}/stage0/{c}_s0.npz')
    yF = D.y[S['foreign']]
    bigF = np.flatnonzero(np.bincount(yF, minlength=10) >= 30)
    for f in ('ae', 's2'):
        P = S[f'P_F_oof_{f}']
        oa, ba = scores(yF, P.argmax(1), bigF)
        out.append(dict(target=c, feat=f, foreign_oof_ba=ba, foreign_oof_oa=oa))
O = pd.DataFrame(out)
O.to_csv(os.path.join(OUT, 'foreign_oof_ba.csv'), index=False)
m = ref[ref.k == 0].rename(columns={'clf': 'feat'}).merge(O, on=['target', 'feat'])
print(m[['target', 'feat', 'foreign_oof_ba', 'ba_ge30', 'foreign_oof_oa', 'oa']].round(1).to_string())
print(m.groupby('feat')[['foreign_oof_ba', 'ba_ge30', 'foreign_oof_oa', 'oa']].mean().round(2))
