import sys, glob, numpy as np, pandas as pd
sys.path.insert(0, '.')
import guidance_G as G
from sklearn.decomposition import PCA
OUT = G.OUT
D = G.GData()
S = dict(np.load(f'{OUT}/stage0/AT_s0.npz'))
fr, allT = S['foreign'], S['allT']
# 1) 稳健 PCA 独立实现
for f in ('ae', 's2'):
    Z = np.vstack([D.X[f][fr], D.X[f][allT]]).astype(np.float64)
    Z = np.clip((Z - Z.mean(0)) / (Z.std(0) + 1e-9), -5, 5)
    T = PCA(32, random_state=0).fit_transform(Z)[len(fr):]
    Gs = S[f'G_{f}'].astype(np.float64)
    sgn = np.sign((T * Gs).sum(0))
    print(f, 'G 与独立实现最大差', float(np.abs(T * sgn - Gs).max()), '特征数', D.X[f].shape[1])
# 2) 独立重训一个模型：AT s0 r0, cond AE, kcenter, k=50, clf s2
import lightgbm as lgb
from common import LGBM_PARAMS
print('LGBM_PARAMS', LGBM_PARAMS)
z = np.load(f'{OUT}/sel/AT_s0_r0_AE_kcenter_k50.npz'); sel = z['sel']
R = pd.read_csv(f'{OUT}/rows_t1_AT_s0_r0.csv')
row = R[(R.cond == 'AE') & (R.method == 'kcenter') & (R.k == 50) & (R.clf == 's2')].iloc[0]
yF = D.y[fr]; ys = D.y[sel]
n = np.bincount(yF, minlength=10).astype(float)
cw = np.where(n > 0, len(yF) / (np.count_nonzero(n) * np.maximum(n, 1)), 1.0)
w = np.r_[cw[yF], 50.0 * cw[ys]]
X = D.X['s2']
p = dict(objective='multiclass', num_class=10, learning_rate=LGBM_PARAMS['learning_rate'], num_leaves=LGBM_PARAMS['num_leaves'],
         min_data_in_leaf=LGBM_PARAMS['min_child_samples'], bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, max_bin=63,
         force_col_wise=True, num_threads=4, seed=0 * 1000 + 0, deterministic=True, verbose=-1)
m = lgb.train(p, lgb.Dataset(np.vstack([X[fr], X[sel]]), np.r_[yF, ys], weight=w), num_boost_round=300)
P = m.predict(X[S['test']])
pz = np.load(glob.glob(f'{OUT}/probs/AT_s0_r0_s2_k50_{row.sel_hash}.npz')[0])
print('probs 文件键', pz.files, '与重训 i300 最大差', float(np.abs(P - pz['i300']).max()),
      'i200 最大差', float(np.abs(m.predict(X[S['test']], num_iteration=200) - pz['i200']).max()))
yT = D.y[S['test']]; big = np.flatnonzero(np.bincount(yT, minlength=10) >= 30)
ph = P.argmax(1); rec = np.array([(ph[yT == c] == c).mean() for c in big])
print('ba_ge30 重算', 100 * rec.mean(), '行内', row.ba_ge30, 'big', big.tolist(), '类计数', np.bincount(yT, minlength=10).tolist())
ef = S['efar'].astype(bool); rec2 = [np.mean(ph[ef][yT[ef] == c] == c) for c in big if (yT[ef] == c).any()]
print('ba_far 重算', 100 * np.mean(rec2), '行内', row.ba_far)
