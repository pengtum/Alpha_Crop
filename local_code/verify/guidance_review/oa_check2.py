"""用检验 2 保存的逐地块预测补算 OA，并估计 OA 口径下 Δ（ae−s2）的方差分量。"""
import sys
sys.path.insert(0, '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion')
import numpy as np, pandas as pd
from common import load_dataset
P = '/Users/perng_mini/AlphaCrops_local/results/preds'
df = load_dataset(columns=None) if False else load_dataset()
classes = sorted(df['label'].unique())
y = df['label'].map({c: i for i, c in enumerate(classes)}).to_numpy()
rows = []
for t in ['PT', 'SE', 'SK', 'LV']:
    te = np.load(f'{P}/{t}_test_idx.npy'); yt = y[te]
    for strat, tag in [('ae_rep', 'ae_rep'), ('s2_rep', 's2_rep'), ('ae_gap', 'ae_gap'), ('s2_gap', 's2_gap'),
                       ('random', 'random'), ('img_rep_ae', 'betti-parcel-img_rep_ae'), ('img_rep_s2', 'betti-parcel-img_rep_s2')]:
        for k in [50, 200]:
            for s in [0, 1, 2]:
                for cf in ['ae', 's2']:
                    p = np.load(f'{P}/{t}_{tag}_{k}_{s}_{cf}.npy')
                    rows.append(dict(target=t, strategy=strat, budget=k, seed=s, clf_feat=cf, oa=100 * (p == yt).mean()))
d = pd.DataFrame(rows)
w = d.pivot_table(index=['target', 'budget', 'clf_feat', 'seed'], columns='strategy', values='oa').reset_index()
w['d_rep'] = w.ae_rep - w.s2_rep; w['d_gap'] = w.ae_gap - w.s2_gap; w['d_betti'] = w.img_rep_ae - w.img_rep_s2
cols = ['d_rep', 'd_gap', 'd_betti']
w['d_fam'] = w[cols].mean(axis=1)
print('OA 口径：逐单元 Δ 的均值/SD\n', w[cols + ['d_fam']].agg(['mean', 'std']).round(3))
print('\n国×k×clf 内跨种子 SD：', {c: round(np.sqrt(w.groupby(['target', 'budget', 'clf_feat'])[c].var().mean()), 3) for c in cols + ['d_fam']})
print('逐国 SD（国内全部单元）：\n', w.groupby('target')[cols + ['d_fam']].std().round(3))
print('\n逐国 × 分类器均值：\n', w.groupby(['target', 'clf_feat'])[cols + ['d_fam']].mean().round(3))
r = w.copy()
for c in cols:
    r[c] -= r.groupby(['target', 'budget', 'clf_feat'])[c].transform('mean')
print('\n方法间残差相关：\n', r[cols].corr().round(2))
print('\nrandom 跨种子 SD（OA）：', round(np.sqrt(w.groupby(['target', 'budget', 'clf_feat'])['random'].var().mean()), 3))
w.to_csv('check2_cell_diffs_oa.csv', index=False)
