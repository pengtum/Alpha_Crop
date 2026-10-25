"""免训练：阶段 0 外国分组留出袋外概率 -> OA / ba（合并 与 逐国），核对 fit 代理的 oof_stage0.csv。"""
import os, sys
PROJ = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np, pandas as pd
import guidance_G as G
GRES = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
D = G.GData()
def ba(y, ph):
    cnt = np.bincount(y, minlength=10); big = np.flatnonzero(cnt >= 30)
    return 100 * np.mean([(ph[y == c] == c).mean() for c in big])
rows = []
for tgt in D.countries:
    for split in (0, 1):
        S = np.load(os.path.join(GRES, 'stage0', f'{tgt}_s{split}.npz'))
        fr = S['foreign']; y = D.y[fr]; cc = D.ctry[fr]
        for f in ('ae', 's2'):
            ph = S[f'P_F_oof_{f}'].argmax(1)
            per = np.mean([ba(y[cc == c], ph[cc == c]) for c in np.unique(cc)])
            rows.append(dict(tgt=tgt, split=split, feat=f, oa=100 * (ph == y).mean(), ba_pooled=ba(y, ph), ba_percountry=per))
df = pd.DataFrame(rows)
print(df.groupby(['split', 'feat'])[['oa', 'ba_pooled', 'ba_percountry']].mean().round(2))
