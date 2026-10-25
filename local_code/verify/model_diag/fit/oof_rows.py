"""免训练部分：阶段 0 外国分组留出袋外（300 轮，种子 0）指标 + rows 中 k=0 / SRS 的目标指标。"""
import glob, os, sys
PROJ = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np, pandas as pd
import guidance_G as G
from fitdiag import ba_oa
GRES = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
OUT = '/Users/perng_mini/AlphaCrops_local/verify/model_diag/fit'
D = G.GData()
rows = []
for c in D.countries:
    for s in (0, 1):
        S = np.load(f'{GRES}/stage0/{c}_s{s}.npz')
        yF = D.y[S['foreign']]
        for f in ('ae', 's2'):
            m = ba_oa(yF, S[f'P_F_oof_{f}'].astype(np.float64))
            # 按外国各国分别算，再平均（留组 = 每个国家都在袋外）
            fc = D.ctry[S['foreign']]
            per = [ba_oa(yF[fc == k], S[f'P_F_oof_{f}'][fc == k].astype(np.float64)) for k in np.unique(fc)]
            rows.append(dict(tgt=c, split=s, feat=f, oof_oa=m['oa'], oof_ba=m['ba'], oof_nll=m['nll'],
                             oof_percountry_oa=np.mean([p['oa'] for p in per]),
                             oof_percountry_ba=np.mean([p['ba'] for p in per]),
                             temp=float(S[f'temp_{f}'])))
oof = pd.DataFrame(rows)
oof.to_csv(f'{OUT}/oof_stage0.csv', index=False)
print(oof.groupby(['feat', 'split'])[['oof_oa', 'oof_ba', 'oof_percountry_oa', 'oof_percountry_ba', 'temp']].mean().round(2))
d = pd.concat([pd.read_csv(f) for f in glob.glob(f'{GRES}/rows_t1_*.csv')])
d = d[d.method.isin(['foreign_only', 'srs'])]
lc = d.groupby(['clf', 'target', 'split', 'k'])[['ba_ge30', 'oa', 'ba_i200']].mean().reset_index()
lc.to_csv(f'{OUT}/rows_k0_srs.csv', index=False)
print(lc.groupby(['clf', 'k'])[['ba_ge30', 'oa']].mean().round(2))
k0 = lc[lc.k == 0].pivot_table(index=['target'], columns=['clf', 'split'], values='ba_ge30').round(1)
print(k0)
