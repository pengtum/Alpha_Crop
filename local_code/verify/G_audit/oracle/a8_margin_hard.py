"""审计 8：ORACLE-margin 所选地块是"边界难样本"还是"疑似错标"：真类是否在 ORACLE 前二、真类概率分布；RO 的类别构成。只读。"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import guidance_G as G
OUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
D = G.GData()
rows = []
for tgt in G.COUNTRIES_G:
    for split in (0, 1):
        S = dict(np.load(os.path.join(OUT, 'stage0', f'{tgt}_s{split}.npz')))
        O = dict(np.load(os.path.join(OUT, 'oracle', f'{tgt}_s{split}.npz')))
        pp = G.pool_positions(S, split, 0); y = D.y[S['allT'][pp]]
        P = O['P_T']; top2 = np.argsort(-P, 1)[:, :2]
        in2 = (top2 == y[:, None]).any(1); pt = P[np.arange(len(y)), y]
        for cond in ('ORACLE', 'S2', 'AE'):
            for k in (50, 200):
                sp = np.load(os.path.join(OUT, 'sel', f'{tgt}_s{split}_r0_{cond}_margin_k{k}.npz'))['sel_pos']
                rows.append(dict(tgt=tgt, split=split, cond=cond, k=k, true_in_top2=in2[sp].mean(),
                                 ptrue_lt_0p1=(pt[sp] < 0.1).mean(), pool_true_in_top2=in2.mean(),
                                 pool_ptrue_lt_0p1=(pt < 0.1).mean(),
                                 comp=' '.join(f'{c}:{n}' for c, n in enumerate(np.bincount(y[sp], minlength=10)) if n),
                                 pool=' '.join(f'{c}:{n/len(y):.2f}' for c, n in enumerate(np.bincount(y, minlength=10)) if n)))
df = pd.DataFrame(rows)
print(df.groupby(['cond', 'k'])[['true_in_top2', 'ptrue_lt_0p1', 'pool_true_in_top2', 'pool_ptrue_lt_0p1']].mean().round(3).to_string())
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 120)
print(df[(df.tgt.isin(['RO', 'SE'])) & (df.k == 200)][['tgt', 'split', 'cond', 'comp', 'pool']].to_string(index=False))
