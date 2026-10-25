"""审计 2：ORACLE 几何（LDA decision_function，C 列）逐列检查：候选池中缺席的类列是否被 sc.min() 填成按折不同的常数。
折划分用 oracle_compute 中同一 KFold(5, shuffle=True, random_state=0) 复现（只看位置，不重训）。只读。"""
import os, sys
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
sys.path.insert(0, os.getcwd())
import guidance_G as G

OUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
D = G.GData()
rows = []
for tgt in G.COUNTRIES_G:
    for split in (0, 1):
        S = dict(np.load(os.path.join(OUT, 'stage0', f'{tgt}_s{split}.npz')))
        O = dict(np.load(os.path.join(OUT, 'oracle', f'{tgt}_s{split}.npz')))
        pp = G.pool_positions(S, split, 0)
        y = D.y[S['allT'][pp]]
        Gm = O['G'].astype(np.float64)
        fold = np.zeros(len(y), int)
        for f, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(np.zeros((len(y), 1)))):
            fold[te] = f
        cnt = np.bincount(y, minlength=10)
        tot_var = Gm.var(0).sum()
        for c in range(10):
            col = Gm[:, c]
            # 按折的列内标准差（=0 表示该折该列是常数填充）
            within = np.array([col[fold == f].std() for f in range(5)])
            n_const_folds = int((within < 1e-9).sum())
            # 各折均值的方差占该列方差的比例（折效应）
            fm = np.array([col[fold == f].mean() for f in range(5)])
            between = np.var(fm[fold]) / max(col.var(), 1e-12)
            rows.append(dict(tgt=tgt, split=split, cls=c, n_pool=int(cnt[c]), col_std=col.std(),
                             var_share=col.var() / tot_var, n_const_folds=n_const_folds, fold_var_share=between))
df = pd.DataFrame(rows)
df.to_csv('/Users/perng_mini/AlphaCrops_local/verify/G_audit/oracle/a2_lda_columns.csv', index=False)
bad = df[df.n_const_folds > 0]
print('有常数填充折的列：', len(bad), '/', len(df))
print(bad.round(3).to_string(index=False))
print('\n按单元：常数填充列的方差份额合计')
print(bad.groupby(['tgt', 'split']).var_share.sum().round(3).to_string())
print('\n所有列 var_share 最大值（按单元）')
print(df.loc[df.groupby(['tgt', 'split']).var_share.idxmax(), ['tgt', 'split', 'cls', 'n_pool', 'col_std', 'var_share', 'n_const_folds', 'fold_var_share']].round(3).to_string(index=False))
