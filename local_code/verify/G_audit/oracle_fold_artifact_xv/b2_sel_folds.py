"""独立核实：ORACLE 选样在 5 个交叉拟合折上的分布（折号与标签无关，正常应近似多项分布均匀）。
对比同单元同方法的 AE、S2 选样。卡方检验（df=4）。只读。"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import chisquare
from sklearn.model_selection import KFold
sys.path.insert(0, os.getcwd())
import guidance_G as G

OUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
W = '/Users/perng_mini/AlphaCrops_local/verify/G_audit/oracle_fold_artifact_xv'
rows = []
for tgt in G.COUNTRIES_G:
    for split in (0, 1):
        S = dict(np.load(os.path.join(OUT, 'stage0', f'{tgt}_s{split}.npz')))
        pp = G.pool_positions(S, split, 0)
        fold = np.zeros(len(pp), int)
        for f, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(np.zeros((len(pp), 1)))):
            fold[te] = f
        for cond in ('ORACLE', 'AE', 'S2'):
            for m in ('lpm2_nogeo', 'typiclust', 'margin'):
                for k in (50, 200):
                    p = os.path.join(OUT, 'sel', f'{tgt}_s{split}_r0_{cond}_{m}_k{k}.npz')
                    if not os.path.exists(p):
                        continue
                    sp = np.load(p)['sel_pos']
                    c = np.bincount(fold[sp], minlength=5)
                    rows.append(dict(tgt=tgt, split=split, cond=cond, method=m, k=k, counts=c.tolist(),
                                     maxshare=c.max() / k, p_chi=chisquare(c).pvalue))
R = pd.DataFrame(rows)
R.to_csv(f'{W}/b2_sel_folds.csv', index=False)
pd.set_option('display.width', 250)
print(R[R.cond == 'ORACLE'].round(4).to_string(index=False))
print('\n各条件 × 方法：p_chi<0.001 的单元数 / 总数')
print(R.assign(sig=R.p_chi < 1e-3).groupby(['cond', 'method']).sig.agg(['sum', 'count']).to_string())
print('\n受影响国家（LV/PT/RO/LT/BE）的 AE/S2 对照：')
print(R[(R.cond != 'ORACLE') & R.tgt.isin(['LV', 'PT', 'RO']) & (R.method != 'margin')].round(4).to_string(index=False))
