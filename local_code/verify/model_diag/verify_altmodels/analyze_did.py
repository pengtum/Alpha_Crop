"""汇总 did_lr_rows.csv（s1r0、s0r1、s1r1）与 altmodels 的 s0r0（part2_rows.csv 的 LR + lgb_e1），
国家层面 AE−S2（方法 × 分类器特征平均）：LR、LGB、差中差；逐单元与合并（先在国家内对单元平均）。"""
import numpy as np, pandas as pd
from scipy import stats
V = '/Users/perng_mini/AlphaCrops_local/verify/model_diag/verify_altmodels/'; A = '/Users/perng_mini/AlphaCrops_local/verify/model_diag/altmodels/'
X = pd.read_csv(V + 'did_lr_rows.csv')
p2 = pd.read_csv(A + 'part2_rows.csv'); p2 = p2[p2.model == 'lr']
e1 = pd.read_csv(A + 'lgb_e1_k200_s0r0.csv')
s0 = p2.merge(e1, on=['target', 'cond', 'method', 'clf'], suffixes=('_lr', '_lgb'))
s0 = s0.rename(columns={'ba_ge30_lr': 'ba_lr', 'oa_lr': 'oa_lr', 'ba_ge30_lgb': 'ba_lgb', 'oa_lgb': 'oa_lgb'}).assign(split=0, rep=0)
X = pd.concat([X, s0[X.columns]])
def tci(v):
    v = np.asarray(v, float); n = len(v); m = v.mean(); h = stats.t.ppf(.975, n-1) * v.std(ddof=1) / np.sqrt(n)
    return f'{m:+.3f} [{m-h:+.3f}, {m+h:+.3f}] p={stats.ttest_1samp(v, 0).pvalue:.3f} {int((v>0).sum())}/{n}'
def ae_s2(df, col):
    p = df.pivot_table(index=['target', 'split', 'rep', 'method', 'clf'], columns='cond', values=col)
    return (p['AE'] - p['S2']).reset_index(name='d')
out = {}
for col in ('ba_lr', 'ba_lgb', 'oa_lr', 'oa_lgb'):
    out[col] = ae_s2(X, col).groupby(['target', 'split', 'rep']).d.mean()
U = pd.DataFrame(out).reset_index()
U['did_ba'] = U.ba_lr - U.ba_lgb
print('n rows per unit:', X.groupby(['split', 'rep']).size().to_dict())
for (sp, rp), g in U.groupby(['split', 'rep']):
    print(f's{sp}r{rp}: LR AE−S2 {tci(g.ba_lr)} | LGB {tci(g.ba_lgb)} | DiD {tci(g.did_ba)} | r={np.corrcoef(g.ba_lr, g.ba_lgb)[0,1]:+.2f}')
C = U.groupby('target').mean(numeric_only=True)
print(f'合并 4 单元（国家内平均）: LR {tci(C.ba_lr)} | LGB {tci(C.ba_lgb)} | DiD {tci(C.did_ba)}')
C3 = U[~((U.split == 0) & (U.rep == 0))].groupby('target').mean(numeric_only=True)
print(f'只用 3 个新单元: LR {tci(C3.ba_lr)} | LGB {tci(C3.ba_lgb)} | DiD {tci(C3.did_ba)}')
print(f'合并 OA: LR {tci(C.oa_lr)} | LGB {tci(C.oa_lgb)}')
for clf in ('ae', 's2'):
    o = {}
    for col in ('ba_lr', 'ba_lgb'):
        o[col] = ae_s2(X[X.clf == clf], col).groupby('target').d.mean()
    o = pd.DataFrame(o); print(f'  clf={clf}: LR {tci(o.ba_lr)} | LGB {tci(o.ba_lgb)} | DiD {tci(o.ba_lr - o.ba_lgb)}')
print(C[['ba_lr', 'ba_lgb', 'did_ba']].round(2).to_string())
print('LR 与 LGB 的平均 ba:', X.ba_lr.mean().round(2), X.ba_lgb.mean().round(2))
