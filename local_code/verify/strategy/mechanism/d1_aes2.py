import sys; sys.path.insert(0,'.')
from load import ci, fmt
from wls import slope
import numpy as np, pandas as pd
from scipy import stats
pd.set_option('display.width',250)
A=pd.read_pickle('A.pkl'); L=pd.read_pickle('L.pkl')
AA=A[A.cond.isin(['AE','S2'])].reset_index(drop=True)
# 1) 剂量斜率：分类器 ae vs s2，逐国配对
bs=L.groupby(['target','clf']).apply(lambda d: slope(d)).unstack('clf')
print('逐国剂量斜率（分类器）：'); print(bs.round(2).T.to_string()); print('ae − s2：',fmt(ci(bs.ae-bs.s2)))
L9=L[~L.target.isin(['RO','PT'])]
bs9=L9.groupby(['target','clf']).apply(lambda d: slope(d)).unstack('clf'); print('9 国 ae − s2：',fmt(ci(bs9.ae-bs9.s2)))
bg=L.groupby(['target','cond']).apply(lambda d: slope(d)).unstack('cond'); print('指导 AE − S2 的剂量斜率：',fmt(ci(bg.AE-bg.S2)))
# 2) predstrat 的构成：AE vs S2 指导（同单元同 k）
p=AA[AA.method=='predstrat_srs'].pivot_table(index=['target','split','rep','clf','k'],columns='cond',values=['d_LD','d_comp_L1','d_ba_ge30','d_maj_share'])
for v in ['d_LD','d_maj_share','d_ba_ge30']:
    dd=(p[(v,'AE')]-p[(v,'S2')]).groupby(['clf','k','target']).mean().groupby('target').mean()
    print(f'predstrat AE − S2 指导 {v}: {fmt(ci(dd),3)}')
# 3) predstrat Δ 按分类器
for clf in ('ae','s2'):
    for cond in ('AE','S2'):
        a=AA[(AA.method=='predstrat_srs')&(AA.clf==clf)&(AA.cond==cond)]
        print(f'predstrat 指导 {cond} 分类器 {clf}: ΔBA {fmt(ci(a.groupby(["k","target"]).d_ba_ge30.mean().groupby("target").mean()))}')
# 4) margin：外国模型质量
Q=pd.read_csv('/Users/perng_mini/AlphaCrops_local/results/guidance_G/guidance_quality_G.csv')
Q['cond']=Q.feat.str.upper()
mg=AA[AA.method.isin(['margin','gist_margin','badge_c','predstrat_srs'])].groupby(['method','target','split','cond'])[['d_ba_ge30','d_oa','d_comp_L1']].mean().reset_index()
mg=mg.merge(Q,on=['target','split','cond'])
print('\n预测类方法：单元层（22 单元 × 2 指导）Δ 与外国模型质量的 Spearman 相关（国家内去均值后）')
for m in ['margin','gist_margin','badge_c','predstrat_srs']:
    d=mg[mg.method==m].copy()
    for v in ['d_ba_ge30','d_oa','nll_pool','auroc_margin','prior_L1_argmax','ece_oof','knn10_ba']:
        d[v+'_w']=d[v]-d.groupby('target')[v].transform('mean')
    out=[]
    for q in ['nll_pool','auroc_margin','prior_L1_argmax','ece_oof']:
        r=stats.spearmanr(d[q+'_w'],d['d_ba_ge30_w']); out.append(f'{q} ρ={r.statistic:+.2f}(p={r.pvalue:.2f})')
    print(f'  {m:14s} ΔBA vs ', '；'.join(out))
# 配对 AE − S2 margin 与质量差
pm=mg[mg.method=='margin'].pivot_table(index=['target','split'],columns='cond',values=['d_ba_ge30','d_oa','nll_pool','auroc_margin','prior_L1_argmax'])
dd=pd.DataFrame({v:pm[(v,'AE')]-pm[(v,'S2')] for v in ['d_ba_ge30','d_oa','nll_pool','auroc_margin','prior_L1_argmax']})
print('\nmargin AE − S2（22 单元）：ΔBA 差', fmt(ci(dd.d_ba_ge30.groupby('target').mean())), '；ΔOA 差', fmt(ci(dd.d_oa.groupby('target').mean())))
for q in ['nll_pool','auroc_margin','prior_L1_argmax']:
    r=stats.spearmanr(dd[q],dd.d_ba_ge30); print(f'   质量差 {q} 与 ΔBA 差：ρ={r.statistic:+.2f} p={r.pvalue:.2f}')
print('质量差均值：',dd[['nll_pool','auroc_margin','prior_L1_argmax']].mean().round(3).to_dict())
# 9 国 margin AE vs S2
for cond in ('AE','S2'):
    a=AA[(AA.method=='margin')&(AA.cond==cond)&~AA.target.isin(['RO','PT'])]
    print(f'margin {cond} 9 国 ΔBA {fmt(ci(a.groupby(["clf","k","target"]).d_ba_ge30.mean().groupby("target").mean()))}')
