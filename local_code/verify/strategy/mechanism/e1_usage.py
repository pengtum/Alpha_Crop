"""(e) 同一批选样在不同标签用法下的"方法 − SRS"（codesign/diag3_sel_rows.csv；C 级，G s0/s1 发现集）。"""
import sys; sys.path.insert(0,'.')
from load import ci, fmt
import numpy as np, pandas as pd
pd.set_option('display.width',250)
d=pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/codesign/diag3_sel_rows.csv')
d=d.rename(columns={'tgt':'target'})
key=['unit','feat','k','learner']
s=d[d.sel=='none_srs'].set_index(key)[['ba','oa','comp_L1']]
x=d[d.sel!='none_srs'].copy()
kk=pd.MultiIndex.from_frame(x[key])
for m in ['ba','oa','comp_L1']: x['d_'+m]=x[m].to_numpy()-s[m].reindex(kk).to_numpy()
x=x.dropna(subset=['d_ba'])
SELS=['AE_lpm2_nogeo','S2_lpm2_nogeo','AE_kcenter','S2_kcenter','AE_typiclust','S2_typiclust','AE_kherding','S2_kherding',
      'AE_margin','S2_margin','AE_predstrat_srs','S2_predstrat_srs','AE_gist_margin','S2_gist_margin','AE_badge_c','S2_badge_c',
      'ORACLE_margin','ORACLE_typiclust','ORACLE2_typiclust','QORACLE_sqrt','QORACLE_equal','QORACLE_prop']
rows=[]
for sel in SELS:
    r=dict(sel=sel)
    for L in ['lgb','bias','feat03']:
        a=x[(x.sel==sel)&(x.learner==L)]
        if not len(a): continue
        v=a.groupby(['feat','k','target']).d_ba.mean().groupby('target').mean()
        o=a.groupby(['feat','k','target']).d_oa.mean().groupby('target').mean()
        r[f'{L}_dBA']=fmt(ci(v)); r[f'{L}_dOA']=round(o.mean(),2)
    rows.append(r)
T=pd.DataFrame(rows); print(T.to_string(index=False)); T.to_csv('e1_usage_vs_srs.csv',index=False)
# SRS 本身的标签价值（相对 k=0 外国模型）：lgb 用 G 的 foreign_only；头部用法相对的是同一外国模型
# 方法间离散：各用法下，16 个 AE/S2 方法的国家均值 Δ 的 SD；以及单元层同格内方法间 SD
print()
for L in ['lgb','bias','feat03']:
    a=x[(x.learner==L)&x.sel.str.match(r'^(AE|S2)_')]
    mm=a.groupby(['sel','target']).d_ba.mean().groupby('sel').mean()
    within=a.groupby(['unit','feat','k']).d_ba.std().mean()
    print(f'{L:7s} 16 个 AE/S2 选样的均值 Δ 的方法间 SD = {mm.std():.2f}；范围 [{mm.min():+.2f}, {mm.max():+.2f}]；同格内方法间 SD 均值 = {within:.2f}')
# 用法 × 选样：头部用法下的 Δ 与 lgb 下的 Δ 的相关（方法×条件×k 层）
p=x[x.sel.isin(SELS[:16])].groupby(['sel','k','learner']).d_ba.mean().unstack('learner')
print('\n方法×k 层（32 点）Δ 的相关：', p.corr().round(2).to_string())
print(p.round(2).to_string())
