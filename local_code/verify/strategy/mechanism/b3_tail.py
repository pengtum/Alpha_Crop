import sys; sys.path.insert(0,'.')
from load import ci, fmt
from wls import slope
import numpy as np, pandas as pd
pd.set_option('display.width',250)
A=pd.read_pickle('A.pkl'); L=pd.read_pickle('L.pkl')
AA=A[A.cond.isin(['AE','S2'])]
# R1 尾部：方法 − SRS < −0.5 的运行，按国家分布
for fam,ms in {'E1':['lpm2_nogeo','kcenter','typiclust','kherding'],'预测/混合':['margin','predstrat_srs','gist_margin','badge_c']}.items():
    a=AA[AA.method.isin(ms)]
    bad=a.d_ba_ge30<-0.5
    print(fam,'运行数',len(a),'<−0.5 比例',round(bad.mean(),3),'；其中来自 RO/PT 的份额',round(a[bad].target.isin(['RO','PT']).mean(),3),
          '；RO/PT 占全部运行的份额',round(a.target.isin(['RO','PT']).mean(),3),'；9 国内 <−0.5 比例',round(bad[~a.target.isin(['RO','PT'])].mean(),3),
          '；RO/PT 内',round(bad[a.target.isin(['RO','PT'])].mean(),3))
# 单元层 SD（同方法、条件、分类器、k）按国家
sd=AA.groupby(['target','method','cond','clf','k']).d_ba_ge30.std().groupby('target').mean()
print('\n同格单元间 SD（方法 − SRS，ba_ge30）按国家：',sd.round(2).to_dict())
sr=A[(A.method=='kcenter')]
# RO 单元细节
print('\nRO 单元细节（predstrat、margin、kcenter、lpm2，AE/S2，两分类器平均）：')
print(AA[AA.target=='RO'].groupby(['method','split','rep','k'])[['d_ba_ge30','d_oa','d_comp_L1','d_LD']].mean().round(2).unstack('k').to_string())
U=pd.read_csv('unit_table.csv'); u=U[U.target=='RO'].iloc[0]
print('RO 候选池占比', {c:round(u[f'pi{c}'],3) for c in range(10)}, 'nT', {c:int(u[f'nT{c}']) for c in range(10)})
