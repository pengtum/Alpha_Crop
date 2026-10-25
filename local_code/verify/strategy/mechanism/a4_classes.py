import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
pd.set_option('display.width',250)
CL=["grassland","maize","orchards","other_crops","potatoes","root_crops","strawberries","tree_wood_forest","vegetables","winter_cereals"]
A=pd.read_pickle('A.pkl'); A=A[A.cond.isin(['AE','S2'])]
# 每类的标签数变化（相对 SRS，k 归一：每 100 个标签），与该类召回变化
rows=[]
for (m,k),a in A.groupby(['method','k']):
    r=dict(method=m,k=k)
    for c in range(10):
        r[CL[c][:8]]=round(100*a[f'd_n{c}'].mean()/k,1)
    rows.append(r)
print('相对 SRS 的逐类标签份额变化（pp of k）'); print(pd.DataFrame(rows).to_string(index=False))
rows=[]
for (m,k),a in A.groupby(['method','k']):
    r=dict(method=m,k=k)
    for c in range(10):
        big=a[f'nT{c}']>=30
        r[CL[c][:8]]=round(100*a.loc[big,f'd_rec{c}'].mean(),2)
    rows.append(r)
print('\n相对 SRS 的逐类召回变化（pp，只算该类在测试集 ≥30 的单元）'); print(pd.DataFrame(rows).to_string(index=False))
# 池中各类占比（跨单元平均）
U=pd.read_csv('unit_table.csv'); print('\n候选池类别占比均值:',{CL[c][:8]:round(U[f'pi{c}'].mean(),3) for c in range(10)})
print('测试集≥30 的单元比例:',{CL[c][:8]:round((U[f'nT{c}']>=30).mean(),2) for c in range(10)})
