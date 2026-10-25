"""质疑者视角 s0：基础核查（只读 G 行）。"""
import os, sys
os.environ['OMP_NUM_THREADS']='4'
sys.path.insert(0,'/Users/perng_mini/AlphaCrops_local/verify/strategy/benchmark')
from common_bm import *
R=load()
t1=R[R.cond.isin(['AE','S2'])&R.method.isin(ALL8)&R.k.isin([50,200])&R.clf.isin(['s2','ae'])]
# 1) 两种分类器是否用同一选样
p=t1.pivot_table(index=['target','split','rep','cond','method','k'],columns='clf',values='sel_hash',aggfunc='first')
print('同一选样(clf s2 vs ae) 比例:', (p['s2']==p['ae']).mean(), len(p))
# 2) 每国 ba_ge30 计入的类数：从 rec 列不可直接得；用 foreign_only 行中非 NaN rec 的数目近似
fo=R[(R.method=='foreign_only')&(R.clf=='s2')]
rc=[f'rec{i}' for i in range(10)]
print(fo.groupby('target')[rc].apply(lambda g: g.notna().sum(axis=1).mean()))
# 3) SRS 单元数
print(R[R.method=='srs'].groupby(['k','clf']).size())
