"""质疑者视角 s2：RO 绝对水平、F5 的 SRS 依赖、predstrat 的 F1、检验 2 去 SE。"""
import os, sys
os.environ['OMP_NUM_THREADS']='4'
sys.path.insert(0,'/Users/perng_mini/AlphaCrops_local/verify/strategy/benchmark')
from common_bm import *
pd.set_option('display.width',250)
R=load()
t=R[R.clf.isin(['s2','ae'])&R.k.isin([0,50,200])&(R.rep==0)]
print('== RO / PT 绝对水平（重复 0，两种分类器、两种指导平均）==')
for c in ('RO','PT'):
    g=t[(t.target==c)&(t.method.isin(['srs','foreign_only','kcenter','margin','gist_margin','predstrat_srs','typiclust']))&(t.cond.isin(['none','AE','S2']))]
    print(c); print(g.groupby(['method','k'])[['oa','oa_P2','ba_ge30','ba_P2']].mean().round(2).to_string())
print('\n== F5 的 SRS 依赖：SRS 自身 (k200−k50) 与 E1 方法自身 (k200−k50)，按重复 ==')
for metric in ('ba_ge30',):
    s=R[(R.method=='srs')&R.clf.isin(['s2','ae'])].groupby(['target','rep','k'])[metric].mean().unstack('k')
    sg=(s[200]-s[50]).unstack('rep')
    e=R[R.cond.isin(['AE','S2'])&R.method.isin(E1)&R.clf.isin(['s2','ae'])].groupby(['target','rep','k'])[metric].mean().unstack('k')
    eg=(e[200]-e[50]).unstack('rep')
    X=pd.concat({'SRS_r0':sg[0],'SRS_r1':sg[1],'E1_r0':eg[0],'E1_r1':eg[1]},axis=1)
    X['E1-SRS_r0']=X.E1_r0-X.SRS_r0; X['E1-SRS_r1']=X.E1_r1-X.SRS_r1
    print(X.round(2).to_string()); print('均值', X.mean().round(3).to_dict())
    for col in ('E1-SRS_r0','E1-SRS_r1','SRS_r0','SRS_r1'):
        s_=ts(X[col]); print(f'  {col}: {s_["mean"]:+.3f} [{s_["lo95"]:+.3f}, {s_["hi95"]:+.3f}] {s_["n_pos"]}/11')
    d=X['SRS_r1']-X['SRS_r0']; s_=ts(d); print(f'  SRS 重复间 (k200−k50) 之差 r1−r0: {s_["mean"]:+.3f} [{s_["lo95"]:+.3f}, {s_["hi95"]:+.3f}]')
print('\n== predstrat_srs 的 macro-F1 与逐类召回（相对 SRS，重复 0，合并）==')
for metric in ('f1_ge30','ba_ge30','oa','ba_P2','oa_P2','ba_far','oa_far'):
    d=pd.concat([vs_srs(R,c,['predstrat_srs'],metric) for c in ('AE','S2')]); d=d[d.rep==0]
    s_=ts(cv(d)); print(f'  {metric:8s}: {s_["mean"]:+.2f} [{s_["lo95"]:+.2f}, {s_["hi95"]:+.2f}] {s_["n_pos"]}/11')
print('\n== 检验 2：uncertainty − random，去 SE ==')
P=pd.read_csv('/Users/perng_mini/AlphaCrops_local/results/pilot_selection.csv')
print(P.columns.tolist()[:40]); print(P.head(2).to_string())
