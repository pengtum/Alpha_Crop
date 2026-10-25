import sys, glob, os; sys.path.insert(0,'.')
from load import ci, fmt
from wls import slope
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy import stats
pd.set_option('display.width',250)
G='/Users/perng_mini/AlphaCrops_local/results/guidance_G'; Cd='/Users/perng_mini/AlphaCrops_local/data/cache_guidance_G'
lon=np.load(f'{Cd}/lon.npy'); lat=np.load(f'{Cd}/lat.npy')
def km_xy(lo,la,lat0): return np.c_[lo*111.32*np.cos(np.radians(lat0)), la*110.57]
# SRS 的 nn_km（与 mediators_G 同定义：选中样本之间最近邻距离的中位数）
rows=[]
for f0 in sorted(glob.glob(f'{G}/stage0/*.npz')):
    tgt,sp=os.path.basename(f0)[:-4].split('_s'); S=np.load(f0); lat0=float(S['lat0'])
    for rep in (0,1):
        for k in (50,200):
            z=np.load(f'{G}/sel/{tgt}_s{sp}_r{rep}_none_srs_k{k}.npz'); s=z['sel']
            xy=km_xy(lon[s],lat[s],lat0); d,_=cKDTree(xy).query(xy,k=2)
            rows.append(dict(target=tgt,split=int(sp),rep=rep,k=k,nn_srs=float(np.median(d[:,1]))))
N=pd.DataFrame(rows)
M=pd.read_csv(f'{G}/mediators_G.csv').merge(N,on=['target','split','rep','k'])
M['d_nn']=M.nn_km-M.nn_srs
print('nn_km 相对 SRS（km）：'); print(M.groupby(['method','k','cond']).d_nn.mean().unstack('cond').round(2).to_string())
print('SRS nn_km 均值：',N.groupby('k').nn_srs.mean().round(2).to_dict())
A=pd.read_pickle('A.pkl'); A=A[A.cond.isin(['AE','S2'])].reset_index(drop=True); A['cell']=np.arange(len(A))
L=pd.read_pickle('L.pkl'); b=slope(L)
A['pred_comp']=b*L.groupby('cell').dlog.mean().reindex(A.cell).to_numpy(); A['res']=A.d_ba_ge30-A.pred_comp
A=A.merge(M[['target','split','rep','method','k','cond','d_nn']],on=['target','split','rep','method','k','cond'],how='left')
T=A.groupby(['method','cond'])[['d_any5','d_sc5','d_spat','d_nn']].mean()
print('\n空间中介（相对 SRS）：'); print(T.round(3).to_string())
print('\nd_any5 / d_sc5 的绝对值上界（方法×条件×k 均值）：',A.groupby(['method','cond','k'])[['d_any5','d_sc5']].mean().abs().max().round(4).to_dict())
# geo_cost：0–5 km 同类局部效应 +2.90 pp；隐含的方法间空间贡献上界 = |Δsc5| × 2.90
print('按 0–5 km 局部效应 +2.90 pp 换算：Δsc5 最大绝对值对应',round(A.groupby(['method','cond','k']).d_sc5.mean().abs().max()*2.90,3),'pp')
# 单元层：残差与空间中介（国家内去均值）
for x in ['d_spat','d_any5','d_nn']:
    d=A[[x,'res','d_ba_ge30','target']].dropna().copy()
    for v in (x,'res','d_ba_ge30'): d[v]=d[v]-d.groupby('target')[v].transform('mean')
    print(f'单元层 {x}: 与残差 r={np.corrcoef(d[x],d.res)[0,1]:+.3f}；与 ΔBA r={np.corrcoef(d[x],d.d_ba_ge30)[0,1]:+.3f}；n={len(d)}')
# d_spat 与构成的共线：方法×条件×k 层
P=A.groupby(['method','cond','k'])[['d_spat','pred_comp','d_ba_ge30']].mean()
print('方法层 d_spat 与构成预测的相关', round(P.d_spat.corr(P.pred_comp),2))
