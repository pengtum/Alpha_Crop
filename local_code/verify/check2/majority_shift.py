import numpy as np, pandas as pd
from scipy.stats import spearmanr
L='/Users/perng_mini/AlphaCrops_local'
classes=['grassland','maize','orchards','other_crops','potatoes','root_crops','strawberries','tree_wood_forest','vegetables','winter_cereals']
s=pd.read_csv('selection_class_counts.csv'); s=s[s.strategy!='_pool_']
s['grass_share']=s.grassland/s[classes].sum(axis=1)*100
# 多数类：各目标国候选池中最大的类（均为 grassland，PT 为 orchards）
pool=pd.read_csv('selection_class_counts.csv'); pool=pool[pool.strategy=='_pool_'].set_index('target')[classes]
maj=pool.idxmax(axis=1).to_dict(); print('pool 多数类', maj)
s['maj_share']=[r[maj[r.target]]/r[classes].sum()*100 for _,r in s.iterrows()]
sel=pd.read_csv(f'{L}/results/pilot_selection.csv'); bet=pd.read_csv(f'{L}/results/pilot_betti_parcel.csv'); bet=bet.assign(strategy='betti_'+bet.strategy)
res=pd.concat([sel,bet])[['target','strategy','budget','seed','clf_feat','ba_ge30']]
m=s.merge(res,on=['target','strategy','budget','seed','clf_feat'])
rnd=m[m.strategy=='random'].set_index(['target','budget','seed','clf_feat'])[['ba_ge30','grass_share','maj_share']]
o=m[m.strategy!='random'].join(rnd,on=['target','budget','seed','clf_feat'],rsuffix='_r')
o['dba']=(o.ba_ge30-o.ba_ge30_r)*100; o['dgrass']=o.grass_share-o.grass_share_r; o['dmaj']=o.maj_share-o.maj_share_r
print('全部非随机策略运行 n=',len(o))
print('Spearman(Δgrassland 占比, Δba_ge30):', spearmanr(o.dgrass,o.dba))
for t in ['PT','SE','SK','LV']:
    x=o[o.target==t]; print(t, 'grass', spearmanr(x.dgrass,x.dba).statistic.round(3), 'maj', spearmanr(x.dmaj,x.dba).statistic.round(3))
x=o[o.strategy!='uncertainty']; print('排除 uncertainty 后 Spearman(Δgrass,Δba):', spearmanr(x.dgrass,x.dba))
# 简单线性回归：Δba ~ Δgrass + uncertainty 指示变量（按目标国分别）
import numpy.linalg as la
for t in ['PT','SE','SK','LV','ALL']:
    x=o if t=='ALL' else o[o.target==t]
    A=np.c_[np.ones(len(x)), x.dgrass, (x.strategy=='uncertainty').astype(float)]
    b=la.lstsq(A,x.dba.to_numpy(),rcond=None)[0]
    A0=np.c_[np.ones(len(x)), (x.strategy=='uncertainty').astype(float)]
    b0=la.lstsq(A0,x.dba.to_numpy(),rcond=None)[0]
    print(f'{t}: 无协变量 uncertainty 系数 {b0[1]:+.2f}；加 Δgrass 后 uncertainty 系数 {b[2]:+.2f}，Δgrass 斜率 {b[1]:+.3f} pp/pp')
