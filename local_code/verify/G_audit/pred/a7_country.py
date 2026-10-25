import sys, os
sys.path.insert(0, "/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion")
os.chdir(sys.path[0])
import numpy as np, pandas as pd
from scipy import stats
import analyze_G as A
from guidance_G import PRED, E1, HYB
OUT = "/Users/perng_mini/AlphaCrops_local/results/guidance_G"
R = A.load_rows(OUT)
d = A.deltas(R,'AE','S2',PRED,reps=[0])
t = d.pivot_table(index='target', columns=['method','clf','k'], values='d', aggfunc='mean').round(2)
print(t.to_string())
print('\nby split:'); print(d.pivot_table(index='target', columns=['split'], values='d').round(3).to_string())
vp = A.country_values(d); vg = A.country_values(A.deltas(R,'AE','S2',E1)); vh = A.country_values(A.deltas(R,'AE','S2',HYB,reps=[0]))
vg0 = A.country_values(A.deltas(R,'AE','S2',E1,reps=[0]))
print('\ncountry values: pred, geom(all reps), geom(rep0), hyb'); print(pd.DataFrame({'pred':vp,'geom':vg,'geom_r0':vg0,'hyb':vh}).round(3).to_string())
print('spearman pred~geom', stats.spearmanr(vp, vg), 'pred~hyb', stats.spearmanr(vp, vh))
# pred minus geom paired on rep0 (does pred differ from geom?)
s = A.t_summary(vp - vg0); print('pred - geom(rep0) country t:', A.fmt(s))
s = A.t_summary(vp - vh); print('pred - hyb country t:', A.fmt(s))
# unit-level t over 22 units (ignoring country clustering) for reference
u = d.groupby(['target','split']).d.mean(); s = A.t_summary(u); print('unit-level (22) t:', A.fmt(s))
# drop two largest countries (RO, SE)
s = A.t_summary(vp.drop(['RO','SE'])); print('drop RO+SE:', A.fmt(s))
# trimmed mean / median
print('median country value', vp.median(), '20% trimmed mean', stats.trim_mean(vp, 0.2))
# pred path with oa  and ba jointly: fraction of cells where ba up and oa down
do = A.deltas(R,'AE','S2',PRED,'oa',reps=[0]).set_index(A.KEY).d
db = d.set_index(A.KEY).d
print('cells: ba>0 & oa<0', ((db>0)&(do<0)).mean().round(3), ' ba>0&oa>0', ((db>0)&(do>0)).mean().round(3), 'corr', np.corrcoef(db, do.reindex(db.index))[0,1].round(3))
