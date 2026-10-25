import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion')
import analyze_G as A
from sklearn.isotonic import IsotonicRegression
KS=np.array([0,50,100,200,400.])
C=pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/G_audit/labeleq/srs_curve_raw.csv',index_col=[0,1]); C.columns=C.columns.astype(int)
D={'+0.3':0.3,'U95':0.1364,'θ_E1':-0.0537,'L95':-0.2437,'−0.3':-0.3}
cs=C.index.get_level_values(0).unique(); rng=np.random.default_rng(0)
Cc=C.groupby(level=0).mean()  # 国家层面（两分类器平均）曲线
def leqs(f):
    fp=IsotonicRegression().fit_transform(KS,f); return {(n,k):A.label_equiv(KS,fp,k,d) for n,d in D.items() for k in (50,200)}
pt=leqs(Cc.mean().to_numpy())
B=[leqs(Cc.loc[rng.choice(cs,len(cs))].mean().to_numpy()) for _ in range(4000)]
B=pd.DataFrame(B)
print('国家平均曲线（两分类器平均后 11 国均值）:',Cc.mean().round(2).to_list())
for key in pt:
    v=B[key].to_numpy(); fin=v[np.isfinite(v)]
    print(f'{key[0]:5s} k={key[1]:3d}: 点估计 {pt[key]:7.1f}；国家 bootstrap 2.5/50/97.5% {np.quantile(v,.025):7.1f} {np.quantile(v,.5):7.1f} {np.quantile(v,.975):7.1f}，超范围 {100*np.mean(~np.isfinite(v)):.1f}%')
# 中位数曲线
Cm=Cc.median(); print('国家中位数曲线',Cm.round(2).to_list(), {k:round(v,1) for k,v in leqs(Cm.to_numpy()).items()})
