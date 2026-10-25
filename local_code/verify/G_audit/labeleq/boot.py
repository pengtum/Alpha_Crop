import sys, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion')
import analyze_G as A
from sklearn.isotonic import IsotonicRegression
S=pd.read_pickle('/Users/perng_mini/AlphaCrops_local/verify/G_audit/labeleq/S.pkl')
KS=np.array([0,50,100,200,400.])
P=S.pivot_table(index=['target','clf','split','rep'],columns='k',values='ba_ge30')
rng=np.random.default_rng(1)
out=[]
for (t,c),g in P.groupby(level=[0,1]):
    M=g.to_numpy()
    for k in (50,200):
        v=[]
        for b in range(1000):
            f=M[rng.integers(0,4,4)].mean(0)
            v.append(A.label_equiv(KS,IsotonicRegression().fit_transform(KS,f),k,0.3))
        v=np.array(v); fin=v[np.isfinite(v)]
        out.append(dict(target=t,clf=c,k=k,p_inf=np.mean(~np.isfinite(v)),q10=np.quantile(fin,.1) if len(fin) else np.nan,
                        q50=np.median(fin) if len(fin) else np.nan,q90=np.quantile(fin,.9) if len(fin) else np.nan))
O=pd.DataFrame(out)
pd.set_option('display.width',200)
print(O.pivot_table(index=['target','clf'],columns='k',values=['p_inf','q10','q50','q90']).round(2))
print('\n单元 bootstrap 下 +0.3 的 inf 概率：k=50 中位',O[O.k==50].p_inf.median(),' k=200 中位',O[O.k==200].p_inf.median())
print('p_inf 介于 0.2–0.8 的曲线数（结论不稳定）: k=50',((O.k==50)&O.p_inf.between(.2,.8)).sum(),' k=200',((O.k==200)&O.p_inf.between(.2,.8)).sum())
# 跨曲线中位数的 bootstrap（单元层面重抽）
med=[]
for b in range(300):
    vals={50:[],200:[]}
    for (t,c),g in P.groupby(level=[0,1]):
        M=g.to_numpy(); f=IsotonicRegression().fit_transform(KS,M[rng.integers(0,4,4)].mean(0))
        for k in (50,200): vals[k].append(A.label_equiv(KS,f,k,0.3))
    med.append([np.median(vals[50]),np.median(vals[200]),np.mean(~np.isfinite(vals[200]))])
med=np.array(med)
f50=med[:,0]; print('\n+0.3 @k50 跨 22 曲线中位数的单元 bootstrap：inf 比例',np.mean(~np.isfinite(f50)),' 有限值 10/50/90% 分位',np.quantile(f50[np.isfinite(f50)],[.1,.5,.9]).round(1))
print('+0.3 @k200 跨曲线中位数为 inf 的比例',np.mean(np.isinf(med[:,1])),'；超范围曲线比例的 10/50/90%',np.quantile(med[:,2],[.1,.5,.9]).round(2))
