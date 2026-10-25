import sys, os, numpy as np, pandas as pd
sys.path.insert(0,'/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion')
os.chdir('/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion')
import analyze_G as A
from sklearn.isotonic import IsotonicRegression
pd.set_option('display.width',220); pd.set_option('display.max_rows',500)
R=A.load_rows()
KS=np.array([0,50,100,200,400.])
C=pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/G_audit/labeleq/srs_curve_raw.csv',index_col=[0,1]); C.columns=C.columns.astype(int)
rows=[]
for (t,clf),r in C.iterrows():
    f=r[[0,50,100,200,400]].to_numpy()
    ks,fp=A.srs_curve(R,t,clf)
    assert np.allclose(ks,KS)
    pav_changed=int(np.sum(np.abs(fp-f)>1e-9))
    # plateaus: equal consecutive PAV values
    plat=[int(ks[i]) for i in range(1,5) if abs(fp[i]-fp[i-1])<1e-9]
    viol=[f"{int(KS[i-1])}->{int(KS[i])}" for i in range(1,5) if f[i]<f[i-1]]
    rows.append(dict(target=t,clf=clf,f0=f[0],f50=f[1],f100=f[2],f200=f[3],f400=f[4],
        jump0_50=f[1]-f[0],g50_200=f[3]-f[1],g200_400=f[4]-f[3],g50_400=f[4]-f[1],
        nviol=len(viol),viol=','.join(viol),pav_pts_changed=pav_changed,
        pav=' '.join(f'{x:.2f}' for x in fp),
        top_plateau_incl_200=abs(fp[3]-fp[4])<1e-9, bottom_plateau_incl_50=abs(fp[1]-fp[0])<1e-9,
        pav_g200_400=fp[4]-fp[3], pav_g0_50=fp[1]-fp[0]))
T=pd.DataFrame(rows)
T.to_csv('/Users/perng_mini/AlphaCrops_local/verify/G_audit/labeleq/curve_diagnostics.csv',index=False)
print(T.drop(columns=['f0','f50','f100','f200','f400']).round(3).to_string())
print('\n单调违反的曲线数（原始）:',(T.nviol>0).sum(),'/',len(T),' 违反段总数',T.nviol.sum(),'/',4*len(T))
print('PAV 改动过的曲线数:',(T.pav_pts_changed>0).sum())
print('k=200 在顶部平台（PAV f(200)=f(400)）:',T.top_plateau_incl_200.sum(),'/',len(T))
print('k=50 在底部平台（PAV f(0)=f(50)）:',T.bottom_plateau_incl_50.sum(),'/',len(T))
print('PAV g200->400 <0.3:',(T.pav_g200_400<0.3).sum(),'  PAV g0->50 <0.3:',(T.pav_g0_50<0.3).sum())
for c in ['jump0_50','g50_200','g200_400','g50_400']:
    v=T[c]; print(f'{c}: mean {v.mean():+.3f}  median {v.median():+.3f}  Q1 {v.quantile(.25):+.3f} Q3 {v.quantile(.75):+.3f}  min {v.min():+.3f} max {v.max():+.3f}  n<0 {int((v<0).sum())}')
print('\n按分类器')
print(T.groupby('clf')[['jump0_50','g50_200','g200_400','g50_400']].agg(['mean','median']).round(3))
print('\n去 PT、RO 后')
T2=T[~T.target.isin(['PT','RO'])]
print(T2[['jump0_50','g50_200','g200_400','g50_400']].agg(['mean','median']).round(3))
# 国家层面（两分类器平均）
Tc=T.groupby('target')[['jump0_50','g50_200','g200_400','g50_400']].mean()
print('\n国家层面（两个分类器平均）'); print(Tc.round(3)); print(Tc.agg(['mean','median']).round(3))
