import sys, os, numpy as np, pandas as pd
sys.path.insert(0,'/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion')
os.chdir('/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion')
import analyze_G as A
pd.set_option('display.width',220); pd.set_option('display.max_rows',500)
R=A.load_rows()
E1=A.E1
sE1=A.t_summary(A.country_values(A.deltas(R,'AE','S2',E1)))
print('θ_E1',{k:round(v,4) if isinstance(v,float) else v for k,v in sE1.items()})
D={'+0.3':0.3,'−0.3':-0.3,'θ_E1':sE1['mean'],'L95':sE1['lo95'],'U95':sE1['hi95']}
# 复现报告
print(A.label_equiv_table(R,D))
# 完整分布
rows=[]
for t in sorted(R.target.unique()):
    for clf in ('s2','ae'):
        ks,f=A.srs_curve(R,t,clf)
        for k in (50,200):
            for n,dv in D.items():
                rows.append(dict(target=t,clf=clf,k=k,what=n,delta=dv,labels=A.label_equiv(ks,f,k,dv),
                                 at0=A.label_equiv(ks,f,k,0.0)))
L=pd.DataFrame(rows)
L.to_csv('/Users/perng_mini/AlphaCrops_local/verify/G_audit/labeleq/label_equiv_all.csv',index=False)
def summ(v):
    v=np.asarray(v,float)
    return pd.Series(dict(n=len(v),median=np.median(v),Q1=np.quantile(v,.25),Q3=np.quantile(v,.75),
        pos_inf=int(np.isposinf(v).sum()),neg_inf=int(np.isneginf(v).sum()),
        out_of_range_pct=100*np.mean(~np.isfinite(v)),
        finite_median=np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else np.nan))
S=L.groupby(['what','k']).labels.apply(summ).unstack()
print(S.round(1).to_string())
print('\n Δ=0 时的标签当量（应为 0；平台上不为 0）:')
Z=L[L.what=='+0.3'][['target','clf','k','at0']]
print(Z[Z.at0!=0].to_string())
W=L.pivot_table(index=['target','clf'],columns=['k','what'],values='labels')
print(W.round(1).to_string())
