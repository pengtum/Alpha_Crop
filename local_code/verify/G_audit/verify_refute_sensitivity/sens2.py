import glob, numpy as np, pandas as pd, sys
sys.path.insert(0,'.')
import guidance_G as G
import analyze_G as A
from scipy import stats
R=A.load_rows()
# 标签当量：+0.3 在 k=200 为 inf 的比例
rows=[]
for t in sorted(R.target.unique()):
    for clf in ('s2','ae'):
        ks,f=A.srs_curve(R,t,clf)
        for k in (50,200):
            for name,dv in (('+0.3',0.3),('-0.3',-0.3),('hi90',0.101),('lo90',-0.208)):
                rows.append(dict(t=t,clf=clf,k=k,w=name,lab=A.label_equiv(ks,f,k,dv)))
L=pd.DataFrame(rows)
print(L.pivot_table(index=['t','clf'],columns=['k','w'],values='lab').round(0).to_string())
print('+0.3@k200 inf 比例', (L[(L.w=='+0.3')&(L.k==200)].lab==np.inf).mean())
print('90%CI 标签当量中位数', L.groupby(['w','k']).lab.median().unstack('k'))
# 按 SRS 敏感度分组的 θ_E1
vE1=A.country_values(A.deltas(R,'AE','S2',G.E1))
S=R[R.method=='srs'].groupby(['target','k']).ba_ge30.mean().unstack('k')
slope=(S[400]-S[50])
print(pd.DataFrame({'dE1':vE1,'srs50_400':slope}).sort_values('srs50_400').round(3).to_string())
resp=slope[slope>=1].index
def ts(v):
    v=np.asarray(v);n=len(v);m=v.mean();se=v.std(ddof=1)/np.sqrt(n);q=stats.t.ppf(.975,n-1)
    return f'{m:+.3f} [{m-q*se:+.3f},{m+q*se:+.3f}] n={n}'
print('SRS 50→400 ≥1 pp 的国家', list(resp), 'θ_E1:', ts(vE1[resp]))
print('其余', ts(vE1.drop(resp)))
print('spearman(|ΔE1|, slope)', stats.spearmanr(vE1.abs(),slope.reindex(vE1.index)))
