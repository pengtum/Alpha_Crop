import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt; plt.rcParams['font.sans-serif']=['Arial Unicode MS','Heiti TC','PingFang HK']; plt.rcParams['axes.unicode_minus']=False
from sklearn.isotonic import IsotonicRegression
S=pd.read_pickle('/Users/perng_mini/AlphaCrops_local/verify/G_audit/labeleq/S.pkl')
KS=np.array([0,50,100,200,400.])
m=S.groupby(['target','clf','k']).ba_ge30.mean(); se=S.groupby(['target','clf','k']).ba_ge30.std(ddof=1)/2
# k=0 只有 2 个独立值（两划分），SE = SD/sqrt(2)
se0=S[S.k==0].drop_duplicates(['target','clf','split']).groupby(['target','clf']).ba_ge30.std(ddof=1)/np.sqrt(2)
fig,axs=plt.subplots(3,4,figsize=(15,10)); axs=axs.ravel()
cs=sorted(S.target.unique()); col={'s2':'#1f77b4','ae':'#d62728'}
xpos=np.array([0,1,2,3,4])
for i,t in enumerate(cs):
    ax=axs[i]
    for clf in ('s2','ae'):
        y=m.loc[(t,clf)].reindex(KS.astype(int)).to_numpy(); e=se.loc[(t,clf)].reindex(KS.astype(int)).to_numpy().copy(); e[0]=se0.loc[(t,clf)]
        fp=IsotonicRegression().fit_transform(KS,y)
        ax.errorbar(xpos,y,yerr=e,fmt='o',color=col[clf],ms=4,capsize=2,alpha=.8,label=f'{clf} raw mean +- SE')
        ax.plot(xpos,fp,'-',color=col[clf],lw=1.5,label=f'{clf} PAV')
    ax.set_xticks(xpos); ax.set_xticklabels(['0','50','100','200','400']); ax.set_title(t); ax.grid(alpha=.3)
    if i==0: ax.legend(fontsize=7)
axs[-1].axis('off')
fig.supxlabel('SRS target labels k (non-uniform axis)'); fig.supylabel('ba_ge30 (%)')
plt.rcParams['font.sans-serif']=['Arial Unicode MS','PingFang SC','Heiti SC']
fig.tight_layout(); fig.savefig('/Users/perng_mini/AlphaCrops_local/verify/G_audit/labeleq/srs_curves.png',dpi=110)
print('ok')
