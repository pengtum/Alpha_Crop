import glob, os, sys
import numpy as np, pandas as pd
from scipy import stats
OUT='/Users/perng_mini/AlphaCrops_local/results/guidance_G'
E1=['lpm2_nogeo','kcenter','typiclust','kherding']; PRED=['margin','predstrat_srs']; HYB=['gist_margin','badge_c']
ALLM=E1+PRED+HYB
UNIT=['target','split','rep']
def load():
    fs=glob.glob(os.path.join(OUT,'rows_t*_*.csv'))
    R=pd.concat([pd.read_csv(f).assign(tier=os.path.basename(f).split('_')[1]) for f in fs],ignore_index=True)
    return R
def vs_srs(R, conds=('AE','S2'), methods=ALLM, metrics=('ba_ge30','oa','ba_P2','oa_P2','ba_P1','ba_far','f1_ge30'), ks=(50,200), clfs=('s2','ae')):
    R=R[R.k.isin(ks)&R.clf.isin(clfs)]
    A=R[R.cond.isin(conds)&R.method.isin(methods)].copy()
    S=R[R.method=='srs'].set_index(UNIT+['clf','k'])
    key=pd.MultiIndex.from_frame(A[UNIT+['clf','k']])
    for m in metrics+('comp_L1','n_classes_sel'):
        A['d_'+m]=A[m].to_numpy()-S[m].reindex(key).to_numpy()
        A['srs_'+m]=S[m].reindex(key).to_numpy()
    return A
def ci(x):
    x=np.asarray(x,float); x=x[~np.isnan(x)]; n=len(x); m=x.mean(); se=x.std(ddof=1)/np.sqrt(n); t=stats.t.ppf(.975,n-1)
    return dict(mean=m,lo=m-t*se,hi=m+t*se,npos=int((x>0).sum()),n=n,p=2*stats.t.sf(abs(m/se),n-1) if se>0 else np.nan)
def fmt(d,prec=2):
    return f"{d['mean']:+.{prec}f} [{d['lo']:+.{prec}f}, {d['hi']:+.{prec}f}] {d['npos']}/{d['n']}"
def cval(df,col,by=('method','cond','clf','k')):
    """先单元平均，再对格子等权 → 每国一个值"""
    return df.groupby(list(by)+['target'])[col].mean().groupby(['target']).mean()
