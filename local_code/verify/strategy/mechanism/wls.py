import numpy as np, pandas as pd
from scipy import stats
def slope(df, y='drec_w', x='dlog_w', w='w'):
    return float((df[w]*df[x]*df[y]).sum()/(df[w]*df[x]**2).sum())
def slope_ctry(df, y='drec_w', x='dlog_w', w='w'):
    """逐国斜率 → 11 国 t 区间（与项目的国家层面推断一致）；另给合并斜率。"""
    bs=df.groupby('target').apply(lambda d: slope(d,y,x,w))
    n=len(bs); m=bs.mean(); se=bs.std(ddof=1)/np.sqrt(n); t=stats.t.ppf(.975,n-1)
    return dict(pooled=slope(df,y,x,w),mean=m,lo=m-t*se,hi=m+t*se,npos=int((bs>0).sum()),n=n)
