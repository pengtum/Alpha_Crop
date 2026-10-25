import os
os.environ['OMP_NUM_THREADS']='2'
import pandas as pd
R='/Users/perng_mini/AlphaCrops_local/results/'
a=pd.read_csv(R+'check3_arms.csv'); a=a[a.split=='random']
print(sorted(a.arm.unique()))
c=pd.read_csv(R+'check3_capacity_arms.csv'); print(sorted(c.arm.unique()), c.cap.unique())
def summ(df,label):
    p=df.pivot_table(index=['target','feat'],columns='arm',values='oa')
    pb=df.pivot_table(index=['target','feat'],columns='arm',values='ba_ge30')
    for sub in [None,['RO','PT']]:
        q=p if sub is None else p[~p.index.get_level_values(0).isin(sub)]
        qb=pb if sub is None else pb[~pb.index.get_level_values(0).isin(sub)]
        nm='11' if sub is None else '9'
        for f in ['ae','s2']:
            x=q.xs(f,level=1); xb=qb.xs(f,level=1)
            s=f"{label} {nm} {f}: "
            wn=[k for k in ['Wnat','W-nat','WNAT','wnat'] if k in x.columns]
            if 'PO' in x.columns: s+=f"PO-L0 OA {(x.PO-x.L0).mean():+.2f}; "
            if 'PK50' in x.columns: s+=f"PK50-L0 {(x.PK50-x.L0).mean():+.2f}; "
            if wn: s+=f"Wnat-L0 OA {(x[wn[0]]-x.L0).mean():+.2f}; "
            if 'W0' in xb.columns: s+=f"W0-L0 BA {(xb.W0-xb.L0).mean():+.2f}; "
            if 'ORACLE' in x.columns: s+=f"ORACLE-L0 OA {(x.ORACLE-x.L0).mean():+.2f} BA {(xb.ORACLE-xb.L0).mean():+.2f}"
            print(s)
summ(a,'es')
for cap in c.cap.unique():
    summ(c[c.cap==cap],cap)
