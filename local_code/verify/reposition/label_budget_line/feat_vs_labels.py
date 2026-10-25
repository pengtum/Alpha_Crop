# 只读：特征选择与标签预算的"汇率"（LightGBM，G tier1 SRS 行，C 级描述）
import os, glob
os.environ["OMP_NUM_THREADS"]="1"
import pandas as pd, numpy as np
G="/Users/perng_mini/AlphaCrops_local/results/guidance_G/"
d=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(G+"rows_t1_*.csv"))],ignore_index=True)
d=d[d.method.isin(["foreign_only","srs"])].drop_duplicates(subset=["method","clf","k","target","split","rep"])
d=d[~((d.k==0)&(d.rep==1))]
lev=d.groupby(["clf","k","target"])[["ba_ge30","oa"]].mean()
for m in ["ba_ge30","oa"]:
    t=lev[m].unstack("target")
    print("\n==",m); print(t.round(2))
    for a,b in [(("ae",0),("s2",400)),(("ae",50),("s2",400)),(("ae",50),("s2",200)),(("ae",0),("s2",200))]:
        x=t.loc[a]-t.loc[b]
        print(f"{a} - {b}: mean {x.mean():+.2f}, >0 {int((x>0).sum())}/11, noROPT {x.drop(['RO','PT']).mean():+.2f}")
