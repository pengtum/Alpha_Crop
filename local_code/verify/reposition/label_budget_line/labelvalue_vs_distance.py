# 只读：Cheng 的实践建议"远距离迁移需要本地标签"在我们设定下的描述性检验（C 级，n=11，事后）
# 标签价值 = G tier1 SRS 在 k 下相对 k=0（只用外国）的增益；距离 = overlap_audit 算的目标国到外国池的纬度差/大圆距离
import os, glob
os.environ["OMP_NUM_THREADS"]="1"
import pandas as pd, numpy as np
from scipy import stats
G="/Users/perng_mini/AlphaCrops_local/results/guidance_G/"
d=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(G+"rows_t1_*.csv"))],ignore_index=True)
d=d[d.method.isin(["foreign_only","srs"])].drop_duplicates(subset=["method","clf","k","target","split","rep"])
d=d[~((d.k==0)&(d.rep==1))]
lev=d.groupby(["clf","k","target"])[["ba_ge30","oa"]].mean()
dist=pd.read_csv("/Users/perng_mini/AlphaCrops_local/verify/reposition/overlap_audit/multisource_k0_by_target.csv").set_index("target")
rows=[]
for clf in ["ae","s2"]:
    for k in [50,200,400]:
        for m in ["ba_ge30","oa"]:
            gain=lev.loc[(clf,k)][m]-lev.loc[(clf,0)][m]
            k0=lev.loc[(clf,0)][m]
            for dv in ["mean_dlat_to_pool","min_dlat_to_pool","mean_km_to_pool"]:
                r=stats.spearmanr(gain,dist.loc[gain.index,dv])
                g2=gain.drop("RO"); r2=stats.spearmanr(g2,dist.loc[g2.index,dv])
                rows.append(dict(clf=clf,k=k,metric=m,dist=dv,rho=round(r.correlation,3),p=round(r.pvalue,3),rho_noRO=round(r2.correlation,3),p_noRO=round(r2.pvalue,3)))
o=pd.DataFrame(rows); pd.set_option("display.width",200)
print(o[o.dist!="min_dlat_to_pool"].to_string(index=False))
o.to_csv("/Users/perng_mini/AlphaCrops_local/verify/reposition/label_budget_line/labelvalue_vs_distance.csv",index=False)
# k=0 精度与距离（迁移亏损本身）
for clf in ["ae","s2"]:
    for m in ["ba_ge30","oa"]:
        k0=lev.loc[(clf,0)][m]
        for dv in ["mean_dlat_to_pool","mean_km_to_pool"]:
            r=stats.spearmanr(k0,dist.loc[k0.index,dv]); print("k0 level",clf,m,dv,round(r.correlation,3),round(r.pvalue,3))
g=pd.DataFrame({c:(lev.loc[(c,400)]-lev.loc[(c,0)]).ba_ge30 for c in ["ae","s2"]}); g["mean_dlat"]=dist.mean_dlat_to_pool
print(g.sort_values("mean_dlat").round(2))
