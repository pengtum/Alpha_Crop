# 只读：G tier1 行中，同一批 SRS 标签下 AE 分类器 − S2 分类器（ba_ge30 / OA），按 k 与国家汇总
import os, glob
os.environ["OMP_NUM_THREADS"]="1"
import pandas as pd, numpy as np
from scipy import stats
G="/Users/perng_mini/AlphaCrops_local/results/guidance_G/"
d=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(G+"rows_t1_*.csv"))],ignore_index=True)
d=d[d.method.isin(["foreign_only","srs"])].copy()
# k=0 在两次重复间共用：去重
d=d.drop_duplicates(subset=["method","clf","k","target","split","rep"])
k0=d[d.k==0]
print("k0 rows",len(k0), k0.groupby(["target","split"]).size().max())
out=[]
for k in [0,50,100,200,400]:
    s=d[d.k==k]
    if k==0: s=s[s.rep==0]
    w=s.pivot_table(index=["target","split","rep"],columns="clf",values=["ba_ge30","oa"])
    for m in ["ba_ge30","oa"]:
        diff=(w[(m,"ae")]-w[(m,"s2")]).groupby(level=0).mean()
        lev_ae=w[(m,"ae")].groupby(level=0).mean().mean(); lev_s2=w[(m,"s2")].groupby(level=0).mean().mean()
        t=stats.t.ppf(0.975,len(diff)-1); se=diff.std(ddof=1)/np.sqrt(len(diff))
        p=stats.ttest_1samp(diff,0).pvalue
        nd=diff.drop(["RO","PT"],errors="ignore")
        out.append(dict(k=k,metric=m,AE=round(lev_ae,2),S2=round(lev_s2,2),diff=round(diff.mean(),2),lo=round(diff.mean()-t*se,2),hi=round(diff.mean()+t*se,2),p=round(p,3),n_pos=int((diff>0).sum()),n=len(diff),diff_noROPT=round(nd.mean(),2),
                        **{c:round(v,2) for c,v in diff.items()}))
o=pd.DataFrame(out)
pd.set_option("display.width",250); pd.set_option("display.max_columns",40)
print(o)
o.to_csv("/Users/perng_mini/AlphaCrops_local/verify/reposition/label_budget_line/feat_by_k.csv",index=False)
# 学习曲线增益：k0->200、k0->400 分 AE/S2
lc=[]
for clf in ["ae","s2"]:
    s=d[d.clf==clf]
    base=s[(s.k==0)&(s.rep==0)].set_index(["target","split"])[["ba_ge30","oa"]]
    for k in [50,200,400]:
        x=s[s.k==k].groupby(["target","split"])[["ba_ge30","oa"]].mean()
        g=(x-base).groupby(level=0).mean()
        lc.append(dict(clf=clf,k=k,dBA_mean=round(g.ba_ge30.mean(),2),dBA_median=round(g.ba_ge30.median(),2),dBA_noROPT=round(g.ba_ge30.drop(["RO","PT"]).mean(),2),dOA_mean=round(g.oa.mean(),2),dOA_noROPT=round(g.oa.drop(["RO","PT"]).mean(),2)))
print(pd.DataFrame(lc))
