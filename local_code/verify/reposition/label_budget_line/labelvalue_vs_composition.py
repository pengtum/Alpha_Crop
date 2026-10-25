# 只读：标签价值能否由"目标国与外国池的类别构成差异"预测，以及这一差异能否用 50 个随机标签估出（C 级，n=11，事后）
import os, glob, json
os.environ["OMP_NUM_THREADS"]="1"
import numpy as np, pandas as pd
from scipy import stats
C="/Users/perng_mini/AlphaCrops_local/data/cache_guidance_G/"
y=np.load(C+"y.npy"); ctry=np.load(C+"ctry.npy"); meta=json.load(open(C+"meta.json")); cls=meta["classes"]
S0="/Users/perng_mini/AlphaCrops_local/results/guidance_G/stage0/"
G="/Users/perng_mini/AlphaCrops_local/results/guidance_G/"
rng=np.random.default_rng(42)
rows=[]
for f in sorted(glob.glob(S0+"*_s*.npz")):
    t,s=os.path.basename(f)[:-4].split("_"); z=np.load(f,allow_pickle=True)
    pF=np.bincount(y[z["foreign"]],minlength=10)/len(z["foreign"])
    pT=np.bincount(y[z["test"]],minlength=10)/len(z["test"])
    L1=np.abs(pT-pF).sum()
    tv=0.5*L1
    # 目标测试集中占比>=1% 但外国池占比<1% 的类（外国池"稀缺"类）的目标份额
    scarce=pT[(pT>=0.01)&(pF<0.01)].sum()
    # 用 50 个随机候选标签估计的 L1（200 次抽样的中位数与 80% 区间）
    yc=y[z["cand"]]; est=[]
    for _ in range(200):
        smp=rng.choice(yc,50,replace=False); ph=np.bincount(smp,minlength=10)/50; est.append(np.abs(ph-pF).sum())
    est=np.array(est)
    rows.append(dict(target=t,split=s,L1_true=L1,L1_est50_med=np.median(est),L1_est50_p10=np.quantile(est,.1),L1_est50_p90=np.quantile(est,.9),scarce_share=scarce,
                     n_cls_test=int((pT>0.0).sum())))
c=pd.DataFrame(rows).groupby("target").mean(numeric_only=True)
d=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(G+"rows_t1_*.csv"))],ignore_index=True)
d=d[d.method.isin(["foreign_only","srs"])].drop_duplicates(subset=["method","clf","k","target","split","rep"])
d=d[~((d.k==0)&(d.rep==1))]
lev=d.groupby(["clf","k","target"])[["ba_ge30","oa"]].mean()
for clf in ["ae","s2"]:
    for k in [200,400]:
        for m in ["ba_ge30","oa"]:
            c[f"gain_{clf}_{k}_{m}"]=lev.loc[(clf,k)][m]-lev.loc[(clf,0)][m]
pd.set_option("display.width",250); pd.set_option("display.max_columns",30)
print(c.round(3).sort_values("L1_true"))
out=[]
for col in [x for x in c.columns if x.startswith("gain_")]:
    for pv in ["L1_true","L1_est50_med","scarce_share"]:
        r=stats.spearmanr(c[col],c[pv]); cc=c.drop("RO"); r2=stats.spearmanr(cc[col],cc[pv])
        cc2=c.drop(["RO","PT"]); r3=stats.spearmanr(cc2[col],cc2[pv])
        out.append(dict(gain=col,predictor=pv,rho=round(r.correlation,3),p=round(r.pvalue,3),rho_noRO=round(r2.correlation,3),p_noRO=round(r2.pvalue,3),rho_noROPT=round(r3.correlation,3),p_noROPT=round(r3.pvalue,3)))
o=pd.DataFrame(out); print(o.to_string(index=False))
print("L1_true vs L1_est50 spearman", stats.spearmanr(c.L1_true,c.L1_est50_med))
c.to_csv("/Users/perng_mini/AlphaCrops_local/verify/reposition/label_budget_line/composition_by_target.csv")
o.to_csv("/Users/perng_mini/AlphaCrops_local/verify/reposition/label_budget_line/labelvalue_vs_composition.csv",index=False)
