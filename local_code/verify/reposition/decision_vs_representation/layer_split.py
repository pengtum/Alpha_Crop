# 只读、轻量：把检验 3 的逐国结果拆成"决策层可修复"与"表征层残差"，并与到外国池的纬度差做描述性相关（C 级，事后）
import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
import pandas as pd, numpy as np
from scipy.stats import spearmanr, wilcoxon
R="/Users/perng_mini/AlphaCrops_local/results/"
OUT="/Users/perng_mini/AlphaCrops_local/verify/reposition/decision_vs_representation/"
a=pd.read_csv(R+"check3_arms.csv"); a=a[a.split=="random"]
cap=pd.read_csv(R+"check3_capacity_arms.csv")
dl=pd.read_csv("/Users/perng_mini/AlphaCrops_local/verify/reposition/overlap_audit/multisource_k0_by_target.csv").set_index("target")
def piv(d,m):
    return d.pivot_table(index=["target","feat"],columns="arm",values=m)
o=piv(a,"oa"); b=piv(a,"ba_ge30")
rows=[]
for (t,f) in o.index:
    r=dict(target=t,feat=f)
    r["oa_total"]=o.loc[(t,f),"Wnat"]-o.loc[(t,f),"L0"]
    r["oa_comp"]=o.loc[(t,f),"PM"]-o.loc[(t,f),"L0"]
    r["oa_resid"]=o.loc[(t,f),"Wnat"]-o.loc[(t,f),"CMN"]
    r["oa_oracle_gain"]=o.loc[(t,f),"ORACLE_oa"]-o.loc[(t,f),"L0"]
    r["oa_after_oracle"]=o.loc[(t,f),"Wnat"]-o.loc[(t,f),"ORACLE_oa"]
    r["ba_total"]=b.loc[(t,f),"W0"]-b.loc[(t,f),"L0"]
    r["ba_set"]=b.loc[(t,f),"LS"]-b.loc[(t,f),"L0"]
    r["ba_resid"]=b.loc[(t,f),"W0"]-b.loc[(t,f),"CMB"]
    r["ba_oracle_gain"]=b.loc[(t,f),"ORACLE_ba"]-b.loc[(t,f),"L0"]
    r["ba_after_oracle"]=b.loc[(t,f),"W0"]-b.loc[(t,f),"ORACLE_ba"]
    rows.append(r)
d=pd.DataFrame(rows)
d["oa_oracle_share"]=d.oa_oracle_gain/d.oa_total
d["ba_oracle_share"]=d.ba_oracle_gain/d.ba_total
d.to_csv(OUT+"layer_split_by_country.csv",index=False)
print("== 11 国均值（早停，随机划分）==")
print(d.groupby("feat")[["oa_total","oa_comp","oa_resid","oa_oracle_gain","oa_after_oracle","ba_total","ba_set","ba_resid","ba_oracle_gain","ba_after_oracle"]].mean().round(2).T)
g=d.groupby("feat")[["oa_oracle_gain","oa_total","ba_oracle_gain","ba_total"]].sum()
print("合并比例 决策层上界/总损失: OA", (g.oa_oracle_gain/g.oa_total).round(3).to_dict(), " BA", (g.ba_oracle_gain/g.ba_total).round(3).to_dict())
w=d.pivot(index="target",columns="feat")
diff=pd.DataFrame({k:w[(k,"ae")]-w[(k,"s2")] for k in ["oa_resid","ba_resid","oa_oracle_gain","ba_oracle_gain","oa_after_oracle","ba_after_oracle","oa_comp"]})
diff=diff.join(dl[["mean_dlat_to_pool","min_dlat_to_pool","mean_km_to_pool"]])
diff.to_csv(OUT+"ae_minus_s2_by_country.csv")
print("\n== AE − S2 逐国（正 = AE 该部分更大）==")
print(diff.round(2))
print("\n均值与 Wilcoxon：")
for k in ["oa_resid","ba_resid","oa_oracle_gain","ba_oracle_gain","oa_after_oracle","ba_after_oracle"]:
    x=diff[k]; print(f"  {k}: 均值 {x.mean():+.2f}，正 {int((x>0).sum())}/11，Wilcoxon p={wilcoxon(x).pvalue:.3f}")
print("\n== 与到外国池纬度差的 Spearman（n=11，朴素 p）==")
for k in ["oa_resid","ba_resid","ba_after_oracle","oa_after_oracle"]:
    for z in ["mean_dlat_to_pool","min_dlat_to_pool"]:
        rho,p=spearmanr(diff[k],diff[z]); 
        rho2,p2=spearmanr(diff.drop(["PT"])[k],diff.drop(["PT"])[z])
        print(f"  {k} vs {z}: rho={rho:+.3f} p={p:.3f} | 去 PT rho={rho2:+.3f} p={p2:.3f}")
# 固定容量
c=cap[cap.arm.isin(["L0","PM","CMN","Wnat","LS","CMB","W0"])]
po=c.pivot_table(index=["target","feat","cap"],columns="arm",values="oa")
pb=c.pivot_table(index=["target","feat","cap"],columns="arm",values="ba_ge30")
cc=pd.DataFrame({"oa_resid":po.Wnat-po.CMN,"ba_resid":pb.W0-pb.CMB}).reset_index()
cc=cc[cc.cap=="c300"].pivot(index="target",columns="feat")
cd=pd.DataFrame({"oa_resid":cc[("oa_resid","ae")]-cc[("oa_resid","s2")],"ba_resid":cc[("ba_resid","ae")]-cc[("ba_resid","s2")]}).join(dl[["mean_dlat_to_pool"]])
print("\n== 固定 300 棵（事后）AE − S2 域残差 ==")
print(cd.round(2))
for k in ["oa_resid","ba_resid"]:
    x=cd[k]; rho,p=spearmanr(x,cd.mean_dlat_to_pool)
    print(f"  {k}: 均值 {x.mean():+.2f}，正 {int((x>0).sum())}/11，Wilcoxon p={wilcoxon(x).pvalue:.3f}；vs mean_dlat rho={rho:+.3f} p={p:.3f}")
cd.to_csv(OUT+"ae_minus_s2_resid_c300.csv")
