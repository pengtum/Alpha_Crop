"""三维度结论的最终量化汇总。"""
import os, re
import numpy as np
import pandas as pd

OUT_DIR='results/experiments/exp6_scalable_mapping'
COST={'AT':1.05,'BE':1.00,'DE':1.00,'EE':0.36,'LT':0.40,'LV':0.31,
      'PT':0.48,'SE':1.36,'SI':0.63,'SK':0.38}
SOURCES=['AT','BE','DE','EE','LT','LV','PT','SE','SI','SK']
FEATURES=['ae','s2','fusion']
KEY=['source_only','random','balanced','climate_similar','climate_diverse',
     'representativeness','transfer_strong','transfer_weak',
     'cost_only','cost_aware_strong','cost_aware_weak']

def parse_info(s):
    if pd.isna(s) or s=='{}': return {}
    o={}
    for c,n in re.findall(r"['\"]([A-Z]{2})['\"]?\s*:\s*(?:np\.int64\()?(\d+)\)?",str(s)):
        o[c]=int(n)
    return o
def cc(s): return sum(n*COST.get(c,1.0) for c,n in parse_info(s).items())

def load():
    dfs=[]
    for s in SOURCES:
        for ft in FEATURES:
            f=os.path.join(OUT_DIR,f'scalable_{ft}_{s}_tier_b.csv')
            if not os.path.exists(f): continue
            d=pd.read_csv(f); d['source']=s; d['feature']=ft
            d['aug_cost']=d['sampling_info'].apply(cc); dfs.append(d)
    return pd.concat(dfs,ignore_index=True)

df=load()

# ======== 维度 1: 策略横向（fusion @ 10k） ========
print('\n[D1] STRATEGY RANKING — Fusion @ budget=10k (mean over 10 sources × 3 seeds)')
print('-'*78)
print(f'{"Strategy":<22}{"BA":>8}{"std_across_src":>17}{"Cost":>9}{"BA-Bal_pp":>12}')
agg=[]
for st in KEY:
    sub=df[(df.feature=='fusion')&(df.strategy==st)&(df.budget==10000)]
    if len(sub)==0: continue
    ba=sub.groupby('source').mean_ood_ba_equal.mean().mean()
    src_std=sub.groupby('source').mean_ood_ba_equal.mean().std()
    c=sub.aug_cost.mean()
    agg.append((st,ba,src_std,c))
bal_ba=[x[1] for x in agg if x[0]=='balanced'][0]
agg.sort(key=lambda x:-x[1])
for st,ba,sd,c in agg:
    print(f'{st:<22}{ba:>8.4f}{sd:>17.4f}{c:>9.0f}{(ba-bal_ba)*100:>+11.2f}pp')

# 谁最常 rank 1（per source × feature）
print('\n[D1b] Which strategy ranks #1 most often? (across 10 sources × 3 features × 5 budgets)')
wins={k:0 for k in KEY}
total=0
for src in SOURCES:
    for ft in FEATURES:
        for b in [1000,5000,10000,20000,50000]:
            sub=df[(df.source==src)&(df.feature==ft)&(df.budget==b)&(df.strategy.isin(KEY))]
            sub=sub.groupby('strategy').mean_ood_ba_equal.mean()
            if len(sub)==0: continue
            wins[sub.idxmax()]+=1; total+=1
for k,v in sorted(wins.items(),key=lambda x:-x[1]):
    print(f'  {k:<22}{v:>4}/{total} ({v/total*100:>5.1f}%)')

# ======== 维度 2: AE vs S2 vs Fusion sample efficiency ========
print('\n[D2] LABEL EFFICIENCY (Balanced strategy, 10 sources × 3 seeds aggregate)')
print('-'*60)
print(f'{"τ":<8}{"AE":>11}{"S2":>11}{"Fusion":>11}{"AE/S2":>10}{"Fus/S2":>10}')
def b2r(b,bas,t):
    if np.max(bas)<t: return np.nan
    if np.min(bas)>=t: return b[0]
    lb=np.log(b)
    for i in range(len(bas)-1):
        if bas[i]<t<=bas[i+1]:
            return float(np.exp(lb[i]+(t-bas[i])/(bas[i+1]-bas[i])*(lb[i+1]-lb[i])))
    return b[-1]
for t in [0.36,0.38,0.40,0.42,0.44]:
    rs={}
    for ft in FEATURES:
        sub=df[(df.feature==ft)&(df.strategy=='balanced')&(df.budget>0)]
        g=sub.groupby('budget').mean_ood_ba_equal.mean().reset_index().sort_values('budget')
        rs[ft]=b2r(g['budget'].values,g['mean_ood_ba_equal'].values,t)
    a,s,f=rs['ae'],rs['s2'],rs['fusion']
    r1=a/s if not (np.isnan(a) or np.isnan(s)) else np.nan
    r2=f/s if not (np.isnan(f) or np.isnan(s)) else np.nan
    fmt=lambda x: f'{x:.0f}' if not np.isnan(x) else 'X'
    print(f'τ={t:<6}{fmt(a):>11}{fmt(s):>11}{fmt(f):>11}{r1:>9.2f}x{r2:>9.2f}x')

# ======== 维度 3: source 影响 ========
print('\n[D3] PER-SOURCE BEST STRATEGY (Fusion @ 10k)')
print('-'*65)
print(f'{"Source":<8}{"Best strategy":<22}{"BA":>8}{"Worst":<22}{"BA":>8}')
for src in SOURCES:
    sub=df[(df.source==src)&(df.feature=='fusion')&(df.budget==10000)&(df.strategy.isin(KEY))]
    if len(sub)==0: continue
    g=sub.groupby('strategy').mean_ood_ba_equal.mean()
    best=g.idxmax(); worst=g.idxmin()
    print(f'{src:<8}{best:<22}{g[best]:>8.4f}{worst:<22}{g[worst]:>8.4f}')

# 每 source 最佳 feature
print('\n[D3b] PER-SOURCE BEST FEATURE (Balanced @ 10k)')
print('-'*60)
print(f'{"Source":<8}{"AE":>10}{"S2":>10}{"Fusion":>10}{"Best":<10}{"AE-S2_pp":>10}')
for src in SOURCES:
    row={}
    for ft in FEATURES:
        sub=df[(df.source==src)&(df.feature==ft)&(df.strategy=='balanced')&(df.budget==10000)]
        row[ft]=sub.mean_ood_ba_equal.mean()
    best=max(row,key=row.get)
    print(f'{src:<8}{row["ae"]:>10.4f}{row["s2"]:>10.4f}{row["fusion"]:>10.4f}'
          f'{best:<10}{(row["ae"]-row["s2"])*100:>+9.2f}pp')

print('\nDone.')
