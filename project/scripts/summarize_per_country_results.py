"""按 source × feature 输出 per-target-country BA，看跨源/跨目标差异。"""
import os, re
import numpy as np
import pandas as pd

OUT_DIR = 'results/experiments/exp6_scalable_mapping'
SOURCES = ['AT','BE','DE','EE','LT','LV','PT','SE','SI','SK']
FEATURES = ['ae','s2','fusion']

def load():
    dfs=[]
    for s in SOURCES:
        for ft in FEATURES:
            f=os.path.join(OUT_DIR,f'scalable_{ft}_{s}_tier_b.csv')
            if not os.path.exists(f): continue
            d=pd.read_csv(f); d['source']=s; d['feature']=ft
            dfs.append(d)
    return pd.concat(dfs,ignore_index=True)

df=load()
# 取 balanced @ 10k @ 3 seeds 平均
sub = df[(df.strategy=='balanced')&(df.budget==10000)]
ba_cols=[c for c in df.columns if c.startswith('ba_') and len(c)==5]
print('\n=== Per-target BA matrix (Balanced @ 10k, mean of 3 seeds) ===')
for ft in FEATURES:
    print(f'\n[{ft.upper()}]')
    rows=[]
    for src in SOURCES:
        s=sub[(sub.source==src)&(sub.feature==ft)]
        if len(s)==0: continue
        row={'source':src}
        for c in ba_cols:
            row[c]=s[c].mean()
        row['mean']=s['mean_ood_ba_equal'].mean()
        rows.append(row)
    out=pd.DataFrame(rows).round(3)
    print(out.to_string(index=False))

# AE-S2 difference per (source, target)
print('\n\n=== AE − S2 per (source, target) ===')
for ft in ['ae','s2']:
    pass
piv_ae=sub[sub.feature=='ae'].groupby('source')[ba_cols+['mean_ood_ba_equal']].mean()
piv_s2=sub[sub.feature=='s2'].groupby('source')[ba_cols+['mean_ood_ba_equal']].mean()
diff=(piv_ae-piv_s2).round(3)
print(diff.to_string())
