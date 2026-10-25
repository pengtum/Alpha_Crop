import pandas as pd, numpy as np
d=pd.read_csv('/Users/perng_mini/AlphaCrops_local/results/pilot_selection.csv')
d['ba_ge30']*=100
sel=d[d.budget.isin([50,200,1000])]
print(sel.groupby(['strategy','budget']).size().unstack())
w=sel.pivot_table(index=['target','budget','seed','clf_feat'],columns='strategy',values='ba_ge30')
w['dG_rep']=w.ae_rep-w.s2_rep; w['dG_gap']=w.ae_gap-w.s2_gap
for m in ['random','geo_strat','s2_rep','ae_rep','s2_gap','ae_gap','uncertainty']:
    w['d_'+m]=w[m]-w['random']
print('\n== paired dG (AE-S2 guidance), per unit ==')
for c in ['dG_rep','dG_gap']:
    g=w[c]
    print(c, 'mean %.3f sd %.3f n %d'%(g.mean(),g.std(),g.notna().sum()))
    print(w.groupby(['budget','clf_feat'])[c].agg(['mean','std']).round(3))
    # variance components: country vs residual
    x=w[c].reset_index()
    for b in [50,200,1000]:
        xb=x[x.budget==b]
        cm=xb.groupby('target')[c].mean()
        within=xb.groupby(['target'])[c].var().mean()
        print(f'  k={b}: between-country SD of means {cm.std():.3f}, within-country SD {np.sqrt(within):.3f}, country means {cm.round(2).to_dict()}')
print('\n== method - random spread (per unit SD) ==')
for m in ['geo_strat','s2_rep','ae_rep','s2_gap','ae_gap','uncertainty']:
    g=w['d_'+m]
    print(m, w.groupby('budget')['d_'+m].agg(['mean','std',lambda s:np.percentile(s,10)]).round(2).to_dict('index'))
# correlation of dG_rep and dG_gap within unit
print('corr dG_rep,dG_gap', w[['dG_rep','dG_gap']].corr().iloc[0,1].round(3))
# random seed-to-seed SD per target-budget-clf
r=sel[sel.strategy=='random'].groupby(['target','budget','clf_feat']).ba_ge30.std()
print('random SD across 3 seeds (median by budget):', r.groupby('budget').median().round(3).to_dict())
w.to_csv('/Users/perng_mini/AlphaCrops_local/verify/guidance_review/pilot_wide.csv')
