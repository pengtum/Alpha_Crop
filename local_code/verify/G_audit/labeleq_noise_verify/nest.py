import numpy as np, glob, os
SEL='/Users/perng_mini/AlphaCrops_local/results/guidance_G/sel'
C=['AT','BE','DE','EE','LT','LV','PT','RO','SE','SI','SK']
ov={}
for c in C:
  for s in (0,1):
    for r in (0,1):
      S={}
      for k in (50,100,200,400):
        f=f'{SEL}/{c}_s{s}_r{r}_none_srs_k{k}.npz'
        S[k]=set(np.load(f)['sel'].tolist())
      for a,b in [(50,100),(50,200),(50,400),(100,200),(200,400)]:
        ov.setdefault((a,b),[]).append(len(S[a]&S[b]))
for key,v in ov.items():
    v=np.array(v); print(key,'overlap min/median/max',v.min(),np.median(v),v.max(), ' expected under indep', key[0]*key[1]/30000)
# numpy Floyd non-nesting demo
rng=np.random.default_rng(1000); a=set(rng.choice(30000,50,replace=False)); rng=np.random.default_rng(1000); b=set(rng.choice(30000,400,replace=False)); print('demo overlap same seed 50 vs 400:',len(a&b))
