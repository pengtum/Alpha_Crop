import os, sys
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[v]='2'
sys.dont_write_bytecode=True
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import indep_ae_mechanism as M
for tgt in ['DE','BE','AT']:
    S=np.load(f'/Users/perng_mini/AlphaCrops_local/results/guidance_G/stage0/{tgt}_s0.npz')
    nc=len(S['cand'])
    for f in ['ae','s2']:
        vals=np.array([M.mixing_index(S[f'G_{f}'][:nc], S[f'GF_{f}'], 2000, M.seed_of(f'spread_{tgt}_{f}_{r}'))[0] for r in range(40)])
        main=M.mixing_index(S[f'G_{f}'][:nc], S[f'GF_{f}'], 2000, M.seed_of(f'AE6_{tgt}_{f}_all'))[0]
        print(tgt,f,'main',round(main,4),'mean',round(vals.mean(),4),'sd',round(vals.std(ddof=1),4),'min',round(vals.min(),4),'max',round(vals.max(),4), 'q', np.round(np.quantile(vals,[.05,.5,.95]),4))
