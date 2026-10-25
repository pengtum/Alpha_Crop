import sys, os, glob, numpy as np, pandas as pd
sys.path.insert(0,"/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion")
sys.dont_write_bytecode=True
import mediators_G as MG, guidance_G as G
D=G.GData()
OUT="/Users/perng_mini/AlphaCrops_local/results/guidance_G"
rows=[]
for p in sorted(glob.glob(OUT+'/stage0/*_s*.npz')):
    tgt,split=os.path.basename(p)[:-4].split('_s'); split=int(split)
    S=dict(np.load(p)); pp=MG.pool_positions(S,split,0); y=D.y[S['allT'][pp]]
    O=np.load(f'{OUT}/oracle/{tgt}_s{split}.npz'); Go=O['G'].astype(float)
    if len(Go)!=len(pp): print('shape mismatch',tgt,split,len(Go),len(pp)); continue
    ref=np.sort(np.random.default_rng(0).choice(len(pp),min(MG.N_REF,len(pp)),replace=False))
    Z=(Go-Go.mean(0))/(Go.std(0)+1e-12); Zc=np.clip(Z,-5,5)
    r=dict(target=tgt,split=split,oracle_raw=MG.knn_ba(Go[ref],y[ref]),oracle_zclip=MG.knn_ba(Zc[ref],y[ref]))
    # sanity: recompute AE knn to match guidance_quality
    r['ae_check']=MG.knn_ba(MG._part(S['G_ae'][pp],32)[ref],y[ref])
    rows.append(r); print(r,flush=True)
df=pd.DataFrame(rows); df.to_csv('/private/tmp/c2check/oracle_knn.csv',index=False)
print(df.groupby('target').mean(numeric_only=True).round(1)); print(df.mean(numeric_only=True).round(1))
