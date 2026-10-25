# P2 校正所用 pi_train 在 AE 与 S2 选样之间差多少；目标样本在训练先验中的权重占比
import sys, os, glob
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from scipy import stats
import guidance_G as G
from stageA import class_weights_from
OUT = "/Users/perng_mini/AlphaCrops_local/results/guidance_G"
D = G.GData()
rows = []
for f in sorted(glob.glob(f"{OUT}/stage0/*_s*.npz")):
    tgt, sp = os.path.basename(f)[:-4].split('_s'); split = int(sp)
    S = np.load(f)
    fr = S['foreign']; yF = D.y[fr]; cw = class_weights_from(yF)
    pp = G.pool_positions(S, split, 0); idx_pool = S['allT'][pp]
    pi_pool = np.bincount(D.y[idx_pool], minlength=10)/len(idx_pool)
    wF = cw[yF]
    for method in ('margin','predstrat_srs','lpm2_nogeo','kcenter','typiclust','kherding'):
        for k in (50,200):
            pis = {}; 
            for cond in ('AE','S2'):
                z = np.load(f"{OUT}/sel/{tgt}_s{split}_r0_{cond}_{method}_k{k}.npz")
                ys = D.y[z['sel']]
                w = np.r_[wF, G.R_MULT*cw[ys]]
                pi = np.bincount(np.r_[yF, ys], weights=w, minlength=10)/w.sum()
                pis[cond] = pi
                tshare = G.R_MULT*cw[ys].sum()/w.sum()
                comp = np.bincount(ys, minlength=10)/k
                rows.append(dict(target=tgt, split=split, method=method, k=k, cond=cond, tshare=tshare,
                                 L1_train_pool=np.abs(pi-pi_pool).sum(), comp_L1=np.abs(comp-pi_pool).sum(),
                                 nF=len(fr)))
            wa = pi_pool/pis['AE']; wb = pi_pool/pis['S2']
            rows.append(dict(target=tgt, split=split, method=method, k=k, cond='AE-S2',
                             L1_train_AE_S2=np.abs(pis['AE']-pis['S2']).sum(),
                             max_logw_diff=np.max(np.abs(np.log(wa[pi_pool>0])-np.log(wb[pi_pool>0])))))
Q = pd.DataFrame(rows)
Q.to_csv(os.path.join(os.path.dirname(__file__) if '__file__' in dir() else '.', 'pitrain.csv'), index=False)
print("foreign n per unit:", Q.nF.dropna().unique()[:5])
a = Q[Q.cond.isin(['AE','S2'])].groupby(['method','k','cond'])[['tshare','L1_train_pool','comp_L1']].mean().round(4)
print(a.to_string())
b = Q[Q.cond=='AE-S2'].groupby(['method','k'])[['L1_train_AE_S2','max_logw_diff']].agg(['mean','max']).round(4)
print(b.to_string())
