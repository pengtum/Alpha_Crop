import numpy as np, sys
import guidance_G as G
D=G.GData()
print('country split | acc_oracle acc_ae acc_s2 | nll_oracle nll_ae nll_s2 | temp_oracle')
for c in G.COUNTRIES_G:
    for s in (0,1):
        S=dict(np.load(G.os.path.join(G.OUT,'stage0',f'{c}_s{s}.npz')))
        o=np.load(G.oracle_path(c,s))
        pp=G.pool_positions(S,s,0); idx=S['allT'][pp]; y=D.y[idx]
        def q(P):
            P=np.asarray(P,float); acc=(P.argmax(1)==y).mean(); nll=-np.log(np.clip(P[np.arange(len(y)),y],1e-12,1)).mean(); return acc,nll
        ao,no=q(o['P_T']); aa,na=q(S['P_T_ae'][pp]); a2,n2=q(S['P_T_s2'][pp])
        print(f'{c} s{s} | {ao:.3f} {aa:.3f} {a2:.3f} | {no:.3f} {na:.3f} {n2:.3f} | {float(o["temp"]):.2f} n={len(y)} oracleN={len(o["P_T"])}')
