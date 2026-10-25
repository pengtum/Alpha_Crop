import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
import analyze_G as A
R = A.load_rows()
print(len(R), R.cond.value_counts().to_dict())
ms = ['lpm2_nogeo','typiclust','margin']
# pooled reproduce
for met in ['ba_ge30','oa','ba_P2','ba_P1','f1_ge30','ba_i200','ba_i250']:
    s = A.t_summary(A.country_values(A.deltas(R,'ORACLE','S2',ms,met,reps=[0])))
    print(f'H pooled {met}:', A.fmt(s))
print()
for met in ['ba_ge30','oa','ba_P2']:
    for m in ms:
        for b in ['S2','AE']:
            s = A.t_summary(A.country_values(A.deltas(R,'ORACLE',b,[m],met,reps=[0])))
            print(f'{met:8s} {m:11s} ORACLE-{b}:', A.fmt(s))
        s = A.t_summary(A.country_values(A.deltas(R,'AE','S2',[m],met,reps=[0])))
        print(f'{met:8s} {m:11s} AE-S2    :', A.fmt(s))
    print()
# vs SRS
R0 = R[R.rep==0]
for met in ['ba_ge30','oa']:
    for cond in ['ORACLE','S2','AE']:
        for m in ms:
            d = A.vs_srs(R0, cond, [m], met)
            d = d[d.rep==0]
            s = A.t_summary(A.country_values(d))
            byk = d.groupby('k').d.mean().round(3).to_dict()
            print(f'{met} {cond:6s} {m:11s} -SRS:', A.fmt(s), 'byk', byk)
    print()
