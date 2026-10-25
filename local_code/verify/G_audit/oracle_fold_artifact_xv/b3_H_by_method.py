"""原始存档行：ORACLE − S2 按方法、按国家（国家值 = 单元、分类器、k 平均）。只读。"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import analyze_G as A
OUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f'{OUT}/rows_t*_*.csv'))], ignore_index=True)
print('rows', len(R))
ms = ['lpm2_nogeo', 'typiclust', 'margin']
sH = A.t_summary(A.country_values(A.deltas(R, 'ORACLE', 'S2', ms, reps=[0])))
print('H all', {k: round(v, 3) for k, v in sH.items()})
tab = {}
for m in ms:
    cv = A.country_values(A.deltas(R, 'ORACLE', 'S2', [m], reps=[0]))
    s = A.t_summary(cv)
    print(m, 'ORACLE-S2', round(s['mean'], 3), [round(s['lo95'], 3), round(s['hi95'], 3)], s['n_pos'])
    tab[m] = cv
    cvA = A.country_values(A.deltas(R, 'AE', 'S2', [m], reps=[0]))
    tab[m + '_AE-S2'] = cvA
print(pd.DataFrame(tab).round(3).to_string())
