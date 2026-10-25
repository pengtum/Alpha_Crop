import sys, os, itertools
sys.path.insert(0, "/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion")
os.chdir(sys.path[0])
import numpy as np, pandas as pd
import analyze_G as A
from guidance_G import PRED
R = A.load_rows("/Users/perng_mini/AlphaCrops_local/results/guidance_G")
v = A.country_values(A.deltas(R,'AE','S2',PRED,reps=[0])).to_numpy()
signs = np.array(list(itertools.product([-1,1], repeat=len(v))))
m = (signs*v).mean(1); print('exact sign-flip p (mean):', ((np.abs(m) >= abs(v.mean())-1e-12).mean()).round(4))
t = (signs*v).mean(1)/((signs*v).std(1,ddof=1)/np.sqrt(len(v))); t0 = v.mean()/(v.std(ddof=1)/np.sqrt(len(v)))
print('exact sign-flip p (t):', ((np.abs(t) >= abs(t0)-1e-12).mean()).round(4))
L = A.label_equiv_table(R, {'pred θ': 0.340, 'pred L95': 0.037, 'pred U95': 0.642})
print(L.to_string())
# SRS curve slope context: median ba gain from k=50->100 and 200->400 by country x clf
rows=[]
for t_ in sorted(R.target.unique()):
    for clf in ('s2','ae'):
        ks, f = A.srs_curve(R, t_, clf); rows.append(dict(target=t_, clf=clf, **{f'k{int(k)}': y for k,y in zip(ks,f)}))
S = pd.DataFrame(rows); print(S.round(2).to_string())
