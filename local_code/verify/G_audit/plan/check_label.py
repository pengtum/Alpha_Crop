import glob, numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
OUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R = pd.concat([pd.read_csv(f) for f in glob.glob(f'{OUT}/rows_t*_*.csv')], ignore_index=True)
g = R[R.method.isin(['srs', 'foreign_only'])]
print(g.groupby(['k']).size().to_dict())
C = g.groupby(['target', 'clf', 'k']).ba_ge30.mean().unstack('k').round(2)
print(C)
C2 = C.copy()
ok = 0
for (t, c), row in C.iterrows():
    ks = row.index.to_numpy(float); f = IsotonicRegression().fit_transform(ks, row.to_numpy())
    C2.loc[(t, c)] = np.round(f, 2)
print('isotonic:'); print(C2)
print('f(50)-f(0) 中位数', float((C2[50] - C2[0]).median()), '; f(400)-f(200) 中位数', float((C2[400] - C2[200]).median()))
print('f(50)-f(0) < 0.3 的格数', int(((C2[50] - C2[0]) < 0.3).sum()), '/', len(C2), '; f(400)-f(200) < 0.3:', int(((C2[400] - C2[200]) < 0.3).sum()))
