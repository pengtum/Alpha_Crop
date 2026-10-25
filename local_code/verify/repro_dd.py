import os, sys, pandas as pd, numpy as np
sys.dont_write_bytecode=True
CODE="/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion"
sys.path.insert(0, CODE)
from common import load_dataset, feature_cols, COUNTRIES, RANDOM_STATE
from scipy.spatial.distance import jensenshannon
import importlib.util
spec=importlib.util.spec_from_file_location('dd', os.path.join(CODE,'domain_distance.py')); dd=importlib.util.module_from_spec(spec); spec.loader.exec_module(dd)
df=load_dataset()
print('rows', len(df))
rng=np.random.default_rng(RANDOM_STATE)
samp=pd.concat([g.iloc[rng.choice(len(g), 3000, replace=False)] for _, g in df.groupby('country')], ignore_index=True)
classes=sorted(df['label'].unique())
comp={c: g['label'].value_counts(normalize=True).reindex(classes, fill_value=0).to_numpy() for c,g in df.groupby('country')}
centroid=df.groupby('country')[['longitude','latitude']].mean()
print(centroid.round(2).to_string())
spaces={}
for feat in ['ae','s2']:
    X=samp[feature_cols(samp,feat)].to_numpy(np.float64)
    X=(X-X.mean(0))/(X.std(0)+1e-9)
    sub=X[rng.choice(len(X),4000,replace=False)]
    sq=(sub**2).sum(1); d2=np.maximum(sq[:,None]+sq[None,:]-2*sub@sub.T,0)
    med=np.median(d2[np.triu_indices(len(sub),1)])
    spaces[feat]=(X,1.0/med); print(feat, 'dim', X.shape[1], 'median d2', med, 'gamma', 1/med)
ctry=samp['country'].to_numpy()
ref=pd.read_csv('/Users/perng_mini/AlphaCrops_local/results/domain_distance.csv')
out=[]
for _,r in ref.iterrows():
    a,b=r.c1,r.c2
    o=dict(c1=a,c2=b, geo=dd.haversine_km(*centroid.loc[a],*centroid.loc[b]), js=jensenshannon(comp[a],comp[b]))
    if (a,b) in [('AT','BE'),('BE','DE'),('EE','LV'),('PT','RO'),('LV','SE')]:
        for feat,(X,g) in spaces.items():
            o['mmd_'+feat]=dd.rbf_mmd2(X[ctry==a],X[ctry==b],g)
    out.append(o)
o=pd.DataFrame(out).merge(ref,on=['c1','c2'])
print('max |geo diff|', (o.geo-o.geo_km).abs().max(), 'max |js diff|', (o.js-o.class_js).abs().max())
print(o.dropna(subset=['mmd_ae_x'])[['c1','c2','mmd_ae_x','mmd_ae_y','mmd_s2_x','mmd_s2_y']].to_string())
# independent haversine sanity: AT-BE with sklearn
from sklearn.metrics.pairwise import haversine_distances
a=np.radians(centroid.loc[['AT','BE'],['latitude','longitude']].to_numpy())
print('sklearn haversine AT-BE km', haversine_distances(a)[0,1]*6371)
# biased vs unbiased check on small sample
X,g=spaces['ae']; A=X[ctry=='AT'][:500]; B=X[ctry=='AT'][500:1000]
print('same-country AT/AT unbiased mmd (should be ~0):', dd.rbf_mmd2(A,B,g))
