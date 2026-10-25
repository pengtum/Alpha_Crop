import pandas as pd, numpy as np
from scipy.stats import spearmanr
R='/Users/perng_mini/AlphaCrops_local/results/'
dd=pd.read_csv(R+'domain_distance.csv'); pw=pd.read_csv(R+'pairwise_v2.csv')
print('c2st_ae quantiles', dd.c2st_ae.quantile([0,.1,.25,.5,.75,1]).round(6).to_dict())
print('c2st_ae >=0.9999:', (dd.c2st_ae>=0.9999).sum(), '/55 ; ==1.0 exactly', (dd.c2st_ae==1.0).sum(), '; c2st_s2>=0.9999', (dd.c2st_s2>=0.9999).sum())
print('c2st_s2 quantiles', dd.c2st_s2.quantile([0,.25,.5,.75,1]).round(4).to_dict())
per=pd.read_csv(R+'check1_per_target.csv')
print(per[(per.descriptor=='climate_diff')&(per.feature=='ae')][['target','rho','top1','regret']].to_string())
# per-target: number of sources with c2st_ae==1 exactly (ties)
sym=pd.concat([dd, dd.rename(columns={'c1':'c2','c2':'c1'})])
for t in sorted(sym.c1.unique()):
    g=sym[sym.c2==t]
    print(t, 'n sources with c2st_ae>=0.99999:', (g.c2st_ae>=0.99999).sum(), ' exact 1.0:', (g.c2st_ae==1.0).sum(), ' unique vals', g.c2st_ae.round(6).nunique())
# check pw n_test
import pyarrow.parquet as pq
ds=pd.read_parquet('/Users/perng_mini/AlphaCrops_local/data/dataset_v2.parquet', columns=['country','has_ae','has_s2','label'])
b=ds[ds.has_ae&ds.has_s2]; cnt=b.country.value_counts()
cr=pw[pw.source!=pw.target]
print('target n_test == all target parcels:', (cr.n_test==cr.target.map(cnt)).all())
wi=pw[pw.source==pw.target]
print('within n_test/n', (wi.n_test/wi.source.map(cnt)).round(4).unique())
print('pairwise rows', len(pw), 'dups', pw.duplicated(['source','target','feature']).sum())
# classes present per country (both) and number of classes each source can predict
print(b.groupby('country').label.nunique().to_dict())
