import os, pandas as pd, numpy as np
D='/Users/perng_mini/AlphaCrops_local/data'
ds=pd.read_parquet(os.path.join(D,'dataset_v2.parquet'), columns=['parcel_id','country','EC_hcat_n','label','has_ae','has_s2','area_ha'])
ae=ds[ds.has_ae]
print('has_ae but not s2 by country:', ae[~ae.has_s2].country.value_counts().to_dict())
for c in ['EE','LV','SI','SK','DE','BE','AT','SE']:
    g=ae[ae.country==c]
    t=pd.crosstab(g.label, g.has_s2)
    t['frac_missing']=(t[False]/(t[False]+t[True])).round(3)
    print(c, 'overall missing %.3f'%(1-g.has_s2.mean()), 'median area miss/ok %.2f/%.2f'%(g[~g.has_s2].area_ha.median(), g[g.has_s2].area_ha.median()))
    print(t[t[False]+t[True]>=30].sort_values('frac_missing',ascending=False).head(6).to_string())
print('arable_crops by country (both):', ds[(ds.EC_hcat_n=='arable_crops')&ds.has_ae&ds.has_s2].country.value_counts().to_dict())
print('shrubberries by country (both):', ds[(ds.EC_hcat_n=='shrubberries_shrubs')&ds.has_ae&ds.has_s2].country.value_counts().to_dict())
