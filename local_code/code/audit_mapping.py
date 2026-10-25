import sys, pandas as pd
sys.path.insert(0, "/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops")
from config.class_definitions import TIER_B_MAPPING
p = pd.read_parquet('data/parcels_eu.parquet')
print(p.columns.tolist()); print(len(p))
C11 = ['AT','BE','DE','EE','LT','LV','PT','RO','SE','SI','SK']
p = p[p.country.isin(C11) & ~p.parcel_id.str.startswith('DE_BB_')]
print('11-country parcels', len(p))
un = p[~p.EC_hcat_n.isin(TIER_B_MAPPING.keys())]
print('unmapped parcels', len(un), f'{len(un)/len(p):.1%}')
t = un.groupby(['EC_hcat_n']).agg(n=('parcel_id','size'), countries=('country', lambda s: dict(s.value_counts().head(4)))).sort_values('n', ascending=False)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 120)
print(t.to_string())
