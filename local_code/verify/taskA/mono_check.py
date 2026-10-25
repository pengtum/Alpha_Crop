import sys
CODE = "/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion"
sys.path.insert(0, CODE)
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
import common, pilot_betti as pb
df = pd.read_parquet(common.DATASET_V2, columns=['parcel_id', 'country', 'has_ae', 'has_s2'])
df = df[df.has_ae & df.has_s2].reset_index(drop=True)
df = df.merge(pd.read_parquet(common.DATA_DIR + '/parcel_nuts.parquet', columns=['parcel_id', 'nuts3']), on='parcel_id', how='left')
for tgt in ['PT', 'SE', 'SK', 'LV']:
    t_idx = np.flatnonzero(df.country.to_numpy() == tgt)
    pool, _ = train_test_split(t_idx, test_size=0.3, random_state=42)
    g, lab = pd.factorize(df.nuts3.to_numpy()[pool]); cap = np.bincount(g)
    for b in [50, 200, 1000]:
        n = pb.allocate_parcels(cap / len(g), cap, b)
        o = np.argsort(cap, kind='stable')
        mono = np.all(np.diff(n[o]) >= 0)
        if tgt in ('SE', 'PT') and b == 50:
            print(tgt, b, 'smallest groups (pool cap, n_g):', [(lab[i], int(cap[i]), int(n[i])) for i in o[:6]])
        print(tgt, b, 'n_g nondecreasing in gamma:', mono)
