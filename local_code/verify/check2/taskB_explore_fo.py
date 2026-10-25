import sys, os
sys.dont_write_bytecode = True
CODE = ('/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/'
        '我的云端硬盘/Research/Alpha_Crops/label_expansion')
sys.path.insert(0, CODE)
import numpy as np, pandas as pd
from common import load_dataset, RESULTS_DIR
df = pd.read_parquet('/Users/perng_mini/AlphaCrops_local/data/dataset_v2.parquet', columns=['country','label','has_ae','has_s2'])
df = df[df.has_ae & df.has_s2].reset_index(drop=True)
classes = sorted(df['label'].unique()); print(classes)
y = df['label'].map({c: i for i, c in enumerate(classes)}).to_numpy()
P = os.path.join(RESULTS_DIR, 'preds')
for tgt in ['LV', 'SE']:
    test = np.load(f'{P}/{tgt}_test_idx.npy'); yt = y[test]
    vc = pd.Series(yt).value_counts().sort_index()
    print(tgt, 'test class counts:', {classes[c]: int(n) for c, n in vc.items()})
    for cf in ['ae', 's2']:
        for name, k in [('foreign_only', 0), ('ceiling_w25', None)]:
            recs = []
            for s in range(3):
                if k is None:
                    f = [x for x in os.listdir(P) if x.startswith(f'{tgt}_{name}_') and x.endswith(f'_{s}_{cf}.npy')][0]
                else:
                    f = f'{tgt}_{name}_{k}_{s}_{cf}.npy'
                p = np.load(f'{P}/{f}')
                recs.append({classes[c]: (p[yt == c] == c).mean() for c in vc.index})
            r = pd.DataFrame(recs).mean()
            print(f'  {cf} {name:12s}', '  '.join(f'{c[:10]}={v*100:.0f}' for c, v in r.items()))
