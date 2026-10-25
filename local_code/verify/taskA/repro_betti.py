"""用轻量数据 + 假模型运行 pilot_betti.main()，捕获实际使用的 foreign/sel/test/w，与主试点逐行复现的结果比较。
特征列用假值：ae_0 / s2_0 = 行号（用于从 X_tr 反推下标），其余 31 列随机。"""
import os, sys, pickle
os.environ.setdefault('OMP_NUM_THREADS', '2')
CODE = "/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion"
sys.path.insert(0, CODE)
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split as tts
import common
import pilot_target_selection as pts
import pilot_betti as pb
V = '/Users/perng_mini/AlphaCrops_local/verify/taskA/'
MODE = sys.argv[1]            # 'parcel' | 'parcel_shuffled' | 'cluster'

def light_load(require=('ae', 's2')):
    df = pd.read_parquet(common.DATASET_V2, columns=['parcel_id', 'country', 'longitude', 'latitude', 'label', 'has_ae', 'has_s2'])
    for r in require:
        df = df[df[f'has_{r}']]
    df = df.reset_index(drop=True)
    g = np.random.default_rng(7)
    idx = np.arange(len(df), dtype=np.float64)
    fake = {}
    for f in ['ae', 's2']:
        fake[f'{f}_0'] = idx
        for j in range(1, 32):
            fake[f'{f}_{j}'] = g.standard_normal(len(df)).astype(np.float32)
    df = pd.concat([df, pd.DataFrame(fake)], axis=1)
    if MODE == 'parcel_shuffled':   # 打乱标签：若选样 class-blind，选到的下标应完全不变
        df['label'] = np.random.default_rng(123).permutation(df['label'].to_numpy())
    return df

calls, tests, splits = [], [], []
class Fake:
    def predict(self, X):
        tests.append(X[:, 0].astype(np.int64)); return np.zeros(len(X), int)
def fake_fit(X_tr, y_tr, w_tr, X_te, seed):
    calls.append(dict(tr=X_tr[:, 0].astype(np.int64), y=np.asarray(y_tr), w=None if w_tr is None else np.asarray(w_tr), seed=seed))
    return Fake()
def rec_split(*a, **k):
    out = tts(*a, **k); splits.append((a[0].copy(), out[0].copy(), out[1].copy(), k)); return out

pb.load_dataset = light_load
pb.fit_predict = fake_fit
pb.train_test_split = rec_split
pb.CLF_FEATS = ['ae']
if MODE == 'cluster':
    pb.UNIT = 'cluster'
    pb.STRATEGIES = ['rand_cluster', 'admin_rep', 'img_rep_ae', 'img_rep_s2']
    pb.TARGETS = ['LV', 'PT']
    pb.SEEDS = [0]
pb.OUT = V + f'smoke_{MODE}.csv'
if os.path.exists(pb.OUT): os.remove(pb.OUT)
pb.main()
out = pd.read_csv(pb.OUT)
print(out.head(12).to_string()); print('rows', len(out))
with open(V + f'capture_{MODE}.pkl', 'wb') as fh:
    pickle.dump(dict(calls=calls, tests=tests, splits=splits, targets=pb.TARGETS, budgets=(pb.CLUSTER_BUDGETS if MODE=='cluster' else pb.BUDGETS),
                     seeds=pb.SEEDS, strategies=pb.STRATEGIES, csv=out), fh)
