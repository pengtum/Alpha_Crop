import os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion')
import numpy as np, lightgbm as lgb
from sklearn.utils.class_weight import compute_sample_weight
from common import LGBM_PARAMS, EARLY_STOPPING, load_dataset, feature_cols
D='/Users/perng_mini/AlphaCrops_local/results/check3'
PARAMS = {**LGBM_PARAMS, 'n_estimators': 500, 'class_weight': None}
df = load_dataset(); classes = sorted(df['label'].unique())
y = df['label'].map({c: i for i, c in enumerate(classes)}).to_numpy()
R = np.load('replay_cm_idx.npz')
for T, feat, name in [('RO','ae','cmn'), ('PT','s2','cmb'), ('AT','ae','cmn')]:
    X = df[feature_cols(df, feat)].to_numpy(np.float32)
    tr, va = R[f'{T}_cm_tr'], R[f'{T}_cm_va']
    w = name == 'cmb'
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(X[tr], y[tr], sample_weight=compute_sample_weight('balanced', y[tr]) if w else None,
          eval_set=[(X[va], y[va])], eval_sample_weight=[compute_sample_weight('balanced', y[va])] if w else None,
          callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)])
    test = np.load(f'{D}/{T}_split.npz')['test']
    p = np.zeros((len(test), 10), np.float32); p[:, m.classes_] = m.predict_proba(X[test])
    q = np.load(f'{D}/{T}_{feat}_{name}_test.npy')
    print(T, feat, name, 'iter', m.best_iteration_, 'maxabs', float(np.abs(p-q).max()), 'argmax agree', float((p.argmax(1)==q.argmax(1)).mean()))
