import os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion')
import numpy as np, lightgbm as lgb, pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from common import LGBM_PARAMS, EARLY_STOPPING, load_dataset, feature_cols, COUNTRIES
D='/Users/perng_mini/AlphaCrops_local/results/check3'
PARAMS = {**LGBM_PARAMS, 'n_estimators': 500, 'class_weight': None}
df = load_dataset(); classes = sorted(df['label'].unique()); C=10
y = df['label'].map({c: i for i, c in enumerate(classes)}).to_numpy(); ctry=df['country'].to_numpy()
R = np.load('replay_cm_idx.npz')
def fit(X, tr, va, wtr, wva):
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(X[tr], y[tr], sample_weight=wtr, eval_set=[(X[va], y[va])], eval_sample_weight=[wva],
          callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)])
    return m
for T, feat, name in [('RO','ae','loco_pm'), ('LT','s2','loco_pm'), ('AT','ae','within')]:
    sp=np.load(f'{D}/{T}_split.npz'); pool, test, foreign = sp['pool'], sp['test'], sp['foreign']
    X = df[feature_cols(df, feat)].to_numpy(np.float32)
    f_tr, f_va, p_tr, p_va = R[f'{T}_f_tr'], R[f'{T}_f_va'], R[f'{T}_p_tr'], R[f'{T}_p_va']
    if name=='loco_pm':
        pi_T=np.bincount(y[pool],minlength=C)/len(pool); pi_F=np.bincount(y[foreign],minlength=C)/len(foreign)
        ratio=np.divide(pi_T,pi_F,out=np.zeros(C),where=pi_F>0)
        ktr, kva = f_tr[ratio[y[f_tr]]>0], f_va[ratio[y[f_va]]>0]
        m=fit(X,ktr,kva,ratio[y[ktr]],ratio[y[kva]])
        print(T,'ratio',np.round(ratio,3), 'n_train',len(ktr))
    else:
        m=fit(X,p_tr,p_va,compute_sample_weight('balanced',y[p_tr]),compute_sample_weight('balanced',y[p_va]))
    p=np.zeros((len(test),C),np.float32); p[:,m.classes_]=m.predict_proba(X[test])
    q=np.load(f'{D}/{T}_{feat}_{name}_test.npy')
    print(T,feat,name,'iter',m.best_iteration_,'maxabs',float(np.abs(p-q).max()),'argmax agree',float((p.argmax(1)==q.argmax(1)).mean()))
# 基本划分核对
for T in COUNTRIES:
    sp=np.load(f'{D}/{T}_split.npz'); pool,test=sp['pool'],sp['test']
    t_idx=np.flatnonzero(ctry==T)
    dup=df.iloc[t_idx].duplicated(['longitude','latitude','area_ha']).sum()
    print(T,'pool∩test',len(np.intersect1d(pool,test)),'pool+test',len(pool)+len(test),'T all',len(t_idx),'dups removed',dup,
          'all in T',bool((ctry[pool]==T).all() and (ctry[test]==T).all()), 'foreign∩T',int((ctry[sp['foreign']]==T).sum()))
