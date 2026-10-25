"""任务 B-4 补充：按检验 2 的配置复训 foreign_only 与 random-k 模型，拿到概率，
在同一模型、同一批 k 个标签上比较"标签用于先验"与"标签用于训练"（以及两者叠加）。"""
import sys, os, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
CODE = "/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion"
sys.path.insert(0, CODE)
from common import load_dataset, feature_cols
import pilot_target_selection as pts
OUT = 'b5_probs'; os.makedirs(OUT, exist_ok=True)
df = load_dataset()
classes = sorted(df['label'].unique())
y = df['label'].map({c:i for i,c in enumerate(classes)}).to_numpy(); ctry = df['country'].to_numpy()
X = {f: df[feature_cols(df, f)].to_numpy(np.float32) for f in ['ae','s2']}
targets = sys.argv[1].split(',') if len(sys.argv)>1 else ['PT','SE','SK','LV']
for t in targets:
    foreign, pool, test, _ = pts.setup_target(df, ctry, t)
    sp = np.load(f'/Users/perng_mini/AlphaCrops_local/results/check3/{t}_split.npz')
    assert (pool==sp['pool']).all() and (test==sp['test']).all(), '检验 2/3 的 pool/test 不同'
    f_only = np.zeros(len(foreign), bool)
    for f in ['ae','s2']:
        jobs = [('foreign_only', 0, s) for s in range(3)] + [('random', k, s) for k in (50,200) for s in range(3)]
        for strat, k, s in jobs:
            path = f'{OUT}/{t}_{strat}_{k}_{s}_{f}.npz'
            if os.path.exists(path): continue
            t0 = time.time()
            if strat == 'foreign_only':
                tr, is_t, lab = foreign, f_only, np.array([], int)
            else:
                lab = np.random.default_rng(1000*s + k).choice(len(pool), k, replace=False)
                tr = np.r_[foreign, pool[lab]]; is_t = np.r_[f_only, np.ones(k, bool)]
            m, share = pts.train(X[f][tr], y[tr], is_t, s)
            pt = np.zeros((len(test), 10)); pt[:, m.classes_] = m.predict_proba(X[f][test])
            pp = np.zeros((len(pool), 10)); pp[:, m.classes_] = m.predict_proba(X[f][pool])
            saved = np.load(f'/Users/perng_mini/AlphaCrops_local/results/preds/{t}_{strat}_{k}_{s}_{f}.npy')
            agree = (m.predict(X[f][test]) == saved).mean()
            np.savez_compressed(path, test=pt.astype(np.float32), pool=pp.astype(np.float32), lab=lab, agree=agree, share=share)
            print(f'{t} {f} {strat} k={k} s={s}: 与已存预测一致率 {agree:.5f}  {time.time()-t0:.0f}s', flush=True)
