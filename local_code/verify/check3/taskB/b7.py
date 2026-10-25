"""检验 3 的 LOCO 配置（同一外国池 f_tr、同一类别平衡权重），只改树的数量：
A = 原配置（早停，复现 L0 概率）；B = 固定 300 棵、不早停（检验 2 的做法）。"""
import sys, os, time, numpy as np, warnings; warnings.filterwarnings('ignore')
CODE = "/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion"
sys.path.insert(0, CODE)
import lightgbm as lgb
from sklearn.utils.class_weight import compute_sample_weight
from common import load_dataset, feature_cols, LGBM_PARAMS, EARLY_STOPPING, RANDOM_STATE
import check3_decomposition as c3
OUT = 'b7_probs'; os.makedirs(OUT, exist_ok=True)
df = load_dataset(); classes = sorted(df['label'].unique())
y = df['label'].map({c:i for i,c in enumerate(classes)}).to_numpy()
X = {f: df[feature_cols(df, f)].to_numpy(np.float32) for f in ['ae','s2']}
D = '/Users/perng_mini/AlphaCrops_local/results/check3'
for t in sys.argv[1].split(','):
    sp = np.load(f'{D}/{t}_split.npz'); foreign, pool, test = sp['foreign'], sp['pool'], sp['test']
    f_tr, f_va = c3.strat_split(foreign, y, 0.1, RANDOM_STATE)
    for f in sys.argv[2].split(','):
        for variant in ('A', 'B'):
            path = f'{OUT}/{t}_{f}_{variant}.npz'
            if os.path.exists(path): continue
            t0 = time.time()
            w_tr = compute_sample_weight('balanced', y[f_tr])
            if variant == 'A':
                m = lgb.LGBMClassifier(**c3.PARAMS)
                m.fit(X[f][f_tr], y[f_tr], sample_weight=w_tr, eval_set=[(X[f][f_va], y[f_va])],
                      eval_sample_weight=[compute_sample_weight('balanced', y[f_va])],
                      callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)])
                it = m.best_iteration_
            else:
                m = lgb.LGBMClassifier(**{**LGBM_PARAMS, 'n_estimators': 300, 'class_weight': None})
                m.fit(X[f][f_tr], y[f_tr], sample_weight=w_tr); it = 300
            P = {}
            for s, idx in (('pool', pool), ('test', test)):
                p = np.zeros((len(idx), 10)); p[:, m.classes_] = m.predict_proba(X[f][idx]); P[s] = p
            ref = np.load(f'{D}/{t}_{f}_loco_test.npy')
            dif = float(np.abs(P['test'] - ref).max())
            np.savez_compressed(path, pool=P['pool'], test=P['test'], iters=it, maxdiff_vs_loco=dif)
            print(f'{t} {f} {variant}: iters={it} test OA={(P["test"].argmax(1)==y[test]).mean():.4f} '
                  f'与 check3 loco_test 最大|差|={dif:.2e} {time.time()-t0:.0f}s', flush=True)
