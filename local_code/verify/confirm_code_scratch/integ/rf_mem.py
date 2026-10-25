"""
RF 峰值内存实测（方案 §2.3、§7 第 3 步）：旧 G 划分的 1 个单元上，按 confirm_run 的代码路径完整训练一次 500 棵 RF
（外国 5 万 + π_0 前 200 个，RT200 r0），并预测测试 + 候选池行。不计算任何指标，不保存概率。
用法：/usr/bin/time -l /Users/perng_mini/AlphaCrops_local/.venv/bin/python -B rf_mem.py AT 0 s2
"""
import os
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion/confirm')
import confirm_run as CR  # noqa: E402  （导入前设线程环境变量 = 2）
import time  # noqa: E402
import numpy as np  # noqa: E402

tgt, split, feat = sys.argv[1], int(sys.argv[2]), sys.argv[3]
CS, G, ML = CR.CS, CR.G, CR.ML
D = G.GData()
sp, sel = CS.build_smoke_unit(D, tgt, split)
ML.PARAMS['rf'] = dict(CR.RF_PARAMS, n_estimators=CR.TREES['rf'])
fi, ti, seed_str = CR.train_rows('RT200', 0, sp, sel)
y = np.asarray(D.y)
cw = CR.class_weights_from(y[sp['foreign']])
X = D.X[feat]
X_tr = np.vstack([np.asarray(X[fi]), np.asarray(X[ti])])
X_te = np.vstack([np.asarray(X[sp['test']]), np.asarray(X[sp['pool']])])
print(f'{tgt}_s{split} {feat}：n_train {len(X_tr)}（外国 {len(fi)} + 目标 {len(ti)}），n_pred {len(X_te)}，'
      f'特征 {X_tr.shape[1]}；拟合前 RSS {CR.peak_rss_mb()} MB', flush=True)
t0 = time.time()
P = CR.fit_predict('rf', X_tr, y[np.r_[fi, ti]], CR.sample_weights('RT200', y[fi], y[ti], cw), X_te,
                   CS.seed_of(seed_str), CR.MAX_THREADS, CR.TREES['rf'])
print(f'RF {CR.TREES["rf"]} 棵 n_jobs={CR.MAX_THREADS}：拟合 + 预测 {time.time() - t0:.1f}s；P {P.shape}；'
      f'进程峰值 RSS（getrusage）{CR.peak_rss_mb()} MB', flush=True)
