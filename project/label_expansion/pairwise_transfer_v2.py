"""
v2 数据上的两两国家迁移矩阵（11 源 × 10 目标 × 3 特征集）。

每个源国按 70/10/20 分层划分为训练 / 早停验证 / 国内测试；在全部其他国家的全部地块上评估。
三种特征集用同一批地块（AE 与 S2 都有的地块）。不含坐标与面积。
输出：RESULTS_DIR/pairwise_v2.csv（source, target, feature, ba, n_test），target==source 为国内测试。
"""
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import (COUNTRIES, RESULTS_DIR, RANDOM_STATE, LGBM_PARAMS, EARLY_STOPPING,
                    load_dataset, feature_cols, balanced_accuracy_present)

OUT = os.path.join(RESULTS_DIR, 'pairwise_v2.csv')


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = load_dataset()
    classes = sorted(df['label'].unique())
    y_all = df['label'].map({c: i for i, c in enumerate(classes)}).to_numpy()
    done = set()
    rows = []
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        rows = prev.to_dict('records')
        done = set(zip(prev['source'], prev['feature']))

    for feat in ['ae', 's2', 'fusion']:
        X_all = df[feature_cols(df, feat)].to_numpy(np.float32)
        for src in COUNTRIES:
            if (src, feat) in done:
                continue
            t0 = time.time()
            idx = np.flatnonzero(df['country'].to_numpy() == src)
            y = y_all[idx]
            # 极少数类（<3 个）无法分层，归入同一分层组后照常切分
            counts = np.bincount(y, minlength=len(classes))
            strat = np.where(counts[y] >= 3, y, -1)
            tr, te = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE, stratify=strat)
            strat_tr = np.where(np.bincount(y_all[tr], minlength=len(classes))[y_all[tr]] >= 3, y_all[tr], -1)
            tr, va = train_test_split(tr, test_size=0.125, random_state=RANDOM_STATE, stratify=strat_tr)
            m = lgb.LGBMClassifier(**LGBM_PARAMS)
            m.fit(X_all[tr], y_all[tr], eval_set=[(X_all[va], y_all[va])],
                  callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)])
            # 源国训练集没有的类别在目标国无法被预测，按原实验口径仍计入目标国平衡精度
            rows.append(dict(source=src, target=src, feature=feat,
                             ba=balanced_accuracy_present(y_all[te], m.predict(X_all[te])), n_test=len(te)))
            for tgt in COUNTRIES:
                if tgt == src:
                    continue
                ti = np.flatnonzero(df['country'].to_numpy() == tgt)
                rows.append(dict(source=src, target=tgt, feature=feat,
                                 ba=balanced_accuracy_present(y_all[ti], m.predict(X_all[ti])), n_test=len(ti)))
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f'{feat:6s} {src}  iters={m.best_iteration_}  {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
