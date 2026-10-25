"""
选样提升空间诊断（探索性，写于采样优化文献调研进行中，2026-09-24）。

问题：在目标国内怎么挑 k 个地块打标签，理论上最多能比随机好多少？
  random        简单随机，20 个种子（给出随机选样的分布；"20 次里测试最好的一次"是作弊上界）
  oracle_equal  按真实类别分层，各存在类等额分配（作弊：用了目标国标签）
  oracle_sqrt   按真实类别分层，名额 ∝ sqrt(类别占比)（作弊）
划分与外国池沿用 pilot_target_selection.setup_target（与检验 2 相同）。
分类特征 S2；目标国样本权重按"每样本权重倍数 r=10"（share = r·k/N_foreign，检验 2 权重诊断的推荐），
类别平衡权重先在全体训练样本上计算，再缩放目标国样本。
输出：RESULTS_DIR/headroom.csv，预测保存在 RESULTS_DIR/preds_headroom/
"""
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from common import RESULTS_DIR, load_dataset, feature_cols
from pilot_target_selection import TARGETS, PARAMS, setup_target, evaluate, save_rows

R_MULT = 10
BUDGETS = [50, 200]
N_RANDOM = 20
N_ORACLE = 5
FEAT = 's2'
OUT = os.path.join(RESULTS_DIR, 'headroom.csv')
PRED = os.path.join(RESULTS_DIR, 'preds_headroom')


def train(X, y, is_t, seed):
    w = compute_sample_weight('balanced', y)
    if is_t.any():
        share = R_MULT * is_t.sum() / (~is_t).sum()
        w[is_t] *= share * w[~is_t].sum() / w[is_t].sum()
    m = lgb.LGBMClassifier(**{**PARAMS, 'random_state': seed})
    return m.fit(X, y, sample_weight=w)


def oracle_pick(y_pool, k, rng, how):
    cls, cnt = np.unique(y_pool, return_counts=True)
    want = np.ones(len(cls)) if how == 'equal' else np.sqrt(cnt / cnt.sum())
    alloc = np.floor(want / want.sum() * k).astype(int)
    alloc = np.minimum(alloc, cnt)
    while alloc.sum() < k:  # 余数按缺口依次补给还有余量的类
        room = np.flatnonzero(alloc < cnt)
        alloc[room[np.argmax((want / want.sum() * k - alloc)[room])]] += 1
    return np.concatenate([rng.choice(np.flatnonzero(y_pool == c), a, replace=False)
                           for c, a in zip(cls, alloc) if a > 0])


def main():
    os.makedirs(PRED, exist_ok=True)
    df = load_dataset()
    classes = sorted(df['label'].unique())
    y = df['label'].map({c: i for i, c in enumerate(classes)}).to_numpy()
    ctry = df['country'].to_numpy()
    X = df[feature_cols(df, FEAT)].to_numpy(np.float32)
    rows = pd.read_csv(OUT).to_dict('records') if os.path.exists(OUT) else []
    done = {(r['target'], r['strategy'], r['budget'], r['seed']) for r in rows}
    for tgt in TARGETS:
        foreign, pool, test, _ = setup_target(df, ctry, tgt)
        jobs = [('foreign_only', 0, 0)]
        for k in BUDGETS:
            jobs += [('random', k, s) for s in range(N_RANDOM)]
            jobs += [(f'oracle_{h}', k, s) for h in ('equal', 'sqrt') for s in range(N_ORACLE)]
        for strat, k, seed in jobs:
            if (tgt, strat, k, seed) in done:
                continue
            t0 = time.time()
            rng = np.random.default_rng(1000 * seed + k)
            if strat == 'foreign_only':
                sel = np.array([], int)
            elif strat == 'random':
                sel = pool[rng.choice(len(pool), k, replace=False)]
            else:
                sel = pool[oracle_pick(y[pool], k, rng, strat.split('_')[1])]
            tr = np.r_[foreign, sel]
            is_t = np.r_[np.zeros(len(foreign), bool), np.ones(len(sel), bool)]
            m = train(X[tr], y[tr], is_t, seed)
            pred = m.predict(X[test])
            np.save(os.path.join(PRED, f'{tgt}_{strat}_{k}_{seed}.npy'), pred.astype(np.int8))
            rows.append(dict(target=tgt, strategy=strat, budget=k, seed=seed, oa=float((pred == y[test]).mean()),
                             n_classes_sel=len(np.unique(y[sel])), **evaluate(y[test], pred)))
            save_rows(rows, OUT)
            print(f'{tgt} {strat:13s} k={k:4d} s={seed:2d} {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
