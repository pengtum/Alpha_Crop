"""
检验 3：跨境损失分解的模型训练（见 ANALYSIS_PLAN_check3.md）。

对 11 个目标国 T × 分类特征 {ae, s2}，训练三个模型并保存它们在 T 的 pool 与 test 上的预测概率：
  loco     外国池（其余 10 国各 10,000 个）+ 类别平衡权重
  within   T 的 pool（90% 训练 / 10% 早停验证）+ 类别平衡权重
  loco_pm  外国池 + 重要性权重 π_T(y)/π_F(y)（训练先验改为 T 的真实先验）
事后修正（类别集合、先验、EM、k 个标签）全部在 analyze_check3.py 中基于这些概率完成。
输出：RESULTS_DIR/check3/{T}_{feat}_{model}_{pool|test}.npy（float32，n × 10）、{T}_split.npz
"""
import hashlib
import os
import time

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from common import COUNTRIES, RESULTS_DIR, RANDOM_STATE, LGBM_PARAMS, EARLY_STOPPING, load_dataset, feature_cols

N_FOREIGN_PER_COUNTRY = 10_000
FEATS = ['ae', 's2']
MODELS = ['loco', 'within', 'loco_pm']
PARAMS = {**LGBM_PARAMS, 'n_estimators': 500, 'class_weight': None}
OUT_DIR = os.path.join(RESULTS_DIR, 'check3')
PLAN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ANALYSIS_PLAN_check3.md')


def strat_split(idx, y, test_size, seed):
    """分层划分；样本数 <2 的类别不参与分层。
    这些样本原先单独成一组（-1），但当这一组本身也只有 1 个成员时（PT 的草莓）无法分层，
    所以改为并入最大类的分层组。这个改动只影响这类极少数样本落在训练还是早停验证中；
    AT–LV 在改动前已完成，未重跑。"""
    counts = np.bincount(y[idx], minlength=y.max() + 1)
    strat = np.where(counts[y[idx]] >= 2, y[idx], -1)
    if 0 < (strat == -1).sum() < 2:
        strat[strat == -1] = counts.argmax()
    return train_test_split(idx, test_size=test_size, random_state=seed, stratify=strat)


def fit(X, y, tr, va, w_tr, w_va, n_classes):
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(X[tr], y[tr], sample_weight=w_tr, eval_set=[(X[va], y[va])], eval_sample_weight=[w_va],
          callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)])
    return m


def full_proba(m, X, n_classes):
    """把模型只见过的类别的概率放回 10 类的位置，未见过的类概率为 0。"""
    p = np.zeros((len(X), n_classes), np.float32)
    p[:, m.classes_] = m.predict_proba(X)
    return p


def main(targets=COUNTRIES, feats=FEATS):
    with open(PLAN, 'rb') as f:
        print(f'plan sha256[:16] = {hashlib.sha256(f.read()).hexdigest()[:16]}', flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_dataset()
    classes = sorted(df['label'].unique())
    C = len(classes)
    y = df['label'].map({c: i for i, c in enumerate(classes)}).to_numpy()
    ctry = df['country'].to_numpy()
    Xs = {f: df[feature_cols(df, f)].to_numpy(np.float32) for f in feats}

    for tgt in targets:
        rng = np.random.default_rng(RANDOM_STATE)
        foreign = np.concatenate([
            rng.choice(np.flatnonzero(ctry == c), min(N_FOREIGN_PER_COUNTRY, int((ctry == c).sum())), replace=False)
            for c in COUNTRIES if c != tgt])
        t_idx = np.flatnonzero(ctry == tgt)
        t_idx = t_idx[~df.iloc[t_idx].duplicated(['longitude', 'latitude', 'area_ha']).to_numpy()]
        pool, test = train_test_split(t_idx, test_size=0.3, random_state=RANDOM_STATE)
        f_tr, f_va = strat_split(foreign, y, 0.1, RANDOM_STATE)
        p_tr, p_va = strat_split(pool, y, 0.1, RANDOM_STATE)
        np.savez(os.path.join(OUT_DIR, f'{tgt}_split.npz'), pool=pool, test=test, foreign=foreign,
                 y_pool=y[pool], y_test=y[test], classes=np.array(classes))

        pi_T = np.bincount(y[pool], minlength=C) / len(pool)
        pi_F = np.bincount(y[foreign], minlength=C) / len(foreign)
        ratio = np.divide(pi_T, pi_F, out=np.zeros(C), where=pi_F > 0)

        for feat in feats:
            X = Xs[feat]
            for model in MODELS:
                out = {s: os.path.join(OUT_DIR, f'{tgt}_{feat}_{model}_{s}.npy') for s in ('pool', 'test')}
                if all(os.path.exists(p) for p in out.values()):
                    continue
                t0 = time.time()
                if model == 'loco':
                    m = fit(X, y, f_tr, f_va, compute_sample_weight('balanced', y[f_tr]),
                            compute_sample_weight('balanced', y[f_va]), C)
                elif model == 'within':
                    m = fit(X, y, p_tr, p_va, compute_sample_weight('balanced', y[p_tr]),
                            compute_sample_weight('balanced', y[p_va]), C)
                else:
                    # 只保留在 T 中出现的类（权重为 0 的样本直接去掉）
                    keep_tr, keep_va = f_tr[ratio[y[f_tr]] > 0], f_va[ratio[y[f_va]] > 0]
                    m = fit(X, y, keep_tr, keep_va, ratio[y[keep_tr]], ratio[y[keep_va]], C)
                for s, idx in (('pool', pool), ('test', test)):
                    np.save(out[s], full_proba(m, X[idx], C))
                acc = (full_proba(m, X[test], C).argmax(1) == y[test]).mean()
                print(f'{tgt} {feat:2s} {model:8s} iters={m.best_iteration_:3d} test OA={acc:.3f} '
                      f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
