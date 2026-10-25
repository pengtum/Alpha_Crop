"""
检验 3 事后补充：训练容量一致性（见 ANALYSIS_ADDENDUM_check3_capacity.md，非预注册）。

主链 6 个模型（loco、loco_pm、cmn、wnat、cmb、within）用与主分析完全相同的训练集
（f_tr、cm_tr、p_tr，按 check3_decomposition.py / check3_supp.py 的随机数顺序重放），
但一律固定 300 棵树、不早停。
输出：RESULTS_DIR/check3/{T}_{feat}_{model}_c300_{test|pool}.npy
"""
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from check3_decomposition import OUT_DIR, strat_split, full_proba
from check3_supp import count_matched, N_CALIB_PER_COUNTRY
from common import COUNTRIES, RANDOM_STATE, LGBM_PARAMS, load_dataset, feature_cols

PARAMS = {**LGBM_PARAMS, 'n_estimators': 300, 'class_weight': None}


def main():
    df = load_dataset()
    classes = sorted(df['label'].unique())
    C = len(classes)
    y = df['label'].map({c: i for i, c in enumerate(classes)}).to_numpy()
    ctry = df['country'].to_numpy()
    Xs = {f: df[feature_cols(df, f)].to_numpy(np.float32) for f in ['ae', 's2']}
    meta = pd.read_csv(os.path.join(OUT_DIR, 'supp_meta.csv'))
    for tgt in COUNTRIES:
        sp = np.load(os.path.join(OUT_DIR, f'{tgt}_split.npz'))
        foreign, pool, test = sp['foreign'], sp['pool'], sp['test']
        f_tr, f_va = strat_split(foreign, y, 0.1, RANDOM_STATE)
        p_tr, p_va = strat_split(pool, y, 0.1, RANDOM_STATE)
        # 与 check3_supp.py 相同的随机数顺序：先抽 calib，再抽 cm_tr
        extra = np.setdiff1d(np.flatnonzero(ctry != tgt), foreign)
        rng = np.random.default_rng(RANDOM_STATE + 1)
        calib = np.concatenate([rng.choice(extra[ctry[extra] == c], N_CALIB_PER_COUNTRY, replace=False)
                                for c in COUNTRIES if c != tgt])
        assert np.array_equal(calib, np.load(os.path.join(OUT_DIR, f'{tgt}_calib.npz'))['calib'])
        cm_tr, _ = count_matched(f_tr, np.setdiff1d(extra, calib), np.bincount(y[p_tr], minlength=C), y, rng)
        n_cm = meta[(meta.target == tgt) & (meta.model == 'cmb')].n_train.unique()
        assert len(n_cm) == 1 and n_cm[0] == len(cm_tr), (tgt, n_cm, len(cm_tr))
        pi_T = np.bincount(y[pool], minlength=C) / len(pool)
        pi_F = np.bincount(y[foreign], minlength=C) / len(foreign)
        ratio = np.divide(pi_T, pi_F, out=np.zeros(C), where=pi_F > 0)
        pm_tr = f_tr[ratio[y[f_tr]] > 0]
        jobs = [('loco', f_tr, compute_sample_weight('balanced', y[f_tr])),
                ('loco_pm', pm_tr, ratio[y[pm_tr]]),
                ('cmn', cm_tr, None),
                ('wnat', p_tr, None),
                ('cmb', cm_tr, compute_sample_weight('balanced', y[cm_tr])),
                ('within', p_tr, compute_sample_weight('balanced', y[p_tr]))]
        for feat in ['ae', 's2']:
            X = Xs[feat]
            for name, tr, w in jobs:
                path = os.path.join(OUT_DIR, f'{tgt}_{feat}_{name}_c300_test.npy')
                if os.path.exists(path):
                    continue
                t0 = time.time()
                m = lgb.LGBMClassifier(**PARAMS).fit(X[tr], y[tr], sample_weight=w)
                np.save(path, full_proba(m, X[test], C))
                if name == 'loco':
                    np.save(os.path.join(OUT_DIR, f'{tgt}_{feat}_{name}_c300_pool.npy'), full_proba(m, X[pool], C))
                print(f'{tgt} {feat} {name:8s} n_train={len(tr):,} {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
