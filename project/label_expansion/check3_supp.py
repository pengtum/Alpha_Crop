"""
检验 3 补充训练（ANALYSIS_PLAN_check3_rev1.md）：训练时链式分解所需的模型。

对 11 个目标国 T × {ae, s2}：
  wnat     T 的 p_tr，不加权（OA 链的上界）
  cmb/cmn  外国地块按 T 的 p_tr 各类计数抽样（早停集按 p_va 计数），类别平衡 / 不加权
  lcal     重跑 LOCO（deterministic），另存外国校准集的预测，用于 BCTS 与 BBSE
  w0_sp / wnat_sp  空间分块划分（约 20 km 网格，30% 的格子作 test_sp）下的本国模型
划分与 check3_decomposition.py 完全一致（外国池、pool/test 从 {T}_split.npz 读取并核对）。
不打印任何精度，只记录迭代次数与抽样短缺量。
输出：RESULTS_DIR/check3/ 下的 *.npy，以及 {T}_split_sp.npz、{T}_calib.npz、supp_meta.csv
"""
import hashlib
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from check3_decomposition import OUT_DIR, N_FOREIGN_PER_COUNTRY, strat_split, full_proba
from common import COUNTRIES, RANDOM_STATE, LGBM_PARAMS, EARLY_STOPPING, load_dataset, feature_cols

FEATS = ['ae', 's2']
PARAMS = {**LGBM_PARAMS, 'n_estimators': 500, 'class_weight': None}
N_CALIB_PER_COUNTRY = 1000
CELL_KM = 20.0
PLANS = ['ANALYSIS_PLAN_check3.md', 'ANALYSIS_PLAN_check3_rev1.md']
HERE = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(OUT_DIR, 'supp_meta.csv')


def fit(X, y, tr, va, weighted, extra=None):
    w_tr = compute_sample_weight('balanced', y[tr]) if weighted else None
    w_va = compute_sample_weight('balanced', y[va]) if weighted else None
    m = lgb.LGBMClassifier(**{**PARAMS, **(extra or {})})
    m.fit(X[tr], y[tr], sample_weight=w_tr, eval_set=[(X[va], y[va])],
          eval_sample_weight=[w_va] if weighted else None,
          callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)])
    return m


def count_matched(pref, extra, counts, y, rng):
    """按 counts 各类计数抽样：先从 pref 抽，不足时从 extra 补，仍不足则全取。返回下标与短缺量。"""
    out, short = [], {}
    for c in np.flatnonzero(counts):
        a = pref[y[pref] == c]
        if len(a) >= counts[c]:
            out.append(rng.choice(a, counts[c], replace=False))
            continue
        b = extra[y[extra] == c]
        need = counts[c] - len(a)
        take = rng.choice(b, min(need, len(b)), replace=False)
        out += [a, take]
        short[int(c)] = dict(need=int(counts[c]), from_pref=int(len(a)), topup=int(len(take)),
                             missing=int(need - len(take)))
    return np.concatenate(out), short


def save(path, arr):
    tmp = path + '.tmp.npy'
    np.save(tmp, arr)
    os.replace(tmp, path)


def main(targets=COUNTRIES, feats=FEATS):
    for p in PLANS:
        with open(os.path.join(HERE, p), 'rb') as f:
            print(f'{p} sha256[:16] = {hashlib.sha256(f.read()).hexdigest()[:16]}', flush=True)
    df = load_dataset()
    classes = sorted(df['label'].unique())
    C = len(classes)
    y = df['label'].map({c: i for i, c in enumerate(classes)}).to_numpy()
    ctry = df['country'].to_numpy()
    lon, lat = df['longitude'].to_numpy(), df['latitude'].to_numpy()
    Xs = {f: df[feature_cols(df, f)].to_numpy(np.float32) for f in feats}
    meta = pd.read_csv(META).to_dict('records') if os.path.exists(META) else []

    for tgt in targets:
        sp = np.load(os.path.join(OUT_DIR, f'{tgt}_split.npz'))
        foreign, pool, test = sp['foreign'], sp['pool'], sp['test']
        # 核对：与 check3_decomposition.py 相同的外国池
        rng0 = np.random.default_rng(RANDOM_STATE)
        chk = np.concatenate([
            rng0.choice(np.flatnonzero(ctry == c), min(N_FOREIGN_PER_COUNTRY, int((ctry == c).sum())), replace=False)
            for c in COUNTRIES if c != tgt])
        assert np.array_equal(chk, foreign) and np.array_equal(y[pool], sp['y_pool'])
        f_tr, f_va = strat_split(foreign, y, 0.1, RANDOM_STATE)
        p_tr, p_va = strat_split(pool, y, 0.1, RANDOM_STATE)
        extra = np.setdiff1d(np.flatnonzero((ctry != tgt)), foreign)  # 未进入外国池的外国地块
        rng = np.random.default_rng(RANDOM_STATE + 1)
        calib = np.concatenate([rng.choice(extra[ctry[extra] == c], N_CALIB_PER_COUNTRY, replace=False)
                                for c in COUNTRIES if c != tgt])
        extra_cm = np.setdiff1d(extra, calib)
        cm_tr, short_tr = count_matched(f_tr, extra_cm, np.bincount(y[p_tr], minlength=C), y, rng)
        cm_va, short_va = count_matched(f_va, np.setdiff1d(extra_cm, cm_tr),
                                        np.bincount(y[p_va], minlength=C), y, rng)
        # 空间分块：约 20 km 网格，随机 30% 的格子作 test_sp
        t_all = np.concatenate([pool, test])
        lat0 = lat[t_all].mean()
        gx = np.floor(lon[t_all] * 111.32 * np.cos(np.radians(lat0)) / CELL_KM).astype(int)
        gy = np.floor(lat[t_all] * 110.57 / CELL_KM).astype(int)
        cell = pd.factorize(pd.Series(gx).astype(str) + '_' + pd.Series(gy).astype(str))[0]
        cells = np.random.default_rng(RANDOM_STATE).permutation(cell.max() + 1)
        test_cells = cells[:int(round(0.3 * len(cells)))]
        is_test = np.isin(cell, test_cells)
        pool_sp, test_sp = t_all[~is_test], t_all[is_test]
        sp_tr, sp_va = strat_split(pool_sp, y, 0.1, RANDOM_STATE)
        np.savez(os.path.join(OUT_DIR, f'{tgt}_split_sp.npz'), pool_sp=pool_sp, test_sp=test_sp,
                 y_pool_sp=y[pool_sp], y_test_sp=y[test_sp], n_cells=cell.max() + 1)
        np.savez(os.path.join(OUT_DIR, f'{tgt}_calib.npz'), calib=calib, y_calib=y[calib],
                 f_va=f_va, y_f_va=y[f_va], p_va=p_va, y_p_va=y[p_va])
        print(f'{tgt}: CM train {len(cm_tr):,} (short {short_tr})  CM val {len(cm_va):,}  '
              f'calib {len(calib):,}  test_sp {len(test_sp):,}/{len(t_all):,} in {len(test_cells)} cells',
              flush=True)

        for feat in feats:
            X = Xs[feat]
            jobs = [('wnat', p_tr, p_va, False, None, [('pool', pool), ('test', test), ('pva', p_va)]),
                    ('cmb', cm_tr, cm_va, True, None, [('pool', pool), ('test', test)]),
                    ('cmn', cm_tr, cm_va, False, None, [('pool', pool), ('test', test)]),
                    ('lcal', f_tr, f_va, True, {'deterministic': True, 'force_row_wise': True},
                     [('pool', pool), ('test', test), ('calib', calib), ('fva', f_va)]),
                    ('w0_sp', sp_tr, sp_va, True, None, [('testsp', test_sp)]),
                    ('wnat_sp', sp_tr, sp_va, False, None, [('testsp', test_sp)])]
            for name, tr, va, weighted, extra_p, outs in jobs:
                paths = {s: os.path.join(OUT_DIR, f'{tgt}_{feat}_{name}_{s}.npy') for s, _ in outs}
                if all(os.path.exists(p) for p in paths.values()):
                    continue
                t0 = time.time()
                m = fit(X, y, tr, va, weighted, extra_p)
                for s, idx in outs:
                    save(paths[s], full_proba(m, X[idx], C))
                if name == 'lcal':
                    m.booster_.save_model(os.path.join(OUT_DIR, f'{tgt}_{feat}_lcal.txt'))
                meta.append(dict(target=tgt, feat=feat, model=name, best_iter=m.best_iteration_,
                                 n_train=len(tr), n_val=len(va), seconds=round(time.time() - t0)))
                pd.DataFrame(meta).to_csv(META, index=False)
                print(f'  {tgt} {feat} {name:7s} iters={m.best_iteration_:3d} n_train={len(tr):,} '
                      f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
