"""
任务 B（事后探索，不属于预注册分析）：小预算下目标国标签的负增益是否来自份额设定。

对 LV、SE，random 策略（与主试点同样的选样），k∈{50,200,1000}、seed∈{0,1,2}、分类特征 ae/s2，
比较目标国权重份额 share∈{0.01,0.05,0.25,1.0} 与 'natural'（只做类别平衡、不缩放）。
训练自己实现（与 pilot_target_selection.train 同逻辑，只是份额可变）；划分与评估复用 CODE 中的函数。

用法：
  python taskB_share.py --repro   只跑 4 个 share=0.25 配置，与主试点 CSV/preds 逐行逐地块比对
  python taskB_share.py           跑全部 180 次（可断点续跑）
输出（本目录）：taskB_share.csv、taskB_recall.csv（逐类 recall）、preds_taskB/*.npy
"""
import os
import sys
import time

sys.dont_write_bytecode = True  # 不在只读的 CODE 目录里写 __pycache__
CODE = ('/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/'
        '我的云端硬盘/Research/Alpha_Crops/label_expansion')
sys.path.insert(0, CODE)

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402

from common import LGBM_PARAMS, RESULTS_DIR, load_dataset, feature_cols  # noqa: E402
from pilot_target_selection import setup_target, evaluate, PARAMS as PILOT_PARAMS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'taskB_share.csv')
OUT_REC = os.path.join(HERE, 'taskB_recall.csv')
PRED_DIR = os.path.join(HERE, 'preds_taskB')
MAIN_PRED = os.path.join(RESULTS_DIR, 'preds')

TARGETS = ['LV', 'SE']
BUDGETS = [50, 200, 1000]
SEEDS = [0, 1, 2]
CLF_FEATS = ['ae', 's2']
SHARES = [0.25, 0.01, 0.05, 1.0, 'natural']  # 0.25 放最前，便于先核对复现

PARAMS = {**LGBM_PARAMS, 'n_estimators': 300, 'n_jobs': 8}
PARAMS.pop('class_weight')
# 与主试点参数一致性核对（主试点是 class_weight=None，这里是删掉该键，二者等价）
_p = {k: v for k, v in PILOT_PARAMS.items() if k not in ('class_weight', 'n_jobs')}
_q = {k: v for k, v in PARAMS.items() if k != 'n_jobs'}
assert _p == _q, (_p, _q)
assert PILOT_PARAMS['class_weight'] is None

REPRO = [('LV', 50, 0, 'ae'), ('SE', 200, 1, 's2'), ('LV', 1000, 2, 's2'), ('SE', 50, 2, 'ae')]


def train_share(X, y, is_t, seed, share):
    w = compute_sample_weight('balanced', y)
    nat = w[is_t].sum() / w[~is_t].sum()
    if share != 'natural':
        w[is_t] *= share * w[~is_t].sum() / w[is_t].sum()
    m = lgb.LGBMClassifier(**{**PARAMS, 'random_state': seed})
    m.fit(X, y, sample_weight=w)
    eff = w[is_t].sum() / w[~is_t].sum()
    per_t = w[is_t].mean() / w[~is_t].mean()  # 目标国样本平均权重 / 外国样本平均权重
    return m, float(nat), float(eff), float(per_t)


def main():
    repro = '--repro' in sys.argv
    # --shard TGT CF：只跑一个 (目标国, 分类特征)，输出到各自的 CSV，便于多进程并行
    targets, feats, out, out_rec = TARGETS, CLF_FEATS, OUT, OUT_REC
    if '--shard' in sys.argv:
        i = sys.argv.index('--shard')
        targets, feats = [sys.argv[i + 1]], [sys.argv[i + 2]]
        out = OUT.replace('.csv', f'_{targets[0]}_{feats[0]}.csv')
        out_rec = OUT_REC.replace('.csv', f'_{targets[0]}_{feats[0]}.csv')
    # 补充检查用：--budgets 50,200 --shares 0.01,0.05（默认为全部）
    budgets, shares = BUDGETS, SHARES
    if '--budgets' in sys.argv:
        budgets = [int(v) for v in sys.argv[sys.argv.index('--budgets') + 1].split(',')]
    if '--shares' in sys.argv:
        shares = [v if v == 'natural' else float(v) for v in sys.argv[sys.argv.index('--shares') + 1].split(',')]
    os.makedirs(PRED_DIR, exist_ok=True)
    t0 = time.time()
    df = load_dataset()
    classes = sorted(df['label'].unique())
    y_all = df['label'].map({c: i for i, c in enumerate(classes)}).to_numpy()
    ctry = df['country'].to_numpy()
    X = {f: df[feature_cols(df, f)].to_numpy(np.float32) for f in feats}
    df = df[['longitude', 'latitude', 'area_ha']].copy()  # setup_target 只需这三列（去重用），省内存
    print(f'loaded {len(df):,} rows in {time.time() - t0:.0f}s; classes={len(classes)}', flush=True)

    main_csv = pd.read_csv(os.path.join(RESULTS_DIR, 'pilot_selection.csv'))
    rows, rec_rows, done = [], [], set()
    if not repro and os.path.exists(out):
        prev = pd.read_csv(out, dtype={'share': str})
        rows = prev.to_dict('records')
        rec_rows = pd.read_csv(out_rec, dtype={'share': str}).to_dict('records')
        done = set(zip(prev.target, prev.budget, prev.seed, prev.clf_feat, prev.share.astype(str)))

    for tgt in targets:
        foreign, pool, test, _ = setup_target(df, ctry, tgt)
        saved = np.load(os.path.join(MAIN_PRED, f'{tgt}_test_idx.npy'))
        assert np.array_equal(saved, test), f'{tgt} test idx 与主试点不一致'
        y_test = y_all[test]
        n_test_c = pd.Series(y_test).value_counts()
        print(f'{tgt}: foreign {len(foreign):,} pool {len(pool):,} test {len(test):,}（test_idx 与主试点一致）',
              flush=True)
        f_only = np.zeros(len(foreign), bool)
        for cf in feats:
            for k in budgets:
                for seed in SEEDS:
                    if repro and (tgt, k, seed, cf) not in REPRO:
                        continue
                    pk = np.random.default_rng(1000 * seed + k).choice(len(pool), k, replace=False)
                    sel = pool[pk]
                    n_sel_c = pd.Series(y_all[sel]).value_counts()
                    tr = np.r_[foreign, sel]
                    is_t = np.r_[f_only, np.ones(k, bool)]
                    for share in (SHARES[:1] if repro else shares):
                        key = (tgt, k, seed, cf, str(share))
                        if key in done:
                            continue
                        t1 = time.time()
                        m, nat, eff, per_t = train_share(X[cf][tr], y_all[tr], is_t, seed, share)
                        pred = m.predict(X[cf][test]).astype(np.int8)
                        np.save(os.path.join(PRED_DIR, f'{tgt}_random_{k}_{seed}_{cf}_share{share}.npy'), pred)
                        ev = evaluate(y_test, pred)
                        row = dict(target=tgt, budget=k, seed=seed, clf_feat=cf, share=str(share),
                                   nat_share=nat, eff_share=eff, w_ratio=per_t,
                                   n_classes_sel=int(len(n_sel_c)), **ev, secs=round(time.time() - t1, 1))
                        if share == 0.25:
                            ref = main_csv[(main_csv.target == tgt) & (main_csv.strategy == 'random') &
                                           (main_csv.budget == k) & (main_csv.seed == seed) &
                                           (main_csv.clf_feat == cf)]
                            assert len(ref) == 1
                            ref = ref.iloc[0]
                            mp = np.load(os.path.join(MAIN_PRED, f'{tgt}_random_{k}_{seed}_{cf}.npy'))
                            row.update(main_ba_ge30=ref.ba_ge30, main_ba=ref.ba, main_f1=ref.macro_f1,
                                       main_ncls=ref.n_classes_sel,
                                       pred_mismatch=int((mp != pred).sum()))
                            print(f'  REPRO {tgt} k={k} s={seed} {cf}: ba_ge30 {ev["ba_ge30"]:.6f} vs 主试点 '
                                  f'{ref.ba_ge30:.6f}; ba {ev["ba"]:.6f} vs {ref.ba:.6f}; '
                                  f'n_cls {len(n_sel_c)} vs {ref.n_classes_sel}; 逐地块不一致 '
                                  f'{row["pred_mismatch"]}/{len(pred)}', flush=True)
                        rows.append(row)
                        for c in n_test_c.index:
                            mask = y_test == c
                            rec_rows.append(dict(target=tgt, budget=k, seed=seed, clf_feat=cf, share=str(share),
                                                 cls=classes[c], n_test=int(n_test_c[c]),
                                                 n_sel=int(n_sel_c.get(c, 0)),
                                                 recall=float((pred[mask] == c).mean()),
                                                 n_pred=int((pred == c).sum())))
                        print(f'{tgt} {cf} k={k:4d} s={seed} share={str(share):4s}: ba_ge30={ev["ba_ge30"]:.4f} '
                              f'nat={nat:.4f} eff={eff:.4f} w_ratio={per_t:.1f} {row["secs"]}s', flush=True)
                        if not repro:
                            pd.DataFrame(rows).to_csv(out + '.tmp', index=False)
                            os.replace(out + '.tmp', out)
                            pd.DataFrame(rec_rows).to_csv(out_rec + '.tmp', index=False)
                            os.replace(out_rec + '.tmp', out_rec)
    if repro:
        pd.DataFrame(rows).to_csv(os.path.join(HERE, 'taskB_repro.csv'), index=False)


if __name__ == '__main__':
    main()
