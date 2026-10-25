"""提升空间诊断的分析（探索性）。best-of-M：把测试集随机对半分为 V 与 E'，在 V 上挑最好的随机子集、在 E' 上报告。"""
import os
import sys

import numpy as np
import pandas as pd

from common import RESULTS_DIR, load_dataset
from pilot_target_selection import setup_target

MIN_N = 30


def ba_ge30(y, p, big):
    return 100 * np.mean([(p[y == c] == c).mean() for c in big])


def main():
    df = load_dataset()
    classes = sorted(df['label'].unique())
    y_all = df['label'].map({c: i for i, c in enumerate(classes)}).to_numpy()
    ctry = df['country'].to_numpy()
    H = pd.read_csv(os.path.join(RESULTS_DIR, 'headroom.csv'))
    P = os.path.join(RESULTS_DIR, 'preds_headroom')
    out = []
    for tgt in H.target.unique():
        _, _, test, _ = setup_target(df, ctry, tgt)
        y = y_all[test]
        half = np.random.default_rng(0).permutation(len(y)) < len(y) // 2
        for k in (50, 200):
            ran = H[(H.target == tgt) & (H.strategy == 'random') & (H.budget == k)]
            if len(ran) < 20:
                continue
            preds = {s: np.load(os.path.join(P, f'{tgt}_random_{k}_{s}.npy')) for s in ran.seed}
            res = {}
            for metric in ('oa', 'ba_ge30'):
                def score(p, m):
                    yy, pp = y[m], p[m]
                    if metric == 'oa':
                        return 100 * (yy == pp).mean()
                    n = np.bincount(yy, minlength=10)
                    return ba_ge30(yy, pp, np.flatnonzero(n >= MIN_N))
                sV = {s: score(p, half) for s, p in preds.items()}
                sE = {s: score(p, ~half) for s, p in preds.items()}
                best = max(sV, key=sV.get)
                arr = np.array(list(sE.values()))
                res[metric] = dict(med=np.median(arr), p90=np.percentile(arr, 90), mx=arr.max(),
                                   bestV_on_E=sE[best], sd=arr.std(ddof=1))
                for o in ('oracle_equal', 'oracle_sqrt'):
                    oo = H[(H.target == tgt) & (H.strategy == o) & (H.budget == k)]
                    col = 'oa' if metric == 'oa' else 'ba_ge30'
                    res[metric][o] = 100 * oo[col].mean() if len(oo) else np.nan
                res[metric]['random_mean_full'] = 100 * ran[col].mean()
            fo = H[(H.target == tgt) & (H.strategy == 'foreign_only')]
            for metric, r in res.items():
                out.append(dict(target=tgt, k=k, metric=metric,
                                foreign_only=100 * fo['oa' if metric == 'oa' else 'ba_ge30'].iloc[0],
                                random_mean=r['random_mean_full'], random_sd=r['sd'],
                                E_median=r['med'], E_p90=r['p90'], E_max=r['mx'], bestV_on_E=r['bestV_on_E'],
                                U2_gain=r['bestV_on_E'] - r['med'],
                                oracle_equal_gain=r['oracle_equal'] - r['random_mean_full'],
                                oracle_sqrt_gain=r['oracle_sqrt'] - r['random_mean_full']))
    R = pd.DataFrame(out)
    pd.set_option('display.width', 250)
    print(R.round(2).to_string(index=False))
    R.to_csv(os.path.join(RESULTS_DIR, 'headroom_summary.csv'), index=False)


if __name__ == '__main__':
    main()
