"""
国家间无标签域距离（检验 1，见 ANALYSIS_PLAN_check1.md）。

输出：RESULTS_DIR/domain_distance.csv（每个无序国家对一行）
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from common import COUNTRIES, RESULTS_DIR, RANDOM_STATE, PROJECT_ROOT, load_dataset, feature_cols

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'experiments'))
from experiment_utils import COUNTRY_TO_ZONE  # noqa: E402

N_PER_COUNTRY = 3000
OUT = os.path.join(RESULTS_DIR, 'domain_distance.csv')


def rbf_mmd2(X, Y, gamma):
    """无偏 MMD² 估计。"""
    def k(A, B):
        d = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
        return np.exp(-gamma * np.maximum(d, 0))
    m, n = len(X), len(Y)
    kxx, kyy, kxy = k(X, X), k(Y, Y), k(X, Y)
    np.fill_diagonal(kxx, 0)
    np.fill_diagonal(kyy, 0)
    return kxx.sum() / (m * (m - 1)) + kyy.sum() / (n * (n - 1)) - 2 * kxy.mean()


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def main():
    df = load_dataset()
    rng = np.random.default_rng(RANDOM_STATE)
    samp = pd.concat([g.iloc[rng.choice(len(g), N_PER_COUNTRY, replace=False)]
                      for _, g in df.groupby('country')], ignore_index=True)
    classes = sorted(df['label'].unique())
    comp = {c: g['label'].value_counts(normalize=True).reindex(classes, fill_value=0).to_numpy()
            for c, g in df.groupby('country')}
    centroid = df.groupby('country')[['longitude', 'latitude']].mean()

    spaces = {}
    for feat in ['ae', 's2']:
        X = samp[feature_cols(samp, feat)].to_numpy(np.float64)
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        sub = X[rng.choice(len(X), 4000, replace=False)]
        sq = (sub ** 2).sum(1)
        d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * sub @ sub.T, 0)
        gamma = 1.0 / np.median(d2[np.triu_indices(len(sub), 1)])
        spaces[feat] = (X, gamma)

    ctry = samp['country'].to_numpy()
    rows = []
    for i, a in enumerate(COUNTRIES):
        for b in COUNTRIES[i + 1:]:
            r = dict(c1=a, c2=b)
            for feat, (X, gamma) in spaces.items():
                Xa, Xb = X[ctry == a], X[ctry == b]
                r[f'mmd_{feat}'] = rbf_mmd2(Xa, Xb, gamma)
                Z = np.vstack([Xa, Xb])
                y = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
                clf = LogisticRegression(max_iter=2000, C=1.0)
                r[f'c2st_{feat}'] = cross_val_score(clf, Z, y, cv=5, scoring='roc_auc').mean()
            r['geo_km'] = haversine_km(*centroid.loc[a], *centroid.loc[b])
            r['climate_diff'] = int(COUNTRY_TO_ZONE[a] != COUNTRY_TO_ZONE[b])
            r['class_js'] = jensenshannon(comp[a], comp[b])
            rows.append(r)
            print(a, b, {k: round(v, 4) for k, v in r.items() if k not in ('c1', 'c2')}, flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print('saved', OUT)


if __name__ == '__main__':
    main()
