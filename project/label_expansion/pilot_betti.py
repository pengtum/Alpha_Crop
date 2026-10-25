"""
检验 2 补充：按 Betti et al.（Mapping on a Budget）的分组代表性效用选样。

与 pilot_target_selection.py 使用完全相同的外国池、目标国候选池与测试集（复用其 setup_target），
以及相同的训练函数（类别平衡 + 目标国 25% 份额，v1 修正）与评估指标；随机基线直接用主试点的 random。

成本模型（UNIT）：
  'parcel'  目视解译：采样单位是单个地块，每个成本 1（当前使用，2026-09-24 作者确定）
  'cluster' 实地调查：采样单位是约 10 km 网格簇，每簇随机标 M 个地块（保留，以后用）

分组（γ_g 为组 g 在候选池中的占比）：
  admin_rep    Admin-Rep：目标国的 NUTS3 行政区（Eurostat NUTS 2021）
  img_rep_ae   Image-Rep：AE 空间（PCA 32 维）k-means 分 8 组（Betti 用 8 组）
  img_rep_s2   Image-Rep：S2 空间（PCA 32 维）k-means 分 8 组
  （cluster 模式另有 rand_cluster 随机选簇基线）

效用函数（Betti 式 (2)，λ = 0.5）：
  U(n) = −λ Σ_g γ_g (n_g + 1)^(−1/2) − (1 − λ)(n + 1)^(−1/2)
目标国内没有初始样本，所以对 n_g 加 1 平滑。原文用 cvxpy + MOSEK 求连续松弛，这里用贪心：
parcel 模式每步给边际增益最大的组加 1 个名额（组内容量为上限），然后在各组内随机抽取；
cluster 模式每步选边际增益最大的簇。
注：parcel 模式下统一成本、没有初始样本，n 固定为预算，λ 只起缩放作用，不影响分配；
此时最优分配近似满足 n_g + 1 ∝ γ_g^(2/3)。
输出：RESULTS_DIR/pilot_betti_{UNIT}.csv（可断点续跑，原子写入）；逐地块预测在 RESULTS_DIR/preds/
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from common import DATA_DIR, RESULTS_DIR, RANDOM_STATE, load_dataset, feature_cols
from pilot_target_selection import (TARGETS, BUDGETS, SEEDS, CLF_FEATS, PRED_DIR,
                                    setup_target, train, evaluate, save_rows, plan_hash)

UNIT = os.environ.get('BETTI_UNIT', 'parcel')
N_GROUPS = 8
LAMBDA = 0.5
CELL_KM = 10.0
M_PER_CLUSTER = 10
CLUSTER_BUDGETS = [5, 20, 100]
STRATEGIES = {'parcel': ['admin_rep', 'img_rep_ae', 'img_rep_s2'],
              'cluster': ['rand_cluster', 'admin_rep', 'img_rep_ae', 'img_rep_s2']}[UNIT]
OUT = os.path.join(RESULTS_DIR, f'pilot_betti_{UNIT}.csv')  # 簇模式用环境变量 BETTI_UNIT=cluster


def utility(n_g, gamma):
    return -LAMBDA * np.sum(gamma * (n_g + 1) ** -0.5) - (1 - LAMBDA) * (n_g.sum() + 1) ** -0.5


def allocate_parcels(gamma, capacity, budget):
    """逐个名额贪心分配到各组，返回每组名额 n_g。"""
    n_g = np.zeros(len(gamma))
    for _ in range(min(budget, int(capacity.sum()))):
        gain = np.where(n_g < capacity, gamma * ((n_g + 1) ** -0.5 - (n_g + 2) ** -0.5), -np.inf)
        n_g[int(np.argmax(gain))] += 1
    return n_g.astype(int)


def greedy_clusters(E, gamma, budget):
    """E: 簇 × 组 的期望样本数矩阵。贪心选 budget 个簇使效用最大。"""
    chosen, n_g = [], np.zeros(E.shape[1])
    avail = np.ones(len(E), bool)
    for _ in range(min(budget, len(E))):
        gains = np.where(avail, [utility(n_g + E[c], gamma) for c in range(len(E))], -np.inf)
        c = int(np.argmax(gains))
        chosen.append(c)
        avail[c] = False
        n_g = n_g + E[c]
    return chosen


def main():
    print(f'plan sha256[:16] = {plan_hash()}', flush=True)
    os.makedirs(PRED_DIR, exist_ok=True)
    df = load_dataset()
    nuts = pd.read_parquet(os.path.join(DATA_DIR, 'parcel_nuts.parquet'), columns=['parcel_id', 'nuts3'])
    df = df.merge(nuts, on='parcel_id', how='left', validate='one_to_one')
    assert df['nuts3'].notna().all()
    classes = sorted(df['label'].unique())
    y_all = df['label'].map({c: i for i, c in enumerate(classes)}).to_numpy()
    ctry = df['country'].to_numpy()
    X = {f: df[feature_cols(df, f)].to_numpy(np.float32) for f in ['ae', 's2']}
    lon, lat = df['longitude'].to_numpy(), df['latitude'].to_numpy()

    done, rows = set(), []
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        rows = prev.to_dict('records')
        done = set(zip(prev.target, prev.strategy, prev.budget, prev.seed, prev.clf_feat))

    for tgt in TARGETS:
        # 与 pilot_target_selection.py 完全相同的外国池与候选池/测试集
        foreign, pool, test, _ = setup_target(df, ctry, tgt)

        groups = {'admin_rep': pd.factorize(df['nuts3'].to_numpy()[pool])[0]}
        for f in ['ae', 's2']:
            Z = np.vstack([X[f][foreign], X[f][pool]]).astype(np.float64)
            Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
            Z = PCA(32, random_state=RANDOM_STATE).fit_transform(Z)[len(foreign):]
            groups[f'img_rep_{f}'] = KMeans(N_GROUPS, n_init=4, random_state=RANDOM_STATE).fit_predict(Z)
        print(f'{tgt}: pool {len(pool):,}; NUTS3 groups {groups["admin_rep"].max() + 1}', flush=True)

        if UNIT == 'cluster':
            lat0 = lat[pool].mean()
            gx = np.floor(lon[pool] * 111.32 * np.cos(np.radians(lat0)) / CELL_KM).astype(int)
            gy = np.floor(lat[pool] * 110.57 / CELL_KM).astype(int)
            cell = pd.factorize(pd.Series(gx).astype(str) + '_' + pd.Series(gy).astype(str))[0]
            members = [np.flatnonzero(cell == c) for c in range(cell.max() + 1)]
            sizes = np.array([len(m) for m in members])
            chosen = {}
            for s, g in groups.items():
                comp = np.zeros((len(members), g.max() + 1))
                np.add.at(comp, (cell, g), 1)
                E = comp / sizes[:, None] * np.minimum(M_PER_CLUSTER, sizes)[:, None]
                chosen[s] = greedy_clusters(E, np.bincount(g) / len(g), max(CLUSTER_BUDGETS))
            budgets = CLUSTER_BUDGETS
        else:
            budgets = BUDGETS

        for b in budgets:
            for seed in SEEDS:
                r = np.random.default_rng(20_000 + 1000 * seed + b)
                for strat in STRATEGIES:
                    if UNIT == 'parcel':
                        g = groups[strat]
                        cap = np.bincount(g)
                        n_g = allocate_parcels(cap / len(g), cap, b)
                        pick = np.concatenate([r.choice(np.flatnonzero(g == k), n_g[k], replace=False)
                                               for k in range(len(cap)) if n_g[k] > 0])
                    else:
                        cl = (r.choice(len(members), min(b, len(members)), replace=False)
                              if strat == 'rand_cluster' else chosen[strat][:b])
                        pick = np.concatenate([r.choice(members[c], min(M_PER_CLUSTER, sizes[c]), replace=False)
                                               for c in cl])
                    sel = pool[pick]
                    for cf in CLF_FEATS:
                        if (tgt, strat, b, seed, cf) in done:
                            continue
                        t0 = time.time()
                        tr = np.r_[foreign, sel]
                        is_t = np.r_[np.zeros(len(foreign), bool), np.ones(len(sel), bool)]
                        m, share = train(X[cf][tr], y_all[tr], is_t, seed)
                        pred = m.predict(X[cf][test])
                        np.save(os.path.join(PRED_DIR, f'{tgt}_betti-{UNIT}-{strat}_{b}_{seed}_{cf}.npy'),
                                pred.astype(np.int8))
                        rows.append(dict(target=tgt, strategy=strat, unit=UNIT, budget=b, n_labels=len(sel),
                                         seed=seed, clf_feat=cf, eff_share=share,
                                         n_classes_sel=len(np.unique(y_all[sel])), **evaluate(y_all[test], pred)))
                        save_rows(rows, OUT)
                        print(f'{tgt} b={b:4d} s={seed} {strat:11s} {cf}: ba={rows[-1]["ba"]:.3f} '
                              f'ba_ge30={rows[-1]["ba_ge30"]:.3f} (labels {len(sel)}, '
                              f'classes {rows[-1]["n_classes_sel"]}) {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
