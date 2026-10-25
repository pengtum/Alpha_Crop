"""树数对 AE/S2 指导对比的影响（小规模验证，不属于任何预注册分析）。

沿用检验 2 的划分与策略（ae_rep/s2_rep/ae_gap/s2_gap/random，已看过的设置），但训练改用阶段 A / G 实验的
配置：S2 特征、只按外国频率的类别权重、目标样本 r=50 倍、lgb.train 固定 300 棵、deterministic。
一次训练 300 棵，在 100/150/200/250/300 轮分别预测（前 t 棵与单独训练 t 棵完全相同），
比较 Δ = ae − s2 与 方法 − random 在不同树数下的一致性。另记录到达各轮次的耗时。
"""
import os
import sys
import time

CODE = ('/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/'
        'Alpha_Crops/label_expansion')
sys.path.insert(0, CODE)
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

from common import RANDOM_STATE, load_dataset, feature_cols  # noqa: E402
from pilot_target_selection import setup_target, kmeans_pick  # noqa: E402
from stageA import lgb_params, class_weights_from, metrics  # noqa: E402

OUT = '/Users/perng_mini/AlphaCrops_local/verify/guidance_review/tree_check.csv'
TARGETS = os.environ.get('TC_TARGETS', 'PT,SE').split(',')
SEEDS = [int(x) for x in os.environ.get('TC_SEEDS', '0,1').split(',')]
BUDGETS = [50, 200]
STRATS = ['ae_rep', 's2_rep', 'ae_gap', 's2_gap', 'random']
ITERS = [100, 150, 200, 250, 300]
R_MULT = 50
THREADS = 2


class Timer:
    def __init__(self):
        self.t = {}
        self.t0 = None

    def __call__(self, env):
        it = env.iteration + 1
        if it in ITERS:
            self.t[it] = time.time() - self.t0


def main():
    df = load_dataset()
    classes = sorted(df['label'].unique())
    y = df['label'].map({c: i for i, c in enumerate(classes)}).to_numpy()
    ctry = df['country'].to_numpy()
    X = {f: df[feature_cols(df, f)].to_numpy(np.float32) for f in ['ae', 's2']}
    rows = pd.read_csv(OUT).to_dict('records') if os.path.exists(OUT) else []
    done = {(r['target'], r['strategy'], r['budget'], r['seed']) for r in rows}
    for tgt in TARGETS:
        foreign, pool, test, _ = setup_target(df, ctry, tgt)
        desc = {}
        for f in ['ae', 's2']:
            Z = np.vstack([X[f][foreign], X[f][pool]]).astype(np.float64)
            Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
            Z = PCA(32, random_state=RANDOM_STATE).fit_transform(Z)
            desc[f] = (Z[:len(foreign)], Z[len(foreign):])
        gap = {f: NearestNeighbors(n_neighbors=1).fit(desc[f][0]).kneighbors(desc[f][1])[0].ravel()
               for f in ['ae', 's2']}
        yT = y[test]
        big = np.flatnonzero(np.bincount(yT, minlength=10) >= 30)
        cw = class_weights_from(y[foreign])
        Xs = X['s2']
        for k in BUDGETS:
            for seed in SEEDS:
                for strat in STRATS:
                    if (tgt, strat, k, seed) in done:
                        continue
                    r = np.random.default_rng(1000 * seed + k)
                    if strat == 'random':
                        pk = r.choice(len(pool), k, replace=False)
                    elif strat.endswith('_rep'):
                        pk = kmeans_pick(desc[strat[:2]][1], k, seed)
                    else:
                        f = strat[:2]
                        cand = np.argsort(-gap[f])[:max(k, int(0.3 * len(pool)))]
                        pk = cand[kmeans_pick(desc[f][1][cand], k, seed)]
                    sel = pool[pk]
                    t0 = time.time()
                    Xtr = np.vstack([Xs[foreign], Xs[sel]])
                    ytr = np.r_[y[foreign], y[sel]]
                    w = np.r_[cw[y[foreign]], R_MULT * cw[y[sel]]]
                    ds = lgb.Dataset(Xtr, ytr, weight=w, free_raw_data=True)
                    tm = Timer()
                    tm.t0 = time.time()
                    m = lgb.train(lgb_params(THREADS, seed), ds, num_boost_round=300, callbacks=[tm])
                    t_train = time.time() - t0
                    for it in ITERS:
                        pred = m.predict(Xs[test], num_iteration=it).argmax(1)
                        mt = metrics(yT, pred, big)
                        rows.append(dict(target=tgt, strategy=strat, budget=k, seed=seed, iters=it,
                                         oa=mt['oa'], ba_ge30=mt['ba_ge30'], t_at_iter=round(tm.t.get(it, np.nan), 1),
                                         t_train_total=round(t_train, 1)))
                    tmp = OUT + '.tmp'
                    pd.DataFrame(rows).to_csv(tmp, index=False)
                    os.replace(tmp, OUT)
                    print(f'{tgt} k={k} s={seed} {strat}: {t_train:.0f}s (150:{tm.t.get(150, 0):.0f}s '
                          f'200:{tm.t.get(200, 0):.0f}s)', flush=True)


if __name__ == '__main__':
    main()
