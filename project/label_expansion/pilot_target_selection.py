"""
检验 2 试点：在目标国采集少量标签时，采在哪里最好？（v1，2026-09-24 按代码审查修正）

场景：目标国 T 没有标签；其余 10 国的现成标签作为外国池（每国抽 5,000 个，共 50,000）。
T 的地块（先去掉质心与面积完全相同的重复地块）随机分为候选池 70% 与留出测试 30%。
策略从候选池选 k 个地块"打标签"，与外国池合并训练，在留出测试上评估。
测试地块从不进入任何池，避免 exp6 的覆盖混淆。

选样策略（选样依据与分类特征分开，便于区分"AE 作为特征"与"AE 作为选样工具"）：
  random       随机
  geo_strat    在公里坐标上 k-means 分层，取各簇中心最近的地块
  s2_rep       S2 空间 k-means 代表性样本
  ae_rep       AE 空间 k-means 代表性样本
  s2_gap       S2 空间中离外国池最远的 30% 地块里再做 k-means 分层（外国数据覆盖不到的地方）
  ae_gap       同上，AE 空间
  uncertainty  外国池训练模型预测熵最高的 30% 地块里再做 k-means 分层（分类特征空间）

训练权重（v1 修正）：先对全部训练样本计算类别平衡权重，再把目标国样本整体缩放，
使目标国样本的权重总和恰为外国样本权重总和的 25%。v0 用 LightGBM 的 class_weight='balanced'
乘名义权重，实际份额随国家（PT 约 0.5、LV 约 0.1）和策略（PT k=50 时 0.48–0.67）变化，已作废。
固定 300 棵树、不早停（没有目标国验证集可用）。

参照线（3 个种子）：
  foreign_only       只用外国池（k=0）
  target_full        只用目标国全部候选池
  ceiling_w25        外国池 + 全部候选池，按与策略相同的 25% 规则加权（补回比例的分母）
  foreign+target_nat 外国池 + 全部候选池，只做类别平衡、不做域缩放

指标：ba（测试集中出现的全部类别）、ba_ge30（只算测试集地块数 ≥30 的类别）、macro_f1。
逐地块预测保存在 RESULTS_DIR/preds/，可事后重算任何指标。
输出：RESULTS_DIR/pilot_selection.csv（可断点续跑，原子写入）
"""
import hashlib
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.class_weight import compute_sample_weight

from common import COUNTRIES, RESULTS_DIR, RANDOM_STATE, LGBM_PARAMS, load_dataset, feature_cols

TARGETS = ['PT', 'SE', 'SK', 'LV']
BUDGETS = [50, 200, 1000]
SEEDS = [0, 1, 2]
STRATEGIES = ['random', 'geo_strat', 's2_rep', 'ae_rep', 's2_gap', 'ae_gap', 'uncertainty']
CLF_FEATS = ['ae', 's2']
N_FOREIGN_PER_COUNTRY = 5000
TARGET_WEIGHT_SHARE = 0.25
MIN_TEST_PER_CLASS = 30
PARAMS = {**LGBM_PARAMS, 'n_estimators': 300, 'class_weight': None}
OUT = os.path.join(RESULTS_DIR, 'pilot_selection.csv')
PRED_DIR = os.path.join(RESULTS_DIR, 'preds')
PLAN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ANALYSIS_PLAN_check2.md')


def setup_target(df, ctry, tgt):
    """外国池、候选池、测试集。pilot_betti.py 复用此函数以保证划分完全一致。"""
    rng = np.random.default_rng(RANDOM_STATE)
    foreign = np.concatenate([
        rng.choice(np.flatnonzero(ctry == c), N_FOREIGN_PER_COUNTRY, replace=False)
        for c in COUNTRIES if c != tgt])
    t_idx = np.flatnonzero(ctry == tgt)
    dup = df.iloc[t_idx].duplicated(['longitude', 'latitude', 'area_ha']).to_numpy()
    t_idx = t_idx[~dup]
    pool, test = train_test_split(t_idx, test_size=0.3, random_state=RANDOM_STATE)
    return foreign, pool, test, int(dup.sum())


def train(X, y, is_target, seed, rescale=True):
    """类别平衡权重；若同时有外国与目标国样本且 rescale，把目标国样本整体缩放到 25% 份额。
    返回模型与目标国样本的实际权重份额（目标国权重和 / 外国权重和）。"""
    w = compute_sample_weight('balanced', y)
    mixed = is_target.any() and not is_target.all()
    if rescale and mixed:
        w[is_target] *= TARGET_WEIGHT_SHARE * w[~is_target].sum() / w[is_target].sum()
    m = lgb.LGBMClassifier(**{**PARAMS, 'random_state': seed})
    m.fit(X, y, sample_weight=w)
    share = w[is_target].sum() / w[~is_target].sum() if mixed else (np.inf if is_target.all() else 0.0)
    return m, float(share)


def evaluate(y_true, y_pred):
    present, counts = np.unique(y_true, return_counts=True)
    rec = {c: (y_pred[y_true == c] == c).mean() for c in present}
    big = present[counts >= MIN_TEST_PER_CLASS]
    return dict(ba=float(np.mean(list(rec.values()))),
                ba_ge30=float(np.mean([rec[c] for c in big])),
                macro_f1=float(f1_score(y_true, y_pred, labels=present, average='macro', zero_division=0)))


def save_rows(rows, path):
    tmp = path + '.tmp'
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


def plan_hash():
    with open(PLAN, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def kmeans_pick(Z, k, seed):
    """k-means 分 k 簇，取离各簇中心最近的样本下标。"""
    k = min(k, len(Z))
    km = KMeans(n_clusters=k, n_init=1, random_state=seed, max_iter=100).fit(Z)
    nn = NearestNeighbors(n_neighbors=1).fit(Z)
    idx = np.unique(nn.kneighbors(km.cluster_centers_, return_distance=False).ravel())
    if len(idx) < k:  # 多个中心共享最近点时随机补足
        rest = np.setdiff1d(np.arange(len(Z)), idx)
        idx = np.r_[idx, np.random.default_rng(seed).choice(rest, k - len(idx), replace=False)]
    return idx


def main():
    print(f'plan sha256[:16] = {plan_hash()}', flush=True)
    os.makedirs(PRED_DIR, exist_ok=True)
    df = load_dataset()
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

    def run(tgt, strat, k, seed, cf, tr_idx, is_t, test, n_sel_cls=np.nan, rescale=True):
        m, share = train(X[cf][tr_idx], y_all[tr_idx], is_t, seed, rescale)
        pred = m.predict(X[cf][test])
        np.save(os.path.join(PRED_DIR, f'{tgt}_{strat}_{k}_{seed}_{cf}.npy'), pred.astype(np.int8))
        rows.append(dict(target=tgt, strategy=strat, budget=k, seed=seed, clf_feat=cf,
                         eff_share=share, n_classes_sel=n_sel_cls, **evaluate(y_all[test], pred)))
        save_rows(rows, OUT)
        return m, rows[-1]

    for tgt in TARGETS:
        foreign, pool, test, n_dup = setup_target(df, ctry, tgt)
        np.save(os.path.join(PRED_DIR, f'{tgt}_test_idx.npy'), test)
        print(f'{tgt}: foreign {len(foreign):,}  pool {len(pool):,}  test {len(test):,}  '
              f'(dropped {n_dup} exact duplicates)', flush=True)

        # 选样用的描述空间：外国池 + 候选池合并标准化后 PCA 到 32 维
        desc = {}
        for f in ['ae', 's2']:
            Z = np.vstack([X[f][foreign], X[f][pool]]).astype(np.float64)
            Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
            Z = PCA(32, random_state=RANDOM_STATE).fit_transform(Z)
            desc[f] = (Z[:len(foreign)], Z[len(foreign):])
        gap = {f: NearestNeighbors(n_neighbors=1).fit(desc[f][0]).kneighbors(desc[f][1])[0].ravel()
               for f in ['ae', 's2']}
        # 公里坐标（两个轴同一尺度）
        lat0 = lat[pool].mean()
        km_xy = np.c_[lon[pool] * 111.32 * np.cos(np.radians(lat0)), lat[pool] * 110.57]
        km_xy = km_xy - km_xy.mean(0)

        f_only = np.zeros(len(foreign), bool)
        both = np.r_[foreign, pool]
        both_t = np.r_[f_only, np.ones(len(pool), bool)]
        for cf in CLF_FEATS:
            for seed in SEEDS:
                for name, tr, is_t, rescale in [('foreign_only', foreign, f_only, True),
                                                ('target_full', pool, np.ones(len(pool), bool), True),
                                                ('ceiling_w25', both, both_t, True),
                                                ('foreign+target_nat', both, both_t, False)]:
                    kk = 0 if name == 'foreign_only' else len(pool)
                    if (tgt, name, kk, seed, cf) in done:
                        continue
                    m, r = run(tgt, name, kk, seed, cf, tr, is_t, test, rescale=rescale)
                    if name == 'foreign_only' and seed == 0:
                        np.save(os.path.join(RESULTS_DIR, f'pilot_probs_{tgt}_{cf}.npy'),
                                m.predict_proba(X[cf][pool]))
                    print(f'{tgt} {cf} s={seed} {name:18s}: ba={r["ba"]:.3f} ba_ge30={r["ba_ge30"]:.3f} '
                          f'f1={r["macro_f1"]:.3f}', flush=True)
            probs_path = os.path.join(RESULTS_DIR, f'pilot_probs_{tgt}_{cf}.npy')
            if not os.path.exists(probs_path):  # 续跑时 foreign_only seed 0 已完成但概率文件缺失
                m, _ = train(X[cf][foreign], y_all[foreign], f_only, 0)
                np.save(probs_path, m.predict_proba(X[cf][pool]))

        for k in BUDGETS:
            for seed in SEEDS:
                for strat in STRATEGIES:
                    r = np.random.default_rng(1000 * seed + k)
                    if strat == 'random':
                        pk = r.choice(len(pool), k, replace=False)
                    elif strat == 'geo_strat':
                        pk = kmeans_pick(km_xy, k, seed)
                    elif strat in ('s2_rep', 'ae_rep'):
                        pk = kmeans_pick(desc[strat[:2]][1], k, seed)
                    elif strat in ('s2_gap', 'ae_gap'):
                        f = strat[:2]
                        cand = np.argsort(-gap[f])[:max(k, int(0.3 * len(pool)))]
                        pk = cand[kmeans_pick(desc[f][1][cand], k, seed)]
                    else:
                        pk = None  # uncertainty 依赖分类特征，下面按 cf 计算
                    for cf in CLF_FEATS:
                        if (tgt, strat, k, seed, cf) in done:
                            continue
                        t0 = time.time()
                        if strat == 'uncertainty':
                            probs0 = np.load(os.path.join(RESULTS_DIR, f'pilot_probs_{tgt}_{cf}.npy'))
                            ent = -(probs0 * np.log(probs0 + 1e-12)).sum(1)
                            cand = np.argsort(-ent)[:max(k, int(0.3 * len(pool)))]
                            sel = pool[cand[kmeans_pick(desc[cf][1][cand], k, seed)]]
                        else:
                            sel = pool[pk]
                        tr = np.r_[foreign, sel]
                        is_t = np.r_[f_only, np.ones(len(sel), bool)]
                        _, row = run(tgt, strat, k, seed, cf, tr, is_t, test, len(np.unique(y_all[sel])))
                        print(f'{tgt} k={k:4d} s={seed} {strat:11s} {cf}: ba={row["ba"]:.3f} '
                              f'ba_ge30={row["ba_ge30"]:.3f} share={row["eff_share"]:.3f} '
                              f'(classes {row["n_classes_sel"]}) {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
