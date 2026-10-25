"""
NC 试点第 6 步：160 次 LightGBM 拟合，只保存测试集预测概率（方案第 5 节）。不计算任何精度。

读 splits/（nc_split.py）与 features/{T1,T2}_{ae,s2}.parquet（nc_extract.py fetch）。
缺失（方案第 2 节）：某样本 AE 或 S2 整行为空 → 两种特征都剔除（源池、测试、候选同样处理）；部分为空保留 NaN。
臂：Z0（每任务每特征 1 次，预测 3 个目标域）；RT_k（k ∈ 50..400，r ∈ 0..2，抽样顺序剔除后取前 k 个）；
    RT全（全部剩余候选，按编号）。行序：源池按编号在前，目标在后。权重：源 = 类别权重，目标 = 50 × 类别权重。
输出 {root}/probs/{任务}/{域}/{特征}_{臂}_r{r}.npz（P_test float32，sid_test）；{root}/runs.csv；{root}/removed.csv。

用法：python nc_run.py [--root DIR] [--smoke]       # --smoke：合成特征 + 5 棵树，输出到 DATA/smoke/
"""
import argparse
import hashlib
import json
import os
import sys
import time

os.environ['OMP_NUM_THREADS'] = '2'
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nc_common as K  # noqa: E402

TREES = 300
ID = {'T1': 'row', 'T2': 'id'}
DOM = {'T1': 'city', 'T2': 'nuts0'}
NC = {'T1': K.T1_C, 'T2': K.T2_C}


def code_hash():
    h = hashlib.sha256()
    for f in ('nc_common.py', 'nc_run.py'):
        h.update(open(os.path.join(K.HERE, f), 'rb').read())
    return h.hexdigest()[:16]


def synth_features(samples, task, feat, seed):
    """冒烟测试用合成特征：类别相关的均值平移 + 噪声；少量整行 NaN 与部分 NaN。"""
    rng = np.random.default_rng(seed)
    from nc_extract import AE_BANDS, s2_names, out_columns
    cols = out_columns(AE_BANDS if feat == 'ae' else s2_names())
    y = samples.y.values
    X = rng.normal(size=(len(y), len(cols))).astype(np.float32) + 0.3 * y[:, None] * (np.arange(len(cols)) % 3 == 0)
    X[rng.random(len(y)) < 0.005] = np.nan
    X[rng.random(X.shape) < 0.01] = np.nan
    df = pd.DataFrame(X, columns=cols)
    df.insert(0, 'sid', samples[ID[task]].values.astype(np.int64))
    return df


def load_task(task, smoke):
    S = pd.read_parquet(os.path.join(K.SPLITS, f'{task.lower()}_samples.parquet'))
    S = S[S.role.isin(['source', 'test', 'cand'])].sort_values(ID[task]).reset_index(drop=True)
    F = {}
    for f in K.FEATS:
        df = synth_features(S, task, f, K.seed_of(f'smoke_{task}_{f}')) if smoke else \
            pd.read_parquet(os.path.join(K.FEATS_DIR, f'{task}_{f}.parquet'))
        df = df.set_index('sid').reindex(S[ID[task]].values)
        F[f] = df
    empty = np.zeros(len(S), bool)
    for f in K.FEATS:
        empty |= F[f].isna().all(1).values
    removed = S[empty].assign(task=task)[['task', DOM[task], 'role', 'y', ID[task]]]
    keep = ~empty
    grp = np.where(S.role.values == 'source', 'source', S[DOM[task]].values)
    rates = []                                                        # 保留样本的部分缺失率（方案第 2 节）
    for f in K.FEATS:
        V = F[f].values[keep]
        for (g, yy), idx in pd.Series(np.arange(keep.sum())).groupby([grp[keep], S.y.values[keep]]):
            sub = V[idx.values]
            rates.append(dict(task=task, group=g, y=int(yy), feat=f, n=len(sub),
                              rows_any_nan=float(np.isnan(sub).any(1).mean()), cells_nan=float(np.isnan(sub).mean())))
    S = S[keep].reset_index(drop=True)
    X = {f: F[f].values[keep].astype(np.float32) for f in K.FEATS}
    cols = {f: list(F[f].columns) for f in K.FEATS}
    return S, X, cols, removed, pd.DataFrame(rates)


def fit_predict(Xtr, ytr, wtr, Xte, seed, C, trees):
    import lightgbm as lgb
    ds = lgb.Dataset(Xtr, ytr, weight=wtr, free_raw_data=True)
    m = lgb.train(K.lgb_params(seed, C), ds, num_boost_round=trees)
    return m.predict(Xte).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=None)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke and a.root and os.path.realpath(a.root) == os.path.realpath(os.path.join(K.DATA, 'runs')):
        sys.exit('冒烟模式不得写入正式 runs 目录')
    if not a.smoke:
        assert 'NC_SPLITS' not in os.environ, '正式运行不得改指划分目录'
        assert a.root is None, '正式运行固定写入 DATA/runs'
    root = a.root or os.path.join(K.DATA, 'smoke' if a.smoke else 'runs')
    trees = 5 if a.smoke else TREES
    os.makedirs(root, exist_ok=True)
    targets = json.load(open(os.path.join(K.SPLITS, 'targets.json')))
    orders = np.load(os.path.join(K.SPLITS, 'orders.npz'))
    ch = code_hash()
    import lightgbm
    runs_path = os.path.join(root, 'runs.csv')
    done = set(pd.read_csv(runs_path).key) if os.path.exists(runs_path) else set()
    rem_all, rate_all = [], []
    split_sha = K.sha256(os.path.join(K.SPLITS, 'manifest.json'))[:16]
    colmap = {}
    for task in K.TASKS:
        S, X, cols, removed, rates = load_task(task, a.smoke)
        rem_all.append(removed)
        rate_all.append(rates)
        feat_sha = {f: ('synthetic' if a.smoke else K.sha256(os.path.join(K.FEATS_DIR, f'{task}_{f}.parquet'))[:16])
                    for f in K.FEATS}
        cols_sha = {f: hashlib.sha256('\n'.join(cols[f]).encode()).hexdigest()[:16] for f in K.FEATS}
        colmap.update({f'{task}_{f}': cols[f] for f in K.FEATS})
        C = NC[task]
        idc, dom = ID[task], DOM[task]
        src = np.flatnonzero((S.role == 'source').values)            # 已按编号排序
        y = S.y.values.astype(int)
        cw = K.class_weights_from(y[src], C)
        tests = {d: np.flatnonzero(((S[dom] == d) & (S.role == 'test')).values) for d in targets[task]}
        pos = pd.Series(np.arange(len(S)), index=S[idc].values)
        plan = []
        for f in K.FEATS:
            plan.append((f, 'Z0', 0, None, None, K.seed_of(f'nc_pilot_v1_z0_{task}')))
        for d in targets[task]:
            for f in K.FEATS:
                for k in K.KS:
                    for r in range(K.R):
                        plan.append((f, f'RT{k}', r, d, k, K.seed_of(f'nc_pilot_v1_fit_{d}_k{k}_r{r}')))
                plan.append((f, 'ALL', 0, d, None, K.seed_of(f'nc_pilot_v1_fit_{d}_all')))
        for f, arm, r, d, k, seed in plan:
            key = f'{task}|{d or "-"}|{f}|{arm}|r{r}'
            if key in done:
                continue
            if arm == 'Z0':
                tgt = np.array([], dtype=int)
            elif arm == 'ALL':
                tgt = np.flatnonzero(((S[dom] == d) & (S.role == 'cand')).values)          # 按编号
            else:
                order = orders[f'{d}_r{r}']
                order = order[np.isin(order, S[idc].values)]                              # 剔除后保持相对次序
                tgt = pos.loc[order[:k]].values
                assert len(tgt) == k
            tr = np.concatenate([src, tgt])
            w = np.concatenate([cw[y[src]], K.TARGET_W * cw[y[tgt]]])
            te_doms = targets[task] if arm == 'Z0' else [d]
            te_all = np.concatenate([tests[x] for x in te_doms])
            t0 = time.time()
            P = fit_predict(X[f][tr], y[tr], w, X[f][te_all], seed, C, trees)
            dt = time.time() - t0
            a0 = 0
            for x in te_doms:
                n = len(tests[x])
                out = os.path.join(root, 'probs', task, x)
                os.makedirs(out, exist_ok=True)
                np.savez(os.path.join(out, f'{f}_{arm}_r{r}.npz'), P_test=P[a0:a0 + n],
                         sid_test=S[idc].values[tests[x]])
                a0 += n
            row = dict(key=key, task=task, dom=d or '-', feat=f, arm=arm, k=k if k else (len(tgt) if arm == 'ALL' else 0),
                       r=r, seed=seed, n_src=len(src), n_tgt=len(tgt),
                       train_sha=hashlib.sha256(S[idc].values[tr].astype(np.int64).tobytes()).hexdigest()[:16],
                       n_features=X[f].shape[1], trees=trees, threads=K.THREADS, t_fit=round(dt, 2), code_hash=ch,
                       feat_sha=feat_sha[f], cols_sha=cols_sha[f], split_sha=split_sha, n_removed=len(removed),
                       lightgbm=lightgbm.__version__, numpy=np.__version__, status='ok')
            pd.DataFrame([row]).to_csv(runs_path, mode='a', header=not os.path.exists(runs_path), index=False)
            print(key, f'n_tgt={len(tgt)}', f'{dt:.1f}s', flush=True)
    pd.concat(rem_all).to_csv(os.path.join(root, 'removed.csv'), index=False)
    json.dump(colmap, open(os.path.join(root, 'columns.json'), 'w'), indent=0)
    pd.concat(rate_all).to_csv(os.path.join(root, 'missing_rates.csv'), index=False)
    print('done')


if __name__ == '__main__':
    main()
