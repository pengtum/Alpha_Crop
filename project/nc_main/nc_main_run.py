"""NC 主实验第 5 步：全部拟合，只保存预测概率（冻结方案 NC_PLAN_v6 第 1–2 节）。不计算任何精度。

输入：nc_main_split.py 的 splits/；试点已登记的样本表与特征（features/{T1,T2}_{ae,s2}.parquet）。
缺失：AE 或 S2 整行为空的样本两种特征都剔除（源池、测试、候选同样处理）；部分为空保留 NaN
      （LightGBM 原生处理；随机森林、逻辑回归用训练集不加权中位数填补）。
每个目标域 d：
  源池 = 同任务其他全部域的全部样本，按 (域名, sid) 升序；
  臂：Z0；RT_k（k ∈ KS 且 k ≤ 剔除后候选数，r = 0、1、2；剔除缺失后的抽样顺序取前 k 个）；
  学习器：lgb 做全部臂；rf、lr 只做 Z0 与 RT200（r = 0、1、2）；
  权重：类别权重 = 该源池类别频率的倒数（class_weights_from）；源样本取类别权重，目标样本取 50 × 类别权重；
  行序：源池在前，目标样本按抽样顺序在后。
输出 {root}/probs/{task}/{d}/{learner}_{feat}_{arm}_r{r}.npz：P_test（float32）、sid_test；
      lgb 的 AE Z0 另存 P_m、sid_m：剔除缺失后 r = 0 抽样顺序第 201–250 个候选样本（用于 M）。
      {root}/runs_{shard}.csv（每次拟合一行）、{root}/removed.csv、{root}/columns.json、{root}/budgets.csv。

用法：python nc_main_run.py --shard i --nshard n [--learners lgb,rf,lr]      # 正式运行
      python nc_main_run.py --smoke [--root DIR] [--smoke-domains n] [--shard i --nshard n]   # 合成特征、极小模型
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
import warnings

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '2'
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NCP = os.path.join(os.path.dirname(HERE), 'nc_pilot')
sys.path.insert(0, NCP)
import nc_common as K  # noqa: E402

LOCAL = os.path.expanduser('~/AlphaCrops_local/nc_main')
SPLITS = os.path.join(LOCAL, 'splits')
PILOT_SPLITS = os.path.join(K.DATA, 'splits')
FEATS_DIR = K.FEATS_DIR
PREFIX = 'nc_main_v1'
TASKS = {'lcz': ('T1', 'city', 'row', K.T1_C), 'lucas': ('T2', 'nuts0', 'id', K.T2_C)}
FEATS = ('ae', 's2')
KS = (50, 100, 200, 400, 800, 1600)
R = 3
M_SLICE = slice(200, 250)            # 抽样顺序第 201–250 个
APPX_K = 200                         # 附录学习器只做 Z0 与 RT200
TREES = 300                          # 与 RSE、试点相同
RF_PARAMS = dict(n_estimators=500, max_features='sqrt', min_samples_leaf=1, bootstrap=True, class_weight=None)
LR_PARAMS = dict(C=1e-3, max_iter=1000, tol=1e-4)
VERSIONS = {'python': '3.12.13', 'numpy': '2.5.3', 'scipy': '1.18.1', 'sklearn': '1.9.1', 'lightgbm': '4.7.0'}   # 与 RSE confirm_split.LIBS 相同
CODE_FILES = ('nc_main_run.py',)


def code_hash():
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update(open(os.path.join(HERE, f), 'rb').read())
    h.update(open(os.path.join(NCP, 'nc_common.py'), 'rb').read())
    return h.hexdigest()[:16]


def check_versions():
    import lightgbm
    import scipy
    import sklearn
    got = {'python': sys.version.split()[0], 'numpy': np.__version__, 'scipy': scipy.__version__,
           'sklearn': sklearn.__version__, 'lightgbm': lightgbm.__version__}
    assert got == VERSIONS, f'库版本与 RSE 不同：{got}'
    return got


# ---------------------------------------------------------------- 数据
def synth_features(S, f, seed):
    """冒烟测试用合成特征：类别相关的均值平移 + 噪声；少量整行 NaN 与部分 NaN。"""
    rng = np.random.default_rng(seed)
    p = 128 if f == 'ae' else 168
    y = S.y.values
    X = rng.normal(size=(len(y), p)).astype(np.float32) + 0.4 * y[:, None] * (np.arange(p) % 3 == 0)
    X[rng.random(len(y)) < 0.005] = np.nan
    X[rng.random(X.shape) < 0.01] = np.nan
    df = pd.DataFrame(X, columns=[f'{f}_{j:03d}' for j in range(p)])
    df.insert(0, 'sid', S.sid.values)
    return df


def load_task(task, smoke):
    """样本表（全部 32 城 / 28 国、全部样本）+ 本实验角色；剔除 AE 或 S2 整行为空的样本。"""
    T, dom, idc, C = TASKS[task]
    S = pd.read_parquet(os.path.join(PILOT_SPLITS, f'{T.lower()}_samples.parquet'))
    S = S.rename(columns={idc: 'sid', dom: 'domain'})[['sid', 'domain', 'y']]
    S['sid'] = S.sid.astype(np.int64)
    S = S.sort_values(['domain', 'sid']).reset_index(drop=True)
    F = {}
    for f in FEATS:
        df = synth_features(S, f, K.seed_of(f'smoke_{task}_{f}')) if smoke else \
            pd.read_parquet(os.path.join(FEATS_DIR, f'{T}_{f}.parquet'))
        df = df.set_index('sid').reindex(S.sid.values)
        F[f] = df
    empty = np.zeros(len(S), bool)
    for f in FEATS:
        empty |= F[f].isna().all(axis=1).values
    removed = S[empty].assign(task=task)
    keep = ~empty
    X = {f: F[f].values[keep].astype(np.float32) for f in FEATS}
    cols = {f: list(F[f].columns) for f in FEATS}
    return S[keep].reset_index(drop=True), X, cols, removed


# ---------------------------------------------------------------- 学习器
def _full(classes, P_part, n, C):
    P = np.zeros((n, C))
    P[:, classes] = P_part
    return P


def fit_predict(learner, Xtr, ytr, wtr, Xte_list, seed, C, smoke):
    """返回 (Xte_list 中每个矩阵的概率 (n, C) float32 列表, 信息字典)。"""
    info = {}
    if learner == 'lgb':
        import lightgbm as lgb
        ds = lgb.Dataset(Xtr, ytr, weight=wtr, free_raw_data=True)
        m = lgb.train(K.lgb_params(seed, C), ds, num_boost_round=5 if smoke else TREES)
        return [m.predict(B).astype(np.float32) for B in Xte_list], info
    from threadpoolctl import threadpool_limits
    Xtr = np.asarray(Xtr, dtype=np.float64)
    allnan = np.isnan(Xtr).all(0)
    med = np.zeros(Xtr.shape[1])
    med[~allnan] = np.nanmedian(Xtr[:, ~allnan], axis=0)       # 训练集整列为空时取 0（只为不报错，正式数据不会出现）
    info['n_allnan_cols'] = int(allnan.sum())

    def imp(A):
        A = np.asarray(A, dtype=np.float64).copy()
        r, c = np.nonzero(np.isnan(A))
        A[r, c] = med[c]
        return A
    A = imp(Xtr)
    with threadpool_limits(limits=K.THREADS), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if learner == 'lr':
            from sklearn.linear_model import LogisticRegression
            mu, sd = A.mean(0), A.std(0) + 1e-6
            m = LogisticRegression(**LR_PARAMS).fit((A - mu) / sd, ytr, sample_weight=wtr / wtr.mean())
            out = [_full(m.classes_, m.predict_proba((imp(B) - mu) / sd), len(B), C) for B in Xte_list]
            info['n_iter'] = int(np.max(m.n_iter_))
            info['converged'] = bool(np.max(m.n_iter_) < LR_PARAMS['max_iter'])
        elif learner == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            prm = dict(RF_PARAMS, n_estimators=5) if smoke else RF_PARAMS
            m = RandomForestClassifier(random_state=seed, n_jobs=K.THREADS, **prm).fit(A, ytr, sample_weight=wtr)
            out = [_full(m.classes_, m.predict_proba(imp(B)), len(B), C) for B in Xte_list]
        else:
            raise ValueError(learner)
    from sklearn.exceptions import ConvergenceWarning
    info['n_convwarn'] = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    info['n_otherwarn'] = len(caught) - info['n_convwarn']
    return [P.astype(np.float32) for P in out], info


# ---------------------------------------------------------------- 计划
def plan_task(task, S, targets, orders, learners):
    """每次拟合一项：(learner, feat, d, arm, r, k, 目标样本在 S 中的行号)。"""
    pos = pd.Series(np.arange(len(S)), index=S.sid.values)
    items, budgets = [], []
    for d in targets[task]:
        ords = []
        for r in range(R):
            o = orders[f'{task}/{d}/r{r}']
            o = o[np.isin(o, S.sid.values[(S.domain == d).values])]      # 剔除缺失后保持相对次序
            ords.append(o)
        n_c = len(ords[0])
        ks = [k for k in KS if k <= n_c]
        budgets.append(dict(task=task, domain=d, n_cand_after=n_c, ks=' '.join(map(str, ks))))
        for lr_ in learners:
            arms = [('Z0', 0, 0)] + [(f'RT{k}', r, k) for k in ks for r in range(R)] if lr_ == 'lgb' else \
                [('Z0', 0, 0)] + [(f'RT{APPX_K}', r, APPX_K) for r in range(R)]
            for f in FEATS:
                for arm, r, k in arms:
                    tgt = pos.loc[ords[r][:k]].values if k else np.array([], dtype=np.int64)
                    assert len(tgt) == k
                    items.append((lr_, f, d, arm, r, k, tgt))
        items_m = pos.loc[ords[0][M_SLICE]].values
        assert len(items_m) == M_SLICE.stop - M_SLICE.start
        budgets[-1]['m_rows'] = items_m
    return items, budgets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=1)
    ap.add_argument('--learners', default='lgb,rf,lr')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--root', default=None)
    ap.add_argument('--smoke-domains', type=int, default=2, help='冒烟时每任务取前 n 个域；0 = 全部')
    a = ap.parse_args()
    learners = a.learners.split(',')
    assert set(learners) <= {'lgb', 'rf', 'lr'}
    if a.smoke:
        root = a.root or os.path.join(LOCAL, 'smoke')
        assert os.path.realpath(root) != os.path.realpath(os.path.join(LOCAL, 'runs')), '冒烟模式不得写入正式 runs 目录'
    else:
        assert a.root is None, '正式运行固定写入 LOCAL/runs'
        root = os.path.join(LOCAL, 'runs')
        check_versions()
    os.makedirs(root, exist_ok=True)
    targets = json.load(open(os.path.join(SPLITS, 'targets.json')))
    if a.smoke and a.smoke_domains > 0:                            # 冒烟：每任务前 n 个域
        targets = {t: v[:a.smoke_domains] for t, v in targets.items()}
    roles = pd.read_parquet(os.path.join(SPLITS, 'roles.parquet'))
    orders = np.load(os.path.join(SPLITS, 'orders.npz'))
    split_sha = K.sha256(os.path.join(SPLITS, 'manifest.json'))[:16]
    ch = code_hash()
    runs_path = os.path.join(root, f'runs_{a.shard}.csv')
    done = set()
    for fn in os.listdir(root):
        if fn.startswith('runs_') and fn.endswith('.csv'):
            done |= set(pd.read_csv(os.path.join(root, fn)).key)
    rem_all, colmap, bud_all, idx = [], {}, [], -1
    for task in TASKS:
        S, X, cols, removed = load_task(task, a.smoke)
        rem_all.append(removed)
        colmap.update({f'{task}_{f}': cols[f] for f in FEATS})
        feat_sha = {f: 'synthetic' if a.smoke else K.sha256(os.path.join(FEATS_DIR, f'{TASKS[task][0]}_{f}.parquet'))[:16]
                    for f in FEATS}
        C = TASKS[task][3]
        rl = roles[roles.task == task].set_index(['domain', 'sid']).role
        S_role = rl.reindex(pd.MultiIndex.from_arrays([S.domain.values, S.sid.values])).values
        y = S.y.values.astype(int)
        items, budgets = plan_task(task, S, targets, orders, learners)
        m_rows = {b['domain']: b.pop('m_rows') for b in budgets}
        bud_all += budgets
        cache = {}
        for lr_, f, d, arm, r, k, tgt in items:
            idx += 1
            if idx % a.nshard != a.shard:
                continue
            key = f'{task}|{d}|{lr_}|{f}|{arm}|r{r}'
            if key in done:
                continue
            if d not in cache:
                src = np.flatnonzero((S.domain != d).values)            # S 已按 (域名, sid) 升序
                test = np.flatnonzero(((S.domain == d) & (S_role == 'test')).values)
                cache = {d: (src, test, K.class_weights_from(y[src], C))}
            src, test, cw = cache[d]
            tr = np.concatenate([src, tgt])
            w = np.concatenate([cw[y[src]], K.TARGET_W * cw[y[tgt]]])
            seed = K.seed_of(f'{PREFIX}_{task}_fit_{d}_{lr_}_{f}_{arm}_r{r}')
            with_m = lr_ == 'lgb' and f == 'ae' and arm == 'Z0'
            te_list = [X[f][test]] + ([X[f][m_rows[d]]] if with_m else [])
            status, err, attempts, info = 'ok', '', 0, {}
            t0 = time.time()
            for attempts in (1, 2):                                   # 失败用同一种子重试一次
                try:
                    out, info = fit_predict(lr_, X[f][tr], y[tr], w, te_list, seed, C, a.smoke)
                    assert all(np.isfinite(P).all() for P in out)
                    status, err = 'ok', ''
                    break
                except Exception:                                    # noqa: BLE001
                    status, err = 'fail', traceback.format_exc(limit=2).replace('\n', ' | ')[-500:]
            dt = time.time() - t0
            if status == 'ok':
                od = os.path.join(root, 'probs', task, d)
                os.makedirs(od, exist_ok=True)
                extra = dict(P_m=out[1], sid_m=S.sid.values[m_rows[d]]) if with_m else {}
                np.savez(os.path.join(od, f'{lr_}_{f}_{arm}_r{r}.npz'), P_test=out[0], sid_test=S.sid.values[test], **extra)
            row = dict(key=key, task=task, domain=d, learner=lr_, feat=f, arm=arm, r=r, k=k, seed=seed,
                       n_src=len(src), n_tgt=len(tgt), n_test=len(test),
                       train_sha=hashlib.sha256(S.sid.values[tr].astype(np.int64).tobytes()).hexdigest()[:16],
                       n_features=X[f].shape[1], threads=K.THREADS, t_fit=round(dt, 2), status=status,
                       attempts=attempts, error=err, info=json.dumps(info), code_hash=ch, feat_sha=feat_sha[f],
                       split_sha=split_sha)
            pd.DataFrame([row]).to_csv(runs_path, mode='a', header=not os.path.exists(runs_path), index=False)
            print(key, f'n_tgt={len(tgt)}', f'{dt:.1f}s', status, flush=True)
    if a.shard == 0:
        pd.concat(rem_all).to_csv(os.path.join(root, 'removed.csv'), index=False)
        json.dump(colmap, open(os.path.join(root, 'columns.json'), 'w'), indent=0)
        pd.DataFrame(bud_all).to_csv(os.path.join(root, 'budgets.csv'), index=False)
    print('done shard', a.shard)


if __name__ == '__main__':
    main()
