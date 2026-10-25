"""
采样优化实测·阶段 A 运行器（ANALYSIS_PLAN_stageA.md）。

用法：
  python stageA.py stage0 --jobs 3            # 每个（开发国, 划分）训练外国模型、留一国委员会、域判别器、PCA
  python stageA.py rgrid  --jobs 4            # 仅 SRS，r ∈ R_GRID，选目标样本权重倍数
  python stageA.py run    --jobs 4 --r50 R --r1000 R   # 全部方法，r 取选定值
每个工作进程只处理分配给它的（国家, 划分, 重复）任务，结果写入自己的 CSV（原子写入），可断点续跑。
输出目录：RESULTS_DIR/stageA/
"""
import argparse
import hashlib
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd

def _out():
    from common import RESULTS_DIR
    return os.environ.get('STAGEA_OUT', os.path.join(RESULTS_DIR, 'stageA'))  # 冒烟测试可另指目录


OUT = _out()  # 模块级计算：spawn 启动的子进程也能拿到


def stable_hash(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16)
DEV = os.environ.get('STAGEA_DEV', 'PT,BE,SI').split(',')
SPLITS = [int(x) for x in os.environ.get('STAGEA_SPLITS', '0,1,2').split(',')]
REPS = [int(x) for x in os.environ.get('STAGEA_REPS', '0,1,2').split(',')]
BUDGETS = [50, 1000]
R_GRID = [1, 5, 20, 50]
N_POOL = 30_000
N_FOREIGN_PER = 5_000
CELL_KM = 10.0
TEST_FRAC = 0.3
BUFFER_KM = 1.0
FEAT = 's2'
TREES = int(os.environ.get('STAGEA_TREES', 300))  # 冒烟测试可调小
N_PCA = 32
MIN_N = 30
ORACLE_ALLOWED = {'margin_r2', 'hybrid', 'oracle_prop', 'oracle_sqrt', 'oracle_equal', 'oracle_err'}
# 阶段 A 的方法（按文献调研第 3 节：每个家族取主要变体；共 33 个方法 + 4 个神谕）
METHODS_A = ['srs', 'srs_nuts3', 'lpm2', 'predstrat', 'predstrat_prop', 'predstrat_equal', 'pmht', 'cube', 'clhs',
             'margin', 'entropy', 'margin_prior', 'margin_r2', 'gist_margin', 'gist_maxmin', 'gist_geo', 'kcenter',
             'kcenter_foreign', 'badge', 'kdpp', 'clue', 'uherding', 'maxherding', 'typiclust', 'typiclust_strat',
             'fass_quota', 'fass_quota_equal', 'probcover', 'kherding', 'fscs', 'tqs', 'bait', 'hybrid',
             'oracle_prop', 'oracle_sqrt', 'oracle_equal', 'oracle_err']
# 超参数覆盖（集成审查的建议）：外国模型用了按外国频率的类别平衡权重 → 训练先验取均匀；UHerding 近邻截断放宽
HP = {'margin_prior': {'p_train': 'uniform'}, 'uherding': {'M_nn': 1000}, 'maxherding': {'M_nn': 1000}}
NEEDS_FIT = {'margin_r2', 'hybrid'}
PLAN = 'ANALYSIS_PLAN_stageA.md'


# ---------------------------------------------------------------- 公共
def lgb_params(threads, seed):
    from common import LGBM_PARAMS
    p = dict(objective='multiclass', num_class=10, learning_rate=LGBM_PARAMS['learning_rate'],
             num_leaves=LGBM_PARAMS['num_leaves'], min_data_in_leaf=LGBM_PARAMS['min_child_samples'],
             bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, max_bin=63,
             force_col_wise=True, num_threads=threads, seed=seed, deterministic=True, verbose=-1)
    return p


def class_weights_from(y_ref, C=10):
    """只按外国池的类别频率计算类别权重（外国与目标样本共用）。外国没有的类权重取 1。"""
    n = np.bincount(y_ref, minlength=C).astype(float)
    w = np.where(n > 0, len(y_ref) / (np.count_nonzero(n) * np.maximum(n, 1)), 1.0)
    return w


def fit_predict(X_tr, y_tr, w_tr, X_te, threads, seed, trees=TREES):
    import lightgbm as lgb
    ds = lgb.Dataset(X_tr, y_tr, weight=w_tr, free_raw_data=True)
    m = lgb.train(lgb_params(threads, seed), ds, num_boost_round=trees)
    return m.predict(X_te).astype(np.float32), m


def km_xy(lon, lat, lat0):
    return np.c_[lon * 111.32 * np.cos(np.radians(lat0)), lat * 110.57]


def save_csv(rows, path):
    tmp = path + '.tmp'
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


def metrics(y, p_pred, big):
    rec = np.array([(p_pred[y == c] == c).mean() if (y == c).any() else np.nan for c in range(10)])
    from sklearn.metrics import f1_score
    return dict(oa=100 * float((y == p_pred).mean()), ba_ge30=100 * float(np.mean(rec[big])),
                f1_ge30=100 * float(f1_score(y, p_pred, labels=big, average='macro', zero_division=0)),
                **{f'rec{c}': rec[c] for c in range(10)})


class Data:
    """一次性加载数据（每个进程各自加载）。"""
    def __init__(self):
        from common import load_dataset, feature_cols
        df = load_dataset()
        self.classes = sorted(df['label'].unique())
        self.y = df['label'].map({c: i for i, c in enumerate(self.classes)}).to_numpy()
        self.ctry = df['country'].to_numpy()
        self.lon, self.lat = df['longitude'].to_numpy(), df['latitude'].to_numpy()
        self.area = df['area_ha'].to_numpy()
        self.X = {f: df[feature_cols(df, f)].to_numpy(np.float32) for f in ('ae', 's2')}
        from common import DATA_DIR, COUNTRIES
        self.countries = COUNTRIES
        nuts = pd.read_parquet(os.path.join(DATA_DIR, 'parcel_nuts.parquet'), columns=['parcel_id', 'nuts3'])
        self.nuts3 = df[['parcel_id']].merge(nuts, on='parcel_id', how='left')['nuts3'].fillna('NA').to_numpy()


# ---------------------------------------------------------------- 阶段 0
def stage0_one(D, tgt, split, threads):
    from scipy.spatial import cKDTree
    from sklearn.decomposition import PCA
    from sklearn.model_selection import StratifiedKFold
    path = os.path.join(OUT, 'stage0', f'{tgt}_s{split}.npz')
    if os.path.exists(path):
        return
    t0 = time.time()
    rng = np.random.default_rng(1000 + 17 * split + stable_hash(tgt) % 997)
    t_idx = np.flatnonzero(D.ctry == tgt)
    keep = ~pd.DataFrame({'a': D.lon[t_idx], 'b': D.lat[t_idx], 'c': D.area[t_idx]}).duplicated().to_numpy()
    t_idx = t_idx[keep]
    lat0 = D.lat[t_idx].mean()
    xy = km_xy(D.lon[t_idx], D.lat[t_idx], lat0)
    cell = pd.factorize(pd.Series(np.floor(xy[:, 0] / CELL_KM).astype(int)).astype(str) + '_' +
                        pd.Series(np.floor(xy[:, 1] / CELL_KM).astype(int)).astype(str))[0]
    cells = rng.permutation(cell.max() + 1)
    test_cells = cells[:int(round(TEST_FRAC * len(cells)))]
    is_test = np.isin(cell, test_cells)
    d_test, _ = cKDTree(xy[is_test]).query(xy[~is_test])
    cand_mask = np.zeros(len(t_idx), bool)
    cand_mask[np.flatnonzero(~is_test)[d_test > BUFFER_KM]] = True
    test, cand = t_idx[is_test], t_idx[cand_mask]
    foreign = np.concatenate([rng.choice(np.flatnonzero(D.ctry == c), N_FOREIGN_PER, replace=False)
                              for c in D.countries if c != tgt])
    allT = np.concatenate([cand, test])
    # PCA（在外国 ∪ 目标全部地块上拟合，只用特征不用标签）
    pca = {}
    for f in ('ae', 's2'):
        Z = np.vstack([D.X[f][foreign], D.X[f][allT]]).astype(np.float64)
        mu, sd = Z.mean(0), Z.std(0) + 1e-9
        p = PCA(N_PCA, random_state=0).fit((Z - mu) / sd)
        pca[f] = (p.transform((D.X[f][foreign] - mu) / sd).astype(np.float32),
                  p.transform((D.X[f][allT] - mu) / sd).astype(np.float32))
    # 外国模型（类别权重只按外国频率）
    Xc = D.X[FEAT]
    yF = D.y[foreign]
    cw = class_weights_from(yF)
    P_T_raw, _ = fit_predict(Xc[foreign], yF, cw[yF], Xc[allT], threads, 0)
    # 留一国委员会：外国袋外概率 + 对目标的概率
    fc = D.ctry[foreign]
    others = [c for c in D.countries if c != tgt]
    P_F_oof = np.zeros((len(foreign), 10), np.float32)
    P_T_loco = np.zeros((len(others), len(allT), 10), np.float16)
    for j, c in enumerate(others):
        tr, te = fc != c, fc == c
        preds, _ = fit_predict(Xc[foreign[tr]], yF[tr], cw[yF[tr]],
                               np.vstack([Xc[foreign[te]], Xc[allT]]), threads, 0)
        P_F_oof[te] = preds[:te.sum()]
        P_T_loco[j] = preds[te.sum():].astype(np.float16)
    # 温度校准（在外国留一国袋外概率上拟合标量温度）
    from scipy.optimize import minimize_scalar
    zF = np.log(np.clip(P_F_oof, 1e-12, None))

    def nll(s):
        l = s * zF
        l -= l.max(1, keepdims=True)
        q = np.exp(l)
        q /= q.sum(1, keepdims=True)
        return -np.log(np.clip(q[np.arange(len(yF)), yF], 1e-300, None)).mean()
    s_opt = minimize_scalar(nll, bounds=(0.05, 20), method='bounded').x
    zT = s_opt * np.log(np.clip(P_T_raw, 1e-12, None))
    zT -= zT.max(1, keepdims=True)
    P_T = np.exp(zT)
    P_T /= P_T.sum(1, keepdims=True)
    # 域判别器（外国 vs 目标，5 折交叉拟合，对目标给出袋外 P(目标|x)）
    import lightgbm as lgb
    Xd = np.vstack([Xc[foreign], Xc[allT]])
    yd = np.r_[np.zeros(len(foreign), int), np.ones(len(allT), int)]
    dom_all = np.zeros(len(yd))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xd, yd):
        pr = dict(objective='binary', learning_rate=0.05, num_leaves=31, num_threads=threads, seed=0,
                  deterministic=True, force_col_wise=True, max_bin=63, verbose=-1)
        m = lgb.train(pr, lgb.Dataset(Xd[tr], yd[tr]), num_boost_round=200)
        dom_all[te] = m.predict(Xd[te])
    dom = dom_all[len(foreign):].astype(np.float32)
    np.savez_compressed(path, foreign=foreign, cand=cand, test=test, allT=allT,
                        F_ae=pca['ae'][0], T_ae=pca['ae'][1], F_s2=pca['s2'][0], T_s2=pca['s2'][1],
                        P_T=P_T.astype(np.float32), P_T_loco=P_T_loco, P_F_oof=P_F_oof, dom=dom,
                        temp=s_opt, lat0=lat0)
    print(f'[stage0] {tgt} s{split}: cand {len(cand):,} test {len(test):,} temp {s_opt:.2f} '
          f'{time.time() - t0:.0f}s', flush=True)


# ---------------------------------------------------------------- 选样与训练
def build_ctx(D, S, pool_pos):
    """pool_pos：候选池在 S['allT'] 中的位置。"""
    import samplers.base as B
    allT = S['allT']
    idx = allT[pool_pos]
    lat0 = float(S['lat0'])
    fr = S['foreign']
    ctx = B.Ctx(N=len(pool_pos), C=10, T_ae=S['T_ae'][pool_pos], T_s2=S['T_s2'][pool_pos],
                T_xy=km_xy(D.lon[idx], D.lat[idx], lat0), T_nuts3=pd.factorize(D.nuts3[idx])[0],
                F_ae=S['F_ae'], F_s2=S['F_s2'], F_xy=km_xy(D.lon[fr], D.lat[fr], lat0),
                F_y=D.y[fr], F_country=pd.factorize(D.ctry[fr])[0], P_T=S['P_T'][pool_pos].astype(np.float64),
                P_T_loco=S['P_T_loco'][:, pool_pos].astype(np.float32), dom=S['dom'][pool_pos],
                P_F_oof=S['P_F_oof'])
    y_pool = D.y[idx]
    ctx._true_y = y_pool
    ctx.oracle = lambda i: y_pool[np.asarray(i)]
    return ctx


def fit_hook(ctx, S, y_S, seed, **_):
    """注入多轮 margin 与 Hybrid 的正式训练函数：S2 原始特征、300 棵树、外国频率类别权重、目标样本 r 倍、
    用阶段 0 的温度校准。返回 (目标池概率 (N, C), 训练集加权类别比例 (C,))。"""
    raw = ctx._raw
    S = np.asarray(S, dtype=np.int64)
    y_S = np.asarray(y_S, dtype=np.int64)
    X = np.vstack([raw['XF'], raw['XT'][S]])
    y = np.r_[raw['yF'], y_S]
    w = np.r_[raw['cw'][raw['yF']], raw['r'] * raw['cw'][y_S]]
    probs, _ = fit_predict(X, y, w, raw['XT'], raw['threads'], seed)
    z = raw['temp'] * np.log(np.clip(probs.astype(np.float64), 1e-12, None))
    z -= z.max(1, keepdims=True)
    P = np.exp(z)
    P /= P.sum(1, keepdims=True)
    p_tr = np.bincount(y, weights=w, minlength=10) / w.sum()
    return P, p_tr


def oracle_select(name, ctx, k, seed):
    """神谕上界（只在分析中解读，不参与晋级）。"""
    import samplers.base as B
    r = np.random.default_rng(seed)
    y = ctx._true_y
    if name == 'oracle_err':
        err = np.flatnonzero(ctx.P_T.argmax(1) != y)
        pick = r.choice(err, min(k, len(err)), replace=False)
        if len(pick) < k:
            rest = np.setdiff1d(np.arange(ctx.N), pick)
            pick = np.r_[pick, r.choice(rest, k - len(pick), replace=False)]
        return B.finalize(pick, k, ctx.N)
    cls, cnt = np.unique(y, return_counts=True)
    want = {'oracle_prop': cnt.astype(float), 'oracle_sqrt': np.sqrt(cnt), 'oracle_equal': np.ones(len(cls))}[name]
    alloc = np.minimum(np.floor(want / want.sum() * k).astype(int), cnt)
    while alloc.sum() < k:
        room = np.flatnonzero(alloc < cnt)
        alloc[room[np.argmax((want / want.sum() * k - alloc)[room])]] += 1
    pick = np.concatenate([r.choice(np.flatnonzero(y == c), a, replace=False) for c, a in zip(cls, alloc) if a > 0])
    return B.finalize(pick, k, ctx.N)


def run_task(args):
    """一个（国家, 划分, 重复）任务：对所有 (方法, k, r) 选样、训练、评估。"""
    mode, tgt, split, rep, methods, r_map, threads, worker = args
    import samplers.base as B
    if mode == 'run':
        import samplers.designs, samplers.uncertainty, samplers.coverage, samplers.transfer  # noqa: F401 注册方法
    D = run_task.D
    S = dict(np.load(os.path.join(OUT, 'stage0', f'{tgt}_s{split}.npz')))
    allT = S['allT']
    n_cand = len(S['cand'])
    rng = np.random.default_rng(10_000 * split + 100 * rep + 7)
    pool_pos = np.sort(rng.choice(n_cand, min(N_POOL, n_cand), replace=False))  # cand 在 allT 的前 n_cand 个
    test = S['test']
    X, y = D.X[FEAT], D.y
    yT = y[test]
    big = np.flatnonzero(np.bincount(yT, minlength=10) >= MIN_N)
    fr = S['foreign']
    cw = class_weights_from(y[fr])
    rows_path = os.path.join(OUT, f'rows_{mode}_{tgt}_s{split}_r{rep}.csv')  # 每个任务一个文件，避免并发写
    np.save(os.path.join(OUT, 'pool', f'{tgt}_s{split}_r{rep}.npy'), allT[pool_pos])
    rows = pd.read_csv(rows_path).to_dict('records') if os.path.exists(rows_path) else []
    done = {(r_['target'], r_['split'], r_['rep'], r_['method'], r_['k'], r_['r']) for r_ in rows}
    seed = 1000 * split + rep  # 公共随机数：同一单元内所有方法用同一个选样种子与训练种子
    for name in methods:
        for k in BUDGETS:
            for r_mult in r_map[k]:
                key = (tgt, split, rep, name, k, r_mult)
                if key in done:
                    continue
                t0 = time.time()
                ctx = build_ctx(D, S, pool_pos)
                if name not in ORACLE_ALLOWED:
                    ctx.oracle = None
                    del ctx._true_y
                ctx._raw = dict(XF=X[fr], XT=X[allT[pool_pos]], yF=y[fr], cw=cw, r=r_mult, temp=float(S['temp']),
                                threads=threads)
                if name == 'srs_rgrid':  # r 网格专用的简单随机抽样（不依赖注册表）
                    sel_pos = np.sort(np.random.default_rng(seed).choice(ctx.N, k, replace=False))
                elif name.startswith('oracle_'):
                    sel_pos = oracle_select(name, ctx, k, seed)
                else:
                    hp = dict(HP.get(name, {}))
                    if name in NEEDS_FIT:
                        hp['fit_predict'] = fit_hook
                    sel_pos = B.REGISTRY[name](ctx, k, seed, **hp)
                t_sel = time.time() - t0
                sel = allT[pool_pos[sel_pos]]
                meta = ctx.meta.get(name, {})
                w_tr = np.r_[cw[y[fr]], r_mult * cw[y[sel]]]
                probs, _ = fit_predict(np.vstack([X[fr], X[sel]]), np.r_[y[fr], y[sel]], w_tr, X[test],
                                       threads, seed)
                tag = f'{tgt}_s{split}_r{rep}_{name}_k{k}_R{r_mult}'
                np.save(os.path.join(OUT, 'probs', tag + '.npy'), probs.astype(np.float16))
                np.savez_compressed(os.path.join(OUT, 'sel', tag + '.npz'), sel=sel,
                                    pi=np.asarray(meta.get('pi', []))[sel_pos] if 'pi' in meta else np.array([]),
                                    prior_idx=np.asarray(meta.get('prior_idx', [])))
                m = metrics(yT, probs.argmax(1), big)
                rows.append(dict(target=tgt, split=split, rep=rep, method=name, k=k, r=r_mult,
                                 n_classes_sel=len(np.unique(y[sel])), t_select=round(t_sel, 1),
                                 t_total=round(time.time() - t0, 1), **m))
                save_csv(rows, rows_path)
                print(f'[w{worker}] {tag} sel {t_sel:.0f}s total {time.time() - t0:.0f}s', flush=True)
    return len(rows)


def init_worker():
    run_task.D = Data()


def main():
    for d in ('stage0', 'probs', 'sel', 'pool'):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['stage0', 'rgrid', 'run'])
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--r50', type=float, default=None)
    ap.add_argument('--r1000', type=float, default=None)
    ap.add_argument('--methods', type=str, default=None, help='逗号分隔；默认全部登记方法 + 神谕')
    a = ap.parse_args()
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), PLAN), 'rb') as f:
        print(f'{PLAN} sha256[:16] = {hashlib.sha256(f.read()).hexdigest()[:16]}', flush=True)
    threads = max(1, 10 // a.jobs)
    if a.mode == 'stage0':
        tasks = [(tgt, s) for tgt in DEV for s in SPLITS]
        with mp.Pool(a.jobs, initializer=init_worker) as pool:
            pool.starmap(stage0_wrap, [(t, s, threads) for t, s in tasks])
        return
    if a.mode == 'rgrid':
        methods, r_map = ['srs_rgrid'], {k: R_GRID for k in BUDGETS}
    else:
        import samplers.designs, samplers.uncertainty, samplers.coverage, samplers.transfer  # noqa: F401
        import samplers.base as B
        methods = a.methods.split(',') if a.methods else METHODS_A
        missing = [m for m in methods if not m.startswith('oracle_') and m not in B.REGISTRY]
        assert not missing, missing
        r_map = {50: [a.r50], 1000: [a.r1000]}
    tasks = [(tgt, s, rep) for tgt in DEV for s in SPLITS for rep in REPS]
    jobs = [(a.mode, t, s, rep, methods, r_map, threads, i % a.jobs) for i, (t, s, rep) in enumerate(tasks)]
    with mp.Pool(a.jobs, initializer=init_worker) as pool:
        for n in pool.imap_unordered(run_task, jobs):
            print(f'task done ({n} rows in worker file)', flush=True)


def stage0_wrap(tgt, split, threads):
    stage0_one(run_task.D, tgt, split, threads)


if __name__ == '__main__':
    main()
