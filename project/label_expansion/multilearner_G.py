"""
多学习器稳健性分析的运行器（ANALYSIS_PLAN_guidance_rev2.md）：用 LR、MLP 集成、RF 重训 G 已保存的选样（方案 A），不重新选样。

用法：
  python multilearner_G.py --learner lr  --jobs 8 --threads 1
  python multilearner_G.py --learner mlp --jobs 8 --threads 1
  python multilearner_G.py --learner rf  --jobs 2 --threads 4
  python multilearner_G.py --all                               # 依次 lr → mlp → rf（各用上面的进程与线程数）
训练集、权重、测试集、评估与 G 完全相同（guidance_G.evaluate）；每个 (学习器, 单元) 一个 CSV，可断点续跑。
输出：RESULTS_DIR/guidance_G_ml/（ML_OUT 可覆盖）：rows_{learner}_{unit}.csv、probs/{learner}/、lr_C/、manifest.csv
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
import warnings

import numpy as np
import pandas as pd

import guidance_G as G
from guidance_G import E1, MIN_N, R_MULT, save_csv, sel_hash
from stageA import class_weights_from


def _out():
    from common import RESULTS_DIR
    return os.environ.get('ML_OUT', os.path.join(RESULTS_DIR, 'guidance_G_ml'))


OUT = _out()
G_OUT = G.OUT  # 阶段 0 与选样文件的来源（只读）
PLAN = 'ANALYSIS_PLAN_guidance_rev2.md'
LEARNERS = ('lr', 'mlp', 'rf')
RUN_CFG = {'lr': (8, 1), 'mlp': (8, 1), 'rf': (2, 4)}  # (进程数, 每进程线程数)
C_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0, 10.0)
# 超参数（方案 rev2 第 1 节）；测试可在导入后覆盖
PARAMS = {
    'lr': dict(max_iter=1000, tol=1e-4),
    'mlp': dict(hidden_layer_sizes=(256, 128), alpha=1e-4, batch_size=256, learning_rate_init=1e-3, max_iter=20,
                early_stopping=False, n_iter_no_change=10 ** 9, tol=0.0),
    'mlp_seeds': 3,
    'rf': dict(n_estimators=500, max_features='sqrt', min_samples_leaf=1, bootstrap=True, class_weight=None),
}
SRS_KS = (50, 100, 200, 400)
E1_KS = (50, 200)
TIER1_INDEX = os.path.expanduser('~/AlphaCrops_local/verify/multilearner/feasibility_compute/tier1_sel_index.csv')
THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
               'NUMEXPR_NUM_THREADS')


def params_hash(learner, C=None):
    """学习器超参数（LR 另含 C）的哈希，写进概率缓存并在命中时核对。"""
    spec = {'lr': PARAMS['lr'], 'mlp': [PARAMS['mlp'], PARAMS['mlp_seeds']], 'rf': PARAMS['rf']}[learner]
    return hashlib.md5(json.dumps([learner, spec, C], sort_keys=True, default=str).encode()).hexdigest()[:16]


def mlp_seed(seed, s):
    """MLP 集成的第 s 个种子：10·seed + s，使不同单元（含相邻重复）的种子不重叠。"""
    return 10 * seed + s


# ---------------------------------------------------------------- 作业清单（方案 A）
def unit_jobs(split, rep, sel_dir):
    """一个 (国家, 划分, 重复) 的作业：k=0（每种特征一次）、E1（AE/S2 指导 × 4 方法 × k∈{50,200}）、SRS（4 档 k）。
    返回 [(cond, method, k)]，由调用方补上国家与单元名以定位选样文件。"""
    J = [('none', 'foreign_only', 0)]
    J += [(c, m, k) for c in ('AE', 'S2') for m in E1 for k in E1_KS]
    J += [('none', 'srs', k) for k in SRS_KS]
    return J


def sel_path(tgt, split, rep, cond, method, k):
    return os.path.join(G_OUT, 'sel', f'{tgt}_s{split}_r{rep}_{cond}_{method}_k{k}.npz')


def build_manifest():
    """列出本次用到的全部选样文件及其哈希（缺文件直接报错）。"""
    rows = []
    for tgt, split, rep in G.unit_order():
        for cond, method, k in unit_jobs(split, rep, None):
            if k == 0:
                continue
            p = sel_path(tgt, split, rep, cond, method, k)
            assert os.path.exists(p), p
            sel = np.load(p)['sel']
            rows.append(dict(tgt=tgt, split=split, rep=rep, cond=cond, method=method, k=k, h=sel_hash(sel), n=len(sel)))
    M = pd.DataFrame(rows)
    assert len(M) == 880, len(M)  # 704 个 E1 + 176 个 SRS
    if os.path.exists(TIER1_INDEX):  # 与冻结前生成的 tier1_sel_index.csv 的对应子集逐行比对
        T = pd.read_csv(TIER1_INDEX).rename(columns={'h': 'h_idx'})
        J = M.merge(T[['tgt', 'split', 'rep', 'cond', 'method', 'k', 'h_idx']], how='left',
                    on=['tgt', 'split', 'rep', 'cond', 'method', 'k'])
        assert J.h_idx.notna().all() and (J.h == J.h_idx).all(), '选样与 tier1_sel_index.csv 不一致'
    return M


# ---------------------------------------------------------------- 学习器
def _std(A):
    mu, sd = A.mean(0), A.std(0) + 1e-6
    return mu, sd


def _full(model, B, C=10):
    P = np.zeros((len(B), C))
    P[:, model.classes_] = model.predict_proba(B)
    return P


def fit_predict_learner(learner, X_tr, y_tr, w_tr, X_te, seed, threads, C=None):
    """返回 (测试集概率 (n, 10) float32, 信息字典)。LR 与 MLP 在训练集上标准化；RF 不标准化。"""
    from threadpoolctl import threadpool_limits
    info = {}
    X_tr = np.asarray(X_tr, dtype=np.float64)
    X_te = np.asarray(X_te, dtype=np.float64)
    with threadpool_limits(limits=threads), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if learner == 'lr':
            from sklearn.linear_model import LogisticRegression
            mu, sd = _std(X_tr)
            m = LogisticRegression(C=C, **PARAMS['lr']).fit((X_tr - mu) / sd, y_tr, sample_weight=w_tr / w_tr.mean())
            P = _full(m, (X_te - mu) / sd)
            info['n_iter'] = int(np.max(m.n_iter_))
            info['converged'] = bool(np.max(m.n_iter_) < PARAMS['lr']['max_iter'])
        elif learner == 'mlp':
            from sklearn.neural_network import MLPClassifier
            mu, sd = _std(X_tr)
            A, B = (X_tr - mu) / sd, (X_te - mu) / sd
            P = np.zeros((len(B), 10))
            its = []
            for s in range(PARAMS['mlp_seeds']):
                m = MLPClassifier(random_state=mlp_seed(seed, s), **PARAMS['mlp']).fit(A, y_tr,
                                                                                       sample_weight=w_tr / w_tr.mean())
                P += _full(m, B)
                its.append(int(m.n_iter_))
            P /= PARAMS['mlp_seeds']
            info['n_iter_min'] = min(its)
        elif learner == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            m = RandomForestClassifier(random_state=seed, n_jobs=threads, **PARAMS['rf']).fit(X_tr, y_tr, sample_weight=w_tr)
            P = _full(m, X_te)
        else:
            raise ValueError(learner)
    from sklearn.exceptions import ConvergenceWarning
    info['n_convwarn'] = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    info['n_otherwarn'] = len(caught) - info['n_convwarn']
    P = P.astype(np.float32)
    if not np.isfinite(P).all():
        raise FloatingPointError(f'{learner}: 概率中有 NaN/Inf')
    return P, info


def select_C(D, S, tgt, split, feat, threads, out=None):
    """LR 的 C：只用外国池，按 fold_groups 的 5 组留国家组，加权对数损失最小者（并列取较小 C）。结果写入 lr_C/。"""
    from threadpoolctl import threadpool_limits
    from sklearn.linear_model import LogisticRegression
    out = out or OUT
    path = os.path.join(out, 'lr_C', f'{tgt}_s{split}_{feat}.json')
    spec = dict(grid=list(C_GRID), lr=PARAMS['lr'])
    if os.path.exists(path):
        js = json.load(open(path))
        assert js.get('spec') == spec, f'{path} 的网格或参数与当前不一致'
        return js['C']
    fr = S['foreign']
    y = D.y[fr]
    X = np.asarray(D.X[feat][fr], dtype=np.float64)
    cw = class_weights_from(y)
    w = cw[y]
    groups = G.fold_groups(D, tgt, split, fr)
    num = {C: 0.0 for C in C_GRID}
    iters = {C: [] for C in C_GRID}
    den = 0.0
    with threadpool_limits(limits=threads), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for g in np.unique(groups):
            tr, te = groups != g, groups == g
            mu, sd = _std(X[tr])
            A, B = (X[tr] - mu) / sd, (X[te] - mu) / sd
            for C in C_GRID:
                m = LogisticRegression(C=C, **PARAMS['lr']).fit(A, y[tr], sample_weight=w[tr] / w[tr].mean())
                iters[C].append(int(np.max(m.n_iter_)))
                P = np.clip(_full(m, B), 1e-12, None)
                num[C] += float(-(w[te] * np.log(P[np.arange(te.sum()), y[te]])).sum())
            den += float(w[te].sum())
    loss = {C: num[C] / den for C in C_GRID}
    best = min(C_GRID, key=lambda C: (round(loss[C], 12), C))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.{os.getpid()}.tmp'
    json.dump(dict(C=best, spec=spec, boundary=best in (C_GRID[0], C_GRID[-1]),
                   loss={str(k): v for k, v in loss.items()}, n_iter={str(k): v for k, v in iters.items()}),
              open(tmp, 'w'))
    os.replace(tmp, path)
    return best


# ---------------------------------------------------------------- 一个 (学习器, 单元)
def run_unit_ml(D, learner, tgt, split, rep, threads, out=None, log=print):
    out = out or OUT
    os.makedirs(os.path.join(out, 'probs', learner), exist_ok=True)
    unit = f'{tgt}_s{split}_r{rep}'
    S = dict(np.load(os.path.join(G_OUT, 'stage0', f'{tgt}_s{split}.npz')))
    pool_pos = G.pool_positions(S, split, rep)
    idx_pool = S['allT'][pool_pos]
    test, fr = S['test'], S['foreign']
    yT = D.y[test]
    big = np.flatnonzero(np.bincount(yT, minlength=10) >= MIN_N)
    efar = S['efar'].astype(bool)
    cw = class_weights_from(D.y[fr])
    pi_pool = np.bincount(D.y[idx_pool], minlength=10) / len(idx_pool)
    seed = 1000 * split + rep
    rows_path = os.path.join(out, f'rows_{learner}_{unit}.csv')
    rows = pd.read_csv(rows_path).to_dict('records') if os.path.exists(rows_path) else []
    done = {(r_['cond'], r_['method'], r_['k'], r_['clf']) for r_ in rows}
    Cs = {f: select_C(D, S, tgt, split, f, threads, out) for f in ('ae', 's2')} if learner == 'lr' else {}

    for cond, method, k in unit_jobs(split, rep, None):
        if k == 0:
            sel, pi_sel = np.array([], np.int64), np.array([])
        else:
            z = np.load(sel_path(tgt, split, rep, cond, method, k))
            sel, pi_sel = z['sel'], z['pi_sel']
        assert not np.intersect1d(sel, test).size and not np.intersect1d(sel, fr).size
        h = sel_hash(sel) if k else 'k0'
        ys = D.y[sel]
        w_tr = np.r_[cw[D.y[fr]], R_MULT * cw[ys]]
        pi_train = np.bincount(np.r_[D.y[fr], ys], weights=w_tr, minlength=10) / w_tr.sum()
        pi_des = None
        if len(pi_sel):
            ipw = 1 / pi_sel
            pi_des = np.bincount(ys, weights=ipw, minlength=10) / ipw.sum()
        for clf in ('s2', 'ae'):
            if (cond, method, k, clf) in done:
                continue
            t1 = time.time()
            # k=0：每个 (国家, 划分, 特征) 只拟合一次，两个重复共用（种子取 rep=0 的值）
            pp = (os.path.join(out, 'probs', learner, f'{tgt}_s{split}_{clf}_k0.npz') if k == 0 else
                  os.path.join(out, 'probs', learner, f'{unit}_{clf}_k{k}_{h}.npz'))
            ph = params_hash(learner, Cs.get(clf))
            hit = os.path.exists(pp)
            status, err = 'ok', ''
            try:
                if hit:
                    z = np.load(pp)
                    assert str(z['params']) == ph, f'{pp} 的超参数与当前不一致'
                    P, info = z['P'], json.loads(str(z['info']))
                else:
                    X = D.X[clf]
                    P, info = fit_predict_learner(learner, np.vstack([X[fr], X[sel]]), np.r_[D.y[fr], ys], w_tr,
                                                  X[test], 1000 * split if k == 0 else seed, threads, Cs.get(clf))
                    tmp = f'{pp}.{os.getpid()}.tmp.npz'
                    np.savez_compressed(tmp, P=P, info=json.dumps(info), params=ph)
                    os.replace(tmp, pp)
                m = G.evaluate({0: P}, yT, big, efar, pi_train, pi_pool, pi_des)
                for key in ('oa_i0', 'ba_i0', 'f1_i0'):
                    m.pop(key, None)
            except (FloatingPointError, ValueError, np.linalg.LinAlgError) as e:  # 方案 rev2：技术失败剔除，不中止运行
                status, err, info, m = 'failed', repr(e)[:300], {}, {}
            rows.append(dict(learner=learner, cond=cond, method=method, clf=clf, k=k, target=tgt, split=split, rep=rep,
                             status=status, error=err, params_hash=ph,
                             sel_hash=h, C=Cs.get(clf, np.nan), t_fit=round(time.time() - t1, 1), cache_hit=hit,
                             n_classes_sel=len(np.unique(ys)),
                             comp_L1=float(np.abs(np.bincount(ys, minlength=10) / k - pi_pool).sum()) if k else np.nan,
                             **{f'info_{a}': b for a, b in info.items()}, **m))
            save_csv(rows, rows_path)
            log(f'{learner} {unit} {cond} {method} k{k} {clf} {time.time() - t1:.1f}s{" (cache)" if hit else ""}')
    return rows


# ---------------------------------------------------------------- 进程与主程序
def init_worker():
    run_task.D = G.GData()


def run_task(args):
    learner, tgt, split, rep, threads, worker = args
    rows = run_unit_ml(run_task.D, learner, tgt, split, rep, threads,
                       log=lambda s: print(f'[w{worker}] {s}', flush=True))
    return f'{learner} {tgt} s{split} r{rep}: {len(rows)} rows'


def c_task(args):
    tgt, split, feat, threads = args
    S = dict(np.load(os.path.join(G_OUT, 'stage0', f'{tgt}_s{split}.npz')))
    return f'C {tgt} s{split} {feat} = {select_C(run_task.D, S, tgt, split, feat, threads)}'


def file_hash(name):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), name), 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def check_frozen():
    """方案、运行器、分析脚本的哈希必须都已记入 plan_hashes.txt（冻结后才能正式运行）。"""
    rec = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plan_hashes.txt')).read()
    for name in (PLAN, 'multilearner_G.py', 'analyze_ML.py'):
        h = file_hash(name)
        assert f'{name} sha256[:16]={h}' in rec, f'{name} 的当前哈希 {h} 不在 plan_hashes.txt 中（未冻结或冻结后被改动）'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--learner', choices=LEARNERS)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--jobs', type=int, default=None)
    ap.add_argument('--threads', type=int, default=None)
    ap.add_argument('--unfrozen', action='store_true', help='只用于冒烟测试：跳过冻结校验')
    a = ap.parse_args()
    assert not (a.all and (a.jobs or a.threads)), '--all 使用方案规定的进程与线程数，不接受覆盖'
    if not a.unfrozen:
        check_frozen()
    os.makedirs(OUT, exist_ok=True)
    print(f'{PLAN} sha256[:16] = {file_hash(PLAN)}；multilearner_G.py = {file_hash("multilearner_G.py")}；'
          f'analyze_ML.py = {file_hash("analyze_ML.py")}；输出 {OUT}', flush=True)
    G.GData.build_cache()
    M = build_manifest()
    M.to_csv(os.path.join(OUT, 'manifest.csv'), index=False)
    print(f'选样清单：{len(M)} 个，sha256[:16] = '
          f'{hashlib.sha256(M.to_csv(index=False).encode()).hexdigest()[:16]}', flush=True)
    learners = LEARNERS if a.all else (a.learner,)
    for learner in learners:
        jobs, threads = RUN_CFG[learner]
        jobs, threads = a.jobs or jobs, a.threads or threads
        for v in THREAD_VARS:  # spawn 出的子进程在导入 numpy 之前继承这些变量（BLAS 为 Accelerate，运行时再设无效）
            os.environ[v] = str(threads)
        print(f'{learner}：{jobs} 个进程 × {threads} 线程', flush=True)
        with mp.Pool(jobs, initializer=init_worker) as pool:
            if learner == 'lr':  # 先为全部 (国家, 划分, 特征) 选好 C，避免多个单元重复计算
                ct = [(t, s, f, threads) for t in G.COUNTRIES_G for s in G.SPLITS for f in ('ae', 's2')]
                for msg in pool.imap_unordered(c_task, ct):
                    print(msg, flush=True)
            tasks = [(learner, t, s, r, threads, i % jobs) for i, (t, s, r) in enumerate(G.unit_order())]
            for msg in pool.imap_unordered(run_task, tasks):
                print(f'task done: {msg}', flush=True)


if __name__ == '__main__':
    main()
