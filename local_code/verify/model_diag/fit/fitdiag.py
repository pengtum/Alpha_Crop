"""
拟合诊断（只读项目代码与已有结果；输出只写到本目录）。
作业类型：
  k0     外国 5 万（G 配置，种子 0，行顺序与 G 相同）→ 训练集、目标测试集的逐检查点指标
  hold   外国随机 80/20（同分布留出）→ 训练集、外国留出、目标测试集
  tgt    只用目标候选池（pool_positions(S, split, 0)，类别权重按池频率）→ 目标测试集
  fp     外国 5 万 + 全部候选池（r=1，类别权重按外国频率）→ 目标测试集
  cap    容量扫描：训练集 k0 或 k200（SRS, rep 0, r=50），配置 L15/L63/L255/reg
每个作业写 jobs/<name>.json；已存在则跳过（可断点续跑）。
"""
import json
import os
import sys
import time

PROJ = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, PROJ)
os.chdir(PROJ)
import numpy as np  # noqa: E402

OUT = '/Users/perng_mini/AlphaCrops_local/verify/model_diag/fit'
GRES = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
CPS = [25, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1500, 2000]
CONFIGS = {  # 相对 stageA.lgb_params 的覆盖
    'L63': {},
    'L15': {'num_leaves': 15},
    'L255': {'num_leaves': 255},
    'reg': {'min_data_in_leaf': 100, 'lambda_l2': 10.0},
    'reg2': {'min_data_in_leaf': 200, 'lambda_l2': 10.0, 'feature_fraction': 0.5},
}


def ba_oa(y, P, min_n=30, big=None):
    ph = P.argmax(1)
    cnt = np.bincount(y, minlength=10)
    if big is None:
        big = np.flatnonzero(cnt >= min_n)
    rec = np.array([(ph[y == c] == c).mean() if cnt[c] else np.nan for c in range(10)])
    ll = float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-15, None)).mean())
    return dict(oa=100 * float((ph == y).mean()), ba=100 * float(np.mean(rec[big])), nll=ll,
                rec=[None if np.isnan(r) else float(r) for r in rec], big=[int(b) for b in big])


def fit_curve(X_tr, y_tr, w_tr, evals, rounds, seed, threads, over=None, keep_probs=()):
    """evals: {name: (X, y)}；'train' 自动加入。返回 {name: {it: metrics}}，外加逐轮 multi_logloss。
    keep_probs: [(name, it)] 需要返回概率的检查点（用于核对）。"""
    import lightgbm as lgb
    from stageA import lgb_params
    p = lgb_params(threads, seed)
    p.update(over or {})
    p['metric'] = 'multi_logloss'
    dtr = lgb.Dataset(X_tr, y_tr, weight=w_tr, free_raw_data=False)
    vs, names, ys = [dtr], ['train'], {'train': np.asarray(y_tr)}
    for n, (X, y) in evals.items():
        vs.append(lgb.Dataset(X, y, reference=dtr, free_raw_data=False))
        names.append(n)
        ys[n] = np.asarray(y)
    idmap = {}
    cnt = {}
    store = {n: {} for n in names}
    probs = {}
    cps = set(c for c in CPS if c <= rounds) | {rounds}

    def fe(preds, data):
        n = idmap.get(id(data))
        if n is None:  # 首次调用时按 Dataset 对象绑定名字
            n = names[len(idmap)] if len(idmap) < len(names) else '?'
        cnt[n] = cnt.get(n, 0) + 1
        it = cnt[n]
        if it in cps:
            P = np.asarray(preds, dtype=np.float64).reshape(len(ys[n]), 10) if preds.ndim == 1 else np.asarray(preds)
            store[n][it] = ba_oa(ys[n], P, big=store[n].get('_big'))
            if (n, it) in keep_probs:
                probs[(n, it)] = P.astype(np.float32)
        return 'dummy', 0.0, True

    rec = {}
    for d, n in zip(vs, names):
        idmap[id(d)] = n
    t0 = time.time()
    lgb.train(p, dtr, num_boost_round=rounds, valid_sets=vs, valid_names=names, feval=fe,
              callbacks=[lgb.record_evaluation(rec)])
    t = time.time() - t0
    ll = {n: [float(v) for v in rec[n]['multi_logloss']] for n in names}
    return store, ll, probs, t


def job_run(job, D=None):
    import guidance_G as G
    from stageA import class_weights_from
    kind, tgt, split, feat, cfg, rounds, trset = job['kind'], job['tgt'], job['split'], job['feat'], \
        job.get('cfg', 'L63'), job['rounds'], job.get('trset', 'k0')
    threads = job.get('threads', 2)
    D = D or G.GData()
    S = np.load(os.path.join(GRES, 'stage0', f'{tgt}_s{split}.npz'))
    fr, test, allT = S['foreign'], S['test'], S['allT']
    X = D.X[feat]
    yF = D.y[fr]
    cw = class_weights_from(yF)
    XT, yT = np.asarray(X[test]), D.y[test]
    evals = {'test': (XT, yT)}
    extra = {}
    keep = ()
    if kind in ('k0', 'cap') and trset == 'k0':
        Xtr, ytr, wtr, seed = np.asarray(X[fr]), yF, cw[yF], 0
        if cfg == 'L63':
            keep = (('test', 300),)
    elif (kind == 'cap' and trset == 'k200') or kind == 'guide':
        sname = f'{tgt}_s{split}_r0_none_srs_k200.npz' if kind == 'cap' else \
            f"{tgt}_s{split}_r0_{job['cond']}_{job['method']}_k{job['k']}.npz"
        sel = np.load(os.path.join(GRES, 'sel', sname))['sel']
        ys = D.y[sel]
        Xtr = np.vstack([X[fr], X[sel]])
        ytr = np.r_[yF, ys]
        wtr = np.r_[cw[yF], 50.0 * cw[ys]]
        seed = 1000 * split + 0
        extra['sel_hash'] = G.sel_hash(sel)
        if cfg == 'L63':
            keep = (('test', 300),)
    elif kind == 'hold':
        rng = np.random.default_rng(12345 + split)
        perm = rng.permutation(len(fr))
        ntr = int(0.8 * len(fr))
        a, b = perm[:ntr], perm[ntr:]
        cwh = class_weights_from(yF[a])
        Xtr, ytr, wtr, seed = np.asarray(X[fr[a]]), yF[a], cwh[yF[a]], 0
        evals['fhold'] = (np.asarray(X[fr[b]]), yF[b])
    elif kind == 'tgt':
        pp = G.pool_positions(S, split, 0)
        idx = allT[pp]
        yp = D.y[idx]
        cwp = class_weights_from(yp)
        Xtr, ytr, wtr, seed = np.asarray(X[idx]), yp, cwp[yp], 0
        extra['n_pool'] = int(len(idx))
    elif kind == 'fp':
        pp = G.pool_positions(S, split, 0)
        idx = allT[pp]
        yp = D.y[idx]
        Xtr = np.vstack([X[fr], X[idx]])
        ytr = np.r_[yF, yp]
        wtr = np.r_[cw[yF], job.get('r', 1.0) * cw[yp]]
        seed = 0
        extra['n_pool'] = int(len(idx))
    else:
        raise ValueError(job)
    # 目标测试集的 big 类（>=30）固定为 G 的定义
    store, ll, probs, t = fit_curve(Xtr, ytr, wtr, evals, rounds, seed, threads, CONFIGS[cfg], keep)
    res = dict(job=job, t=t, ll=ll, metrics={n: {str(k): v for k, v in d.items()} for n, d in store.items()},
               n_train=int(len(ytr)), **extra)
    # 核对：k0 L63 第 300 轮与阶段 0 的 P0test_*_i300；k200 与 G probs 缓存
    if ('test', 300) in probs:
        P = probs[('test', 300)]
        if trset == 'k0':
            ref = S[f'P0test_{feat}_i300'].astype(np.float32)
        else:
            z = np.load(os.path.join(GRES, 'probs', f'{tgt}_s{split}_r0_{feat}_k200_{extra["sel_hash"]}.npz'))
            ref = z['i300']
        res['check_maxabs'] = float(np.abs(P - ref).max())
        res['check_argmax_agree'] = float((P.argmax(1) == ref.argmax(1)).mean())
    return res


def name_of(job):
    return '_'.join(str(job[k]) for k in ('kind', 'tgt', 'split', 'feat')) + \
        f"_{job.get('trset', 'k0')}_{job.get('cfg', 'L63')}_R{job['rounds']}" + (f"_r{job['r']}" if 'r' in job else '') + \
        (f"_{job['cond']}_{job['method']}_k{job['k']}" if job['kind'] == 'guide' else '')


def worker(args):
    job, = args
    path = os.path.join(OUT, 'jobs', name_of(job) + '.json')
    if os.path.exists(path):
        return name_of(job) + ' (skip)'
    D = worker.D
    t0 = time.time()
    res = job_run(job, D)
    tmp = path + '.tmp'
    json.dump(res, open(tmp, 'w'))
    os.replace(tmp, path)
    return f'{name_of(job)} {time.time() - t0:.0f}s'


def init_worker():
    import guidance_G as G
    worker.D = G.GData()


if __name__ == '__main__':
    import multiprocessing as mp
    os.makedirs(os.path.join(OUT, 'jobs'), exist_ok=True)
    jobs = json.load(open(sys.argv[1]))
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    with mp.Pool(nproc, initializer=init_worker) as pool:
        for msg in pool.imap(worker, [(j,) for j in jobs]):
            print(time.strftime('%H:%M:%S'), msg, flush=True)
