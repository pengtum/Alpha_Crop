"""独立复核（视角 verify_fit）：划分 1 上重训，检验 fit 代理的关键结论。
不复用 fit 代理的 fitdiag.py；直接 lgb.train + predict(num_iteration=it)。只读项目代码与结果。
作业：
  k0      外国 5 万，类别权重按外国频率（G 配置），种子 0            -> 与阶段 0 P0test_i300 核对
  tgt     只用目标候选池（pool_positions(S,1,0)），类别权重按池频率
  k0L15   同 k0，num_leaves=15
  k0uw    同 k0，不加类别权重（样本权重全 1）
  tgtfw   只用目标候选池，类别权重按外国频率（与 G 相同的权重来源）
  tgt400  只用目标 SRS k=400 的标签（G 的 sel 文件，rep 0），类别权重按这 400 个的频率
"""
import json, os, sys, time
PROJ = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np
OUT = '/Users/perng_mini/AlphaCrops_local/verify/model_diag/verify_fit'
GRES = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
CPS = [25, 50, 100, 150, 200, 250, 300]
THREADS = 2


def metr(y, P, big):
    ph = P.argmax(1)
    rec = np.array([(ph[y == c] == c).mean() if (y == c).any() else np.nan for c in range(10)])
    nll = float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-15, None)).mean())
    return dict(oa=100 * float((ph == y).mean()), ba=100 * float(np.mean(rec[big])), nll=nll,
                rec=[None if np.isnan(r) else round(float(r), 4) for r in rec])


def run(job):
    import lightgbm as lgb
    import guidance_G as G
    from stageA import class_weights_from, lgb_params
    D = run.D
    kind, tgt, split, feat = job['kind'], job['tgt'], job['split'], job['feat']
    S = np.load(os.path.join(GRES, 'stage0', f'{tgt}_s{split}.npz'))
    fr, test, allT = S['foreign'], S['test'], S['allT']
    X = D.X[feat]
    yF = D.y[fr]
    cwF = class_weights_from(yF)
    yT = D.y[test]
    big = np.flatnonzero(np.bincount(yT, minlength=10) >= 30)
    p = lgb_params(THREADS, 0)
    if kind in ('k0', 'k0L15', 'k0uw'):
        Xtr, ytr = np.asarray(X[fr]), yF
        w = np.ones(len(yF)) if kind == 'k0uw' else cwF[yF]
        if kind == 'k0L15':
            p['num_leaves'] = 15
    elif kind in ('tgt', 'tgtfw'):
        idx = allT[G.pool_positions(S, split, 0)]
        assert not np.isin(idx, test).any()
        Xtr, ytr = np.asarray(X[idx]), D.y[idx]
        w = (class_weights_from(ytr) if kind == 'tgt' else cwF)[ytr]
    elif kind == 'tgt400':
        sel = np.load(os.path.join(GRES, 'sel', f'{tgt}_s{split}_r0_none_srs_k400.npz'))['sel']
        assert not np.isin(sel, test).any()
        Xtr, ytr = np.asarray(X[sel]), D.y[sel]
        w = class_weights_from(ytr)[ytr]
    t0 = time.time()
    m = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=w, free_raw_data=True), num_boost_round=300)
    t = time.time() - t0
    XT = np.asarray(X[test])
    res = dict(job=job, t=t, n_train=int(len(ytr)), big=[int(b) for b in big], test={}, train={})
    for it in CPS:
        P = m.predict(XT, num_iteration=it)
        res['test'][it] = metr(yT, P, big)
        if it == 300 and kind == 'k0':
            ref = S[f'P0test_{feat}_i300'].astype(np.float32)
            res['check_maxabs'] = float(np.abs(P - ref).max())
            res['check_agree'] = float((P.argmax(1) == ref.argmax(1)).mean())
    bigtr = np.flatnonzero(np.bincount(ytr, minlength=10) >= 30)
    for it in (100, 300):
        res['train'][it] = metr(ytr, m.predict(Xtr, num_iteration=it), bigtr)
    return res


def worker(job):
    name = '_'.join(str(job[k]) for k in ('kind', 'tgt', 'split', 'feat'))
    path = os.path.join(OUT, 'jobs', name + '.json')
    if os.path.exists(path):
        return name + ' skip'
    r = run(job)
    json.dump(r, open(path + '.tmp', 'w')); os.replace(path + '.tmp', path)
    return f'{name} {r["t"]:.0f}s'


def init():
    import guidance_G as G
    run.D = G.GData()


if __name__ == '__main__':
    import multiprocessing as mp
    os.makedirs(os.path.join(OUT, 'jobs'), exist_ok=True)
    C11 = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']
    C5 = ['AT', 'DE', 'LV', 'PT', 'SE']
    jobs = []
    for kind, cs in (('tgt400', C5), ('k0', C11), ('tgt', C11), ('k0uw', C5), ('k0L15', C5), ('tgtfw', C5)):
        for c in cs:
            for f in ('ae', 's2'):
                jobs.append(dict(kind=kind, tgt=c, split=1, feat=f))
    with mp.Pool(2, initializer=init) as pool:
        for msg in pool.imap_unordered(worker, jobs):
            print(time.strftime('%H:%M:%S'), msg, flush=True)
