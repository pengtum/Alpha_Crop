"""同分布留出的 ba 口径核对：fit 代理的 hold（外国随机 80/20，rng 12345+split）只给合并 ba；
这里重做 3 个单元，另算逐国 ba 再平均，与阶段 0 留组袋外的逐国 ba 同口径比较。300 轮与 100 轮。"""
import json, os, sys, time
PROJ = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np
OUT = '/Users/perng_mini/AlphaCrops_local/verify/model_diag/verify_fit'
GRES = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'


def ba(y, ph):
    big = np.flatnonzero(np.bincount(y, minlength=10) >= 30)
    return 100 * float(np.mean([(ph[y == c] == c).mean() for c in big]))


def run(job):
    import lightgbm as lgb
    import guidance_G as G
    from stageA import class_weights_from, lgb_params
    D = run.D
    tgt, split, feat = job['tgt'], job['split'], job['feat']
    S = np.load(os.path.join(GRES, 'stage0', f'{tgt}_s{split}.npz'))
    fr = S['foreign']; yF = D.y[fr]; X = D.X[feat]
    perm = np.random.default_rng(12345 + split).permutation(len(fr)); ntr = int(0.8 * len(fr))
    a, b = perm[:ntr], perm[ntr:]
    cw = class_weights_from(yF[a])
    m = lgb.train(lgb_params(2, 0), lgb.Dataset(np.asarray(X[fr[a]]), yF[a], weight=cw[yF[a]]), num_boost_round=300)
    yb, cb = yF[b], D.ctry[fr[b]]
    oof = S[f'P_F_oof_{feat}'].argmax(1)
    res = dict(job=job)
    for it in (100, 300):
        ph = m.predict(np.asarray(X[fr[b]]), num_iteration=it).argmax(1)
        res[it] = dict(oa=100 * float((ph == yb).mean()), ba_pooled=ba(yb, ph),
                       ba_percountry=float(np.mean([ba(yb[cb == c], ph[cb == c]) for c in np.unique(cb)])))
    # 阶段 0 留组袋外，只取同一批 20% 行，逐国
    res['oof_same_rows'] = dict(oa=100 * float((oof[b] == yb).mean()), ba_pooled=ba(yb, oof[b]),
                                ba_percountry=float(np.mean([ba(yb[cb == c], oof[b][cb == c]) for c in np.unique(cb)])))
    return res


def worker(job):
    name = 'hold_' + '_'.join(str(job[x]) for x in ('tgt', 'split', 'feat'))
    path = os.path.join(OUT, 'jobs', name + '.json')
    r = run(job)
    json.dump(r, open(path, 'w'))
    return name


def init():
    import guidance_G as G
    run.D = G.GData()


if __name__ == '__main__':
    import multiprocessing as mp
    jobs = [dict(tgt=c, split=0, feat=f) for c in ('AT', 'PT', 'SE') for f in ('ae', 's2')]
    with mp.Pool(2, initializer=init) as pool:
        for msg in pool.imap_unordered(worker, jobs):
            print(time.strftime('%H:%M:%S'), msg, flush=True)
