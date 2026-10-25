"""独立复核 θ(AE−S2) 在低轮数下是否仍≈0：换一个子集（划分 1、重复 0、k=50、kcenter（为省时只用一种方法）），
这与 fit 代理用的子集（划分 0、k=200、lpm2_nogeo + typiclust）不重叠。训练到 200 轮，第 200 轮与 G 的 probs 缓存核对。"""
import json, os, sys, time
PROJ = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np
OUT = '/Users/perng_mini/AlphaCrops_local/verify/model_diag/verify_fit'
GRES = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
CPS = [25, 50, 100, 150, 200]


def run(job):
    import lightgbm as lgb
    import guidance_G as G
    from stageA import class_weights_from, lgb_params
    D = run.D
    tgt, split, rep, cond, method, k, clf = (job[x] for x in ('tgt', 'split', 'rep', 'cond', 'method', 'k', 'clf'))
    S = np.load(os.path.join(GRES, 'stage0', f'{tgt}_s{split}.npz'))
    fr, test = S['foreign'], S['test']
    sel = np.load(os.path.join(GRES, 'sel', f'{tgt}_s{split}_r{rep}_{cond}_{method}_k{k}.npz'))['sel']
    X = D.X[clf]
    yF, ys, yT = D.y[fr], D.y[sel], D.y[test]
    cw = class_weights_from(yF)
    big = np.flatnonzero(np.bincount(yT, minlength=10) >= 30)
    w = np.r_[cw[yF], 50.0 * cw[ys]]
    m = lgb.train(lgb_params(2, 1000 * split + rep), lgb.Dataset(np.vstack([X[fr], X[sel]]), np.r_[yF, ys], weight=w),
                  num_boost_round=200)
    XT = np.asarray(X[test])
    res = dict(job=job, test={})
    for it in CPS:
        P = m.predict(XT, num_iteration=it)
        ph = P.argmax(1)
        rec = np.array([(ph[yT == c] == c).mean() for c in big])
        res['test'][it] = dict(ba=100 * float(rec.mean()), oa=100 * float((ph == yT).mean()))
        if it == 200:
            z = np.load(os.path.join(GRES, 'probs', f'{tgt}_s{split}_r{rep}_{clf}_k{k}_{G.sel_hash(sel)}.npz'))
            res['check_maxabs'] = float(np.abs(P.astype(np.float32) - z['i200']).max())
    return res


def worker(job):
    name = 'theta_' + '_'.join(str(job[x]) for x in ('tgt', 'split', 'rep', 'cond', 'method', 'k', 'clf'))
    path = os.path.join(OUT, 'jobs', name + '.json')
    if os.path.exists(path):
        return name + ' skip'
    t0 = time.time()
    r = run(job)
    json.dump(r, open(path + '.tmp', 'w')); os.replace(path + '.tmp', path)
    return f'{name} {time.time() - t0:.0f}s maxabs {r["check_maxabs"]:.2g}'


def init():
    import guidance_G as G
    run.D = G.GData()


if __name__ == '__main__':
    import multiprocessing as mp
    C11 = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']
    jobs = [dict(tgt=c, split=1, rep=0, cond=cd, method=mt, k=50, clf=cl) for c in C11 for mt in ('kcenter',)
            for cd in ('AE', 'S2') for cl in ('ae', 's2')]
    with mp.Pool(2, initializer=init) as pool:
        for msg in pool.imap_unordered(worker, jobs):
            print(time.strftime('%H:%M:%S'), msg, flush=True)
