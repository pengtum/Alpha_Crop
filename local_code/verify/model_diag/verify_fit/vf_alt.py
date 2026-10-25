"""独立小实验：换模型家族（sklearn 逻辑回归、MLP）能否缩小"只用目标池 − 只用外国(k0)"的差距？
划分 1，5 国（AT, DE, LV, PT, SE）× {ae, s2} × {k0（外国 5 万，外国频率类别权重）, tgt（目标池 3 万，池频率类别权重）}。
MLP：标准化，(256,128)，adam 1e-3，batch 256，alpha 1e-4，partial_fit 逐轮（Adam 状态保留），记录 40 轮内的测试/训练曲线。"""
import json, os, sys, time, warnings
PROJ = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np
OUT = '/Users/perng_mini/AlphaCrops_local/verify/model_diag/verify_fit'
GRES = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
EPS = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40]


def metr(y, ph, big):
    rec = np.array([(ph[y == c] == c).mean() for c in big])
    return dict(oa=100 * float((ph == y).mean()), ba=100 * float(rec.mean()))


def run(job):
    import guidance_G as G
    from stageA import class_weights_from
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    D = run.D
    model, tset, tgt, split, feat = (job[x] for x in ('model', 'tset', 'tgt', 'split', 'feat'))
    S = np.load(os.path.join(GRES, 'stage0', f'{tgt}_s{split}.npz'))
    fr, test, allT = S['foreign'], S['test'], S['allT']
    X = D.X[feat]
    idx = fr if tset == 'k0' else allT[G.pool_positions(S, split, 0)]
    ytr = D.y[idx]
    w = class_weights_from(ytr)[ytr]
    w = w / w.mean()
    sc = StandardScaler().fit(np.asarray(X[idx], dtype=np.float64))
    Xtr = sc.transform(np.asarray(X[idx], dtype=np.float64))
    XT = sc.transform(np.asarray(X[test], dtype=np.float64))
    yT = D.y[test]
    big = np.flatnonzero(np.bincount(yT, minlength=10) >= 30)
    bigtr = np.flatnonzero(np.bincount(ytr, minlength=10) >= 30)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(ytr), min(10000, len(ytr)), replace=False)
    res = dict(job=job, curve={})
    t0 = time.time()
    if model == 'lr':
        for C in (0.01, 1.0):
            m = LogisticRegression(C=C, max_iter=1000).fit(Xtr, ytr, sample_weight=w)
            res['curve'][f'C{C}'] = dict(test=metr(yT, m.predict(XT), big), train=metr(ytr[sub], m.predict(Xtr[sub]), bigtr),
                                         n_iter=int(np.max(m.n_iter_)))
    else:
        m = MLPClassifier((256, 128), alpha=1e-4, batch_size=256, learning_rate_init=1e-3, random_state=0)
        classes = np.arange(10)
        for ep in range(1, max(EPS) + 1):
            m.partial_fit(Xtr, ytr, classes=classes, sample_weight=w)
            if ep in EPS:
                res['curve'][ep] = dict(test=metr(yT, m.predict(XT), big), train=metr(ytr[sub], m.predict(Xtr[sub]), bigtr),
                                        loss=float(m.loss_))
    res['t'] = time.time() - t0
    return res


def worker(job):
    name = 'alt_' + '_'.join(str(job[x]) for x in ('model', 'tset', 'tgt', 'split', 'feat'))
    path = os.path.join(OUT, 'jobs', name + '.json')
    if os.path.exists(path):
        return name + ' skip'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = run(job)
    json.dump(r, open(path + '.tmp', 'w')); os.replace(path + '.tmp', path)
    return f'{name} {r["t"]:.0f}s'


def init():
    import guidance_G as G
    run.D = G.GData()


if __name__ == '__main__':
    import multiprocessing as mp
    C5 = ['AT', 'DE', 'LV', 'PT', 'SE']
    jobs = [dict(model=mo, tset=ts, tgt=c, split=1, feat=f) for mo in ('mlp', 'lr') for c in C5 for f in ('ae', 's2')
            for ts in ('k0', 'tgt')]
    with mp.Pool(2, initializer=init) as pool:
        for msg in pool.imap_unordered(worker, jobs):
            print(time.strftime('%H:%M:%S'), msg, flush=True)
