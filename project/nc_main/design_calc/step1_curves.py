"""第 1 步：从三个任务的测试概率算出每个臂的 τ 曲线（OA、BA、宏 F1、κ），另存随机两半的曲线用于交叉拟合诊断。
只读已有概率；输出只写到本脚本所在的 scratch 目录。"""
import os
import sys
import pickle

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
UF = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/unified_figs'
sys.path.insert(0, UF)
import prep_unified as PU  # noqa: E402  只 import，不运行 main

AC, B, B2 = PU.AC, PU.B, PU.B2
TAUS = AC.TAUS


def curve_set(yT, big, Pa, halves):
    C = Pa.shape[1]
    Z = B.log_or_neginf(B.norm64(Pa))
    PH = B2.tau_preds_grid(Z, AC.pihat_of(Z), TAUS).astype(np.int64)
    out = {}
    for name, m in [('all', np.ones(len(yT), bool))] + list(halves.items()):
        y = yT[m].astype(np.int64)
        ph = PH[:, m]
        n = len(y)
        conf = np.stack([np.bincount(y * C + ph[t], minlength=C * C).reshape(C, C) for t in range(len(TAUS))])
        tp = conf[:, np.arange(C), np.arange(C)].astype(float)
        tr = conf.sum(2).astype(float)
        pr = conf.sum(1).astype(float)
        bg = np.asarray(big)
        oa = 100 * tp.sum(1) / n
        with np.errstate(invalid='ignore', divide='ignore'):
            ba = 100 * np.nanmean(tp[:, bg] / tr[:, bg], 1)
            f1 = 100 * np.nanmean(np.where(tr[:, bg] + pr[:, bg] > 0, 2 * tp[:, bg] / (tr[:, bg] + pr[:, bg]), 0.0), 1)
        po = tp.sum(1) / n
        pe = (tr * pr).sum(1) / n ** 2
        kap = 100 * (po - pe) / (1 - pe)
        out[name] = dict(oa=oa, ba=ba, f1=f1, kappa=kap)
    return out


def run(task, units):
    res = {}
    for U in units:
        yT, big, P = U['yT'], U['big'], U['P']
        rng = np.random.default_rng(abs(hash(U['unit'])) % (2 ** 31) if False else 12345 + len(res))
        h = rng.permutation(len(yT)) < len(yT) // 2
        halves = {'h1': h, 'h2': ~h}
        cv = {}
        for key, Pa in P.items():
            cv[key] = curve_set(yT, big, Pa, halves)
        # 与冻结实现核对 OA、BA
        ref = AC.unit_curves(yT, big, {k: P[k] for k in [('ae', 'Z0', 0), ('s2', 'RT200', 1)]})
        for k, (o, b) in ref.items():
            assert np.allclose(o, cv[k]['all']['oa'], atol=1e-9) and np.allclose(b, cv[k]['all']['ba'], atol=1e-9), k
        res[U['unit']] = dict(domain=U['domain'], primary=U['primary'], n_test=len(yT), big=list(big), cv=cv)
        print(task, U['unit'], len(yT), flush=True)
    return res


if __name__ == '__main__':
    allres = {}
    allres['crop'] = run('crop', PU.crop_units())
    allres['lcz'] = run('lcz', PU.pilot_units('lcz'))
    allres['lucas'] = run('lucas', PU.pilot_units('lucas'))
    pickle.dump(allres, open(os.path.join(HERE, 'curves.pkl'), 'wb'))
    print('done')
