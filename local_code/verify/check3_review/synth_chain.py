"""
训练时链式分解（不依赖概率校准）的合成验证：
  OA：L0（外国、平衡权重）→ PM（外国、权重 π_T/π_S）→ CMN（外国、按 T 的 pool 类别计数抽样、自然权重）→ Wnat（T 的 pool、自然权重）
      先验部分 = PM − L0；样本量/类别计数部分 = CMN − PM；条件部分 = Wnat − CMN
  BA：L0 → LS → CMB（外国、按 T 计数抽样、平衡权重）→ W0（T、平衡权重）
      条件部分 = W0 − CMB
δ=0（无条件偏移）时，条件部分应 ≈ 0。
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from synth_decomp import C, SRC_COUNTS, TGT, make_world, sample, fit, fp, po, oa, ba


def sample_counts(means, counts, r):
    y = np.repeat(np.arange(C), counts.astype(int)); r.shuffle(y)
    X = (means[y] + r.normal(0, 1, (len(y), means.shape[1]))).astype(np.float32)
    return X, y


def run(seed, tname, delta, n_src=30000, n_pool=10000, n_test=10000):
    r = np.random.default_rng(1000 + seed)
    means, dirs = make_world(seed)
    shift = None if delta == 0 else delta * dirs
    piS = SRC_COUNTS / SRC_COUNTS.sum(); piT = TGT[tname] / TGT[tname].sum()
    Xs, ys = sample(means, piS, n_src, r)
    Xp, yp = sample(means, piT, n_pool, r, shift)
    Xt, yt = sample(means, piT, n_test, r, shift)
    U = np.ones(C) / C
    cnt = np.bincount(yp, minlength=C); S = cnt > 0
    piT_pool = cnt / cnt.sum()
    res = {}
    m, _ = fit(Xs, ys, compute_sample_weight('balanced', ys), seed); pL = fp(m, Xt)
    res['L0'] = pL; res['LS'] = (pL * S) / (pL * S).sum(1, keepdims=True); res['PO'] = po(pL, piT_pool, U)
    piS_emp = np.bincount(ys, minlength=C) / len(ys)
    ratio = np.divide(piT_pool, piS_emp, out=np.zeros(C), where=piS_emp > 0)
    keep = ratio[ys] > 0
    m, _ = fit(Xs[keep], ys[keep], ratio[ys[keep]], seed); res['PM'] = fp(m, Xt)
    Xc, yc = sample_counts(means, cnt, r)                       # 外国（源）数据，类别计数与 T 的 pool 相同
    m, _ = fit(Xc, yc, None, seed); res['CMN'] = fp(m, Xt)
    m, _ = fit(Xc, yc, compute_sample_weight('balanced', yc), seed); res['CMB'] = fp(m, Xt)
    m, _ = fit(Xp, yp, None, seed); res['Wnat'] = fp(m, Xt)
    m, _ = fit(Xp, yp, compute_sample_weight('balanced', yp), seed); res['W0'] = fp(m, Xt)
    res['WPO'] = po(res['W0'], piT_pool, U)
    return {k: (oa(v, yt) * 100, ba(v, yt) * 100) for k, v in res.items()}


if __name__ == '__main__':
    rows = []
    for tname in ['SE', 'PT']:
        for delta in [0.0, 1.0]:
            for s in [0, 1]:
                for k, (a, b) in run(s, tname, delta).items():
                    rows.append(dict(T=tname, delta=delta, seed=s, arm=k, OA=a, BA=b))
                print(tname, delta, s, flush=True)
    R = pd.DataFrame(rows); R.to_csv('/Users/perng_mini/AlphaCrops_local/verify/check3_review/synth_chain.csv', index=False)
    M = R.pivot_table(index=['T', 'delta', 'seed'], columns='arm', values='OA')
    B = R.pivot_table(index=['T', 'delta', 'seed'], columns='arm', values='BA')
    pd.set_option('display.width', 250)
    print(M.groupby(['T', 'delta']).mean().round(2).to_string())
    X = pd.DataFrame(dict(
        OA_total_Wnat=M['Wnat'] - M['L0'], OA_prior_PM=M['PM'] - M['L0'], OA_n_counts=M['CMN'] - M['PM'],
        OA_cond=M['Wnat'] - M['CMN'],
        OA_plan_total=M['WPO'] - M['L0'], OA_plan_prior=M['PO'] - M['L0'], OA_plan_cond=M['WPO'] - M['PO'],
        BA_total=B['W0'] - B['L0'], BA_n_counts=B['CMB'] - B['LS'], BA_cond=B['W0'] - B['CMB'],
        BA_plan_cond=B['W0'] - B['LS']))
    print(X.groupby(['T', 'delta']).mean().round(2).to_string())
