"""
在阶段 A（PT，期中 4 个单元，S2 分类器，r=50，300 轮）的全部 37 个方法上叠加 F3b（只读；无训练）。
F3b：(1−α)·log P_重训 + α·log softmax(s·log P_外国 + b)，(s, b) 用该方法选出的 k 个标签在冻结外国模型
（阶段 A 的 P_T，已温度校准）上按类内 Hájek 归一加权拟合，λ=0.1。设计权重：有 π 的用 π，否则等权。
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import glob
import numpy as np
import pandas as pd
from scipy import stats

__file__ = os.path.abspath(__file__)
exec(open(os.path.join(os.path.dirname(__file__), 'calib_diag.py')).read().split('rows = []')[0])
RA = '/Users/perng_mini/AlphaCrops_local/results/stageA'


def lsm(L):
    L = L - L.max(1, keepdims=True)
    return L - np.log(np.exp(L).sum(1, keepdims=True))


out = []
for f in sorted(glob.glob(os.path.join(RA, 'rows_run_PT_*.csv'))):
    R_ = pd.read_csv(f)
    if len(R_) < 70:
        continue
    tgt, split, rep = 'PT', int(R_.split.iloc[0]), int(R_['rep'].iloc[0])
    S = np.load(os.path.join(RA, 'stage0', f'{tgt}_s{split}.npz'))
    allT, test = S['allT'], S['test']
    pos_of = pd.Series(np.arange(len(allT)), index=allT)
    yT = y_all[test]
    big = np.flatnonzero(np.bincount(yT, minlength=C) >= MIN_N)
    PT_ = S['P_T'].astype(np.float64)
    Z_all = np.log(np.clip(PT_, 1e-12, None))
    ZT = Z_all[len(S['cand']):]
    for _, r in R_.iterrows():
        tag = f"{tgt}_s{split}_r{rep}_{r['method']}_k{r['k']}_R{r['r']}"
        pp = os.path.join(RA, 'probs', tag + '.npy')
        sp = os.path.join(RA, 'sel', tag + '.npz')
        if not (os.path.exists(pp) and os.path.exists(sp)):
            continue
        z = np.load(sp)
        sel = z['sel']
        pis = z['pi'] if len(z['pi']) == len(sel) else np.ones(len(sel))
        ys = y_all[sel]
        Zs = Z_all[pos_of.loc[sel].to_numpy()]
        s_b, b_b = fit_bcts(Zs, ys, weights(ys, pis, 'bal'), LAM0)
        Pr = np.load(pp).astype(np.float64)
        Lr = np.log(np.clip(Pr / Pr.sum(1, keepdims=True), 1e-12, None))
        o0, b0, _ = scores(yT, Lr.argmax(1), big)
        row = dict(unit=f's{split}_r{rep}', method=r['method'], k=int(r['k']), oa=o0, ba=b0)
        for a in (0.4, 0.5):
            o, bb, _ = scores(yT, ((1 - a) * Lr + a * lsm(s_b * ZT + b_b)).argmax(1), big)
            row[f'oa_a{a}'], row[f'ba_a{a}'] = o, bb
        out.append(row)
    print(f, flush=True)

df = pd.DataFrame(out)
df.to_csv(os.path.join(OUT, 'stageA_f3b_rows.csv'), index=False)
L = []
for k in (50, 1000):
    d = df[df.k == k]
    srs = d[d.method == 'srs'].set_index('unit')
    t = []
    for m, g in d.groupby('method'):
        g = g.set_index('unit')
        t.append(dict(method=m, n=len(g),
                      d_ba_retrain=(g.ba - srs.ba).mean(), d_oa_retrain=(g.oa - srs.oa).mean(),
                      d_ba_F3b04=(g['ba_a0.4'] - srs.ba).mean(), d_oa_F3b04=(g['oa_a0.4'] - srs.oa).mean(),
                      gain_F3b04_same=(g['ba_a0.4'] - g.ba).mean(), oa_F3b04_same=(g['oa_a0.4'] - g.oa).mean(),
                      d_ba_F3b05=(g['ba_a0.5'] - srs.ba).mean(), d_oa_F3b05=(g['oa_a0.5'] - srs.oa).mean()))
    t = pd.DataFrame(t).sort_values('d_ba_retrain', ascending=False)
    L.append(f'\n== PT 阶段 A k={k}（4 单元；相对 SRS 重训；F3b 用该方法自己的标签）==')
    L.append(t.round(2).to_string(index=False))
txt = '\n'.join(L)
open(os.path.join(OUT, 'stageA_f3b_report.txt'), 'w').write(txt)
print(txt)
