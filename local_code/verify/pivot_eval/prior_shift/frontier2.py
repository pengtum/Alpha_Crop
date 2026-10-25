"""
F3 的对照（只读；无训练）：F3 的增益来自"bal 决策层"，还是只是"两个模型集成"？
  F3b  (1−α)·log P_retrain + α·log softmax(s_bal·Z + b_bal)      ← 方案
  F3n  同上，但第二个专家用 nat 头（同一组标签、自然先验加权）
  F3z  同上，但第二个专家是未用任何目标标签的外国模型（温度校准后的 k0）
  F3u  同上，第二个专家为 k0 除以其在候选池上的平均预测（无标签"均衡化"：log P0 − log mean_pool P0）
另报告：predstrat_AE（重训）+ F3b 相对 SRS 重训的合计。所有对比都是 11 国等权，国家层 t 区间。
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
import pandas as pd
from scipy import stats

__file__ = os.path.abspath(__file__)
exec(open(os.path.join(os.path.dirname(__file__), 'calib_diag.py')).read().split('rows = []')[0])

ALPHAS = [0.3, 0.4, 0.5, 0.6]


def lsm(L):
    L = L - L.max(1, keepdims=True)
    return L - np.log(np.exp(L).sum(1, keepdims=True))


rows = []
for tgt in COUNTRIES:
    for split in (0, 1):
        S = np.load(os.path.join(R, 'stage0', f'{tgt}_s{split}.npz'))
        n_cand = len(S['cand'])
        allT, test, fr = S['allT'], S['test'], S['foreign']
        yT = y_all[test]
        big = np.flatnonzero(np.bincount(yT, minlength=C) >= MIN_N)
        for rep in (0, 1):
            unit = f'{tgt}_s{split}_r{rep}'
            pool_pos = pool_positions(n_cand, split, rep)
            yP = y_all[allT[pool_pos]]
            N = len(pool_pos)
            designs = [('srs_G', k, f'{unit}_none_srs_k{k}.npz') for k in (50, 100, 200, 400)]
            if rep == 0:
                designs += [('predstrat_AE', k, f'{unit}_AE_predstrat_srs_k{k}.npz') for k in (50, 200)]
                designs += [('predstrat_S2', k, f'{unit}_S2_predstrat_srs_k{k}.npz') for k in (50, 200)]
            for clf in ('s2', 'ae'):
                PP = S[f'P_T_{clf}'][pool_pos].astype(np.float64)
                PT = S[f'P_T_{clf}'][n_cand:].astype(np.float64)
                ZP, ZT = np.log(np.clip(PP, 1e-12, None)), np.log(np.clip(PT, 1e-12, None))
                Lz = lsm(ZT)
                Lu = lsm(ZT - np.log(PP.mean(0)))
                base = dict(target=tgt, split=split, rep=rep, unit=unit, clf=clf)
                for des, k, fn in designs:
                    z = np.load(os.path.join(R, 'sel', fn))
                    sel_pos, sel = z['sel_pos'], z['sel']
                    pis = z['pi_sel'] if len(z['pi_sel']) else np.full(k, k / N)
                    ys = yP[sel_pos]
                    s_n, b_n = fit_bcts(ZP[sel_pos], ys, weights(ys, pis, 'nat'), LAM0)
                    s_b, b_b = fit_bcts(ZP[sel_pos], ys, weights(ys, pis, 'bal'), LAM0)
                    Pr = np.load(os.path.join(R, 'probs', f'{unit}_{clf}_k{k}_{sel_hash(sel)}.npz'))['i300']
                    Pr = Pr.astype(np.float64)
                    Lr = np.log(np.clip(Pr / Pr.sum(1, keepdims=True), 1e-12, None))
                    o, bb, _ = scores(yT, Lr.argmax(1), big)
                    rows.append(dict(base, design=des, k=k, fam='retrain', alpha=0.0, oa=o, ba=bb))
                    experts = {'F3b': lsm(s_b * ZT + b_b), 'F3n': lsm(s_n * ZT + b_n), 'F3z': Lz, 'F3u': Lu}
                    for fam, Le in experts.items():
                        for a in ALPHAS:
                            o, bb, _ = scores(yT, ((1 - a) * Lr + a * Le).argmax(1), big)
                            rows.append(dict(base, design=des, k=k, fam=fam, alpha=a, oa=o, ba=bb))
        print(tgt, split, flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'frontier2_rows.csv'), index=False)


def summ(j, label):
    c = j.groupby('target')[['dba', 'doa']].mean()
    out = [label]
    for col in ('dba', 'doa'):
        v = c[col]
        h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        pos = int((v > 0).sum()) if col == 'dba' else int((v >= -0.3).sum())
        out.append(f'{col} {v.mean():+.2f} [{v.mean() - h:+.2f},{v.mean() + h:+.2f}] {pos}/11')
    cc = c.drop(['RO', 'PT'], errors='ignore')
    out.append(f'| 去RO/PT dba {cc.dba.mean():+.2f} doa {cc.doa.mean():+.2f}')
    return '  '.join(out)


key = ['unit', 'target', 'clf', 'k']
lines = []
ref_srs = df[(df.design == 'srs_G') & (df.fam == 'retrain')].set_index(key)[['ba', 'oa']]
for des in ('srs_G', 'predstrat_AE', 'predstrat_S2'):
    for k in sorted(df[df.design == des].k.unique()):
        lines.append(f'\n== {des} k={k} ==')
        ref_same = df[(df.design == des) & (df.fam == 'retrain') & (df.k == k)].set_index(key)[['ba', 'oa']]
        if des != 'srs_G':
            x = ref_same
            j = (x - ref_srs).dropna().rename(columns={'ba': 'dba', 'oa': 'doa'}).reset_index()
            lines.append(summ(j, f'  重训 {des} − 重训 SRS            '))
        for fam in ('F3b', 'F3n', 'F3z', 'F3u'):
            for a in ALPHAS:
                x = df[(df.design == des) & (df.fam == fam) & (df.alpha == a) & (df.k == k)].set_index(key)[['ba', 'oa']]
                j = (x - ref_same).dropna().rename(columns={'ba': 'dba', 'oa': 'doa'}).reset_index()
                lines.append(summ(j, f'  {fam} α={a} − 重训同标签          '))
                if des != 'srs_G' and fam == 'F3b':
                    j2 = (x - ref_srs).dropna().rename(columns={'ba': 'dba', 'oa': 'doa'}).reset_index()
                    lines.append(summ(j2, f'  {fam} α={a}（{des}）− 重训 SRS   '))
txt = '\n'.join(lines)
open(os.path.join(OUT, 'frontier2_report.txt'), 'w').write(txt)
print(txt)
