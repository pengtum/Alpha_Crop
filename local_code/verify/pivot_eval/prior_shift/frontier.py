"""
前沿诊断（只读；无训练）：同一组 k 个目标标签，不同"决策层"在 (OA, BA) 平面上的位置。
  F1 冻结外国模型：偏置在 nat（OA 向）与 bal（BA 向）之间插值 b(α) = (1−α)·b_nat + α·b_bal（温度同样插值）
  F2 重训模型（G 缓存）+ α·(b_bal − b_nat)（偏置方向取自冻结模型的样本外预测；启发式代理）
  F3 两模型对数概率混合：(1−α)·log P_retrain + α·log softmax(s_bal·Z_frozen + b_bal)
  F4 重训模型沿先验方向：log P_retrain − α·log π_train（α>1 为过度去偏斜）
标签：G 的 SRS 选样（4 个单元）与 predstrat_srs（AE 指导，重复 0）；k ∈ {50, 200}。λ=0.1 预先固定。
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
import pandas as pd

__file__ = os.path.abspath(__file__)
src = open(os.path.join(os.path.dirname(__file__), 'calib_diag.py')).read().split('rows = []')[0]
exec(src)

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
ALPHA_PRIOR = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]


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
        cw = class_weights_from(y_all[fr])
        for rep in (0, 1):
            unit = f'{tgt}_s{split}_r{rep}'
            pool_pos = pool_positions(n_cand, split, rep)
            idx_pool = allT[pool_pos]
            yP = y_all[idx_pool]
            N = len(pool_pos)
            designs = [('srs_G', k, f'{unit}_none_srs_k{k}.npz') for k in (50, 200)]
            if rep == 0:
                designs += [('predstrat_AE', k, f'{unit}_AE_predstrat_srs_k{k}.npz') for k in (50, 200)]
            for clf in ('s2', 'ae'):
                PP = S[f'P_T_{clf}'][pool_pos].astype(np.float64)
                PT = S[f'P_T_{clf}'][n_cand:].astype(np.float64)
                ZP, ZT = np.log(np.clip(PP, 1e-12, None)), np.log(np.clip(PT, 1e-12, None))
                base = dict(target=tgt, split=split, rep=rep, unit=unit, clf=clf)
                for des, k, fn in designs:
                    z = np.load(os.path.join(R, 'sel', fn))
                    sel_pos, sel = z['sel_pos'], z['sel']
                    pis = z['pi_sel'] if len(z['pi_sel']) else np.full(k, k / N)
                    ys = yP[sel_pos]
                    s_n, b_n = fit_bcts(ZP[sel_pos], ys, weights(ys, pis, 'nat'), LAM0)
                    s_b, b_b = fit_bcts(ZP[sel_pos], ys, weights(ys, pis, 'bal'), LAM0)
                    pp = os.path.join(R, 'probs', f'{unit}_{clf}_k{k}_{sel_hash(sel)}.npz')
                    Pr = np.load(pp)['i300'].astype(np.float64)
                    Pr /= Pr.sum(1, keepdims=True)
                    Lr = np.log(np.clip(Pr, 1e-12, None))
                    w = np.r_[cw[y_all[fr]], R_MULT * cw[ys]]
                    pit = np.bincount(np.r_[y_all[fr], ys], weights=w, minlength=C) / w.sum()
                    o_ref, b_ref, _ = scores(yT, Pr.argmax(1), big)
                    Lbal = lsm(s_b * ZT + b_b)
                    for a in ALPHAS:
                        L1 = ((1 - a) * s_n + a * s_b) * ZT + (1 - a) * b_n + a * b_b
                        o, bb, _ = scores(yT, L1.argmax(1), big)
                        rows.append(dict(base, design=des, k=k, fam='F1_frozen_interp', alpha=a, oa=o, ba=bb,
                                         oa_ref=o_ref, ba_ref=b_ref))
                        L2 = Lr + a * (b_b - b_n)
                        o, bb, _ = scores(yT, L2.argmax(1), big)
                        rows.append(dict(base, design=des, k=k, fam='F2_retrain_biasdir', alpha=a, oa=o, ba=bb,
                                         oa_ref=o_ref, ba_ref=b_ref))
                        L3 = (1 - a) * Lr + a * Lbal
                        o, bb, _ = scores(yT, L3.argmax(1), big)
                        rows.append(dict(base, design=des, k=k, fam='F3_mix_retrain_bal', alpha=a, oa=o, ba=bb,
                                         oa_ref=o_ref, ba_ref=b_ref))
                    for a in ALPHA_PRIOR:
                        L4 = Lr - a * np.log(np.clip(pit, 1e-12, None))
                        o, bb, _ = scores(yT, L4.argmax(1), big)
                        rows.append(dict(base, design=des, k=k, fam='F4_retrain_prior', alpha=a, oa=o, ba=bb,
                                         oa_ref=o_ref, ba_ref=b_ref))
        print(tgt, split, flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'frontier_rows.csv'), index=False)

df['dba'] = df.ba - df.ba_ref
df['doa'] = df.oa - df.oa_ref
lines = []
for des in ('srs_G', 'predstrat_AE'):
    for k in (50, 200):
        d = df[(df.design == des) & (df.k == k)]
        lines.append(f'\n== {des} k={k}：相对同一批标签的 G 重训（300 轮, r=50）；国家内平均后 11 国等权 ==')
        c = d.groupby(['fam', 'alpha', 'target'])[['dba', 'doa']].mean()
        m = c.groupby(['fam', 'alpha']).mean()
        pos = c.groupby(['fam', 'alpha']).dba.apply(lambda v: int((v > 0).sum()))
        oapos = c.groupby(['fam', 'alpha']).doa.apply(lambda v: int((v >= -0.3).sum()))
        noropt = d[~d.target.isin(['RO', 'PT'])].groupby(['fam', 'alpha', 'target'])[['dba', 'doa']].mean() \
            .groupby(['fam', 'alpha']).mean().add_suffix('_noROPT')
        t = m.join(noropt)
        t['ba_pos'] = pos
        t['oa>=-0.3'] = oapos
        lines.append(t.round(2).to_string())
txt = '\n'.join(lines)
open(os.path.join(OUT, 'frontier_report.txt'), 'w').write(txt)
print(txt)
