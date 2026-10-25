"""独立构造修正几何并重选（只选样，不训练）：
  V2 = 只保留池中 ≥30 个地块的类的列，再 z 截断 ±5（与原审查者的 V2 定义相同，但本脚本独立实现）
  F1 = 方案忠实的最小修复：去掉"某折训练集缺该类"而被 sc.min() 填充的列，其余列全保留，再 z 截断 ±5
  PF = 方案字面：不删列，只 z 截断 ±5
输出：折 η²、typiclust/lpm2 选样的折分布卡方 p，以及与原审查者 sel_fix/V2 的逐位一致性。只读。"""
import os, sys, time
import numpy as np, pandas as pd
from scipy.stats import chisquare
from sklearn.model_selection import KFold
sys.path.insert(0, os.getcwd())
import guidance_G as G

OUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
W = '/Users/perng_mini/AlphaCrops_local/verify/G_audit/oracle_fold_artifact_xv'
OTHER = '/Users/perng_mini/AlphaCrops_local/verify/G_audit/oracle/sel_fix'
os.makedirs(f'{W}/sel', exist_ok=True)
D = G.GData()


def zc(M):
    M = np.asarray(M, np.float64)
    return np.clip((M - M.mean(0)) / (M.std(0) + 1e-9), -G.CLIP, G.CLIP)


def eta2(M, fold):
    M = M - M.mean(0)
    return sum(((fold == f).sum() * (M[fold == f].mean(0) ** 2)).sum() for f in range(5)) / (M ** 2).sum()


rows = []
for tgt in G.COUNTRIES_G:
    for split in (0, 1):
        S = dict(np.load(os.path.join(OUT, 'stage0', f'{tgt}_s{split}.npz')))
        O = dict(np.load(os.path.join(OUT, 'oracle', f'{tgt}_s{split}.npz')))
        pp = G.pool_positions(S, split, 0)
        y = D.y[S['allT'][pp]]
        fold = np.zeros(len(y), int)
        for f, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(np.zeros((len(y), 1)))):
            fold[te] = f
        cnt = np.bincount(y, minlength=10)
        filled = {c for f in range(5) for c in np.flatnonzero(np.bincount(y[fold != f], minlength=10) == 0)}
        Gm = O['G'].astype(np.float64)
        V = {'V2': zc(Gm[:, cnt >= 30]),
             'F1': zc(Gm[:, [c for c in range(10) if c not in filled]]),
             'PF': zc(Gm)}
        seed = 1000 * split
        for vn, Gv in V.items():
            orc = dict(O, G=Gv.astype(np.float32))
            for m in ('lpm2_nogeo', 'typiclust'):
                for k in (50, 200):
                    if vn == 'PF' and m == 'lpm2_nogeo':
                        continue
                    t0 = time.time()
                    sp, _ = G.select_one(D, S, pp, 'ORACLE', m, k, seed, orc)
                    np.savez_compressed(f'{W}/sel/{tgt}_s{split}_{vn}_{m}_k{k}.npz', sel_pos=sp)
                    c = np.bincount(fold[sp], minlength=5)
                    same = np.nan
                    op = f'{OTHER}/{tgt}_s{split}_V2_{m}_k{k}.npz'
                    if vn == 'V2' and os.path.exists(op):
                        same = bool(np.array_equal(np.sort(np.load(op)['sel_pos']), np.sort(sp)))
                    rows.append(dict(tgt=tgt, split=split, var=vn, method=m, k=k, eta2=eta2(Gv, fold),
                                     counts=c.tolist(), p_chi=chisquare(c).pvalue, same_as_other_V2=same,
                                     n_cls=len(np.unique(y[sp])), t=round(time.time() - t0, 1)))
        print(tgt, split, 'filled cols', sorted(int(c) for c in filled), flush=True)
R = pd.DataFrame(rows)
R.to_csv(f'{W}/b4_select.csv', index=False)
pd.set_option('display.width', 250)
print(R.groupby('var').eta2.describe().round(3))
print(R[R.method == 'typiclust'].assign(sig=lambda d: d.p_chi < 1e-3).groupby(['var', 'k']).sig.sum())
print('与原审查者 V2 逐位一致：', R[R['var'] == 'V2'].same_as_other_V2.value_counts(dropna=False).to_dict())
print(R[R.tgt.isin(['LT', 'LV', 'PT', 'RO']) & (R.method == 'typiclust') & (R.k == 200)].round(4).to_string(index=False))
