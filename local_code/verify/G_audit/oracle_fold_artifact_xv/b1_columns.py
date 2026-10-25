"""独立核实：ORACLE 几何 G 的缺席类列是否等于该折 sc.min()（按折变化的常数），以及折效应占总方差的份额。
另外模拟方案第 2 节的"C 列 z 截断 ±5"后的折效应份额（不删列），看方案忠实实现是否也有问题。只读。"""
import os, sys
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
sys.path.insert(0, os.getcwd())
import guidance_G as G

OUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
W = '/Users/perng_mini/AlphaCrops_local/verify/G_audit/oracle_fold_artifact_xv'
D = G.GData()


def eta2(M, fold):
    """矩阵 M 的总方差中由折均值解释的份额（组间平方和 / 总平方和，对列求和）。"""
    M = M - M.mean(0)
    tot = (M ** 2).sum()
    bet = sum(((fold == f).sum() * (M[fold == f].mean(0) ** 2)).sum() for f in range(5))
    return bet / tot


def zclip(M, clip=5.0):
    return np.clip((M - M.mean(0)) / (M.std(0) + 1e-9), -clip, clip)


cols, units = [], []
for tgt in G.COUNTRIES_G:
    for split in (0, 1):
        S = dict(np.load(os.path.join(OUT, 'stage0', f'{tgt}_s{split}.npz')))
        O = dict(np.load(os.path.join(OUT, 'oracle', f'{tgt}_s{split}.npz')))
        pp = G.pool_positions(S, split, 0)
        y = D.y[S['allT'][pp]]
        Gm = O['G'].astype(np.float64)
        assert len(Gm) == len(y)
        fold = np.zeros(len(y), int)
        # 与 oracle_compute 相同：KFold(n_folds, shuffle=True, random_state=0).split(Xr)，只依赖行数
        for f, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(np.zeros((len(y), 1)))):
            fold[te] = f
        cnt = np.bincount(y, minlength=10)
        # 每折训练集中缺席的类
        absent_tr = {f: set(np.flatnonzero(np.bincount(y[fold != f], minlength=10) == 0)) for f in range(5)}
        # 核对缺席列 == 该折所有 LDA 分数最小值（等价于该折存在列的最小值与填充值本身的最小值）
        ok_fill = True
        for f in range(5):
            A = sorted(absent_tr[f])
            if not A:
                continue
            Gf = Gm[fold == f]
            pres = [c for c in range(10) if c not in absent_tr[f]]
            fill = Gf[:, pres].min()
            ok_fill &= bool(np.allclose(Gf[:, A], fill, atol=1e-4))
        tot = Gm.var(0).sum()
        art = sorted(set.union(*absent_tr.values()))
        rare = [c for c in range(10) if 0 < cnt[c] < 30]
        for c in range(10):
            col = Gm[:, c]
            within_const = int(sum(Gm[fold == f, c].std() < 1e-9 for f in range(5)))
            cols.append(dict(tgt=tgt, split=split, cls=c, n_pool=int(cnt[c]), var_share=col.var() / tot,
                             n_const_folds=within_const, fold_eta2=eta2(col[:, None], fold)))
        Z = zclip(Gm)
        keep = [c for c in range(10) if cnt[c] >= 30]
        units.append(dict(
            tgt=tgt, split=split, n_pool=len(y), absent_in_pool=[int(c) for c in np.flatnonzero(cnt == 0)],
            rare_1_29=[(int(c), int(cnt[c])) for c in rare], fill_check=ok_fill,
            art_var_share_raw=sum(Gm[:, c].var() for c in art) / tot if art else 0.0,
            eta2_raw=eta2(Gm, fold),
            eta2_zclip=eta2(Z, fold),
            eta2_drop_lt30_z=eta2(zclip(Gm[:, keep]), fold),
            fold_means_art=None if not art else np.round([Gm[fold == f][:, art[0]].mean() for f in range(5)], 2).tolist(),
        ))
pd.DataFrame(cols).to_csv(f'{W}/b1_columns.csv', index=False)
U = pd.DataFrame(units)
U.to_csv(f'{W}/b1_units.csv', index=False)
pd.set_option('display.width', 250)
print(U.round(3).to_string(index=False))
