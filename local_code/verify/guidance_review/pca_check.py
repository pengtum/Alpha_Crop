"""
问题 3：AE 与 S2 标准化后 PCA 的解释方差、截断后几何的有效维数、以及几项与尺度/原点相关的量。
(1) 全局 2 万地块随机样本：标准化 → PCA，前 8/16/32 个主成分累计解释方差（另报 AE 的 mean 组 / stdDev 组各自的结果）。
(2) 按 stageA.py 阶段 0 的做法逐国仿真（外国 10 国各 5,000 + 目标国全部地块，标准化后 PCA 32，随机数与 stageA 不同，
    只为估计量级）：并集上的累计解释方差；目标池在前 d 维的方差保留比例；X_of 缩放后前 d 维的有效维数（参与比）；
    目标池均值偏离原点的程度（BADGE 相关）；直径 / 中位成对距离（GIST 的 d_max 归一化与 k-center 相关）；
    10-NN 平衡精度（d=16、d=32、全维标准化特征；目标国内 5 折）衡量截断保留的类别信息。
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

CODE = "/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion"
sys.path.insert(0, CODE)
from samplers.uncertainty import _diam_blocked  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
COUNTRIES = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']


def load():
    import pyarrow.parquet as pq
    names = pq.ParquetFile('/Users/perng_mini/AlphaCrops_local/data/dataset_v2.parquet').schema_arrow.names
    ae = [c for c in names if c.startswith('ae_')]
    s2 = [c for c in names if c.startswith('s2_')]
    df = pd.read_parquet('/Users/perng_mini/AlphaCrops_local/data/dataset_v2.parquet',
                         columns=['country', 'label', 'has_ae', 'has_s2'] + ae + s2)
    df = df[df.has_ae & df.has_s2].reset_index(drop=True)
    X = {'ae': df[ae].to_numpy(np.float32), 's2': df[s2].to_numpy(np.float32)}
    y = pd.factorize(df['label'])[0]
    return df['country'].to_numpy(), y, X, ae, s2


def evr(Z, n=32):
    mu, sd = Z.mean(0), Z.std(0) + 1e-9
    p = PCA(n, random_state=0).fit((Z - mu) / sd)
    c = np.cumsum(p.explained_variance_ratio_)
    return c, p, mu, sd


def part_scale(A, d):
    A = A[:, :d].astype(np.float64)
    return A / np.sqrt(A.var(0).sum() / d)


def pr(v):
    return float(v.sum() ** 2 / (v ** 2).sum())


def knn_bacc(X, y, seed=0):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import balanced_accuracy_score
    ok = np.isin(y, np.flatnonzero(np.bincount(y) >= 10))
    X, y = X[ok], y[ok]
    pred = cross_val_predict(KNeighborsClassifier(10, n_jobs=2), X, y,
                             cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    return 100 * balanced_accuracy_score(y, pred)


def main():
    t0 = time.time()
    ctry, y, X, ae_cols, s2_cols = load()
    print(f'loaded {len(y):,} rows {time.time() - t0:.0f}s', flush=True)
    rng = np.random.default_rng(0)
    # ---------------- (1) 全局 2 万
    idx = rng.choice(len(y), 20_000, replace=False)
    rows = []
    for f in ('ae', 's2'):
        Z = X[f][idx].astype(np.float64)
        c, p, mu, sd = evr(Z, 32)
        full = PCA(random_state=0).fit((Z - mu) / sd).explained_variance_ratio_
        rows.append(dict(sample='global20k', feat=f, dim=Z.shape[1], cum8=c[7], cum16=c[15], cum32=c[31],
                         pc1=c[0], n90=int(np.searchsorted(np.cumsum(full), 0.90) + 1),
                         n95=int(np.searchsorted(np.cumsum(full), 0.95) + 1)))
        if f == 'ae':
            m = np.array(['_mean' in s for s in ae_cols])
            raw_sd = Z.std(0)
            print(f'AE 原始标准差：mean 组中位 {np.median(raw_sd[m]):.4f}，stdDev 组中位 {np.median(raw_sd[~m]):.4f}')
            load_sq = p.components_ ** 2
            for d in (8, 16, 32):
                sh = (load_sq[:d] * p.explained_variance_[:d, None]).sum(0)
                print(f'  AE 前 {d} 个 PC 中来自 stdDev 组的方差份额 {sh[~m].sum() / sh.sum():.3f}')
            for g, mm in (('ae_mean', m), ('ae_std', ~m)):
                cg, *_ = evr(Z[:, mm], 32)
                rows.append(dict(sample='global20k', feat=g, dim=int(mm.sum()), cum8=cg[7], cum16=cg[15],
                                 cum32=cg[31], pc1=cg[0]))
        else:
            ms = np.array(['_mean' in s.rsplit('_', 1)[-1] or s.endswith('_mean') for s in s2_cols])
            raw_sd = Z.std(0)
            print(f'S2 原始标准差范围 {raw_sd.min():.4g} – {raw_sd.max():.4g}（中位 {np.median(raw_sd):.4g}）')
    g = pd.DataFrame(rows)
    print(g.round(3).to_string(index=False), flush=True)
    g.to_csv(os.path.join(OUT, 'pca_global20k.csv'), index=False)

    # ---------------- (2) 阶段 0 仿真
    res = []
    for tgt in COUNTRIES:
        r = np.random.default_rng(1000 + COUNTRIES.index(tgt))
        foreign = np.concatenate([r.choice(np.flatnonzero(ctry == c), 5000, replace=False)
                                  for c in COUNTRIES if c != tgt])
        allT = np.flatnonzero(ctry == tgt)
        pool = r.choice(allT, min(30_000, len(allT)), replace=False)   # 模拟 3 万候选池
        pool_pos = np.searchsorted(allT, np.sort(pool))
        for f in ('ae', 's2'):
            Z = np.vstack([X[f][foreign], X[f][allT]]).astype(np.float64)
            mu, sd = Z.mean(0), Z.std(0) + 1e-9
            p = PCA(32, random_state=0).fit((Z - mu) / sd)
            c = np.cumsum(p.explained_variance_ratio_)
            Zt = (X[f][allT].astype(np.float64) - mu) / sd
            T = p.transform(Zt)[pool_pos]
            tot_T = Zt[pool_pos].var(0).sum()
            row = dict(target=tgt, feat=f, cum8=c[7], cum16=c[15], cum32=c[31])
            for d in (16, 32):
                v = T[:, :d].var(0)
                row[f'poolkeep{d}'] = v.sum() / tot_T             # 目标池总方差中被前 d 维保留的比例
                row[f'PR{d}'] = pr(v)                              # X_of 缩放后前 d 维的有效维数
                row[f'top1share{d}'] = v[0] / v.sum()
            S = part_scale(T, 32)
            row['offset_ratio32'] = float((S.mean(0) ** 2).sum() / S.var(0).sum())   # ‖均值‖² / 总方差（BADGE 的 h 未居中）
            sub = r.choice(len(S), 5000, replace=False)
            i, j = r.integers(0, len(S), 20000), r.integers(0, len(S), 20000)
            med = np.median(np.linalg.norm(S[i] - S[j], axis=1))
            dmax, _, _ = _diam_blocked(S, np.einsum('ij,ij->i', S, S), 1024)
            row['dmax_over_med'] = dmax / med
            nn_idx = r.choice(len(T), min(20_000, len(T)), replace=False)
            yt = y[allT][pool_pos][nn_idx]
            row['knn16'] = knn_bacc(part_scale(T[nn_idx], 16), yt)
            row['knn32'] = knn_bacc(part_scale(T[nn_idx], 32), yt)
            row['knnfull'] = knn_bacc(Zt[pool_pos][nn_idx], yt)
            res.append(row)
            print(tgt, f, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
                           if k not in ('target', 'feat')}, f'{time.time() - t0:.0f}s', flush=True)
    R = pd.DataFrame(res)
    R.to_csv(os.path.join(OUT, 'pca_stage0_sim.csv'), index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    print(R.round(3).to_string(index=False))
    W = R.pivot(index='target', columns='feat')
    print('\n== AE − S2（逐国）均值与范围 ==')
    for col in ['cum8', 'cum16', 'cum32', 'poolkeep16', 'poolkeep32', 'PR16', 'PR32', 'top1share16', 'offset_ratio32',
                'dmax_over_med', 'knn16', 'knn32', 'knnfull']:
        a, s = W[(col, 'ae')], W[(col, 's2')]
        dd = a - s
        print(f'{col:15s} AE {a.mean():7.3f} [{a.min():.3f},{a.max():.3f}]  S2 {s.mean():7.3f} [{s.min():.3f},{s.max():.3f}]'
              f'  Δ {dd.mean():+.3f} [{dd.min():+.3f},{dd.max():+.3f}]')


if __name__ == '__main__':
    main()
