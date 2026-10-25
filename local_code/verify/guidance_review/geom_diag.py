"""几何诊断（只读阶段 0 与原始特征；不读 rows_run/probs/sel）。开发国 PT/BE/SI，第 0 种划分。"""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
sys.path.insert(0, '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion')
from common import load_dataset, feature_cols
df = load_dataset()
classes = sorted(df['label'].unique())
y = df['label'].map({c: i for i, c in enumerate(classes)}).to_numpy()
X = {f: df[feature_cols(df, f)].to_numpy(np.float32) for f in ('ae', 's2')}
aecols = feature_cols(df, 'ae'); ae_mean_idx = [i for i, c in enumerate(aecols) if c.endswith('_mean')]
lon, lat = df.longitude.to_numpy(), df.latitude.to_numpy()
ctry = df.country.to_numpy()
del df
S0 = '/Users/perng_mini/AlphaCrops_local/results/stageA/stage0'
rng = np.random.default_rng(0)
out = []

def pr(v):  # participation ratio
    return v.sum() ** 2 / (v ** 2).sum()

def knn_purity(Z, yy, K=10):
    nn = NearestNeighbors(n_neighbors=K + 1, n_jobs=2).fit(Z)
    _, I = nn.kneighbors(Z)
    I = I[:, 1:]
    agree = (yy[I] == yy[:, None]).mean(1)
    # 类平衡版本：每类平均
    cls = np.unique(yy)
    return agree.mean(), np.mean([agree[yy == c].mean() for c in cls if (yy == c).sum() >= 30]), I

def jacc(I1, I2):
    return np.mean([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(I1, I2)])

for tgt in ['PT', 'BE', 'SI']:
    S = dict(np.load(f'{S0}/{tgt}_s0.npz'))
    fr, allT, cand = S['foreign'], S['allT'], S['cand']
    ncand = len(cand)
    sub = np.sort(rng.choice(ncand, min(15000, ncand), replace=False))
    idx = allT[sub]
    yy = y[idx]
    lat0 = float(S['lat0'])
    xy = np.c_[lon[idx] * 111.32 * np.cos(np.radians(lat0)), lat[idx] * 110.57]
    res = {'tgt': tgt, 'n_sub': len(sub)}
    Is = {}
    for f in ('ae', 's2'):
        Z = np.vstack([X[f][fr], X[f][allT]]).astype(np.float64)
        mu, sd = Z.mean(0), Z.std(0) + 1e-9
        Zs = (Z - mu) / sd
        p = PCA(64, random_state=0).fit(Zs)
        evr = p.explained_variance_ratio_
        res[f'{f}_evr8'] = evr[:8].sum(); res[f'{f}_evr16'] = evr[:16].sum(); res[f'{f}_evr32'] = evr[:32].sum()
        T = S[f'T_{f}'][sub].astype(np.float64)   # 阶段 0 保存的 32 维 PCA（目标池）
        vT = T.var(0)
        # 目标池内 PC 方差 / 合并方差（衡量该 PC 是否主要编码"国家间"差异）
        res[f'{f}_tgtvar_ratio_pc1_4'] = float(np.mean(vT[:4] / p.explained_variance_[:4]))
        res[f'{f}_tgtvar_ratio_pc5_32'] = float(np.mean(vT[4:32] / p.explained_variance_[4:32]))
        for d in (8, 16, 32):
            A = T[:, :d] / np.sqrt(T[:, :d].var(0).sum() / d)   # 与 X_of 相同
            res[f'{f}_PR{d}'] = pr(A.var(0))
            a, b, I = knn_purity(A, yy)
            res[f'{f}_pur{d}'] = a; res[f'{f}_purbal{d}'] = b
            Is[(f, d)] = I
        # 原始空间（z 标准化，全部维）
        R = (X[f][idx] - mu) / sd
        a, b, I = knn_purity(R, yy); res[f'{f}_purRAW'] = a; res[f'{f}_purbalRAW'] = b; Is[(f, 'raw')] = I
        # 目标池内 PCA 32（不含外国）
        Zt = (X[f][allT] - X[f][allT].mean(0)) / (X[f][allT].std(0) + 1e-9)
        pt = PCA(32, random_state=0).fit(Zt)
        At = pt.transform((X[f][idx] - X[f][allT].mean(0)) / (X[f][allT].std(0) + 1e-9))
        a, b, I = knn_purity(At, yy); res[f'{f}_purTgtPCA32'] = a; res[f'{f}_purbalTgtPCA32'] = b
        # 线性 R^2：坐标 ~ 32 维 PC
        res[f'{f}_R2xy'] = LinearRegression().fit(T, xy).score(T, xy)
        # kNN 回归 R^2（非线性）：用 10-NN 平均坐标预测
        I32 = Is[(f, 32)]
        pred = xy[I32].mean(1)
        res[f'{f}_knnR2xy'] = 1 - ((xy - pred) ** 2).sum() / ((xy - xy.mean(0)) ** 2).sum()
    # AE 均值部分，余弦（原生度量）
    M = X['ae'][idx][:, ae_mean_idx].astype(np.float64)
    M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    a, b, I = knn_purity(M, yy); res['ae_purMeanCos'] = a; res['ae_purbalMeanCos'] = b; Is[('ae', 'meancos')] = I
    # XY
    a, b, I = knn_purity(xy, yy); res['xy_pur'] = a; res['xy_purbal'] = b; Is[('xy', 0)] = I
    res['jac_ae32_s2_32'] = jacc(Is[('ae', 32)], Is[('s2', 32)])
    res['jac_ae32_xy'] = jacc(Is[('ae', 32)], Is[('xy', 0)])
    res['jac_s2_32_xy'] = jacc(Is[('s2', 32)], Is[('xy', 0)])
    res['jac_ae32_aeraw'] = jacc(Is[('ae', 32)], Is[('ae', 'raw')])
    res['jac_s2_32_s2raw'] = jacc(Is[('s2', 32)], Is[('s2', 'raw')])
    res['jac_ae32_aemeancos'] = jacc(Is[('ae', 32)], Is[('ae', 'meancos')])
    # 随机基线：同类比例
    cnt = np.bincount(yy, minlength=10) / len(yy)
    res['chance_pur'] = (cnt ** 2).sum()
    out.append(res)
    print(tgt, 'done', flush=True)
R = pd.DataFrame(out).set_index('tgt').T
pd.set_option('display.width', 200)
print(R.round(3).to_string())
R.to_csv('/Users/perng_mini/AlphaCrops_local/verify/guidance_review/geom_diag.csv')
