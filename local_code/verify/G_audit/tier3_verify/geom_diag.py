"""几何诊断（不训练）：主分析稳健几何 G 与层级 3 的 RAWZ(RZ)、NATIVE(NT) 在 AE/S2 上的差别。
每个 (国家, 划分) 取重复 0 的候选池：
 - 截断比例：外国∪目标上 |z|>5 的元素比例（AE、S2 原始 128/168 列）
 - 重尾度：直径 / 成对距离中位数（池内随机 2000 点）
 - 10 近邻平衡精度（池标签，5 折），G / RZ / NT
 - AE：G_ae、NT_ae 的 PC1 与 stdDev 组 z 均值的相关"""
import os, sys, json
import numpy as np, pandas as pd
LE = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, LE); os.chdir(LE)
import guidance_G as G
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from scipy.spatial.distance import pdist
D = G.GData()
ST0 = '/Users/perng_mini/AlphaCrops_local/results/guidance_G/stage0'
out = []
for tgt in G.COUNTRIES_G:
    for split in (0, 1):
        S = np.load(os.path.join(ST0, f'{tgt}_s{split}.npz'))
        pp = G.pool_positions(S, split, 0)
        allT, fr = S['allT'], S['foreign']
        idx = allT[pp]
        y = D.y[idx]
        r = dict(tgt=tgt, split=split)
        rng = np.random.default_rng(0)
        sub = rng.choice(len(pp), 2000, replace=False)
        for f in ('ae', 's2'):
            Z = np.vstack([D.X[f][fr], D.X[f][allT]]).astype(np.float64)
            Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
            r[f'clip_{f}'] = float((np.abs(Zs) > 5).mean())
            r[f'rowclip_{f}'] = float((np.abs(Zs) > 5).any(1).mean())
            if f == 'ae':
                std_cols = [j for j, c in enumerate(D.cols[f]) if c.endswith('_stdDev')]
                sm = Zs[len(fr):][pp][:, std_cols].mean(1)
            for g in ('G', 'RZ', 'NT'):
                X = S[f'{g}_{f}'][pp].astype(np.float64)
                dd = pdist(X[sub])
                r[f'tail_{g}_{f}'] = float(dd.max() / np.median(dd))
                kn = KNeighborsClassifier(10, n_jobs=1)
                ph = cross_val_predict(kn, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
                r[f'knnBA_{g}_{f}'] = 100 * balanced_accuracy_score(y, ph)
                if f == 'ae':
                    r[f'corrPC1_std_{g}'] = float(abs(np.corrcoef(X[:, 0], sm)[0, 1]))
        out.append(r)
        print(r, flush=True)
df = pd.DataFrame(out)
df.to_csv('/Users/perng_mini/AlphaCrops_local/verify/G_audit/tier3_verify/geom_diag.csv', index=False)
print(df.drop(columns=['tgt', 'split']).describe().T[['mean', 'min', 'max']].round(3).to_string())
