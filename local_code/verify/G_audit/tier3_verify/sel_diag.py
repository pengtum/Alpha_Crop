"""选样层面诊断（不训练）：层级 3 的 RAWZ / NATIVE 选样与主分析选样差多少。
22 个重复 0 单元 × k=50：RAWZ（E1 四方法）、NATIVE（lpm2_nogeo、typiclust）。
 - Jaccard(层级3 选样, 主分析同特征选样)；主分析 AE 与 S2 选样也算一遍作参照
 - 离群比例：所选点落在'离池中心最远 1%'（在该特征的主分析稳健空间 G 中度量）"""
import os, sys
import numpy as np, pandas as pd
LE = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, LE); os.chdir(LE)
import guidance_G as G
D = G.GData()
ST0 = '/Users/perng_mini/AlphaCrops_local/results/guidance_G/stage0'
SEL = '/Users/perng_mini/AlphaCrops_local/results/guidance_G/sel'
E1 = ['lpm2_nogeo', 'kcenter', 'typiclust', 'kherding']
out = []
for tgt in G.COUNTRIES_G:
    for split in (0, 1):
        S = dict(np.load(os.path.join(ST0, f'{tgt}_s{split}.npz')))
        pp = G.pool_positions(S, split, 0)
        seed = 1000 * split
        far = {}
        for f in ('ae', 's2'):
            X = S[f'G_{f}'][pp]
            dist = np.linalg.norm(X - X.mean(0), axis=1)
            far[f] = dist >= np.quantile(dist, 0.99)
        for m in E1:
            main = {c: np.load(f'{SEL}/{tgt}_s{split}_r0_{c}_{m}_k50.npz')['sel_pos'] for c in ('AE', 'S2')}
            jac = lambda a, b: len(set(a) & set(b)) / len(set(a) | set(b))
            r = dict(tgt=tgt, split=split, method=m, J_main_AE_S2=jac(main['AE'], main['S2']),
                     far_main_AE=far['ae'][main['AE']].mean(), far_main_S2=far['s2'][main['S2']].mean())
            fams = [('RAWZ', 'RAWZ_AE', 'RAWZ_S2')] + ([('NATIVE', 'NATIVE_AE', 'NATIVE_S2')] if m in ('lpm2_nogeo', 'typiclust') else [])
            for fam, ca, cb in fams:
                sa, _ = G.select_one(D, S, pp, ca, m, 50, seed)
                sb, _ = G.select_one(D, S, pp, cb, m, 50, seed)
                r.update({f'J_{fam}AE_mainAE': jac(sa, main['AE']), f'J_{fam}S2_mainS2': jac(sb, main['S2']),
                          f'J_{fam}AE_{fam}S2': jac(sa, sb),
                          f'far_{fam}_AE': far['ae'][sa].mean(), f'far_{fam}_S2': far['s2'][sb].mean()})
            out.append(r)
            print(r, flush=True)
df = pd.DataFrame(out)
df.to_csv('/Users/perng_mini/AlphaCrops_local/verify/G_audit/tier3_verify/sel_diag.csv', index=False)
print(df.groupby('method').mean(numeric_only=True).drop(columns='split').round(3).T.to_string())
