"""只读：按 guidance_G.jobs_for 枚举层级 1 的 (单元, 条件, 方法, k)，读取 sel 文件，统计唯一选样（按 sel_hash），
估计新学习器需要的拟合次数（每种特征一次）。不读测试标签、不读任何精度列。"""
import os, sys, itertools, collections
PROJ = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np
import guidance_G as G
GOUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
recs = []
missing = []
for tgt in G.COUNTRIES_G:
    for split in G.SPLITS:
        for rep in G.REPS:
            for cond, m, k, clfs in G.jobs_for(1, split, rep):
                p = f'{GOUT}/sel/{tgt}_s{split}_r{rep}_{cond}_{m}_k{k}.npz'
                if not os.path.exists(p):
                    missing.append(p); continue
                sel = np.load(p)['sel']
                recs.append(dict(tgt=tgt, split=split, rep=rep, cond=cond, method=m, k=k, h=G.sel_hash(sel), n=len(sel)))
import pandas as pd
df = pd.DataFrame(recs)
print('tier1 sel files found', len(df), 'missing', len(missing))
df['unit'] = df.tgt + '_s' + df.split.astype(str) + '_r' + df.rep.astype(str)
fam = {m: 'E1' for m in G.E1} | {m: 'PRED' for m in G.PRED} | {m: 'HYB' for m in G.HYB} | {'srs': 'SRS'}
df['fam'] = df.method.map(fam)
def uniq(sub):
    return sub.drop_duplicates(['unit', 'k', 'h']).shape[0]
# 方案 A：E1 × {AE,S2} × k{50,200} × 44 单元 + SRS k{50,100,200,400} × 44 单元
A = df[(df.fam == 'E1') | (df.fam == 'SRS')]
Bset = df
for name, sub in (('E1 only', df[df.fam == 'E1']), ('SRS only', df[df.fam == 'SRS']), ('A: E1+SRS', A),
                  ('B: all tier1', Bset), ('PRED', df[df.fam == 'PRED']), ('HYB', df[df.fam == 'HYB'])):
    print(f'{name:14s} files {len(sub):5d}  unique(unit,k,hash) {uniq(sub):5d}')
# 同一单元同一 k，AE 与 S2 指导选到相同样本的比例
g = df[df.cond.isin(['AE', 'S2'])].pivot_table(index=['unit', 'method', 'k'], columns='cond', values='h', aggfunc='first')
same = (g.AE == g.S2)
print('AE vs S2 guidance identical selections:', int(same.sum()), 'of', len(g))
print(same.groupby(level='method').sum().to_string())
# 跨方法重合（同单元同 k 同 hash 不同方法/条件）
dup = df.groupby(['unit', 'k', 'h']).size()
print('hash groups with >1 file:', int((dup > 1).sum()), 'extra files saved:', int((dup - 1).sum()))
# k=0：每 (国家, 划分) 一个外国模型
print('k=0 units (country x split):', df[['tgt', 'split']].drop_duplicates().shape[0])
df.to_csv('/Users/perng_mini/AlphaCrops_local/verify/multilearner/feasibility_compute/tier1_sel_index.csv', index=False)
# 各 k 的拟合数
print(df.drop_duplicates(['unit', 'k', 'h']).groupby(['fam', 'k']).size().to_string())
