"""审计 5：G 各行/各概率文件是否都在同一个测试集上（行数一致、n_far 一致、big 类集合一致）。阶段 A 概率文件同查。只读。"""
import glob, os, re, zipfile, numpy as np, pandas as pd
R = '/Users/perng_mini/AlphaCrops_local/results'
rows = pd.concat([pd.read_csv(f) for f in glob.glob(f'{R}/guidance_G/rows_t*_*.csv')], ignore_index=True)
print('G rows', len(rows), 'conds', sorted(rows.cond.unique()))
print('n_far unique per (target,split):', rows.groupby(['target', 'split']).n_far.nunique().max())
# big 类集合：rec{c} 非空 → 类出现在测试集；ba_ge30 用 big。按 (target,split) 看 recc 非空模式是否恒定
rc = [f'rec{c}' for c in range(10)]
pat = rows[rc].notna().astype(int).astype(str).agg(''.join, axis=1)
print('present-class pattern unique per (target,split):', pat.groupby([rows.target, rows.split]).nunique().max())
# 概率文件行数
nt = {}
for t in rows.target.unique():
    for s in (0, 1):
        nt[(t, s)] = len(np.load(f'{R}/guidance_G/stage0/{t}_s{s}.npz')['test'])
bad = 0; n = 0
for p in glob.glob(f'{R}/guidance_G/probs/*.npz'):
    m = re.match(r'(\w\w)_s(\d)_r(\d)_', os.path.basename(p))
    t, s = m.group(1), int(m.group(2))
    with zipfile.ZipFile(p) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            ver = np.lib.format.read_magic(f)
            shape = (np.lib.format.read_array_header_1_0(f) if ver == (1, 0) else np.lib.format.read_array_header_2_0(f))[0]
    n += 1
    if shape[0] != nt[(t, s)]:
        bad += 1
print(f'G probs files {n}, wrong test length {bad}')
# 阶段 A
bad = 0; n = 0
ntA = {}
for p in glob.glob(f'{R}/stageA/probs/*.npy'):
    m = re.match(r'(\w\w)_s(\d)_r(\d)_', os.path.basename(p))
    t, s = m.group(1), int(m.group(2))
    if (t, s) not in ntA:
        ntA[(t, s)] = len(np.load(f'{R}/stageA/stage0/{t}_s{s}.npz')['test'])
    a = np.load(p, mmap_mode='r')
    n += 1
    bad += a.shape[0] != ntA[(t, s)]
print(f'stageA probs files {n}, wrong test length {bad}')
# G 中同一 (unit, clf, k, sel_hash) 被不同条件共用时指标是否一致
key = ['target', 'split', 'rep', 'clf', 'k', 'sel_hash']
g = rows[rows.sel_hash.notna() & (rows.sel_hash != '')].groupby(key).ba_ge30.nunique()
print('same selection across conds -> ba_ge30 unique values max:', g.max(), '; groups with >1 cond:',
      (rows[rows.sel_hash.notna()].groupby(key).cond.nunique() > 1).sum())
# 种子：选样文件名中无种子；检查同单元 srs 在 tier1 与 tier3(k=1000 未跑) 的一致性略
