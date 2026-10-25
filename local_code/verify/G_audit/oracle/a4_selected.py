"""审计 4：各条件（ORACLE/AE/S2/srs）所选地块的性质（重复 0）：
- 类别构成（覆盖类数、与池先验 L1、稀有类占比）；
- "难度"：ORACLE 袋外模型 / 外国 AE / 外国 S2 模型在所选地块上的误判率、真类概率；
- sel 文件与 rows 的 comp_L1 / n_classes_sel / sel_hash 一致性。只读。"""
import glob, hashlib, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import guidance_G as G

OUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
D = G.GData()
R = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT, 'rows_t*_*.csv'))], ignore_index=True)
rows = []
mism = 0
for tgt in G.COUNTRIES_G:
    for split in (0, 1):
        S = dict(np.load(os.path.join(OUT, 'stage0', f'{tgt}_s{split}.npz')))
        O = dict(np.load(os.path.join(OUT, 'oracle', f'{tgt}_s{split}.npz')))
        pp = G.pool_positions(S, split, 0)
        idx = S['allT'][pp]
        y = D.y[idx]
        pi_pool = np.bincount(y, minlength=10) / len(y)
        rare = pi_pool < 0.05
        P = {'orc': O['P_T'], 'ae': S['P_T_ae'][pp], 's2': S['P_T_s2'][pp]}
        wrong = {k: v.argmax(1) != y for k, v in P.items()}
        ptrue = {k: v[np.arange(len(y)), y] for k, v in P.items()}
        base = dict(tgt=tgt, split=split)
        # 池基线
        rows.append(dict(base, cond='pool', method='pool', k=0, n_cls=int((pi_pool > 0).sum()), L1=0.0,
                         rare_frac=rare[y].mean(), **{f'err_{k}': w.mean() for k, w in wrong.items()},
                         **{f'ptrue_{k}': p.mean() for k, p in ptrue.items()}))
        for f in sorted(glob.glob(os.path.join(OUT, 'sel', f'{tgt}_s{split}_r0_*.npz'))):
            name = os.path.basename(f)[len(f'{tgt}_s{split}_r0_'):-4]
            cond, rest = name.split('_', 1)
            method, k = rest.rsplit('_k', 1)
            k = int(k)
            if cond not in ('ORACLE', 'AE', 'S2', 'none') or method not in ('lpm2_nogeo', 'typiclust', 'margin', 'srs'):
                continue
            z = np.load(f)
            sp = z['sel_pos']
            assert np.array_equal(idx[sp], z['sel']), f
            ys = y[sp]
            L1 = float(np.abs(np.bincount(ys, minlength=10) / k - pi_pool).sum())
            h = hashlib.md5(np.sort(z['sel'].astype(np.int64)).tobytes()).hexdigest()[:16]
            rr = R[(R.target == tgt) & (R.split == split) & (R.rep == 0) & (R.cond == cond) & (R.method == method) & (R.k == k)]
            if len(rr) == 0 or not (np.allclose(rr.comp_L1, L1) and (rr.sel_hash == h).all()
                                    and (rr.n_classes_sel == len(np.unique(ys))).all()):
                mism += 1
            rows.append(dict(base, cond=cond, method=method, k=k, n_cls=len(np.unique(ys)), L1=L1,
                             rare_frac=rare[ys].mean(), **{f'err_{kk}': w[sp].mean() for kk, w in wrong.items()},
                             **{f'ptrue_{kk}': p[sp].mean() for kk, p in ptrue.items()}))
print('sel 与 rows 不一致的条目：', mism)
df = pd.DataFrame(rows)
df.to_csv('/Users/perng_mini/AlphaCrops_local/verify/G_audit/oracle/a4_selected.csv', index=False)
pd.set_option('display.width', 250)
print(df.groupby(['method', 'cond', 'k'])[['n_cls', 'L1', 'rare_frac', 'err_orc', 'err_ae', 'err_s2', 'ptrue_orc',
                                           'ptrue_ae', 'ptrue_s2']].mean().round(3).to_string())
