"""审计 5：
(a) 用存档 oracle 重新选样，核对 sel_hash 与 rows 一致（选样可复现）；
(b) 几何修正的敏感性（只改选样，不训练）：
    V1 = 方案第 2 节字面要求：对 C 列做稳健 z 截断（±5，不做 PCA）；
    V2 = 只保留候选池中 ≥30 个地块的类的列，再做 V1（去掉缺席类 sc.min() 填充与极稀有类的折伪影）；
    输出与原选样的 Jaccard、覆盖类数、L1。只读，写到 verify 目录。"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import guidance_G as G

OUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
AUD = '/Users/perng_mini/AlphaCrops_local/verify/G_audit/oracle'
D = G.GData()
R = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT, 'rows_t2_*.csv'))], ignore_index=True)


def zclip(A):
    A = np.asarray(A, np.float64)
    return np.clip((A - A.mean(0)) / (A.std(0) + 1e-9), -G.CLIP, G.CLIP)


def variants(O, y):
    cnt = np.bincount(y, minlength=10)
    keep = np.flatnonzero(cnt >= 30)
    return {'orig': O['G'], 'V1': zclip(O['G']), 'V2': zclip(O['G'][:, keep])}


rows = []
os.makedirs(os.path.join(AUD, 'sel_fix'), exist_ok=True)
for tgt in G.COUNTRIES_G:
    for split in (0, 1):
        S = dict(np.load(os.path.join(OUT, 'stage0', f'{tgt}_s{split}.npz')))
        O = dict(np.load(os.path.join(OUT, 'oracle', f'{tgt}_s{split}.npz')))
        pp = G.pool_positions(S, split, 0)
        idx = S['allT'][pp]
        y = D.y[idx]
        pi_pool = np.bincount(y, minlength=10) / len(y)
        seed = 1000 * split + 0
        for vname, Gv in variants(O, y).items():
            orc = dict(O, G=np.asarray(Gv, np.float32))
            for m in ('lpm2_nogeo', 'typiclust') + (('margin',) if vname == 'orig' else ()):
                for k in (50, 200):
                    sp, _ = G.select_one(D, S, pp, 'ORACLE', m, k, seed, orc)
                    z = np.load(os.path.join(OUT, 'sel', f'{tgt}_s{split}_r0_ORACLE_{m}_k{k}.npz'))
                    a, b = set(sp.tolist()), set(z['sel_pos'].tolist())
                    ys = y[sp]
                    r = dict(tgt=tgt, split=split, var=vname, method=m, k=k, jacc_vs_stored=len(a & b) / len(a | b),
                             n_cls=len(np.unique(ys)), L1=float(np.abs(np.bincount(ys, minlength=10) / k - pi_pool).sum()))
                    if vname == 'orig':
                        h = G.sel_hash(idx[sp])
                        rr = R[(R.target == tgt) & (R.split == split) & (R.cond == 'ORACLE') & (R.method == m) & (R.k == k)]
                        r['hash_match'] = bool((rr.sel_hash == h).all() and len(rr) == 2)
                    else:
                        np.savez_compressed(os.path.join(AUD, 'sel_fix', f'{tgt}_s{split}_{vname}_{m}_k{k}.npz'), sel_pos=sp)
                    rows.append(r)
        print(tgt, split, flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(AUD, 'a5_reselect.csv'), index=False)
print('orig 可复现（hash 全一致）：', df[df['var'] == 'orig'].hash_match.all(), df[df['var'] == 'orig'].hash_match.sum())
print(df.groupby(['var', 'method', 'k'])[['jacc_vs_stored', 'n_cls', 'L1']].mean().round(3).to_string())
print(df[(df['var'] == 'V2')].pivot_table(index=['tgt', 'split'], columns=['method', 'k'], values='jacc_vs_stored').round(2).to_string())
