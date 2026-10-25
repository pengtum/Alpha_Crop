"""按方案第 2 节对 ORACLE 几何补做 z 截断（候选池行，±5，不 PCA），重跑 lpm2_nogeo / typiclust 选样，
与原选样比较；并做校准：原 G 重跑（应完全复现）、原 G 加 1e-9 相对扰动、同单元 AE vs S2 的 Jaccard。"""
import numpy as np, os, sys, json
sys.path.insert(0, '.')
import guidance_G as G
OUTD = '/Users/perng_mini/AlphaCrops_local/verify/G_audit/oracle_prep_check'
D = G.GData()
def jac(a, b):
    a, b = set(a.tolist()), set(b.tolist()); return len(a & b) / len(a | b)
def zclip(Gm):
    Gm = np.asarray(Gm, np.float64)
    return np.clip((Gm - Gm.mean(0)) / (Gm.std(0) + 1e-9), -G.CLIP, G.CLIP).astype(np.float32)
res = []
os.makedirs(OUTD + '/sel_fixed', exist_ok=True)
for tgt in D.countries:
    for split in (0, 1):
        S = dict(np.load(os.path.join(G.OUT, 'stage0', f'{tgt}_s{split}.npz')))
        o = dict(np.load(G.oracle_path(tgt, split)))
        pool_pos = G.pool_positions(S, split, 0)
        seed = 1000 * split
        o_fix = dict(o); o_fix['G'] = zclip(o['G'])
        rng = np.random.default_rng(1)
        o_eps = dict(o); o_eps['G'] = (o['G'].astype(np.float64) * (1 + 1e-9 * rng.standard_normal(o['G'].shape))).astype(np.float32)
        for method in ('lpm2_nogeo', 'typiclust'):
            for k in (50, 200):
                unit = f'{tgt}_s{split}_r0'
                z = np.load(os.path.join(G.OUT, 'sel', f'{unit}_ORACLE_{method}_k{k}.npz'))
                orig = z['sel_pos']
                rep_, _ = G.select_one(D, S, pool_pos, 'ORACLE', method, k, seed, o)
                fix, _ = G.select_one(D, S, pool_pos, 'ORACLE', method, k, seed, o_fix)
                eps, _ = G.select_one(D, S, pool_pos, 'ORACLE', method, k, seed, o_eps)
                zae = np.load(os.path.join(G.OUT, 'sel', f'{unit}_AE_{method}_k{k}.npz'))['sel_pos']
                zs2 = np.load(os.path.join(G.OUT, 'sel', f'{unit}_S2_{method}_k{k}.npz'))['sel_pos']
                np.savez_compressed(f'{OUTD}/sel_fixed/{unit}_ORACLEfix_{method}_k{k}.npz', sel_pos=fix)
                r = dict(tgt=tgt, split=split, method=method, k=k, reproduced=bool(np.array_equal(np.sort(rep_), np.sort(orig))),
                         J_fix=jac(fix, orig), J_eps=jac(eps, orig), J_ae_s2=jac(zae, zs2),
                         J_fix_s2=jac(fix, zs2), J_orig_s2=jac(orig, zs2))
                res.append(r); print(r, flush=True)
import pandas as pd
df = pd.DataFrame(res); df.to_csv(OUTD + '/resel.csv', index=False)
print(df.groupby(['method', 'k'])[['reproduced', 'J_fix', 'J_eps', 'J_ae_s2', 'J_fix_s2', 'J_orig_s2']].mean().round(3))
