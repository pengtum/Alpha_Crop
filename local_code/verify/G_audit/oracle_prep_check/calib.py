import numpy as np, os, sys
sys.path.insert(0, '.')
import guidance_G as G
D = G.GData()
def jac(a, b):
    a, b = set(a.tolist()), set(b.tolist()); return len(a & b) / len(a | b)
out=[]
for tgt, split in [('AT',0),('PT',0),('SE',1),('RO',0)]:
    S = dict(np.load(os.path.join(G.OUT, 'stage0', f'{tgt}_s{split}.npz')))
    o = dict(np.load(G.oracle_path(tgt, split)))
    pool_pos = G.pool_positions(S, split, 0); seed = 1000*split
    Gm = o['G'].astype(np.float64)
    for name, Gp in [('rel1e-4', Gm*(1+1e-4*np.random.default_rng(1).standard_normal(Gm.shape))),
                     ('col1%', Gm*np.exp(0.01*np.random.default_rng(2).standard_normal(10))),
                     ('col10%', Gm*np.exp(0.10*np.random.default_rng(3).standard_normal(10)))]:
        op = dict(o); op['G'] = Gp.astype(np.float32)
        for method in ('lpm2_nogeo','typiclust'):
            for k in (50,200):
                orig = np.load(os.path.join(G.OUT,'sel',f'{tgt}_s{split}_r0_ORACLE_{method}_k{k}.npz'))['sel_pos']
                s,_ = G.select_one(D,S,pool_pos,'ORACLE',method,k,seed,op)
                out.append((tgt,split,name,method,k,round(jac(s,orig),3)))
                print(out[-1], flush=True)
