import sys, os, glob
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
import guidance_G as G
D = G.GData()
OUT = G.OUT
rows = []
for tgt in G.COUNTRIES_G:
    for split in (0, 1):
        S = dict(np.load(os.path.join(OUT, 'stage0', f'{tgt}_s{split}.npz')))
        pool_pos = G.pool_positions(S, split, 0)
        idx_pool = S['allT'][pool_pos]
        y = D.y[idx_pool]
        o = np.load(os.path.join(OUT, 'oracle', f'{tgt}_s{split}.npz'))
        PO = o['P_T']
        assert len(PO) == len(y)
        Ps2 = S['P_T_s2'][pool_pos]; Pae = S['P_T_ae'][pool_pos]
        def stats_(P, pos):
            yy = y[pos]; Pp = P[pos]
            top2 = np.argsort(-Pp, 1)[:, :2]
            return dict(err=(Pp.argmax(1) != yy).mean(), top2=(top2 == yy[:, None]).any(1).mean())
        base = {f'pool_{n}': stats_(P, np.arange(len(y))) for n, P in (('O', PO), ('S2', Ps2), ('AE', Pae))}
        pi_pool = np.bincount(y, minlength=10) / len(y)
        for cond in ('ORACLE', 'S2', 'AE'):
            for m in ('margin', 'typiclust', 'lpm2_nogeo'):
                for k in (50, 200):
                    f = os.path.join(OUT, 'sel', f'{tgt}_s{split}_r0_{cond}_{m}_k{k}.npz')
                    z = np.load(f)
                    sp = z['sel_pos']
                    assert np.array_equal(idx_pool[sp], z['sel'])
                    sO = stats_(PO, sp)
                    comp = np.bincount(y[sp], minlength=10) / k
                    rows.append(dict(tgt=tgt, split=split, cond=cond, method=m, k=k,
                                     errO_sel=sO['err'], top2O_sel=sO['top2'],
                                     errO_pool=base['pool_O']['err'], top2O_pool=base['pool_O']['top2'],
                                     errS2_pool=base['pool_S2']['err'], errAE_pool=base['pool_AE']['err'],
                                     comp_L1=np.abs(comp - pi_pool).sum(), n_cls=len(np.unique(y[sp])),
                                     n_cls_pool=int((pi_pool > 0).sum()),
                                     n_cls_pool_ge=int((np.bincount(y, minlength=10) >= 30).sum())))
T = pd.DataFrame(rows)
T.to_csv(os.path.join(sys.argv[1], 'mech.csv'), index=False)
pd.set_option('display.width', 200)
print(T.groupby(['cond', 'method']).agg(errO_sel=('errO_sel', 'mean'), top2O_sel=('top2O_sel', 'mean'),
                                        comp_L1=('comp_L1', 'mean'), n_cls=('n_cls', 'mean')).round(3))
print(T.groupby(['cond', 'method', 'k']).agg(errO_sel=('errO_sel', 'mean'), top2O_sel=('top2O_sel', 'mean'),
                                             comp_L1=('comp_L1', 'mean'), n_cls=('n_cls', 'mean')).round(3))
u = T.drop_duplicates(['tgt', 'split'])
print('pool errO', u.errO_pool.mean().round(4), 'top2O', u.top2O_pool.mean().round(4),
      'errS2', u.errS2_pool.mean().round(4), 'errAE', u.errAE_pool.mean().round(4), 'n_cls_pool', u.n_cls_pool.mean())
