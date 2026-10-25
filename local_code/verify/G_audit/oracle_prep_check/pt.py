import numpy as np, sys, os
sys.path.insert(0, '.')
import guidance_G as G
D = G.GData()
for tgt in ['PT','LV','SE','AT','RO']:
  for split in [0]:
    S = dict(np.load(os.path.join(G.OUT,'stage0',f'{tgt}_s{split}.npz')))
    o = np.load(G.oracle_path(tgt, split))
    pool_pos = G.pool_positions(S, split, 0)
    idx = S['allT'][pool_pos]
    y = D.y[idx]
    Gm = o['G'].astype(np.float64)
    assert len(y)==len(Gm)
    cnt = np.bincount(y, minlength=10)
    sd = Gm.std(0)
    # variance share of each column
    vs = Gm.var(0)/Gm.var(0).sum()
    print(tgt, 'counts', cnt.tolist())
    print('   sd', np.round(sd,2).tolist())
    print('   var share', np.round(vs,3).tolist())
    # check constant-fill: rows equal to min in a column
    for c in range(10):
        m = Gm[:,c].min()
        nmin = (Gm[:,c]==m).sum()
        if nmin>5: print('   col',c,'n at min',nmin)
