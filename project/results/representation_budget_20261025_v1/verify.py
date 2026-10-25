"""数值复核：逐段插值独立重算共同OA包络，核对冻结输入和论文数字。"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import t

P = Path(__file__).resolve().parent
D = pd.read_csv(P/'curves.csv')
U = pd.read_csv(P/'unit_values.csv')
max_error = 0.0
count = 0
for unit, d in D.groupby('unit'):
    anchor = min(d[(d.feat==f)&(d.k==0)&(d.tau==0)].OA.item() for f in ('ae','s2'))
    grid = anchor + np.linspace(-4,0,41)
    envelopes = {}
    for key,g in d.groupby(['feat','k','r']):
        g = g.sort_values('tau')
        ox,by = g.OA.to_numpy(),g.BA.to_numpy()
        vals = []
        for o in grid:
            candidates = []
            for i in range(len(ox)-1):
                a,b = ox[i:i+2]
                if min(a,b)-1e-9<=o<=max(a,b)+1e-9:
                    if abs(a-b)<1e-9:
                        candidates.append(max(by[i:i+2]))
                    else:
                        w = min(1,max(0,(o-a)/(b-a)))
                        candidates.append(by[i]*(1-w)+by[i+1]*w)
            vals.append(max(candidates) if candidates else np.nan)
        envelopes[key] = np.array(vals)
    mask = np.all(np.isfinite(np.stack(list(envelopes.values()))),axis=0)
    assert mask.sum()>=21
    for (f,k,r),v in envelopes.items():
        a = U[(U.unit==unit)&(U.feat==f)&(U.k==k)&(U.r==r)].BA_common.item()
        max_error = max(max_error,abs(a-v[mask].mean()))
        count += 1
assert count == 572 and max_error < 1e-9
C = pd.read_csv(P/'country_curves.csv')
E = pd.read_csv(P/'contrasts.csv')
for row in E.itertuples():
    d = C[C.country.isin(['AT','BE','DE','EE','LT','LV','SE','SI','SK'])] if row.country_set==9 else C
    p = d.pivot(index='country',columns=['feat','k'],values=row.metric)
    x = p[('ae',0)]-p[('s2',row.k)]
    n = len(x); m = x.sum()/n
    sem = np.sqrt(((x-m)**2).sum()/(n-1)/n)
    lo,hi = m-t.ppf(.975,n-1)*sem,m+t.ppf(.975,n-1)*sem
    assert np.allclose([m,lo,hi],[row.mean,row.lo95,row.hi95],atol=1e-10)
    assert (x>0).sum() == row.n_pos
hashes=json.loads((P/'input_sha256.json').read_text())
for f,v in hashes.items():
    assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==v,f
result={'independent_envelope_comparisons':count,'max_abs_error':max_error,'all_contrasts_and_t_intervals_verified':True,'unchanged_input_files':len(hashes),'coverage_units':22,'countries':11}
(P/'verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
