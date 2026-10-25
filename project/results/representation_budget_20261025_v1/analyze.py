"""只读现有概率；在所有标签档位共同的 OA 支持域上分析表征的标签等价量。"""
import os
import sys
from pathlib import Path
import hashlib
import json
import time
sys.dont_write_bytecode = True
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[v] = '2'
OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[1]
sys.path.insert(0, str(PROJECT / 'label_expansion/confirm'))
import analyze_confirm as C
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression

ROOT = Path(C.FORMAL_ROOT)
KS = np.array([0, 50, 100, 200, 400])
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    start = time.time()
    cache = OUT / 'curves.csv'
    old = pd.read_csv(ROOT / 'analysis/unit_rows.csv')
    if not cache.exists():
        y = C.G.GData().y
        rows, hashes = [], {}
        for unit in C.expected_units():
            path = ROOT / 'splits' / f'{unit}.npz'
            hashes[str(path)] = sha(path)
            z = np.load(path)
            yt, big = y[z['test']], z['big']
            assert np.array_equal(big, np.flatnonzero(np.bincount(yt, minlength=10) >= 30))
            for f in C.FEATS:
                for k in KS:
                    arm = 'Z0' if k == 0 else f'RT{k}'
                    for r in ([0] if k == 0 else range(3)):
                        path = ROOT / 'probs' / unit / f'lgb_{arm}_{f}_r{r}.npz'
                        hashes[str(path)] = sha(path)
                        P = np.load(path)['P_test']
                        oa, ba = C.unit_curves(yt, big, {('lgb', f, arm, r): P})[('lgb', f, arm, r)]
                        ref = old[(old.unit == unit) & (old.level == 'arm') & (old.learner == 'lgb') & (old.feat == f) & (old.name == arm) & (old.r == r)].iloc[0]
                        assert abs(oa[C.I_T0] - ref.oa_t0) < 1e-9
                        assert abs(ba[C.I_T0] - ref.ba_t0) < 1e-9
                        for t, a, b in zip(C.TAUS, oa, ba):
                            rows.append(dict(unit=unit, country=unit[:2], feat=f, k=int(k), r=r, tau=t, OA=a, BA=b))
            print(f'{unit} completed, {time.time()-start:.1f}s', flush=True)
        pd.DataFrame(rows).to_csv(cache, index=False)
        for path in (ROOT / 'analysis/unit_rows.csv', Path(C.G.GData.cache_dir()) / 'y.npy', Path(C.G.GData.cache_dir()) / 'meta.json', Path(C.__file__), OUT / 'ANALYSIS_PLAN.txt'):
            hashes[str(path)] = sha(path)
        (OUT / 'input_sha256.json').write_text(json.dumps(hashes, indent=2))
    d = pd.read_csv(cache)
    common, coverage, grids, paircheck = [], [], [], []
    for unit, df in d.groupby('unit'):
        curves = {(f, int(k), int(r)): (g.sort_values('tau').OA.to_numpy(), g.sort_values('tau').BA.to_numpy()) for (f,k,r), g in df.groupby(['feat','k','r'])}
        anchor = min(curves[(f,0,0)][0][C.I_T0] for f in C.FEATS)
        grid = anchor + C.OFFS
        env = {key: C.B.envelope([cv], grid)[0] for key, cv in curves.items()}
        mask = np.isfinite(np.stack(list(env.values()))).all(0)
        cov = mask.mean()
        coverage.append(dict(unit=unit, country=unit[:2], anchor=anchor, common_points=int(mask.sum()), total_points=len(grid), coverage=cov, valid=bool(cov >= C.MIN_COV)))
        # 与冻结分析的原始零标签表征差异对账，随后使用更严格的所有曲线共同支持域。
        v, cv = C.paired(env[('ae',0,0)], env[('s2',0,0)])
        ref = old[(old.unit==unit)&(old.level=='D')&(old.learner=='lgb')&(old.name=='D_Z0')].iloc[0]
        assert (np.isnan(v) and np.isnan(ref.value)) or abs(v-ref.value)<1e-9
        paircheck.append(dict(unit=unit, recomputed=v, frozen=ref.value))
        for (f,k,r), e in env.items():
            oa, ba = curves[(f,k,r)]
            value = e[mask].mean() if cov>=C.MIN_COV else np.nan
            common.append(dict(unit=unit,country=unit[:2],feat=f,k=k,r=r,BA_common=value,OA_default=oa[C.I_T0],BA_default=ba[C.I_T0]))
            for i,o in enumerate(grid):
                grids.append(dict(unit=unit,feat=f,k=k,r=r,OA=o,BA=e[i],in_common=bool(mask[i])))
    u = pd.DataFrame(common)
    u.to_csv(OUT/'unit_values.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
    pd.DataFrame(grids).to_csv(OUT/'envelopes.csv',index=False)
    pd.DataFrame(paircheck).to_csv(OUT/'frozen_reconciliation.csv',index=False)
    units = u.groupby(['unit','country','feat','k'],as_index=False)[['BA_common','BA_default','OA_default']].mean()
    countries = units.groupby(['country','feat','k'],as_index=False)[['BA_common','BA_default','OA_default']].mean()
    countries.to_csv(OUT/'country_curves.csv',index=False)
    estimates, curves_out, inversions, country_diffs = [], [], [], []
    for metric in ['BA_common','BA_default','OA_default']:
        p = countries.pivot(index='country',columns=['feat','k'],values=metric).dropna()
        for c,row in p.iterrows():
            country_diffs.append(dict(metric=metric,country=c,ae0=row[('ae',0)],s2400=row[('s2',400)],diff=row[('ae',0)]-row[('s2',400)]))
        for setname, keep in C.SETS.items():
            q = p.loc[p.index.isin(keep)]
            for k in KS:
                x = q[('ae',0)]-q[('s2',k)]
                n = len(x)
                ci = stats.t.interval(.95,n-1,loc=x.mean(),scale=stats.sem(x))
                estimates.append(dict(metric=metric,country_set=setname,k=int(k),n=n,mean=x.mean(),lo95=ci[0],hi95=ci[1],n_pos=int((x>0).sum())))
            for f in C.FEATS:
                for k in KS:
                    curves_out.append(dict(metric=metric,country_set=setname,feat=f,k=int(k),mean=q[(f,k)].mean()))
            if metric != 'BA_common':
                continue
            arr = np.stack([q.loc[:,[(f,k) for k in KS]].to_numpy() for f in C.FEATS],axis=1)
            # 先国家均值后单调拟合；同时保存未经单调化的曲线。
            def inv(ys, target):
                iso = IsotonicRegression().fit_transform(KS,ys)
                if target <= iso[0]:
                    return 0.0
                if target > iso[-1]:
                    return np.inf
                j = np.flatnonzero(iso>=target)[0]
                return float(KS[j-1]+(KS[j]-KS[j-1])*(target-iso[j-1])/(iso[j]-iso[j-1]))
            def get(a):
                av = a.mean(0)
                return inv(av[1],av[0,0]),inv(av[0],av[1,-1])
            point = get(arr)
            rng = np.random.default_rng(42)
            boot = np.array([get(arr[rng.integers(len(arr),size=len(arr))]) for _ in range(2000)])
            np.savetxt(OUT/f'bootstrap_{setname}.csv',boot,delimiter=',',header='s2_count_matching_ae0,ae_count_matching_s2400',comments='')
            for i,name in enumerate(['s2_count_matching_ae0','ae_count_matching_s2400']):
                so = np.sort(boot[:,i])
                inversions.append(dict(country_set=setname,name=name,point=point[i],lo90=so[99],hi90=so[1899],fraction_above400=float(np.isinf(so).mean()),fraction_zero=float((so==0).mean())))
    pd.DataFrame(estimates).to_csv(OUT/'contrasts.csv',index=False)
    pd.DataFrame(curves_out).to_csv(OUT/'mean_curves.csv',index=False)
    pd.DataFrame(inversions).to_csv(OUT/'label_equivalents.csv',index=False)
    pd.DataFrame(country_diffs).to_csv(OUT/'country_contrasts.csv',index=False)
    print(pd.DataFrame(coverage).to_string(index=False))
    print(pd.DataFrame(estimates).query('k == 400').to_string(index=False))
    print(pd.DataFrame(inversions).to_string(index=False))
    print(pd.DataFrame(country_diffs).query('metric == "BA_common"').to_string(index=False))
    print(f'Completed in {time.time()-start:.1f}s', flush=True)

if __name__ == '__main__':
    main()
