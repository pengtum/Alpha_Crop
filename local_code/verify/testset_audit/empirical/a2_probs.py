"""审计 2：G 与阶段 A 的全部 probs 文件——行数 = 测试集大小；按行重算指标与 rows CSV 比对；单元内 n_far、big 类集合、rec 缺失模式恒定。只读。"""
import os, sys, glob, re, zipfile
LE = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, LE)
import numpy as np
import pandas as pd
import multiprocessing as mp

R = '/Users/perng_mini/AlphaCrops_local/results'
OUTD = '/Users/perng_mini/AlphaCrops_local/verify/testset_audit/empirical'
SC = np.load(f'{OUTD}/splits_cache.npz')
Y = np.load('/Users/perng_mini/AlphaCrops_local/data/cache_guidance_G/y.npy')
MIN_N = 30


def npz_shapes(path):
    """只读 npy 头，不解压全部数据。"""
    out = {}
    with zipfile.ZipFile(path) as zf:
        for n in zf.namelist():
            with zf.open(n) as f:
                v = np.lib.format.read_magic(f)
                shape, _, dt = (np.lib.format.read_array_header_1_0(f) if v == (1, 0) else np.lib.format.read_array_header_2_0(f))
                out[n[:-4]] = (shape, str(dt))
    return out


def scores(yT, ph, big, mask=None):
    from sklearn.metrics import f1_score
    if mask is not None:
        yT, ph = yT[mask], ph[mask]
    rec = np.array([(ph[yT == c] == c).mean() if (yT == c).any() else np.nan for c in range(10)])
    return (100 * float((ph == yT).mean()), 100 * float(np.nanmean(rec[big])),
            100 * float(f1_score(yT, ph, labels=big, average='macro', zero_division=0)), rec)


def g_unit(args):
    tgt, s = args
    test = SC[f'{tgt}_s{s}_test']
    efar = SC[f'{tgt}_s{s}_efar'].astype(bool)
    yT = Y[test]
    cnt = np.bincount(yT, minlength=10)
    big = np.flatnonzero(cnt >= MIN_N)
    out = []
    for f in sorted(glob.glob(f'{R}/guidance_G/rows_t*_{tgt}_s{s}_r*.csv')):
        tier, rep = re.search(r'rows_t(\d)_[A-Z]+_s\d_r(\d)\.csv', f).groups()
        df = pd.read_csv(f)
        for _, r in df.iterrows():
            d = dict(tgt=tgt, split=s, rep=int(rep), tier=int(tier), cond=r['cond'], method=r['method'], clf=r['clf'],
                     k=int(r['k']), n_far_row=int(r['n_far']))
            if r['method'] == 'foreign_only':
                d['probs_rows'] = -1
                d['recomputed'] = False
            else:
                pp = f"{R}/guidance_G/probs/{tgt}_s{s}_r{rep}_{r['clf']}_k{int(r['k'])}_{r['sel_hash']}.npz"
                if not os.path.exists(pp):
                    d['probs_rows'] = -2
                    out.append(d)
                    continue
                z = np.load(pp)
                last = max(int(k[1:]) for k in z.files)
                P = z[f'i{last}'].astype(np.float64)
                d['probs_rows'] = P.shape[0]
                P /= P.sum(1, keepdims=True)
                ph = P.argmax(1)
                oa, ba, f1, rec = scores(yT, ph, big)
                oaf, baf, _, _ = scores(yT, ph, big, efar)
                d.update(recomputed=True, d_oa=abs(oa - r['oa']), d_ba=abs(ba - r['ba_ge30']), d_f1=abs(f1 - r['f1_ge30']),
                         d_oa_far=abs(oaf - r['oa_far']), d_ba_far=abs(baf - r['ba_far']))
                # 以其他测试集重算，确认指标能区分测试集（取 1-s 划分的同大小前缀不行，改用打乱标签对照）
            d['nan_pattern_row'] = ''.join('N' if pd.isna(r[f'rec{c}']) else 'v' for c in range(10))
            d['nan_pattern_test'] = ''.join('N' if cnt[c] == 0 else 'v' for c in range(10))
            d['n_far_test'] = int(efar.sum())
            d['n_test'] = len(test)
            out.append(d)
    return out


def g_stage0_p0(args):
    tgt, s = args
    sh = npz_shapes(f'{R}/guidance_G/stage0/{tgt}_s{s}.npz')
    n_test = len(SC[f'{tgt}_s{s}_test'])
    n_cand = len(SC[f'{tgt}_s{s}_cand'])
    p0 = {k: v[0] for k, v in sh.items() if k.startswith('P0test')}
    other = {k: v[0] for k, v in sh.items() if k in ('P_T_ae', 'P_T_s2', 'G_ae', 'G_s2', 'RZ_ae', 'NT_ae', 'efar', 'd_far')}
    return dict(tgt=tgt, split=s, n_test=n_test, n_cand=n_cand, p0_ok=all(v[0] == n_test for v in p0.values()),
                n_p0=len(p0), allT_arrays_ok=all(v[0] == n_test + n_cand for k, v in other.items() if k not in ('efar', 'd_far')),
                efar_ok=other['efar'][0] == n_test and other['d_far'][0] == n_test)


def g_orphans():
    """probs 目录中每个文件的行数 = 该 (国家, 划分) 的测试集大小。"""
    bad, n = [], 0
    for p in sorted(glob.glob(f'{R}/guidance_G/probs/*.npz')):
        m = re.match(r'([A-Z]+)_s(\d)_r(\d)_(\w+?)_k(\d+)_([0-9a-f]+)\.npz', os.path.basename(p))
        tgt, s = m.group(1), int(m.group(2))
        sh = npz_shapes(p)
        n += 1
        nt = len(SC[f'{tgt}_s{s}_test'])
        for k, (shape, dt) in sh.items():
            if shape[0] != nt or shape[1] != 10:
                bad.append((os.path.basename(p), k, shape, nt))
    return n, bad


def a_unit(path):
    tgt, s, rep = re.search(r'rows_\w+?_([A-Z]+)_s(\d)_r(\d)\.csv', path).groups()
    s, rep = int(s), int(rep)
    mode = 'rgrid' if 'rgrid' in path else 'run'
    test = SC[f'{tgt}_s{s}_test']
    yT = Y[test]
    cnt = np.bincount(yT, minlength=10)
    big = np.flatnonzero(cnt >= MIN_N)
    out = []
    for _, r in pd.read_csv(path).iterrows():
        rm = r['r']
        rtag = f"R{int(rm)}" if mode == 'rgrid' else f"R{rm}"
        tag = f"{tgt}_s{s}_r{rep}_{r['method']}_k{int(r['k'])}_{rtag}"
        pp = f'{R}/stageA/probs/{tag}.npy'
        d = dict(mode=mode, tgt=tgt, split=s, rep=rep, method=r['method'], k=int(r['k']), r=rm, n_test=len(test))
        if not os.path.exists(pp):
            d['probs_rows'] = -2
            out.append(d)
            continue
        P = np.load(pp, mmap_mode='r')
        d['probs_rows'] = P.shape[0]
        ph = np.asarray(P).astype(np.float32).argmax(1)
        oa, ba, f1, rec = scores(yT, ph, big)
        d.update(d_oa=abs(oa - r['oa']), d_ba=abs(ba - r['ba_ge30']), d_f1=abs(f1 - r['f1_ge30']),
                 nan_ok=''.join('N' if pd.isna(r[f'rec{c}']) else 'v' for c in range(10)) ==
                 ''.join('N' if cnt[c] == 0 else 'v' for c in range(10)))
        out.append(d)
    return out


if __name__ == '__main__':
    import guidance_G as G
    units = [(t, s) for t in G.COUNTRIES_G for s in (0, 1)]
    with mp.Pool(4) as pool:
        g_rows = [x for L in pool.map(g_unit, units) for x in L]
        p0 = pool.map(g_stage0_p0, units)
        a_rows = [x for L in pool.map(a_unit, sorted(glob.glob(f'{R}/stageA/rows_*_s*_r*.csv'))) for x in L]
    gdf = pd.DataFrame(g_rows)
    gdf.to_csv(f'{OUTD}/a2_G_rows.csv', index=False)
    print('G rows:', len(gdf), 'recomputed:', int(gdf.recomputed.sum()), 'missing probs:', int((gdf.probs_rows == -2).sum()))
    rc = gdf[gdf.recomputed == True]
    print('probs_rows == n_test (recomputed rows):', bool((rc.probs_rows == rc.n_test).all()))
    for c in ('d_oa', 'd_ba', 'd_f1', 'd_oa_far', 'd_ba_far'):
        print(f'  max |{c}| =', rc[c].max())
    print('n_far_row == n_far_test all rows:', bool((gdf.n_far_row == gdf.n_far_test).all()))
    print('rec NaN pattern == test absent-class pattern all rows:', bool((gdf.nan_pattern_row == gdf.nan_pattern_test).all()))
    per = gdf.groupby(['tgt', 'split']).agg(n_rows=('k', 'size'), n_far_unique=('n_far_row', 'nunique'),
                                            nan_unique=('nan_pattern_row', 'nunique'), n_test=('n_test', 'first'))
    print(per.to_string())
    print('stage0 P0/arrays:', pd.DataFrame(p0)[['p0_ok', 'allT_arrays_ok', 'efar_ok']].all().to_dict(),
          'n_p0 per npz', set(d['n_p0'] for d in p0))
    n, bad = g_orphans()
    print('G probs files scanned:', n, 'bad shapes:', len(bad), bad[:5])
    adf = pd.DataFrame(a_rows)
    adf.to_csv(f'{OUTD}/a2_A_rows.csv', index=False)
    print('StageA rows:', len(adf), 'missing probs:', int((adf.probs_rows == -2).sum()),
          'rows==n_test:', bool((adf[adf.probs_rows >= 0].probs_rows == adf[adf.probs_rows >= 0].n_test).all()))
    ok = adf[adf.probs_rows >= 0]
    for c in ('d_oa', 'd_ba', 'd_f1'):
        print(f'  A max |{c}| =', ok[c].max(), ' n>1e-6:', int((ok[c] > 1e-6).sum()))
    print('  A nan pattern ok:', bool(ok.nan_ok.all()))
    print(ok.groupby(['mode', 'tgt', 'split']).size().to_string())
