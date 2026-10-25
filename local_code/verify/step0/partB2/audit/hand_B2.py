# 审计：手算 3 个单元的 FA(PK-200)、FA(RT-200)、FA(Z0)（新网格），不用 step0B 的任何函数
import os, sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True
import glob, hashlib
import numpy as np
import pandas as pd
L = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, L)
os.chdir(L)
import guidance_G as G
R = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
C = 10
TAUS = [round(-2.0 + 0.05 * i, 2) for i in range(81)]
OFFS = [round(-4.0 + 0.1 * i, 1) for i in range(41)]

UNITS = [('DE', 1, 0), ('SE', 0, 1), ('LV', 1, 1)]
D = G.GData()
y = np.asarray(D.y)
FB = pd.read_csv('/Users/perng_mini/AlphaCrops_local/verify/step0/partB2/fa_rows_B2.csv', float_precision='round_trip')
FB = FB[FB.grid == 'new']


def curve(Z, yT, big):
    Pa = np.exp(Z - Z.max(1, keepdims=True)); Pa /= Pa.sum(1, keepdims=True)
    pih = Pa.mean(0)
    lp = np.array([np.log(p) if p > 0 else 0.0 for p in pih])
    oa, ba = [], []
    for t in TAUS:
        ph = (Z - t * lp).argmax(1)
        oa.append(100 * np.mean(ph == yT))
        ba.append(100 * np.mean([np.mean(ph[yT == c] == c) for c in big]))
    return np.array(oa), np.array(ba)


def fa(oa, ba, ostar):
    vals = []
    for off in OFFS:
        o = ostar + off
        best = None
        for i in range(len(oa) - 1):
            lo, hi = min(oa[i], oa[i + 1]) - 1e-9, max(oa[i], oa[i + 1]) + 1e-9
            if not (lo <= o <= hi):
                continue
            d = oa[i + 1] - oa[i]
            if abs(d) < 1e-9:
                v = max(ba[i], ba[i + 1])
            else:
                lam = min(max((o - oa[i]) / d, 0.0), 1.0)
                v = ba[i] + lam * (ba[i + 1] - ba[i])
            best = v if best is None else max(best, v)
        vals.append(best)
    r = [v for v in vals if v is not None]
    cov = len(r) / len(vals)
    return (np.mean(r) if cov >= 0.5 else np.nan), cov


out = []
for tgt, sp, rep in UNITS:
    unit = f'{tgt}_s{sp}_r{rep}'
    S = np.load(os.path.join(R, 'stage0', f'{tgt}_s{sp}.npz'))
    nc = len(S['cand'])
    allT, test, fr = S['allT'], S['test'], S['foreign']
    yT = y[test]
    big = [c for c in range(C) if (yT == c).sum() >= G.MIN_N]
    yF = y[fr]
    cw = G.class_weights_from(yF)
    pitr = np.array([cw[yF][yF == c].sum() for c in range(C)]) / cw[yF].sum()
    pool = G.pool_positions(S, sp, rep)
    rows = pd.read_csv(os.path.join(R, f'rows_t1_{unit}.csv'))
    zs = np.load(os.path.join(R, 'sel', f'{unit}_none_srs_k200.npz'))
    sel = zs['sel']
    assert np.array_equal(np.sort(allT[pool][zs['sel_pos']]), np.sort(sel))
    ys = y[sel]
    assert len(ys) == 200
    phat = np.array([(ys == c).sum() + 1.0 for c in range(C)]) / (200 + C)
    for f in ('ae', 's2'):
        g = rows[(rows.method == 'srs') & (rows.cond == 'none') & (rows.k == 200) & (rows.clf == f)]
        assert len(g) == 1
        h = g.iloc[0].sel_hash
        assert hashlib.md5(np.sort(sel.astype(np.int64)).tobytes()).hexdigest()[:16] == h
        P0 = S[f'P_T_{f}'][nc:].astype(np.float64); P0 /= P0.sum(1, keepdims=True)
        with np.errstate(divide='ignore'):
            Z0 = np.log(P0)
            w = np.where(pitr > 0, phat / np.where(pitr > 0, pitr, 1), 0.0)
            Q = P0 * w; Q /= Q.sum(1, keepdims=True)
            Zpk = np.log(Q)
            Pr = np.load(os.path.join(R, 'probs', f'{unit}_{f}_k200_{h}.npz'))['i300'].astype(np.float64)
            Pr /= Pr.sum(1, keepdims=True)
            Zrt = np.log(Pr)
        o0, b0 = curve(Z0, yT, big)
        ostar = o0[TAUS.index(0.0)]
        res = {}
        for arm, Z in (('Z0', Z0), ('PK200', Zpk), ('RT200', Zrt)):
            oa, ba = curve(Z, yT, big)
            res[arm] = fa(oa, ba, ostar)
        fb = FB[(FB.unit == unit) & (FB.clf == f)].set_index('arm')
        for arm in res:
            out.append(dict(unit=unit, clf=f, arm=arm, FA_hand=res[arm][0], reach_hand=res[arm][1],
                            FA_main=fb.loc[arm, 'FA'], reach_main=fb.loc[arm, 'reach']))
        out.append(dict(unit=unit, clf=f, arm='A200', FA_hand=res['PK200'][0] - res['Z0'][0],
                        FA_main=fb.loc['PK200', 'FA'] - fb.loc['Z0', 'FA']))
        out.append(dict(unit=unit, clf=f, arm='R200', FA_hand=res['RT200'][0] - res['PK200'][0],
                        FA_main=fb.loc['RT200', 'FA'] - fb.loc['PK200', 'FA']))
O = pd.DataFrame(out)
O['diff'] = O.FA_hand - O.FA_main
pd.set_option('display.width', 200)
print(O.to_string())
print('max |diff|', O['diff'].abs().max())
