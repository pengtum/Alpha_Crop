"""probs 文件核对：clf=ae 与 clf=s2 的概率不同；文件完整性（形状、NaN、行和）；孤儿文件；P0test 的 float16 并列。"""
import glob
import os

import numpy as np
import pandas as pd

RES = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
OUTD = '/Users/perng_mini/AlphaCrops_local/verify/G_audit/rawdata'
R = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, 'rows_t*_*.csv'))], ignore_index=True)
R = R[R.method != 'foreign_only'].copy()
R['unit'] = R.target + '_s' + R.split.astype(str) + '_r' + R.rep.astype(str)
R['pf'] = R.unit + '_' + R.clf + '_k' + R.k.astype(str) + '_' + R.sel_hash + '.npz'
files = set(os.listdir(os.path.join(RES, 'probs')))
ref = set(R.pf)
print('probs files', len(files), 'referenced', len(ref), 'missing', len(ref - files), 'orphans', len(files - ref))
n_test = {}
out = []
bad = []
for (unit, k, h), g in R.groupby(['unit', 'k', 'sel_hash']):
    tgt, s = unit.split('_')[0], unit.split('_')[1]
    if (tgt, s) not in n_test:
        n_test[(tgt, s)] = len(np.load(os.path.join(RES, 'stage0', f'{tgt}_{s}.npz'))['test'])
    P = {}
    for clf in ('ae', 's2'):
        z = np.load(os.path.join(RES, 'probs', f'{unit}_{clf}_k{k}_{h}.npz'))
        if sorted(z.files) != ['i200', 'i250', 'i300']:
            bad.append((unit, clf, k, h, 'keys', z.files))
        for it in z.files:
            a = z[it]
            if a.shape != (n_test[(tgt, s)], 10) or a.dtype != np.float32 or not np.isfinite(a).all() \
                    or np.abs(a.sum(1) - 1).max() > 1e-4:
                bad.append((unit, clf, k, h, it, a.shape, a.dtype))
        P[clf] = z['i300']
        if np.array_equal(z['i200'], z['i300']):
            bad.append((unit, clf, k, h, 'i200==i300'))
    d = np.abs(P['ae'] - P['s2'])
    out.append(dict(unit=unit, k=k, sel_hash=h, maxdiff=float(d.max()), meandiff=float(d.mean()),
                    argmax_agree=float((P['ae'].argmax(1) == P['s2'].argmax(1)).mean()),
                    identical=bool(np.array_equal(P['ae'], P['s2']))))
O = pd.DataFrame(out)
O.to_csv(os.path.join(OUTD, 'probs_ae_vs_s2.csv'), index=False)
print('pairs', len(O), 'identical', O.identical.sum(), 'min maxdiff', O.maxdiff.min(), 'min meandiff', O.meandiff.min())
print('argmax agreement ae vs s2: mean %.3f min %.3f max %.3f' % (O.argmax_agree.mean(), O.argmax_agree.min(),
                                                                   O.argmax_agree.max()))
print('bad', len(bad), bad[:5])
# 同一 sel 下 ae 与 s2 行的指标完全相同的次数
W = R.pivot_table(index=['unit', 'cond', 'method', 'k'], columns='clf', values='ba_ge30')
print('rows ae==s2 ba_ge30 exactly', int((W['ae'] == W['s2']).sum()), 'of', len(W))
# P0test：ae 与 s2 不同；float16 并列
tie = []
for f in sorted(glob.glob(os.path.join(RES, 'stage0', '*.npz'))):
    S = np.load(f)
    for clf in ('ae', 's2'):
        for it in (200, 250, 300):
            a = S[f'P0test_{clf}_i{it}'].astype(np.float32)
            mx = a.max(1, keepdims=True)
            nt = int(((a == mx).sum(1) > 1).sum())
            tie.append(dict(file=os.path.basename(f), clf=clf, it=it, n_tie=nt, n=len(a),
                            rowsum_dev=float(np.abs(a.sum(1) - 1).max())))
    tie[-1]['ae_eq_s2'] = bool(np.array_equal(S['P0test_ae_i300'], S['P0test_s2_i300']))
T = pd.DataFrame(tie)
print(T.groupby(['clf', 'it'])[['n_tie', 'n']].sum())
print('max rowsum dev float16', T.rowsum_dev.max(), 'P0 ae==s2 any', T.ae_eq_s2.fillna(False).any())
T.to_csv(os.path.join(OUTD, 'p0test_ties.csv'), index=False)
