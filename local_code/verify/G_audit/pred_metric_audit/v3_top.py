"""补充：最大类 vs 其余 ≥30 类的 Δrec（国家层面配对）；去 RO；选样中最大类占比（机制）；
E1 的 OA 分解修正（按 rep 分组）。"""
import glob, os, json
import numpy as np, pandas as pd
from scipy import stats

OUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
CACHE = '/Users/perng_mini/AlphaCrops_local/data/cache_guidance_G'
y = np.asarray(np.load(os.path.join(CACHE, 'y.npy'), mmap_mode='r'))
classes = json.load(open(os.path.join(CACHE, 'meta.json')))['classes']
R = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT, 'rows_t*_*.csv'))], ignore_index=True)
KEY = ['target', 'split', 'rep', 'method', 'clf', 'k']
PRED = ['margin', 'predstrat_srs']
E1 = ['lpm2_nogeo', 'kcenter', 'typiclust', 'kherding']
recs = [f'rec{c}' for c in range(10)]
prev = {}
for t in sorted(R.target.unique()):
    for s in (0, 1):
        z = np.load(os.path.join(OUT, 'stage0', f'{t}_s{s}.npz'))
        prev[(t, s)] = np.bincount(y[z['test']], minlength=10)


def ts(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]; n = len(v); m = v.mean(); se = v.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(.975, n - 1)
    return f'{m:+.3f} [{m - t * se:+.3f},{m + t * se:+.3f}] p={2 * stats.t.sf(abs(m / se), n - 1):.3f} pos={int((v > 0).sum())}/{n}'


def cv(d, col='d'):
    return d.groupby(['target', 'method', 'clf', 'k'])[col].mean().groupby('target').mean()


def dl(methods, metric, reps):
    a = R[(R.cond == 'AE') & R.method.isin(methods)].set_index(KEY)[metric]
    b = R[(R.cond == 'S2') & R.method.isin(methods)].set_index(KEY)[metric]
    d = (a - b).dropna().rename('d').reset_index()
    return d[d.rep.isin(reps)]


for label, ms, reps in (('pred', PRED, [0]), ('geom', E1, [0, 1])):
    print(f'\n######## {label} ########')
    A = R[(R.cond == 'AE') & R.method.isin(ms) & R.rep.isin(reps)].set_index(KEY)[recs]
    B = R[(R.cond == 'S2') & R.method.isin(ms) & R.rep.isin(reps)].set_index(KEY)[recs]
    D = ((A - B) * 100).reset_index()
    top_d, rest_d, oa_top, oa_rest = [], [], [], []
    for _, r in D.iterrows():
        n = prev[(r.target, r.split)]
        big = np.flatnonzero(n >= 30); top = int(np.argmax(n))
        rest = [c for c in big if c != top]
        dr = r[recs].to_numpy(float)
        top_d.append(dr[top]); rest_d.append(np.nanmean(dr[rest]))
        oa_top.append(n[top] / n.sum() * dr[top]); oa_rest.append(np.nansum((n / n.sum() * dr)[[c for c in range(10) if c != top]]))
    D['top'] = top_d; D['rest'] = rest_d; D['oa_top'] = oa_top; D['oa_rest'] = oa_rest
    vt, vr = cv(D, 'top'), cv(D, 'rest')
    print('最大类 Δrec          ', ts(vt))
    print('其余 ≥30 类平均 Δrec  ', ts(vr))
    print('其余 − 最大类         ', ts(vr - vt))
    print('ΔOA 中最大类贡献      ', ts(cv(D, 'oa_top')))
    print('ΔOA 中其余类贡献      ', ts(cv(D, 'oa_rest')))
    print(pd.DataFrame({'top_drec': vt, 'rest_drec': vr, 'oa_top': cv(D, 'oa_top'), 'oa_rest': cv(D, 'oa_rest')}).round(3).to_string())
    for drop in (['RO'], ['RO', 'SE'], ['PT']):
        print(f'去 {drop}：')
        for m in ('ba_ge30', 'oa', 'f1_ge30', 'ba_P2', 'oa_P2', 'ba_far'):
            v = cv(dl(ms, m, reps)); v = v.drop(drop)
            print(f'   {m:8s}', ts(v))
        vv = (vr - vt).drop(drop)
        print('   其余−最大类 Δrec', ts(vv))

# 机制：预测通路下 AE 与 S2 选样里最大类（测试集最大类）的占比
print('\n######## 选样中测试集最大类的占比（pred，重复 0） ########')
rows = []
for f in glob.glob(os.path.join(OUT, 'sel', '*_r0_*.npz')):
    b = os.path.basename(f)[:-4]
    t, s, r, rest = b.split('_', 3)
    cond, rest2 = rest.split('_', 1)
    meth, kk = rest2.rsplit('_k', 1)
    if cond not in ('AE', 'S2') or meth not in PRED:
        continue
    sel = np.load(f)['sel']
    sp = int(s[1:]); n = prev[(t, sp)]; top = int(np.argmax(n))
    ys = y[sel]
    rows.append(dict(target=t, split=sp, cond=cond, method=meth, k=int(kk), top_share=(ys == top).mean(),
                     top_test=n[top] / n.sum()))
S = pd.DataFrame(rows)
P = S.pivot_table(index=['target', 'split', 'method', 'k'], columns='cond', values='top_share')
P['d'] = P.AE - P.S2
v = P.reset_index().groupby(['target', 'method', 'k']).d.mean().groupby('target').mean()
print('AE − S2 选样中最大类占比（国家层面）', ts(v * 100), '(pp)')
print((v * 100).round(2).to_dict())
print('平均占比 AE', round(P.AE.mean(), 3), 'S2', round(P.S2.mean(), 3), '测试集', round(S.top_test.mean(), 3))
