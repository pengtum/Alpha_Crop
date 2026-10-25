"""诊断 3：换成"冻结外国模型 + 目标头"后，选样杠杆是否变大？
对 G 的全部 k50/k200 选样（44 个单元 × 两种分类器特征）与本脚本构造的类别配额神谕（真实类分层，层内 SRS），
分别用目标头评估，并与 G 的 LightGBM 重训结果（同一选样、同一分类器）配对比较。"""
import sys, time, glob
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/codesign')
from headlib import *  # noqa

y_all = np.load(f'{CACHE}/y.npy')
G = pd.concat([pd.read_csv(f) for f in glob.glob(f'{GOUT}/rows_t*.csv')])
G = G[G.k.isin([50, 200])]
HEADS = {'bias': ('bias', 0.01, 1.0), 'feat03': ('feat', 0.03, 1.0), 'feat1': ('feat', 0.1, 1.0)}


def quota_oracle(y_pool, k, rule, seed):
    """真实类分层：equal / sqrt / prop 分配（m_min=1），层内 SRS。只作神谕上限。"""
    import importlib.util
    rng = np.random.default_rng(seed)
    N_h = np.bincount(y_pool, minlength=C)
    w = {'equal': (N_h > 0).astype(float), 'sqrt': np.sqrt(N_h), 'prop': N_h.astype(float)}[rule]
    n_h = alloc(N_h, w, k, 1)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y_pool == c), n_h[c], replace=False)
                                   for c in range(C) if n_h[c] > 0]))


spec = __import__('importlib.util').util.spec_from_file_location(
    'designs_alloc', '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/'
    'Alpha_Crops/label_expansion/samplers/designs.py')
# 只取 alloc 函数（避免导入整个包的副作用）：复制其源码
src = open(spec.origin).read()
s0 = src.index('def alloc(')
s1 = src.index('\n# ====', s0)
exec(src[s0:s1])

rows = []
t0 = time.time()
for tgt in COUNTRIES:
    for split in (0, 1):
        S = load_stage0(tgt, split)
        n_cand = len(S['cand']); test = S['test']; yT = y_all[test]
        big = np.flatnonzero(np.bincount(yT, minlength=C) >= MIN_N)
        for rep in (0, 1):
            unit = f'{tgt}_s{split}_r{rep}'
            pp = pool_positions(S, split, rep)
            idx_pool = S['allT'][pp]
            y_pool = y_all[idx_pool]
            sels = {}
            for fpath in glob.glob(f'{GOUT}/sel/{unit}_*_k50.npz') + glob.glob(f'{GOUT}/sel/{unit}_*_k200.npz'):
                name = fpath.split('/')[-1][len(unit) + 1:-4]
                cm, k = name.rsplit('_k', 1)
                z = np.load(fpath)
                sels[(cm, int(k))] = z['sel_pos']
            for k in (50, 200):
                for rule in ('equal', 'sqrt', 'prop'):
                    sels[(f'QORACLE_{rule}', k)] = quota_oracle(y_pool, k, rule, 1000 * split + rep + 77)
            gu = G[(G.target == tgt) & (G.split == split) & (G.rep == rep)]
            for f in ('ae', 's2'):
                Zall = logp(S[f'P_T_{f}']); Zp, Zt = Zall[pp], Zall[n_cand:]
                Gf = S[f'G_{f}'].astype(np.float64); Gf = Gf / Gf[:, :1].std()
                Xp, Xt = Gf[pp][:, :16], Gf[n_cand:][:, :16]
                for (cm, k), sp in sels.items():
                    ys = y_pool[sp]
                    base = dict(unit=unit, tgt=tgt, split=split, rep=rep, feat=f, k=k, sel=cm,
                                n_cls=len(np.unique(ys)), comp_L1=float(np.abs(np.bincount(ys, minlength=C) / k
                                                                             - np.bincount(y_pool, minlength=C) / len(y_pool)).sum()))
                    if cm.startswith('QORACLE'):
                        lg = None
                    else:
                        cond, meth = (cm.split('_', 1) if not cm.startswith(('NATIVE', 'RAWZ')) else
                                      ('_'.join(cm.split('_')[:2]), '_'.join(cm.split('_')[2:])))
                        g = gu[(gu.cond == cond) & (gu.method == meth) & (gu.k == k) & (gu.clf == f)]
                        lg = g.iloc[0] if len(g) else None
                    if lg is not None:
                        rows.append(dict(base, learner='lgb', ba=lg.ba_ge30, oa=lg.oa))
                    for hn, (kind, lam, alpha) in HEADS.items():
                        ux = kind == 'feat'
                        pr = fit_head(Zp[sp], ys, kind, Xp[sp] if ux else None, lam, alpha)
                        oa, ba = scores(yT, pr(Zt, Xt if ux else None).argmax(1), big)
                        rows.append(dict(base, learner=hn, ba=ba, oa=oa))
        print(tgt, split, f'{time.time() - t0:.0f}s', flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/diag3_sel_rows.csv', index=False)
