"""独立核实：预测通路 AE−S2 的逐类召回差与测试集流行度的关系；OA 差的逐类分解。
国家层面汇总（先在国家内平均，再跨 11 国 t 检验），避免按行合并的伪重复。"""
import glob, os, json
import numpy as np, pandas as pd
from scipy import stats

OUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
CACHE = '/Users/perng_mini/AlphaCrops_local/data/cache_guidance_G'
y = np.load(os.path.join(CACHE, 'y.npy'), mmap_mode='r')
classes = json.load(open(os.path.join(CACHE, 'meta.json')))['classes']
R = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT, 'rows_t*_*.csv'))], ignore_index=True)
KEY = ['target', 'split', 'rep', 'method', 'clf', 'k']
PRED = ['margin', 'predstrat_srs']
E1 = ['lpm2_nogeo', 'kcenter', 'typiclust', 'kherding']
recs = [f'rec{c}' for c in range(10)]

# 测试集流行度
prev = {}
for t in sorted(R.target.unique()):
    for s in (0, 1):
        z = np.load(os.path.join(OUT, 'stage0', f'{t}_s{s}.npz'))
        yt = np.asarray(y)[z['test']]
        prev[(t, s)] = np.bincount(yt, minlength=10)
# 核对：用 rec 与流行度重构 OA
chk = R[R.cond == 'AE'].head(200)
err = []
for _, r in chk.iterrows():
    n = prev[(r.target, r.split)]
    rc = r[recs].to_numpy(float)
    oa = 100 * np.nansum(n * rc) / n.sum()
    big = n >= 30
    ba = 100 * np.nanmean(rc[big])
    err.append((abs(oa - r.oa), abs(ba - r.ba_ge30)))
print('OA/BA 重构最大误差', np.max(err, 0))


def ts(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]; n = len(v); m = v.mean(); se = v.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(.975, n - 1)
    return f'{m:+.3f} [{m - t * se:+.3f},{m + t * se:+.3f}] p={2 * stats.t.sf(abs(m / se), n - 1):.3f} pos={int((v > 0).sum())}/{n}'


def run(ms, reps, label):
    print(f'\n######## {label} ########')
    A = R[(R.cond == 'AE') & R.method.isin(ms) & R.rep.isin(reps)].set_index(KEY)[recs]
    B = R[(R.cond == 'S2') & R.method.isin(ms) & R.rep.isin(reps)].set_index(KEY)[recs]
    D = (A - B).dropna(how='all') * 100  # pp
    L = D.stack().rename('drec').reset_index().rename(columns={'level_6': 'rc'})
    L['c'] = L.rc.str[3:].astype(int)
    L['n'] = [prev[(t, s)][c] for t, s, c in zip(L.target, L.split, L.c)]
    L['N'] = [prev[(t, s)].sum() for t, s in zip(L.target, L.split)]
    L['pi'] = L.n / L.N
    big = L[L.n >= 30].copy()
    # 在每个 (国家, 划分) 内按流行度把 ≥30 类分成两半（中位数），较少的一半 vs 较多的一半
    meds = big.drop_duplicates(['target', 'split', 'c']).groupby(['target', 'split']).n.median()
    big['med'] = [meds[(t, s)] for t, s in zip(big.target, big.split)]
    big['half'] = np.where(big.n <= big.med, 'rare', 'common')
    # 国家值：先在 (方法, 分类器, k, 半区) 格子内对单元与类平均，再等权
    cvh = big.groupby(['target', 'half', 'method', 'clf', 'k']).drec.mean().groupby(['target', 'half']).mean().unstack()
    print('较少的一半（国家层面）', ts(cvh['rare']))
    print('较多的一半（国家层面）', ts(cvh['common']))
    print('较少 − 较多           ', ts(cvh['rare'] - cvh['common']))
    # 国家内 Spearman：类的 Δrec（在国家内对单元/格子平均）与 log 流行度
    rhos = []
    for t, g in big.groupby('target'):
        cm = g.groupby('c').agg(drec=('drec', 'mean'), pi=('pi', 'mean'))
        rhos.append(stats.spearmanr(cm.drec, cm.pi)[0])
    print('国家内 Spearman(Δrec, 流行度) 均值', ts(rhos))
    # 按行合并的 Spearman（复现审查者的做法）
    print('按行合并 Spearman', round(stats.spearmanr(big.drec, big.pi)[0], 3), 'n', len(big))
    # 逐类（国家层面）
    pc = big.groupby(['target', 'c']).drec.mean().unstack()
    print('逐类 Δrec（国家层面均值，pp）：')
    for c in range(10):
        if c in pc:
            v = pc[c].dropna()
            if len(v) >= 3:
                print(f'   {classes[c]:18s} n国={len(v):2d}', ts(v))
    # OA 分解：ΔOA = Σ_c π_c Δrec_c ；分成最大类、其余 ≥30 类、<30 类
    L['contrib'] = L.pi * L.drec.fillna(0)
    L['rank'] = L.groupby(['target', 'split', 'method', 'clf', 'k']).n.rank(ascending=False, method='first')
    L['grp'] = np.select([L['rank'] == 1, L.n >= 30], ['top1', 'other_big'], 'small')
    oa_parts = L.groupby(['target', 'split', 'method', 'clf', 'k', 'grp']).contrib.sum().unstack(fill_value=0)
    cvo = oa_parts.groupby(['target', 'method', 'clf', 'k']).mean().groupby('target').mean()
    print('ΔOA 分解（国家层面，pp）：')
    for g in cvo.columns:
        print(f'   {g:10s}', ts(cvo[g]))
    print('   合计      ', ts(cvo.sum(1)))
    tops = L[L['rank'] == 1].drop_duplicates(['target', 'split'])[['target', 'split', 'c', 'pi']]
    tops['cls'] = [classes[c] for c in tops.c]
    print('各国最大类：', tops.groupby('target').agg(cls=('cls', 'first'), pi=('pi', 'mean')).round(3).to_dict('index'))
    print(cvo.round(3).to_string())
    return big, cvo


big_p, cvo_p = run(PRED, [0], 'pred（margin、predstrat_srs，重复 0）')
big_g, cvo_g = run(E1, [0, 1], 'geom E1（对照）')
