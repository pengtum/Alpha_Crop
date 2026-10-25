"""汇总 jobs/*.json → 表格（report_fit.txt 与若干 csv）。"""
import glob
import json
import os

import numpy as np
import pandas as pd

OUT = '/Users/perng_mini/AlphaCrops_local/verify/model_diag/fit'
recs = []
checks = []
for p in sorted(glob.glob(f'{OUT}/jobs/*.json')):
    r = json.load(open(p))
    j = r['job']
    base = dict(kind=j['kind'], tgt=j['tgt'], split=j['split'], feat=j['feat'], trset=j.get('trset', 'k0'),
                cfg=j.get('cfg', 'L63'), R=j['rounds'], t=r['t'])
    if j['kind'] == 'k0':
        base['kind'] = 'cap'  # k0 L63 同时是容量扫描的基线
    if 'check_maxabs' in r:
        checks.append(dict(base, maxabs=r['check_maxabs'], agree=r['check_argmax_agree']))
    for ev, d in r['metrics'].items():
        ll = r['ll'][ev]
        for it, m in d.items():
            it = int(it)
            recs.append(dict(base, ev=ev, it=it, oa=m['oa'], ba=m['ba'], nll=m['nll'], lgb_ll=ll[it - 1],
                             **{f'rec{c}': (m['rec'][c] if c in m['big'] else np.nan) for c in range(10)}))
df = pd.DataFrame(recs)
df.to_csv(f'{OUT}/curves_long.csv', index=False)
pd.DataFrame(checks).to_csv(f'{OUT}/repro_checks.csv', index=False)
lines = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    lines.append(s)


P('== 复现核对（第 300 轮目标测试概率 vs 阶段 0 / G probs 缓存）==')
if checks:
    c = pd.DataFrame(checks)
    P(c.groupby(['trset', 'feat'])[['maxabs', 'agree']].agg(['max', 'min']).to_string())

# (1) 源域拟合：训练集、外国随机留出、目标（k0 L63）@300
P('\n== (1)(2) 源域拟合与迁移，L63，300 轮，11 国划分 0 平均 ==')
k0 = df[(df.kind == 'cap') & (df.trset == 'k0') & (df.cfg == 'L63') & (df.split == 0)]
hold = df[(df.kind == 'hold') & (df.split == 0)]
for it in (50, 100, 300, 500, 1000):
    a = k0[(k0.it == it)].groupby(['feat', 'ev'])[['oa', 'ba', 'lgb_ll']].mean()
    b = hold[(hold.it == it)].groupby(['feat', 'ev'])[['oa', 'ba', 'lgb_ll']].mean()
    P(f'-- 第 {it} 轮 --')
    P('外国 5 万训练（k0）：\n' + a.round(2).to_string())
    if len(b):
        P('外国 80/20：\n' + b.round(2).to_string())
    P(f"国家数 k0={k0[k0.it == it].tgt.nunique()}，hold={hold[hold.it == it].tgt.nunique()}")

# 逐国 @300 表
t300 = k0[(k0.it == 300)].pivot_table(index='tgt', columns=['feat', 'ev'], values=['ba', 'oa'])
h300 = hold[(hold.it == 300) & (hold.ev == 'fhold')].pivot_table(index='tgt', columns='feat', values=['ba', 'oa'])
oof = pd.read_csv(f'{OUT}/oof_stage0.csv')
oof0 = oof[oof.split == 0].pivot_table(index='tgt', columns='feat', values=['oof_ba', 'oof_oa', 'oof_percountry_ba'])
def _flat(t, pre):
    t = t.copy()
    t.columns = [pre + '_' + '_'.join(map(str, c)) if isinstance(c, tuple) else pre + '_' + str(c) for c in t.columns]
    return t
tab = pd.concat([_flat(t300, 'k0'), _flat(h300, 'hold'), _flat(oof0, 'oof')], axis=1)
tab.round(1).to_csv(f'{OUT}/table_source_vs_target_300.csv')

# 汇总 ba / OA 阶梯
P('\n== 阶梯（300 轮，11 国平均）：训练集 → 外国随机留出 → 外国留国家组（阶段 0 袋外）→ 目标 k=0 ==')
for f in ('ae', 's2'):
    tr = k0[(k0.it == 300) & (k0.ev == 'train') & (k0.feat == f)]
    te = k0[(k0.it == 300) & (k0.ev == 'test') & (k0.feat == f)]
    ho = hold[(hold.it == 300) & (hold.ev == 'fhold') & (hold.feat == f)]
    of = oof[(oof.split == 0) & (oof.feat == f)]
    P(f"{f}: 训练 ba {tr.ba.mean():.1f} OA {tr.oa.mean():.1f} | 随机留出 ba {ho.ba.mean():.1f} OA {ho.oa.mean():.1f} "
      f"(n国={ho.tgt.nunique()}) | 留组袋外 ba(合并) {of.oof_ba.mean():.1f} ba(逐国均) {of.oof_percountry_ba.mean():.1f} "
      f"OA {of.oof_oa.mean():.1f} | 目标 ba_ge30 {te.ba.mean():.1f} OA {te.oa.mean():.1f} (n国={te.tgt.nunique()})")

# (3) 目标域上限
P('\n== (3) 目标域上限（划分 0，同一目标测试集）==')
tg = df[(df.kind == 'tgt') & (df.ev == 'test')]
fp = df[(df.kind == 'fp') & (df.ev == 'test')]
for it in (100, 300, 500):
    x = tg[tg.it == it].groupby('feat')[['ba', 'oa']].mean()
    P(f'只用目标候选池 @{it}:\n' + x.round(2).to_string())
x = fp[fp.it == 300].groupby('feat')[['ba', 'oa']].mean()
P('外国 + 全部候选池(r=1) @300:\n' + x.round(2).to_string())
rows = pd.read_csv(f'{OUT}/rows_k0_srs.csv')
r0 = rows[rows.split == 0]
cmp = []
for f in ('ae', 's2'):
    for c in sorted(df.tgt.unique()):
        g = lambda d, it: d[(d.tgt == c) & (d.feat == f) & (d.it == it)]
        a = g(k0[k0.ev == 'test'], 300)
        b = g(tg, 300)
        e = g(tg, 500)
        d_ = g(fp, 300)
        s = r0[(r0.target == c) & (r0.clf == f)].set_index('k')
        if len(a) and len(b):
            cmp.append(dict(feat=f, tgt=c, k0_ba=a.ba.iloc[0], k0_oa=a.oa.iloc[0],
                            srs200_ba=s.loc[200, 'ba_ge30'] if 200 in s.index else np.nan,
                            srs400_ba=s.loc[400, 'ba_ge30'] if 400 in s.index else np.nan,
                            tgt300_ba=b.ba.iloc[0], tgt300_oa=b.oa.iloc[0],
                            tgt500_ba=e.ba.iloc[0] if len(e) else np.nan,
                            fp_ba=d_.ba.iloc[0] if len(d_) else np.nan, fp_oa=d_.oa.iloc[0] if len(d_) else np.nan))
cmp = pd.DataFrame(cmp, columns=['feat', 'tgt', 'k0_ba', 'k0_oa', 'srs200_ba', 'srs400_ba', 'tgt300_ba', 'tgt300_oa',
                                 'tgt500_ba', 'fp_ba', 'fp_oa'])
cmp['gap_ba'] = cmp.tgt300_ba - cmp.k0_ba
cmp['gap_oa'] = cmp.tgt300_oa - cmp.k0_oa
cmp.round(2).to_csv(f'{OUT}/table_upper_bound.csv', index=False)
P(cmp.round(1).to_string())
P(cmp.groupby('feat')[['k0_ba', 'srs200_ba', 'srs400_ba', 'tgt300_ba', 'fp_ba', 'gap_ba', 'k0_oa', 'tgt300_oa',
                       'fp_oa', 'gap_oa']].mean().round(2).to_string())
if len(cmp):
    w = cmp.pivot(index='tgt', columns='feat', values='tgt300_ba').dropna()
    if len(w):
        P(f'目标上限 AE−S2 ba 平均 {(w.ae - w.s2).mean():+.2f}，>0 国家 {(w.ae > w.s2).sum()}/{len(w)}')
    w = cmp.pivot(index='tgt', columns='feat', values='k0_ba').dropna()
    P(f'k=0 AE−S2 ba 平均 {(w.ae - w.s2).mean():+.2f}，>0 国家 {(w.ae > w.s2).sum()}/{len(w)}')

# (4) 容量扫描
P('\n== (4) 容量扫描：目标 ba_ge30 / OA / 训练集加权 logloss（按国家平均；只含已完成的国家）==')
cap = df[(df.kind == 'cap')]
for trs in ('k0', 'k200'):
    for cfg in ('L15', 'L63', 'reg', 'L255'):
        x = cap[(cap.trset == trs) & (cap.cfg == cfg)]
        if not len(x):
            continue
        n = x.tgt.nunique()
        te = x[x.ev == 'test'].groupby(['feat', 'it'])[['ba', 'oa', 'nll']].mean()
        tr = x[x.ev == 'train'].groupby(['feat', 'it'])[['lgb_ll', 'oa']].mean().rename(
            columns={'lgb_ll': 'train_wll', 'oa': 'train_oa'})
        P(f'-- {trs} {cfg}（{n} 国：{sorted(x.tgt.unique())}）--')
        P(pd.concat([te, tr], axis=1).round(2).to_string())
cap.to_csv(f'{OUT}/capacity_long.csv', index=False)
open(f'{OUT}/report_fit.txt', 'w').write('\n'.join(lines))

# (5) G 的指导对比（θ：AE 指导 − S2 指导，k=200，lpm2_nogeo + typiclust，单元 s0 r0）随轮数
P('\n== (5) 指导对比 θ(AE−S2) 随轮数（k=200，lpm2_nogeo+typiclust，两种分类器平均，s0 r0）==')
gd = df[(df.kind == 'guide') & (df.ev == 'test')]
if len(gd):
    GRES = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
    rr = pd.concat([pd.read_csv(f'{GRES}/rows_t1_{c}_s0_r0.csv') for c in sorted(gd.tgt.unique())])
    rr = rr[(rr.k == 200) & rr.method.isin(['lpm2_nogeo', 'typiclust']) & rr.cond.isin(['AE', 'S2'])]
    import re
    # 本脚本的 guide 作业：从 json 名中取 cond/method
    gl = []
    for p in sorted(glob.glob(f'{OUT}/jobs/guide_*.json')):
        r = json.load(open(p))
        j = r['job']
        for it, m in r['metrics']['test'].items():
            gl.append(dict(tgt=j['tgt'], clf=j['feat'], cond=j['cond'], method=j['method'], it=int(it), ba=m['ba'],
                           oa=m['oa']))
    gl = pd.DataFrame(gl)
    for it in (200, 250, 300):
        x = rr[['target', 'clf', 'cond', 'method', f'ba_i{it}', f'oa_i{it}']].rename(
            columns={'target': 'tgt', f'ba_i{it}': 'ba', f'oa_i{it}': 'oa'})
        x['it'] = it
        gl = pd.concat([gl, x], ignore_index=True)
    # 只保留每个国家四种组合都齐全的
    piv = gl.pivot_table(index=['tgt', 'clf', 'method', 'it'], columns='cond', values=['ba', 'oa'])
    th = pd.DataFrame({'ba': piv['ba']['AE'] - piv['ba']['S2'], 'oa': piv['oa']['AE'] - piv['oa']['S2']}).reset_index()
    full = th.dropna().groupby(['tgt', 'it']).size()
    th_c = th.groupby(['tgt', 'it'])[['ba', 'oa']].mean().reset_index()
    th_c = th_c[th_c.set_index(['tgt', 'it']).index.map(full) == 4]
    from scipy import stats
    W = th_c.pivot(index='tgt', columns='it', values='ba').dropna()
    P(f'国家数 {len(W)}')
    for it in W.columns:
        v = W[it]
        ci = stats.t.interval(0.95, len(v) - 1, v.mean(), v.std(ddof=1) / np.sqrt(len(v)))
        P(f'  第 {it} 轮：θ_ba 平均 {v.mean():+.3f} pp，95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]，>0 国家 {(v > 0).sum()}/{len(v)}')
    P('  逐国 θ_ba：\n' + W.round(2).to_string())
    P('  国家间 Pearson 相关（θ_ba）：\n' + W.corr().round(2).to_string())
    Wo = th_c.pivot(index='tgt', columns='it', values='oa').dropna()
    P('  θ_oa 平均：' + ' '.join(f'{it}:{Wo[it].mean():+.3f}' for it in Wo.columns))
    # 方法 − SRS 与 k=0 → k=200 增益（6 个容量国家）
    ab = gl.groupby(['tgt', 'clf', 'it'])['ba'].mean().rename('guided').reset_index()
    th_c.to_csv(f'{OUT}/theta_by_rounds.csv', index=False)
    gl.to_csv(f'{OUT}/guide_long.csv', index=False)

# SRS k=200 相对 k=0 的增益随轮数（对选样的敏感度）
P('\n== k=200 SRS 相对 k=0 的 ba 增益随轮数（L63，6 个容量国家）==')
c0 = cap[(cap.cfg == 'L63') & (cap.trset == 'k0') & (cap.ev == 'test')]
c2 = cap[(cap.cfg == 'L63') & (cap.trset == 'k200') & (cap.ev == 'test')]
if len(c2):
    m = c0.merge(c2, on=['tgt', 'feat', 'it'], suffixes=('_0', '_2'))
    m['gain_ba'] = m.ba_2 - m.ba_0
    m['gain_oa'] = m.oa_2 - m.oa_0
    P(m.groupby(['feat', 'it'])[['ba_0', 'ba_2', 'gain_ba', 'oa_0', 'oa_2', 'gain_oa']].mean().round(2).to_string())
    P(f"国家：{sorted(m.tgt.unique())}")
    if len(gd):
        g2 = gl[gl.tgt.isin(m.tgt.unique())].groupby(['tgt', 'clf', 'it'])['ba'].mean().reset_index()
        g2 = g2.merge(m[['tgt', 'feat', 'it', 'ba_2', 'ba_0']].rename(columns={'feat': 'clf'}), on=['tgt', 'clf', 'it'])
        g2['guided_minus_srs'] = g2.ba - g2.ba_2
        g2['guided_gain_vs_k0'] = g2.ba - g2.ba_0
        P('指导选样（AE、S2、两种方法平均）− SRS，及相对 k=0：\n' +
          g2.groupby(['clf', 'it'])[['guided_minus_srs', 'guided_gain_vs_k0']].mean().round(2).to_string())
open(f'{OUT}/report_fit.txt', 'w').write('\n'.join(lines))

# (6) 补充：轮数的"可实现"选择与神谕选择；逐类召回随轮数
P('\n== (6) 轮数选择：外国随机留出 logloss 最小的轮 vs 目标测试 ba 最优轮（k0 L63）==')
sel_rows = []
for p in sorted(glob.glob(f'{OUT}/jobs/hold_*.json')):
    r = json.load(open(p))
    j = r['job']
    ll = np.array(r['ll']['fhold'])
    best = int(np.argmin(ll)) + 1
    k0j = f"{OUT}/jobs/k0_{j['tgt']}_0_{j['feat']}_k0_L63_R1000.json"
    if not os.path.exists(k0j):
        continue
    k = json.load(open(k0j))
    cps = sorted(int(x) for x in k['metrics']['test'])
    near = min(cps, key=lambda c: abs(c - best))
    tb = {c: k['metrics']['test'][str(c)]['ba'] for c in cps}
    to = {c: k['metrics']['test'][str(c)]['oa'] for c in cps}
    tll = np.array(k['ll']['test'])
    fh = {c: r['metrics']['fhold'][str(c)] for c in (int(x) for x in r['metrics']['fhold'])}
    sel_rows.append(dict(tgt=j['tgt'], feat=j['feat'], es_round=best, es_cp=near, ba_es=tb[near], oa_es=to[near],
                         ba_300=tb[300], oa_300=to[300], ba_best=max(tb.values()),
                         best_cp=max(tb, key=tb.get), tll_min_round=int(np.argmin(tll)) + 1,
                         fhold_ba_100=fh[100]['ba'], fhold_ba_300=fh[300]['ba'], fhold_ba_500=fh[500]['ba'],
                         fhold_oa_100=fh[100]['oa'], fhold_oa_300=fh[300]['oa'], fhold_oa_500=fh[500]['oa']))
if sel_rows:
    sr = pd.DataFrame(sel_rows)
    sr.to_csv(f'{OUT}/table_round_selection.csv', index=False)
    P(sr.round(1).to_string())
    P(sr.groupby('feat')[['es_round', 'ba_es', 'oa_es', 'ba_300', 'oa_300', 'ba_best', 'tll_min_round', 'fhold_ba_100',
                          'fhold_ba_300', 'fhold_ba_500', 'fhold_oa_300']].mean().round(2).to_string())
    w = sr.pivot(index='tgt', columns='feat', values=['ba_es', 'ba_300'])
    if ('ba_es', 'ae') in w and ('ba_es', 's2') in w:
        d_es = (w['ba_es']['ae'] - w['ba_es']['s2']).dropna()
        d_3 = (w['ba_300']['ae'] - w['ba_300']['s2']).dropna()
        P(f'k=0 分类器 AE−S2 ba：300 轮 {d_3.mean():+.2f}（>0 {int((d_3 > 0).sum())}/{len(d_3)}）；'
          f'按外国留出早停 {d_es.mean():+.2f}（>0 {int((d_es > 0).sum())}/{len(d_es)}）')

P('\n== 逐类召回（k0 L63，目标测试，11 国平均，仅 big 类）：第 50 / 100 / 300 / 1000 轮 ==')
t = df[(df.kind == 'cap') & (df.trset == 'k0') & (df.cfg == 'L63') & (df.ev == 'test') & df.it.isin([50, 100, 300, 1000])]
classes = ['grassland', 'maize', 'orchards', 'other_crops', 'potatoes', 'root_crops', 'strawberries',
           'tree_wood_forest', 'vegetables', 'winter_cereals']
for f in ('ae', 's2'):
    x = t[t.feat == f]
    out = {}
    for c in range(10):
        # 只在 big 类的国家上平均：big 由测试集计数决定，这里用召回非空且国家 big 包含 c 的近似
        out[classes[c]] = x.groupby('it')[f'rec{c}'].mean() * 100
    P(f'{f}:\n' + pd.DataFrame(out).T.round(1).to_string())
open(f'{OUT}/report_fit.txt', 'w').write('\n'.join(lines))

# (7) 容量配置对比：只在该配置已完成的（国家, 特征, 训练集）上与 L63 配对
P('\n== (7) 容量配置 vs L63（配对：同一国家/特征/训练集；目标测试）==')
t = cap[cap.ev == 'test']
tr_ = cap[cap.ev == 'train']
for cfg in ('L15', 'L255', 'reg'):
    ok = t[t.cfg == cfg][['tgt', 'feat', 'trset']].drop_duplicates()
    if not len(ok):
        continue
    x = t.merge(ok, on=['tgt', 'feat', 'trset'])
    x = x[x.cfg.isin([cfg, 'L63'])]
    its = sorted(t[t.cfg == cfg].it.unique())
    x = x[x.it.isin(its)]
    p = x.pivot_table(index=['trset', 'feat', 'it'], columns='cfg', values=['ba', 'oa', 'nll'])
    y = tr_.merge(ok, on=['tgt', 'feat', 'trset'])
    y = y[y.cfg.isin([cfg, 'L63']) & y.it.isin(its)].pivot_table(index=['trset', 'feat', 'it'], columns='cfg',
                                                                   values=['lgb_ll', 'oa'])
    y.columns = [f'train_{a}_{b}' for a, b in y.columns]
    P(f'-- {cfg}（配对单元数：' + ', '.join(f'{k}={v}' for k, v in ok.groupby(['trset', 'feat']).size().items()) + '）--')
    P(pd.concat([p.round(2), y.round(3)], axis=1).to_string())
    # 最优轮（按目标 ba，神谕）与第 300 轮（或最大轮）的比较
    best = x.groupby(['trset', 'feat', 'cfg', 'tgt']).ba.max().groupby(['trset', 'feat', 'cfg']).mean().unstack()
    P('各国最优轮 ba（神谕，目标测试选轮）平均：\n' + best.round(2).to_string())
    # AE−S2（分类器特征）差：同一配置、同一轮
    both = x.pivot_table(index=['trset', 'tgt', 'it', 'cfg'], columns='feat', values='ba').dropna()
    both['d'] = both['ae'] - both['s2']
    P('分类器 AE−S2 ba（同国配对）：\n' + both.groupby(['trset', 'cfg', 'it']).d.agg(['mean', 'count']).unstack('cfg').round(2).to_string())
open(f'{OUT}/report_fit.txt', 'w').write('\n'.join(lines))
