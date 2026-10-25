"""汇总独立复核实现的逐次抽样行（indep_rows.csv）→ indep_summary.txt 与 indep_country.csv。
国家值 = 两个划分 × 20 次抽样的平均；报 11 国与去掉 RO、PT 的 9 国：均值、95% CI（t，df = n−1）、>0 的国家数。
恢复率 = 国家均值之比 mean_c(OA_E − OA_none) / mean_c(OA_oracle − OA_none)。
另给出每个国家值的蒙特卡罗标准误（划分内 20 次抽样的方差），以及国家层均值的蒙特卡罗标准误，便于与主实现对比。"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

D_OUT = sys.argv[1] if len(sys.argv) > 1 else '/Users/perng_mini/AlphaCrops_local/verify/step0/partA_indep'
C11 = 'AT,BE,DE,EE,LT,LV,PT,RO,SE,SI,SK'.split(',')
C9 = [c for c in C11 if c not in ('RO', 'PT')]
N0S = (100, 500, 2000)
FEATS = ('ae', 's2')

df = pd.read_csv(os.path.join(D_OUT, 'indep_rows.csv'))
ref = df[df.scen == 'ref']
draw = df[df.scen != 'ref'].copy()

# 参照（与抽样无关）：国家值 = 两个划分的平均
REF = {}
for f in FEATS:
    for est in ('E_none', 'E_oracle'):
        s = ref[(ref.feat == f) & (ref.est == est)].groupby('target')[['oa', 'ba_ge30', 'prior_L1']].mean()
        REF[(f, est)] = s


def sel(scen, n0, feat, arm, est):
    q = draw[(draw.scen == scen) & (draw.n0 == n0) & (draw.feat == feat) & (draw.arm == arm) & (draw.est == est)]
    return q.set_index(['target', 'split', 'r'])


def country(x):
    """x：以 (target, split, r) 为索引的逐次抽样值 → (国家值 Series, 国家值的蒙特卡罗标准误 Series)。"""
    g = x.groupby(level=['target', 'split'])
    m = g.mean().groupby(level='target').mean()
    v = g.var(ddof=1)                    # 划分内 20 次抽样的方差
    n = g.count()
    se = np.sqrt((v / n).groupby(level='target').sum()) / 2.0
    return m.reindex(C11), se.reindex(C11)


def stat(v, cs):
    a = v.reindex(cs).to_numpy(float)
    n = len(a)
    m = a.mean()
    h = stats.t.ppf(0.975, n - 1) * a.std(ddof=1) / np.sqrt(n)
    return m, m - h, m + h, int((a > 0).sum()), n


def fmt(v, cs, se=None, nd=2):
    m, lo, hi, npos, n = stat(v, cs)
    s = f'{m:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}] >0 {npos}/{n}'
    if se is not None:
        s += f' (MC SE {np.sqrt((se.reindex(cs) ** 2).sum()) / len(cs):.{nd}f})'
    return s


def lvl(v, cs, nd=2):
    a = v.reindex(cs).to_numpy(float)
    return f'{a.mean():.{nd}f}'


LINES = []
CROWS = []


def P(s=''):
    LINES.append(s)


def put(contrast, n0, feat, est, stat_name, v, se=None):
    for c in C11:
        CROWS.append(dict(contrast=contrast, n0=n0, feat=feat, est=est, stat=stat_name, country=c,
                          value=float(v[c]), mc_se=float(se[c]) if se is not None else np.nan))


def percountry(v, nd=2):
    return '  '.join(f'{c} {v[c]:+.{nd}f}' for c in C11)


P('第 0 步 A 部分：独立复核实现的汇总（只依据规格 A；未读取 verify/step0/partA/）')
P(f'数据：{len(draw):,} 条逐次抽样行；22 个单元 × R = 20；OA、BA 为百分数；先验 L1 = ‖p̂ − π_test‖₁（0–2）。')
P('CI：国家值的 t 分布 95% CI（11 国 df = 10，9 国 df = 8）；"MC SE" 为国家层均值的蒙特卡罗标准误（仅抽样随机性）。')
P()

# ------------------------------------------------------------------ 参照
P('=' * 100)
P('参照：E_none 与 E_oracle（P_T 测试行底座；国家值 = 两个划分平均）')
for f in FEATS:
    no, orc = REF[(f, 'E_none')], REF[(f, 'E_oracle')]
    gap = (orc.oa - no.oa).reindex(C11)
    P(f'  [{f}] OA_none 11 国均值 {lvl(no.oa, C11)}，9 国 {lvl(no.oa, C9)}；OA_oracle 11 国 {lvl(orc.oa, C11)}，9 国 {lvl(orc.oa, C9)}')
    P(f'       OA_oracle − OA_none：11 国 {fmt(gap, C11)}；9 国 {fmt(gap, C9)}')
    P(f'       国家值：{percountry(gap)}')
    put('ref', 0, f, 'E_oracle-E_none', 'dOA', gap)
    put('ref', 0, f, 'E_none', 'OA', no.oa.reindex(C11))
    put('ref', 0, f, 'E_oracle', 'OA', orc.oa.reindex(C11))
    P(f'       BA：none 11 国 {lvl(no.ba_ge30, C11)}，oracle 11 国 {lvl(orc.ba_ge30, C11)}；oracle − none 11 国 '
      f'{fmt((orc.ba_ge30 - no.ba_ge30).reindex(C11), C11)}')
P()

# ------------------------------------------------------------------ A-c1
P('=' * 100)
P('A-c1 初始样本的偏差：E_S0 在 clus 与 rand 下的先验 L1 与 OA 恢复率')
for n0 in N0S:
    P('-' * 100)
    P(f'n0 = {n0}')
    xc = sel('clus', n0, 'ae', '-', 'E_S0')
    xr = sel('rand', n0, 'ae', '-', 'E_S0')
    for name, col in (('S0 偏差强度 L1(S0 构成, 抽样框构成)', 'bias_S0'), ('先验 L1(E_S0)', 'prior_L1')):
        vc, sc = country(xc[col])
        vr, sr = country(xr[col])
        vd, sd = country(xc[col] - xr[col])
        P(f'  {name}：clus 11 国 {lvl(vc, C11, 3)}，rand {lvl(vr, C11, 3)}；clus − rand 11 国 {fmt(vd, C11, sd, 3)}；'
          f'9 国 {fmt(vd, C9, sd, 3)}')
        P(f'     国家值 clus − rand：{percountry(vd, 3)}')
        put('A-c1', n0, 'all', 'E_S0', f'{col}_clus', vc, sc)
        put('A-c1', n0, 'all', 'E_S0', f'{col}_rand', vr, sr)
        put('A-c1', n0, 'all', 'E_S0', f'{col}_clus-rand', vd, sd)
    nz, _ = country(xc['n_zones'])
    P(f'  clus 所用区数：11 国均值 {lvl(nz, C11)}；国家值 ' + '  '.join(f'{c} {nz[c]:.2f}' for c in C11))
    put('A-c1', n0, 'all', 'E_S0', 'n_zones_clus', nz)
    for f in FEATS:
        no = REF[(f, 'E_none')].oa.reindex(C11)
        gap = (REF[(f, 'E_oracle')].oa - REF[(f, 'E_none')].oa).reindex(C11)
        res = {}
        for scen in ('clus', 'rand'):
            x = sel(scen, n0, f, '-', 'E_S0')
            v, s = country(x['oa'])
            res[scen] = (v, s)
            g = v - no
            put('A-c1', n0, f, 'E_S0', f'OA_{scen}', v, s)
            put('A-c1', n0, f, 'E_S0', f'dOA_vs_none_{scen}', g, s)
            rec11 = g.reindex(C11).mean() / gap.reindex(C11).mean()
            rec9 = g.reindex(C9).mean() / gap.reindex(C9).mean()
            P(f'  [{f}] {scen}: OA(E_S0) − OA_none 11 国 {fmt(g, C11, s)}；9 国 {fmt(g, C9, s)}；'
              f'恢复率 11 国 {rec11:.3f}，9 国 {rec9:.3f}')
        vd, sd = country(sel('clus', n0, f, '-', 'E_S0')['oa'] - sel('rand', n0, f, '-', 'E_S0')['oa'])
        put('A-c1', n0, f, 'E_S0', 'dOA_clus-rand', vd, sd)
        P(f'  [{f}] OA(E_S0) clus − rand：11 国 {fmt(vd, C11, sd)}；9 国 {fmt(vd, C9, sd)}')
        P(f'       国家值：{percountry(vd)}')
        vb, sb = country(sel('clus', n0, f, '-', 'E_S0')['ba_ge30'] - sel('rand', n0, f, '-', 'E_S0')['ba_ge30'])
        P(f'  [{f}] （附）BA(E_S0) clus − rand：11 国 {fmt(vb, C11, sb)}；9 国 {fmt(vb, C9, sb)}')
P()

# ------------------------------------------------------------------ A-c2
P('=' * 100)
P('A-c2 先验用哪些标签估（k = 200，SRS 补采）：OA(E_S1) − OA(E_pool) 与 OA(E_S1) − OA(E_S0)')
for scen in ('clus', 'rand'):
    for n0 in N0S:
        P('-' * 100)
        P(f'S0 = {scen}，n0 = {n0}')
        xs1 = sel(scen, n0, 'ae', 'SRS', 'E_S1')
        xpl = sel(scen, n0, 'ae', 'SRS', 'E_pool')
        xs0 = sel(scen, n0, 'ae', '-', 'E_S0')
        for nm, a, b in (('S1 − pool', xs1, xpl), ('S1 − S0', xs1, xs0)):
            vd, sd = country(a['prior_L1'] - b['prior_L1'])
            P(f'  先验 L1 {nm}：11 国 {fmt(vd, C11, sd, 3)}；9 国 {fmt(vd, C9, sd, 3)}')
            put('A-c2', n0, 'all', f'{scen}|SRS', f'L1_{nm.replace(" ", "")}', vd, sd)
        for f in FEATS:
            xs1 = sel(scen, n0, f, 'SRS', 'E_S1')
            xpl = sel(scen, n0, f, 'SRS', 'E_pool')
            xs0 = sel(scen, n0, f, '-', 'E_S0')
            no = REF[(f, 'E_none')].oa.reindex(C11)
            gap = (REF[(f, 'E_oracle')].oa - REF[(f, 'E_none')].oa).reindex(C11)
            lv = {}
            for nm, x in (('S1', xs1), ('pool', xpl), ('S0', xs0)):
                v, _ = country(x['oa'])
                lv[nm] = v
            P(f'  [{f}] OA 水平（11 国）：E_S1 {lvl(lv["S1"], C11)}，E_pool {lvl(lv["pool"], C11)}，'
              f'E_S0 {lvl(lv["S0"], C11)}，none {lvl(no, C11)}；恢复率 11 国 S1 '
              f'{(lv["S1"] - no).mean() / gap.mean():.3f}，pool {(lv["pool"] - no).mean() / gap.mean():.3f}，'
              f'S0 {(lv["S0"] - no).mean() / gap.mean():.3f}')
            for nm, a, b in (('S1 − pool', xs1, xpl), ('S1 − S0', xs1, xs0)):
                vd, sd = country(a['oa'] - b['oa'])
                put('A-c2', n0, f, f'{scen}|SRS', f'dOA_{nm.replace(" ", "")}', vd, sd)
                P(f'  [{f}] OA(E_{nm.split(" − ")[0]}) − OA(E_{nm.split(" − ")[1]})：11 国 {fmt(vd, C11, sd)}；'
                  f'9 国 {fmt(vd, C9, sd)}')
                P(f'       国家值：{percountry(vd)}')
P()

# ------------------------------------------------------------------ A-c3
P('=' * 100)
P('A-c3 成本口径下补采放在哪里（clus，B = 20，每簇 10 个）：EXT、SPR（c2 = 2）、SPR=（c2 = 1）')
for n0 in N0S:
    P('-' * 100)
    P(f'n0 = {n0}')
    xspr = sel('clus', n0, 'ae', 'SPR', 'E_S1')
    nl, snl = country(xspr['n_labels'])
    ncl, _ = country(xspr['spr_clusters'])
    nin, _ = country(xspr['spr_in'])
    P(f'  所得标签数（S1）：EXT = SPR= = 200；SPR 11 国均值 {lvl(nl, C11, 1)}（9 国 {lvl(nl, C9, 1)}），'
      f'簇数 {lvl(ncl, C11, 2)}，其中区内簇 {lvl(nin, C11, 2)}')
    P('     SPR 国家值：' + '  '.join(f'{c} {nl[c]:.1f}' for c in C11))
    put('A-c3', n0, 'all', 'SPR', 'n_labels_S1', nl, snl)
    sq_in, _ = country(sel('clus', n0, 'ae', 'SPR', 'E_S1')['sprq_in'])
    P(f'  SPR= 的 20 个簇中区内簇数 11 国均值 {lvl(sq_in, C11, 2)}')
    for est in ('E_pool', 'E_S1'):
        P(f'  估计量 {est}：')
        X = {arm: sel('clus', n0, 'ae', arm, est) for arm in ('EXT', 'SPR', 'SPR=')}
        L1v = {arm: country(X[arm]['prior_L1']) for arm in X}
        P('    先验 L1 水平（11 国）：' + '，'.join(f'{a} {lvl(L1v[a][0], C11, 3)}' for a in X))
        for arm in X:
            put('A-c3', n0, 'all', f'{est}|{arm}', 'L1', *L1v[arm])
        for a, b in (('SPR', 'EXT'), ('SPR=', 'EXT'), ('SPR', 'SPR=')):
            vd, sd = country(X[a]['prior_L1'] - X[b]['prior_L1'])
            P(f'    先验 L1 {a} − {b}：11 国 {fmt(vd, C11, sd, 3)}；9 国 {fmt(vd, C9, sd, 3)}')
            put('A-c3', n0, 'all', est, f'L1_{a}-{b}', vd, sd)
        for f in FEATS:
            no = REF[(f, 'E_none')].oa.reindex(C11)
            gap = (REF[(f, 'E_oracle')].oa - REF[(f, 'E_none')].oa).reindex(C11)
            X = {arm: sel('clus', n0, f, arm, est) for arm in ('EXT', 'SPR', 'SPR=')}
            O = {arm: country(X[arm]['oa']) for arm in X}
            for arm in X:
                put('A-c3', n0, f, f'{est}|{arm}', 'OA', *O[arm])
            P(f'    [{f}] OA 水平（11 国）：' + '，'.join(
                f'{a} {lvl(O[a][0], C11)}（恢复率 {(O[a][0] - no).mean() / gap.mean():.3f}）' for a in X))
            for a, b in (('SPR', 'EXT'), ('SPR=', 'EXT'), ('SPR', 'SPR=')):
                vd, sd = country(X[a]['oa'] - X[b]['oa'])
                put('A-c3', n0, f, est, f'dOA_{a}-{b}', vd, sd)
                P(f'    [{f}] OA {a} − {b}：11 国 {fmt(vd, C11, sd)}；9 国 {fmt(vd, C9, sd)}')
                P(f'         国家值：{percountry(vd)}')
P()

# ------------------------------------------------------------------ 附：EXT-p 与 none
P('=' * 100)
P('附（非 A-c1–c3 必报）：口径 1 的 EXT-p 与 SRS（clus，k = 200）及 none 型 S0 的 SRS')
for n0 in N0S:
    for est in ('E_pool', 'E_S1'):
        for f in FEATS:
            a = sel('clus', n0, f, 'EXT-p', est)
            b = sel('clus', n0, f, 'SRS', est)
            vd, sd = country(a['oa'] - b['oa'])
            put('extra', n0, f, est, 'dOA_EXT-p-SRS', vd, sd)
            vl, sl = country(a['prior_L1'] - b['prior_L1'])
            P(f'  n0 {n0} {est} [{f}] OA EXT-p − SRS：11 国 {fmt(vd, C11, sd)}；9 国 {fmt(vd, C9, sd)}；'
              f'L1 差 11 国 {fmt(vl, C11, sl, 3)}')
for f in FEATS:
    x = sel('none', 0, f, 'SRS', 'E_S1')
    v, s = country(x['oa'])
    no = REF[(f, 'E_none')].oa.reindex(C11)
    gap = (REF[(f, 'E_oracle')].oa - REF[(f, 'E_none')].oa).reindex(C11)
    put('extra', 0, f, 'none|SRS|E_S1', 'OA', v, s)
    P(f'  none 型 S0，SRS k = 200，E_S1 [{f}]：OA − OA_none 11 国 {fmt(v - no, C11, s)}；'
      f'恢复率 11 国 {(v - no).mean() / gap.mean():.3f}，9 国 {(v - no).reindex(C9).mean() / gap.reindex(C9).mean():.3f}')
    vl, sl = country(x['prior_L1'])
    P(f'       先验 L1 11 国 {lvl(vl, C11, 3)}')

# ------------------------------------------------------------------ 对账与设计诊断
P()
P('=' * 100)
P('对账（P0test_{f}_i300 底座，G 的 foreign_only 行，44 个单元×重复 × 2 特征）')
rc = pd.read_csv(os.path.join(D_OUT, 'indep_recon.csv'))
P(f'  E_none OA 与 G oa 最大绝对差 {np.abs(rc.oa_none_P0 - rc.G_oa).max():.2e}；BA 与 G ba_ge30 最大绝对差 '
  f'{np.abs(rc.ba_none_P0 - rc.G_ba).max():.2e}')
P(f'  用 G 的 3 万候选池先验（pool_positions）重加权 → 与 G oa_P2 最大绝对差 '
  f'{np.abs(rc.oa_poolprior_P0 - rc.G_oa_P2).max():.2e}，ba_P2 {np.abs(rc.ba_poolprior_P0 - rc.G_ba_P2).max():.2e}'
  f'（说明 reweight、π_train、评估函数与 G 逐位一致）')
P(f'  评估函数与 G._scores 最大绝对差 {rc.d_scores.max():.2e}')
for f in FEATS:
    q = rc[rc.feat == f].groupby('target')[['oa_oracle_P0', 'G_oa_P2', 'G_oa']].mean().reindex(C11)
    P(f'  [{f}] P0 底座 E_oracle OA 11 国 {q.oa_oracle_P0.mean():.2f} vs G oa_P2（候选池先验）{q.G_oa_P2.mean():.2f}，'
      f'差 {(q.oa_oracle_P0 - q.G_oa_P2).mean():+.2f}；G oa（无修正）{q.G_oa.mean():.2f}。'
      f'差异来源：P2 用候选池先验而非测试先验，'
      f'L1(π_pool, π_test) 11 国均值 {rc[rc.feat == f].groupby("target").L1_pool_test.mean().reindex(C11).mean():.3f}')
P('  两者不同是预期的：E_oracle 用测试集真实先验，P2 用 3 万候选池先验；上面已用池先验逐位复现 oa_P2。')

P()
P('=' * 100)
P('设计诊断')
un = pd.read_csv(os.path.join(D_OUT, 'indep_units.csv'))
P(f'  KMeans labels_ 与最近中心一致的比例（全部单元）最小 {un.km_label_consistent.min():.3f}')
P(f'  clus 抽样中"所选调查区里没有落入任何 S0 地块"的次数：共 {int(un.draws_with_empty_selected_zone.sum())} / '
  f'{22 * 60} 次（所以"S0 所在区"取所选区还是 S0 实际落入的区几乎没有差别）')
for _, u in un.iterrows():
    P(f'  {u.target} s{u.split}: 候选 {u.n_cand:,}，格 {u.n_cells}（≥10 个 {u.n_cells_ge10}），区规模 {u.zone_sizes}，'
      f'clus 所用区数分布 {u.zones_used_dist}')

SENS = os.path.join(D_OUT, 'sens_square', 'indep_country.csv')
if os.path.exists(SENS) and os.path.abspath(SENS) != os.path.abspath(os.path.join(D_OUT, 'indep_country.csv')):
    P()
    P('=' * 100)
    P('敏感性：调查区 KMeans 用"格内候选地块坐标均值"（主结果）还是"10 km 方格几何中心"（sens_square/）')
    a = pd.DataFrame(CROWS)
    b = pd.read_csv(SENS)
    m = a.merge(b, on=['contrast', 'n0', 'feat', 'est', 'stat', 'country'], suffixes=('_m', '_s'))
    m = m[m.contrast.isin(['A-c1', 'A-c2', 'A-c3']) & m.stat.str.startswith(('dOA', 'L1', 'prior_L1', 'bias'))]
    g = m.groupby(['contrast', 'n0', 'feat', 'est', 'stat'])
    t = pd.DataFrame({'mass': g.value_m.mean(), 'square': g.value_s.mean(),
                      'mcse': np.sqrt(g.apply(lambda x: (x.mc_se_m ** 2 + x.mc_se_s ** 2).sum())) / 11})
    t['d'] = t.square - t.mass
    t['z'] = t.d / t.mcse
    t = t[np.isfinite(t.z)]
    P(f'  11 国均值差 |z|（以两次运行的 MC SE 合成）：中位数 {t.z.abs().median():.2f}，最大 {t.z.abs().max():.2f}；'
      f'|z| > 2 的条目 {int((t.z.abs() > 2).sum())} / {len(t)}')
    for idx, rr in t[t.z.abs() > 1.5].iterrows():
        P(f'    {idx}: mass {rr.mass:+.3f}  square {rr.square:+.3f}  差 {rr.d:+.3f}  z {rr.z:+.2f}')

open(os.path.join(D_OUT, 'indep_summary.txt'), 'w').write('\n'.join(LINES) + '\n')
pd.DataFrame(CROWS).to_csv(os.path.join(D_OUT, 'indep_country.csv'), index=False)
print('\n'.join(LINES))
