"""RSE 故事线配图的源数据整理（只读已有结果，不训练、不改任何冻结文件）。

输出到 rse_figs/source_data/：每张图读这里的 CSV/GeoJSON，图中每个数都能追溯到这里。
- 确认运行（v5 + rev1）的估计量、国家值、X3、A2 比例：原样抽取；
- 示例单元（BE_F1）的 τ 族 OA–BA 曲线：用冻结的 analyze_confirm.unit_curves 重算（只 import）；
- 示例单元的空间布局：测试格、候选地块、调查区、S0 与三种补采（r0）；
- 11 国边界：NUTS 2021 三级区按国家合并；
- 早先的已登记结果：G 实验 θ_E1（四个学习器）、检验 1 选源 ρ、AE 按类/按地块大小的前沿差（C 级）、几何选样方法 − SRS。
用法：~/AlphaCrops_local/.venv/bin/python prep_data.py
"""
import os
import re
import sys

for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(k, '2')

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'source_data')
LE = os.path.dirname(os.path.dirname(HERE))          # label_expansion/
LOCAL = '/Users/perng_mini/AlphaCrops_local'
CONF = os.path.join(LOCAL, 'results/confirm_v1')
ANA = os.path.join(CONF, 'analysis')
EX_UNIT = 'BE_F1'                                    # 示例单元：ΔFA(RT200) 与 D 都接近 9 国均值
COUNTRIES = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']
os.makedirs(OUT, exist_ok=True)


def save(df, name):
    p = os.path.join(OUT, name)
    df.to_csv(p, index=False)
    print(f'  {name:34s} {len(df):6d} 行')


# ------------------------------------------------------------ 1. 确认运行：原样抽取
def confirm_tables():
    e = pd.read_csv(os.path.join(ANA, 'estimates.csv'))
    save(e[['est_id', 'family', 'learner', 'quantity', 'feat', 'set', 'primary', 'n', 'mean', 'lo95', 'hi95',
            'lo90', 'hi90', 'lo975', 'hi975', 'n_pos', 'p_one']], 'confirm_estimates.csv')
    save(pd.read_csv(os.path.join(ANA, 'country_values.csv')), 'confirm_country_values.csv')
    save(pd.read_csv(os.path.join(ANA, 'x3_equiv.csv')), 'confirm_x3_equiv.csv')
    save(pd.read_csv(os.path.join(ANA, 'x3_cost.csv')), 'confirm_x3_cost.csv')
    save(pd.read_csv(os.path.join(ANA, 'a2_ratio.csv')), 'confirm_a2_ratio.csv')
    save(pd.read_csv(os.path.join(ANA, 'x2_position.csv')), 'confirm_x2_position.csv')
    save(pd.read_csv(os.path.join(ANA, 'tests.csv')), 'confirm_tests.csv')


# ------------------------------------------------------------ 2. 示例单元的 τ 曲线（重算，只 import 冻结代码）
def example_curves(G):
    sys.path.insert(0, os.path.join(LE, 'confirm'))
    cwd = os.getcwd()
    import analyze_confirm as A   # noqa: E402  导入链会 chdir；只调用其函数，不写任何文件
    os.chdir(cwd)
    s = np.load(os.path.join(CONF, 'splits', f'{EX_UNIT}.npz'))
    test, big = s['test'], s['big']
    yT = np.asarray(G.y)[test]
    big = big[big >= 0] if big.dtype.kind == 'i' else big
    P = {}
    for arm, rs in (('Z0', [0]), ('RT200', [0, 1, 2]), ('RT400', [0, 1, 2]), ('TGT', [0])):
        for f in ('ae', 's2'):
            for r in rs:
                P[('lgb', f, arm, r)] = np.load(os.path.join(CONF, 'probs', EX_UNIT, f'lgb_{arm}_{f}_r{r}.npz'))['P_test']
    CV = A.unit_curves(yT, big, P)
    rows = []
    for (lrn, f, arm, r), (oa, ba) in CV.items():
        for t, o, b in zip(A.TAUS, oa, ba):
            rows.append(dict(unit=EX_UNIT, learner=lrn, feat=f, arm=arm, r=r, tau=float(t), OA=float(o), BA=float(b)))
    df = pd.DataFrame(rows)
    save(df, 'example_unit_curves.csv')
    # 与分析输出对账：τ = 0 的 OA/BA 与 unit_rows 一致
    u = pd.read_csv(os.path.join(ANA, 'unit_rows.csv'))
    u = u[(u.unit == EX_UNIT) & (u.level == 'arm') & (u.learner == 'lgb')]
    t0 = df[df.tau == 0.0]
    m = t0.merge(u[['feat', 'name', 'r', 'oa_t0', 'ba_t0', 'value', 'OA_star']], left_on=['feat', 'arm', 'r'],
                 right_on=['feat', 'name', 'r'])
    err = max((m.OA - m.oa_t0).abs().max(), (m.BA - m.ba_t0).abs().max())
    assert len(m) == len(t0) and err < 1e-9, ('示例单元曲线与 unit_rows 不一致', len(m), len(t0), err)
    print(f'  示例单元 τ=0 对账：{len(m)} 条曲线，最大 |Δ| {err:.1e}')
    save(m[['feat', 'arm', 'r', 'OA_star', 'value']].rename(columns={'value': 'FA'}), 'example_unit_fa.csv')
    # 示例单元的逐单元差值与 D（分析输出原样抽取，供图中标注）
    u = pd.read_csv(os.path.join(ANA, 'unit_rows.csv'))
    u = u[(u.unit == EX_UNIT) & (u.learner == 'lgb') & (u.level.isin(['diff', 'D', 't0']))]
    save(u[['level', 'name', 'feat', 'r', 'OA_star', 'value', 'cov']], 'example_unit_diffs.csv')


# ------------------------------------------------------------ 3. 示例单元的空间布局
def example_layout(G):
    s = np.load(os.path.join(CONF, 'splits', f'{EX_UNIT}.npz'))
    z = np.load(os.path.join(CONF, 'selection', f'{EX_UNIT}.npz'))
    lon, lat = np.asarray(G.lon), np.asarray(G.lat)
    lat0 = float(s['lat0'])

    def xy(idx):
        return lon[idx] * 111.32 * np.cos(np.radians(lat0)), lat[idx] * 110.57   # 与 stageA.km_xy 相同

    rng = np.random.default_rng(0)
    rows = []

    def add(idx, role, n_max=None):
        idx = np.asarray(idx)
        if n_max is not None and len(idx) > n_max:
            idx = np.sort(rng.choice(idx, n_max, replace=False))
        x, y = xy(idx)
        cell = np.c_[np.floor(x / 10), np.floor(y / 10)].astype(int)
        rows.extend(dict(role=role, x_km=a, y_km=b, cx=c[0], cy=c[1]) for a, b, c in zip(x, y, cell))

    add(s['test'], 'test', 4000)
    add(np.setdiff1d(s['cand'], np.r_[z['S0_r0'], z['S1_SRS_k300_r0'], z['S1_EXT_k300_r0'], z['S1_BA_k300_r0']]),
        'candidate', 12000)
    add(z['S0_r0'], 'S0')
    for st in ('SRS', 'EXT', 'BA'):
        add(z[f'S1_{st}_k100_r0'], f'{st}100')
    df = pd.DataFrame(rows)
    save(df, 'example_unit_layout.csv')
    cells = pd.DataFrame(dict(cx=s['cell_cx'], cy=s['cell_cy'], n_cand=s['cell_cnt'], region=s['cell_reg']))
    cells['S0_region'] = cells.region.isin(z['S0reg_r0'])
    cells['S0_cell'] = False
    cells.loc[z['S0cell_r0'], 'S0_cell'] = True
    save(cells, 'example_unit_cells.csv')
    save(pd.DataFrame(s['test_cells'], columns=['cx', 'cy']), 'example_unit_test_cells.csv')
    save(pd.DataFrame(dict(region=np.arange(8), gamma=z['BA_gamma_r0'], n0=z['BA_n0_r0'],
                           n_BA100=z['BA_n_k100_r0'], n_BA300=z['BA_n_k300_r0'])), 'example_unit_ba_alloc.csv')


# ------------------------------------------------------------ 4. 国家边界与样本量
def boundaries_and_counts(G):
    import geopandas as gpd
    g = gpd.read_file(os.path.join(LOCAL, 'data/boundaries/NUTS_RG_01M_2021_4326_LEVL_3.geojson'))
    g = g[~g.NUTS_ID.str.match(r'^(FRY|PT2|PT3|ES7)')]       # 去掉远离欧洲本土的海外领地与群岛
    c = g.dissolve(by='CNTR_CODE', as_index=False)[['CNTR_CODE', 'geometry']]
    c['geometry'] = c.geometry.simplify(0.01)
    c['study'] = c.CNTR_CODE.isin(COUNTRIES)
    c.to_file(os.path.join(OUT, 'countries.geojson'), driver='GeoJSON')
    print(f'  countries.geojson                  {len(c):6d} 国')
    # 德国只用了下萨克森（NUTS1 DE9）与北莱茵-威斯特法伦（DEA）两州：单独保存，供地图标出实际研究区
    de = g[g.NUTS_ID.str.match(r'^(DE9|DEA)')].copy()
    de['land'] = de.NUTS_ID.str[:3]
    de = de.dissolve(by='land', as_index=False)[['land', 'geometry']]
    de['geometry'] = de.geometry.simplify(0.01)
    de.to_file(os.path.join(OUT, 'study_regions_de.geojson'), driver='GeoJSON')
    print(f'  study_regions_de.geojson           {len(de):6d} 州')
    # 罗马尼亚的标签只覆盖南部边境地带：取含有 RO 地块的 NUTS3 区并合并，供地图只标出实际覆盖范围
    ro3 = g[g.CNTR_CODE == 'RO'][['NUTS_ID', 'geometry']]
    k = np.asarray(G.ctry) == 'RO'
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(np.asarray(G.lon)[k], np.asarray(G.lat)[k]), crs=4326)
    hit = gpd.sjoin(pts, ro3, predicate='within', how='inner').NUTS_ID.value_counts()
    ro = ro3[ro3.NUTS_ID.isin(hit[hit >= 50].index)].dissolve()[['geometry']]
    ro['geometry'] = ro.geometry.simplify(0.01)
    ro.to_file(os.path.join(OUT, 'study_regions_ro.geojson'), driver='GeoJSON')
    print(f'  study_regions_ro.geojson           {int((hit >= 50).sum()):6d} 个 NUTS3 区，覆盖 {hit[hit >= 50].sum() / k.sum():.1%} 的 RO 地块')
    ctry = np.asarray(G.ctry)
    rows = []
    for cc in COUNTRIES:
        n_test = sum(int(np.load(os.path.join(CONF, 'splits', f'{cc}_{h}.npz'))['n_test']) for h in ('F1', 'F2'))
        rows.append(dict(country=cc, n_parcels=int((ctry == cc).sum()), n_test_F1F2=n_test,
                         primary9=cc not in ('RO', 'PT')))
    save(pd.DataFrame(rows), 'country_counts.csv')
    save(pd.DataFrame(dict(cls=np.arange(len(G.classes)), name=G.classes)), 'classes.csv')


# ------------------------------------------------------------ 5. 早先的已登记结果
def earlier_results():
    R = os.path.join(LOCAL, 'results')
    s = pd.read_csv(os.path.join(R, 'guidance_G_ml/summary_ML.csv'))
    save(s[s.what == 'theta_E1'][['learner', 'metric', 'n', 'mean', 'lo90', 'hi90', 'lo95', 'hi95', 'tost_p', 'verdict']],
         'guidance_theta.csv')

    # 检验 1：报告中的目标国平均 ρ 与 bootstrap CI（原样解析）；逐目标国值另算
    txt = open(os.path.join(R, 'check1_report.txt'), encoding='utf-8').read()
    blk = txt.split('三种分类特征平均后')[1].split('===')[0]
    rows = [dict(descriptor=m[0], rho=float(m[1]), lo95=float(m[2]), hi95=float(m[3]))
            for m in re.findall(r'^\s+(\w+)\s+([+-][\d.]+)\s+\[([+-][\d.]+),\s*([+-][\d.]+)\]', blk, re.M)]
    assert len(rows) == 7, rows
    save(pd.DataFrame(rows), 'source_selection_rho.csv')
    pt = pd.read_csv(os.path.join(R, 'check1_per_target.csv'))
    save(pt.groupby(['target', 'descriptor']).rho.mean().reset_index(), 'source_selection_rho_per_target.csv')
    # AE-MMD − S2-MMD 的 Δρ（三种分类特征平均）与 bootstrap CI，原样解析
    m = re.search(r'分类特征=avg\s+Δρ\(AE−S2\)=([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]\s+AE 更好的目标国 (\d+)/(\d+)', txt)
    save(pd.DataFrame([dict(contrast='mmd_ae-mmd_s2', d_rho=float(m[1]), lo95=float(m[2]), hi95=float(m[3]),
                            n_ae_better=int(m[4]), n=int(m[5]))]), 'source_selection_delta.csv')

    # AE 按类、按地块大小（C 级；原样解析 ae_frontier_summary.txt 中的国家层汇总）
    sm = open(os.path.join(LOCAL, 'verify/preconf/ae/ae_frontier_summary.txt'), encoding='utf-8').read()
    ae3 = sm.split('######## AE3')[1].split('######## AE4')[0]
    num = r'([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]，>0 (\d+)/(\d+)'
    rows = []
    for m in re.finditer(r'\[(\d)\] (\w+)（.*?\n\s+同 OA 点 AE−S2\s+9 国 ' + num + r' \| 11 国 ' + num, ae3):
        g = m.groups()
        rows.append(dict(cls=g[1], d9=float(g[2]), lo9=float(g[3]), hi9=float(g[4]), pos9=int(g[5]), n9=int(g[6]),
                         d11=float(g[7]), lo11=float(g[8]), hi11=float(g[9]), pos11=int(g[10]), n11=int(g[11])))
    assert len(rows) == 10, len(rows)
    save(pd.DataFrame(rows), 'ae_by_class.csv')
    ae4 = sm.split('######## AE4')[1].split('######## 读法')[0]
    rows = []
    for m in re.finditer(r'-- 组 (\S+)（.*?BA 同OA点\s+AE−S2 9 国 ' + num + r' \| 11 国 ' + num, ae4, re.S):
        g = m.groups()
        rows.append(dict(size_group=g[0], d9=float(g[1]), lo9=float(g[2]), hi9=float(g[3]), pos9=int(g[4]),
                         n9=int(g[5]), d11=float(g[6]), lo11=float(g[7]), hi11=float(g[8]), pos11=int(g[9]),
                         n11=int(g[10])))
    shares = dict(re.findall(r'^\s+(\S+)\s+占比 11 国 ([\d.]+)', ae4, re.M))
    for r in rows:
        r['share11'] = float(shares[r['size_group']])
    assert len(rows) == 5, rows
    save(pd.DataFrame(rows), 'ae_by_size.csv')

    # 几何/不确定性选样方法 − SRS（G 实验，ba_ge30，2 分类器 × 2 k 合并；原样解析）
    lg = open(os.path.join(LOCAL, 'verify/strategy/benchmark/a_vs_srs.log'), encoding='utf-8').read()
    blk = lg.split('ba_ge30')[1].split('method cond')[0]
    rows = []
    for m in re.finditer(r'^(\w+)\s+' + r'\s+'.join([r'([+-]?[\d.]+)'] * 12) + r'\s*$', blk, re.M):
        g = m.groups()
        v = list(map(float, g[1:]))
        for j, cond in enumerate(('AE', 'AE+S2', 'S2')):
            rows.append(dict(method=g[0], guidance=cond, hi95=v[j], lo95=v[3 + j], mean=v[6 + j], n_pos=int(v[9 + j])))
    assert len(rows) == 24, len(rows)
    save(pd.DataFrame(rows), 'strategy_vs_srs.csv')


def traceability():
    """图例里引用、但此前不在 source_data 中的设计量：各国零标签 OA★、TGT 训练池大小。"""
    u = pd.read_csv(os.path.join(ANA, 'unit_rows.csv'))
    z = u[(u.level == 'arm') & (u.name == 'Z0')].groupby(['learner', 'country', 'feat']).OA_star.mean().reset_index()
    save(z, 'country_oastar_z0.csv')
    rows = []
    for cc in COUNTRIES:
        for h in ('F1', 'F2'):
            s = np.load(os.path.join(CONF, 'splits', f'{cc}_{h}.npz'))
            rows.append(dict(unit=f'{cc}_{h}', n_tgt_pool=len(s['pool']), n_candidates=len(s['cand']),
                             n_test=int(s['n_test']), n_foreign=len(s['foreign'])))
    save(pd.DataFrame(rows), 'unit_sizes.csv')


def main():
    print('== 确认运行表')
    confirm_tables()
    traceability()
    print('== 早先结果')
    earlier_results()
    sys.path.insert(0, LE)
    from guidance_G import GData  # noqa: E402
    G = GData()
    print('== 边界与样本量')
    boundaries_and_counts(G)
    print('== 示例单元布局')
    example_layout(G)
    print('== 示例单元曲线')
    example_curves(G)


if __name__ == '__main__':
    main()
