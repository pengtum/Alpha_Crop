"""
三个任务（作物 crop、城市形态 lcz、土地覆盖 lucas）的统一分析流程：同一份代码从各任务的测试集预测概率出发，
算出全部逐单元量、域值与任务层估计，写到 unified_figs/source_data/。只读已有结果，不训练。

口径与 RSE 确认分析（label_expansion/confirm/analyze_confirm.py，冻结）相同，度量函数直接 import：
  曲线      AC.unit_curves：τ = −2..2（81 点），ℓ = log P − τ·log π̂；OA、BA（BA 类别 = 各任务的登记口径）
  窗口      OA★ = 同一特征 Z0 在 τ = 0 的 OA；OA★ + [−4, 0]，步长 0.1；包络 step0B.envelope；差值 AC.paired
  dFA_RTk   同一特征 RT_k − Z0（每次抽样）；dFA_REF：REF − Z0（作物 REF = 只用目标国训练 TGT；试点 REF = 全部候选）
  D_arm     AE − S2，锚点 min(OA★_ae, OA★_s2)（Z0）；arm ∈ Z0、RT200、RT400
  t0_*      τ = 0 读数：RT200 − Z0 的 BA/OA（每个特征）；Z0 的 AE − S2 BA/OA
  按类      Z0；同 OA 点 o_mid = 锚点 − 2；两特征各自在包络上取 OA = o_mid 的段与插值系数，逐类召回同系数插值；
            只计该单元的 BA 类别（与 RSE 事后分析 AE3 相同）
  按大小    作物与 LUCAS：面积 ÷ (π·30² m²) 分组 [0,0.5)…[4,∞)；组内 BA 只计组内 ≥ 30 个样本的类，< 2 类记缺失（同 AE4）
聚合（与 analyze_confirm.country_table 相同）：单元内对抽样平均（跳过缺失）→ 域内对单元平均（作物两个半区）→
  both = AE、S2 域值平均（任一缺失则缺失）；inter = AE − S2。任务层：主集域上平均，95% t CI（交互另给 90%），>0 的域数。
对账：作物域值与已登记的 confirm_v1/analysis/country_values.csv、试点域值与 nc_pilot/runs/analysis/domain_values.csv
逐项比对，|差| < 1e-9，否则中止。

用法：~/AlphaCrops_local/.venv/bin/python -B prep_unified.py
"""
import json
import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'source_data')
AC_ROOT = os.path.dirname(HERE)                                  # Alpha_Crops/
LE = os.path.join(AC_ROOT, 'label_expansion')
NCP = os.path.join(AC_ROOT, 'nc_pilot')
LOCAL = '/Users/perng_mini/AlphaCrops_local'
CONF = os.path.join(LOCAL, 'results/confirm_v1')
NCD = os.path.join(LOCAL, 'nc_pilot')
RSE_SRC = os.path.join(LE, 'storyline/rse_figs/source_data')
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.join(LE, 'confirm'))
_cwd = os.getcwd()
import analyze_confirm as AC  # noqa: E402  导入链会 chdir 并核对冻结文件哈希
os.chdir(_cwd)
B, B2 = AC.B, AC.B2
sys.path.insert(0, NCP)
import nc_common as NK  # noqa: E402

FEATS = ('ae', 's2')
KS = (50, 100, 200, 400)
R = 3
CORE_M2 = np.pi * 30 ** 2
SIZE_CUTS = np.array([0.5, 1.0, 2.0, 4.0])
SIZE_LAB = ['<0.5', '0.5–1', '1–2', '2–4', '≥4']
MIN_GRP = 30

LCZ_NAMES = ['LCZ 1 compact high-rise', 'LCZ 2 compact mid-rise', 'LCZ 3 compact low-rise', 'LCZ 4 open high-rise',
             'LCZ 5 open mid-rise', 'LCZ 6 open low-rise', 'LCZ 7 lightweight low-rise', 'LCZ 8 large low-rise',
             'LCZ 9 sparsely built', 'LCZ 10 heavy industry', 'LCZ A dense trees', 'LCZ B scattered trees',
             'LCZ C bush, scrub', 'LCZ D low plants', 'LCZ E bare rock or paved', 'LCZ F bare soil or sand',
             'LCZ G water']
LUCAS_NAMES = ['artificial land', 'cropland', 'woodland', 'shrubland', 'grassland', 'bare land', 'wetland']


def save(df, name):
    df.to_csv(os.path.join(OUT, name), index=False)
    print(f'  {name:36s} {len(df):7d} 行')


# ================================================================ 任务接口：每个单元给出测试标签、BA 类别、概率与面积
def crop_units():
    sys.path.insert(0, LE)
    cwd = os.getcwd()
    os.chdir(LE)
    from guidance_G import GData  # noqa: E402
    G = GData()
    os.chdir(cwd)
    y, area = np.asarray(G.y), np.asarray(G.area)
    for cc in AC.COUNTRIES:
        for h in ('F1', 'F2'):
            u = f'{cc}_{h}'
            s = np.load(os.path.join(CONF, 'splits', f'{u}.npz'))
            big = s['big']
            big = big[big >= 0] if big.dtype.kind == 'i' else np.flatnonzero(big)
            P = {}
            for f in FEATS:
                for arm, rs, src in [('Z0', [0], 'Z0')] + [(f'RT{k}', range(R), f'RT{k}') for k in KS] + [('REF', [0], 'TGT')]:
                    for r in rs:
                        P[(f, arm, r)] = np.load(os.path.join(CONF, 'probs', u, f'lgb_{src}_{f}_r{r}.npz'))['P_test']
            yield dict(unit=u, domain=cc, primary=cc not in AC.DROP, yT=y[s['test']], big=[int(c) for c in big], P=P,
                       area_m2=area[s['test']] * 1e4, classes=list(G.classes))


def pilot_units(task):
    T = {'lcz': 'T1', 'lucas': 'T2'}[task]
    idc = {'T1': 'row', 'T2': 'id'}[T]
    dom = {'T1': 'city', 'T2': 'nuts0'}[T]
    S = pd.read_parquet(os.path.join(NCD, 'splits', f'{T.lower()}_samples.parquet'))
    removed = pd.read_csv(os.path.join(NCD, 'runs', 'removed.csv'))
    S = S[~S[idc].isin(set(removed.loc[removed.task == T, idc].astype(np.int64)))]
    ba = json.load(open(os.path.join(NCD, 'splits', 'ba_classes.json')))
    targets = json.load(open(os.path.join(NCD, 'splits', 'targets.json')))[T]
    meta = pd.read_parquet(os.path.join(NCD, 'splits', 'lucas_meta_raw.parquet')).set_index('id') if T == 'T2' else None
    for d in targets:
        pdir = os.path.join(NCD, 'runs', 'probs', T, d)
        P, sid = {}, None
        for f in FEATS:
            for arm, rs, src in [('Z0', [0], 'Z0')] + [(f'RT{k}', range(R), f'RT{k}') for k in KS] + [('REF', [0], 'ALL')]:
                for r in rs:
                    z = np.load(os.path.join(pdir, f'{f}_{src}_r{r}.npz'))
                    sid = z['sid_test'] if sid is None else sid
                    assert np.array_equal(sid, z['sid_test'])
                    P[(f, arm, r)] = z['P_test']
        te = S.set_index(idc).loc[sid]
        area = meta.loc[sid, 'area'].values.astype(float) if meta is not None else None
        yield dict(unit=d, domain=d, primary=True, yT=te.y.values.astype(int), big=list(ba[d]), P=P, area_m2=area,
                   classes=LCZ_NAMES if T == 'T1' else LUCAS_NAMES)


# ================================================================ 逐单元计算（三个任务同一份代码）
def interp(v, s_, l_):
    return (1 - l_) * v[s_] + l_ * v[s_ + 1]


def unit_rows(U):
    yT, big, P = U['yT'], U['big'], U['P']
    CV = AC.unit_curves(yT, big, P)
    star = {f: CV[(f, 'Z0', 0)][0][AC.I_T0] for f in FEATS}
    base = dict(unit=U['unit'], domain=U['domain'])
    rows, env = [], {}
    arms = [('Z0', 0)] + [(f'RT{k}', r) for k in KS for r in range(R)] + [('REF', 0)]
    for f in FEATS:
        for arm, r in arms:
            env[(f, arm, r)] = B.envelope([CV[(f, arm, r)]], star[f] + AC.OFFS)[0]
        for arm, r in arms[1:]:
            v, cov = AC.paired(env[(f, arm, r)], env[(f, 'Z0', 0)])
            rows.append(dict(base, name=f'dFA_{arm}', feat=f, r=r, value=v, cov=cov))
        z0 = CV[(f, 'Z0', 0)]
        for r in range(R):
            rc = CV[(f, 'RT200', r)]
            for j, met in ((1, 'BA'), (0, 'OA')):
                rows.append(dict(base, name=f't0_RT200-Z0_{met}', feat=f, r=r, value=rc[j][AC.I_T0] - z0[j][AC.I_T0]))
    anc = min(star.values())
    for arm, rs in (('Z0', [0]), ('RT200', range(R)), ('RT400', range(R))):
        for r in rs:
            ea = B.envelope([CV[('ae', arm, r)]], anc + AC.OFFS)[0]
            es = B.envelope([CV[('s2', arm, r)]], anc + AC.OFFS)[0]
            v, cov = AC.paired(ea, es)
            rows.append(dict(base, name=f'D_{arm}', feat='ae-s2', r=r, value=v, cov=cov))
    ca, cs = CV[('ae', 'Z0', 0)], CV[('s2', 'Z0', 0)]
    for j, met in ((1, 'BA'), (0, 'OA')):
        rows.append(dict(base, name=f't0_Z0_AE-S2_{met}', feat='ae-s2', r=0, value=ca[j][AC.I_T0] - cs[j][AC.I_T0]))
    # ---- 按类与按大小（Z0，同 OA 点 o_mid = 锚点 − 2）
    o_mid = anc - 2.0
    det = {}
    for f in FEATS:
        Z = B.log_or_neginf(B.norm64(P[(f, 'Z0', 0)]))
        PH = B2.tau_preds_grid(Z, AC.pihat_of(Z), AC.TAUS)
        corr = PH == yT[None, :]
        best, _, seg, lam = B.envelope([CV[(f, 'Z0', 0)]], np.array([o_mid]))
        s_, l_ = int(seg[0]), float(lam[0])
        rec = {c: corr[:, yT == c].mean(1) for c in big}
        if s_ >= 0:
            ba_chk = 100.0 * np.mean([interp(rec[c], s_, l_) for c in big])
            assert abs(ba_chk - best[0]) < 1e-6, (U['unit'], f, ba_chk, best[0])
        det[f] = dict(corr=corr, s=s_, l=l_, rec=rec)
    cls_rows = []
    for c in big:
        r_ = dict(base, cls=int(c), cls_name=U['classes'][c], n_test=int((yT == c).sum()))
        for f in FEATS:
            d = det[f]
            r_[f'rec_mid_{f}'] = 100.0 * interp(d['rec'][c], d['s'], d['l']) if d['s'] >= 0 else np.nan
            r_[f'rec_t0_{f}'] = 100.0 * d['rec'][c][AC.I_T0]
        r_['d_mid'] = r_['rec_mid_ae'] - r_['rec_mid_s2']
        r_['d_t0'] = r_['rec_t0_ae'] - r_['rec_t0_s2']
        cls_rows.append(r_)
    size_rows = []
    if U['area_m2'] is not None:
        grp = np.digitize(U['area_m2'] / CORE_M2, SIZE_CUTS)
        for j in range(len(SIZE_LAB)):
            mj = grp == j
            ok = [c for c in np.unique(yT) if (mj & (yT == c)).sum() >= MIN_GRP]
            r_ = dict(base, grp=j, grp_lab=SIZE_LAB[j], n_grp=int(mj.sum()), share=float(mj.mean()), n_cls=len(ok))
            for f in FEATS:
                d = det[f]
                if len(ok) >= 2 and d['s'] >= 0:
                    r_[f'ba_mid_{f}'] = 100.0 * float(np.mean([interp(d['corr'][:, mj & (yT == c)].mean(1), d['s'], d['l'])
                                                              for c in ok]))
                else:
                    r_[f'ba_mid_{f}'] = np.nan
            r_['d_mid'] = r_['ba_mid_ae'] - r_['ba_mid_s2']
            size_rows.append(r_)
    arm_rows = [dict(base, feat=f, arm=arm, r=r, OA_star=star[f], oa_t0=CV[(f, arm, r)][0][AC.I_T0],
                     ba_t0=CV[(f, arm, r)][1][AC.I_T0], reach=float(np.isfinite(env[(f, arm, r)]).mean()))
                for f in FEATS for arm, r in arms]
    return rows, cls_rows, size_rows, arm_rows, CV, star


# ================================================================ 聚合
def domain_table(U):
    """单元 → 域：对抽样平均（跳过缺失）→ 对单元平均（跳过缺失）；both、inter 按特征组合。"""
    x = U.groupby(['domain', 'unit', 'name', 'feat']).value.mean().reset_index()
    d = x.groupby(['domain', 'name', 'feat']).value.mean().reset_index()
    fe = d[d.feat.isin(FEATS)].set_index(['domain', 'name', 'feat']).value.unstack('feat')
    both = fe[list(FEATS)].mean(axis=1, skipna=False).rename('value').reset_index().assign(feat='both')
    inter = (fe['ae'] - fe['s2']).rename('value').reset_index().assign(feat='inter')
    return pd.concat([d, both, inter], ignore_index=True)


def tci(v, level):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 2:
        return n, (float(v.mean()) if n else np.nan), np.nan, np.nan, int((v > 0).sum())
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(n)
    q = stats.t.ppf(0.5 + level / 2, n - 1)
    return n, float(m), float(m - q * se), float(m + q * se), int((v > 0).sum())


def estimates(D, task, primary):
    rows = []
    for (name, feat), g in D.groupby(['name', 'feat']):
        for setname, doms in (('primary', primary), ('all', sorted(D.domain.unique()))):
            v = g.set_index('domain').value.reindex(doms)
            n, m, lo, hi, npos = tci(v, 0.95)
            _, _, lo90, hi90, _ = tci(v, 0.90)
            rows.append(dict(task=task, name=name, feat=feat, set=setname, n=n, mean=m, lo95=lo, hi95=hi,
                             lo90=lo90, hi90=hi90, n_pos=npos))
    return pd.DataFrame(rows)


def group_summary(df, key, task, primary, val='d_mid'):
    rows = []
    dom = df.groupby(['domain', key])[val].mean().reset_index()
    for k, g in dom.groupby(key):
        v = g.set_index('domain')[val].reindex(primary)
        n, m, lo, hi, npos = tci(v, 0.95)
        rows.append(dict(task=task, **{key: k}, n=n, mean=m, lo95=lo, hi95=hi, n_pos=npos))
    return pd.DataFrame(rows)


# ================================================================ 对账
def reconcile(task, D):
    if task == 'crop':
        ref = pd.read_csv(os.path.join(CONF, 'analysis', 'country_values.csv'))
        ref = ref[ref.learner == 'lgb'].rename(columns={'country': 'domain'})
        mine = D.copy()
        mine['name'] = mine.name.replace({'dFA_REF': 'dFA_TGT'})
        m = mine.merge(ref[['name', 'feat', 'domain', 'value']], on=['name', 'feat', 'domain'], suffixes=('', '_ref'))
        names = {'dFA_RT50', 'dFA_RT100', 'dFA_RT200', 'dFA_RT400', 'dFA_TGT', 'D_Z0', 'D_RT200',
                 't0_RT200-Z0_BA', 't0_RT200-Z0_OA', 't0_Z0_AE-S2_BA', 't0_Z0_AE-S2_OA'}
        assert names <= set(m.name), names - set(m.name)
    else:
        T = {'lcz': 'T1', 'lucas': 'T2'}[task]
        ref = pd.read_csv(os.path.join(NCD, 'runs', 'analysis', 'domain_values.csv'))
        ref = ref[ref.task == T]
        long = []
        for _, r in ref.iterrows():
            long.append(dict(name='D_Z0', feat='ae-s2', domain=r.dom, value_ref=r.dR))
            for k in list(KS) + ['all']:
                nm = f'dFA_RT{k}' if k != 'all' else 'dFA_REF'
                long.append(dict(name=nm, feat='both', domain=r.dom, value_ref=r[f'dL_{k}']))
                long.append(dict(name=nm, feat='inter', domain=r.dom, value_ref=r[f'I_{k}']))
                for f in FEATS:
                    long.append(dict(name=nm, feat=f, domain=r.dom, value_ref=r[f'dL_{k}_{f}']))
        m = D.merge(pd.DataFrame(long), on=['name', 'feat', 'domain'])
    a, b = m.value.values.astype(float), m.value_ref.values.astype(float)
    assert np.array_equal(np.isnan(a), np.isnan(b)), f'{task}: 缺失位置与登记值不一致'
    err = float(np.nanmax(np.abs(a - b))) if np.isfinite(a).any() else 0.0
    assert err < 1e-9, f'{task}: 与登记值最大差 {err}'
    print(f'  对账通过：{task} {len(m)} 个域值与登记结果一致（最大 |差| {err:.1e}）')


# ================================================================ 地图、示例单元
def example_domain(D, primary):
    """示例域：dFA_RT200（both）与 D_Z0 都最接近主集均值（按主集标准差标准化后的距离最小）。"""
    a = D[(D.name == 'dFA_RT200') & (D.feat == 'both')].set_index('domain').value.reindex(primary)
    b = D[(D.name == 'D_Z0')].set_index('domain').value.reindex(primary)
    z = ((a - a.mean()) / a.std(ddof=1)) ** 2 + ((b - b.mean()) / b.std(ddof=1)) ** 2
    return z.idxmin()


def curves_long(task, unit, CV):
    rows = []
    for (f, arm, r), (oa, ba) in CV.items():
        if arm in ('Z0', 'RT200', 'RT400', 'REF') and r == 0:
            for t, o, b_ in zip(AC.TAUS, oa, ba):
                rows.append(dict(task=task, unit=unit, feat=f, arm=arm, r=r, tau=float(t), OA=float(o), BA=float(b_)))
    return rows


def layout_crop(unit):
    sys.path.insert(0, LE)
    cwd = os.getcwd(); os.chdir(LE)
    from guidance_G import GData  # noqa: E402
    G = GData(); os.chdir(cwd)
    s = np.load(os.path.join(CONF, 'splits', f'{unit}.npz'))
    z = np.load(os.path.join(CONF, 'selection', f'{unit}.npz'))
    lon, lat, lat0 = np.asarray(G.lon), np.asarray(G.lat), float(s['lat0'])
    rng = np.random.default_rng(0)
    rt = s['pool'][z['pi_r0'][:200]]
    rows = []
    for idx, role, nmax in ((s['test'], 'test', 4000), (np.setdiff1d(s['cand'], rt), 'candidate', 12000), (rt, 'RT200', None)):
        idx = np.asarray(idx)
        if nmax and len(idx) > nmax:
            idx = np.sort(rng.choice(idx, nmax, replace=False))
        x, y = lon[idx] * 111.32 * np.cos(np.radians(lat0)), lat[idx] * 110.57      # 与 stageA.km_xy 相同
        rows += [dict(role=role, x_km=a, y_km=b) for a, b in zip(x, y)]
    cells = pd.DataFrame(s['test_cells'], columns=['cx', 'cy']).assign(size_km=10.0, off_km=0.0)
    return pd.DataFrame(rows), cells


def layout_pilot(task, d):
    T = {'lcz': 'T1', 'lucas': 'T2'}[task]
    idc, dom = {'T1': ('row', 'city'), 'T2': ('id', 'nuts0')}[T]
    S = pd.read_parquet(os.path.join(NCD, 'splits', f'{T.lower()}_samples.parquet'))
    removed = pd.read_csv(os.path.join(NCD, 'runs', 'removed.csv'))
    rm = set(removed.loc[removed.task == T, idc].astype(np.int64))
    S = S[(S[dom] == d) & ~S[idc].isin(rm)]
    o = np.load(os.path.join(NCD, 'splits', 'orders.npz'))[f'{d}_r0']
    rt = o[np.isin(o, S[idc].values)][:200]
    S = S.assign(role=np.where(S[idc].isin(rt), 'RT200', np.where(S.role == 'cand', 'candidate', S.role)))
    df = pd.DataFrame(dict(role=S.role.values, x_km=S.x.values / 1000, y_km=S.y_m.values / 1000))
    grid, off = (NK.T1_GRID, NK.T1_GRID_OFF) if T == 'T1' else (NK.T2_GRID, NK.T2_GRID_OFF)
    te = S[S.role == 'test']
    xr, yr = np.round(te.x.values), np.round(te.y_m.values)
    cells = pd.DataFrame(sorted(set(zip(np.floor((xr - off) / grid).astype(int), np.floor((yr - off) / grid).astype(int)))),
                         columns=['cx', 'cy']).assign(size_km=grid / 1000, off_km=off / 1000)
    return df, cells


NAME = {'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 'CY': 'Cyprus', 'CZ': 'Czechia', 'DE': 'Germany',
        'DK': 'Denmark', 'EE': 'Estonia', 'EL': 'Greece', 'ES': 'Spain', 'FI': 'Finland', 'FR': 'France', 'HR': 'Croatia',
        'HU': 'Hungary', 'IE': 'Ireland', 'IT': 'Italy', 'LT': 'Lithuania', 'LU': 'Luxembourg', 'LV': 'Latvia',
        'MT': 'Malta', 'NL': 'Netherlands', 'PL': 'Poland', 'PT': 'Portugal', 'RO': 'Romania', 'SE': 'Sweden',
        'SI': 'Slovenia', 'SK': 'Slovakia', 'UK': 'United Kingdom'}


def maps():
    """底图 world_lsib.geojson（GEE USDOS/LSIB_SIMPLE/2017，3 km 简化，三个任务共用）；国家按名称匹配。"""
    import shutil
    assert os.path.exists(os.path.join(OUT, 'world_lsib.geojson')), '先取回 world_lsib.geojson'
    tg = json.load(open(os.path.join(NCD, 'splits', 'targets.json')))
    # LCZ 城市点：可用 32 城（中位坐标），角色 target / source
    t1 = pd.read_parquet(os.path.join(NCD, 'splits', 't1_samples.parquet'))
    c = t1.groupby('city').agg(lon=('lon', 'median'), lat=('lat', 'median'), n=('row', 'size')).reset_index()
    c['role'] = np.where(c.city.isin(tg['T1']), 'target', 'source')
    save(c, 'lcz_domains.csv')
    # LUCAS 国家：多边形数、角色
    t2 = pd.read_parquet(os.path.join(NCD, 'splits', 't2_samples.parquet'))
    k = t2.groupby('nuts0').agg(n=('id', 'size'), lon=('lon', 'median'), lat=('lat', 'median')).reset_index()
    k['name'] = k.nuts0.map(NAME)
    k['role'] = np.where(k.nuts0.isin(tg['T2']), 'target', 'source')
    save(k.rename(columns={'nuts0': 'domain'}), 'lucas_domains.csv')
    # 作物：11 国（每国轮流作目标，其余 10 国作源池），样本量沿用 RSE 图 1；DE、RO 的实际研究区另存
    cc = pd.read_csv(os.path.join(RSE_SRC, 'country_counts.csv')).rename(columns={'country': 'domain', 'n_parcels': 'n'})
    cc['name'] = cc.domain.map(NAME)
    cc['role'] = np.where(cc.primary9, 'target', 'target_other')
    save(cc[['domain', 'name', 'n', 'role']], 'crop_domains.csv')
    for f in ('study_regions_de.geojson', 'study_regions_ro.geojson'):
        shutil.copy(os.path.join(RSE_SRC, f), os.path.join(OUT, f'crop_{f}'))


# ================================================================ 主程序
def run_task(task, units_iter):
    print(f'== {task}')
    U, CLS, SZ, ARM, CVs = [], [], [], [], {}
    prim = set()
    for u in units_iter:
        rows, cls_rows, size_rows, arm_rows, CV, star = unit_rows(u)
        U += rows; CLS += cls_rows; SZ += size_rows; ARM += arm_rows
        CVs[u['unit']] = CV
        if u['primary']:
            prim.add(u['domain'])
        print(f'   {u["unit"]}: n_test {len(u["yT"])}, BA 类 {len(u["big"])}', flush=True)
    U = pd.DataFrame(U)
    D = domain_table(U)
    reconcile(task, D)
    primary = sorted(prim)
    E = estimates(D, task, primary)
    C = pd.DataFrame(CLS)
    save(U.assign(task=task), f'{task}_unit_rows.csv')
    save(D.assign(task=task), f'{task}_domain_values.csv')
    save(E, f'{task}_estimates.csv')
    save(pd.DataFrame(ARM).assign(task=task), f'{task}_arm_rows.csv')
    save(C.assign(task=task), f'{task}_class_rows.csv')
    save(group_summary(C, 'cls_name', task, primary), f'{task}_class_summary.csv')
    if SZ:
        Z = pd.DataFrame(SZ)
        save(Z.assign(task=task), f'{task}_size_rows.csv')
        g = group_summary(Z, 'grp_lab', task, primary)
        share = Z[Z.domain.isin(primary)].groupby(['domain', 'grp_lab']).share.mean().groupby('grp_lab').mean()
        g['share'] = g.grp_lab.map(share)
        g['order'] = g.grp_lab.map({l: i for i, l in enumerate(SIZE_LAB)})
        save(g.sort_values('order'), f'{task}_size_summary.csv')
    ex = example_domain(D, primary)
    ex_unit = 'BE_F1' if task == 'crop' else ex          # 作物沿用 RSE 图 2 的示例单元
    save(pd.DataFrame(curves_long(task, ex_unit, CVs[ex_unit])), f'{task}_example_curves.csv')
    lay, cells = layout_crop(ex_unit) if task == 'crop' else layout_pilot(task, ex_unit)
    save(lay, f'{task}_example_layout.csv')
    save(cells, f'{task}_example_test_cells.csv')
    return dict(task=task, primary=primary, all=sorted(D.domain.unique()), example_unit=ex_unit,
                n_units=int(U.unit.nunique()))


def main():
    meta = []
    meta.append(run_task('crop', crop_units()))
    meta.append(run_task('lcz', pilot_units('lcz')))
    meta.append(run_task('lucas', pilot_units('lucas')))
    print('== 地图')
    maps()
    json.dump(meta, open(os.path.join(OUT, 'tasks_meta.json'), 'w'), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
