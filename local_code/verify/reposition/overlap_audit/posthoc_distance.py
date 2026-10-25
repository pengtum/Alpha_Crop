"""
事后描述性分析（重合审计，2026-09-27）：用检验 1 的单源两两迁移（pairwise_v2.csv）
检查 Cheng et al. 2026 (RSE 347:115694) 的模式——"AE 相对 S2 的优势随源-目标热量差增大"。

没有 GDD，用国家质心纬度差 |Δlat| 与大圆距离作代理（Cheng 自己给出 1400–1500 °C·d ≈ 9–10° 纬度）。
另外用 G 的 k=0（多源外国池，只用外国样本）逐目标国的 AE−S2 作补充描述。

只读项目与结果文件；不训练模型；输出写到本目录。
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '2')

import glob
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr, pearsonr

OUT = os.path.dirname(os.path.abspath(__file__))
RES = '/Users/perng_mini/AlphaCrops_local/results'
DATA = '/Users/perng_mini/AlphaCrops_local/data/dataset_v2.parquet'
C = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']
RNG = np.random.default_rng(42)
NPERM = 20000
NBOOT = 5000


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def ols(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    r2 = 1 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return b, r2


def fe_slope(df, xcol, ycol):
    """双向固定效应（源国 + 目标国）下 x 的斜率。"""
    S = pd.get_dummies(df['source'], drop_first=True, dtype=float).to_numpy()
    T = pd.get_dummies(df['target'], drop_first=True, dtype=float).to_numpy()
    X = np.column_stack([df[xcol].to_numpy(float), S, T])
    b, _ = ols(df[ycol].to_numpy(float), X)
    return b[1]


# ---------- 1. 国家质心（与 load_dataset 相同：AE 与 S2 都有的地块） ----------
tab = pq.read_table(DATA, columns=['country', 'longitude', 'latitude', 'has_ae', 'has_s2']).to_pandas()
tab = tab[tab.has_ae & tab.has_s2]
cen = tab.groupby('country').agg(lat=('latitude', 'mean'), lon=('longitude', 'mean'),
                                 lat_p10=('latitude', lambda s: s.quantile(.1)),
                                 lat_p90=('latitude', lambda s: s.quantile(.9)),
                                 n=('latitude', 'size')).loc[C]
cen.round(3).to_csv(os.path.join(OUT, 'country_centroids.csv'))

# ---------- 2. 两两迁移表 ----------
p = pd.read_csv(os.path.join(RES, 'pairwise_v2.csv'))
w = p.pivot_table(index=['source', 'target'], columns='feature', values='ba').reset_index()
w[['ae', 's2', 'fusion']] *= 100  # pp
local = w[w.source == w.target].set_index('source')
x = w[w.source != w.target].copy()
x['dlat'] = [abs(cen.lat[s] - cen.lat[t]) for s, t in zip(x.source, x.target)]
x['geo_km'] = [haversine_km(cen.lon[s], cen.lat[s], cen.lon[t], cen.lat[t]) for s, t in zip(x.source, x.target)]
dd = pd.read_csv(os.path.join(RES, 'domain_distance.csv'))
cz = {}
for r in dd.itertuples():
    cz[(r.c1, r.c2)] = cz[(r.c2, r.c1)] = r.climate_diff
x['climate_diff'] = [cz[(s, t)] for s, t in zip(x.source, x.target)]
x['d_ae_s2'] = x.ae - x.s2
x['d_fu_s2'] = x.fusion - x.s2
# 相对本国（源国国内测试）的迁移损失：Cheng 的 ALR 是绝对精度对 |ΔGDD| 的斜率，这里另给"损失"版本
x['loss_ae'] = [local.loc[s, 'ae'] for s in x.source] - x.ae
x['loss_s2'] = [local.loc[s, 's2'] for s in x.source] - x.s2
x.round(4).to_csv(os.path.join(OUT, 'pairwise_with_distance.csv'), index=False)


def perm_p(df, xcol, ycol, fe=False, nperm=NPERM):
    """节点置换（QAP 式）：把国家质心随机重新分配给国家，重算距离与斜率。保持二元依赖结构。"""
    obs = fe_slope(df, xcol, ycol) if fe else ols(df[ycol].to_numpy(float), df[xcol].to_numpy(float))[0][1]
    lat = cen.lat.to_numpy(); lon = cen.lon.to_numpy()
    idx = {c: i for i, c in enumerate(C)}
    si = df.source.map(idx).to_numpy(); ti = df.target.map(idx).to_numpy()
    y = df[ycol].to_numpy(float)
    null = np.empty(nperm)
    if fe:
        S = pd.get_dummies(df['source'], drop_first=True, dtype=float).to_numpy()
        T = pd.get_dummies(df['target'], drop_first=True, dtype=float).to_numpy()
    for b in range(nperm):
        pi = RNG.permutation(len(C))
        if xcol == 'dlat':
            xv = np.abs(lat[pi][si] - lat[pi][ti])
        else:
            xv = haversine_km(lon[pi][si], lat[pi][si], lon[pi][ti], lat[pi][ti])
        if fe:
            bb, _ = ols(y, np.column_stack([xv, S, T])); null[b] = bb[1]
        else:
            bb, _ = ols(y, xv); null[b] = bb[1]
    return obs, float((np.abs(null) >= abs(obs)).mean())


def cluster_boot(df, xcol, ycol, nboot=NBOOT):
    """国家块 bootstrap（二元数据）：有放回抽 11 个国家，取抽中国家之间的全部有向对。"""
    out = []
    pairs = {(s, t): i for i, (s, t) in enumerate(zip(df.source, df.target))}
    xv = df[xcol].to_numpy(float); yv = df[ycol].to_numpy(float)
    for _ in range(nboot):
        draw = RNG.choice(C, len(C), replace=True)
        rows = [pairs[(a, b)] for i, a in enumerate(draw) for j, b in enumerate(draw)
                if i != j and a != b and (a, b) in pairs]
        if len(rows) < 10 or np.ptp(xv[rows]) == 0:
            continue
        out.append(ols(yv[rows], xv[rows])[0][1])
    return np.percentile(out, [2.5, 97.5])


res = {}
# ---------- 3. 各特征的"精度 ~ 距离"斜率（ALR 代理）与 AE−S2 差 ~ 距离 ----------
for xcol, unit in [('dlat', 1.0), ('geo_km', 1000.0)]:
    for ycol in ['ae', 's2', 'fusion', 'd_ae_s2', 'd_fu_s2', 'loss_ae', 'loss_s2']:
        b, r2 = ols(x[ycol].to_numpy(float), x[xcol].to_numpy(float))
        obs, pp = perm_p(x, xcol, ycol)
        ci = cluster_boot(x, xcol, ycol)
        feo, fep = perm_p(x, xcol, ycol, fe=True, nperm=5000)
        rho, rp = spearmanr(x[xcol], x[ycol])
        res[f'{ycol}~{xcol}'] = dict(
            intercept=round(b[0], 3), slope_per_unit=round(b[1] * unit, 3),
            unit='1 deg' if xcol == 'dlat' else '1000 km',
            r2=round(r2, 3), spearman=round(rho, 3), spearman_naive_p=round(rp, 4),
            perm_p=round(pp, 4), boot95=[round(v * unit, 3) for v in ci],
            fe_slope=round(feo * unit, 3), fe_perm_p=round(fep, 4))
        if ycol in ('d_ae_s2', 'd_fu_s2') and b[1] != 0:
            res[f'{ycol}~{xcol}']['crossover'] = round(-b[0] / b[1] * (1 if xcol == 'dlat' else 1), 2)

# ---------- 4. 分箱（对照 Cheng Fig. 6b） ----------
bins = [0, 3, 6, 9, 12, 20]
x['dlat_bin'] = pd.cut(x.dlat, bins, right=False)
bt = x.groupby('dlat_bin', observed=True).agg(n=('d_ae_s2', 'size'), ae=('ae', 'mean'), s2=('s2', 'mean'),
                                               d_mean=('d_ae_s2', 'mean'), d_sd=('d_ae_s2', 'std'),
                                               ae_wins=('d_ae_s2', lambda s: int((s > 0).sum())))
bt.round(2).to_csv(os.path.join(OUT, 'bins_dlat.csv'))

# ---------- 5. 敏感性：去掉某些国家 ----------
sens = {}
for drop in [[], ['PT'], ['RO'], ['PT', 'RO'], ['SE'], ['LV']]:
    s = x[~x.source.isin(drop) & ~x.target.isin(drop)]
    b, r2 = ols(s.d_ae_s2.to_numpy(float), s.dlat.to_numpy(float))
    ba, _ = ols(s.ae.to_numpy(float), s.dlat.to_numpy(float))
    bs, _ = ols(s.s2.to_numpy(float), s.dlat.to_numpy(float))
    sens['drop_' + ('none' if not drop else '+'.join(drop))] = dict(
        n=len(s), d_slope_per_deg=round(b[1], 3), d_intercept=round(b[0], 3), r2=round(r2, 3),
        ae_slope=round(ba[1], 3), s2_slope=round(bs[1], 3), mean_d=round(s.d_ae_s2.mean(), 3),
        far_ge9_n=int((s.dlat >= 9).sum()),
        far_ge9_mean_d=round(s[s.dlat >= 9].d_ae_s2.mean(), 3) if (s.dlat >= 9).any() else None,
        near_lt9_mean_d=round(s[s.dlat < 9].d_ae_s2.mean(), 3))
# 按方向：PT 作目标 / 作源
for role in ['source', 'target']:
    s = x[x[role] == 'PT']
    sens[f'PT_as_{role}'] = dict(n=len(s), mean_d=round(s.d_ae_s2.mean(), 3),
                                 ae_wins=int((s.d_ae_s2 > 0).sum()),
                                 rows=s[['source', 'target', 'dlat', 'ae', 's2', 'd_ae_s2']].round(2).to_dict('records'))

# ---------- 6. 本地（源国国内测试，随机划分无缓冲） ----------
loc = dict(ae_mean=round(local.ae.mean(), 2), s2_mean=round(local.s2.mean(), 2),
           d_mean=round((local.ae - local.s2).mean(), 2),
           ae_lower_n=int((local.ae < local.s2).sum()), n=len(local))

# ---------- 7. 多源 k=0（G stage0 的 foreign_only 行） ----------
rows = []
for f in sorted(glob.glob(os.path.join(RES, 'guidance_G', 'rows_t1_*_r0.csv'))):
    d = pd.read_csv(f)
    rows.append(d[d.method == 'foreign_only'])
g0 = pd.concat(rows)
g0 = g0.pivot_table(index=['target', 'split'], columns='clf', values=['oa', 'ba_ge30']).reset_index()
g0.columns = ['_'.join([c for c in col if c]) for col in g0.columns]
g0 = g0.groupby('target')[['oa_ae', 'oa_s2', 'ba_ge30_ae', 'ba_ge30_s2']].mean().loc[C]
g0['d_oa'] = g0.oa_ae - g0.oa_s2
g0['d_ba'] = g0.ba_ge30_ae - g0.ba_ge30_s2
g0['mean_dlat_to_pool'] = [np.mean([abs(cen.lat[t] - cen.lat[s]) for s in C if s != t]) for t in C]
g0['min_dlat_to_pool'] = [np.min([abs(cen.lat[t] - cen.lat[s]) for s in C if s != t]) for t in C]
g0['mean_km_to_pool'] = [np.mean([haversine_km(cen.lon[t], cen.lat[t], cen.lon[s], cen.lat[s]) for s in C if s != t]) for t in C]
g0['lat'] = cen.lat
g0.round(3).to_csv(os.path.join(OUT, 'multisource_k0_by_target.csv'))
ms = {}
for yc in ['d_oa', 'd_ba']:
    for xc in ['mean_dlat_to_pool', 'min_dlat_to_pool', 'mean_km_to_pool']:
        r, pv = spearmanr(g0[xc], g0[yc])
        ms[f'{yc}~{xc}'] = dict(spearman=round(r, 3), p=round(pv, 3))
    ms[f'{yc}_mean'] = round(g0[yc].mean(), 3)
    ms[f'{yc}_ae_better_n'] = int((g0[yc] > 0).sum())

# ---------- 8. 代理与 Cheng 的换算 ----------
cheng = dict(
    note='Cheng: 1400-1500 °C·d ≈ 9-10° 纬度 → 约 150 °C·d/°（本分析采用的换算，仅作量级参照）',
    predicted_d_slope_pp_per_deg=round(9.0 * 0.15, 2),  # ΔOA 斜率 0.09 / 1000 °C·d（RF, Fig. 6b）
    predicted_alr_ae_pp_per_deg=round(14 * 0.15, 2), predicted_alr_s2_pp_per_deg=round(23 * 0.15, 2),
    predicted_crossover_deg='9-10')
corr_proxy = dict(dlat_vs_geo_spearman=round(spearmanr(x.dlat, x.geo_km)[0], 3))

summary = dict(ols_and_tests=res, sensitivity=sens, local=loc, multisource_k0=ms,
               cheng_conversion=cheng, proxy_corr=corr_proxy,
               n_pairs=len(x), n_far_ge9=int((x.dlat >= 9).sum()), n_far_ge10=int((x.dlat >= 10).sum()),
               dlat_range=[round(x.dlat.min(), 2), round(x.dlat.max(), 2)])
with open(os.path.join(OUT, 'posthoc_summary.json'), 'w') as fh:
    json.dump(summary, fh, ensure_ascii=False, indent=1, default=str)
print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
print(cen.round(2))
print(bt.round(2))
print(g0.round(2))
