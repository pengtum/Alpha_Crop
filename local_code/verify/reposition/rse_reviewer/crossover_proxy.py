"""只读轻量计算：在已有的单源两两迁移（pairwise_v2.csv，LightGBM，BA）上，
用纬度差 |Δlat| 与地理距离作为 Cheng 2026 热量差 |ΔGDD| 的粗代理，
检查 AE−S2 迁移差是否随距离增大而转正（Cheng 的"交叉点"现象）。
不训练模型；只读 dataset_v2 的 country/lat/lon/has_* 列与 pairwise_v2.csv。
注意：110 对来自 11 国，不独立；以国家为单位做 bootstrap（源、目标同时按国家重抽的简化：重抽国家集合后保留两端都在集合中的对）。
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RES = '/Users/perng_mini/AlphaCrops_local/results'
DATA = '/Users/perng_mini/AlphaCrops_local/data/dataset_v2.parquet'

t = pq.read_table(DATA, columns=['country', 'longitude', 'latitude', 'label', 'has_ae', 'has_s2']).to_pandas()
t = t[t.has_ae & t.has_s2]
cent = t.groupby('country')[['latitude', 'longitude']].mean()
ncls = t.groupby('country')['label'].nunique()
print('国家质心纬度（AE、S2 都有的地块）：')
print(cent.round(2).assign(n_class=ncls).to_string())

pw = pd.read_csv(f'{RES}/pairwise_v2.csv')
pw = pw[pw.source != pw.target]
w = pw.pivot_table(index=['source', 'target'], columns='feature', values='ba').reset_index()
dd = pd.read_csv(f'{RES}/domain_distance.csv')
geo = {}
for _, r in dd.iterrows():
    geo[(r.c1, r.c2)] = r.geo_km
    geo[(r.c2, r.c1)] = r.geo_km
w['geo_km'] = [geo[(a, b)] for a, b in zip(w.source, w.target)]
w['dlat'] = [abs(cent.latitude[a] - cent.latitude[b]) for a, b in zip(w.source, w.target)]
w['d'] = (w.ae - w.s2) * 100
w['ae'] *= 100
w['s2'] *= 100
print(f'\n对数 n={len(w)}；AE 均值 {w.ae.mean():.2f}，S2 均值 {w.s2.mean():.2f}，AE−S2 {w.d.mean():+.2f} pp；AE 更高 {int((w.d>0).sum())}/{len(w)}')


def fit(df, x):
    X = np.c_[np.ones(len(df)), df[x].to_numpy()]
    out = {}
    for y in ['ae', 's2', 'd']:
        b = np.linalg.lstsq(X, df[y].to_numpy(), rcond=None)[0]
        out[y] = b
    return out


def fit_fe(df, x):
    """带目标国固定效应的斜率（目标国决定了类别集合与基准难度）。"""
    D = pd.get_dummies(df.target, drop_first=False).to_numpy(float)
    X = np.c_[D, df[x].to_numpy()]
    return {y: np.linalg.lstsq(X, df[y].to_numpy(), rcond=None)[0][-1] for y in ['ae', 's2', 'd']}


countries = sorted(set(w.source))
rng = np.random.default_rng(42)
for x, unit, scale in [('dlat', '每 1° 纬度差', 1.0), ('geo_km', '每 1000 km', 1000.0)]:
    f = fit(w, x)
    fe = fit_fe(w, x)
    sl_ae, sl_s2, sl_d = f['ae'][1] * scale, f['s2'][1] * scale, f['d'][1] * scale
    cross = -f['d'][0] / f['d'][1] if f['d'][1] != 0 else np.nan
    boots = []
    for _ in range(2000):
        s = rng.choice(countries, size=len(countries), replace=True)
        cnt = pd.Series(s).value_counts()
        sub = w[w.source.isin(cnt.index) & w.target.isin(cnt.index)]
        wt = sub.source.map(cnt) * sub.target.map(cnt)
        sub = sub.loc[sub.index.repeat(wt.astype(int))]
        if sub[x].nunique() < 3:
            continue
        g = fit(sub, x)
        boots.append([g['ae'][1] * scale, g['s2'][1] * scale, g['d'][1] * scale])
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [5, 95], axis=0)
    rs = pd.Series(w.d).corr(w[x], method='spearman')
    print(f'\n== 代理变量 {x}（{unit}）==  范围 {w[x].min():.1f}–{w[x].max():.1f}')
    print(f'  BA 斜率（pp/{unit}）：AE {sl_ae:+.2f} [90% 国家 bootstrap {lo[0]:+.2f}, {hi[0]:+.2f}]；'
          f'S2 {sl_s2:+.2f} [{lo[1]:+.2f}, {hi[1]:+.2f}]')
    print(f'  AE−S2 差的斜率：{sl_d:+.2f} [{lo[2]:+.2f}, {hi[2]:+.2f}]；Spearman ρ(差, {x})={rs:+.3f}；'
          f'截距 {f["d"][0]:+.2f}；线性交叉点 {x}={cross:.1f}')
    print(f'  目标国固定效应下斜率：AE {fe["ae"]*scale:+.2f}，S2 {fe["s2"]*scale:+.2f}，差 {fe["d"]*scale:+.2f}')
    # 分箱
    bins = [0, 3, 6, 9, 12, 25] if x == 'dlat' else [0, 500, 1000, 1500, 2000, 4000]
    w['bin'] = pd.cut(w[x], bins)
    tab = w.groupby('bin', observed=True).agg(n=('d', 'size'), ae=('ae', 'mean'), s2=('s2', 'mean'),
                                              d=('d', 'mean'), ae_win=('d', lambda v: (v > 0).mean()))
    print(tab.round(2).to_string())

# 去掉 RO（只有 3 类，标签映射伪影）与 LV（S2 第 0008 批缺失）后的稳健性
for drop in [['RO'], ['LV'], ['RO', 'LV'], ['PT']]:
    sub = w[~w.source.isin(drop) & ~w.target.isin(drop)]
    f = fit(sub, 'dlat')
    print(f'\n去掉 {drop}：n={len(sub)}，AE−S2 均值 {sub.d.mean():+.2f}，斜率(每1°) AE {f["ae"][1]:+.2f} S2 {f["s2"][1]:+.2f} 差 {f["d"][1]:+.2f}，'
          f'ρ(差,dlat)={sub.d.corr(sub.dlat, method="spearman"):+.3f}')

w.drop(columns='bin').to_csv('crossover_proxy_pairs.csv', index=False)
