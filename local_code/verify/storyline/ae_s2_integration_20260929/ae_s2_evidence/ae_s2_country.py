"""
只读描述性重算：AE 相对 S2 的逐国模式及其与国家协变量的关系（不训练任何模型）。
输入全部是已有输出文件；输出写到本脚本所在目录。
运行：/Users/perng_mini/AlphaCrops_local/.venv/bin/python -B ae_s2_country.py
"""
import glob
import json
import os

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import numpy as np
import pandas as pd
from scipy import stats

OUT = os.path.dirname(os.path.abspath(__file__))
L = '/Users/perng_mini/AlphaCrops_local'
G_DIR = f'{L}/results/guidance_G'
ML_DIR = f'{L}/results/guidance_G_ml'
C = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']
C9 = [c for c in C if c not in ('RO', 'PT')]
CLASSES = ['grassland', 'maize', 'orchards', 'other_crops', 'potatoes', 'root_crops',
           'strawberries', 'tree_wood_forest', 'vegetables', 'winter_cereals']
UNIT = ['target', 'split', 'rep']
LEARNERS = ['lgb', 'lr', 'mlp', 'rf']


def tsum(v):
    v = pd.Series(v).dropna()
    n = len(v)
    m = v.mean()
    se = v.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    p = 2 * stats.t.sf(abs(m / se), n - 1) if se > 0 else np.nan
    return dict(n=n, mean=m, lo95=m - h, hi95=m + h, p=p, npos=int((v > 0).sum()))


def fmt(s):
    return f"{s['mean']:+.2f} [{s['lo95']:+.2f}, {s['hi95']:+.2f}]，p={s['p']:.3f}，>0 {s['npos']}/{s['n']}"


# ---------------------------------------------------------------- 1. 读 SRS / k=0 行（四个学习器）
def load_rows():
    G = pd.concat([pd.read_csv(f) for f in glob.glob(f'{G_DIR}/rows_t*_*.csv')], ignore_index=True)
    G = G[G.method.isin(['srs', 'foreign_only']) & G.clf.isin(['ae', 's2'])].assign(learner='lgb')
    M = pd.concat([pd.read_csv(f) for f in glob.glob(f'{ML_DIR}/rows_*_*.csv')], ignore_index=True)
    M = M[M.method.isin(['srs', 'foreign_only']) & M.clf.isin(['ae', 's2'])]
    if 'status' in M:
        M = M[M.status.fillna('ok') != 'failed']
    R = pd.concat([G, M], ignore_index=True)
    return R[R.k.isin([0, 50, 100, 200, 400])]


R = load_rows()
log = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    log.append(s)


P('# AE − S2 逐国描述性重算（只读，不训练）')
P('行数（srs + foreign_only，四学习器）：', len(R), R.groupby('learner').size().to_dict())


def feat_delta(metric):
    P_ = R.pivot_table(index=['learner'] + UNIT + ['k'], columns='clf', values=metric, aggfunc='mean')
    d = (P_['ae'] - P_['s2']).rename('d').reset_index()
    return d.groupby(['learner', 'target', 'k']).d.mean().reset_index()


D = {m: feat_delta(m) for m in ('ba_ge30', 'oa')}
wide = []
for m, d in D.items():
    for k in (0, 200, 400):
        w = d[d.k == k].pivot(index='target', columns='learner', values='d')[LEARNERS]
        w.columns = [f'{m}_k{k}_{c}' for c in w.columns]
        wide.append(w)
W = pd.concat(wide, axis=1).loc[C]
W.round(3).to_csv(f'{OUT}/country_dfeat_4learners.csv')

P('\n## 1. 核对：11 国均值应与 report_ML_rev3.txt 一致')
for m in ('ba_ge30', 'oa'):
    for L_ in LEARNERS:
        for k in (0, 400):
            s = tsum(W[f'{m}_k{k}_{L_}'])
            s9 = tsum(W.loc[C9, f'{m}_k{k}_{L_}'])
            P(f'  {m:8s} {L_:4s} k={k:<3d} 11 国 {fmt(s)} | 去 RO/PT {s9["mean"]:+.2f}（{s9["npos"]}/9）')

P('\n## 2. 逐国 AE − S2（ba_ge30 与 OA，k=0 与 k=400，四学习器）')
for m in ('ba_ge30', 'oa'):
    for k in (0, 400):
        cols = [f'{m}_k{k}_{L_}' for L_ in LEARNERS]
        t = W[cols].copy()
        t.columns = LEARNERS
        t['mean4'] = t.mean(1)
        t['n_neg'] = (t[LEARNERS] < 0).sum(1)
        P(f'\n### {m}, k={k}')
        P(t.round(2).to_string())
        # 学习器之间逐国模式的一致性
        rho = t[LEARNERS].corr(method='spearman').values
        iu = np.triu_indices(4, 1)
        P(f'  学习器两两 Spearman（逐国 11 值）：中位数 {np.median(rho[iu]):+.2f}，范围 [{rho[iu].min():+.2f}, {rho[iu].max():+.2f}]')
        # Kendall W
        rk = t[LEARNERS].rank()
        Sx = ((rk.sum(1) - rk.sum(1).mean()) ** 2).sum()
        m_, n_ = 4, len(t)
        Wk = 12 * Sx / (m_ ** 2 * (n_ ** 3 - n_))
        chi = m_ * (n_ - 1) * Wk
        P(f'  Kendall W = {Wk:.2f}（χ²={chi:.1f}，df={n_-1}，p={stats.chi2.sf(chi, n_-1):.3f}）')

# ---------------------------------------------------------------- 2. 逐类召回 AE − S2（k=0 / k=400）
cnt = pd.read_csv(f'{L}/verify/storyline/paper_plan/G_test_class_counts.csv')
cls_rows = []
for L_ in LEARNERS:
    for k in (0, 400):
        S = R[(R.learner == L_) & (R.k == k)]
        for c_i, cname in enumerate(CLASSES):
            col = f'rec{c_i}'
            Pv = S.pivot_table(index=UNIT, columns='clf', values=col, aggfunc='mean')
            d = (Pv['ae'] - Pv['s2']).rename('d').reset_index()
            d = d.merge(cnt[['country', 'split', f'test_{cname}']].rename(columns={'country': 'target'}),
                        on=['target', 'split'])
            d = d[d[f'test_{cname}'] >= 30]
            cv = d.groupby('target').d.mean() * 100
            if len(cv) >= 3:
                s = tsum(cv)
                cls_rows.append(dict(learner=L_, k=k, cls=cname, n=s['n'], mean=s['mean'], lo95=s['lo95'],
                                     hi95=s['hi95'], p=s['p'], npos=s['npos'], median=cv.median(),
                                     per_country=json.dumps(cv.round(1).to_dict())))
CL = pd.DataFrame(cls_rows)
CL.round(3).to_csv(f'{OUT}/class_recall_ae_minus_s2.csv', index=False)
P('\n## 3. 逐类召回 AE − S2（pp；只含该类在该划分测试集 ≥30 地块的国家；国家为单位 t 区间，描述）')
for k in (0, 400):
    P(f'\n### k={k}')
    t = CL[CL.k == k].pivot(index='cls', columns='learner', values='mean')[LEARNERS]
    t['n'] = CL[(CL.k == k) & (CL.learner == 'lgb')].set_index('cls').n
    t['npos_lgb'] = CL[(CL.k == k) & (CL.learner == 'lgb')].set_index('cls').npos
    t['mean4'] = t[LEARNERS].mean(1)
    P(t.round(2).to_string())
P('\nLightGBM k=0 逐类逐国（AE−S2 召回，pp）：')
for _, r in CL[(CL.learner == 'lgb') & (CL.k == 0)].iterrows():
    P(f"  {r.cls:18s} {fmt(r)}  {r.per_country}")

# ---------------------------------------------------------------- 3. 国家协变量
cov = pd.DataFrame(index=C)
ms = pd.read_csv(f'{L}/verify/reposition/overlap_audit/multisource_k0_by_target.csv').set_index('target')
cen = pd.read_csv(f'{L}/verify/reposition/overlap_audit/country_centroids.csv').set_index('country')
cov['lat'] = cen.lat
cov['lat_span_p10_p90'] = cen.lat_p90 - cen.lat_p10
cov['mean_dlat_to_pool'] = ms.mean_dlat_to_pool
cov['min_dlat_to_pool'] = ms.min_dlat_to_pool
cov['mean_km_to_pool'] = ms.mean_km_to_pool
comp = pd.read_csv(f'{L}/verify/reposition/label_budget_line/composition_by_target.csv').groupby('target').mean(numeric_only=True)
cov['comp_L1_vs_foreign'] = comp.L1_true
cov['n_cls_test'] = comp.n_cls_test
gt = cnt.groupby('country').sum(numeric_only=True)
cov['grass_share_test'] = gt.test_grassland / gt.n_test
cov['n_big_ge30'] = cnt.groupby('country').n_big_ge30.mean()
# 几何质量（G 的指导质量表：候选池 10 近邻 BA、margin AUROC、NLL）
gq = pd.read_csv(f'{G_DIR}/guidance_quality_G.csv')
gqm = gq.groupby(['target', 'feat'])[['knn10_ba', 'auroc_margin', 'nll_pool', 'ece_oof', 'temp']].mean().unstack('feat')
cov['knn10_ae_minus_s2'] = gqm[('knn10_ba', 'ae')] - gqm[('knn10_ba', 's2')]
cov['nll_pool_ae_minus_s2'] = gqm[('nll_pool', 'ae')] - gqm[('nll_pool', 's2')]
# Betti 迁移诊断：分离比与外国计入率（主设定 K=8、q95；计入率与 w、k 无关）
bt = pd.read_csv(f'{L}/verify/betti_transfer/diag_rows.csv')
bm = bt[(bt.K == 8) & (bt.thr == 'q95') & (bt.w == '1/50') & (bt.k == 200)]
bq = bm.groupby(['country', 'space'])[['sep_ratio', 'foreign_counted_rate']].mean().unstack('space')
cov['sep_ratio_ae'] = bq[('sep_ratio', 'ae')]
cov['sep_ratio_s2'] = bq[('sep_ratio', 's2')]
cov['sep_ratio_ae_over_s2'] = cov.sep_ratio_ae / cov.sep_ratio_s2
cov['foreign_in_rate_ae'] = bq[('foreign_counted_rate', 'ae')]
cov['foreign_in_rate_s2'] = bq[('foreign_counted_rate', 's2')]
# 检验 1 的国家对距离：到其余 10 国的平均 MMD、C2ST、类别 JS
dd = pd.read_csv(f'{L}/results/domain_distance.csv')
dd2 = pd.concat([dd, dd.rename(columns={'c1': 'c2', 'c2': 'c1'})])
dm = dd2.groupby('c1')[['mmd_ae', 'mmd_s2', 'c2st_ae', 'c2st_s2', 'class_js', 'geo_km']].mean()
cov['mmd_ae_mean'] = dm.mmd_ae
cov['mmd_s2_mean'] = dm.mmd_s2
cov['mmd_ae_rank_minus_s2_rank'] = dm.mmd_ae.rank() - dm.mmd_s2.rank()
cov['c2st_ae_mean'] = dm.c2st_ae
cov['c2st_s2_mean'] = dm.c2st_s2
cov['class_js_mean'] = dm.class_js
# 国内（大量目标标签）AE − S2：tgt300（缓冲格网划分 0，约 3 万池标签）与 pairwise_v2 国内测试
ub = pd.read_csv(f'{L}/verify/model_diag/fit/table_upper_bound.csv')
ubw = ub.pivot(index='tgt', columns='feat')
cov['tgt300_ba_ae_minus_s2'] = ubw[('tgt300_ba', 'ae')] - ubw[('tgt300_ba', 's2')]
cov['tgt300_oa_ae_minus_s2'] = ubw[('tgt300_oa', 'ae')] - ubw[('tgt300_oa', 's2')]
cov['k0s0_ba_ae_minus_s2'] = ubw[('k0_ba', 'ae')] - ubw[('k0_ba', 's2')]
cov['k0s0_oa_ae_minus_s2'] = ubw[('k0_oa', 'ae')] - ubw[('k0_oa', 's2')]
pw = pd.read_csv(f'{L}/results/pairwise_v2.csv')
pwd = pw[pw.source == pw.target].pivot(index='target', columns='feature', values='ba')
cov['domestic_ba_ae_minus_s2'] = 100 * (pwd['ae'] - pwd['s2'])
pwx = pw[pw.source != pw.target].pivot_table(index='target', columns='feature', values='ba', aggfunc='mean')
cov['single_src_mean_ba_ae_minus_s2'] = 100 * (pwx['ae'] - pwx['s2'])
# 地块面积与 30 m 邻域核：核 = 半径 3 像元的圆（约 29 个 10 m 像元 ≈ 0.29 ha）
ds = pd.read_parquet(f'{L}/data/dataset_v2.parquet', columns=['country', 'area_ha', 'latitude', 'longitude', 'has_ae', 'has_s2', 'label'])
ds = ds[ds.has_ae & ds.has_s2]
g = ds.groupby('country')
cov['n_parcels'] = g.size()
cov['median_area_ha'] = g.area_ha.median()
cov['share_lt_0p29ha'] = g.area_ha.apply(lambda a: (a < 0.29).mean())
cov['share_lt_0p9ha'] = g.area_ha.apply(lambda a: (a < 0.9).mean())


def extent(sub):
    la = sub.latitude.quantile([0.05, 0.95]).values
    lo = sub.longitude.quantile([0.05, 0.95]).values
    km_lat = (la[1] - la[0]) * 111.2
    km_lon = (lo[1] - lo[0]) * 111.2 * np.cos(np.deg2rad(sub.latitude.median()))
    return km_lat * km_lon / 1e3


cov['extent_p5_p95_1e3km2'] = g.apply(extent)
cov.round(4).to_csv(f'{OUT}/country_covariates.csv')

P('\n## 4. 国家协变量（country_covariates.csv）')
P(cov.round(3).T.to_string())

P('\n## 5. 分离比与外国计入率（Betti 迁移诊断主设定 K=8、q95；4 单元平均）')
P(cov[['sep_ratio_ae', 'sep_ratio_s2', 'sep_ratio_ae_over_s2', 'foreign_in_rate_ae', 'foreign_in_rate_s2']].round(3).to_string())
P('  均值：', cov[['foreign_in_rate_ae', 'foreign_in_rate_s2']].mean().round(3).to_dict(),
  '中位数：', cov[['foreign_in_rate_ae', 'foreign_in_rate_s2']].median().round(3).to_dict())

# ---------------------------------------------------------------- 4. 相关（描述，朴素 p，未校正）
outcomes = {
    'dBA_k0_lgb': W['ba_ge30_k0_lgb'], 'dBA_k0_mean4': W[[f'ba_ge30_k0_{x}' for x in LEARNERS]].mean(1),
    'dOA_k0_lgb': W['oa_k0_lgb'], 'dOA_k0_mean4': W[[f'oa_k0_{x}' for x in LEARNERS]].mean(1),
    'dBA_k400_mean4': W[[f'ba_ge30_k400_{x}' for x in LEARNERS]].mean(1),
}
preds = ['lat', 'lat_span_p10_p90', 'mean_dlat_to_pool', 'min_dlat_to_pool', 'mean_km_to_pool',
         'comp_L1_vs_foreign', 'n_cls_test', 'grass_share_test', 'knn10_ae_minus_s2', 'nll_pool_ae_minus_s2',
         'sep_ratio_ae', 'sep_ratio_ae_over_s2', 'foreign_in_rate_ae', 'foreign_in_rate_s2',
         'mmd_ae_mean', 'mmd_s2_mean', 'class_js_mean', 'tgt300_ba_ae_minus_s2', 'domestic_ba_ae_minus_s2',
         'single_src_mean_ba_ae_minus_s2', 'median_area_ha', 'share_lt_0p29ha', 'extent_p5_p95_1e3km2', 'n_parcels']
rows = []
for on, y in outcomes.items():
    for pv in preds:
        x = cov[pv]
        r11 = stats.spearmanr(x.loc[C], y.loc[C])
        r9 = stats.spearmanr(x.loc[C9], y.loc[C9])
        rows.append(dict(outcome=on, predictor=pv, rho11=r11[0], p11=r11[1], rho9=r9[0], p9=r9[1]))
CR = pd.DataFrame(rows)
CR.round(3).to_csv(f'{OUT}/corr_dfeat_vs_covariates.csv', index=False)
P(f'\n## 6. 逐国 AE−S2 与协变量的 Spearman（n=11 与去 RO/PT n=9；朴素 p；{len(preds)} 个协变量 × {len(outcomes)} 个结果 = {len(CR)} 次比较，未校正，只作描述）')
for on in outcomes:
    t = CR[CR.outcome == on].set_index('predictor')[['rho11', 'p11', 'rho9', 'p9']]
    P(f'\n### {on}')
    P(t.round(3).to_string())
nsig = (CR.p11 < 0.05).sum()
P(f'\n  p11<0.05 的格：{nsig}/{len(CR)}（纯噪声期望约 {0.05*len(CR):.1f}）；p9<0.05：{(CR.p9 < 0.05).sum()}/{len(CR)}')

# ---------------------------------------------------------------- 5. 国内 vs 跨境：AE 优势在迁移中增加还是缩小（LightGBM，划分 0）
P('\n## 7. 国内（tgt300，约 3 万目标池标签，缓冲格网划分 0）与跨境（k=0，划分 0）AE−S2 的对照（LightGBM）')
t = cov[['tgt300_ba_ae_minus_s2', 'k0s0_ba_ae_minus_s2', 'tgt300_oa_ae_minus_s2', 'k0s0_oa_ae_minus_s2']].copy()
t['transfer_minus_domestic_ba'] = t.k0s0_ba_ae_minus_s2 - t.tgt300_ba_ae_minus_s2
t['transfer_minus_domestic_oa'] = t.k0s0_oa_ae_minus_s2 - t.tgt300_oa_ae_minus_s2
P(t.round(2).to_string())
for col in t.columns:
    s = tsum(t[col]); s9 = tsum(t.loc[C9, col])
    P(f'  {col:30s} 11 国 {fmt(s)} | 9 国 {fmt(s9)}')
r = stats.spearmanr(t.tgt300_ba_ae_minus_s2, t.k0s0_ba_ae_minus_s2)
P(f'  国内 AE−S2 与跨境 AE−S2（BA）逐国 Spearman ρ={r[0]:+.2f}（p={r[1]:.2f}）')
# pairwise 国内测试（非缓冲，分层 20%）
s = tsum(cov.domestic_ba_ae_minus_s2)
P(f'  pairwise_v2 国内测试 BA AE−S2：{fmt(s)}')
s = tsum(cov.single_src_mean_ba_ae_minus_s2)
P(f'  pairwise_v2 单源跨国（各目标国对 10 个源的平均）BA AE−S2：{fmt(s)}')

# ---------------------------------------------------------------- 6. 学习曲线斜率：AE 与 S2 对目标标签的利用
P('\n## 8. SRS 学习曲线 k=0→400 增益：AE 分类器 − S2 分类器（即 Δ_feat(400) − Δ_feat(0)；逐国，四学习器）')
for m in ('ba_ge30', 'oa'):
    ch = pd.DataFrame({L_: W[f'{m}_k400_{L_}'] - W[f'{m}_k0_{L_}'] for L_ in LEARNERS})
    P(f'### {m}')
    P(ch.round(2).to_string())

# ---------------------------------------------------------------- 7. AE 邻域均值向量的范数（嵌入单位长度时，范数<1 反映 30 m 邻域内的异质性）
X = pd.read_parquet(f'{L}/data/dataset_v2.parquet', columns=['country', 'has_ae', 'has_s2'] + [f'ae_A{i:02d}_mean' for i in range(64)])
X = X[X.has_ae & X.has_s2]
nrm = np.sqrt((X[[f'ae_A{i:02d}_mean' for i in range(64)]].to_numpy(np.float64) ** 2).sum(1))
X['norm'] = nrm
q = X.groupby('country').norm.describe(percentiles=[0.1, 0.5, 0.9])
P('\n## 9. AE 30 m 邻域 mean 向量的 L2 范数（逐国分布；若原始像元为单位向量，则范数越低邻域越异质——单位长度的前提未在本项目内核实）')
P(q.round(3).to_string())
cov2 = cov.copy()
cov2['ae_mean_norm_median'] = q['50%']
r = stats.spearmanr(cov2.ae_mean_norm_median, outcomes['dBA_k0_mean4'])
P(f'  范数中位数 vs dBA_k0_mean4：ρ={r[0]:+.2f}（p={r[1]:.2f}，n=11）')
cov2.round(4).to_csv(f'{OUT}/country_covariates.csv')

with open(f'{OUT}/ae_s2_country_report.txt', 'w') as f:
    f.write('\n'.join(log) + '\n')
print('\n写出：', OUT)
