"""任务 B 汇总（事后探索，不属于预注册分析）。读取各 shard 的 CSV，与主试点的 foreign_only 等参照线比较。"""
import glob
import os
import sys

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = '/Users/perng_mini/AlphaCrops_local/results'
PRED = os.path.join(RES, 'preds')
SH = ['natural', '0.01', '0.05', '0.25', '1.0']
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 50)

d = pd.concat([pd.read_csv(f, dtype={'share': str}) for f in sorted(glob.glob(os.path.join(HERE, 'taskB_share_*_*.csv')))
               if not os.path.basename(f).endswith('_all.csv')], ignore_index=True)
rec = pd.concat([pd.read_csv(f, dtype={'share': str}) for f in sorted(glob.glob(os.path.join(HERE, 'taskB_recall_*_*.csv')))
                 if not os.path.basename(f).endswith('_all.csv')], ignore_index=True)
d.to_csv(os.path.join(HERE, 'taskB_share_all.csv'), index=False)
rec.to_csv(os.path.join(HERE, 'taskB_recall_all.csv'), index=False)
sup = d[d.target.isin(['PT', 'SK'])].copy()      # 补充检查（PT/SK，份额 0.01/0.05/1.0）
d = d[d.target.isin(['LV', 'SE'])].reset_index(drop=True)
rec = rec[rec.target.isin(['LV', 'SE'])].reset_index(drop=True)
print(f'训练次数 {len(d)}；各份额：{d.share.value_counts().to_dict()}')
assert not d.duplicated(['target', 'budget', 'seed', 'clf_feat', 'share']).any()

# ---------- 复现核对 ----------
r25 = d[d.share == '0.25']
print(f'share=0.25 复现：{len(r25)} 行；ba_ge30 最大绝对差 {np.abs(r25.ba_ge30 - r25.main_ba_ge30).max():.2e}；'
      f'ba 最大绝对差 {np.abs(r25.ba - r25.main_ba).max():.2e}；macro_f1 最大绝对差 {np.abs(r25.macro_f1 - r25.main_f1).max():.2e}；'
      f'逐地块不一致总数 {int(r25.pred_mismatch.sum())}')

main = pd.read_csv(os.path.join(RES, 'pilot_selection.csv'))
main = main[main.target.isin(['LV', 'SE'])]
ref = main[main.strategy.isin(['foreign_only', 'ceiling_w25', 'target_full', 'foreign+target_nat'])]
fo_seed = ref[ref.strategy == 'foreign_only'].set_index(['target', 'clf_feat', 'seed'])
refm = ref.groupby(['target', 'clf_feat', 'strategy'])['ba_ge30'].mean().unstack('strategy') * 100
print('\n参照线（ba_ge30 %，3 种子平均）与 foreign_only 的种子间 SD')
refm['fo_sd'] = ref[ref.strategy == 'foreign_only'].groupby(['target', 'clf_feat']).ba_ge30.std() * 100
print(refm.round(2).to_string())

# ---------- 表 1：各份额 ba_ge30 相对 foreign_only 的差 ----------
for m in ['ba_ge30', 'ba', 'macro_f1']:
    d[f'd_{m}'] = [100 * (row[m] - fo_seed.loc[(row.target, row.clf_feat, row.seed), m]) for _, row in d.iterrows()]
g = d.groupby(['target', 'budget', 'clf_feat', 'share'])
mean = g.d_ba_ge30.mean().unstack('share')[SH]
lo = g.d_ba_ge30.min().unstack('share')[SH]
hi = g.d_ba_ge30.max().unstack('share')[SH]
print('\n表 1：Δba_ge30 = 份额 s 的 ba_ge30 − foreign_only（pp，同种子配对，3 种子平均 [最小, 最大]）')
tab = mean.copy().astype(object)
for c in SH:
    tab[c] = [f'{a:+6.2f} [{b:+5.1f},{e:+5.1f}]' for a, b, e in zip(mean[c], lo[c], hi[c])]
print(tab.to_string())
print('\n自然份额（natural）的实际 eff_share（3 种子平均）：')
print(d[d.share == 'natural'].groupby(['target', 'budget', 'clf_feat']).eff_share.mean().unstack('clf_feat').round(4).to_string())

print('\n表 1b：按份额汇总（12 个 (目标国,k,特征) 单元）')
for k in [50, 200, 1000]:
    mk = mean.xs(k, level='budget')
    print(f'  k={k:4d}: ' + '  '.join(f'{s}: 均值{mk[s].mean():+.2f} 正向{int((mk[s] > 0).sum())}/{len(mk)}' for s in SH))
print('  k∈{50,200}: ' + '  '.join(
    f'{s}: 均值{mean.loc[mean.index.get_level_values("budget") <= 200, s].mean():+.2f} '
    f'正向{int((mean.loc[mean.index.get_level_values("budget") <= 200, s] > 0).sum())}/8' for s in SH))
# 种子级：正向次数
dd = d.set_index(['target', 'budget', 'clf_feat', 'seed', 'share']).d_ba_ge30.unstack('share')[SH]
print('\n种子级正向次数（每个 k 下 2 国 × 2 特征 × 3 种子 = 12 次）：')
for k in [50, 200, 1000]:
    x = dd.xs(k, level='budget')
    print(f'  k={k:4d}: ' + '  '.join(f'{s}:{int((x[s] > 0).sum())}/12' for s in SH))

print('\n每个单元的最佳份额：')
best = mean.idxmax(axis=1)
print(pd.concat([best.rename('best_share'), mean.max(axis=1).rename('best_Δ'), mean['0.25'].rename('Δ@0.25')], axis=1).round(2).to_string())

# ---------- 补回比例 ----------
print('\n表 2：补回比例 (BA − foreign_only)/(ceiling_w25 − foreign_only)，3 种子平均的 BA')
bm = g.ba_ge30.mean().unstack('share')[SH] * 100
fo = refm['foreign_only']; ce = refm['ceiling_w25']
cl = bm.copy()
for (t, k, cf), row in bm.iterrows():
    cl.loc[(t, k, cf)] = (row - fo[(t, cf)]) / (ce[(t, cf)] - fo[(t, cf)])
print(cl.map(lambda v: f'{v:+.0%}').to_string())

# ---------- 次要指标 ----------
for m in ['ba', 'macro_f1']:
    mm = g[f'd_{m}'].mean().unstack('share')[SH]
    print(f'\n次要指标 Δ{m}（pp，3 种子平均）')
    print(mm.round(2).to_string())

# ---------- 逐类 recall ----------
lab = pd.read_parquet('/Users/perng_mini/AlphaCrops_local/data/dataset_v2.parquet', columns=['label', 'has_ae', 'has_s2'])
lab = lab[lab.has_ae & lab.has_s2].reset_index(drop=True)
classes = sorted(lab['label'].unique())
y_all = lab['label'].map({c: i for i, c in enumerate(classes)}).to_numpy()
fo_rec = []
for t in ['LV', 'SE']:
    y = y_all[np.load(os.path.join(PRED, f'{t}_test_idx.npy'))]
    for cf in ['ae', 's2']:
        for s in range(3):
            p = np.load(os.path.join(PRED, f'{t}_foreign_only_0_{s}_{cf}.npy'))
            for c in np.unique(y):
                fo_rec.append(dict(target=t, clf_feat=cf, seed=s, cls=classes[c],
                                   fo_recall=float((p[y == c] == c).mean()), fo_npred=int((p == c).sum())))
fo_rec = pd.DataFrame(fo_rec)
rec = rec.merge(fo_rec, on=['target', 'clf_feat', 'seed', 'cls'], how='left', validate='many_to_one')
assert rec.fo_recall.notna().all()
rec['d_rec'] = 100 * (rec.recall - rec.fo_recall)
rec['d_npred'] = rec.n_pred - rec.fo_npred
big = rec[rec.n_test >= 30].copy()
nbig = big.groupby(['target']).cls.nunique()
print(f'\nba_ge30 覆盖的类别数：{nbig.to_dict()}')
# 核对：逐类 recall 的均值 = ba_ge30
chk = big.groupby(['target', 'budget', 'seed', 'clf_feat', 'share']).recall.mean().rename('x').reset_index().merge(
    d[['target', 'budget', 'seed', 'clf_feat', 'share', 'ba_ge30']])
print(f'核对：逐类 recall 平均与 ba_ge30 最大差 {np.abs(chk.x - chk.ba_ge30).max():.1e}')

for t in ['LV', 'SE']:
    b = big[big.target == t]
    print(f'\n表 3（{t}）：逐类 Δrecall（pp，相对 foreign_only，ae/s2 × 3 种子平均）；括号内为对 Δba_ge30 的贡献 = Δrecall/{nbig[t]}')
    x = b.groupby(['budget', 'share', 'cls']).d_rec.mean().unstack('cls')
    x = x.reindex(pd.MultiIndex.from_product([[50, 200, 1000], SH], names=['budget', 'share']))
    print(x.round(1).to_string())
    print(f'  foreign_only 各类 recall（%，ae/s2 × 3 种子平均）：' + '  '.join(
        f'{c}={v:.0f}' for c, v in (b.groupby('cls').fo_recall.mean() * 100).items()))
    print(f'  候选样本中各类平均个数（3 种子平均）：')
    ns = b[b.share == '0.25'].groupby(['budget', 'cls']).n_sel.mean().unstack('cls')
    print(ns.round(1).to_string())

print('\n表 3b：按特征分开（share=0.25 与 1.0，k=50/200），逐类 Δrecall（pp，3 种子平均）')
for t in ['LV', 'SE']:
    b = big[(big.target == t) & big.share.isin(['0.25', '1.0']) & (big.budget <= 200)]
    print(f'--- {t}')
    print(b.groupby(['clf_feat', 'budget', 'share', 'cls']).d_rec.mean().unstack('cls').round(1).to_string())

print('\n表 4：被抽中（n_sel>0）与未被抽中（n_sel=0）的类，Δrecall 的平均（pp，种子级 × 类，≥30 类）')
big['sampled'] = np.where(big.n_sel > 0, 'sampled', 'unsampled')
t4 = big.groupby(['target', 'budget', 'share', 'sampled']).d_rec.agg(['mean', 'count']).unstack('sampled')
t4 = t4.reindex(pd.MultiIndex.from_product([['LV', 'SE'], [50, 200, 1000], SH], names=['target', 'budget', 'share']))
print(t4.round(1).to_string())

print('\n表 4b：多数类（grassland/other_crops/winter_cereals）与少数类（其余 ≥30 类）的 Δrecall（pp）')
maj = {'grassland', 'other_crops', 'winter_cereals'}
big['grp'] = np.where(big.cls.isin(maj), 'majority', 'minority')
t4b = big.groupby(['target', 'budget', 'share', 'grp']).d_rec.mean().unstack('grp')
t4b = t4b.reindex(pd.MultiIndex.from_product([['LV', 'SE'], [50, 200, 1000], SH], names=['target', 'budget', 'share']))
print(t4b.round(1).to_string())

print('\n表 5：预测数的变化（相对 foreign_only；测试集全部类别；ae/s2 × 3 种子平均）——错误流向哪里')
allc = rec.copy()
for t in ['LV', 'SE']:
    b = allc[(allc.target == t) & allc.share.isin(['0.25', '1.0', '0.05']) & (allc.budget <= 200)]
    x = b.groupby(['budget', 'share', 'cls']).d_npred.mean().unstack('cls')
    print(f'--- {t}（测试集 {int(allc[(allc.target == t)].groupby("cls").n_test.first().sum())} 个地块）')
    print(x.round(0).to_string())

# ---------- 总体精度 OA（非预注册，只作诊断）----------
def oa_of(path, y):
    return float((np.load(path) == y).mean())
oa_rows = []
for t in ['LV', 'SE']:
    y = y_all[np.load(os.path.join(PRED, f'{t}_test_idx.npy'))]
    for cf in ['ae', 's2']:
        for sd in range(3):
            fo_oa = oa_of(os.path.join(PRED, f'{t}_foreign_only_0_{sd}_{cf}.npy'), y)
            for k in [50, 200, 1000]:
                for sh in SH:
                    oa = oa_of(os.path.join(HERE, 'preds_taskB', f'{t}_random_{k}_{sd}_{cf}_share{sh}.npy'), y)
                    oa_rows.append(dict(target=t, clf_feat=cf, seed=sd, budget=k, share=sh, d_oa=100 * (oa - fo_oa)))
oa = pd.DataFrame(oa_rows)
print('\n表 6：总体精度 OA 相对 foreign_only 的差（pp，ae/s2 × 3 种子平均）——多数类主导的指标')
print(oa.groupby(['target', 'budget', 'share']).d_oa.mean().unstack('share')[SH].round(2).to_string())

# ---------- 噪声与剂量-反应 ----------
from scipy.stats import wilcoxon  # noqa: E402
print('\n表 7：噪声参照')
print(f'  foreign_only 种子间 SD（ba_ge30，pp）：' + '  '.join(f'{t}-{cf}={v:.2f}' for (t, cf), v in refm['fo_sd'].items()))
nat = d[(d.share == 'natural') & (d.budget <= 200)].d_ba_ge30
print(f'  natural 份额（目标国权重≈0，k≤200，24 次）的 Δba_ge30：均值 {nat.mean():+.2f}，SD {nat.std():.2f}，'
      f'范围 [{nat.min():+.2f}, {nat.max():+.2f}] → 同种子配对下仅"加入极小扰动"就有这么大的波动')
print('\n表 8：每个 k 下，各份额相对 natural 的配对差（Δ_s − Δ_natural，12 次 = 2 国 × 2 特征 × 3 种子），Wilcoxon（探索性，未校正）')
for k in [50, 200, 1000]:
    x = dd.xs(k, level='budget')
    out = []
    for sh in ['0.01', '0.05', '0.25', '1.0']:
        diff = x[sh] - x['natural']
        p = wilcoxon(diff).pvalue
        out.append(f'{sh}: {diff.mean():+.2f} (负向 {int((diff < 0).sum())}/12, p={p:.3f})')
    print(f'  k={k:4d}: ' + '  '.join(out))
print('\n表 8b：剂量-反应（每个 (目标国,特征,种子) 内 Δ 与份额次序 natural<0.01<0.05<0.25<1.0 的 Spearman ρ 的平均）')
from scipy.stats import spearmanr  # noqa: E402
for k in [50, 200, 1000]:
    x = dd.xs(k, level='budget')
    rhos = x.apply(lambda r: spearmanr(range(5), r.values)[0], axis=1)
    print(f'  k={k:4d}: 平均 ρ={rhos.mean():+.2f}；按国家 ' + '  '.join(
        f'{t}={rhos.xs(t, level="target").mean():+.2f}' for t in ['LV', 'SE']) + f'；ρ<0 的比例 {(rhos < 0).mean():.0%}')
print('\n表 9：每个目标国样本相对外国样本的平均权重倍数 w_ratio（= share × 50000 / k），与 Δba_ge30 均值（4 单元 × 3 种子）')
wr = d.groupby(['budget', 'share']).agg(w_ratio=('w_ratio', 'mean'), d=('d_ba_ge30', 'mean')).reset_index()
wr = wr.sort_values('w_ratio')
print(wr.round(3).to_string(index=False))

# ---------- 补充检查：PT/SK ----------
if len(sup):
    print('\n' + '=' * 30 + ' 补充检查 PT/SK（份额 0.01/0.05/1.0；0.25 取主试点 random 行）' + '=' * 30)
    mainall = pd.read_csv(os.path.join(RES, 'pilot_selection.csv'))
    fo2 = mainall[mainall.strategy == 'foreign_only'].set_index(['target', 'clf_feat', 'seed']).ba_ge30
    r25 = mainall[(mainall.strategy == 'random') & mainall.target.isin(['PT', 'SK'])].copy()
    r25['share'] = '0.25'
    z = pd.concat([sup[['target', 'budget', 'seed', 'clf_feat', 'share', 'ba_ge30']],
                   r25[['target', 'budget', 'seed', 'clf_feat', 'share', 'ba_ge30']]], ignore_index=True)
    z['d'] = [100 * (r.ba_ge30 - fo2[(r.target, r.clf_feat, r.seed)]) for _, r in z.iterrows()]
    print(f'补充检查训练次数 {len(sup)}')
    print(z.groupby(['target', 'budget', 'clf_feat', 'share']).d.mean().unstack('share')[['0.01', '0.05', '0.25', '1.0']].round(2).to_string())
