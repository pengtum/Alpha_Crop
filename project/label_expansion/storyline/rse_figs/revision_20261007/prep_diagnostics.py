"""新增描述性分析：空间与标签分布、零标签预测、单源两两迁移；不训练模型。"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '2'
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from pyproj import Transformer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from common import DATA_DIR, RESULTS_DIR, COUNTRIES

OUT = HERE / 'source_data'
OUT.mkdir(exist_ok=True)
CONF = Path(RESULTS_DIR) / 'confirm_v1'
inputs = []

def register(p):
    p = Path(p)
    with p.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    inputs.append(dict(file=p.name, sha256=digest))
    return p

def save(d, name):
    d.to_csv(OUT / name, index=False)

def main():
    p = register(Path(DATA_DIR) / 'dataset_v2.parquet')
    df = pd.read_parquet(p, columns=['country', 'label', 'longitude', 'latitude', 'has_ae', 'has_s2'])
    df = df[df.has_ae & df.has_s2].reset_index(drop=True)
    classes = sorted(df.label.unique())
    y = pd.Categorical(df.label, categories=classes).codes
    # 与正式运行缓存逐行比对，防止概率与标签错位。
    cache = Path(DATA_DIR) / 'cache_guidance_G'
    assert np.array_equal(y, np.load(cache / 'y.npy'))
    assert np.array_equal(df.country, np.load(cache / 'ctry.npy', allow_pickle=True))
    assert np.array_equal(df.longitude, np.load(cache / 'lon.npy'))
    counts = pd.crosstab(df.country, df.label).reindex(index=COUNTRIES, columns=classes, fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0) * 100
    save(counts.rename_axis('country').reset_index(), 'class_counts.csv')
    save(shares.rename_axis('country').reset_index(), 'class_shares.csv')
    summaries = pd.DataFrame(dict(country=COUNTRIES, n=counts.sum(axis=1).values,
        present=(counts > 0).sum(axis=1).values, majority_pct=shares.max(axis=1).values,
        majority=shares.idxmax(axis=1).values, rare_lt_30=((counts > 0) & (counts < 30)).sum(axis=1).values))
    save(summaries, 'country_support.csv')
    tx = Transformer.from_crs(4326, 3035, always_xy=True)
    x, yy = tx.transform(df.longitude.to_numpy(), df.latitude.to_numpy())
    locations = pd.DataFrame(dict(country=df.country, x=x, y=yy)).groupby('country', as_index=False)[['x','y']].median()
    save(locations, 'country_label_positions.csv')
    grid = pd.DataFrame(dict(x=np.floor(x / 10000).astype(int), y=np.floor(yy / 10000).astype(int)))
    grid = grid.value_counts().rename('n').reset_index()
    grid['x'] = (grid.x + .5) * 10000
    grid['y'] = (grid.y + .5) * 10000
    assert grid.n.sum() == len(df)
    save(grid, 'spatial_density.csv')
    unit_rows = pd.read_csv(register(CONF / 'analysis/unit_rows.csv'))
    metrics, confusions, test_idx, errors = [], [], [], []
    for cc in COUNTRIES:
        for half in ('F1', 'F2'):
            unit = f'{cc}_{half}'
            split = np.load(register(CONF / 'splits' / f'{unit}.npz'))
            idx, big = split['test'], split['big']
            assert (df.country.iloc[idx] == cc).all()
            test_idx.extend(idx.tolist())
            yt = y[idx]
            for feat in ('ae', 's2'):
                P = np.load(register(CONF / 'probs' / unit / f'lgb_Z0_{feat}_r0.npz'))['P_test']
                assert P.shape == (len(idx), len(classes)) and np.isfinite(P).all()
                pred = P.argmax(axis=1)
                cm = np.bincount(yt * 10 + pred, minlength=100).reshape(10, 10)
                support = cm.sum(axis=1)
                recalls = np.divide(cm.diagonal(), support, out=np.full(10, np.nan), where=support > 0)
                oa, ba = 100 * (yt == pred).mean(), 100 * recalls[big].mean()
                ref = unit_rows[(unit_rows.unit == unit) & (unit_rows.learner == 'lgb') &
                    (unit_rows.feat == feat) & (unit_rows.level == 'arm') & (unit_rows.name == 'Z0')]
                assert len(ref) == 1
                errors.extend([abs(oa - ref.iloc[0].oa_t0), abs(ba - ref.iloc[0].ba_t0)])
                metrics.append(dict(country=cc, half=half, feat=feat, n_test=len(idx), n_ba=len(big), OA=oa, BA=ba))
                for i, true in enumerate(classes):
                    for j, predicted in enumerate(classes):
                        confusions.append(dict(country=cc, half=half, feat=feat, true=true, predicted=predicted,
                            count=int(cm[i,j]), support=int(support[i]), included_ba=bool(i in big),
                            row_pct=100 * cm[i,j] / support[i] if support[i] else np.nan))
    assert len(test_idx) == len(set(test_idx)), '测试半区不应重叠'
    assert max(errors) < 1e-9, max(errors)
    met = pd.DataFrame(metrics)
    save(met, 'zero_label_units.csv')
    save(met.groupby(['country','feat'], as_index=False)[['OA','BA']].mean(), 'zero_label_countries.csv')
    cm = pd.DataFrame(confusions)
    save(cm, 'confusion_units.csv')
    # 概览：每个真实类别先在合格半区内平均，再按国家等权平均；不把缺席类填为零。
    eligible = cm[cm.included_ba]
    country_cm = eligible.groupby(['country','feat','true','predicted'], as_index=False).row_pct.mean()
    overview = country_cm.groupby(['feat','true','predicted']).row_pct.agg(['mean','count']).reset_index()
    save(overview, 'confusion_country_equal.csv')
    # 逐国混淆矩阵保留所有出现的类；两半合并计数，稀有类显示支持数。
    raw = cm.groupby(['country','feat','true','predicted'], as_index=False)['count'].sum()
    raw['support'] = raw.groupby(['country','feat','true'])['count'].transform('sum')
    raw['row_pct'] = 100 * raw['count'] / raw.support.replace(0, np.nan)
    save(raw, 'confusion_countries.csv')
    pairs = pd.read_csv(register(Path(RESULTS_DIR) / 'pairwise_v2.csv'))
    pairs = pairs[pairs.feature.isin(['ae','s2']) & (pairs.source != pairs.target)].copy()
    assert len(pairs) == 220 and not pairs.duplicated(['source','target','feature']).any()
    assert all(r.n_test == counts.loc[r.target].sum() for r in pairs.itertuples())
    pairs['BA'] = 100 * pairs.ba
    pairs['unseen_target_pct'] = [float(shares.loc[r.target, counts.loc[r.source] == 0].sum()) for r in pairs.itertuples()]
    save(pairs, 'pairwise_transfer.csv')
    diff = pairs.pivot(index=['source','target'], columns='feature', values='BA')
    summary = dict(n_parcels=len(df), n_test_unique=len(test_idx), absent_country_class=int((counts==0).sum().sum()),
        rare_country_class_lt30=int(((counts>0)&(counts<30)).sum().sum()),
        majority_min=float(shares.max(axis=1).min()), majority_max=float(shares.max(axis=1).max()),
        pooled_class_counts={k:int(v) for k,v in counts.sum().items()},
        pairwise_mean_ae=float(diff.ae.mean()), pairwise_mean_s2=float(diff.s2.mean()),
        pairwise_ae_wins=int((diff.ae > diff.s2).sum()), pairwise_n=len(diff),
        zero_label_means=met.groupby('feat')[['OA','BA']].mean().round(6).to_dict('index'),
        reconciliation_max_abs_error=max(errors), inputs=inputs)
    (OUT / 'audit.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k != 'inputs'},indent=2))

if __name__ == '__main__':
    main()
