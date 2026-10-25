"""
D2 独立复核（不 import cheng_metrics.py；只读 results/guidance_G/，不读 guidance_G_ml/；不训练任何模型）。
- k=0：11 国 × 2 划分 × 2 特征，用 sklearn 的 accuracy_score / balanced_accuracy_score 与"把非 4 类列置 −1"的
  argmax 独立重算 oa、ba_all、ba_ge30、oa4、ba4；与 cheng_metrics_rows.csv 逐项比较。
- 另用温度校准后的 P_T（S['P_T_{f}'][n_cand:]）重算，看 4 类/10 类 argmax 是否随概率来源改变。
- SRS k=200：DE、PT、SE 三国（2 划分 × 2 重复）独立重算。
- 国家层 AE−S2 与 t 推断（scipy.stats 独立计算）。
- 全国（非测试集）类别计数，用于核对"稀疏"说法。
"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix  # noqa: E402

warnings.filterwarnings('ignore')
L = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, L)
import guidance_G as G  # noqa: E402  只用 GData 与 OUT

GOUT = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
assert os.path.realpath(G.OUT) == os.path.realpath(GOUT), G.OUT
assert 'guidance_G_ml' not in os.path.realpath(GOUT).split(os.sep)
HERE = os.path.dirname(os.path.abspath(__file__))
D2 = '/Users/perng_mini/AlphaCrops_local/verify/storyline/cheng_comparable/cheng_metrics_rows.csv'
CT = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']
C4 = [0, 1, 4, 9]
NON4 = [c for c in range(10) if c not in C4]


def metrics(P, y):
    P = np.asarray(P, np.float64)
    P = P / P.sum(1, keepdims=True)
    ph = P.argmax(1)
    cm = confusion_matrix(y, ph, labels=range(10))
    n = cm.sum(1)
    rec = np.divide(np.diag(cm), n, out=np.full(10, np.nan), where=n > 0)
    m = np.isin(y, C4)
    Q = P[m].copy()
    Q[:, NON4] = -1.0
    ph4 = Q.argmax(1)
    y4 = y[m]
    return dict(oa=100 * accuracy_score(y, ph), ba_all=100 * balanced_accuracy_score(y, ph),
                ba_ge30=100 * np.nanmean(rec[n >= 30]),
                oa4=100 * accuracy_score(y4, ph4), ba4=100 * balanced_accuracy_score(y4, ph4),
                n4=int(m.sum()), ph=ph, ph4=ph4,
                ties10=int((np.sort(P, 1)[:, -1] == np.sort(P, 1)[:, -2]).sum()),
                ties4=int((np.sort(P[m][:, C4], 1)[:, -1] == np.sort(P[m][:, C4], 1)[:, -2]).sum()))


def tsum(v):
    v = np.asarray(v, float)
    n = len(v)
    se = v.std(ddof=1) / np.sqrt(n)
    q = stats.t.ppf(0.975, n - 1)
    return v.mean(), v.mean() - q * se, v.mean() + q * se, stats.ttest_1samp(v, 0).pvalue, int((v > 0).sum()), n


def main():
    D = G.GData()
    R2 = pd.read_csv(D2)
    out, alt = [], []
    for t in CT:
        for s in (0, 1):
            S = np.load(os.path.join(GOUT, 'stage0', f'{t}_s{s}.npz'))
            test, cand = S['test'], S['cand']
            assert np.array_equal(S['allT'][len(cand):], test)
            y = np.asarray(D.y[test])
            for f in ('ae', 's2'):
                its = sorted(int(k.split('_i')[-1]) for k in S.files if k.startswith(f'P0test_{f}_i'))
                P = S[f'P0test_{f}_i{its[-1]}']
                assert P.dtype == np.float16 and P.shape == (len(y), 10)
                m = metrics(P.astype(np.float32), y)
                PT = S[f'P_T_{f}'][len(cand):]
                mt = metrics(PT, y)
                alt.append(dict(target=t, split=s, feat=f, its=str(its),
                                d_argmax10=int((m['ph'] != mt['ph']).sum()),
                                d_argmax4=int((m['ph4'] != mt['ph4']).sum()),
                                d_oa4=mt['oa4'] - m['oa4'], d_ba4=mt['ba4'] - m['ba4'], d_ba_all=mt['ba_all'] - m['ba_all']))
                out.append(dict(k=0, target=t, split=s, rep=-1, feat=f,
                                **{k: v for k, v in m.items() if k not in ('ph', 'ph4')}))
    # SRS k=200：三国
    for t in ('DE', 'PT', 'SE'):
        for s in (0, 1):
            S = np.load(os.path.join(GOUT, 'stage0', f'{t}_s{s}.npz'))
            y = np.asarray(D.y[S['test']])
            for r in (0, 1):
                rows = pd.read_csv(os.path.join(GOUT, f'rows_t1_{t}_s{s}_r{r}.csv'))
                for f in ('ae', 's2'):
                    rr = rows[(rows.method == 'srs') & (rows.k == 200) & (rows.clf == f)]
                    hs = rr.sel_hash.unique()
                    assert len(hs) == 1, (t, s, r, f, hs)
                    z = np.load(os.path.join(GOUT, 'probs', f'{t}_s{s}_r{r}_{f}_k200_{hs[0]}.npz'))
                    it = max(int(k[1:]) for k in z.files)
                    m = metrics(z[f'i{it}'], y)
                    assert abs(m['oa'] - rr.oa.iloc[0]) < 1e-9 and abs(m['ba_ge30'] - rr.ba_ge30.iloc[0]) < 1e-9
                    out.append(dict(k=200, target=t, split=s, rep=r, feat=f,
                                    **{k: v for k, v in m.items() if k not in ('ph', 'ph4')}))
    O = pd.DataFrame(out)
    A = pd.DataFrame(alt)
    O.to_csv(os.path.join(HERE, 'indep_rows.csv'), index=False)
    A.to_csv(os.path.join(HERE, 'indep_calibrated_vs_raw.csv'), index=False)

    # 与 D2 逐项比较
    key = ['k', 'target', 'split', 'rep', 'feat']
    M = O.merge(R2[R2.learner == 'lgbm'], on=key, suffixes=('', '_d2'))
    print(f'合并行数 {len(M)}（k=0 {int((M.k == 0).sum())}，k=200 {int((M.k == 200).sum())}）')
    for c in ('oa', 'ba_all', 'ba_ge30', 'oa4', 'ba4', 'ties10', 'ties4'):
        print(f'  {c:<8} 最大绝对差 {np.abs(M[c] - M[c + "_d2"]).max():.3e}')
    print(f'  n4 与 n_test4 一致：{(M.n4 == M.n_test4).all()}')
    k0 = M[M.k == 0]
    print(f'k=0 并列：10 类 {int(k0.ties10.sum())}（最大单行 {int(k0.ties10.max())}，占比 '
          f'{100 * (k0.ties10 / k0.n_test).max():.3f}%），4 类 {int(k0.ties4.sum())}；'
          f'k=200（全部 132 行由 D2）：10 类 {int(R2[R2.k == 200].ties10.sum())}，4 类 {int(R2[R2.k == 200].ties4.sum())}')
    print(f'温度校准概率与原始 P0test 的 argmax 差异：10 类合计 {A.d_argmax10.sum()}，4 类合计 {A.d_argmax4.sum()}；'
          f'|Δoa4| 最大 {A.d_oa4.abs().max():.4f}，|Δba4| 最大 {A.d_ba4.abs().max():.4f}，|Δba_all| 最大 {A.d_ba_all.abs().max():.4f}')
    print('迭代轮数键：', A.its.unique())

    # 国家层
    for k in (0, 200):
        sub = O[O.k == k]
        cts = sorted(sub.target.unique())
        print(f'\n== k={k}，国家层 AE−S2（{len(cts)} 国，独立重算） ==')
        for c in ('ba_all', 'oa4', 'ba4', 'oa', 'ba_ge30'):
            w = sub.pivot_table(index=['target', 'split', 'rep'], columns='feat', values=c)
            d = (w.ae - w.s2).groupby('target').mean()
            line = ' '.join(f'{t}{d[t]:+.2f}' for t in cts)
            if k == 0:
                a = tsum(d.values)
                b = tsum(d.drop(['RO', 'PT']).values)
                e = tsum(d.drop(['LT', 'LV', 'RO']).values)
                print(f'{c:<8} 11 国 {a[0]:+.2f} [{a[1]:+.2f},{a[2]:+.2f}] p={a[3]:.3f} {a[4]}/{a[5]} | '
                      f'去 RO,PT {b[0]:+.2f} [{b[1]:+.2f},{b[2]:+.2f}] {b[4]}/{b[5]} | '
                      f'去 LT,LV,RO {e[0]:+.2f} [{e[1]:+.2f},{e[2]:+.2f}] {e[4]}/{e[5]}')
                lv = w.groupby('target').mean().mean()
                print(f'         水平 AE {lv.ae:.2f} S2 {lv.s2:.2f}')
            print(f'         逐国 {line}')
        if k == 0:
            w = sub.pivot_table(index=['target', 'split'], columns='feat', values='ba4')
            d = (w.ae - w.s2).groupby('target').mean()
            print(f'ba4 贡献分解：11 国总和 {d.sum():+.2f}；LT+LV+RO {d[["LT", "LV", "RO"]].sum():+.2f}；'
                  f'DE {d["DE"]:+.2f}；PT {d["PT"]:+.2f}；其余 6 国 {d.drop(["LT", "LV", "RO", "DE", "PT"]).sum():+.2f}')

    # 全国计数（原始与 make_split 同样去重后）
    print('\n== 全国类别计数（raw / 去重） ==')
    cls = list(D.classes)
    for t in CT:
        idx = np.flatnonzero(D.ctry == t)
        keep = ~pd.DataFrame({'a': D.lon[idx], 'b': D.lat[idx], 'c': D.area[idx]}).duplicated().to_numpy()
        raw = np.bincount(D.y[idx], minlength=10)
        dd = np.bincount(D.y[idx[keep]], minlength=10)
        print(t, ' '.join(f'{cls[c][:6]}={raw[c]}/{dd[c]}' for c in C4),
              '| 出现类', int((raw > 0).sum()), '缺:', ','.join(cls[c] for c in range(10) if raw[c] == 0))


if __name__ == '__main__':
    main()
