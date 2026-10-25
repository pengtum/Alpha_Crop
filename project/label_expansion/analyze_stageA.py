"""
阶段 A 分析（ANALYSIS_PLAN_stageA.md）。

用法：
  python analyze_stageA.py rgrid            # 选 r*(k)：SRS 在开发国上的 ba_ge30 均值最大者
  python analyze_stageA.py run [--noise]    # 各方法相对 SRS 的配对差、晋级与无效判定、三档先验处理、构成、E_far
--noise：把每行指标换成随机数（只用于在不看结果的情况下测试流程）。
输出：RESULTS_DIR/stageA/report_{rgrid|run}.txt
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

from stageA import OUT, DEV, BUDGETS, MIN_N, class_weights_from, km_xy
from check3_priors import em_prior

NOISE = '--noise' in sys.argv
# 期中查看：--targets PT[,BE] 只分析这些国家；--tag 为报告文件名加后缀（不覆盖正式报告）
TARGETS = next((a.split('=')[1].split(',') for a in sys.argv if a.startswith('--targets=')), None)
TAG = next((a.split('=')[1] for a in sys.argv if a.startswith('--tag=')), '')
COMPLETE_ONLY = '--complete-only' in sys.argv  # 只保留全部方法都已完成的单元


class Tee:
    def __init__(self, path):
        self.f = open(path, 'w')

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        self.f.flush()


def load_rows(mode):
    fs = glob.glob(os.path.join(OUT, f'rows_{mode}_*.csv'))
    R = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    if TARGETS:
        R = R[R.target.isin(TARGETS)]
    if COMPLETE_ONLY:
        n = R.groupby(['target', 'split', 'rep']).size()
        full = n[n == n.max()].index
        R = R.set_index(['target', 'split', 'rep']).loc[full].reset_index()
    if NOISE:
        rng = np.random.default_rng(0)
        for m in ('oa', 'ba_ge30', 'f1_ge30'):
            R[m] = rng.normal(50, 5, len(R))
    return R


def boot_ci(d, strata, B=5000, level=0.90, seed=0):
    """按国家分层、在单元层面重抽的 bootstrap（每国内重抽其单元）。"""
    rng = np.random.default_rng(seed)
    d, strata = np.asarray(d), np.asarray(strata)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    m = [np.mean(np.concatenate([d[g][rng.integers(0, len(g), len(g))] for g in groups])) for _ in range(B)]
    a = (1 - level) / 2
    return np.quantile(m, [a, 1 - a])


def rgrid():
    sys.stdout = Tee(os.path.join(OUT, 'report_rgrid.txt'))
    R = load_rows('rgrid')
    R = R[R.method == 'srs_rgrid']
    print(f'行数 {len(R)}；单元 {R.groupby(["target", "split", "rep"]).ngroups}')
    for k in BUDGETS:
        t = R[R.k == k].groupby(['r', 'target'])[['ba_ge30', 'oa']].mean().unstack('target')
        t['ba_ge30_mean'] = R[R.k == k].groupby('r').ba_ge30.mean()
        t['oa_mean'] = R[R.k == k].groupby('r').oa.mean()
        print(f'\nk={k}：\n{t.round(2).to_string()}')
        best = t['ba_ge30_mean'].idxmax()
        print(f'  → r*({k}) = {best}（按 ba_ge30 均值；OA 同时列出供参考）')


def prior_correct(p, pi_new, pi_train):
    w = np.divide(pi_new, pi_train, out=np.zeros_like(pi_new), where=pi_train > 0)
    q = p * w
    s = q.sum(1, keepdims=True)
    return np.where(s > 0, q / np.where(s > 0, s, 1), p)


def run():
    sys.stdout = Tee(os.path.join(OUT, ('report_run_NOISE' if NOISE else 'report_run') + (f'_{TAG}' if TAG else '') + '.txt'))
    if NOISE:
        print('!!! 噪声模式，结果无意义 !!!')
    if TAG:
        print(f'!!! 期中查看（{TAG}）：只含部分国家/单元，晋级判定以全部开发国完成后的正式分析为准 !!!')
    from stageA import Data
    D = Data()
    R = load_rows('run')
    unit = ['target', 'split', 'rep']
    print(f'行数 {len(R)}；方法 {R.method.nunique()}；单元 {R.groupby(unit).ngroups}；r 取值 {sorted(R.r.unique())}')

    # ---------- 三档先验处理：从保存的概率与选样重算
    extra = []
    for (tgt, split, rep), g in R.groupby(unit):
        S = np.load(os.path.join(OUT, 'stage0', f'{tgt}_s{split}.npz'))
        test, fr = S['test'], S['foreign']
        yT = D.y[test]
        big = np.flatnonzero(np.bincount(yT, minlength=10) >= MIN_N)
        pool = np.load(os.path.join(OUT, 'pool', f'{tgt}_s{split}_r{rep}.npy'))
        pi_pool = np.bincount(D.y[pool], minlength=10) / len(pool)
        cw = class_weights_from(D.y[fr])
        lat0 = float(S['lat0'])
        xy_test = km_xy(D.lon[test], D.lat[test], lat0)
        for _, row in g.iterrows():
            tag = f'{tgt}_s{split}_r{rep}_{row.method}_k{row.k}_R{row.r:g}'
            if not os.path.exists(os.path.join(OUT, 'probs', tag + '.npy')):
                tag = f'{tgt}_s{split}_r{rep}_{row.method}_k{row.k}_R{float(row.r)}'  # 运行器用浮点 r 命名（R50.0）
            P = np.load(os.path.join(OUT, 'probs', tag + '.npy')).astype(np.float64)
            P /= P.sum(1, keepdims=True)
            z = np.load(os.path.join(OUT, 'sel', tag + '.npz'))
            sel = z['sel']
            ysel = D.y[sel]
            w = np.r_[cw[D.y[fr]], row.r * cw[ysel]]
            yy = np.r_[D.y[fr], ysel]
            pi_train = np.bincount(yy, weights=w, minlength=10) / w.sum()
            if len(z['pi']):
                ipw = 1 / z['pi']
                pi_des = np.bincount(ysel, weights=ipw, minlength=10) / ipw.sum()  # Hájek
                src = 'design'
            else:
                pi_des = em_prior(P, pi_train)[0]  # 零标签 EM（在测试集的无标签概率上）
                src = 'em'
            pi_naive = np.bincount(ysel, minlength=10) / len(ysel)
            out = {}
            for tag2, pin in (('P1', pi_des), ('P2', pi_pool), ('Pnaive', pi_naive)):
                ph = prior_correct(P, pin, pi_train).argmax(1)
                rec = np.array([(ph[yT == c] == c).mean() if (yT == c).any() else np.nan for c in range(10)])
                out[f'oa_{tag2}'] = 100 * (ph == yT).mean()
                out[f'ba_{tag2}'] = 100 * np.mean(rec[big])
            d_sel, _ = __import__('scipy.spatial', fromlist=['cKDTree']).cKDTree(
                km_xy(D.lon[sel], D.lat[sel], lat0)).query(xy_test)
            far = d_sel > 5
            ph0 = P.argmax(1)
            recf = np.array([(ph0[far & (yT == c)] == c).mean() if (far & (yT == c)).sum() else np.nan
                             for c in range(10)])
            extra.append(dict(target=tgt, split=split, rep=rep, method=row.method, k=row.k, r=row.r,
                              prior_src=src, prior_L1=np.abs(pi_des - pi_pool).sum(),
                              comp_L1=np.abs(pi_naive - pi_pool).sum(), far_frac=far.mean(),
                              ba_far=100 * np.nanmean(recf[big]), oa_far=100 * (ph0[far] == yT[far]).mean(), **out))
    E = pd.DataFrame(extra)
    if NOISE:
        for c in [c for c in E.columns if c.startswith(('oa_', 'ba_'))]:
            E[c] = np.random.default_rng(1).normal(50, 5, len(E))
    R = R.merge(E, on=unit + ['method', 'k', 'r'])
    R.to_csv(os.path.join(OUT, 'stageA_merged' + (f'_{TAG}' if TAG else '') + '.csv'), index=False)

    # ---------- 各方法相对 SRS 的配对差
    base = R[R.method == 'srs'].set_index(unit + ['k'])
    print('\n' + '=' * 25 + ' 主结果：Δ(方法 − SRS)，P0，ba_ge30 与 OA（27 个单元平均）' + '=' * 25)
    summ = []
    for (m, k), g in R.groupby(['method', 'k']):
        if m == 'srs':
            continue
        g = g.set_index(unit + ['k'])
        b = base.loc[g.index]
        row = dict(method=m, k=k, n=len(g))
        for met, col in (('ba', 'ba_ge30'), ('oa', 'oa'), ('ba_P1', 'ba_P1'), ('ba_P2', 'ba_P2'),
                         ('oa_P1', 'oa_P1'), ('oa_P2', 'oa_P2'), ('ba_far', 'ba_far')):
            d = (g[col] - b[col]).to_numpy()
            lo, hi = boot_ci(d, g.index.get_level_values('target'))
            row[f'd_{met}'] = d.mean()
            row[f'lo_{met}'], row[f'hi_{met}'] = lo, hi
            if met == 'ba':
                row['pos_countries'] = int((pd.Series(d, index=g.index.get_level_values('target'))
                                            .groupby(level=0).mean() > 0).sum())
        row['comp_L1'] = g['comp_L1'].mean()
        row['n_classes'] = g['n_classes_sel'].mean()
        summ.append(row)
    Sm = pd.DataFrame(summ)
    for k in BUDGETS:
        t = Sm[Sm.k == k].sort_values('d_ba', ascending=False)
        print(f'\n--- k={k} ---')
        print(t[['method', 'n', 'd_ba', 'lo_ba', 'hi_ba', 'pos_countries', 'd_oa', 'lo_oa', 'd_ba_P1', 'd_oa_P1',
                 'd_ba_P2', 'd_oa_P2', 'd_ba_far', 'comp_L1', 'n_classes']].round(2).to_string(index=False))
    # ---------- 晋级与无效
    print('\n' + '=' * 25 + ' 晋级规则（预注册）' + '=' * 25)
    margin = Sm[Sm.method == 'margin'].set_index('k')['d_ba'] if 'margin' in set(Sm.method) else None
    prom = []
    for m, g in Sm[~Sm.method.str.startswith('oracle')].groupby('method'):
        g = g.set_index('k')
        crit = any(g.loc[k, 'd_ba'] >= 0.2 or (margin is not None and g.loc[k, 'd_ba'] >= margin.get(k, np.inf))
                   for k in g.index)
        futile = all(g.loc[k, 'hi_ba'] < 0.2 for k in g.index)
        prom.append(dict(method=m, mean_d=g['d_ba'].mean(), meets=crit, futile=futile))
    P = pd.DataFrame(prom).sort_values('mean_d', ascending=False)
    P['promote'] = P.meets & ~P.futile
    uncond = {'srs', 'lpm2', 'predstrat', 'margin', 'gist_margin'}  # M1–M4 与 GIST（研究者点名）
    chosen = P[P.promote & ~P.method.isin(uncond)].head(6).method.tolist()
    print(P.round(2).to_string(index=False))
    print(f'\n无条件进入阶段 B：{sorted(uncond & set(R.method))}；按规则晋级（≤6）：{chosen}')
    # ---------- 神谕与转向规则
    print('\n' + '=' * 25 + ' 神谕（只用于解读提升空间）' + '=' * 25)
    o = Sm[Sm.method.str.startswith('oracle')]
    print(o[['method', 'k', 'd_ba', 'd_oa', 'd_ba_P1', 'd_oa_P1', 'd_ba_P2', 'd_oa_P2']].round(2).to_string(index=False))
    srsP1 = base.groupby('k')[['ba_P1']].mean()
    for k in BUDGETS:
        if 'oracle_prop' in set(R.method):
            u1 = R[(R.method == 'oracle_prop') & (R.k == k)].ba_ge30.mean()
            print(f'  k={k}：U1a − (SRS+P1) = {u1 - srsP1.loc[k, "ba_P1"]:+.2f} pp（转向规则阈值 0.3）')


if __name__ == '__main__':
    {'rgrid': rgrid, 'run': run}[sys.argv[1]]()
