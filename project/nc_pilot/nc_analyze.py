"""
NC 试点一次性分析（方案第 6、7 节）。度量函数 import 自 RSE 已冻结的 analyze_confirm（其导入链会核对冻结文件哈希、
chdir 到 label_expansion；本脚本只用绝对路径）。

两个任务相同：同 OA 下的 BA。
  曲线      AC.unit_curves(yT, BA 类别, P)：τ = −2..2（81 点），ℓ = log P − τ·log π̂
  OA★       同一特征 Z0 在 τ = 0 的 OA；窗口 OA★ + [−4, 0] 步长 0.1（41 点）；包络 step0B.envelope
  两臂之差  AC.paired：共同可达格点上逐点 BA 差的平均；共同可达 < 50% 记缺失
  Δ_R       Z0 下 AE − S2，锚点 min(OA★_ae, OA★_s2)
  Δ_L(k)    同一特征 RT_k − Z0 → 对有值的抽样平均 → 两特征平均（任一缺失则缺失）；Δ_L(全) 同理
  I(k)      AE 的 Δ_L(k) − S2 的 Δ_L(k)；k ∈ {50,100,200,400,全}
  m         对有值的域平均；有值的域 < 2 → 缺失
  判定      任务成立 ⇔ m_R > 0 且 m_R ≥ m_L(200)；缺失 → 不成立
  L_AE      AC.k_eq(m_R, [0,50,100,200,400], AC.pav([0, m_L(50..400)]))，只用五个量全有值的域；< 2 个域 → 不可计算
输出 {out}/：domain_values.csv、arm_rows.csv、task_summary.csv、decision.txt、dose_curve.{png,pdf}

用法：python nc_analyze.py [--root DIR] [--out DIR] [--smoke]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SELF_PATH = os.path.abspath(__file__)
sys.path.insert(0, HERE)
import nc_common as K  # noqa: E402
sys.path.insert(0, '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/'
                   'label_expansion/confirm')
import analyze_confirm as AC  # noqa: E402  （导入时 chdir；之后只用绝对路径）

B = AC.B
ID = {'T1': 'row', 'T2': 'id'}
DOM = {'T1': 'city', 'T2': 'nuts0'}
KN = np.array([0, 50, 100, 200, 400], dtype=float)


def nanmean_or_nan(v, min_n=1):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) >= min_n else np.nan


def domain_rows(task, d, root, ba, S):
    pdir = os.path.join(root, 'probs', task, d)
    te = S[(S[DOM[task]] == d) & (S.role == 'test')].sort_values(ID[task])
    P, sid0 = {}, None
    arms = [('Z0', 0)] + [(f'RT{k}', r) for k in K.KS for r in range(K.R)] + [('ALL', 0)]
    for f in K.FEATS:
        for arm, r in arms:
            z = np.load(os.path.join(pdir, f'{f}_{arm}_r{r}.npz'))
            sid0 = z['sid_test'] if sid0 is None else sid0
            assert np.array_equal(z['sid_test'], sid0)
            P[(f, arm, r)] = z['P_test']
    yT = te.set_index(ID[task]).loc[sid0, 'y'].values.astype(int)
    CV = AC.unit_curves(yT, list(ba), P)
    star = {f: CV[(f, 'Z0', 0)][0][AC.I_T0] for f in K.FEATS}
    env, arm_rows, draw_rows = {}, [], []
    for f in K.FEATS:
        grid = star[f] + AC.OFFS
        for arm, r in arms:
            oa, ba_ = CV[(f, arm, r)]
            env[(f, arm, r)] = B.envelope([CV[(f, arm, r)]], grid)[0]
            arm_rows.append(dict(task=task, dom=d, feat=f, arm=arm, r=r, OA_star=star[f], oa_t0=oa[AC.I_T0],
                                 ba_t0=ba_[AC.I_T0], oa_min=float(oa.min()), oa_max=float(oa.max()),
                                 reach=float(np.isfinite(env[(f, arm, r)]).mean())))
    out = dict(task=task, dom=d, n_test=len(yT), n_ba_classes=len(ba), OA_star_ae=star['ae'], OA_star_s2=star['s2'])
    anc = min(star.values())
    ea = B.envelope([CV[('ae', 'Z0', 0)]], anc + AC.OFFS)[0]
    es = B.envelope([CV[('s2', 'Z0', 0)]], anc + AC.OFFS)[0]
    out['dR'], out['dR_cov'] = AC.paired(ea, es)
    per_feat = {}
    for f in K.FEATS:
        for k in list(K.KS) + ['all']:
            if k == 'all':
                v, c = AC.paired(env[(f, 'ALL', 0)], env[(f, 'Z0', 0)])
                vals, covs = [v], [c]
            else:
                pr = [AC.paired(env[(f, f'RT{k}', r)], env[(f, 'Z0', 0)]) for r in range(K.R)]
                vals, covs = [p[0] for p in pr], [p[1] for p in pr]
            for r, (v, c) in enumerate(zip(vals, covs)):
                draw_rows.append(dict(task=task, dom=d, feat=f, k=k, r=r, value=v, cov=c, missing=bool(not c >= AC.MIN_COV)))
            out[f'n_draws_{k}_{f}'] = int(np.isfinite(vals).sum())
            per_feat[(f, k)] = nanmean_or_nan(vals)
            out[f'dL_{k}_{f}'] = per_feat[(f, k)]
            out[f'cov_{k}_{f}'] = float(np.nanmin(covs)) if np.isfinite(covs).any() else np.nan
    for k in list(K.KS) + ['all']:
        a, s = per_feat[('ae', k)], per_feat[('s2', k)]
        out[f'dL_{k}'] = (a + s) / 2 if np.isfinite(a) and np.isfinite(s) else np.nan
        out[f'I_{k}'] = a - s if np.isfinite(a) and np.isfinite(s) else np.nan
    out['all_ge_dR'] = bool(out['dL_all'] >= out['dR']) if np.isfinite(out['dL_all']) and np.isfinite(out['dR']) else None
    out['all_lt_400'] = bool(out['dL_all'] < out['dL_400']) if np.isfinite(out['dL_all']) and np.isfinite(out['dL_400']) else None
    return out, arm_rows, draw_rows


def cluster_coverage(task, d, S_full, S, orders):
    """T1：在剔除前的候选集上按 330 m 求连通分量（与第 4 节的候选集一致）；
    报告每次抽样实际用于 RT200 的前 200 个样本（剔除后的顺序）落入的分量数。"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    c = S_full[(S_full[DOM[task]] == d) & (S_full.role == 'cand')].sort_values(ID[task])
    kept = set(S[ID[task]])
    xy = c[['x', 'y_m']].values
    pr = cKDTree(xy).query_pairs(K.T1_CLUSTER, output_type='ndarray')
    n = len(xy)
    A = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(n, n))
    ncomp, lab = connected_components(A, directed=False)
    comp = pd.Series(lab, index=c[ID[task]].values)
    res = []
    for r in range(K.R):
        o = orders[f'{d}_r{r}']
        o = o[np.isin(o, list(kept))][:200]
        res.append(int(comp.loc[o].nunique()))
    return ncomp, res


def task_summary(task, D):
    s = dict(task=task)
    both = D[np.isfinite(D.dR) & np.isfinite(D.dL_200)]           # 判定：只用 Δ_R 与 Δ_L(200) 都有值的域
    s['decide_doms'] = ','.join(both.dom)
    s['m_R'] = float(both.dR.mean()) if len(both) >= 2 else np.nan
    for k in list(K.KS) + ['all']:
        s[f'm_L_{k}'] = (float(both.dL_200.mean()) if len(both) >= 2 else np.nan) if k == 200 else \
            nanmean_or_nan(D[f'dL_{k}'], 2)                         # 其余 k 只作报告：对有值的域平均
        s[f'I_{k}'] = nanmean_or_nan(D[f'I_{k}'], 1)
    s['holds'] = bool(np.isfinite(s['m_R']) and np.isfinite(s['m_L_200']) and s['m_R'] > 0 and s['m_R'] >= s['m_L_200'])
    full = D[np.isfinite(D[['dR'] + [f'dL_{k}' for k in K.KS]]).all(axis=1)]
    s['L_AE_doms'] = ','.join(full.dom)
    if len(full) < 2:
        s['L_AE'] = '不可计算'
    else:
        fv = AC.pav(np.r_[0.0, [full[f'dL_{k}'].mean() for k in K.KS]])
        s['L_AE'] = AC.fmt_k(AC.k_eq(full.dR.mean(), KN, fv))
    return s


def dose_figure(D, T, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ks = [0] + list(K.KS)
    for ax, task in zip(axs, K.TASKS):
        Dt = D[D.task == task]
        doms = [x for x in str(T.loc[T.task == task, 'L_AE_doms'].iloc[0]).split(',') if x]
        M = Dt[Dt.dom.isin(doms)]
        for f, col in (('ae', '#3B6FB6'), ('s2', '#C8553D')):
            for _, row in Dt.iterrows():
                ax.plot(ks, [0] + [row[f'dL_{k}_{f}'] for k in K.KS], color=col, lw=0.6, alpha=0.35)
            if len(M):
                ax.plot(ks, [0] + [M[f'dL_{k}_{f}'].mean() for k in K.KS], color=col, lw=1.6, marker='o', ms=3,
                        label=f'{f.upper()}: label gain')
        if len(M):
            ax.axhline(M.dR.mean(), color='k', lw=1, ls='--', label='AE − S2 at 0 labels')
        ax.set_xlabel('Target labels (k)')
        ax.set_ylabel('Δ BA at matched OA (pp)')
        ax.set_title({'T1': 'Urban form (LCZ)', 'T2': 'Land cover (LUCAS)'}[task] + f"  [means: {','.join(doms) or 'none'}]")
        ax.set_xticks(ks)
    axs[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out, f'dose_curve.{ext}'), dpi=300)


REGISTERED_DEPS = {AC.SELF: '4dad4750221edcf5'}


def registered():
    """plan_hashes.txt 中登记过的 sha256 前 16 位（任意行中 'sha256[:16]=' 或 'code_hash=' 之后的值）。"""
    import re
    txt = open(K.PLAN_HASHES, encoding='utf-8').read()
    return set(re.findall(r'(?:sha256\[:16\]=|code_hash=)([0-9a-f]{16})', txt))


def formal_checks(runs):
    reg = registered()
    for f in ('PILOT_PLAN_v4.md', 'nc_common.py', 'nc_run.py', 'nc_analyze.py'):
        h = K.sha256(os.path.join(HERE, f))[:16]
        assert h in reg, f'{f} 的哈希 {h} 未登记'
    for p, h in REGISTERED_DEPS.items():
        assert K.sha256(p)[:16] == h, f'冻结依赖哈希不符：{p}'
    assert 'NC_SPLITS' not in os.environ
    man = json.load(open(os.path.join(K.SPLITS, 'manifest.json')))
    for f, h in man.items():
        assert K.sha256(os.path.join(K.SPLITS, f)) == h and h[:16] in reg, f'划分文件与登记不符：{f}'
    assert runs.code_hash.iloc[0] in reg, 'runs 的 code_hash 未登记'
    assert (runs.trees == 300).all() and (runs.threads == 2).all()
    assert runs.split_sha.nunique() == 1 and runs.groupby(['task', 'feat'])[['feat_sha', 'cols_sha']].nunique().max().max() == 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    formal_root = os.path.join(K.DATA, 'runs')
    if a.smoke:
        root = a.root or os.path.join(K.DATA, 'smoke')
        assert os.path.realpath(root) != os.path.realpath(formal_root), '冒烟模式不得读取正式 runs'
        out = a.out or os.path.join(root, 'analysis')
    else:
        assert a.root is None and a.out is None, '正式分析的输入输出目录固定'
        root, out = formal_root, os.path.join(formal_root, 'analysis')
        if os.path.exists(out):
            sys.exit(f'输出目录已存在，拒绝覆盖：{out}')
    tmp = out + '.tmp'
    if os.path.exists(tmp):
        import shutil
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    targets = json.load(open(os.path.join(K.SPLITS, 'targets.json')))
    ba = json.load(open(os.path.join(K.SPLITS, 'ba_classes.json')))
    orders = np.load(os.path.join(K.SPLITS, 'orders.npz'))
    runs = pd.read_csv(os.path.join(root, 'runs.csv'))
    assert len(runs) == 160 and runs.key.is_unique and (runs.status == 'ok').all() and runs.code_hash.nunique() == 1
    if not a.smoke:
        formal_checks(runs)
    removed = pd.read_csv(os.path.join(root, 'removed.csv'))
    rows, arms, draws = [], [], []
    for task in K.TASKS:
        S_full = pd.read_parquet(os.path.join(K.SPLITS, f'{task.lower()}_samples.parquet'))
        rm = set(removed.loc[removed.task == task, ID[task]].astype(np.int64))
        S = S_full[~S_full[ID[task]].isin(rm)]
        for d in targets[task]:
            r, ar, dr = domain_rows(task, d, root, ba[d], S)
            if task == 'T1':
                ncomp, cov = cluster_coverage(task, d, S_full, S, orders)
                r['cand_clusters'], r['k200_clusters'] = ncomp, ';'.join(map(str, cov))
            rows.append(r)
            arms += ar
            draws += dr
    D = pd.DataFrame(rows)
    T = pd.DataFrame([task_summary(t, D[D.task == t]) for t in K.TASKS])
    D.to_csv(os.path.join(tmp, 'domain_values.csv'), index=False)
    pd.DataFrame(arms).to_csv(os.path.join(tmp, 'arm_rows.csv'), index=False)
    pd.DataFrame(draws).to_csv(os.path.join(tmp, 'draw_rows.csv'), index=False)
    T.to_csv(os.path.join(tmp, 'task_summary.csv'), index=False)
    dom_col = removed['city'].fillna(removed['nuts0']) if 'nuts0' in removed else removed['city']
    rem_sum = removed.assign(dom=dom_col).groupby(['task', 'dom', 'role', 'y']).size().rename('n_removed').reset_index()
    rem_sum.to_csv(os.path.join(tmp, 'removed_summary.csv'), index=False)
    rates = pd.read_csv(os.path.join(root, 'missing_rates.csv'))
    n_hold = int(T.holds.sum())
    verdict = {2: 'GO', 1: '部分 GO', 0: 'STOP'}[n_hold]
    lines = [f'决策：{verdict}（成立的任务：{", ".join(T.task[T.holds]) or "无"}）',
             f'代码：nc_analyze {K.sha256(SELF_PATH)[:16]}；runs code_hash {runs.code_hash.iloc[0]}；'
             f'方案 {K.sha256(os.path.join(HERE, "PILOT_PLAN_v4.md"))[:16]}', '']
    for _, s in T.iterrows():
        lines.append(f"{s.task}: 判定 m_R = {s.m_R:+.2f}，m_L(200) = {s.m_L_200:+.2f}（域：{s.decide_doms}），成立 = {s.holds}")
        lines.append(f"    L_AE = {s.L_AE}（域：{s.L_AE_doms}）")
        lines.append(f"    描述性 m_L(50/100/400/全)（各自对有值的域平均，至少 2 个域）= "
                     f"{s.m_L_50:+.2f}/{s.m_L_100:+.2f}/{s.m_L_400:+.2f}/{s.m_L_all:+.2f}")
        lines.append(f"    I(50/100/200/400/全)（对有值的域平均）= "
                     f"{s.I_50:+.2f}/{s.I_100:+.2f}/{s.I_200:+.2f}/{s.I_400:+.2f}/{s.I_all:+.2f}")
    lines.append('')
    for _, r in D.iterrows():
        lines.append(f"  {r.task} {r.dom}: Δ_R = {r.dR:+.2f}（共同可达 {r.dR_cov:.2f}）；Δ_L(50/100/200/400/全) = "
                     + '/'.join(f'{r[f"dL_{k}"]:+.2f}' for k in list(K.KS) + ['all'])
                     + f"；全 ≥ Δ_R：{r.all_ge_dR}；全 < 400：{r.all_lt_400}"
                     + (f"；候选簇 {r.cand_clusters}，k=200 覆盖 {r.k200_clusters}" if r.task == 'T1' else ''))
    lines.append('')
    lines.append('剔除样本（按任务、角色）：' + removed.groupby(['task', 'role']).size().to_string().replace('\n', '; '))
    lines.append('部分缺失（保留样本中含 NaN 的行比例，按任务 × 特征）：'
                 + rates.assign(w=rates.n * rates.rows_any_nan).groupby(['task', 'feat'])
                 .apply(lambda g: g.w.sum() / g.n.sum(), include_groups=False).round(3).to_string().replace('\n', '; '))
    open(os.path.join(tmp, 'decision.txt'), 'w').write('\n'.join(lines) + '\n')
    dose_figure(D, T, tmp)
    os.rename(tmp, out)
    print('\n'.join(lines))
    if not a.smoke:
        for f in ('domain_values.csv', 'arm_rows.csv', 'draw_rows.csv', 'task_summary.csv', 'removed_summary.csv',
                  'decision.txt', 'dose_curve.png', 'dose_curve.pdf'):
            K.register(f'nc_pilot analysis output {f} sha256[:16]={K.sha256(os.path.join(out, f))[:16]} '
                       '(one-shot analysis, registered immediately)')


if __name__ == '__main__':
    main()
