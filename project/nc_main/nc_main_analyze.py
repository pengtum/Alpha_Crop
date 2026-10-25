"""NC 主实验分析：一次性计算全部登记量（冻结方案 NC_PLAN_v6 第 3–5 节）。

输入：nc_main_run.py 的 {root}/probs、runs_*.csv、budgets.csv；splits/（targets、ba_classes）；试点样本表（标签）；
      作物锚点读 RSE 确认运行（confirm_v1 的 probs、splits、selection）。
度量：τ 曲线用冻结的 analyze_confirm.unit_curves；Δ_V 用 nc_metrics.delta_v；保序回归用 analyze_confirm.pav；
      bootstrap 区间用 analyze_confirm.q_interval(level=0.95)。
输出 {root}/analysis/：domain_values.csv、curves_e_g.csv、confirm.json、estimates.json、crop_anchor.csv、report.md。

用法：python nc_main_analyze.py [--root DIR] [--no-crop]
"""
import argparse
import json
import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
AC_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(AC_ROOT, 'nc_pilot'))
sys.path.insert(0, os.path.join(AC_ROOT, 'label_expansion', 'confirm'))
import nc_metrics as NM  # noqa: E402
import nc_common as K  # noqa: E402
_cwd = os.getcwd()
import analyze_confirm as AC  # noqa: E402  导入链会 chdir 并核对冻结文件哈希
os.chdir(_cwd)

LOCAL = os.path.expanduser('~/AlphaCrops_local/nc_main')
SPLITS = os.path.join(LOCAL, 'splits')
PILOT_SPLITS = os.path.join(K.DATA, 'splits')
CONF = os.path.expanduser('~/AlphaCrops_local/results/confirm_v1')
TASKS = {'lcz': ('T1', 'city', 'row'), 'lucas': ('T2', 'nuts0', 'id')}
FAMILIES = ('lcz', 'lucas')
FEATS = ('ae', 's2')
LEARNERS = ('lgb', 'rf', 'lr')
KS = (50, 100, 200, 400, 800, 1600)
R = 3
K_D = 200                              # d = e(200)；g(200)；附录 RT200
ALPHA = 0.025
N_BOOT = 2000
BOOT_SEED = 'nc_main_v1_boot_M1'
CROP_KS = (50, 100, 200, 400)


# ---------------------------------------------------------------- 逐域的量
def labels_of(task):
    T, dom, idc = TASKS[task]
    S = pd.read_parquet(os.path.join(PILOT_SPLITS, f'{T.lower()}_samples.parquet'))
    return pd.Series(S.y.values.astype(int), index=S[idc].values.astype(np.int64))


def m_of(P, y):
    """τ = 0（argmax，并列取首个）的正确率（%）。"""
    return 100.0 * float(np.mean(np.argmax(np.asarray(P, dtype=np.float64), axis=1) == np.asarray(y)))


def e_g(CV, ks):
    """e(k)、g(k)：每次抽样内配对计算，再对 3 次抽样平均。CV 键 (feat, arm, r)。"""
    ae0 = CV[('ae', 'Z0', 0)]
    e = {0: NM.delta_v(CV[('s2', 'Z0', 0)], ae0)}
    g = {}
    for k in ks:
        e[k] = float(np.mean([NM.delta_v(CV[('s2', f'RT{k}', r)], ae0) for r in range(R)]))
        g[k] = float(np.mean([NM.delta_v(ae0, CV[('ae', f'RT{k}', r)]) for r in range(R)]))
    return e, g


def l_saved(e):
    """e：{k: e(k)}，k 含 0。ê = −pav(−e)（单调不增）；首个 ê ≤ 0 的格点与前一格点之间线性插值。
    返回 (L_saved, 是否右删失, k_max, ê 字典)。"""
    kk = sorted(e)
    eh = -AC.pav(-np.array([e[k] for k in kk]))
    hit = np.flatnonzero(eh <= 0)
    if len(hit) == 0:
        return float(kk[-1]), True, kk[-1], dict(zip(kk, eh))
    j = int(hit[0])
    if j == 0:
        return 0.0, False, kk[-1], dict(zip(kk, eh))
    L = kk[j - 1] + (kk[j] - kk[j - 1]) * eh[j - 1] / (eh[j - 1] - eh[j])
    return float(L), False, kk[-1], dict(zip(kk, eh))


def check_runs(root):
    """所有登记的拟合都应有一行且成功；有失败（含重试后）的域整体剔除并报告。"""
    R_ = pd.concat([pd.read_csv(os.path.join(root, f)) for f in sorted(os.listdir(root))
                    if f.startswith('runs_') and f.endswith('.csv')], ignore_index=True)
    assert R_.key.is_unique, '同一拟合出现多行'
    bud = pd.read_csv(os.path.join(root, 'budgets.csv'))
    expect, fails = {}, set()
    for _, b in bud.iterrows():
        ks = [int(x) for x in str(b.ks).split()]
        n = 2 * (1 + R * len(ks)) + 2 * 2 * (1 + R)          # lgb 全部臂；rf、lr 各 Z0 + RT200×3
        expect[(b.task, b.domain)] = n
    got = R_.groupby(['task', 'domain']).size().to_dict()
    for key, n in expect.items():
        if got.get(key, 0) != n:
            raise SystemExit(f'拟合未完成：{key} 应 {n} 次，实有 {got.get(key, 0)} 次')
    for _, r in R_[R_.status != 'ok'].iterrows():
        fails.add((r.task, r.domain))
    return R_, bud, sorted(fails)


def domain_values(root, task, d, ks, big, yser, learner):
    """读一个域一个学习器的概率 → τ 曲线 → e、g、d、M（M 只对 lgb）。"""
    pdir = os.path.join(root, 'probs', task, d)
    arms = [('Z0', 0)] + ([(f'RT{k}', r) for k in ks for r in range(R)] if learner == 'lgb' else
                          [(f'RT{K_D}', r) for r in range(R)])
    P, sid, Pm = {}, None, None
    for f in FEATS:
        for arm, r in arms:
            z = np.load(os.path.join(pdir, f'{learner}_{f}_{arm}_r{r}.npz'))
            sid = z['sid_test'] if sid is None else sid
            assert np.array_equal(sid, z['sid_test']), (task, d, learner, f, arm, r)
            P[(f, arm, r)] = z['P_test']
            if learner == 'lgb' and f == 'ae' and arm == 'Z0':
                Pm, sid_m = z['P_m'], z['sid_m']
    yT = yser.loc[sid].values
    CV = AC.unit_curves(yT, big, P)
    e, g = e_g(CV, ks if learner == 'lgb' else [K_D])
    out = dict(task=task, domain=d, learner=learner, n_test=len(sid), d=e[K_D], g200=g[K_D], e=e, g=g)
    if learner == 'lgb':
        out['M'] = m_of(Pm, yser.loc[sid_m].values)
        out['n_m'] = len(sid_m)
        L, cens, kmax, eh = l_saved(e)
        out.update(L_saved=L, censored=cens, k_max=kmax, e_hat=eh)
    return out


# ---------------------------------------------------------------- 检验
def ols_b(y, X):
    return np.linalg.inv(X.T @ X) @ X.T @ y


def ols_hc3(y, X):
    XtX = np.linalg.inv(X.T @ X)
    b = XtX @ X.T @ y
    e = y - X @ b
    h = np.einsum('ij,jk,ik->i', X, XtX, X)
    V = XtX @ (X.T * (e ** 2 / (1 - h) ** 2)) @ X @ XtX
    return b, np.sqrt(np.diag(V))


def design(D):
    return np.c_[np.ones(len(D)), (D.task == 'lucas').values.astype(float), D.M.values]


def m1_of(b):
    """每族 g(200) 拟合值降到 1 pp 处的 M：LCZ (1 − a)/β，LUCAS (1 − a − γ)/β。"""
    a, gam, beta = b
    return {'lcz': (1 - a) / beta, 'lucas': (1 - a - gam) / beta}


def holm(p1, p2, a=ALPHA):
    lo, hi = min(p1, p2), max(p1, p2)
    r_lo = lo <= a / 2
    r_hi = r_lo and hi <= a
    return bool((p1 <= p2 and r_lo) or (p1 > p2 and r_hi)), bool((p2 < p1 and r_lo) or (p2 >= p1 and r_hi))


def bootstrap_m1(D):
    """按族分层的域级 bootstrap：每族域按名称升序，同一生成器先 LCZ、后 LUCAS 各一次 integers(0, n, (2000, n))。"""
    rng = K.rng_of(BOOT_SEED)
    parts = {fam: D[D.task == fam].sort_values('domain').reset_index(drop=True) for fam in FAMILIES}
    idx = {fam: rng.integers(0, len(parts[fam]), size=(N_BOOT, len(parts[fam]))) for fam in FAMILIES}
    vals, n_bad, n_sing = {fam: [] for fam in FAMILIES}, 0, 0
    for b in range(N_BOOT):
        Db = pd.concat([parts[fam].iloc[idx[fam][b]] for fam in FAMILIES], ignore_index=True)
        X = design(Db)
        if np.linalg.matrix_rank(X) < X.shape[1]:            # 设计矩阵秩不足：β 不可估计，不计入（正式数据中不会出现）
            n_sing += 1
            continue
        bb = ols_b(Db.g200.values, X)
        if bb[2] >= 0:
            n_bad += 1
            continue
        for fam, v in m1_of(bb).items():
            vals[fam].append(v)
    q = n_bad / N_BOOT
    unb = (n_bad + n_sing) / N_BOOT > 0.025
    ci = {fam: (None if unb else [float(x) for x in AC.q_interval(np.array(vals[fam]), level=0.95)])
          for fam in FAMILIES}
    return q, n_sing, unb, ci


def confirm(D):
    n = len(D)
    pos = int((D.d > 0).sum())
    p1 = float(stats.binom.sf(pos - 1, n, 0.5))
    b, se = ols_hc3(D.g200.values, design(D))
    t2 = float(b[2] / se[2])
    p2 = float(stats.t.cdf(t2, n - 3))
    r1, r2 = holm(p1, p2)
    out = dict(n=n, n_by_family={f: int((D.task == f).sum()) for f in FAMILIES},
               H1=dict(n_pos=pos, p=p1, rejected=r1),
               H2=dict(a=float(b[0]), gamma=float(b[1]), beta=float(b[2]), se_hc3=[float(x) for x in se], t=t2, df=n - 3,
                       p=p2, rejected=r2))
    m1 = m1_of(b)
    q, n_sing, unb, ci = bootstrap_m1(D) if b[2] < 0 else (None, None, None, None)
    out['M1'] = {fam: dict(value=float(m1[fam]) if b[2] < 0 else None,
                           ci95=None if ci is None else ci[fam],
                           unbounded=unb,
                           extrapolated=None if b[2] >= 0 else bool(not (D.M[D.task == fam].min() <= m1[fam]
                                                                          <= D.M[D.task == fam].max())),
                           M_range=[float(D.M[D.task == fam].min()), float(D.M[D.task == fam].max())])
                 for fam in FAMILIES}
    out['M1_q_beta_nonneg'] = q
    out['M1_n_singular'] = n_sing
    out['M1_reportable'] = r2                                  # 只在 H2 被拒绝时写预算规则与 M₁
    return out


# ---------------------------------------------------------------- 估计值
def summarize(V, L):
    """V：lgb 的逐域值；L：e、g 长表（全部学习器）。"""
    est = {}
    for grp, sub in [('pooled', V)] + [(f, V[V.task == f]) for f in FAMILIES]:
        est[f'd_{grp}'] = dict(n=len(sub), mean=float(sub.d.mean()), median=float(sub.d.median()),
                               frac_pos=float((sub.d > 0).mean()))
    lg = L[L.learner == 'lgb']
    est['dose'] = (lg.groupby(['task', 'quantity', 'k']).value.agg(['mean', 'median', 'count']).reset_index()
                   .to_dict('records'))
    rows = []
    for grp, sub in [('pooled', V)] + [(f, V[V.task == f]) for f in FAMILIES]:
        for k in (0,) + KS:
            has = sub[sub.e_hat.apply(lambda h: k in h)]
            if len(has):
                rows.append(dict(group=grp, k=k, n=len(has), frac_L_gt_k=float(np.mean([h[k] > 0 for h in has.e_hat]))))
    est['frac_L_saved_gt_k'] = rows
    est['L_saved'] = V[['task', 'domain', 'L_saved', 'censored', 'k_max']].to_dict('records')
    return est


def appendix(Vall, h1_rejected):
    out = {}
    for lr_ in ('rf', 'lr'):
        sub = Vall[Vall.learner == lr_]
        out[lr_] = dict(n=len(sub), d_mean=float(sub.d.mean()), d_median=float(sub.d.median()),
                        g200_mean=float(sub.g200.mean()), g200_median=float(sub.g200.median()),
                        frac_d_pos=float((sub.d > 0).mean()))
    agree = h1_rejected and all(out[lr_]['frac_d_pos'] > 0.5 for lr_ in ('rf', 'lr'))
    out['statement'] = '三种学习器方向一致' if agree else '结论限于 LightGBM'
    return out


# ---------------------------------------------------------------- 作物锚点（RSE 重读，描述性）
def crop_anchor():
    sys.path.insert(0, os.path.join(AC_ROOT, 'unified_figs'))
    import prep_unified as PU
    sys.path.insert(0, os.path.join(AC_ROOT, 'label_expansion'))
    cwd = os.getcwd()
    os.chdir(os.path.join(AC_ROOT, 'label_expansion'))
    from guidance_G import GData
    y_all = np.asarray(GData().y)
    os.chdir(cwd)
    rows = []
    for U in PU.crop_units():
        CV = AC.unit_curves(U['yT'], U['big'], U['P'])
        e, g = e_g(CV, CROP_KS)
        u = U['unit']
        S = np.load(os.path.join(CONF, 'splits', f'{u}.npz'))
        Sel = np.load(os.path.join(CONF, 'selection', f'{u}.npz'))
        Pp = np.load(os.path.join(CONF, 'probs', u, 'lgb_Z0_ae_r0.npz'))['P_pool']
        pos = np.asarray(Sel['pi_r0'][200:250], dtype=np.int64)
        M = m_of(Pp[pos], y_all[np.asarray(S['pool'], dtype=np.int64)[pos]])
        rows.append(dict(unit=u, domain=U['domain'], primary=U['primary'], M=M,
                         **{f'e{k}': v for k, v in e.items()}, **{f'g{k}': v for k, v in g.items()}))
    U_ = pd.DataFrame(rows)
    C = U_.groupby(['domain', 'primary']).mean(numeric_only=True).reset_index()   # F1、F2 两半平均
    Ls = [l_saved({k: r[f'e{k}'] for k in (0,) + CROP_KS}) for _, r in C.iterrows()]
    C['L_saved'], C['censored'] = [x[0] for x in Ls], [x[1] for x in Ls]
    C = C.rename(columns={f'e{K_D}': 'd'})
    return U_, C


# ---------------------------------------------------------------- 主程序
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.join(LOCAL, 'runs'))
    ap.add_argument('--no-crop', action='store_true')
    a = ap.parse_args()
    out = os.path.join(a.root, 'analysis')
    os.makedirs(out, exist_ok=True)
    runs, bud, fails = check_runs(a.root)
    ba = json.load(open(os.path.join(SPLITS, 'ba_classes.json')))
    rows = []
    for task in FAMILIES:
        yser = labels_of(task)
        for _, b in bud[bud.task == task].iterrows():
            if (task, b.domain) in fails:
                continue
            ks = [int(x) for x in str(b.ks).split()]
            for lr_ in LEARNERS:
                rows.append(domain_values(a.root, task, b.domain, ks, ba[task][b.domain], yser, lr_))
            print(task, b.domain, flush=True)
    Vall = pd.DataFrame(rows)
    L = pd.DataFrame([dict(task=r['task'], domain=r['domain'], learner=r['learner'], quantity=q, k=k, value=v)
                      for r in rows for q in ('e', 'g') for k, v in r[q].items()])
    V = Vall[Vall.learner == 'lgb'].reset_index(drop=True)
    C = confirm(V)
    C['excluded_domains'] = [list(x) for x in fails]
    est = summarize(V, L)
    est['appendix'] = appendix(Vall, C['H1']['rejected'])
    keep = ['task', 'domain', 'learner', 'n_test', 'd', 'g200', 'M', 'n_m', 'L_saved', 'censored', 'k_max']
    Vall[[c for c in keep if c in Vall]].to_csv(os.path.join(out, 'domain_values.csv'), index=False)
    L.to_csv(os.path.join(out, 'curves_e_g.csv'), index=False)
    json.dump(C, open(os.path.join(out, 'confirm.json'), 'w'), indent=1, ensure_ascii=False)
    crop = None
    if not a.no_crop:
        Uc, crop = crop_anchor()
        Uc.to_csv(os.path.join(out, 'crop_anchor_units.csv'), index=False)
        crop.to_csv(os.path.join(out, 'crop_anchor.csv'), index=False)
        est['crop'] = {nm: dict(n=len(c_), d_n_pos=int((c_.d > 0).sum()), d_mean=float(c_.d.mean()),
                                d_median=float(c_.d.median()),
                                frac_L_saved_gt_k=[dict(k=k, n=len(c_), frac_L_gt_k=float((c_.censored | (c_.L_saved > k)).mean()))
                                                   for k in (0,) + CROP_KS])
                       for nm, c_ in (('9', crop[crop.primary]), ('11', crop))}
    json.dump(est, open(os.path.join(out, 'estimates.json'), 'w'), indent=1, ensure_ascii=False, default=float)

    # 报告（措辞按第 4 节写死的规则）
    lines = ['# NC 主实验结果（自动生成）', '',
             f"确认性域 n = {C['n']}（LCZ {C['n_by_family']['lcz']}、LUCAS {C['n_by_family']['lucas']}）；剔除：{C['excluded_domains'] or '无'}", '',
             f"- H1：d > 0 的域 {C['H1']['n_pos']}/{C['n']}，精确二项单侧 p = {C['H1']['p']:.3g}，"
             f"{'拒绝' if C['H1']['rejected'] else '未拒绝'}（Holm）",
             f"- H2：β̂ = {C['H2']['beta']:.4f}（HC3 se {C['H2']['se_hc3'][2]:.4f}），t = {C['H2']['t']:.3f}，df = {C['H2']['df']}，"
             f"单侧 p = {C['H2']['p']:.3g}，{'拒绝' if C['H2']['rejected'] else '未拒绝'}（Holm）", '']
    if C['H1']['rejected']:
        lines.append('主张（H1）：在多数目标地区，零标签的 AE 胜过 S2 加 200 个本地标签。')
    if C['H2']['rejected']:
        lines.append('主张（H2）：先用 50 个本地标签测零标签迁移精度。高于 M₁ 时，再采 200 个标签预期提升不足 1 pp；低于 M₁ 时，应采标签。')
        for fam in FAMILIES:
            m = C['M1'][fam]
            ci = '无界' if m['unbounded'] else f"[{m['ci95'][0]:.1f}, {m['ci95'][1]:.1f}]"
            lines.append(f"  - {fam}：M₁ = {m['value']:.1f}%，95% 区间 {ci}{'，外推' if m['extrapolated'] else ''}")
    if not (C['H1']['rejected'] or C['H2']['rejected']):
        lines.append('两个假设均未被拒绝：只报告估计值，不写成结论。')
    lines += ['', f"附录：{est['appendix']['statement']}（RF d > 0 比例 {est['appendix']['rf']['frac_d_pos']:.2f}，"
              f"LR {est['appendix']['lr']['frac_d_pos']:.2f}）"]
    if crop is not None:
        for nm, c_ in (('9', crop[crop.primary]), ('11', crop)):
            lines.append(f"作物锚点（{nm} 国）：d > 0 的国家 {int((c_.d > 0).sum())}/{len(c_)}，d 中位数 {c_.d.median():+.2f}")
    open(os.path.join(out, 'report.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
