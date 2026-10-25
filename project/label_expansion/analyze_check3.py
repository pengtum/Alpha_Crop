"""
检验 3 分析（ANALYSIS_PLAN_check3.md，以及优先的修订 ANALYSIS_PLAN_check3_rev1.md）。

主分解：训练时链式替换
  OA：   L0 → PM → CMN → W-nat   （类别构成 / 样本量 / 域残差）
  ba_ge30：L0 → LS → CMB → W0    （类别集合 / 样本量 / 域残差）
次要：事后修正 2×2（Shapley、交互项）、先验估计（PEM、PEM∘LS、PEM_bcts、PK、BBSE）、BIAS_ORACLE、
      按类归因、描述量相关、空间分块与各项敏感性、校准诊断。
用法：python analyze_check3.py [--noise]
  --noise 在每个目标国内打乱 y，用于在不看结果的情况下测试流程（结果无意义）。
输出：RESULTS_DIR/check3_report.txt、check3_arms.csv
"""
import hashlib
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

from check3_priors import as64, trained_mask, uniform_prior, reweight, label_set, em_prior, laplace_prior, bbse_prior
from common import COUNTRIES, RESULTS_DIR, RANDOM_STATE, load_dataset

D = os.path.join(RESULTS_DIR, 'check3')
HERE = os.path.dirname(os.path.abspath(__file__))
C = 10
MIN_N = 30
GROUPS = {'同质': ['maize', 'potatoes', 'strawberries'],
          '异质': ['other_crops', 'orchards', 'vegetables'],
          '其余': ['grassland', 'winter_cereals', 'root_crops', 'tree_wood_forest']}
NOISE = '--noise' in sys.argv


class Tee:
    def __init__(self, path):
        self.f = open(path, 'w')

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        self.f.flush()


def P(tgt, feat, model, part):
    return as64(np.load(os.path.join(D, f'{tgt}_{feat}_{model}_{part}.npy')))


def metrics(y, yhat, drop=None):
    n = np.bincount(y, minlength=C)
    rec = np.array([(yhat[y == c] == c).mean() if n[c] else np.nan for c in range(C)])
    big = n >= MIN_N
    if drop is not None:
        big[drop] = False
    present = np.flatnonzero(n)
    return dict(oa=100 * float((y == yhat).mean()), ba_ge30=100 * float(np.nanmean(rec[big])),
                ba=100 * float(np.nanmean(rec)),
                f1=100 * float(f1_score(y, yhat, labels=present, average='macro', zero_division=0)))


def fit_bcts(p, y, w):
    """温度 + 偏置校准（Alexandari et al. 2020）：logits = s·log p + b，最小化加权 NLL。"""
    k = trained_mask(p)
    z = np.log(np.clip(p[:, k], 1e-12, None))
    Y = np.eye(k.sum())[np.searchsorted(np.flatnonzero(k), y)]
    w = w / w.sum()

    def f(theta):
        s, b = theta[0], theta[1:]
        l = s * z + b
        l = l - l.max(1, keepdims=True)
        q = np.exp(l)
        q /= q.sum(1, keepdims=True)
        nll = -(w * np.log(np.clip((q * Y).sum(1), 1e-300, None))).sum()
        g = (q - Y) * w[:, None]
        return nll, np.r_[(g * z).sum(), g.sum(0)]

    r = minimize(f, np.r_[1.0, np.zeros(k.sum())], jac=True, method='L-BFGS-B',
                 bounds=[(1e-3, None)] + [(None, None)] * k.sum())

    def apply(pp):
        l = r.x[0] * np.log(np.clip(pp[:, k], 1e-12, None)) + r.x[1:]
        l = l - l.max(1, keepdims=True)
        q = np.exp(l)
        out = np.zeros_like(pp)
        out[:, k] = q / q.sum(1, keepdims=True)
        return out
    return apply, r.x[0]


def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    e = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return 100 * e


def pooled(df, part, total, rng, B=5000):
    """合并比例 Σ部分/Σ总损失，按目标国 cluster bootstrap，另给留一国范围。"""
    t = df.index.to_numpy()
    val = df[part].sum() / df[total].sum()
    boot = []
    for _ in range(B):
        s = rng.choice(len(t), len(t))
        den = df[total].iloc[s].sum()
        boot.append(df[part].iloc[s].sum() / den if den != 0 else np.nan)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    loo = [df.drop(i)[part].sum() / df.drop(i)[total].sum() for i in t]
    return val, lo, hi, min(loo), max(loo)


def main():
    out = 'check3_report_NOISE.txt' if NOISE else 'check3_report.txt'
    sys.stdout = Tee(os.path.join(RESULTS_DIR, out))
    for p in ['ANALYSIS_PLAN_check3.md', 'ANALYSIS_PLAN_check3_rev1.md']:
        with open(os.path.join(HERE, p), 'rb') as f:
            print(f'{p} sha256[:16] = {hashlib.sha256(f.read()).hexdigest()[:16]}')
    if NOISE:
        print('!!! 噪声模式：y 已在目标国内打乱，结果无意义 !!!')
    rng = np.random.default_rng(RANDOM_STATE)

    df_all = load_dataset()
    classes = sorted(df_all['label'].unique())
    cid = {c: i for i, c in enumerate(classes)}
    y_all = df_all['label'].map(cid).to_numpy()
    hcat = df_all['EC_hcat_n'].to_numpy()
    grp_idx = {g: [cid[c] for c in cs] for g, cs in GROUPS.items()}
    other_id = cid['other_crops']

    rows, perclass, estim, calib_rows = [], [], [], []
    boot_store = {}
    targets = [t for t in COUNTRIES if os.path.exists(os.path.join(D, f'{t}_calib.npz'))
               and all(os.path.exists(os.path.join(D, f'{t}_{f}_wnat_sp_testsp.npy')) for f in ['ae', 's2'])]
    print(f'可用目标国（{len(targets)}）：{targets}')
    for tgt in targets:
        sp = np.load(os.path.join(D, f'{tgt}_split.npz'))
        cal = np.load(os.path.join(D, f'{tgt}_calib.npz'))
        ssp = np.load(os.path.join(D, f'{tgt}_split_sp.npz'))
        pool, test, foreign = sp['pool'], sp['test'], sp['foreign']
        y_pool, y_test = sp['y_pool'].copy(), sp['y_test'].copy()
        if NOISE:
            r2 = np.random.default_rng(1)
            y_pool, y_test = r2.permutation(y_pool), r2.permutation(y_test)
        n_pool = np.bincount(y_pool, minlength=C)
        pi_T = n_pool / n_pool.sum()
        present = n_pool > 0
        present20 = n_pool >= 20
        pi_F = np.bincount(y_all[foreign], minlength=C) / len(foreign)
        # 描述量
        js_unif = jensenshannon(pi_T, np.ones(C) / C)
        js_F = jensenshannon(pi_T, pi_F)
        fine = []
        for c in np.flatnonzero(present):
            a = pd.Series(hcat[pool][y_all[pool] == c]).value_counts(normalize=True)
            b = pd.Series(hcat[foreign][y_all[foreign] == c]).value_counts(normalize=True)
            u = a.index.union(b.index)
            fine.append(pi_T[c] * jensenshannon(a.reindex(u, fill_value=0), b.reindex(u, fill_value=0)))
        js_fine = float(np.sum(fine) / pi_T[present].sum())
        # 空间分块测试集在 pool ∪ test 中的位置
        t_all = np.concatenate([pool, test])
        pos = pd.Series(np.arange(len(t_all)), index=t_all)
        sp_pos = pos.loc[ssp['test_sp']].to_numpy()
        y_all_T = np.r_[y_pool, y_test]
        y_sp = y_all_T[sp_pos]
        pool_sp_pos = pos.loc[ssp['pool_sp']].to_numpy()
        n_pool_sp = np.bincount(y_all_T[pool_sp_pos], minlength=C)

        for feat in ['ae', 's2']:
            L = {s: P(tgt, feat, 'loco', s) for s in ('pool', 'test')}
            Lc = {s: P(tgt, feat, 'lcal', s) for s in ('pool', 'test', 'calib', 'fva')}
            pi_tr_L = uniform_prior(trained_mask(L['test']))
            arms_p = {
                'L0': L['test'], 'PM': P(tgt, feat, 'loco_pm', 'test'),
                'CMN': P(tgt, feat, 'cmn', 'test'), 'Wnat': P(tgt, feat, 'wnat', 'test'),
                'CMB': P(tgt, feat, 'cmb', 'test'), 'W0': P(tgt, feat, 'within', 'test'),
            }
            arms_p['LS'] = label_set(L['test'], present, pi_tr_L)[0]
            arms_p['LS20'] = label_set(L['test'], present20, pi_tr_L)[0]
            arms_p['PO'] = reweight(L['test'], pi_T, pi_tr_L)[0]
            pi_tr_W = uniform_prior(trained_mask(arms_p['W0']))
            arms_p['WPO'] = reweight(arms_p['W0'], pi_T, pi_tr_W)[0]
            # 先验估计（在 LOCO 的 pool 预测上，无标签）
            pem, it, conv, ll = em_prior(L['pool'], pi_tr_L)
            arms_p['PEM'] = reweight(L['test'], pem, pi_tr_L)[0]
            pi_ls = uniform_prior(present & trained_mask(L['test']))
            pem_ls, it2, conv2, _ = em_prior(label_set(L['pool'], present, pi_tr_L)[0], pi_ls)
            arms_p['PEM_LS'] = reweight(arms_p['LS'], pem_ls, pi_ls)[0]
            w_cal = compute_sample_weight('balanced', cal['y_calib'])
            bcts, s_T = fit_bcts(Lc['calib'], cal['y_calib'], w_cal)
            Lb = {s: bcts(Lc[s]) for s in ('pool', 'test')}
            pi_tr_c = uniform_prior(trained_mask(Lb['test']))
            arms_p['L0cal'] = Lc['test']
            arms_p['PO_bcts'] = reweight(Lb['test'], pi_T, pi_tr_c)[0]
            pem_b, it3, conv3, _ = em_prior(Lb['pool'], pi_tr_c)
            arms_p['PEM_bcts'] = reweight(Lb['test'], pem_b, pi_tr_c)[0]
            bb_h, cond_h, _ = bbse_prior(Lc['calib'], cal['y_calib'], w_cal, Lc['pool'], trained_mask(Lc['calib']))
            bb_s, cond_s, _ = bbse_prior(Lc['calib'], cal['y_calib'], w_cal, Lc['pool'], trained_mask(Lc['calib']),
                                         soft=True)
            arms_p['BBSE_hard'] = reweight(Lc['test'], bb_h, pi_tr_L)[0]
            arms_p['BBSE_soft'] = reweight(Lc['test'], bb_s, pi_tr_L)[0]
            # BIAS_ORACLE：用 pool 标签在 LOCO 的 pool 预测上拟合温度+偏置（目标国大量标签，上界）
            ora, _ = fit_bcts(L['pool'], y_pool, np.ones(len(y_pool)))
            q_ora = ora(L['test'])
            arms_p['ORACLE_oa'] = q_ora
            # ba 版本：除以 π_T 回到均匀先验；T 中不存在的类权重为 0（reweight 对 pi_train=0 的类置零）
            arms_p['ORACLE_ba'] = reweight(q_ora, present / present.sum(), pi_T)[0]
            est = {'PEM': (pem, it, conv), 'PEM_LS': (pem_ls, it2, conv2), 'PEM_bcts': (pem_b, it3, conv3),
                   'BBSE_hard': (bb_h, cond_h, None), 'BBSE_soft': (bb_s, cond_s, None)}
            for k_lab in (50, 200):
                for seed in range(3):
                    lab = np.random.default_rng(1000 + seed).choice(len(y_pool), k_lab, replace=False)
                    for tag, alpha, restrict in (('PK', 1.0, False), ('PKj', 0.5, False), ('PK_LS', 1.0, True)):
                        if restrict:
                            n = np.bincount(y_pool[lab], minlength=C).astype(float)
                            pk = np.where(present, n + 1.0, 0)
                            pk /= pk.sum()
                        else:
                            pk = laplace_prior(y_pool[lab], C, alpha)
                        name = f'{tag}{k_lab}_s{seed}'
                        arms_p[name] = reweight(L['test'], pk, pi_tr_L)[0]
                        est[name] = (pk, None, None)
            for name, (pi_hat, a, b) in est.items():
                estim.append(dict(target=tgt, feat=feat, est=name, l1=float(np.abs(pi_hat - pi_T).sum()),
                                  iters_or_cond=a, converged=b))

            preds = {k: v.argmax(1) for k, v in arms_p.items()}
            for arm, yh in preds.items():
                m = metrics(y_test, yh)
                m_noo = metrics(y_test, yh, drop=other_id)
                rows.append(dict(target=tgt, feat=feat, arm=arm, split='random', **m, ba_ge30_noother=m_noo['ba_ge30']))
            # 空间分块：W0_sp、Wnat_sp 与其余各臂在 test_sp 上
            full = {}
            for arm, src in (('L0', 'loco'), ('PM', 'loco_pm'), ('CMN', 'cmn'), ('CMB', 'cmb')):
                full[arm] = np.vstack([P(tgt, feat, src, 'pool'), P(tgt, feat, src, 'test')])[sp_pos]
            full['LS'] = label_set(full['L0'], n_pool_sp > 0, pi_tr_L)[0]
            full['W0'] = P(tgt, feat, 'w0_sp', 'testsp')
            full['Wnat'] = P(tgt, feat, 'wnat_sp', 'testsp')
            for arm, pp in full.items():
                rows.append(dict(target=tgt, feat=feat, arm=arm, split='spatial', **metrics(y_sp, pp.argmax(1))))
            # 按类召回（OA 链、ba 链的各步），用于按类归因
            n_te = np.bincount(y_test, minlength=C)
            for arm in ('L0', 'PM', 'CMN', 'Wnat', 'LS', 'CMB', 'W0'):
                rec = np.array([(preds[arm][y_test == c] == c).mean() if n_te[c] else np.nan for c in range(C)])
                perclass.append(dict(target=tgt, feat=feat, arm=arm, **{f'r{c}': rec[c] for c in range(C)},
                                     **{f'n{c}': n_te[c] for c in range(C)}))
            # 配对地块 bootstrap（主链各部分）
            B = 2000
            idx = np.random.default_rng(7).integers(0, len(y_test), (B, len(y_test)))
            chain_arms = ['L0', 'PM', 'CMN', 'Wnat', 'LS', 'CMB', 'W0']
            bm = {a: {'oa': [], 'ba_ge30': []} for a in chain_arms}
            for b in range(B):
                yb = y_test[idx[b]]
                nb = np.bincount(yb, minlength=C)
                big = nb >= MIN_N
                for a in chain_arms:
                    ph = preds[a][idx[b]]
                    corr = ph == yb
                    bm[a]['oa'].append(100 * corr.mean())
                    hit = np.bincount(yb[corr], minlength=C)
                    bm[a]['ba_ge30'].append(100 * np.mean(hit[big] / nb[big]))
            boot_store[(tgt, feat)] = {a: {k: np.array(v) for k, v in d.items()} for a, d in bm.items()}
            # 校准诊断
            w_fva = compute_sample_weight('balanced', cal['y_f_va'])
            p_va_pos = pd.Series(np.arange(len(pool)), index=pool).loc[cal['p_va']].to_numpy()
            w0_pva = P(tgt, feat, 'within', 'pool')[p_va_pos]
            calib_rows.append(dict(
                target=tgt, feat=feat, bcts_temp=s_T,
                loco_implied_max=float(np.average(Lc['fva'], axis=0, weights=w_fva).max()),
                loco_implied_min=float(np.average(Lc['fva'], axis=0, weights=w_fva)[trained_mask(Lc['fva'])].min()),
                loco_ece_fva=ece(Lc['fva'], cal['y_f_va']), w0_ece_pva=ece(w0_pva, cal['y_p_va']),
                lcal_vs_loco_maxdiff=float(np.abs(Lc['test'] - L['test']).max()),
                js_unif=js_unif, js_F=js_F, js_fine=js_fine))

    A = pd.DataFrame(rows)
    A.to_csv(os.path.join(RESULTS_DIR, 'check3_arms_NOISE.csv' if NOISE else 'check3_arms.csv'), index=False)
    R = A[A.split == 'random'].pivot_table(index=['feat', 'target'], columns='arm', values=['oa', 'ba_ge30', 'f1'])
    Cal = pd.DataFrame(calib_rows).set_index(['feat', 'target'])

    print('\n' + '=' * 30 + ' 主分解：训练时链式替换 ' + '=' * 30)
    for metric, chain, names in [
            ('oa', ['L0', 'PM', 'CMN', 'Wnat'], ['类别构成', '样本量', '域残差']),
            ('ba_ge30', ['L0', 'LS', 'CMB', 'W0'], ['类别集合', '样本量', '域残差'])]:
        print(f'\n### {metric} 链：' + ' → '.join(chain))
        for feat in ['ae', 's2']:
            r = R[metric].loc[feat]
            dfp = pd.DataFrame({'total': r[chain[-1]] - r[chain[0]]}, index=r.index)
            for i, nm in enumerate(names):
                dfp[nm] = r[chain[i + 1]] - r[chain[i]]
            print(f'\n[{feat.upper()}] 各国（pp）：')
            print(pd.concat([r[chain].round(1), dfp.round(1)], axis=1).to_string())
            print(f'  均值：' + '  '.join(f'{c}={dfp[c].mean():+.2f}' for c in dfp.columns)
                  + '   中位数：' + '  '.join(f'{c}={dfp[c].median():+.2f}' for c in dfp.columns))
            for nm in names:
                v, lo, hi, l1, l2 = pooled(dfp, nm, 'total', rng)
                print(f'  合并比例 {nm}：{v:.0%}  95% CI [{lo:.0%}, {hi:.0%}]  留一国范围 [{l1:.0%}, {l2:.0%}]')
            if metric == 'oa':
                v, lo, hi, _, _ = pooled(dfp, '类别构成', 'total', rng)
                verdict = '成立' if (v >= 0.30 and lo >= 0.15) else '方向性支持' if v >= 0.30 else '不成立'
                print(f'  → 主要问题 1（{feat.upper()}）：类别构成比例 {v:.0%}，判定：{verdict}')
            # 单国 CI（配对地块 bootstrap）
            ci = []
            for t in r.index:
                bs = boot_store[(t, feat)]
                parts = [bs[chain[i + 1]][metric] - bs[chain[i]][metric] for i in range(3)]
                ci.append(', '.join(f'{nm}[{np.percentile(x, 2.5):+.1f},{np.percentile(x, 97.5):+.1f}]'
                                    for nm, x in zip(names, parts)))
            print('  单国 95% CI：' + ' | '.join(f'{t}: {c}' for t, c in zip(r.index, ci)))
        print(f'\n主要问题 2：AE − S2 的域残差（{metric}，pp）')
        dom = {}
        for feat in ['ae', 's2']:
            r = R[metric].loc[feat]
            dom[feat] = r[chain[-1]] - r[chain[-2]]
        d = dom['ae'] - dom['s2']
        p = wilcoxon(d).pvalue if (d != 0).any() else np.nan
        same = int((np.sign(d) == np.sign(d.mean())).sum())
        verdict = '有差异' if abs(d.mean()) >= 1 and p < 0.05 else '无差异'
        print('  各国：' + '  '.join(f'{t}{v:+.1f}' for t, v in d.items()))
        print(f'  平均 {d.mean():+.2f} pp，Wilcoxon p={p:.3f}，同向 {same}/{len(d)} → {verdict}')

    print('\n' + '=' * 30 + ' 次要分析 ' + '=' * 30)
    print('\n1. 事后修正 2×2（OA；角：LS、PO、W0、WPO；另对照 W-nat）')
    for feat in ['ae', 's2']:
        r = R['oa'].loc[feat]
        PA, CA = r['PO'] - r['LS'], r['WPO'] - r['PO']
        CB, PB = r['W0'] - r['LS'], r['WPO'] - r['W0']
        I = PB - PA
        tot = r['WPO'] - r['L0']
        S = r['LS'] - r['L0']
        shap = S + (PA + PB) / 2
        flag = (I.abs() >= 0.5 * tot.abs())
        print(f'  [{feat.upper()}] 合并：先修先验 {(S + PA).sum() / tot.sum():.0%}，先修条件 {(S + PB).sum() / tot.sum():.0%}，'
              f'Shapley {shap.sum() / tot.sum():.0%}；|交互项|≥0.5|总损失| 的国家 {int(flag.sum())}/{len(flag)}；'
              f'W-nat − WPO 均值 {(r["Wnat"] - r["WPO"]).mean():+.2f} pp')
    print('\n2. 先验修正对平衡精度的影响（ba_ge30，负值 = 伤害）')
    for feat in ['ae', 's2']:
        r = R['ba_ge30'].loc[feat]
        a, b = r['PO'] - r['LS'], r['WPO'] - r['W0']
        print(f'  [{feat.upper()}] ba(PO)−ba(LS) 均值 {a.mean():+.2f}（负 {int((a < 0).sum())}/{len(a)}）；'
              f'ba(WPO)−ba(W0) 均值 {b.mean():+.2f}（负 {int((b < 0).sum())}/{len(b)}）')
    print('\n3. 先验估计（OA；恢复率 = Σ(X−L0)/Σ(PO−L0)，bcts 类以 L0cal、PO_bcts 为基准）')
    E = pd.DataFrame(estim)
    for feat in ['ae', 's2']:
        r = R['oa'].loc[feat]
        base = (r['PO'] - r['L0']).sum()
        base_c = (r['PO_bcts'] - r['L0cal']).sum()
        out = [f'PO={base / len(r):+.2f}pp/国', f'PO_bcts 相对 L0cal={base_c / len(r):+.2f}pp/国']
        for name in ['PEM', 'PEM_LS', 'BBSE_hard', 'BBSE_soft']:
            out.append(f'{name} {(r[name] - r["L0"]).sum() / base:.0%}')
        out.append(f'PEM_bcts {(r["PEM_bcts"] - r["L0cal"]).sum() / base_c:.0%}')
        for tag in ('PK', 'PKj', 'PK_LS'):
            for k in (50, 200):
                cols = [f'{tag}{k}_s{s}' for s in range(3)]
                out.append(f'{tag}{k} {(r[cols].mean(1) - r["L0"]).sum() / base:.0%}')
        print(f'  [{feat.upper()}] ' + '；'.join(out))
        e = E[E.feat == feat].groupby('est')['l1'].mean()
        print('     π̂ 的 L1 误差（均值）：' + '  '.join(f'{k}={v:.3f}' for k, v in e.items() if not re.search(r'_s[12]$', k)))
    conv = E[E.est.isin(['PEM', 'PEM_LS', 'PEM_bcts'])]
    print(f'     EM 收敛比例 {conv.converged.mean():.0%}，迭代次数中位数 {conv.iters_or_cond.median():.0f}')
    print('\n4. BIAS_ORACLE（用目标国 pool 标签校准 LOCO 的决策层，上界）')
    for feat in ['ae', 's2']:
        r_oa, r_ba = R['oa'].loc[feat], R['ba_ge30'].loc[feat]
        print(f'  [{feat.upper()}] OA：ORACLE−L0 {(r_oa["ORACLE_oa"] - r_oa["L0"]).mean():+.2f}，'
              f'Wnat−ORACLE {(r_oa["Wnat"] - r_oa["ORACLE_oa"]).mean():+.2f}；'
              f'ba_ge30：ORACLE−L0 {(r_ba["ORACLE_ba"] - r_ba["L0"]).mean():+.2f}，'
              f'W0−ORACLE {(r_ba["W0"] - r_ba["ORACLE_ba"]).mean():+.2f} pp')
    print('\n5. 按类归因（OA 链：ΔOA = Σ π_test(c)·Δrecall_c，11 国平均，pp）')
    Pc = pd.DataFrame(perclass).set_index(['target', 'feat', 'arm'])
    for feat in ['ae', 's2']:
        out = []
        for a, b, nm in (('L0', 'PM', '类别构成'), ('PM', 'CMN', '样本量'), ('CMN', 'Wnat', '域残差')):
            g = {k: [] for k in GROUPS}
            for t in targets:
                n = np.array([Pc.loc[(t, feat, a), f'n{c}'] for c in range(C)], float)
                d = np.array([Pc.loc[(t, feat, b), f'r{c}'] - Pc.loc[(t, feat, a), f'r{c}'] for c in range(C)])
                contrib = 100 * np.nan_to_num(d) * n / n.sum()
                for k, ids in grp_idx.items():
                    g[k].append(contrib[ids].sum())
            out.append(nm + '（' + '，'.join(f'{k} {np.mean(v):+.2f}' for k, v in g.items()) + '）')
        print(f'  [{feat.upper()}] ' + '；'.join(out))
    print('\n6. 描述量相关（Spearman，11 国）')
    for feat in ['ae', 's2']:
        r = R['oa'].loc[feat]
        comp = r['PM'] - r['L0']
        dom = r['Wnat'] - r['CMN']
        cal_f = Cal.loc[feat]
        print(f'  [{feat.upper()}] 类别构成 vs JS(π_T,均匀) ρ={spearmanr(comp, cal_f.js_unif).correlation:+.2f}；'
              f'vs class_js ρ={spearmanr(comp, cal_f.js_F).correlation:+.2f}；'
              f'域残差 vs 细类构成 JS ρ={spearmanr(dom, cal_f.js_fine).correlation:+.2f}')
    print('\n7. 敏感性')
    S = A[A.split == 'spatial'].pivot_table(index=['feat', 'target'], columns='arm', values=['oa', 'ba_ge30'])
    for feat in ['ae', 's2']:
        for metric, chain in (('oa', ['L0', 'PM', 'CMN', 'Wnat']), ('ba_ge30', ['L0', 'LS', 'CMB', 'W0'])):
            for lab, tab in (('随机划分', R), ('空间分块', S)):
                r = tab[metric].loc[feat]
                tot = (r[chain[-1]] - r[chain[0]]).sum()
                parts = [(r[chain[i + 1]] - r[chain[i]]).sum() / tot for i in range(3)]
                print(f'  [{feat.upper()} {metric} {lab}] 总损失均值 {tot / len(r):+.2f} pp；'
                      f'三部分比例 {parts[0]:.0%} / {parts[1]:.0%} / {parts[2]:.0%}')
        for drop in (['RO'], ['LV'], ['RO', 'LV']):
            r = R['oa'].loc[feat].drop(index=[d for d in drop if d in R['oa'].loc[feat].index])
            tot = (r['Wnat'] - r['L0']).sum()
            print(f'  [{feat.upper()} oa 去掉 {"+".join(drop)}] 类别构成比例 {(r["PM"] - r["L0"]).sum() / tot:.0%}')
        rn = A[(A.split == 'random') & (A.feat == feat)].pivot(index='target', columns='arm', values='ba_ge30_noother')
        tot = (rn['W0'] - rn['L0']).sum()
        print(f'  [{feat.upper()} ba_ge30 去掉 other_crops] 三部分比例 '
              f'{(rn["LS"] - rn["L0"]).sum() / tot:.0%} / {(rn["CMB"] - rn["LS"]).sum() / tot:.0%} / '
              f'{(rn["W0"] - rn["CMB"]).sum() / tot:.0%}')
        r = R['ba_ge30'].loc[feat]
        print(f'  [{feat.upper()} ba_ge30 类别集合阈值 ≥20] LS20−L0 均值 {(r["LS20"] - r["L0"]).mean():+.2f}，'
              f'LS−L0 均值 {(r["LS"] - r["L0"]).mean():+.2f} pp')
    print('\n8. 校准诊断')
    print(Cal[['bcts_temp', 'loco_implied_min', 'loco_implied_max', 'loco_ece_fva', 'w0_ece_pva',
               'lcal_vs_loco_maxdiff']].round(3).to_string())
    meta = pd.read_csv(os.path.join(D, 'supp_meta.csv'))
    print('  最优迭代次数（补充模型）中位数：' + '  '.join(
        f'{m}={int(v)}' for m, v in meta.groupby('model')['best_iter'].median().items())
        + f'；封顶 500 轮的比例 {(meta.best_iter >= 499).mean():.0%}')
    log = open(os.path.join(RESULTS_DIR, 'check3.log')).read()
    its = [int(x) for x in re.findall(r'iters=\s*(\d+)', log)]
    print(f'  原始三模型最优迭代次数：中位数 {np.median(its):.0f}，最小 {min(its)}，封顶比例 {np.mean(np.array(its) >= 499):.0%}')


if __name__ == '__main__':
    main()
