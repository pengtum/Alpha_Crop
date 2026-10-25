"""(e) 方差分解：同一 (单元, 分类器, k) 格内，方法、指导、方法×指导；再把格内方差拆成可复现（跨格一致）与格特有部分。"""
import itertools

from common_bm import *

R = load()


def factorial_ss(arr, names):
    """平衡全因子（无重复）数组的各效应平方和（Möbius 反演边际均值）。返回 {frozenset: SS}。"""
    axes = list(range(arr.ndim))
    means = {}
    for r in range(arr.ndim + 1):
        for H in itertools.combinations(axes, r):
            other = tuple(a for a in axes if a not in H)
            means[H] = arr.mean(axis=other, keepdims=True) if other else arr
    ss = {}
    for r in range(1, arr.ndim + 1):
        for F in itertools.combinations(axes, r):
            e = 0
            for rr in range(len(F) + 1):
                for H in itertools.combinations(F, rr):
                    e = e + (-1) ** (len(F) - len(H)) * means[H]
            e = np.broadcast_to(e, arr.shape)
            ss[frozenset(names[a] for a in F)] = float((e ** 2).sum())
    return ss


lines = []
for metric in ('ba_ge30', 'oa'):
    # ---------- 全 8 方法，重复 0（22 单元）
    sub = R[R.cond.isin(['AE', 'S2']) & R.method.isin(ALL8) & R.k.isin([50, 200]) & R.clf.isin(['s2', 'ae']) & (R.rep == 0)]
    P = sub.set_index(['target', 'split', 'clf', 'k', 'method', 'cond'])[metric]
    idx = pd.MultiIndex.from_product([COUNTRIES, [0, 1], ['s2', 'ae'], [50, 200], ALL8, ['AE', 'S2']])
    arr = P.reindex(idx).to_numpy().reshape(11, 2, 2, 2, 8, 2)
    assert not np.isnan(arr).any()
    names = ['T', 'S', 'C', 'K', 'M', 'G']
    ss = factorial_ss(arr, names)
    within = {F: v for F, v in ss.items() if F & {'M', 'G'}}
    tot = sum(within.values())
    # 简单格内三分（与 analyze_G 的定义相同）：M 类 = 所有只含 M（不含 G）的项；G 类同理；交互 = 含 M 与 G 的项
    sm = sum(v for F, v in within.items() if 'M' in F and 'G' not in F)
    sg = sum(v for F, v in within.items() if 'G' in F and 'M' not in F)
    si = sum(v for F, v in within.items() if 'M' in F and 'G' in F)
    lines.append(f'\n===== {metric}：8 方法 × 2 指导，重复 0 的 22 单元 × 2 分类器 × 2 k = 88 格 =====')
    lines.append(f'格内三分：方法 {sm / tot:.1%}，指导 {sg / tot:.1%}，方法×指导 {si / tot:.1%}')

    # 细分：把含 M/G 的项按"与国家/划分/分类器/k 的交互"归类
    def lab(F):
        core = ''.join(x for x in 'MG' if x in F)
        rest = F - {'M', 'G'}
        if not rest:
            ctx = '全局'
        elif rest <= {'C', 'K'}:
            ctx = '×设置(clf/k)'
        elif 'S' in rest:
            ctx = '×划分(国家内)'
        elif rest == {'T'}:
            ctx = '×国家'
        else:
            ctx = '×国家×设置'
        return {'M': '方法', 'G': '指导', 'MG': '方法×指导'}[core] + ctx
    grp = {}
    for F, v in within.items():
        grp[lab(F)] = grp.get(lab(F), 0) + v
    G_ = pd.Series(grp).sort_values(ascending=False) / tot
    lines.append('细分（占格内方差）：\n' + G_.map(lambda x: f'{x:.1%}').to_string())
    # 均方比：方法×国家 对 方法×划分(国家)，指导×国家、方法×指导×国家 对 各自的划分项
    def ms(F):
        F = frozenset(F)
        df = np.prod([{'T': 10, 'S': 1, 'C': 1, 'K': 1, 'M': 7, 'G': 1}[x] for x in F])
        return ss[F], df
    for eff, err in ((('M', 'T'), [('M', 'S'), ('M', 'T', 'S')]), (('G', 'T'), [('G', 'S'), ('G', 'T', 'S')]),
                     (('M', 'G', 'T'), [('M', 'G', 'S'), ('M', 'G', 'T', 'S')]),
                     (('M',), [('M', 'T')]), (('G',), [('G', 'T')]), (('M', 'G'), [('M', 'G', 'T')])):
        s1, d1 = ms(eff)
        s2_ = sum(ms(e)[0] for e in err)
        d2 = sum(ms(e)[1] for e in err)
        F = (s1 / d1) / (s2_ / d2)
        lines.append(f'  {"×".join(eff)} 对 {"+".join("×".join(e) for e in err)}：F({d1},{d2}) = {F:.2f}，p = {stats.f.sf(F, d1, d2):.3g}' +
                     ('（国家为随机因素：误差取与国家的交互）' if err[0] == eff + ('T',) else '（误差取国家内划分间的交互）'))
    # 去掉 kcenter / 去掉 RO 的三分
    for lab2, keepM, keepT in (('去掉 kcenter', [m for m in ALL8 if m != 'kcenter'], COUNTRIES),
                               ('去掉 RO', ALL8, [c for c in COUNTRIES if c != 'RO']),
                               ('只 E1（重复 0）', E1, COUNTRIES), ('只预测/混合法', PRED + HYB, COUNTRIES)):
        a2 = arr[[COUNTRIES.index(c) for c in keepT]][:, :, :, :, [ALL8.index(m) for m in keepM]]
        s2s = factorial_ss(a2, names)
        w2 = {F: v for F, v in s2s.items() if F & {'M', 'G'}}
        t2 = sum(w2.values())
        lines.append(f'{lab2}：方法 {sum(v for F, v in w2.items() if "M" in F and "G" not in F) / t2:.1%}，'
                     f'指导 {sum(v for F, v in w2.items() if "G" in F and "M" not in F) / t2:.1%}，'
                     f'方法×指导 {sum(v for F, v in w2.items() if "M" in F and "G" in F) / t2:.1%}；'
                     f'其中全局方法主效应 {w2[frozenset("M")] / t2:.1%}，全局指导主效应 {w2[frozenset("G")] / t2:.1%}，全局方法×指导 {w2[frozenset("MG")] / t2:.1%}')
    # ---------- E1 4 方法，44 单元（复现 analyze_G 的 66.8/6.9/26.3）
    sub = R[R.cond.isin(['AE', 'S2']) & R.method.isin(E1) & R.k.isin([50, 200]) & R.clf.isin(['s2', 'ae'])]
    P = sub.set_index(['target', 'split', 'rep', 'clf', 'k', 'method', 'cond'])[metric]
    idx = pd.MultiIndex.from_product([COUNTRIES, [0, 1], [0, 1], ['s2', 'ae'], [50, 200], E1, ['AE', 'S2']])
    arr = P.reindex(idx).to_numpy().reshape(11, 2, 2, 2, 2, 4, 2)
    names7 = ['T', 'S', 'P', 'C', 'K', 'M', 'G']
    ss = factorial_ss(arr, names7)
    within = {F: v for F, v in ss.items() if F & {'M', 'G'}}
    tot = sum(within.values())
    lines.append(f'E1 × 44 单元（对照 report_G）：方法 {sum(v for F, v in within.items() if "M" in F and "G" not in F) / tot:.1%}，'
                 f'指导 {sum(v for F, v in within.items() if "G" in F and "M" not in F) / tot:.1%}，'
                 f'方法×指导 {sum(v for F, v in within.items() if "M" in F and "G" in F) / tot:.1%}；'
                 f'全局方法 {within[frozenset("M")] / tot:.1%}，全局指导 {within[frozenset("G")] / tot:.1%}，全局方法×指导 {within[frozenset("MG")] / tot:.1%}，'
                 f'含重复(候选池)P 的项 {sum(v for F, v in within.items() if "P" in F) / tot:.1%}')
open(f'{OUTD}/e_variance.log', 'w').write('\n'.join(lines))
print('\n'.join(lines))
