"""
单一来源数字表（ANALYSIS_PLAN_guidance_rev4.md 第 D3 节）。

从结果文件重新计算论文 headline 数字（REPOSITION_AFTER_CHENG2026_2026-09-27.md 第 4 节 H1–H6 与主要次要结果），
输出 NUMBERS.csv、NUMBERS.md、discrepancies.txt（均写在本脚本所在目录）。

规则（rev4 与任务说明）：
- 只读结果文件，不训练任何模型，不拟合决策头（决策头数字直接从已有逐单元行重算）。
- 严禁读取 results/guidance_G_ml/ 下的任何文件：所有读路径经 _chk() 断言。rev2 学习器（LR、MLP、RF）的行一律占位"待 rev2"。
- 置信区间统一为 95%；需要 90% 的（TOST 等价判定）另列，并注明用途。
- 推断单位为国家（11 国 t 检验，df=10；去掉 RO、PT 时 9 国，df=8），与 analyze_G.t_summary 一致。
- 能从原始行重算的就重算，并与文档数字核对；只能从报告读取的，写明读取位置。
线程 ≤2。不写入项目目录（禁止生成 .pyc）。

v2（2026-09-27）：落实 d3_review 复核意见（证据等级、测试集复用、预注册标注、剂量斜率拆行、补去 RO/PT 值、
补 Cheng 讨论引用的数字、M9/M9b 改为从 alt_long.csv / part1_rows.csv 重算、计数更正、discrepancies 补条目），
并把 rev4 D1（decision_curve/）与 D2（cheng_comparable/）的已有输出填入对应占位行。旧版见 numbers_table_v1.py。

用法：/Users/perng_mini/AlphaCrops_local/.venv/bin/python numbers_table.py
"""
import os
import sys

sys.dont_write_bytecode = True  # 不在项目目录生成 __pycache__
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '2'
os.environ.pop('GUIDE_OUT', None)

import glob  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

LE = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, LE)
import analyze_G as A  # noqa: E402
from guidance_G import E1, FAMILIES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = '/Users/perng_mini/AlphaCrops_local'
RES = f'{LOCAL}/results'
GRES = f'{RES}/guidance_G'
VER = f'{LOCAL}/verify'
STORY = f'{VER}/storyline'
D1DIR = f'{STORY}/decision_curve'
D2DIR = f'{STORY}/cheng_comparable'
PROJ = os.path.dirname(LE)  # Alpha_Crops 项目根（只读）
OLDEXP = f'{PROJ}/results/experiments'  # 原论文（修复前数据）的实验结果，只读
FORBIDDEN = 'guidance_G_ml'
COUNTRIES = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']
ROPT = ['RO', 'PT']
ALL8 = E1 + FAMILIES['pred'] + FAMILIES['hyb']
assert FORBIDDEN not in A.OUT and os.path.realpath(A.OUT) == os.path.realpath(GRES), A.OUT


def _chk(path):
    assert FORBIDDEN not in path and FORBIDDEN not in os.path.realpath(path), f'禁止读取 {path}'
    return path


def rcsv(path, **kw):
    return pd.read_csv(_chk(path), **kw)


def rtxt(path):
    with open(_chk(path), encoding='utf-8') as f:
        return f.read()


def rpkl(path):
    return pd.read_pickle(_chk(path))


# ------------------------------------------------------------------ 公共
TS = A.t_summary


def cv_from_unit(d, col='d'):
    """单元值 → 国家值（单元等权）。d 需含 target。"""
    return d.groupby('target')[col].mean()


def pct_fmt(x):
    return None if x is None else float(x)


ROWS = []


def add(id, group, desc, value, unit='pp', s=None, lo95=None, hi95=None, lo90=None, hi90=None, ci90_use='',
        n=None, n_pos=None, p=None, p_note='', noROPT=None, noROPT_ci=None, noROPT_na='', grade='', prereg='', reuse='',
        learner='LightGBM', metric='', source='', func='', mode='重算', doc=None, doc_src='', per_country=None,
        note=''):
    """一行。s 为 t_summary 字典时自动填 CI/n/p。doc = {'v': '字符串', 'lo': .., 'hi': .., 'lo90': .., 'hi90': ..}。
    noROPT_na：去 RO/PT 值不适用或未算时的原因（v2：不再留空）。"""
    if s is not None:
        value = s['mean'] if value is None else value
        lo95, hi95, lo90, hi90 = s['lo95'], s['hi95'], s['lo90'], s['hi90']
        n, n_pos, p = s['n'], s['n_pos'], s['p']
    if not ci90_use:
        lo90 = hi90 = None
    if noROPT is None and not noROPT_na and mode == '占位':
        noROPT_na = '待 rev2' if 'rev2' in str(value) else '待定'
    r = dict(id=id, group=group, desc=desc, value=value, unit=unit, ci95_lo=lo95, ci95_hi=hi95,
             ci90_lo=lo90, ci90_hi=hi90, ci90_use=ci90_use, n=n, n_pos=n_pos, p=p, p_note=p_note,
             noROPT=noROPT, noROPT_ci=noROPT_ci, noROPT_na=noROPT_na, grade=grade, prereg=prereg, test_reuse=reuse,
             learner=learner,
             metric=metric, source=source, func=func, mode=mode, doc=json.dumps(doc, ensure_ascii=False) if doc else '',
             doc_src=doc_src,
             per_country=json.dumps({k: round(float(v), 3) for k, v in per_country.items()}, ensure_ascii=False)
             if per_country is not None else '',
             note=note)
    ROWS.append(r)
    return r


def noropt_str(v, digits=3):
    """去掉 RO、PT 的 9 国 t 汇总（值, CI 字符串）。"""
    s = TS(v.drop([c for c in ROPT if c in v.index]))
    return s['mean'], f'[{s["lo95"]:+.{digits}f}, {s["hi95"]:+.{digits}f}]，{s["n_pos"]}/{s["n"]}'


# 测试集复用口径
RU_G_PRE = 'G s0/s1：层级 1+2 首次使用（预注册，合规）；此后被多次复用'
RU_G_POST = '是：G s0/s1 已复用（事后分析）'
RU_C3_PRE = ('检验 3 预注册分析（随机 70/30、无缓冲）；其中 PT、SE、SK、LV 的测试集与检验 2 逐位相同，检验 2 已在其上评估过；'
             '之后又用于选开发国、headroom 与 G 设计审查（TESTSET_AUDIT §3–4）')
RU_C3_SP = ('检验 3 预注册敏感性，20 km 格网 test_sp（与随机划分不同，也无缓冲；TESTSET_AUDIT §2）；'
            'TESTSET_AUDIT §3–4 未单列 test_sp 与检验 2 的重合及其后续用途')
RU_C3_POST = '是：检验 3 划分已看过（事后）；PT、SE、SK、LV 的测试集与检验 2 逐位相同（TESTSET_AUDIT §3）'
RU_NOTEST = '不用测试标签（重复 0 的候选池）'
RU_C1 = '国别留出（pairwise_v2 两两迁移），未用于调参'
RU_MD = '是：G s0/s1 划分 0/1 已复用（模型诊断，事后）'

SRC_G = f'{GRES}/rows_t*_*.csv'


# ================================================================== H1：缺口解剖（检验 3）
def pooled_share(dfp, part, total, rng, B=5000):
    """与 analyze_check3.pooled 逐行相同（同一随机数消耗顺序）。"""
    t = dfp.index.to_numpy()
    val = dfp[part].sum() / dfp[total].sum()
    boot = []
    for _ in range(B):
        s = rng.choice(len(t), len(t))
        den = dfp[total].iloc[s].sum()
        boot.append(dfp[part].iloc[s].sum() / den if den != 0 else np.nan)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return val, lo, hi


def chain_df(r, chain, names):
    dfp = pd.DataFrame({'total': r[chain[-1]] - r[chain[0]]}, index=r.index)
    for i, nm in enumerate(names):
        dfp[nm] = r[chain[i + 1]] - r[chain[i]]
    return dfp


CHAINS = [('oa', ['L0', 'PM', 'CMN', 'Wnat'], ['comp', 'size', 'resid']),
          ('ba_ge30', ['L0', 'LS', 'CMB', 'W0'], ['set', 'size', 'resid'])]


def h1_check3():
    src = f'{RES}/check3_arms.csv'
    Aa = rcsv(src)
    Rr = Aa[Aa.split == 'random'].pivot_table(index=['feat', 'target'], columns='arm', values=['oa', 'ba_ge30', 'f1'])
    Ss = Aa[Aa.split == 'spatial'].pivot_table(index=['feat', 'target'], columns='arm', values=['oa', 'ba_ge30'])
    rng = np.random.default_rng(42)  # analyze_check3：RANDOM_STATE=42，只被 pooled 消耗
    res = {}
    for metric, chain, names in CHAINS:
        for feat in ('ae', 's2'):
            dfp = chain_df(Rr[metric].loc[feat], chain, names)
            for nm in names:
                res[('es', metric, feat, nm)] = pooled_share(dfp, nm, 'total', rng)
            if metric == 'oa':
                pooled_share(dfp, 'comp', 'total', rng)  # 原脚本判定行再消耗一次
            res[('es', metric, feat, 'dfp')] = dfp
    # 固定 300 棵（事后）：analyze_check3_capacity 的随机数顺序
    srcc = f'{RES}/check3_capacity_arms.csv'
    Cc = rcsv(srcc)
    T = Cc.pivot_table(index=['feat', 'cap', 'target'], columns='arm', values=['oa', 'ba_ge30'])
    rng2 = np.random.default_rng(42)
    for metric, chain, names in CHAINS:
        for feat in ('ae', 's2'):
            for cap in ('es', 'c300'):
                dfp = chain_df(T[metric].loc[(feat, cap)], chain, names)
                for nm in names:
                    res[(f'cap_{cap}', metric, feat, nm)] = pooled_share(dfp, nm, 'total', rng2)
                res[(f'cap_{cap}', metric, feat, 'dfp')] = dfp
    fn = 'numbers_table.h1_check3（复现 analyze_check3.pooled 与 analyze_check3_capacity 的随机数顺序）'
    docs = {('es', 'oa', 'ae'): ('63', '41', '76'), ('es', 'oa', 's2'): ('67', '53', '78'),
            ('cap_c300', 'oa', 'ae'): ('46', '22', '69'), ('cap_c300', 'oa', 's2'): ('44', '26', '60')}
    # --- OA 类别构成份额
    for key, lab, setting, srcx, pre, ru, gr in (
            ('es', '早停（预注册设置）', 'es', src, '是（check3 08b3d271 + rev1 fc6aeffd）', RU_C3_PRE, 'B'),
            ('cap_c300', '固定 300 棵（事后容量检验）', 'c300', srcc, '否（事后补充 ANALYSIS_ADDENDUM_check3_capacity）', RU_C3_POST, 'B（区间 44–67% 作效应大小）')):
        for feat in ('ae', 's2'):
            v, lo, hi = res[(key, 'oa', feat, 'comp')]
            dfp = res[(key, 'oa', feat, 'dfp')]
            d9 = dfp.drop(ROPT)
            d = docs[(key, 'oa', feat)]
            add(f'H1.oa_comp.{setting}.{feat}', 'H1', f'OA 跨境损失中类别构成的合并份额（{lab}，{feat.upper()} 特征）',
                100 * v, '%', lo95=100 * lo, hi95=100 * hi, n=11, noROPT=100 * d9['comp'].sum() / d9['total'].sum(),
                grade=gr, prereg=pre, reuse=ru, metric='OA；Σ(PM−L0)/Σ(Wnat−L0)',
                source=srcx, func=fn, doc={'v': d[0], 'lo': d[1], 'hi': d[2]}, doc_src='REPOSITION §4 H1；CHECK3_RESULTS',
                per_country=(100 * dfp['comp'] / dfp['total']).to_dict(),
                note='95% CI = 按目标国 cluster bootstrap（B=5000），逐位复现原脚本随机数；去 RO/PT 只给点估计。')
    # 空间分块（预注册敏感性，无 CI）
    for feat, dv in (('ae', '66'), ('s2', '72')):
        r = Ss['oa'].loc[feat]
        dfp = chain_df(r, CHAINS[0][1], CHAINS[0][2])
        d9 = dfp.drop(ROPT)
        add(f'H1.oa_comp.spatial.{feat}', 'H1', f'OA 类别构成合并份额（20 km 空间分块测试，{feat.upper()}）',
            100 * dfp['comp'].sum() / dfp['total'].sum(), '%', n=11, noROPT=100 * d9['comp'].sum() / d9['total'].sum(),
            grade='B', prereg='是（rev1 敏感性）', reuse=RU_C3_SP, metric='OA', source=src, func='numbers_table.h1_check3',
            doc={'v': dv}, doc_src='REPOSITION §4 H1；CONCLUSIONS C3a', note='原报告未给 CI；去 RO/PT 只给点估计。')
    # BA 类别集合份额与域残差份额
    for key, setting, srcx, ru, pre in (('es', 'es', src, RU_C3_PRE, '是'), ('cap_c300', 'c300', srcc, RU_C3_POST, '否（事后）')):
        for feat in ('ae', 's2'):
            for nm, lab, dmap in (('set', '类别集合', {('es', 'ae'): ('13', '0', '34'), ('es', 's2'): ('15', '0', '33'),
                                                     ('c300', 'ae'): ('12', '0', '27'), ('c300', 's2'): ('11', '0', '26')}),
                                  ('resid', '域残差', {('es', 'ae'): ('115', '80', '156'), ('es', 's2'): ('106', '79', '136'),
                                                    ('c300', 'ae'): ('143', '93', '204'), ('c300', 's2'): ('131', '93', '183')})):
                v, lo, hi = res[(key, 'ba_ge30', feat, nm)]
                dfp = res[(key, 'ba_ge30', feat, 'dfp')]
                d9 = dfp.drop(ROPT)
                d = dmap[(setting, feat)]
                add(f'H1.ba_{nm}.{setting}.{feat}', 'H1', f'ba_ge30 跨境损失中{lab}的合并份额（{setting}，{feat.upper()}）',
                    100 * v, '%', lo95=100 * lo, hi95=100 * hi, n=11, noROPT=100 * d9[nm].sum() / d9['total'].sum(),
                    grade='B', prereg=pre, reuse=ru, metric='ba_ge30；L0→LS→CMB→W0', source=srcx, func=fn,
                    doc={'v': d[0], 'lo': d[1], 'hi': d[2]}, doc_src='CHECK3_RESULTS；check3(_capacity)_report.txt')
    for feat, dv in (('ae', '114'), ('s2', '108')):
        r = Ss['ba_ge30'].loc[feat]
        dfp = chain_df(r, CHAINS[1][1], CHAINS[1][2])
        d9 = dfp.drop(ROPT)
        add(f'H1.ba_resid.spatial.{feat}', 'H1', f'ba_ge30 域残差合并份额（20 km 空间分块，{feat.upper()}）',
            100 * dfp['resid'].sum() / dfp['total'].sum(), '%', n=11, noROPT=100 * d9['resid'].sum() / d9['total'].sum(),
            grade='B', prereg='是（rev1 敏感性）', reuse=RU_C3_SP,
            metric='ba_ge30', source=src, func='numbers_table.h1_check3', doc={'v': dv}, doc_src='CONCLUSIONS C3b',
            note='原报告未给 CI；去 RO/PT 只给点估计（v2 补算）。')
    # 总损失（t 区间为本表新算）
    for key, setting, srcx, ru in (('es', 'es', src, RU_C3_PRE), ('cap_c300', 'c300', srcc, RU_C3_POST)):
        for metric, lab in (('oa', 'OA'), ('ba_ge30', 'ba_ge30')):
            for feat in ('ae', 's2'):
                tot = res[(key, metric, feat, 'dfp')]['total']
                dmap = {('es', 'oa', 'ae'): '16.59', ('es', 'oa', 's2'): '20.03', ('c300', 'oa', 'ae'): '11.33',
                        ('c300', 'oa', 's2'): '13.07', ('es', 'ba_ge30', 'ae'): '17.82', ('es', 'ba_ge30', 's2'): '18.65',
                        ('c300', 'ba_ge30', 'ae'): '12.88', ('c300', 'ba_ge30', 's2'): '13.47'}
                m9, c9 = noropt_str(tot, 2)
                add(f'H1.total.{metric}.{setting}.{feat}', 'H1', f'{lab} 跨境总损失（本国模型 − L0，{setting}，{feat.upper()}）',
                    None, 'pp', s=TS(tot), noROPT=m9, noROPT_ci=c9, grade='B', prereg='是' if setting == 'es' else '否（事后）',
                    reuse=ru, metric=lab, source=srcx, func='numbers_table.h1_check3 + analyze_G.t_summary',
                    doc={'v': dmap[(setting, metric, feat)]}, doc_src='check3(_capacity)_report.txt', per_country=tot.to_dict(),
                    note='95% t 区间为本表新算（原报告只给均值）。')
    # AE − S2 域残差（主要问题 2）
    for metric, lab, dv, dp in (('oa', 'OA', '+0.00', '0.577'), ('ba_ge30', 'ba_ge30', '+0.73', '0.700')):
        d = res[('es', metric, 'ae', 'dfp')]['resid'] - res[('es', metric, 's2', 'dfp')]['resid']
        w = stats.wilcoxon(d).pvalue
        m9, c9 = noropt_str(d, 2)
        add(f'H1.resid_ae_s2.{metric}.es', 'H1', f'AE − S2 的域残差（{lab}，早停，主要问题 2）', None, 'pp', s=TS(d),
            p_note=f'原判定用 Wilcoxon p={w:.3f}；p 列为 t 检验', noROPT=m9, noROPT_ci=c9, grade='B',
            prereg='是（主要问题 2）', reuse=RU_C3_PRE, metric=lab, source=src,
            func='numbers_table.h1_check3（wilcoxon 与 analyze_check3 相同）',
            doc={'v': dv, 'p_wilcoxon': dp}, doc_src='CHECK3_RESULTS；REPOSITION H1', per_country=d.to_dict(),
            note=f'Wilcoxon p={w:.3f}（预注册检验）；95% t 区间为本表新算。')
        ROWS[-1]['_wilcoxon'] = w
    d = res[('cap_c300', 'oa', 'ae', 'dfp')]['resid'] - res[('cap_c300', 'oa', 's2', 'dfp')]['resid']
    w = stats.wilcoxon(d).pvalue
    m9, c9 = noropt_str(d, 2)
    add('H1.resid_ae_s2.oa.c300', 'H1', 'AE − S2 的 OA 域残差（固定 300 棵，事后边缘信号）', None, 'pp', s=TS(d),
        p_note=f'Wilcoxon p={w:.3f}', noROPT=m9, noROPT_ci=c9, grade='C', prereg='否（事后）', reuse=RU_C3_POST, metric='OA',
        source=srcc, func='numbers_table.h1_check3', doc={'v': '-2.4', 'p_wilcoxon': '0.054'}, doc_src='CHECK3_RESULTS',
        per_country=d.to_dict(), note='负值 = AE 域残差更小。')
    # BIAS_ORACLE（检验 3 的决策层上界，早停）
    for metric, arm, lab in (('oa', 'ORACLE_oa', 'OA'), ('ba_ge30', 'ORACLE_ba', 'ba_ge30')):
        for feat in ('ae', 's2'):
            r = Rr[metric].loc[feat]
            g = r[arm] - r['L0']
            dmap = {('oa', 'ae'): '+11.28', ('oa', 's2'): '+13.51', ('ba_ge30', 'ae'): '+6.96', ('ba_ge30', 's2'): '+5.88'}
            m9, c9 = noropt_str(g, 2)
            tot = res[('es', metric, feat, 'dfp')]['total']
            add(f'H1.bias_oracle.{metric}.{feat}', 'H1附', f'检验 3 BIAS_ORACLE：只校准决策层（池标签拟合温度+偏置）相对 L0 的增益（{lab}，早停，{feat.upper()}）',
                None, 'pp', s=TS(g), noROPT=m9, noROPT_ci=c9, grade='C', prereg='是（rev1 次要 4，描述）', reuse=RU_C3_PRE,
                metric=lab, source=src, func='numbers_table.h1_check3', doc={'v': dmap[(metric, feat)]},
                doc_src='CHECK3_RESULTS；check3_report 次要 4', per_country=g.to_dict(),
                note=f'上界；用 pool 全部标签。合并份额 = {100 * g.sum() / tot.sum():.1f}%（占该链总损失）。')
    # 先验：PO/PK/PEM（H5 使用，放这里是因为同一 arms 文件）
    return Rr, res, Cc


# ------------------------------------------------------------------ H1：只用目标池 vs 只用外国（模型诊断）
GR_TGTONLY = 'C（事后、缓冲格网、复用测试集）'
NOTE_TGTONLY = ('单个数字为 C（事后、G 缓冲格网、复用测试集）；它支持 H1 的 B 级方向结论'
                '（CONCLUSIONS §2 第 3 条〔检验 3 + 模型诊断；B〕，§3 建议 M3 由 A 降为 B）。v1 误标为 A。')
SRC_ALT = f'{VER}/model_diag/verify_fit/alt_long.csv'


def alt_pivot(model, ck):
    """模型诊断 5 国试点（划分 1）：index=(tgt, feat)，columns=tset∈{k0, tgt}，值为测试 ba。"""
    a = rcsv(SRC_ALT)
    x = a[(a.model == model) & (a.ck == ck)]
    return x.pivot_table(index=['tgt', 'feat'], columns='tset', values='ba')


def h1_target_only():
    src0 = f'{VER}/model_diag/fit/table_upper_bound.csv'
    U = rcsv(src0)
    for feat in ('ae', 's2'):
        u = U[U.feat == feat].set_index('tgt')
        for metric, col_t, col_0, dv in (('ba_ge30', 'tgt300_ba', 'k0_ba', {'ae': '12.81', 's2': '13.03'}),
                                          ('OA', 'tgt300_oa', 'k0_oa', {'ae': '10.87', 's2': '11.90'})):
            g = u[col_t] - u[col_0]
            m9, c9 = noropt_str(g, 2)
            add(f'H1.tgtonly.s0.{metric}.{feat}', 'H1', f'只用目标池约 3 万标签 − 只用外国（LightGBM 300 轮，划分 0，{metric}，{feat.upper()}）',
                None, 'pp', s=TS(g), noROPT=m9, noROPT_ci=c9, grade=GR_TGTONLY, prereg='否（模型诊断，事后）',
                reuse=RU_MD, metric=metric, source=src0, func='numbers_table.h1_target_only',
                doc={'v': dv[feat]}, doc_src='report_fit.txt；MODEL_DIAGNOSIS M3（"约 13 pp BA、11–12 pp OA"）',
                per_country=g.to_dict(), note=NOTE_TGTONLY)
    src1 = f'{VER}/model_diag/verify_fit/vf_long.csv'
    V = rcsv(src1)
    x = V[(V.ev == 'test') & (V.it == 300)]
    P = x.pivot_table(index=['tgt', 'feat'], columns='kind', values=['ba', 'oa'])
    for metric, dv in (('ba', {'ae': '14.4', 's2': '14.4'}), ('oa', {'ae': '12.2', 's2': '12.4'})):
        g = (P[metric]['tgt'] - P[metric]['k0'])
        npos_all = int((g > 0).sum())
        for feat in ('ae', 's2'):
            gf = g.xs(feat, level='feat')
            m9, c9 = noropt_str(gf, 2)
            add(f'H1.tgtonly.s1.{metric}.{feat}', 'H1', f'只用目标池 − 只用外国（划分 1 独立重训，{metric}，{feat.upper()}）',
                None, 'pp', s=TS(gf), noROPT=m9, noROPT_ci=c9, grade=GR_TGTONLY, prereg='否（事后复核）', reuse=RU_MD,
                metric=metric, source=src1, func='numbers_table.h1_target_only', doc={'v': dv[feat]},
                doc_src='MODEL_DIAGNOSIS §3（"划分 1 上高 14.4/14.4 pp ba、12.2/12.4 pp OA，22/22"）',
                per_country=gf.to_dict(), note=f'22 个（国家×特征）组合中 {npos_all}/22 为正。' + NOTE_TGTONLY)
    # MLP / LR（模型诊断的 5 国试点，非 rev2）：v2 改为从 alt_long.csv 重算（与 analyze_alt.py 的"最终点"同式）
    for lrn, ck, name in (('mlp', '40', 'MLP'), ('lr', 'C1.0', 'LR（C=1）')):
        x = alt_pivot(lrn, ck)
        g = (x['tgt'] - x['k0']).unstack('feat')
        g9 = g.drop([c for c in ROPT if c in g.index])
        sa, ss = TS(g['ae']), TS(g['s2'])
        add(f'H1.tgtonly.s1.ba.{lrn}', 'H1', f'只用目标池 − 只用外国的 ba 差距（{name}，模型诊断 5 国试点，划分 1）',
            f'AE {g["ae"].mean():+.2f} / S2 {g["s2"].mean():+.2f}', 'pp', n=len(g),
            noROPT=f'AE {g9["ae"].mean():+.2f} / S2 {g9["s2"].mean():+.2f}',
            noROPT_ci=f'去 PT 后 n={len(g9)}（5 国中无 RO）', grade='C（n=5）', prereg='否（事后）', reuse=RU_MD,
            learner=name + '（MODEL_DIAGNOSIS 试点，不属于 rev2 估计量）', metric='ba_ge30', source=SRC_ALT,
            func='numbers_table.alt_pivot（与 verify_fit/analyze_alt.py 的"最终点"同式）', mode='重算',
            doc={'v_ae': {'mlp': '15', 'lr': '23'}[lrn], 'v_s2': {'mlp': '15', 'lr': '19'}[lrn]},
            doc_src='MODEL_DIAGNOSIS M3（MLP 15、LR 19–23 pp）',
            per_country={f'{c}|{f}': g.loc[c, f] for c in g.index for f in ('ae', 's2')},
            note=f'只有 5 国（{",".join(g.index)}，划分 1），n=5，不是 11 国。95% t 区间：AE [{sa["lo95"]:+.2f}, {sa["hi95"]:+.2f}]，'
                 f'S2 [{ss["lo95"]:+.2f}, {ss["hi95"]:+.2f}]。')


# ------------------------------------------------------------------ H1 附：缓冲格网决策层上界
def h1_decision_upper():
    src = f'{VER}/reposition/decision_vs_representation/oracle_G_buffered_rows.csv'
    d = rcsv(src)
    cvv = d.groupby(['target', 'feat'])[['dec_oa', 'dec_ba']].mean()
    dm = {('dec_oa', 'ae'): '+6.66', ('dec_oa', 's2'): '+6.38', ('dec_ba', 'ae'): '+10.45', ('dec_ba', 's2'): '+9.28'}
    for col, lab in (('dec_oa', 'OA'), ('dec_ba', 'ba_ge30（BA 版：回到出现类均匀先验）')):
        for feat in ('ae', 's2'):
            g = cvv.xs(feat, level='feat')[col]
            m9, c9 = noropt_str(g, 2)
            add(f'H1附.dec_upper.{col}.{feat}', 'H1附', f'缓冲格网上的决策层上界：候选池约 3 万标签拟合温度+11 个偏置后相对 k=0（{lab}，{feat.upper()}，两划分平均）',
                None, 'pp', s=TS(g), noROPT=m9, noROPT_ci=c9, grade='C（正文，标明上界）', prereg='否（事后）', reuse=RU_G_POST,
                metric=lab, source=src, func='numbers_table.h1_decision_upper（行来自 oracle_G_buffered.py 的 fit_bcts，无 λ）',
                doc={'v': dm[(col, feat)]}, doc_src='REPOSITION H1 附；oracle_G_buffered.log', per_country=g.to_dict())
    s0 = d[d.split == 0].copy()
    for feat in ('ae', 's2'):
        z = s0[s0.feat == feat].set_index('target')
        for m, lab, dv, dv9 in (('oa', 'OA', {'ae': '57', 's2': '53'}, {'ae': '37', 's2': '38'}),
                                ('ba', 'ba_ge30', {'ae': '78', 's2': '71'}, {'ae': '87', 's2': '80'})):
            tot = z[f'tgt300_{m}'] - z[f'k0_{m}']
            dec = z[f'dec_{m}']
            keep = ~z.index.isin(ROPT)
            add(f'H1附.dec_share.s0.{m}.{feat}', 'H1附', f'决策层上界占"只用目标池"差距的合并份额（划分 0，{lab}，{feat.upper()}）',
                100 * dec.sum() / tot.sum(), '%', n=11, noROPT=100 * dec[keep].sum() / tot[keep].sum(), grade='C',
                prereg='否（事后）', reuse=RU_G_POST, metric=lab, source=src,
                func='numbers_table.h1_decision_upper', doc={'v': dv[feat], 'noROPT': dv9[feat]},
                doc_src='REPOSITION H1 附；oracle_G_buffered.log', note='SE 的 BA 剩余为负，参照不是严格上限；只报 pp 更稳。')
            left = z[f'tgt300_{m}'] - z[f'ora_{m}']
            m9, c9 = noropt_str(left, 2)
            add(f'H1附.repr_left.s0.{m}.{feat}', 'H1附', f'表征层剩余（只用目标池 − 决策层上界，划分 0，{lab}，{feat.upper()}）',
                None, 'pp', s=TS(left), noROPT=m9, noROPT_ci=c9, grade='C', prereg='否（事后）', reuse=RU_G_POST, metric=lab,
                source=src, func='numbers_table.h1_decision_upper',
                doc={'v': {('oa', 'ae'): '+4.62', ('oa', 's2'): '+5.54', ('ba', 'ae'): '+2.78', ('ba', 's2'): '+3.73'}[(m, feat)]},
                doc_src='oracle_G_buffered.log', per_country=left.to_dict())
        add(f'H1附.PT_left_oa.{feat}', 'H1附', f'PT：决策层校准后 OA 仍差（只用目标池 − 上界，划分 0，{feat.upper()}）',
            float(z.loc['PT', 'tgt300_oa'] - z.loc['PT', 'ora_oa']), 'pp', n=1, noROPT_na='不适用：单国（PT）行，n=1',
            grade='C（n=1）', prereg='否', reuse=RU_G_POST,
            metric='OA', source=src, func='numbers_table.h1_decision_upper', doc={'v': {'ae': '17.3', 's2': '15.6'}[feat]},
            doc_src='REPOSITION H1 附')
        add(f'H1附.RO_upper_oa.{feat}', 'H1附', f'RO：决策层上界 OA 增益（划分 0，{feat.upper()}）',
            float(z.loc['RO', 'ora_oa'] - z.loc['RO', 'k0_oa']), 'pp', n=1, noROPT_na='不适用：单国（RO）行，n=1',
            grade='C（n=1）', prereg='否', reuse=RU_G_POST,
            metric='OA', source=src, func='numbers_table.h1_decision_upper', doc={'v': {'ae': '45', 's2': '35'}[feat]},
            doc_src='REPOSITION H1 附（"+45/+35"）')
    # AE − S2：校准后剩余（划分 0）
    w0 = s0.assign(left_oa=s0.tgt300_oa - s0.ora_oa).pivot(index='target', columns='feat', values='left_oa')
    v = w0['ae'] - w0['s2']
    m9, c9 = noropt_str(v, 2)
    add('H1附.left_oa_ae_s2.G', 'H1附', 'AE − S2：决策层校准后的 OA 表征层剩余（缓冲格网，划分 0）', None, 'pp', s=TS(v),
        noROPT=m9, noROPT_ci=c9, grade='C（多对比中挑出的一项）', prereg='否（事后）', reuse=RU_G_POST, metric='OA', source=src,
        func='numbers_table.h1_decision_upper', doc={'v': '-0.91', 'lo': '-1.73', 'hi': '-0.10'}, doc_src='oracle_G_buffered.log',
        per_country=v.to_dict(), note='负 = AE 剩余更小；与 Cheng 同向的唯一信号。')
    # 早停 / 固定 300 棵下的 BA 份额与 AE−S2 剩余（检验 3 划分）
    src2 = f'{VER}/reposition/decision_vs_representation/oracle_es_vs_c300_by_country.csv'
    E = rcsv(src2)
    for cap in ('es', 'c300'):
        for feat in ('ae', 's2'):
            z = E[(E.cap == cap) & (E.feat == feat)]
            z9 = z[~z.target.isin(ROPT)]
            dv = {('es', 'ae'): '39', ('es', 's2'): '32', ('c300', 'ae'): '68', ('c300', 's2'): '62'}[(cap, feat)]
            add(f'H1附.dec_share_c3.{cap}.{feat}', 'H1附', f'决策层上界占 BA 跨境总损失的份额（检验 3 划分，{cap}，{feat.upper()}）',
                100 * z.ba_gain.sum() / z.ba_total.sum(), '%', n=len(z), noROPT=100 * z9.ba_gain.sum() / z9.ba_total.sum(),
                noROPT_ci=f'点估计，n={len(z9)}（v2 补算）', grade='C', prereg='否（事后）', reuse=RU_C3_POST,
                metric='ba_ge30', source=src2, func='numbers_table.h1_decision_upper', doc={'v': dv},
                doc_src='oracle_c300.log；REPOSITION H1 附（"BA 份额 32–78%"）')
        z = E[E.cap == cap].pivot(index='target', columns='feat', values='oa_left')
        v = z['ae'] - z['s2']
        m9, c9 = noropt_str(v, 2)
        add(f'H1附.left_oa_ae_s2.{cap}', 'H1附', f'AE − S2：决策层校准后的 OA 剩余（检验 3 划分，{cap}）', None, 'pp', s=TS(v),
            noROPT=m9, noROPT_ci=c9, grade='C', prereg='否（事后）', reuse=RU_C3_POST, metric='OA', source=src2,
            func='numbers_table.h1_decision_upper', doc={'v': {'es': '-1.21', 'c300': '-1.76'}[cap]}, doc_src='oracle_c300.log',
            per_country=v.to_dict())


# ================================================================== H2：选样指导与选源
def guidance_quality_rows(R):
    """与 analyze_G.guidance_quality 逐行同式，但返回逐（国家, 划分, 指导）行，便于算去 RO/PT。只用重复 0 候选池标签。"""
    import guidance_G as G
    D = G.GData()
    rows = []
    for (t, s), _ in R[R.cond == 'ORACLE'].groupby(['target', 'split']):
        S = dict(np.load(_chk(os.path.join(GRES, 'stage0', f'{t}_s{s}.npz'))))
        pp = G.pool_positions(S, s, 0)
        y = D.y[S['allT'][pp]]
        for name, P in (('AE', S['P_T_ae'][pp]), ('S2', S['P_T_s2'][pp])):
            P = np.asarray(P, float)
            ph = P.argmax(1)
            rows.append(dict(target=t, split=s, guide=name, acc=100 * (ph == y).mean(),
                             nll=-np.log(np.clip(P[np.arange(len(y)), y], 1e-300, None)).mean()))
    return pd.DataFrame(rows)


def h2_guidance(R):
    dE1 = A.deltas(R, 'AE', 'S2', E1)
    vE1 = A.country_values(dE1)
    s = TS(vE1)
    q = stats.t.ppf(0.95, s['n'] - 1)
    p_tost = max(stats.t.sf((s['mean'] + A.SESOI) / s['se'], s['n'] - 1), stats.t.sf((A.SESOI - s['mean']) / s['se'], s['n'] - 1))
    minm = max(abs(s['mean'] - q * s['se']), abs(s['mean'] + q * s['se']))
    m9, c9 = noropt_str(vE1)
    s9 = TS(vE1.drop(ROPT))
    add('H2.theta_E1', 'H2', 'θ_E1：只用几何的通路，AE 指导 − S2 指导（4 方法 × 2 分类器 × k∈{50,200} × 4 单元）', None, 'pp', s=s,
        ci90_use='TOST 等价判定（±0.3 pp，冻结方案第 8 节）', p_note=f'TOST p={p_tost:.4f}', noROPT=m9,
        noROPT_ci=c9 + f'；90% [{s9["lo90"]:+.3f}, {s9["hi90"]:+.3f}] → {A.verdict(s9)}',
        grade='方向 A；等价 B', prereg='是（ff55edf8 主估计量）', reuse=RU_G_PRE, metric='ba_ge30',
        source=SRC_G, func='analyze_G.deltas → country_values → t_summary',
        doc={'v': '-0.054', 'lo': '-0.244', 'hi': '+0.136', 'lo90': '-0.208', 'hi90': '+0.101', 'p_tost': '0.008'},
        doc_src='G_RESULTS §1；report_G.txt；REPOSITION H2', per_country=vE1.to_dict(), note=f'判定：{A.verdict(s)}')
    ROWS[-1]['_tost'] = p_tost
    q9 = stats.t.ppf(0.95, s9['n'] - 1)
    add('H2.theta_E1.tost_min_margin', 'H2', 'θ_E1 能通过 TOST 的最小界值', minm, 'pp', n=11,
        noROPT=max(abs(s9['mean'] - q9 * s9['se']), abs(s9['mean'] + q9 * s9['se'])), noROPT_ci='9 国同式（v2 补算）',
        grade='B', prereg='是（派生）',
        reuse=RU_G_PRE, metric='ba_ge30', source=SRC_G, func='analyze_G.main 同式', doc={'v': '0.208'}, doc_src='report_G.txt')
    lo = A.loo(vE1)
    add('H2.theta_E1.loo', 'H2', '留一国 θ_E1 的均值范围（11 次）', f'{lo["mean"].min():+.3f} 至 {lo["mean"].max():+.3f}', 'pp',
        n=11, noROPT_na='不适用：本行本身是国家子集敏感性；去 RO/PT 的 θ_E1 见 H2.theta_E1 的去 RO/PT 列',
        grade='B（事后）', prereg='否（rev1 B3 事后）', reuse=RU_G_POST, metric='ba_ge30', source=SRC_G, func='analyze_G.loo',
        doc={'v_min': '-0.108', 'v_max': '+0.012'}, doc_src='report_G.txt',
        note=f'判等价 {int(lo.verdict.str.startswith("等价").sum())}/11；最小去 {lo.loc[lo["mean"].idxmin(), "drop"]}，最大去 {lo.loc[lo["mean"].idxmax(), "drop"]}。')
    ROWS[-1]['_loo'] = (lo['mean'].min(), lo['mean'].max())
    # 各指标 θ_E1 与 AE 优势的 90% 上界
    his, his9, byk = {}, {}, {}
    MET7 = ('ba_ge30', 'oa', 'f1_ge30', 'ba_P1', 'ba_P2', 'ba_far', 'oa_P2')
    for met in MET7:
        dm_ = A.deltas(R, 'AE', 'S2', E1, met)
        sm = TS(A.country_values(dm_))
        his[met] = sm['hi90']
        his9[met] = TS(A.country_values(dm_).drop(ROPT))['hi90']
        for k in (50, 200):
            byk[(met, k)] = TS(A.country_values(dm_[dm_.k == k]))['hi90']
        if met in ('oa', 'ba_P2', 'ba_far', 'f1_ge30'):
            v = A.country_values(A.deltas(R, 'AE', 'S2', E1, met))
            m9, c9 = noropt_str(v)
            dmap = {'oa': {'v': '-0.149', 'lo': '-0.497', 'hi': '+0.198', 'lo90': '-0.432', 'hi90': '+0.133'},
                    'ba_P2': {'v': '-0.051', 'lo90': '-0.168', 'hi90': '+0.066'},
                    'ba_far': {'v': '-0.038', 'lo90': '-0.209', 'hi90': '+0.134'},
                    'f1_ge30': {'v': '+0.044', 'lo90': '-0.176', 'hi90': '+0.264'}}
            if met == 'oa':
                pre, ru = '次要（OA 为 H2 门控，未进行）', RU_G_PRE
            elif met in ('ba_P2', 'ba_far'):
                pre, ru = '是（冻结方案 §8 次要，不校正）；逐指标 TOST 判定为事后', RU_G_PRE
            else:
                pre, ru = '否（rev1 事后；f1_ge30 不在冻结方案 §7 指标中）', RU_G_POST
            add(f'H2.theta_E1.{met}', 'H2', f'θ_E1 的 {met} 版', None, 'pp', s=sm, ci90_use='TOST 判定（非预注册，事后逐指标）',
                noROPT=m9, noROPT_ci=c9, grade='B（OA 为"不确定"）' if met == 'oa' else 'B',
                prereg=pre, reuse=ru,
                metric=met, source=SRC_G, func='analyze_G.deltas(metric) → country_values → t_summary', doc=dmap[met],
                doc_src='report_G.txt；REPOSITION H2', per_country=v.to_dict(), note=f'判定：{A.verdict(sm)}')
    mx = max(his, key=his.get)
    mx9 = max(his9, key=his9.get)
    add('H2.ae_adv_hi90_max', 'H2', 'AE 优势的 90% 上界在 7 个已报指标上的最大值（k=50 与 k=200 两档合并）', his[mx], 'pp', n=11,
        noROPT=his9[mx9], noROPT_ci=f'9 国，最大者为 {mx9}（v2 补算）', grade='B',
        prereg='否（派生）', reuse=RU_G_POST, metric=f'max over {list(his)}', source=SRC_G, func='numbers_table.h2_guidance',
        doc={'v': '0.27'}, doc_src='REPOSITION H2（"≤+0.27"）',
        per_country={k: v for k, v in his.items()},
        note=f'两档 k 合并；最大者为 {mx}；per_country 列此处为各指标的 hi90。按 k 拆开见 H2.ae_adv_hi90_max.byk，'
             '"≤+0.27" 只在 k 合并时成立。')
    ROWS[-1]['_his'] = his
    bmx = {k: max(MET7, key=lambda m: byk[(m, k)]) for k in (50, 200)}
    byk9 = {}
    for met in MET7:
        dm_ = A.deltas(R, 'AE', 'S2', E1, met)
        for k in (50, 200):
            byk9[(met, k)] = TS(A.country_values(dm_[dm_.k == k]).drop(ROPT))['hi90']
    bmx9 = {k: max(MET7, key=lambda m: byk9[(m, k)]) for k in (50, 200)}
    add('H2.ae_adv_hi90_max.byk', 'H2', 'AE 优势的 90% 上界按 k 拆开：7 个已报指标中的最大值（k=50；k=200）',
        f'k=50：{byk[(bmx[50], 50)]:+.3f}（{bmx[50]}）；k=200：{byk[(bmx[200], 200)]:+.3f}（{bmx[200]}）', 'pp', n=11,
        noROPT=f'k=50：{byk9[(bmx9[50], 50)]:+.3f}（{bmx9[50]}）；k=200：{byk9[(bmx9[200], 200)]:+.3f}（{bmx9[200]}）',
        noROPT_ci='9 国（v2 补算）', grade='B（描述）',
        prereg='否（派生；按 k 做 TOST 非预注册）', reuse=RU_G_POST, metric='7 个已报指标', source=SRC_G,
        func='analyze_G.deltas(metric) 按 k 切分 → country_values → t_summary', mode='重算',
        doc={'v_f1_k50': '+0.36', 'v_ba_k50': '+0.150', 'v_ba_k200': '+0.083'}, doc_src='d3_review/r6_metrics.py（复核重算值）',
        per_country={f'{m}|k{k}': byk[(m, k)] for m in MET7 for k in (50, 200)},
        note='per_country 列此处为各（指标|k）的 hi90。只有 ba_ge30 在两档 k 下都 ≤+0.27；'
             f'f1_ge30 在 k=50 为 {byk[("f1_ge30", 50)]:+.3f}。')
    ROWS[-1]['_byk'] = byk
    for k in (50, 200):
        vk = A.country_values(dE1[dE1.k == k])
        sm = TS(vk)
        m9, c9 = noropt_str(vk)
        add(f'H2.theta_E1.k{k}', 'H2', f'θ_E1 只取 k={k}', None, 'pp', s=sm, ci90_use='按 k 的 TOST（非预注册）',
            noROPT=m9, noROPT_ci=c9,
            grade='B（次要）', prereg='是（冻结方案 §8 次要，按 k 的 θ）；按 k 做 TOST 非预注册', reuse=RU_G_PRE, metric='ba_ge30',
            source=SRC_G, func='analyze_G.country_values(dE1[k])',
            doc={50: {'v': '+0.013', 'lo90': '-0.125', 'hi90': '+0.150'},
                 200: {'v': '-0.120', 'lo90': '-0.323', 'hi90': '+0.083'}}[k], doc_src='report_G.txt',
            note=f'判定：{A.verdict(sm)}')
    for lrn in ('LR', 'MLP', 'RF'):
        add(f'H2.theta_E1.{lrn}', 'H2', f'θ_E1（{lrn}）', '待 rev2', 'pp', grade='待定', prereg='是（rev2 2589217a）',
            reuse=RU_G_POST, learner=lrn, metric='ba_ge30', source='results/guidance_G_ml/（未读取）', func='analyze_ML.py（待正式运行）',
            mode='占位', noROPT_na='待 rev2')
    add('H2.theta_E1.learner_verdict', 'H2', '学习器稳健性判定（rev2 §4："等价不依赖学习器"等三档）', '待 rev2', '', grade='待定',
        prereg='是（rev2）', reuse=RU_G_POST, learner='LightGBM+LR+MLP+RF', source='analyze_ML.py', func='analyze_ML.py', mode='占位',
        noROPT_na='待 rev2')
    # 预测通路（C）与族错误率
    sec = {}
    g = dE1.pivot_table(index=A.UNIT + ['method', 'k'], columns='clf', values='d').dropna()
    g['gamma'] = (g['ae'] - g['s2']) / 2
    sec['γ'] = g.reset_index().groupby(['target', 'method', 'k']).gamma.mean().groupby('target').mean()
    for clf in ('s2', 'ae'):
        sec[f'E1|clf={clf}'] = A.country_values(dE1[dE1.clf == clf])
    for k in (50, 200):
        sec[f'E1|k={k}'] = A.country_values(dE1[dE1.k == k])
    for name in ('pred', 'hyb'):
        sec[f'通路 {name}'] = A.country_values(A.deltas(R, 'AE', 'S2', FAMILIES[name], reps=[0]))
    for lab, met in (('Δ(P2)', 'ba_P2'), ('Δ(P1)', 'ba_P1'), ('E_far', 'ba_far')):
        sec[lab] = A.country_values(A.deltas(R, 'AE', 'S2', E1, met))
    for a_, b_ in (('AE', 'S2XY'), ('AE', 'XY'), ('S2XY', 'S2')):
        sec[f'{a_}−{b_}'] = A.country_values(A.deltas(R, a_, b_, ['lpm2_nogeo', 'kcenter'], reps=[0]))
    sec['H_ORACLE'] = A.country_values(A.deltas(R, 'ORACLE', 'S2', ['lpm2_nogeo', 'typiclust', 'margin'], reps=[0]))
    V = pd.concat(sec, axis=1).dropna()
    praw = pd.Series({c: TS(V[c])['p'] for c in V.columns})
    hol = A.holm(praw)
    vp = sec['通路 pred']
    m9, c9 = noropt_str(vp)
    add('H2.pred_pathway', 'H2', '只用预测的通路（margin、predstrat_srs）AE − S2（重复 0 的 22 单元）', None, 'pp', s=TS(vp),
        p_note=f'14 项族 Holm p={hol["通路 pred"]:.3f}', noROPT=m9, noROPT_ci=c9, grade='C', prereg='是（次要，不校正）',
        reuse=RU_G_PRE, metric='ba_ge30', source=SRC_G, func='analyze_G.deltas(FAMILIES[pred], reps=[0]) + analyze_G.holm',
        doc={'v': '+0.340', 'lo': '+0.037', 'hi': '+0.642', 'holm': '0.438'}, doc_src='report_G.txt；G_RESULTS §3',
        per_country=vp.to_dict())
    ROWS[-1]['_holm'] = hol['通路 pred']
    # ORACLE2 typiclust：同一测定能检出约 1 pp 的阳性例子
    v = A.country_values(A.deltas(R, 'ORACLE2', 'S2', ['typiclust'], reps=[0]))
    m9, c9 = noropt_str(v)
    add('H2.oracle2_typiclust', 'H2', '阳性例子：修正几何神谕 ORACLE2 − S2（typiclust，重复 0）', None, 'pp', s=TS(v), noROPT=m9,
        noROPT_ci=c9, grade='C（事后、单方法、未校正）', prereg='否（rev1 B1 事后）', reuse=RU_G_POST, metric='ba_ge30', source=SRC_G,
        func="analyze_G.deltas(R,'ORACLE2','S2',['typiclust'],reps=[0])", doc={'v': '+0.929', 'lo': '+0.343', 'hi': '+1.515'},
        doc_src='report_G.txt', per_country=v.to_dict())
    # 标签当量
    La = A.label_equiv_detail(R, {'+0.3': A.SESOI, 'θ_E1': s['mean']}, A.label_equiv_anchored)
    cs = A.censored_summary(La)
    cs9 = A.censored_summary(La[~La.target.isin(ROPT)])
    mc = A.mean_curve_equiv(R, {'+0.3': A.SESOI})
    for k in (50, 200):
        c = cs.loc[('+0.3', k)]
        c9 = cs9.loc[('+0.3', k)]
        add(f'H2.labelequiv.k{k}', 'H2', f'+0.3 pp 在 k={k} 处约合多少个随机标签（锚定定义，22 条曲线中位数）', float(c['median']),
            '个标签', n=22, noROPT=float(c9['median']), noROPT_ci=f'18 条曲线中位数；+∞ 删失 {int(c9["n_pinf"])}/18（v2 补算）',
            grade='C', prereg='否（rev1 B4 锚定定义事后）', reuse=RU_G_POST, metric='ba_ge30',
            source=SRC_G, func='analyze_G.label_equiv_detail(label_equiv_anchored) + censored_summary',
            doc={50: {'v': '89.6'}, 200: {'v': 'inf'}}[k], doc_src='report_G.txt；G_RESULTS §5',
            note=f'+∞ 删失 {int(c["n_pinf"])}/22；有限值 IQR [{c["q25_finite"]:.1f}, {c["q75_finite"]:.1f}]；'
                 f'跨国平均曲线：S2 分类器 {mc.loc[("s2", "+0.3"), k]:.1f}，AE 分类器 {mc.loc[("ae", "+0.3"), k]:.1f}。')
        ROWS[-1]['_mc'] = (mc.loc[('s2', '+0.3'), k], mc.loc[('ae', '+0.3'), k])
    # 指导质量（可分性与外国模型质量）
    srcq = f'{GRES}/guidance_quality_G.csv'
    Q = rcsv(srcq)
    kn = Q.pivot_table(index='target', columns='feat', values='knn10_ba')
    gap = kn['ae'] - kn['s2']
    add('H2.knn10', 'H2', '候选池 10 近邻平衡精度：AE 几何 vs S2 几何（可分性前提）',
        f'{Q[Q.feat == "ae"].knn10_ba.mean():.1f} vs {Q[Q.feat == "s2"].knn10_ba.mean():.1f}', '%', n=11, n_pos=int((gap > 0).sum()),
        noROPT=float(gap.drop(ROPT).mean()), grade='B', prereg='否（描述）', reuse=RU_NOTEST, learner='—（几何）',
        metric='10-NN BA', source=srcq, func='numbers_table.h2_guidance（均值与国家差）',
        doc={'v_ae': '46.6', 'v_s2': '38.8'}, doc_src='G_RESULTS §2；REPOSITION H2', per_country=gap.to_dict(),
        note=f'AE − S2 国家差 {gap.min():+.1f} 至 {gap.max():+.1f}，均值 {gap.mean():+.2f}；noROPT 列为 9 国差均值。'
             '只用重复 0 候选池标签（mediators_G.unit_quality）。G 冻结前已对每国约 3 万个目标地块看过类似的 kNN 几何纯度诊断（TESTSET_AUDIT §5）。')
    ROWS[-1]['_knn'] = (Q[Q.feat == 'ae'].knn10_ba.mean(), Q[Q.feat == 's2'].knn10_ba.mean())
    au = Q.groupby('feat').auroc_margin.mean()
    au9 = Q[~Q.target.isin(ROPT)].groupby('feat').auroc_margin.mean()
    add('H2.auroc_margin', 'H2', '外国模型 margin 识别误判的 AUROC：AE vs S2', f'{au["ae"]:.4f} vs {au["s2"]:.4f}', 'AUROC', n=11,
        noROPT=f'{au9["ae"]:.4f} vs {au9["s2"]:.4f}', noROPT_ci='9 国均值（v2 补算）',
        grade='B', prereg='是（冻结方案 §7 中介变量，描述）', reuse=RU_NOTEST, metric='AUROC', source=srcq,
        func='numbers_table.h2_guidance', doc={'v_ae': '0.743', 'v_s2': '0.743'}, doc_src='G_RESULTS §2',
        note='只用重复 0 候选池标签（mediators_G.unit_quality）。')
    ROWS[-1]['_auroc'] = (au['ae'], au['s2'])
    gq = A.guidance_quality(R)
    gq9 = guidance_quality_rows(R)
    assert np.allclose(gq9.groupby('guide')[['acc', 'nll']].mean().loc[['AE', 'S2']].to_numpy(),
                       gq.loc[['AE', 'S2'], ['acc', 'nll']].to_numpy())
    gq9 = gq9[~gq9.target.isin(ROPT)].groupby('guide')[['acc', 'nll']].mean()
    add('H2.guide_acc', 'H2', '外国模型在候选池（重复 0）的准确率：AE vs S2（NLL 附注）',
        f'{gq.loc["AE", "acc"]:.1f} vs {gq.loc["S2", "acc"]:.1f}', '%', n=11,
        noROPT=f'{gq9.loc["AE", "acc"]:.1f} vs {gq9.loc["S2", "acc"]:.1f}', noROPT_ci='9 国均值（v2 补算）', grade='B',
        prereg='否（事后描述量；NLL 属冻结方案 §7 "外国模型质量"中介变量，准确率不在其中）', reuse=RU_NOTEST,
        metric='acc / NLL', source=f'{GRES}/stage0/*.npz + {GRES}/oracle/*.npz',
        func='analyze_G.guidance_quality（v2：另按国家展开以算去 RO/PT）',
        doc={'v_ae': '71.1', 'v_s2': '70.0'}, doc_src='report_G.txt',
        note=f'NLL {gq.loc["AE", "nll"]:.3f} vs {gq.loc["S2", "nll"]:.3f}；ORACLE 准确率 {gq.loc["ORACLE", "acc"]:.1f}。只用重复 0 候选池标签。')
    ROWS[-1]['_acc'] = (gq.loc['AE', 'acc'], gq.loc['S2', 'acc'])


def h2_check1():
    src = f'{RES}/check1_per_target.csv'
    per = rcsv(src)
    DESC = ['mmd_ae', 'mmd_s2', 'c2st_ae', 'c2st_s2', 'geo_km', 'climate_diff', 'class_js']
    rng = np.random.default_rng(42)

    def boot_ci(v, B=5000):
        v = np.asarray(v)
        v = v[~np.isnan(v)]
        m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
        return np.percentile(m, [2.5, 97.5])
    avg_t = per.groupby(['descriptor', 'target'])['rho'].mean().unstack()
    cis = {d: boot_ci(avg_t.loc[d].to_numpy()) for d in DESC}
    dcis = {}
    for feat in ('ae', 's2', 'fusion', 'avg'):
        if feat == 'avg':
            a, b = avg_t.loc['mmd_ae'], avg_t.loc['mmd_s2']
        else:
            p = per[per['feature'] == feat].pivot(index='target', columns='descriptor', values='rho')
            a, b = p['mmd_ae'], p['mmd_s2']
        diff = (a - b).dropna()
        dcis[feat] = (diff, boot_ci(diff.to_numpy()))
    fn = 'numbers_table.h2_check1（复现 analyze_check1 的 bootstrap 随机数顺序）'
    dv = {'geo_km': ('0.588', '0.37', '0.76'), 'mmd_ae': ('0.532', '0.27', '0.71'), 'mmd_s2': ('0.511', '0.23', '0.71')}
    for d in ('geo_km', 'mmd_ae', 'mmd_s2'):
        v = avg_t.loc[d]
        add(f'H2.check1.rho.{d}', 'H2', f'检验 1：以 −{d} 给候选源国排序与迁移 BA 的平均 Spearman ρ（三种分类特征平均）', float(v.mean()),
            'ρ', lo95=cis[d][0], hi95=cis[d][1], n=int(v.notna().sum()), noROPT=float(v.drop(ROPT).mean()), grade='B',
            prereg='是（ANALYSIS_PLAN_check1）', reuse=RU_C1, learner='LightGBM（pairwise_v2）', metric='Spearman ρ',
            source=src, func=fn, doc={'v': dv[d][0], 'lo': dv[d][1], 'hi': dv[d][2]}, doc_src='CHECK1_RESULTS；check1_report.txt',
            per_country=v.to_dict(), note='95% CI = 按目标国 bootstrap（B=5000）。')
    diff, ci = dcis['avg']
    w = stats.wilcoxon(diff.to_numpy()).pvalue
    d9 = diff.drop(ROPT)
    add('H2.check1.drho_aemmd_s2mmd', 'H2', '检验 1 主比较：Δρ = ρ(AE-MMD) − ρ(S2-MMD)', float(diff.mean()), 'Δρ', lo95=ci[0], hi95=ci[1],
        n=len(diff), n_pos=int((diff > 0).sum()), p=w, p_note='Wilcoxon（预注册）', noROPT=float(d9.mean()),
        noROPT_ci=f'{int((d9 > 0).sum())}/9，Wilcoxon p={stats.wilcoxon(d9).pvalue:.3f}', grade='B',
        prereg='是（主比较）', reuse=RU_C1, learner='LightGBM（pairwise_v2）', metric='Spearman ρ', source=src, func=fn,
        doc={'v': '+0.021', 'lo': '-0.018', 'hi': '+0.062', 'p': '0.43'}, doc_src='CHECK1_RESULTS；REPOSITION H2',
        per_country=diff.to_dict(), note=f'去 RO（预注册敏感性）：{diff.drop("RO").mean():+.3f}，p={stats.wilcoxon(diff.drop("RO")).pvalue:.3f}。')
    ROWS[-1]['_noRO'] = diff.drop('RO').mean()
    dg = (avg_t.loc['mmd_ae'] - avg_t.loc['geo_km']).dropna()
    add('H2.check1.drho_aemmd_geo', 'H2', '检验 1：Δρ = ρ(AE-MMD) − ρ(地理距离)', float(dg.mean()), 'Δρ', n=len(dg),
        n_pos=int((dg > 0).sum()), p=stats.wilcoxon(dg).pvalue, p_note='Wilcoxon', noROPT=float(dg.drop(ROPT).mean()), grade='B',
        prereg='是', reuse=RU_C1, learner='LightGBM（pairwise_v2）', metric='Spearman ρ', source=src, func=fn,
        doc={'v': '-0.055', 'p': '0.250'}, doc_src='CHECK1_RESULTS', per_country=dg.to_dict(), note='原报告无 CI。')


# ================================================================== H3：随机标签价值
def srs_frame(R):
    S = R[R.method.isin(['srs', 'foreign_only']) & R.clf.isin(['ae', 's2'])]
    S = S[S.k.isin([0, 50, 100, 200, 400])]
    return S


def h3_label_value(R):
    S = srs_frame(R)
    kg = S.groupby(['target', 'clf', 'k']).ba_ge30.mean().unstack('k')
    g50 = kg[50] - kg[0]
    nonmono = int((kg[[0, 50, 100, 200, 400]].diff(axis=1).iloc[:, 1:] < 0).any(axis=1).sum())
    keep = ~kg.index.get_level_values('target').isin(ROPT)
    add('H3.srs_k0_50_median', 'H3', 'SRS 学习曲线 k=0→50 增益的中位数（22 条曲线 = 11 国 × 2 分类器）', float(g50.median()), 'pp', n=22,
        noROPT=float(g50[keep].median()), grade='B', prereg='否（描述）', reuse=RU_G_POST, metric='ba_ge30', source=SRC_G,
        func='analyze_G.main 同式（ks_gain）', doc={'v': '+0.31'}, doc_src='report_G.txt；REPOSITION H3',
        note=f'不单调曲线 {nonmono}/22；noROPT 为 18 条曲线中位数。')
    ROWS[-1]['_nonmono'] = nonmono
    g400 = kg[400] - kg[0]
    add('H3.srs_k0_400_median', 'H3', 'SRS k=0→400 增益的中位数（22 条曲线）', float(g400.median()), 'pp', n=22,
        noROPT=float(g400[keep].median()), grade='B', prereg='否（描述）', reuse=RU_G_POST, metric='ba_ge30', source=SRC_G,
        func='numbers_table.h3_label_value', doc={'v': '+1.41', 'mean': '3.33'}, doc_src='MODEL_DIAGNOSIS M4',
        note=f'22 条曲线均值 {g400.mean():+.2f}。')
    ROWS[-1]['_mean400'] = g400.mean()
    # rev3 B2：G(L=LightGBM, k)
    U = S.groupby(A.UNIT + ['clf', 'k']).ba_ge30.mean().unstack('k')
    Uo = S.groupby(A.UNIT + ['clf', 'k']).oa.mean().unstack('k')
    dm = {(200, 's2'): ('+2.41', '+0.66'), (200, 'ae'): ('+3.10', '+0.71'), (400, 'ae'): ('+3.36', None), (400, 's2'): ('+3.31', None)}
    b2 = {}
    for k in (50, 100, 200, 400):
        for clf in ('s2', 'ae', 'avg'):
            for met, UU in (('ba_ge30', U), ('oa', Uo)):
                u = (UU[k] - UU[0]).rename('d').reset_index()
                if clf != 'avg':
                    u = u[u.clf == clf]
                v = u.groupby(['target', 'clf']).d.mean().groupby('target').mean()
                s = TS(v)
                s9 = TS(v.drop(ROPT))
                b2[(k, clf, met)] = (s, s9)
                if met == 'oa' and k not in (200, 400):
                    continue
                d = dm.get((k, clf)) if met == 'ba_ge30' else None
                add(f'H3.G_lgb.k{k}.{clf}.{met}', 'H3', f'随机标签价值 G(LightGBM, k={k})：SRS(k) − k=0，{"两种特征平均" if clf == "avg" else clf.upper() + " 特征"}（rev3 B2）',
                    None, 'pp', s=s, noROPT=s9['mean'], noROPT_ci=f'[{s9["lo95"]:+.2f}, {s9["hi95"]:+.2f}]，{s9["n_pos"]}/9',
                    grade='B（待 rev3 B2 四学习器判据；国家分布 C）', prereg='rev3 B2 预先指定（LightGBM 部分此前已看过）', reuse=RU_G_POST, metric=met,
                    source=SRC_G, func='numbers_table.h3_label_value（单元配对 → 国家 → t）',
                    doc={'v': d[0], 'noROPT': d[1]} if d and d[1] else ({'v': d[0]} if d else None),
                    doc_src='REPOSITION H3/H6；CONCLUSIONS V8' if d else '', per_country=v.to_dict())
    s9 = b2[(200, 'avg', 'ba_ge30')][1]
    ok = s9['hi95'] < 2
    add('H3.B2_criterion.lgb', 'H3', 'rev3 B2 判据（LightGBM 部分）：去 RO/PT 后 G(L,200)（两种特征平均）95% CI 上界 < 2 pp',
        s9['hi95'], 'pp', n=9, noROPT_na='不适用：判据本身即去 RO/PT 的 9 国版', grade='B（需四个学习器都满足）', prereg='是（rev3 B2）', reuse=RU_G_POST, metric='ba_ge30',
        source=SRC_G, func='numbers_table.h3_label_value', note=f'LightGBM {"满足" if ok else "不满足"}（上界 {s9["hi95"]:+.2f}）；四学习器合判待 rev2。')
    for lrn in ('LR', 'MLP', 'RF'):
        add(f'H3.G.{lrn}', 'H3', f'随机标签价值 G({lrn}, k∈{{50,200,400}}) 与 B2 判据', '待 rev2', 'pp', grade='待定', prereg='是（rev3 B2）',
            reuse=RU_G_POST, learner=lrn, source='results/guidance_G_ml/（未读取）', func='analyze_ML_rev3.py', mode='占位',
            noROPT_na='待 rev2', note='LR 的 SRS/k=0 行已非盲（rev3 A1），论文中标"非盲"。')
    # 标签价值与距离、构成
    src = f'{VER}/reposition/label_budget_line/labelvalue_vs_distance.csv'
    Dd = rcsv(src)
    sub = Dd[Dd.dist.isin(['mean_dlat_to_pool', 'mean_km_to_pool'])]
    rr = pd.concat([sub.rho, sub.rho_noRO])
    pp = pd.concat([sub.p, sub.p_noRO])
    add('H3.labelvalue_vs_distance', 'H3', '标签价值（SRS k − k=0）与到外国池距离的 Spearman ρ 范围（平均纬度差、平均公里；11 国与去 RO）',
        f'{rr.min():+.3f} 至 {rr.max():+.3f}', 'ρ', n=11, p=float(pp.min()), p_note='所列 p 的最小值（朴素）',
        noROPT_na='未算：源文件只给 11 国与去 RO 两版（去 RO 已并入范围）；去 RO/PT 需改写 labelvalue_vs_distance.py（方案外）',
        grade='C（n=11，事后）',
        prereg='否（事后）', reuse=RU_G_POST, metric='ba_ge30 与 OA；k=50/200/400；2 分类器', source=src,
        func='读取 labelvalue_vs_distance.py 的输出', mode='读取（该脚本从 G 行重算）',
        doc={'v_min': '-0.31', 'v_max': '+0.33', 'p_min': '0.33'}, doc_src='REPOSITION H3')
    ROWS[-1]['_rng'] = (rr.min(), rr.max(), pp.min())
    sm = Dd[Dd.dist == 'min_dlat_to_pool']
    add('H3.labelvalue_vs_min_dlat', 'H3', '同上，但距离取"到外国池的最小纬度差"（原日志未打印的一组）',
        f'{sm.rho.min():+.3f} 至 {sm.rho.max():+.3f}（去 RO {sm.rho_noRO.min():+.3f} 至 {sm.rho_noRO.max():+.3f}）', 'ρ', n=11,
        p=float(sm.p.min()), p_note=f'朴素 p 范围 {sm.p.min():.3f}–{sm.p.max():.3f}',
        noROPT_na='未算：源文件只给 11 国与去 RO 两版（去 RO 已列在数值中）', grade='C（n=11，事后，分叉路径）', prereg='否',
        reuse=RU_G_POST, metric='ba_ge30 与 OA', source=src, func='读取', mode='读取',
        note='"标签价值与距离无关"依赖距离定义：最小纬度差给出 ρ≈0.5–0.75 的正相关。见 discrepancies.txt。')
    # 构成 L1（目标测试集 vs 外国池）
    y = np.load(_chk(f'{LOCAL}/data/cache_guidance_G/y.npy'))
    L1 = {}
    for f in sorted(glob.glob(_chk(f'{GRES}/stage0/*_s*.npz'))):
        t = os.path.basename(f).split('_')[0]
        z = np.load(f, allow_pickle=True)
        pF = np.bincount(y[z['foreign']], minlength=10) / len(z['foreign'])
        pT = np.bincount(y[z['test']], minlength=10) / len(z['test'])
        L1.setdefault(t, []).append(np.abs(pT - pF).sum())
    L1 = pd.Series({t: np.mean(v) for t, v in L1.items()})
    for t, dv in (('RO', '1.41'), ('PT', '0.86')):
        add(f'H3.compL1.{t}', 'H3', f'{t}：目标测试集与外国池的类别构成 L1（两划分平均）', float(L1[t]), 'L1', n=1,
            noROPT_na=f'不适用：单国（{t}）行', grade='C（n=2 解释）',
            prereg='否', reuse=RU_G_POST, learner='—', metric='L1', source=f'{GRES}/stage0/*.npz + data/cache_guidance_G/y.npy',
            func='numbers_table.h3_label_value', doc={'v': dv}, doc_src='REPOSITION H3')
    add('H3.compL1.other9', 'H3', '其余 9 国的构成 L1 范围', f'{L1.drop(ROPT).min():.3f} 至 {L1.drop(ROPT).max():.3f}', 'L1', n=9,
        noROPT_na='不适用：本行即去 RO/PT 的 9 国',
        grade='C', prereg='否', reuse=RU_G_POST, learner='—', metric='L1', source=f'{GRES}/stage0/*.npz',
        func='numbers_table.h3_label_value', per_country=L1.to_dict())
    return b2


# ================================================================== H4：标哪些地块
def h4_strategy(R):
    for cond, dv in (('AE', {'v': '-0.156', 'lo': '-0.728', 'hi': '+0.416'}), ('S2', {'v': '-0.102', 'lo': '-0.647', 'hi': '+0.442'})):
        d = A.vs_srs(R, cond, E1)
        v = A.country_values(d)
        m9, c9 = noropt_str(v)
        add(f'H4.geom_vs_srs.{cond}', 'H4', f'4 个几何覆盖法相对 SRS 的平均差（{cond} 指导，44 单元）', None, 'pp', s=TS(v), noROPT=m9,
            noROPT_ci=c9, grade='B−', prereg='否（rev1 B3 事后；冻结方案 §8 R3 预注册的是胜率与 Δ>0 的国家数）', reuse=RU_G_POST,
            metric='ba_ge30', source=SRC_G,
            func='analyze_G.vs_srs → country_values', doc=dv, doc_src='report_G.txt R3；STRATEGY ①；REPOSITION H4',
            per_country=v.to_dict(), note=f'运行层面 P(<−0.5) = {(d.d < -0.5).mean():.3f}。')
    # 4 方法间 F（复现 skeptic/s1_checks.method_F，重复 0）
    def all_vs(metric, rep0):
        out = []
        for cond in ('AE', 'S2'):
            RR = R[R.rep == 0] if rep0 else R
            d = A.vs_srs(RR, cond, ALL8, metric)
            out.append(d.assign(cond=cond))
        return pd.concat(out)

    def method_F(CM):
        Y = CM.values
        n, m = Y.shape
        gm, rm, cmn = Y.mean(), Y.mean(1, keepdims=True), Y.mean(0, keepdims=True)
        ssm = n * ((cmn - gm) ** 2).sum()
        sse = ((Y - rm - cmn + gm) ** 2).sum()
        F = (ssm / (m - 1)) / (sse / ((m - 1) * (n - 1)))
        return F, stats.f.sf(F, m - 1, (m - 1) * (n - 1)), m - 1, (m - 1) * (n - 1)
    for rep0, lab in ((True, '重复 0（原分析）'), (False, '全部 44 单元（本表新算）')):
        d = all_vs('ba_ge30', rep0)
        CM = d.groupby(['target', 'method', 'cond', 'clf', 'k']).d.mean().groupby(['target', 'method']).mean().unstack('method')[E1]
        F, p, a, b = method_F(CM)
        F9, p9, _, _ = method_F(CM.drop('RO'))
        F92, p92, a92, b92 = method_F(CM.drop(ROPT))
        add(f'H4.geom_F.{"rep0" if rep0 else "all"}', 'H4', f'4 个几何方法之间的差异：两因素（国家 × 方法）F 检验，{lab}', F, 'F',
            p=p, p_note=f'F({a},{b})；去 RO：F={F9:.2f}，p={p9:.3f}', n=11, noROPT=F92, noROPT_ci=f'F({a92},{b92})，p={p92:.3f}（v2 补算）',
            grade='B−', prereg='否（事后）', reuse=RU_G_POST,
            metric='ba_ge30', source=SRC_G, func='numbers_table.h4_strategy（复现 strategy/skeptic/s1_checks.method_F）',
            doc={'v': '1.01', 'p': '0.402'} if rep0 else None, doc_src='STRATEGY ①；skeptic/s1_checks_clean.log' if rep0 else '')
    # predstrat 与 Holm 族
    fam = {}
    cvals = {}
    for cond in ('AE', 'S2'):
        for m in ALL8:
            v = A.country_values(A.vs_srs(R, cond, [m]))
            cvals[(cond, m)] = v
            fam[f'{m}|{cond}'] = TS(v)['p']
    hol = A.holm(pd.Series(fam))
    for cond, dv in (('AE', {'v': '+1.23', 'lo': '+0.33', 'hi': '+2.14', 'holm16': '0.198'}),
                     ('S2', {'v': '+0.84', 'lo': '-0.02', 'hi': '+1.69'})):
        v = cvals[(cond, 'predstrat_srs')]
        m9, c9 = noropt_str(v)
        add(f'H4.predstrat.{cond}', 'H4', f'predstrat_srs − SRS（{cond} 指导，重复 0 的 22 单元）', None, 'pp', s=TS(v),
            p_note=f'16 项族（8 方法 × 2 指导）Holm p={hol[f"predstrat_srs|{cond}"]:.3f}', noROPT=m9, noROPT_ci=c9, grade='C',
            prereg='否（事后）', reuse=RU_G_POST, metric='ba_ge30', source=SRC_G, func='analyze_G.vs_srs + analyze_G.holm',
            doc=dv, doc_src='STRATEGY ②；strategy/benchmark/a_vs_srs.log', per_country=v.to_dict())
        ROWS[-1]['_holm'] = hol[f'predstrat_srs|{cond}']
    dd = pd.concat([A.vs_srs(R, c, ['predstrat_srs']).assign(cond=c) for c in ('AE', 'S2')])
    v = dd.groupby(['target', 'method', 'clf', 'k', 'cond']).d.mean().groupby('target').mean()
    m9, c9 = noropt_str(v)
    add('H4.predstrat.pooled', 'H4', 'predstrat_srs − SRS（AE、S2 指导合并）', None, 'pp', s=TS(v), noROPT=m9, noROPT_ci=c9,
        p_note=f'Holm（16 项族）最小 p={min(hol["predstrat_srs|AE"], hol["predstrat_srs|S2"]):.3f}', grade='C', prereg='否（事后）',
        reuse=RU_G_POST, metric='ba_ge30', source=SRC_G, func='analyze_G.vs_srs（合并 cond）',
        doc={'v': '+1.04', 'lo': '+0.17', 'hi': '+1.90', 'holm_REPOSITION': '0.44'},
        doc_src='REPOSITION H4（写 "Holm 校正后 p=0.44"）；STRATEGY ②（写 "最小 p=0.198"）', per_country=v.to_dict(),
        note='REPOSITION H4 的 "Holm p=0.44" 实为预测通路 AE−S2（+0.34）的 Holm p，不是 predstrat 相对 SRS；见 discrepancies.txt。')
    ROWS[-1]['_holm'] = min(hol['predstrat_srs|AE'], hol['predstrat_srs|S2'])
    # 收缩估计（冻结方案规定逐方法只报收缩值）
    M = A.shrink_methods(A.vs_srs(R, 'AE', ALL8))
    r_ = M.set_index('method').loc['predstrat_srs']
    M9 = A.shrink_methods(A.vs_srs(R[~R.target.isin(ROPT)], 'AE', ALL8))
    r9 = M9.set_index('method').loc['predstrat_srs']
    add('H4.predstrat.shrunk_AE', 'H4', 'predstrat_srs − SRS 的收缩估计（AE 指导，8 方法部分合并）', float(r_['shrunk']), 'pp',
        n=11, noROPT=float(r9['shrunk']), noROPT_ci=f'9 国重做部分合并；SE {r9["shrunk_se"]:.2f}，τ̂={r9["tau_methods"]:.2f}（v2 补算）',
        grade='C', prereg='收缩规则来自冻结方案（应用对象事后）', reuse=RU_G_POST, metric='ba_ge30', source=SRC_G,
        func='analyze_G.shrink_methods(vs_srs(R,"AE",8 方法))', doc={'v': '+0.41'}, doc_src='STRATEGY ②；a_shrink.csv',
        note=f'SE {r_["shrunk_se"]:.2f}；方法间 τ̂={r_["tau_methods"]:.2f}。')
    # kcenter OA
    dd = pd.concat([A.vs_srs(R, c, ['kcenter'], 'oa').assign(cond=c) for c in ('AE', 'S2')])
    v = dd.groupby(['target', 'method', 'clf', 'k', 'cond']).d.mean().groupby('target').mean()
    m9, c9 = noropt_str(v)
    add('H4.kcenter_oa', 'H4', 'kcenter − SRS 的 OA（AE、S2 指导合并，44 单元）', None, 'pp', s=TS(v), noROPT=m9, noROPT_ci=c9,
        grade='B−/C', prereg='否（事后）', reuse=RU_G_POST, metric='OA', source=SRC_G, func='analyze_G.vs_srs(metric="oa")',
        doc={'v': '-1.76', 'lo': '-4.31', 'hi': '+0.79', 'RO': '-12.55', 'PT': '-4.32'}, doc_src='STRATEGY ④；REPOSITION H4',
        per_country=v.to_dict(), note=f'RO {v["RO"]:+.2f}，PT {v["PT"]:+.2f}；中位数 {v.median():+.2f}。')
    ROWS[-1]['_ROPT'] = (v['RO'], v['PT'])
    # 剂量–反应（strategy/mechanism/L.pkl）
    src = f'{VER}/strategy/mechanism/L.pkl'
    Lk = rpkl(src)

    def slope(df):
        return float((df.w * df.dlog_w * df.drec_w).sum() / (df.w * df.dlog_w ** 2).sum())
    bs = Lk.groupby('target').apply(slope)
    s = TS(bs)
    # v2：拆成两行。合并斜率的 CI 按国家 cluster bootstrap（本表新算，B=5000，种子 42）；逐国均值用 t 区间。
    # 合并 WLS 斜率 = Σ w·x·y / Σ w·x²，按国家分块求和后抽国家（与逐行 concat 等价，更快）
    num = (Lk.w * Lk.dlog_w * Lk.drec_w).groupby(Lk.target).sum()
    den = (Lk.w * Lk.dlog_w ** 2).groupby(Lk.target).sum()
    assert abs(num.sum() / den.sum() - slope(Lk)) < 1e-9
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(5000):
        pick = rng.choice(len(num), len(num))
        boot.append(num.to_numpy()[pick].sum() / den.to_numpy()[pick].sum())
    blo, bhi = np.percentile(boot, [2.5, 97.5])
    L9 = Lk[~Lk.target.isin(ROPT)]
    add('H4.dose_slope.pooled', 'H4', '类别构成剂量–反应：某类比 SRS 多拿标签时该类召回的斜率（pp / 单位 Δlog(1+n)），合并（WLS）斜率', slope(Lk),
        'pp/Δlog', lo95=blo, hi95=bhi, n=11, noROPT=slope(L9), noROPT_ci='9 国合并斜率，点估计（v2 补算）', grade='C',
        prereg='否（事后）', reuse=RU_G_POST, metric='逐类召回', source=src,
        func='numbers_table.h4_strategy（斜率与 strategy/mechanism/wls.slope 同式；CI 为本表新算的国家 cluster bootstrap）',
        doc={'v': '3.3'}, doc_src='STRATEGY §3.1；REPOSITION H4',
        note='95% CI = 按目标国 cluster bootstrap（B=5000，种子 42，本表新算；原报告未给合并斜率的 CI）。r=50 加权下近乎机械。'
             'v1 把逐国均值的 t 区间放在本行，已拆开。')
    add('H4.dose_slope.country_mean', 'H4', '同上，逐国斜率的均值（每国一条 WLS 斜率）', None, 'pp/Δlog', s=s,
        noROPT=float(bs.drop(ROPT).mean()), noROPT_ci=noropt_str(bs, 2)[1], grade='C', prereg='否（事后）', reuse=RU_G_POST,
        metric='逐类召回', source=src, func='numbers_table.h4_strategy（与 strategy/mechanism/wls.slope_ctry 同式）',
        doc={'v': '4.2', 'lo': '0.9', 'hi': '7.5'}, doc_src='STRATEGY §3.1（"逐国均值 +4.2 [+0.9, +7.5]"）',
        per_country=bs.to_dict(), note='95% t 区间（df=10）。')


# ================================================================== H5：标签怎么用
H5_REG = {}


def h5_usage(Rr, Cc):
    src = f'{RES}/check3_arms.csv'
    srcc = f'{RES}/check3_capacity_arms.csv'
    for feat, dv, dpem in (('ae', '95', '31'), ('s2', '97', '58')):
        r = Rr['oa'].loc[feat]
        base = (r['PO'] - r['L0']).sum()
        pk = r[[f'PK50_s{s}' for s in range(3)]].mean(axis=1)
        keep = ~r.index.isin(ROPT)
        add(f'H5.pk50_recovery.es.{feat}', 'H5', f'50 个随机目标标签估计先验的 OA 恢复率 Σ(PK50−L0)/Σ(PO−L0)（早停，3 种子，{feat.upper()}）',
            100 * (pk - r['L0']).sum() / base, '%', n=11,
            noROPT=100 * (pk - r['L0'])[keep].sum() / (r['PO'] - r['L0'])[keep].sum(), grade='B', prereg='是（rev1 次要 3）',
            reuse=RU_C3_PRE, metric='OA', source=src, func='numbers_table.h5_usage（analyze_check3 同式）', doc={'v': dv},
            doc_src='check3_report 次要 3；REPOSITION H5', per_country=(100 * (pk - r['L0']) / (r['PO'] - r['L0'])).to_dict(),
            note=f'RO 占先验增益 {100 * (r["PO"] - r["L0"])["RO"] / base:.0f}%；PK50 − PO 均值 {(pk - r["PO"]).mean():+.2f} pp。')
        add(f'H5.pem_recovery.{feat}', 'H5', f'不用标签的 EM 先验估计 OA 恢复率（早停，{feat.upper()}）',
            100 * (r['PEM'] - r['L0']).sum() / base, '%', n=11, noROPT=100 * (r['PEM'] - r['L0'])[keep].sum() / (r['PO'] - r['L0'])[keep].sum(),
            grade='B', prereg='是（rev1 次要 3）', reuse=RU_C3_PRE, metric='OA', source=src, func='numbers_table.h5_usage',
            doc={'v': dpem}, doc_src='CHECK3_RESULTS；REPOSITION H5')
        rb = Rr['ba_ge30'].loc[feat]
        g = rb['PO'] - rb['LS']
        m9, c9 = noropt_str(g, 2)
        add(f'H5.prior_hurts_ba.{feat}', 'H5', f'按目标先验修正对 ba_ge30 的影响 ba(PO) − ba(LS)（早停，{feat.upper()}）', None, 'pp',
            s=TS(g), noROPT=m9, noROPT_ci=c9, grade='B（定义的推论）', prereg='是（rev1 次要 2）', reuse=RU_C3_PRE, metric='ba_ge30',
            source=src, func='numbers_table.h5_usage', doc={'v': {'ae': '-12.21', 's2': '-12.77'}[feat]},
            doc_src='check3_report 次要 2；REPOSITION H5', per_country=g.to_dict(), note=f'为负的国家 {int((g < 0).sum())}/11。')
    T = Cc.pivot_table(index=['feat', 'cap', 'target'], columns='arm', values=['oa'])['oa']
    for feat in ('ae', 's2'):
        for cap, dv in (('es', {'ae': '95', 's2': '97'}), ('c300', {'ae': '85', 's2': '93'})):
            r = T.loc[(feat, cap)]
            pk = r[[f'PK50_s{s}' for s in range(20)]].mean(axis=1)
            keep = ~r.index.isin(ROPT)
            add(f'H5.pk50_recovery20.{cap}.{feat}', 'H5', f'PK50 恢复率（20 种子，{cap}，{feat.upper()}）',
                100 * (pk - r.L0).sum() / (r.PO - r.L0).sum(), '%', n=11,
                noROPT=100 * (pk - r.L0)[keep].sum() / (r.PO - r.L0)[keep].sum(), grade='B', prereg='否（事后容量检验）',
                reuse=RU_C3_POST, metric='OA', source=srcc, func='numbers_table.h5_usage（analyze_check3_capacity 同式）',
                doc={'v': dv[feat]}, doc_src='check3_capacity_report.txt', note=f'PK50 − PO 均值 {(pk - r.PO).mean():+.2f} pp。')
    # 区域聚集的 50 个标签（检验 3 复核 task B，事后）
    srcb = f'{VER}/check3/taskB/b3_rows.pkl'
    df = rpkl(srcb)
    REG = {}
    for f in ('ae', 's2'):
        base = df.groupby('target')[f'{f}_PO'].first() - df.groupby('target')[f'{f}_L0'].first()
        L0 = df.groupby('target')[f'{f}_L0'].first()
        keep = ~base.index.isin(ROPT)
        for m, dv in ((1, '约80'), (3, '91–94')):
            d = df[(df.ncl == 20) & (df.m == m)]
            gm = d.groupby('target')[f'{f}_PK'].mean()
            rec = 100 * (gm - L0).sum() / base.sum()
            rec9 = 100 * (gm - L0)[keep].sum() / base[keep].sum()
            d50 = df[(df.ncl == 50) & (df.m == m)].groupby('target')[f'{f}_PK'].mean()
            rec50 = 100 * (d50 - L0).sum() / base.sum()
            REG[(m, f)] = (rec, rec50)
            add(f'H5.region{m}.{f}', 'H5', f'50 个标签只来自 {m} 个区域时的先验 OA 恢复率（区域 = 候选池 20 个 KMeans 簇之一；{f.upper()}）', rec, '%',
                n=11, noROPT=rec9, noROPT_ci='9 国合并比，点估计（v2 补算）', grade='C（不进摘要）',
                prereg='否（检验 3 复核 task B，事后）', reuse=RU_C3_POST, metric='OA', source=srcb,
                func='numbers_table.h5_usage（复现 verify/check3/taskB/b3_sum.py）', doc={'v': dv}, doc_src='CHECK3_RESULTS；REPOSITION H5',
                note=f'"区域" = 候选池 20 个 KMeans 簇之一，结果依赖这一定义。50 簇敏感性：{rec50:.1f}%。每配置 60 次重复。'
                     'CONCLUSIONS C3e 把"标签的空间分布效应"定为 C，但第 68、988 行把这组数字列在 B 级条目下；'
                     'REPOSITION H5 把它列在"先验估计 B（摘要）"的数字里，应移出摘要（discrepancies 第 19 条）。')
    H5_REG.update(REG)
    # 决策头：F3b（重训 ⊕ 类均衡头，α=0.4）− 重训，k=200，λ=0.1
    srcf = f'{VER}/pivot_eval/prior_shift/frontier2_rows.csv'
    F = rcsv(srcf)
    key = ['unit', 'target', 'clf', 'k']
    for k in (50, 200, 400):
        ref = F[(F.design == 'srs_G') & (F.fam == 'retrain') & (F.k == k)].set_index(key)[['ba', 'oa']]
        x = F[(F.design == 'srs_G') & (F.fam == 'F3b') & (F.alpha == 0.4) & (F.k == k)].set_index(key)[['ba', 'oa']]
        j = (x - ref).dropna().reset_index()
        c = j.groupby('target')[['ba', 'oa']].mean()
        s = TS(c.ba)
        so = TS(c.oa)
        s9 = TS(c.ba.drop(ROPT))
        add(f'H5.F3b_a0.4.k{k}', 'H5', f'类均衡决策头与重训的对数线性混合 F3b（α=0.4，λ=0.1）− 同一批 SRS 标签重训（k={k}）', None, 'pp',
            s=s, noROPT=s9['mean'], noROPT_ci=f'[{s9["lo95"]:+.2f}, {s9["hi95"]:+.2f}]，{s9["n_pos"]}/9', grade='C', prereg='否（事后；α、λ 在同一测试集上选）',
            reuse=RU_G_POST, metric='ba_ge30', source=srcf, func='numbers_table.h5_usage（frontier2.py summ 同式）',
            doc={'v': '+0.94', 'lo': '+0.69', 'hi': '+1.19', 'oa': '-0.05', 'noROPT': '+1.01'} if k == 200 else
            {50: {'v': '+0.91'}, 400: {'v': '+0.88'}}[k], doc_src='prior_shift/frontier2_report.txt；REPOSITION H5',
            per_country=c.ba.to_dict(), note=f'OA {so["mean"]:+.2f} [{so["lo95"]:+.2f}, {so["hi95"]:+.2f}]。注意：这是"重训 ⊕ 头"的混合，不是只用头。')
        if k == 200:
            ROWS[-1]['_oa'] = so['mean']
            ROWS[-1]['note'] += ('CONCLUSIONS 第 991 行把它写成"只用 k 个目标标签拟合一个类均衡的决策层"，与 frontier2.py 中 F3b 的定义'
                                 '（(1−α)·log P_重训 + α·log softmax(头)）不符。与 rev4 D1 的 Δhead 不是同一个量，不能并列。')
    # F3b 的"标签当量"（CONCLUSIONS 第 992 行 / PIVOT §2）：50 个标签加 F3b 的水平 vs 更多标签重训的水平
    lv = {}
    for fam, k, alpha in (('F3b', 50, 0.4), ('retrain', 200, None), ('retrain', 400, None)):
        z = F[(F.design == 'srs_G') & (F.fam == fam) & (F.k == k)]
        if alpha is not None:
            z = z[z.alpha == alpha]
        lv[(fam, k)] = z.groupby('target')[['ba', 'oa']].mean()
    f50, r200, r400 = lv[('F3b', 50)], lv[('retrain', 200)], lv[('retrain', 400)]
    f50a5 = F[(F.design == 'srs_G') & (F.fam == 'F3b') & (F.k == 50) & (F.alpha == 0.5)].groupby('target').ba.mean()
    keep9 = ~f50.index.isin(ROPT)
    add('H5.F3b_labelequiv', 'H5', '50 个 SRS 标签加 F3b（α=0.4）的 BA 水平，对比更多 SRS 标签重训的 BA 水平（国家均值）',
        f'9 国：F3b k=50 {f50.ba[keep9].mean():.2f} vs 重训 k=400 {r400.ba[keep9].mean():.2f}；11 国：F3b k=50 {f50.ba.mean():.2f} vs 重训 k=200 {r200.ba.mean():.2f}',
        '%', n=11, noROPT_na='不适用：本行的主口径即去 RO/PT 的 9 国（11 国版并列）', grade='C（事后；α、λ 在同一测试集上选）',
        prereg='否（PIVOT 事后）', reuse=RU_G_POST, metric='ba_ge30（OA 附注）', source=srcf,
        func='numbers_table.h5_usage（frontier2_rows 水平值的国家均值）',
        doc={'v9_f3b': '48.81', 'v9_rt400': '48.80', 'v11_f3b': '46.83', 'v11_rt200': '46.84', 'oa9_f3b': '80.1', 'oa9_rt400': '81.4'},
        doc_src='PIVOT_EVALUATION §2（标签当量）；CONCLUSIONS 第 992 行',
        note=f'OA（9 国）：F3b k=50 {f50.oa[keep9].mean():.1f} vs 重训 k=400 {r400.oa[keep9].mean():.1f}。只在 BA 上相当，OA 更低，'
             '不能叫"标签当量"（CONCLUSIONS 第 1238 行）。'
             f'PIVOT §2 的 11 国"46.83"与 α=0.5 的值（{f50a5.mean():.2f}）一致，而 9 国"48.81"是 α=0.4：同一段混用了两个 α（见 discrepancies 第 20 条）。')
    ROWS[-1]['_a5'] = (f50a5.mean(), f50a5.drop(ROPT).mean())
    ROWS[-1]['_lq'] = dict(v9_f3b=f50.ba[keep9].mean(), v9_rt400=r400.ba[keep9].mean(), v11_f3b=f50.ba.mean(),
                           v11_rt200=r200.ba.mean(), oa9_f3b=f50.oa[keep9].mean(), oa9_rt400=r400.oa[keep9].mean())
    txt = rtxt(f'{VER}/pivot_eval/prior_shift/loco_alpha.txt')
    vals = [float(x) for x in re.findall(r'留出 ΔBA ([+-][\d.]+)', txt)]
    add('H5.F3b_loco_alpha', 'H5', '留一国选 α 后被留出国家的 ΔBA（4 个设计 × k）', f'{min(vals):+.2f} 至 {max(vals):+.2f}', 'pp', n=11,
        noROPT_na='未算：只读到 loco_alpha.txt 的 11 国结果；去 RO/PT 需重跑留一选 α 流程（方案外）',
        grade='C', prereg='否（事后）', reuse=RU_G_POST, metric='ba_ge30', source=f'{VER}/pivot_eval/prior_shift/loco_alpha.txt',
        func='读取（正则）', mode='读取', doc={'v_min': '+0.46', 'v_max': '+1.50'}, doc_src='REPOSITION H5',
        note='SRS k=200 行：' + (re.search(r'srs_G k=200:.*', txt).group(0) if re.search(r'srs_G k=200:.*', txt) else ''))
    # λ 敏感性：v2 从 frontier3_rows.csv 重算（frontier3_report 同式：F3b(i300, α=0.5, 设计权重=1) − 重训 i300）
    srcl = f'{VER}/pivot_eval/prior_shift/frontier3_rows.csv'
    F3 = rcsv(srcl)
    x3 = F3[(F3.design == 'srs_G') & (F3.k == 200) & (F3.it == 'i300')]
    ref3 = x3[x3['var'] == 'retrain'].set_index(['unit', 'target', 'clf'])[['ba', 'oa']]
    lam, lam9 = {}, {}
    for lm in ('0.02', '0.1', '0.5'):
        h3 = x3[(x3['var'] == f'F3b_l{lm}_dw1') & (x3.alpha == 0.5)].set_index(['unit', 'target', 'clf'])[['ba', 'oa']]
        c3 = (h3 - ref3).dropna().groupby('target').mean()
        s3 = TS(c3.ba)
        lam[lm] = s3
        lam9[lm] = TS(c3.ba.drop(ROPT))['mean']
    add('H5.F3b_lambda_sens', 'H5', 'F3b 对 λ 的敏感性（α=0.5，k=200，SRS；F3b(i300) − 重训 i300）',
        '；'.join(f'λ={k_}: {v_["mean"]:+.2f} [{v_["lo95"]:+.2f}, {v_["hi95"]:+.2f}]' for k_, v_ in lam.items()), 'pp',
        n=11, noROPT='；'.join(f'λ={k_}: {v_:+.2f}' for k_, v_ in lam9.items()), noROPT_ci='9 国均值（v2 补算）',
        grade='C', prereg='否', reuse=RU_G_POST, metric='ba_ge30', source=srcl,
        func='numbers_table.h5_usage（frontier3_report 同式；v1 为读取报告）', mode='重算',
        doc={'v_0.02': '+2.46', 'v_0.1': '+1.28', 'v_0.5': '+0.25'}, doc_src='CONCLUSIONS V1；frontier3_report.txt',
        note='注意 α=0.5，而 headline +0.94 为 α=0.4；CONCLUSIONS V1 未注明 α。')
    ROWS[-1]['_lam'] = {f'v_{k_}': v_['mean'] for k_, v_ in lam.items()}
    # 只用头（bal，λ=0.1）：相对重训与相对 k=0（calib_rows；不是 rev4 D1）
    srcc2 = f'{VER}/pivot_eval/prior_shift/calib_rows.csv'
    Cr = rcsv(srcc2)
    k0 = Cr[Cr['head'] == 'k0'].drop_duplicates(['unit', 'clf']).set_index(['unit', 'clf'])[['ba', 'oa']]
    for k in (50, 200, 400):
        h = Cr[(Cr.design == 'srs_G') & (Cr['head'] == 'bal_l0.1') & (Cr.k == k)].set_index(['unit', 'clf', 'target'])[['ba', 'oa']]
        rt = Cr[(Cr.design == 'srs_G') & (Cr['head'] == 'retrain') & (Cr.k == k)].set_index(['unit', 'clf', 'target'])[['ba', 'oa']]
        for ref_name, ref in (('k0', None), ('retrain', rt)):
            if ref is None:
                hh = h.reset_index()
                r0 = k0.reindex(pd.MultiIndex.from_frame(hh[['unit', 'clf']])).to_numpy()
                dd = hh.assign(dba=hh.ba - r0[:, 0], doa=hh.oa - r0[:, 1])
            else:
                dd = (h - ref).rename(columns={'ba': 'dba', 'oa': 'doa'}).reset_index()
            c = dd.groupby('target')[['dba', 'doa']].mean()
            s = TS(c.dba)
            so = TS(c.doa)
            s9 = TS(c.dba.drop(ROPT))
            dmap = {('k0', 50): {'v': '+4.71'}, ('k0', 200): {'v': '+5.95'}, ('k0', 400): {'v': '+6.38'},
                    ('retrain', 200): {'v': '+3.20', 'lo': '+1.37', 'hi': '+5.02', 'oa': '-5.69'}}
            add(f'H5.balhead_only.vs_{ref_name}.k{k}', 'H5', f'只用类均衡决策头（冻结外国模型，λ=0.1，G 的 SRS 标签 k={k}）相对{"k=0 外国模型" if ref_name == "k0" else "同标签重训"}',
                None, 'pp', s=s, noROPT=s9['mean'], noROPT_ci=f'[{s9["lo95"]:+.2f}, {s9["hi95"]:+.2f}]，{s9["n_pos"]}/9',
                grade='C', prereg='否（PIVOT 事后）', reuse=RU_G_POST, metric='ba_ge30', source=srcc2,
                func='numbers_table.h5_usage（calib_diag 的逐单元行）', doc=dmap.get((ref_name, k)),
                doc_src='prior_shift/calib_report.txt' if (ref_name, k) in dmap else '', per_country=c.dba.to_dict(),
                note=f'OA {so["mean"]:+.2f} [{so["lo95"]:+.2f}, {so["hi95"]:+.2f}]，{so["n_pos"]}/11 为正。只有重复 0（22 单元）；'
                     '不是 rev4 D1（D1 另行子抽样、44 单元），与 D1 的 Δhead 数值接近但不完全相同。')
            if ref_name == 'retrain' and k == 200:
                ROWS[-1]['_oa'] = so['mean']
    full = Cr[Cr['head'] == 'bal_full'].set_index(['unit', 'clf'])[['ba', 'oa', 'target']]
    r0 = k0.reindex(full.index)
    dd = full.assign(dba=full.ba - r0.ba, doa=full.oa - r0.oa).reset_index()
    c = dd.groupby('target')[['dba', 'doa']].mean()
    s = TS(c.dba)
    so = TS(c.doa)
    add('H5.balhead_full.vs_k0', 'H5', '只用类均衡决策头，全池约 3 万标签（λ=1e-4）相对 k=0', None, 'pp', s=s,
        noROPT=float(c.dba.drop(ROPT).mean()), noROPT_ci=noropt_str(c.dba, 2)[1], grade='C（上界）', prereg='否', reuse=RU_G_POST,
        metric='ba_ge30', source=srcc2,
        func='numbers_table.h5_usage（calib_diag.py 第 202 行：fit_bcts(…, 1e-4)）', doc={'v': '+9.26', 'lo': '+5.30', 'hi': '+13.21'},
        doc_src='prior_shift/calib_report.txt', per_country=c.dba.to_dict(),
        note=f'OA {so["mean"]:+.2f} [{so["lo95"]:+.2f}, {so["hi95"]:+.2f}]。v1 误写为 λ=0.1：calib_diag.py 的全池头用 λ=1e-4'
             '（rev4 D1 核对 (2) 也按 λ=1e-4 逐位复现了这 44 行）；同 λ=0.1 的全池头见 D1.ba_ge30.dhead.full。'
             '与 H1附 的 BA 版上界（oracle_G_buffered，无 λ、重加权）口径也不同。')
    # 重训与偏置头下方法排序的相关
    src3 = f'{VER}/pivot_eval/codesign/diag3_sel_rows.csv'
    d = rcsv(src3)
    kk = ['unit', 'feat', 'k', 'learner']
    sref = d[d.sel == 'none_srs'].set_index(kk)[['ba', 'oa']]
    x = d[d.sel != 'none_srs'].copy()
    ix = pd.MultiIndex.from_frame(x[kk])
    for m in ('ba', 'oa'):
        x['d_' + m] = x[m].to_numpy() - sref[m].reindex(ix).to_numpy()
    x = x.dropna(subset=['d_ba'])
    sels = [f'{c}_{m}' for m in ALL8 for c in ('AE', 'S2')]
    out = {}
    for tag, xx in (('', x), ('9', x[~x.tgt.isin(ROPT)])):
        for m in ('ba', 'oa'):
            p = xx[xx.sel.isin(sels)].groupby(['sel', 'k', 'learner'])['d_' + m].mean().unstack('learner')
            out[m + tag] = p[['lgb', 'bias']].corr().iloc[0, 1]
            out[m + '_sp' + tag] = p[['lgb', 'bias']].corr(method='spearman').iloc[0, 1]
    add('H5.rank_corr_retrain_vs_bias', 'H5', '同一批选样下，LightGBM 重训与偏置头之间"方法 − SRS"的相关（16 选样 × 2 k = 32 点；BA，Pearson）',
        float(out['ba']), 'r', n=32, noROPT=float(out['ba9']),
        noROPT_ci=f'去 RO/PT 的单元重算；BA Spearman {out["ba_sp9"]:.2f}，OA Pearson {out["oa9"]:.2f}、Spearman {out["oa_sp9"]:.2f}（v2 补算）',
        grade='C', prereg='否', reuse=RU_G_POST, metric='ba（OA 附注）', source=src3,
        func='numbers_table.h5_usage（复现 strategy/mechanism/e1_usage.py）', doc={'v': '0.12', 'oa_spearman': '0.09'},
        doc_src='STRATEGY ⑥（附录 D 写明 "BA 为 Pearson 0.12，OA 为 Spearman 0.09"）；REPOSITION H5',
        note=f'BA：Pearson {out["ba"]:.2f}，Spearman {out["ba_sp"]:.2f}；OA：Pearson {out["oa"]:.2f}，Spearman {out["oa_sp"]:.2f}。'
             '单元等权（非国家等权）。文档对 BA、OA 用了不同的相关系数。')
    ROWS[-1]['_oa_sp'] = out['oa_sp']
    ROWS[-1]['_corr'] = out


# ================================================================== rev4 D1：决策层标签效率曲线（读取 D1 输出的国家值，重算 t 汇总）
D1_OUT = {}


def d1_rows():
    src = f'{D1DIR}/decision_curve_country.csv'
    C = rcsv(src)
    summ = rtxt(f'{D1DIR}/decision_curve_summary.txt')
    assert '7e90ccd5b1c3828f' in summ
    base = dict(grade='C（rev4 预先指定；测试集复用）', prereg='是（rev4 D1，7e90ccd5）', reuse=RU_G_POST,
                learner='LightGBM（冻结的外国模型 + 决策头）', source=src,
                func='numbers_table.d1_rows（decision_curve.py 输出的国家值 → analyze_G.t_summary）', mode='重算（D1 输出）')
    NOTE = ('主口径为"两特征国家值平均"（作者选定的汇总口径，rev4 未规定跨特征合并），逐特征值见注。'
            '"决策层可挽回部分"= 冻结外国 LightGBM（P_T 温度校准底座）上只重拟温度与类偏置；λ=0.1、类均衡加权 NLL、'
            '从 3 万候选池抽 k 个；与 H1附 上界（oracle_G_buffered）和 H5.balhead_full（λ=1e-4）都不是同一个量。')
    docs = {('ba_ge30', 'dhead_PT', 50): {'v': '+4.48', 'lo': '+1.64', 'hi': '+7.32'},
            ('ba_ge30', 'dhead_PT', 400): {'v': '+6.53', 'lo': '+4.05', 'hi': '+9.01'},
            ('ba_ge30', 'dret', 50): {'v': '+1.52', 'lo': '-0.87', 'hi': '+3.91'},
            ('ba_ge30', 'diff_PT', 50): {'v': '+2.95', 'lo': '+2.02', 'hi': '+3.88'},
            ('ba_ge30', 'diff_PT', 400): {'v': '+3.19', 'lo': '+0.78', 'hi': '+5.61'},
            ('ba_ge30', 'dfull_PT', 0): {'v': '+6.55', 'lo': '+3.99', 'hi': '+9.12', 'noROPT': '+5.67'},
            ('oa', 'dfull_PT', 0): {'v': '-0.79', 'noROPT': '-4.02'}}
    out = {}
    for met, lab in (('ba_ge30', 'ba_ge30'), ('oa', 'OA')):
        for col, what, ks in (('dhead_PT', 'Δhead(k)：只调决策层相对 P_T 底座', (50, 100, 200, 400, 1000)),
                              ('dret', 'Δretrain(k)：同单元 G 的 SRS(k) 重训 − foreign_only', (50, 100, 200, 400)),
                              ('diff_PT', 'Δhead(k) − Δretrain(k)', (50, 100, 200, 400))):
            for k in ks:
                z = C[(C.metric == met) & (C.k == k)]
                v = z[z.clf == 'both'].set_index('target')[col]
                sa = z[z.clf == 'ae'][col].mean()
                ss = z[z.clf == 's2'][col].mean()
                s = TS(v)
                m9, c9 = noropt_str(v, 2)
                out[(met, col, k)] = (s, TS(v.drop(ROPT)))
                add(f'D1.{met}.{col.replace("_PT", "")}.k{k}', 'H5/D1', f'rev4 D1：{what}（{lab}，k={k}）', None, 'pp', s=s,
                    noROPT=m9, noROPT_ci=c9, metric=lab, doc=docs.get((met, col, k)),
                    doc_src='decision_curve_summary.txt' if (met, col, k) in docs else '', per_country=v.to_dict(),
                    note=f'逐特征国家均值：AE {sa:+.2f}，S2 {ss:+.2f}。' + (
                        '头与重训用的不是同一批标签子样本（头：D1 种子 5 次；重训：G 的 SRS 选样 1 次）；标签配对版见 decision_curve_summary.txt 附加部分。'
                        if col == 'diff_PT' else '') + NOTE, **base)
        z = C[(C.metric == met) & (C.k == 50)]
        v = z[z.clf == 'both'].set_index('target')['dfull_PT']
        s = TS(v)
        m9, c9 = noropt_str(v, 2)
        out[(met, 'dfull_PT', 0)] = (s, TS(v.drop(ROPT)))
        add(f'D1.{met}.dhead.full', 'H5/D1', f'rev4 D1：Δhead(全池约 3 万标签，λ=0.1)（{lab}）', None, 'pp', s=s, noROPT=m9,
            noROPT_ci=c9, metric=lab, doc=docs.get((met, 'dfull_PT', 0)), doc_src='decision_curve_summary.txt',
            per_country=v.to_dict(), note=f'逐特征国家均值：AE {z[z.clf == "ae"].dfull_PT.mean():+.2f}，S2 {z[z.clf == "s2"].dfull_PT.mean():+.2f}。'
            + ('SI、EE、LV 的全池值低于部分 k 的值，全池不是逐国严格上界（D1 报告第 5 条）。' if met == 'ba_ge30' else
               'OA 上全池头的增益接近 0 或为负（类均衡头面向 BA）。') + NOTE, **base)
        full = out[(met, 'dfull_PT', 0)]
        rat = '  '.join(f'k{k}:{out[(met, "dhead_PT", k)][0]["mean"] / full[0]["mean"]:.2f}' for k in (50, 100, 200, 400, 1000))
        rat9 = '  '.join(f'k{k}:{out[(met, "dhead_PT", k)][1]["mean"] / full[1]["mean"]:.2f}' for k in (50, 100, 200, 400, 1000))
        add(f'D1.{met}.ratio', 'H5/D1', f'rev4 D1：Δhead(k)/Δhead(全池)，国家均值之比（{lab}；描述，不推断）', rat, '', n=11, noROPT=rat9,
            noROPT_ci='9 国均值之比', metric=lab, per_country=None,
            note=('OA 的全池分母接近 0 或为负，比例没有解释意义，只为与规格一致而列出。' if met == 'oa' else
                  '同一 λ=0.1 下，k≈200–400 已达到全池水平；λ 决定平台高度（BA–OA 工作点），见 d1_review/lam_sens_rows.csv（复核事后，C 级，不用于选 λ）。')
            + NOTE, **base)
        if met == 'ba_ge30':
            D1_OUT.update(out)


# ================================================================== H6：AE 作为特征
def h6_feature(R):
    S = srs_frame(R)
    W = S.groupby(A.UNIT + ['clf', 'k'])[['ba_ge30', 'oa']].mean()
    dm_ba = {0: ('+2.42', '-0.36', '+5.20', '3.28'), 50: ('+3.02', '-0.28', '+6.32', '3.11'), 100: ('+2.79', '-0.25', '+5.84', '3.05'),
             200: ('+3.11', '+0.49', '+5.74', '3.33'), 400: ('+2.47', '+0.22', '+4.73', '2.81')}
    dm_oa = {0: ('+1.05', '-0.98', '+3.07', '2.25'), 50: ('+2.79', '+1.36', '+4.22', '2.42'), 100: ('+2.71', '+0.93', '+4.48', '2.38'),
             200: ('+2.54', '+1.10', '+3.98', '2.27'), 400: ('+2.09', '+1.02', '+3.16', '2.07')}
    unitd = {}
    for met, dm in (('ba_ge30', dm_ba), ('oa', dm_oa)):
        for k in (0, 50, 100, 200, 400):
            w = W[met].xs(k, level='k').unstack('clf')
            u = (w['ae'] - w['s2']).rename('d').reset_index()
            unitd[(met, k)] = u.set_index(A.UNIT).d
            v = cv_from_unit(u)
            s = TS(v)
            s9 = TS(v.drop(ROPT))
            d = dm[k]
            add(f'H6.dfeat_lgb.{met}.k{k}', 'H6', f'Δ_feat(LightGBM, k={k})：同一批标签下 AE 特征分类器 − S2 特征分类器（rev3 B1）', None, 'pp',
                s=s, noROPT=s9['mean'], noROPT_ci=f'[{s9["lo95"]:+.2f}, {s9["hi95"]:+.2f}]，{s9["n_pos"]}/9，p={s9["p"]:.3f}',
                grade='C（待 rev2 确认）', prereg='rev3 B1 预先指定（LightGBM 部分此前已看过）', reuse=RU_G_POST, metric=met,
                source=SRC_G, func='numbers_table.h6_feature（单元配对 → 国家 → t）',
                doc={'v': d[0], 'lo': d[1], 'hi': d[2], 'noROPT': d[3]}, doc_src='label_budget_line/feat_by_k.csv；REPOSITION H6',
                per_country=v.to_dict(),
                note='k=0 的两个重复共用同一外国模型。' + (f'为负的国家：{", ".join(v[v < 0].index)}。' if met == 'ba_ge30' else ''))
            if met == 'ba_ge30' and k in (0, 400):
                ROWS[-1]['_neg'] = list(v[v < 0].index)
        ch = (unitd[(met, 400)] - unitd[(met, 0)]).rename('d').reset_index()
        v = cv_from_unit(ch)
        s = TS(v)
        s9 = TS(v.drop(ROPT))
        add(f'H6.dfeat_change_lgb.{met}', 'H6', f'Δ_feat(LightGBM, 400) − Δ_feat(LightGBM, 0)（单元层变化量，rev3 B1）', None, 'pp', s=s,
            noROPT=s9['mean'], noROPT_ci=f'[{s9["lo95"]:+.2f}, {s9["hi95"]:+.2f}]，{s9["n_pos"]}/9', grade='C',
            prereg='rev3 B1', reuse=RU_G_POST, metric=met, source=SRC_G, func='numbers_table.h6_feature',
            doc={'v': '+0.05', 'lo': '-0.92', 'hi': '+1.02'} if met == 'ba_ge30' else None, doc_src='REPOSITION H6' if met == 'ba_ge30' else '',
            per_country=v.to_dict(), note='只是没检出变化，不是等价。')
        if met == 'ba_ge30':
            s0 = TS(cv_from_unit(unitd[(met, 0)].rename('d').reset_index()))
            s4 = TS(cv_from_unit(unitd[(met, 400)].rename('d').reset_index()))
            same = np.sign(s0['mean']) == np.sign(s4['mean'])
            inside = (s['lo95'] > -1) and (s['hi95'] < 1)
            v0 = cv_from_unit(unitd[(met, 0)].rename('d').reset_index()).drop(ROPT)
            v4 = cv_from_unit(unitd[(met, 400)].rename('d').reset_index()).drop(ROPT)
            same9 = np.sign(v0.mean()) == np.sign(v4.mean())
            inside9 = (s9['lo95'] > -1) and (s9['hi95'] < 1)
            add('H6.shift_criterion.lgb', 'H6', 'rev3 B1"整体平移"判据（LightGBM 部分）：Δ_feat(0) 与 Δ_feat(400) 同号，且变化量 95% CI 在 (−1, +1) 内',
                '满足' if (same and inside) else '不满足', '', n=11,
                noROPT='满足' if (same9 and inside9) else '不满足',
                noROPT_ci=f'9 国：同号 {"是" if same9 else "否"}；变化量 95% CI [{s9["lo95"]:+.3f}, {s9["hi95"]:+.3f}]（描述；rev3 B1 判据按 11 国）',
                grade='判据（需 ≥3/4 学习器）', prereg='是（rev3 B1）',
                reuse=RU_G_POST, metric='ba_ge30', source=SRC_G, func='numbers_table.h6_feature',
                note=f'同号：{"是" if same else "否"}；变化量 95% CI [{s["lo95"]:+.3f}, {s["hi95"]:+.3f}] {"在" if inside else "不在"} (−1, +1) 内。')
            ROWS[-1]['_crit'] = (same, inside, s['lo95'], s['hi95'])
    # SRS 学习曲线平行（k0→400 两种分类器）
    for clf, dv in (('ae', '+3.36'), ('s2', '+3.31')):
        w = W['ba_ge30'].xs(clf, level='clf').unstack('k')
        v = (w[400] - w[0]).groupby('target').mean()
        m9, c9 = noropt_str(v, 2)
        add(f'H6.srs_curve_k0_400.{clf}', 'H6', f'SRS 学习曲线 k=0→400 增益（{clf.upper()} 分类器，国家均值）', None, 'pp', s=TS(v), noROPT=m9,
            noROPT_ci=c9, grade='B（国家分布 C）', prereg='否（描述）', reuse=RU_G_POST, metric='ba_ge30', source=SRC_G,
            func='numbers_table.h6_feature', doc={'v': dv}, doc_src='REPOSITION H6（"AE +3.36，S2 +3.31"）', per_country=v.to_dict())
    for lrn in ('LR', 'MLP', 'RF'):
        add(f'H6.dfeat.{lrn}', 'H6', f'Δ_feat({lrn}, k∈{{0,50,100,200,400}}) 与变化量', '待 rev2', 'pp', grade='待定', prereg='是（rev3 B1）',
            reuse=RU_G_POST, learner=lrn, source='results/guidance_G_ml/（未读取）', func='analyze_ML_rev3.py', mode='占位',
            note='LR 的 k=0 AE−S2 曾在简报中出现（rev3 A2 已披露为非盲），本表不引用。' if lrn == 'LR' else '')
    add('H6.shift_criterion.all', 'H6', '"整体平移"判据的学习器计数 X/4', '待 rev2', '', grade='待定', prereg='是（rev3 B1）', reuse=RU_G_POST,
        learner='LightGBM+LR+MLP+RF', source='analyze_ML_rev3.py', func='analyze_ML_rev3.py', mode='占位')
    add('H6.B3_rf_k0', 'H6', 'RF 在 k=0 时的 AE − S2（OA 与 ba_ge30；与 Cheng 主分类器对照口径，rev3 B3）', '待 rev2', 'pp', grade='描述',
        prereg='是（rev3 B3）', reuse=RU_G_POST, learner='RF', source='results/guidance_G_ml/（未读取）', func='analyze_ML_rev3.py', mode='占位',
        note='RF 的 AT k=0 已看过（rev2 §5）。')
    # M9：模型诊断 5 国试点里的 LR / MLP k=0 AE−S2（与 rev2 口径不同）。v2：从 alt_long.csv 重算（v1 为读取 out_alt.txt）
    for lrn, ck, lab, dv in (('mlp', '40', 'MLP（最终点，5 国，划分 1）', '+2.86'), ('lr', 'C1.0', 'LR（C=1，5 国，划分 1）', '-1.48'),
                             ('lr', 'C0.01', 'LR（C=0.01，5 国，划分 1）', '+0.49')):
        x = alt_pivot(lrn, ck)['k0'].unstack('feat')
        v = x.ae - x.s2
        v9 = v.drop([c for c in ROPT if c in v.index])
        rid = f'H6.M9.{lrn}_k0' if ck != 'C0.01' else 'H6.M9.lr_C0.01_k0'
        add(rid, 'H6', f'M9：k=0 分类特征 AE − S2 的 ba（{lab}）', None, 'pp', s=TS(v), noROPT=float(v9.mean()),
            noROPT_ci=f'去 PT 后 n={len(v9)}（5 国中无 RO）', grade='C（n=5）', prereg='否（模型诊断）', reuse=RU_MD,
            learner=lab + '（MODEL_DIAGNOSIS 试点，不属于 rev2 估计量）', metric='ba_ge30', source=SRC_ALT,
            func='numbers_table.alt_pivot（k0 行 AE − S2）', mode='重算', doc={'v': dv},
            doc_src='MODEL_DIAGNOSIS §4 第 5 点（M9）', per_country=v.to_dict(),
            note='n=5 国、单划分；与 rev2 的 11 国、按规则选 C 的口径不同，不能直接对比；rev2 出结果后不与 rev2 数字并排放。')
    src3 = f'{VER}/model_diag/altmodels/part1_rows.csv'
    P1 = rcsv(src3)
    k0m = P1[P1.k == 0].pivot_table(index='target', columns=['model', 'feat'], values='ba_ge30')
    allk0 = {}
    for mdl in sorted(set(k0m.columns.get_level_values(0))):
        if (mdl, 'ae') in k0m.columns and (mdl, 's2') in k0m.columns:
            dd_ = (k0m[(mdl, 'ae')] - k0m[(mdl, 's2')]).dropna()
            allk0[mdl] = (dd_.mean(), len(dd_))
    allk0_txt = '；'.join(f'{m_} {v_:+.2f}' for m_, (v_, n_) in allk0.items() if n_ == 11)
    partial = [m_ for m_, (v_, n_) in allk0.items() if n_ < 11]
    if partial:
        allk0_txt += (f'（{"、".join(f"{m_}（n={allk0[m_][1]}）" for m_ in partial)} 在 k=0 不足 11 国，不列数值；'
                      'RF 的 k=0 AE − S2 属 rev3 B3 待 rev2 的量）')
    for mdl, lab in (('lr', 'LR（part1 设置，11 国，划分 0 重复 0）'), ('mlp2', 'MLP (256,128) 早停（part1，11 国）'),
                     ('mlp3', 'MLP 三层（part1，11 国）')):
        x = P1[(P1.model == mdl) & (P1.k == 0)].pivot(index='target', columns='feat', values='ba_ge30')
        v = x.ae - x.s2
        m9, c9 = noropt_str(v, 2)
        add(f'H6.M9b.{mdl}_k0', 'H6', f'补充：k=0 AE − S2 的 ba（{lab}）', None, 'pp', s=TS(v), noROPT=m9, noROPT_ci=c9, grade='C',
            prereg='否', reuse=RU_MD, learner=lab + '（MODEL_DIAGNOSIS 试点，不属于 rev2 估计量）', metric='ba_ge30', source=src3,
            func='numbers_table.h6_feature', per_country=v.to_dict(),
            note=('与 M9 的 −1.48 同为"LR"，但设置不同，符号与幅度都不同；说明 M9 的"符号不稳"依赖设置。' if mdl == 'lr' else '')
            + f'同一文件 k=0 有 11 国的全部学习器的 AE − S2（11 国均值）：{allk0_txt}。v2 补 mlp3，避免只列部分学习器。')


# ================================================================== 补充与占位
def supp(res):
    src = f'{VER}/reposition/overlap_audit/posthoc_summary.json'
    J = json.load(open(_chk(src)))
    o = J['ols_and_tests']['d_ae_s2~dlat']
    txt = rtxt(f'{VER}/reposition/rse_reviewer/crossover_proxy.log')
    m = re.search(r'AE−S2 差的斜率：([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]', txt)
    mfe = re.search(r'目标国固定效应下斜率：AE ([+-][\d.]+)，S2 ([+-][\d.]+)，差 ([+-][\d.]+)', txt)
    pw = rcsv(f'{VER}/reposition/overlap_audit/pairwise_with_distance.csv')
    assert len(pw) == 110
    pw9 = pw[~pw.source.isin(ROPT) & ~pw.target.isin(ROPT)]
    sl9 = float(np.polyfit(pw9.dlat, pw9.d_ae_s2, 1)[0])
    sl11 = float(np.polyfit(pw.dlat, pw.d_ae_s2, 1)[0])
    assert abs(sl11 - o['slope_per_unit']) < 5e-3, (sl11, o['slope_per_unit'])
    add('S.single_source_slope', '补充', '单源两两迁移：AE − S2 的 BA 差对 |Δlat| 的 OLS 斜率（110 对，pairwise_v2）', o['slope_per_unit'], 'pp/°',
        lo95=o['boot95'][0], hi95=o['boot95'][1], n=110, noROPT=sl9,
        noROPT_ci=f'去掉涉及 RO 或 PT 的对后 n={len(pw9)}，OLS 点估计（v2 由 pairwise_with_distance.csv 重算；未做 bootstrap）',
        grade='C（事后，看过 Cheng 之后）', prereg='否',
        reuse='国别留出（pairwise_v2，随机划分；不与 G 混用）', learner='LightGBM（pairwise_v2）', metric='全类 BA',
        source=src, func='读取（JSON）；去 RO/PT 为 np.polyfit 重算', mode='读取',
        doc={'v': '-0.30', 'lo': '-0.91', 'hi': '+0.04'}, doc_src='REPOSITION §3、附录 A',
        note=f'交叉点 {o["crossover"]}°。正文只用国家块 bootstrap 的 95% CI（本行）。'
             f'不引用：(1) 另一脚本（crossover_proxy.log）的 90% CI [{m.group(2)}, {m.group(3)}]，两者来自不同 bootstrap；'
             f'(2) perm_p={o["perm_p"]}（按"对"置换，忽略国家聚类）；'
             f'(3) 两个固定效应斜率不一致：posthoc_summary.json 的 fe_slope={o["fe_slope"]}，crossover_proxy.log 的目标国固定效应差 '
             f'{mfe.group(3) if mfe else "?"}，只报其一并写明脚本。')
    ROWS[-1]['_fe'] = (o['fe_slope'], mfe.group(3) if mfe else '?', o['perm_p'], m.group(2), m.group(3))
    add('S.cheng_pred_slope', '补充', '按 Cheng 数据换算的 AE − S2 斜率预测（Fig. 6b 每 1000 °C·d ΔOA +0.09 × 约 150 °C·d/°）',
        0.09 / 1000 * 150 * 100, 'pp/°', noROPT_na='不适用：文献换算值，非本研究数据', grade='不适用（文献换算）', prereg='否',
        reuse='—', learner='—（Cheng 2026）', metric='OA（Cheng）', source='REPOSITION 附录 A §3（换算说明）',
        func='0.09/1000 × 150 × 100', mode='文献换算', doc={'v': '+1.35'}, doc_src='REPOSITION §3；附录 A',
        note='换算系数（约 150 °C·d/°）取自 Cheng 的"1400–1500 °C·d ≈ 9–10°"；附录 A 另写"换算系数偏大 3 倍时约 +0.45"。')
    # 讨论中 Cheng 对照：pairwise_v2 国内测试
    pv = rcsv(f'{RES}/pairwise_v2.csv')
    dom = pv[pv.source == pv.target].pivot(index='target', columns='feature', values='ba') * 100
    dd = dom.ae - dom.s2
    m9, c9 = noropt_str(dd, 2)
    add('S.cheng.pairwise_domestic', '补充', 'pairwise_v2 国内测试（源国 = 目标国，分层 20%）的 BA：AE vs S2，及 AE − S2', None, 'pp', s=TS(dd),
        noROPT=m9, noROPT_ci=c9 + f'；水平 AE {dom.ae.drop(ROPT).mean():.2f} vs S2 {dom.s2.drop(ROPT).mean():.2f}',
        grade='C（事后引用；国内部分无缓冲、未去重）', prereg='否（检验 1 的副产品）',
        reuse='检验 1 的国内部分：源国分层抽 20%，无缓冲、未去重（TESTSET_AUDIT §2），未用于调参', learner='LightGBM（pairwise_v2）',
        metric='全类 BA（目标国出现的类）', source=f'{RES}/pairwise_v2.csv', func='numbers_table.supp（source==target 行）',
        doc={'v_ae': '55.54', 'v_s2': '53.35', 'v': '+2.19'}, doc_src='REPOSITION §1 表；附录 A 第 331 行',
        per_country=dd.to_dict(),
        note=f'水平：AE {dom.ae.mean():.2f}，S2 {dom.s2.mean():.2f}（11 国均值）；AE 更低的国家 {int((dd < 0).sum())}/11。'
             '与 Cheng"本地 AEF 低 0.02"方向相反，文档归因于基线不同。')
    ROWS[-1]['_lv'] = {'v_ae': dom.ae.mean(), 'v_s2': dom.s2.mean()}
    # PT 的链式 OA 域残差 AE − S2 随训练设置翻号（检验 3 的早停 / 固定 300 棵）
    pt = {}
    for key, lab in (('es', '早停（check3_arms）'), ('cap_es', '早停（capacity 文件）'), ('cap_c300', '固定 300 棵')):
        pt[key] = float(res[(key, 'oa', 'ae', 'dfp')]['resid']['PT'] - res[(key, 'oa', 's2', 'dfp')]['resid']['PT'])
    add('S.cheng.PT_resid_flip', '补充', 'PT：检验 3 链式 OA 域残差的 AE − S2，早停 vs 固定 300 棵',
        f'早停 {pt["es"]:+.2f}（capacity 文件早停 {pt["cap_es"]:+.2f}）；固定 300 棵 {pt["cap_c300"]:+.2f}', 'pp', n=1,
        noROPT_na='不适用：单国（PT）行', grade='C（n=1，事后）', prereg='否', reuse=RU_C3_POST, metric='OA',
        source=f'{RES}/check3_arms.csv；{RES}/check3_capacity_arms.csv', func='numbers_table.h1_check3 的 dfp（resid 列，PT 行）',
        doc={'v_es': '+4.9', 'v_c300': '-11.9'}, doc_src='REPOSITION §3（"PT 为 +4.9 或 −11.9"）、第 940 行',
        note='说明国家级表征层 AE−S2 差值随训练设置翻号，国家级数据无法检验热量梯度。')
    ROWS[-1]['_pt'] = {'v_es': pt['es'], 'v_c300': pt['cap_c300']}
    # D2：rev4 与 Cheng 可比的口径（LightGBM，k=0），由 D2 行文件重算国家层 t 汇总
    add_d2_rows()
    # P3 协议效应：v2 从原论文结果文件重算（修复前数据）
    idj = json.load(open(_chk(f'{OLDEXP}/exp1_id_baseline/percountry_tier_b_nocoords_2021.json')))
    lvl, gap = {}, {}
    for f in ('ae', 's2', 'fusion'):
        pc = pd.DataFrame({r: {c: v['balanced_accuracy'] for c, v in idj[f'{f}|{r}']['per_country'].items()} for r in range(5)}).mean(axis=1)
        lo = rcsv(f'{OLDEXP}/exp2_loco/loco_{f}_nocoords_tier_b.csv').set_index('test_country').balanced_accuracy * 100
        lvl[f] = (pc, lo)
        gap[f] = (pc - lo).dropna()
    add('S.P3_protocol', '补充', 'P3 协议效应：逐国口径 ID（pooled CV 逐国 BA）→ LOCO 落差（AE / S2 / Fusion）',
        ' / '.join(f'{gap[f].mean():.2f}' for f in ('ae', 's2', 'fusion')), 'pp', n=11,
        noROPT=' / '.join(f'{gap[f].drop(ROPT).mean():.2f}' for f in ('ae', 's2', 'fusion')), noROPT_ci='9 国均值（v2 补算）',
        grade='B（修复前数据）', prereg='否', reuse='原论文（修复前数据；LOCO 为国别留出）', learner='LightGBM', metric='BA（逐国口径）',
        source=f'{OLDEXP}/exp1_id_baseline/percountry_tier_b_nocoords_2021.json；{OLDEXP}/exp2_loco/loco_{{ae,s2,fusion}}_nocoords_tier_b.csv',
        func='numbers_table.supp（ID 5 次重复逐国 BA 平均 − LOCO 逐国 BA）', mode='重算',
        doc={'v_ae': '11.99', 'v_s2': '11.74', 'v_fu': '13.63'},
        doc_src='PROGRESS_REVIEW_2026-09-23 §3.2 第 183 行（该行注明"未写入"手稿）；CONCLUSIONS P3',
        note='v1 为"读取（文档）"；v2 在项目 results/experiments 中找到结果文件并重算。CONCLUSIONS 的"+15.4"（窄训练减宽训练）仍没有结果文件，本表不收。')
    ROWS[-1]['_p3'] = {f'v_{k_[:2]}': gap[f].mean() for f, k_ in (('ae', 'ae'), ('s2', 's2'), ('fusion', 'fu'))}
    la, ls = lvl['ae'][1], lvl['s2'][1]
    dl = (la - ls).dropna()
    w = stats.wilcoxon(dl).pvalue
    add('S.P3_loco', '补充', '原论文 LOCO（逐国 BA）：AE vs S2 与 AE − S2', None, 'pp', s=TS(dl),
        p_note=f'Wilcoxon p={w:.3f}（原报告所用）；p 列为 t 检验', noROPT=float(dl.drop(ROPT).mean()),
        noROPT_ci=noropt_str(dl, 2)[1], grade='B（修复前数据）', prereg='否', reuse='原论文（国别留出）', learner='LightGBM',
        metric='BA（逐国口径）', source=f'{OLDEXP}/exp2_loco/loco_{{ae,s2}}_nocoords_tier_b.csv', func='numbers_table.supp', mode='重算',
        doc={'v_ae': '42.71', 'v_s2': '43.17', 'v': '-0.46', 'p_wilcoxon': '0.413'},
        doc_src='REPOSITION §1 表（"42.71 对 43.17，p=0.41"）；CONCLUSIONS P2/9', per_country=dl.to_dict(),
        note=f'水平 AE {la.mean():.2f}，S2 {ls.mean():.2f}；中位数 {dl.median():+.2f}，{int((dl > 0).sum())}/11 为正。'
             '修复前数据（当时 9.7% 地块被误归入 other_crops）。')
    ROWS[-1]['_loco'] = dict(v_ae=la.mean(), v_s2=ls.mean(), v=dl.mean(), p_wilcoxon=w)


def add_d2_rows():
    """rev4 D2（LightGBM，k=0）：由 cheng_metrics_rows.csv 重算国家层 AE − S2。"""
    src = f'{D2DIR}/cheng_metrics_rows.csv'
    summ = rtxt(f'{D2DIR}/cheng_metrics_summary.txt')
    assert '7e90ccd5b1c3828f' in summ
    Dr = rcsv(src)
    z = Dr[(Dr.learner == 'lgbm') & (Dr.k == 0)]
    assert z.method.unique().tolist() == ['foreign_only'] and len(z) == 44
    P = z.pivot_table(index=['target', 'split'], columns='feat', values=['ba_all', 'oa4', 'ba4'])
    NOTE = ('k=0 的指标用 P0test（float16，与主文 foreign_only 相同）；D1 用温度校准的 P_T，两种底座在 ba_ge30 上单元级最大差 0.15 pp、'
            'OA 0.015 pp（decision_curve_summary.txt）。Cheng 的 −0.04 OA 是完整 SITS 基线；类均衡指标（全类 BA、4 类 BA）的 11 国与去 RO/PT 版 95% CI '
            '都同时含 0 与 +4（Cheng 只用 S2 的对照），所以只能说方向与 Cheng 只用 S2 的对照一致；4 类 OA 以 grassland 为主，不宜与 Cheng 的 OA 直接比；'
            'RF 的 k=0 也只能消除学习器差异，本数据检验不了"比完整 SITS 低 0.04"。')
    docs = {'ba_all': {'v': '+1.98', 'lo': '-0.69', 'hi': '+4.65', 'noROPT': '+2.66'},
            'oa4': {'v': '+1.03', 'lo': '-0.92', 'hi': '+2.98', 'noROPT': '+0.62'},
            'ba4': {'v': '+1.36', 'lo': '-3.28', 'hi': '+5.99', 'noROPT': '+2.11'}}
    labs = {'ba_all': '(a) 全类平衡精度（测试集出现的所有类）', 'oa4': '(b) 4 类子任务 OA', 'ba4': '(b) 4 类子任务平衡精度'}
    for met in ('ba_all', 'ba4'):  # 注中"CI 同时含 0 与 +4"的说法由此核对
        lv_ = P[met].groupby('target').mean()
        v = (lv_['ae'] - lv_['s2'])
        for ss_ in (TS(v), TS(v.drop(ROPT))):
            assert ss_['lo95'] < 0 < 4 < ss_['hi95'], (met, ss_)
    for met in ('ba_all', 'oa4', 'ba4'):
        lv_ = P[met].groupby('target').mean()
        v = (lv_['ae'] - lv_['s2'])
        m9, c9 = noropt_str(v, 2)
        add(f'D2.lgbm.{met}.k0', '补充/D2', f'rev4 D2：k=0 下 AE − S2 的{labs[met]}（LightGBM）', None, 'pp', s=TS(v), noROPT=m9, noROPT_ci=c9,
            grade='描述（补充材料；C：测试集复用）', prereg='是（rev4 D2，7e90ccd5）', reuse=RU_G_POST, metric=met, source=src,
            func='numbers_table.add_d2_rows（cheng_metrics.py 行 → 国家 → t）', mode='重算（D2 输出）', doc=docs[met],
            doc_src='cheng_metrics_summary.txt', per_country=v.to_dict(),
            note=f'水平：AE {lv_["ae"].mean():.2f}，S2 {lv_["s2"].mean():.2f}。'
                 + ('RO 的 4 类子任务只剩 grassland，LT、LV 缺 potatoes；4 类 OA 以 grassland 为主。' if met != 'ba_all' else
                    '含 1–29 个测试地块的格子共 30 个，噪声大，应与 ba_ge30 一起报。') + NOTE)
        if met == 'ba4':
            keep = ~v.index.isin(['LT', 'LV', 'RO'])
            s8 = TS(v[keep])
            add('D2.lgbm.ba4.k0.complete8', '补充/D2', '方案外，描述：rev4 D2 的 4 类平衡精度 AE − S2，只保留 4 类齐全的 8 国（去掉 LT、LV、RO）',
                None, 'pp', s=s8, noROPT=float(v[keep & ~v.index.isin(ROPT)].mean()),
                noROPT_ci=f'再去 PT 后 n={int((keep & ~v.index.isin(ROPT)).sum())}（点估计）', grade='方案外，描述',
                prereg='否（方案外）', reuse=RU_G_POST, metric='ba4', source=src, func='numbers_table.add_d2_rows', mode='重算（D2 输出）',
                doc={'v': '+0.11', 'lo': '-6.18', 'hi': '+6.40'}, doc_src='cheng_metrics_summary.txt（方案外行）',
                per_country=v[keep].to_dict(),
                note='DE +12.43 与 PT −12.87 相互抵消；11 国的 +1.36 几乎全部来自 LT、LV、RO。此时 Cheng 的 −4 落在 CI 内，无法区分。')
    for lrn in ('LR', 'MLP', 'RF'):
        add(f'D2.cheng_metrics.{lrn}', '补充/D2', f'rev4 D2：k=0 下 AE − S2 的全类 BA 与 4 类子任务 OA/BA（{lrn}）', '待 rev2 + D2', 'pp',
            grade='描述（补充材料）', prereg='是（rev4 D2）', reuse=RU_G_POST, learner=lrn,
            source='results/guidance_G_ml/（未读取）', func='cheng_metrics.py --src … --ml-unblinded（rev2 完成后）', mode='占位',
            noROPT_na='待 rev2')


# ================================================================== 核对与输出
def _num(sv):
    sv = str(sv).replace('−', '-').replace('+', '').strip()
    try:
        return float(sv)
    except ValueError:
        return None


def _dec(sv):
    sv = str(sv).replace('−', '-')
    return len(sv.split('.')[1]) if '.' in sv else 0


def compare_all():
    """逐行把重算值与文档值比较。|差| > 0.01 的都列出；再区分"文档舍入可解释"与"真实差异"。"""
    out = []
    for r in ROWS:
        if not r['doc']:
            continue
        doc = json.loads(r['doc'])
        cand = {}
        v = r['value']
        if isinstance(v, (int, float, np.floating)):
            cand['v'] = float(v)
        if r['ci95_lo'] is not None:
            cand['lo'], cand['hi'] = r['ci95_lo'], r['ci95_hi']
        if r['ci90_lo'] is not None:
            cand['lo90'], cand['hi90'] = r['ci90_lo'], r['ci90_hi']
        if isinstance(r['noROPT'], (int, float, np.floating)) and r['noROPT'] is not None:
            cand['noROPT'] = float(r['noROPT'])
        if r['p'] is not None and not isinstance(r['p'], str):
            cand['p'] = float(r['p'])
        # 特殊键
        for key, val in r.items():
            if key == '_tost':
                cand['p_tost'] = val
            if key == '_wilcoxon':
                cand['p_wilcoxon'] = val
            if key == '_holm':
                cand['holm'] = val
                cand['holm16'] = val
            if key == '_loo':
                cand['v_min'], cand['v_max'] = val
            if key == '_knn':
                cand['v_ae'], cand['v_s2'] = val
            if key == '_auroc':
                cand['v_ae'], cand['v_s2'] = val
            if key == '_acc':
                cand['v_ae'], cand['v_s2'] = val
            if key == '_rng':
                cand['v_min'], cand['v_max'], cand['p_min'] = val
            if key == '_oa':
                cand['oa'] = val
            if key == '_mean400':
                cand['mean'] = val
            if key == '_cm':
                cand['country_mean'] = val
            if key == '_oa_sp':
                cand['oa_spearman'] = val
            if key in ('_lq', '_lam', '_pt', '_p3', '_loco', '_lv'):
                cand.update(val)
            if key == '_byk':
                cand['v_f1_k50'], cand['v_ba_k50'], cand['v_ba_k200'] = val[('f1_ge30', 50)], val[('ba_ge30', 50)], val[('ba_ge30', 200)]
        if r['id'] in ('H1.tgtonly.s1.ba.mlp', 'H1.tgtonly.s1.ba.lr'):
            m = re.findall(r'([+-][\d.]+)', str(r['value']))
            cand['v_ae'], cand['v_s2'] = float(m[0]), float(m[1])
        if r['id'].startswith('H4.kcenter_oa'):
            cand['RO'], cand['PT'] = r['_ROPT']
        if r['id'] == 'H4.predstrat.pooled':
            cand['holm_REPOSITION'] = r['_holm']
        if r['id'] == 'H5.F3b_loco_alpha':
            m = re.findall(r'([+-][\d.]+)', str(r['value']))
            cand['v_min'], cand['v_max'] = float(m[0]), float(m[1])
        if r['id'] == 'H5.F3b_lambda_sens':
            m = dict(re.findall(r'λ=([\d.]+): ([+-][\d.]+)', str(r['value'])))
            cand['v_0.02'], cand['v_0.5'] = float(m.get('0.02', 'nan')), float(m.get('0.5', 'nan'))
        if r['id'] == 'H2.check1.drho_aemmd_s2mmd':
            pass
        for k, dv in doc.items():
            if k not in cand:
                continue
            if dv in ('inf',):
                ok = np.isinf(cand[k])
                out.append(dict(id=r['id'], field=k, doc=dv, recomputed=cand[k], diff=0.0 if ok else np.nan,
                                cls='一致（≤0.01）' if ok else '真实差异', src=r['doc_src']))
                continue
            if '–' in str(dv):
                lo_, hi_ = [_num(x) for x in str(dv).split('–')]
                x = cand[k]
                ok = lo_ - 0.5 <= x <= hi_ + 0.5
                out.append(dict(id=r['id'], field=k, doc=dv, recomputed=x, diff=0.0 if ok else min(abs(x - lo_), abs(x - hi_)),
                                cls='在文档区间内' if ok else '真实差异', src=r['doc_src']))
                continue
            approx = str(dv).startswith('约')
            dn = _num(str(dv).lstrip('约'))
            if dn is None:
                continue
            x = float(cand[k])
            diff = x - dn
            tol_round = 0.5 * 10 ** (-_dec(dv)) + 1e-9
            if abs(diff) <= 0.01:
                cls = '一致（≤0.01）'
            elif approx and abs(diff) <= 2.5:
                cls = '文档写"约"，可解释'
            elif abs(diff) <= tol_round:
                cls = '文档舍入可解释'
            else:
                cls = '真实差异'
            out.append(dict(id=r['id'], field=k, doc=dv, recomputed=x, diff=diff, cls=cls, src=r['doc_src']))
    return pd.DataFrame(out)


def fmtnum(x, unit):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ''
    if isinstance(x, str):
        return x
    if unit in ('%',):
        return f'{x:.1f}'
    if unit in ('ρ', 'Δρ', 'r', 'AUROC'):
        return f'{x:+.3f}'
    if unit in ('F', 'L1', '个标签'):
        return f'{x:.2f}' if np.isfinite(x) else ('+∞' if x > 0 else '−∞')
    return f'{x:+.3f}'


DOCS = {'REPOSITION': f'{LE}/REPOSITION_AFTER_CHENG2026_2026-09-27.md', 'CONCLUSIONS': f'{LE}/CONCLUSIONS_2026-09-26.md'}


def doc_lines(doc, *pats):
    """返回文档中同时含有全部 pats 的行号（1 起）。只读项目文档。"""
    with open(_chk(DOCS[doc]), encoding='utf-8') as f:
        return [i + 1 for i, ln in enumerate(f) if all(p_ in ln for p_ in pats)]


def _isnum(x):
    return isinstance(x, (int, float, np.floating)) and not (isinstance(x, float) and np.isnan(x))


def write_outputs(C):
    df = pd.DataFrame(ROWS)
    df = df[[c for c in df.columns if not c.startswith('_')]]
    df.to_csv(os.path.join(HERE, 'NUMBERS.csv'), index=False, encoding='utf-8-sig')
    # ---------- discrepancies.txt
    real = C[C.cls == '真实差异']
    rnd = C[C.cls.isin(['文档舍入可解释', '文档写"约"，可解释'])]
    vc = C.cls.value_counts()
    assert vc.sum() == len(C)
    lines = ['差异清单（numbers_table.py v2 生成；重算值 − 文档值）', '',
             f'核对字段总数 {len(C)}：' + '；'.join(f'{k} {int(v)}' for k, v in vc.items()) + f'（合计 {int(vc.sum())}）。', '',
             '一、真实差异（|差| > 0.01 且不能由文档舍入解释）']
    for _, r in real.iterrows():
        lines.append(f'  - {r['id']} [{r['field']}]：文档 {r['doc']}（{r['src']}）；重算 {r['recomputed']:.4f}；差 {r['diff']:+.4f}')
    lines += ['', '二、|差| > 0.01 但可由文档舍入解释（文档只给到较少位数）']
    for _, r in rnd.iterrows():
        lines.append(f'  - {r['id']} [{r['field']}]：文档 {r['doc']}；重算 {r['recomputed']:.4f}；差 {r['diff']:+.4f}')
    lines += ['', '三、口径或引用问题（数值本身重算一致，但文档写法需要改；行号指项目目录 label_expansion/ 下的文档，本表未修改任何文档）']
    lines += NOTES
    lines += ['', '四、本版（v2）相对 v1 的更正（按 d3_review 复核意见；v1 备份为 *_v1.*）']
    lines += CHANGES
    open(os.path.join(HERE, 'discrepancies.txt'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    # ---------- NUMBERS.md
    mc = df['mode'].value_counts()
    n_na = int(((df['mode'] != '占位') & df['noROPT'].isna()).sum())
    md = ['# 单一来源数字表（rev4 D3，v2）', '',
          '由 `numbers_table.py` 从结果文件生成（2026-09-27）。完整字段（逐国值、文档值、生成函数全名）见 `NUMBERS.csv`；差异见 `discrepancies.txt`；'
          'v1 备份为 `NUMBERS_v1.*`。',
          '',
          f'**行数** {len(df)}：' + '；'.join(f'{k} {int(v)}' for k, v in mc.items()) + f'。核对文档字段 {len(C)} 处（见 discrepancies.txt 抬头）。', '',
          '**读法**',
          '- 置信区间统一为 95%（国家层 t 区间，df=10；去 RO/PT 为 9 国，df=8）。90% CI 只在等价（TOST）判定需要时列出，并注明用途。',
          '- "方式"：重算 = 由本脚本从逐行结果重新计算；重算（D1/D2 输出）= 由 rev4 D1/D2 脚本的输出文件重算国家层汇总；'
          '读取 = 只能从报告或日志读取（出处写明）；文献换算 = 由 Cheng 2026 的数字换算，不是本研究数据；占位 = 待 rev2。',
          '- "去 RO/PT"列：给出 9 国值；不能给时写"不适用"或"未算"并说明原因（v2 不再留空）。'
          f'非占位行中去 RO/PT 为"不适用/未算"的共 {n_na} 行。',
          '- 证据等级沿用 CONCLUSIONS 的 A/B/C/D（A 只给预注册主判定；C 为事后分析或用了已看过的测试集）。'
          '这里标的是**单个数字**的等级；结论的等级可以更高（例如 H1.tgtonly 的单个数字为 C，但支持 H1 的 B 级方向结论）。',
          '- "测试集复用"：写明该测试集此前是否被看过或用于调参。检验 3 的随机测试集在 PT、SE、SK、LV 上与检验 2 逐位相同'
          '（TESTSET_AUDIT §3–4）；只用候选池标签的量写"不用测试标签"。',
          '- rev2 学习器（LR、MLP、RF）的所有量都是占位"待 rev2"：本表没有读取 `results/guidance_G_ml/` 的任何文件。',
          '- 表中带"MODEL_DIAGNOSIS 试点"字样的 LR/MLP 行（H1.tgtonly.s1.ba.mlp/lr、H6.M9.*、H6.M9b.*）是已公开的模型诊断试点，'
          '**不属于 rev2 估计量**（5 国或设置不同）；rev2 出结果后，这些行不要与 rev2 的数字并排放。',
          '- rev4 D1 行的主口径是"两特征国家值平均"（作者选定的汇总口径，rev4 未规定跨特征合并），逐特征值在注中。'
          'D1 与 D2 的 k=0 底座不同：D1 用温度校准的 P_T，D2 用 float16 的 P0test（与主文 foreign_only 相同），ba_ge30 单元级最大差 0.15 pp。',
          '- 标"方案外，描述"的行不在 rev4 预先指定之内，只作描述。',
          '- 百分比行（份额、恢复率、水平）的单位为 %；其余为 pp，除非另注。', '',
          '**术语（按 rev4 D1，写入论文定义表时照此限定）**',
          '- 决策层可挽回部分：在冻结的外国 LightGBM 上，用 k 个目标标签只重拟温度与类别偏置所得的增益；λ=0.1、类均衡加权 NLL、'
          '标签从约 3 万候选池简单随机抽取（D1.* 行）。它与 H1附 的缓冲格网上界（oracle_G_buffered，无 λ、重加权）、'
          'H5.balhead_full（calib_diag，λ=1e-4）都不是同一个量。',
          '- 表征层剩余：只用目标池全部标签训练的模型与决策层上界之间的差距（H1附.repr_left.*）。',
          '- 检验 3 链式分解中的"域残差"包含决策层可挽回部分与表征层剩余两部分。',
          '- H5 的 +0.94 是 F3b 混合头（重训 ⊕ 类均衡头，α=0.4）相对同标签重训，不是"只用决策头"，也不是 D1 的 Δhead。', '']
    order = ['H1', 'H1附', 'H2', 'H3', 'H4', 'H5', 'H5/D1', 'H6', '补充', '补充/D2']
    titles = {'H1': 'H1 缺口解剖：OA 与 BA 的迁移缺口来源不同', 'H1附': 'H1 附：决策层上界与表征层剩余（上界，C 级）',
              'H2': 'H2 AE 不是更好的选样指导，也不是更好的选源依据', 'H3': 'H3 合并重训时随机目标标签的边际价值',
              'H4': 'H4 标哪些地块', 'H5': 'H5 标签怎么用', 'H5/D1': 'rev4 D1 决策层标签效率曲线（C 级，预先指定；λ=0.1）',
              'H6': 'H6 AE 作为特征', '补充': '补充材料（含 REPOSITION §1、§3 讨论 Cheng 时引用的数字）',
              '补充/D2': 'rev4 D2 与 Cheng 可比的口径（补充材料，描述）'}
    assert set(df.group) <= set(order), set(df.group) - set(order)
    for g in order:
        sub = df[df.group == g]
        if not len(sub):
            continue
        md += [f'## {titles[g]}', '',
               '| ID | 描述 | 数值 | 95% CI | 90% CI（用途） | n（>0） | p | 去 RO/PT | 等级 | 预注册 | 测试集复用 | 学习器 | 方式 | 出处 |',
               '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for _, r in sub.iterrows():
            u = r['unit'] if isinstance(r['unit'], str) else ''
            val = fmtnum(r['value'], u) + ('' if (r['mode'] == '占位' or not u or isinstance(r['value'], str)) else f' {u}')
            if isinstance(r['value'], str) and u and r['mode'] != '占位':
                val += f'（{u}）'
            ci = f'[{fmtnum(r['ci95_lo'], u)}, {fmtnum(r['ci95_hi'], u)}]' if pd.notna(r['ci95_lo']) and r['ci95_lo'] is not None else ''
            ci90 = f'[{fmtnum(r['ci90_lo'], u)}, {fmtnum(r['ci90_hi'], u)}]（{r['ci90_use']}）' if pd.notna(r['ci90_lo']) and r['ci90_lo'] is not None else ''
            nn = '' if r['n'] is None or (isinstance(r['n'], float) and np.isnan(r['n'])) else f'{int(r['n'])}'
            if r['n_pos'] is not None and not (isinstance(r['n_pos'], float) and np.isnan(r['n_pos'])):
                nn += f'（{int(r['n_pos'])}）'
            pv = '' if r['p'] is None or (isinstance(r['p'], float) and np.isnan(r['p'])) else f'{r['p']:.3g}'
            if isinstance(r['p_note'], str) and r['p_note']:
                pv = (pv + '；' if pv else '') + r['p_note']
            nr = fmtnum(r['noROPT'], u) if r['noROPT'] is not None and not (isinstance(r['noROPT'], float) and np.isnan(r['noROPT'])) else ''
            if isinstance(r['noROPT_ci'], str) and r['noROPT_ci']:
                nr += f' {r['noROPT_ci']}'
            if not nr and isinstance(r['noROPT_na'], str) and r['noROPT_na']:
                nr = r['noROPT_na']
            desc = r['desc'] + (f'。注：{r['note']}' if isinstance(r['note'], str) and r['note'] else '')
            srcs = str(r['source']).replace(LOCAL + '/', '~/AlphaCrops_local/').replace(PROJ + '/', 'Alpha_Crops/')
            cells = [r['id'], desc, val, ci, ci90, nn, pv, nr, r['grade'], r['prereg'], r['test_reuse'], r['learner'], r['mode'], f'`{srcs}`']
            md.append('| ' + ' | '.join(str(c).replace('|', '／').replace('\n', ' ') for c in cells) + ' |')
        md.append('')
    key_ids = ['H2.theta_E1', 'H2.theta_E1.oa', 'H1.oa_comp.es.ae', 'H1.oa_comp.es.s2', 'H1.resid_ae_s2.ba_ge30.es',
               'H1附.dec_upper.dec_ba.ae', 'H3.G_lgb.k200.avg.ba_ge30', 'H4.predstrat.pooled', 'H4.kcenter_oa',
               'H5.F3b_a0.4.k200', 'D1.ba_ge30.dhead.k200', 'D1.ba_ge30.diff.k200', 'D2.lgbm.ba4.k0',
               'H6.dfeat_lgb.ba_ge30.k0', 'H6.dfeat_lgb.ba_ge30.k400', 'H6.dfeat_change_lgb.ba_ge30']
    pcs = {}
    for i in key_ids:
        r = df[df.id == i]
        assert len(r) == 1, i
        if isinstance(r.iloc[0]['per_country'], str) and r.iloc[0]['per_country']:
            pcs[i] = json.loads(r.iloc[0]['per_country'])
    if pcs:
        md += ['## 附：关键行的逐国值', '', '单位与对应行相同（份额行为 %，其余为 pp）。', '',
               '| 国家 | ' + ' | '.join(pcs) + ' |', '|---|' + '---|' * len(pcs)]
        for c in COUNTRIES:
            md.append(f'| {c} | ' + ' | '.join(f'{pcs[i].get(c, float("nan")):+.2f}' for i in pcs) + ' |')
        md.append('')
    md += ['## 未收录的内容', '',
           '- rev4 D2 的方案外内容中，只收了"去掉 4 类不全的国家"一行（D2.lgbm.ba4.k0.complete8，Cheng 对照段要用）。'
           '逐类召回、SRS k=200 一节不收（见 `cheng_comparable/cheng_metrics_summary.txt`，方案外，描述）；'
           '"10 类 OA"与"ba_ge30"两条参照行与 H6.dfeat_lgb.*.k0 相同，不重复列出。',
           '- rev4 D1 的附加部分（全池 λ=1e-4 敏感性、标签配对版头 − 重训、子样本 SD、拟合参数）不收，见 `decision_curve/decision_curve_summary.txt`；'
           '复核中的 λ 敏感性（`d1_review/lam_sens_rows.csv`）为复核事后计算，C 级，不用于选 λ。',
           '- CONCLUSIONS P3 的"+15.4"（窄训练减宽训练的跨境代价）没有结果文件，不收。', '',
           '## 生成函数', '',
           '- G 行：`analyze_G.load_rows`（只读 `results/guidance_G/rows_t*_*.csv`）→ `deltas` / `vs_srs` → `country_values` → `t_summary`；'
           'SRS 学习曲线、Δ_feat 与 G(L,k) 为单元配对后按国家平均（`numbers_table.h3_label_value`、`h6_feature`）。',
           '- 检验 3：`check3_arms.csv`、`check3_capacity_arms.csv`，合并份额的 bootstrap 逐位复现 `analyze_check3.pooled` 与 `analyze_check3_capacity` 的随机数顺序。',
           '- 检验 1：`check1_per_target.csv`，bootstrap 逐位复现 `analyze_check1.boot_ci` 的随机数顺序。',
           '- 决策层：`oracle_G_buffered_rows.csv`、`oracle_es_vs_c300_by_country.csv`、`frontier2_rows.csv`、`frontier3_rows.csv`、`calib_rows.csv` 的逐单元行（本表不重新拟合任何决策头）。',
           '- rev4 D1：`storyline/decision_curve/decision_curve_country.csv` 的国家值（D1 脚本 `decision_curve.py` 产出）→ `t_summary`；'
           'rev4 D2：`storyline/cheng_comparable/cheng_metrics_rows.csv` 的逐行指标 → 国家 → `t_summary`。',
           '- 策略基准：G 行 + `strategy/mechanism/L.pkl`（剂量斜率；合并斜率的 CI 为本表新算的国家 cluster bootstrap）+ `pivot_eval/codesign/diag3_sel_rows.csv`（排序相关）。',
           '- 模型诊断试点：`model_diag/verify_fit/alt_long.csv`（5 国 LR/MLP，与 `analyze_alt.py` 同式）与 `model_diag/altmodels/part1_rows.csv`。',
           '- 原论文（修复前数据）：`Alpha_Crops/results/experiments/exp1_id_baseline/percountry_tier_b_nocoords_2021.json`、'
           '`exp2_loco/loco_*_nocoords_tier_b.csv`（只读）。', '']
    open(os.path.join(HERE, 'NUMBERS.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    return df


NOTES = []
CHANGES = []


def _row(id_):
    for r in ROWS:
        if r['id'] == id_:
            return r
    raise KeyError(id_)


def _ln(lst):
    return '、'.join(f'第 {i} 行' for i in lst) if lst else '（未找到）'


def build_notes():
    """口径与引用问题（数字全部取自本表已算出的行；文档行号由 doc_lines 现查）。"""
    N = NOTES
    pa, ps = _row('H4.predstrat.AE'), _row('H4.predstrat.S2')
    pp_ = _row('H2.pred_pathway')
    rep44 = doc_lines('REPOSITION', 'p=0.44', 'predstrat')
    N.append(f'  1. [H4] REPOSITION 把"predstrat … Holm 校正后 p=0.44"写了 {len(rep44)} 处：{_ln(rep44)}（§4 H4 表与附录 D）。重算：predstrat 相对 SRS '
             f'在 16 项族（8 方法 × 2 指导）中的 Holm p 为 {pa["_holm"]:.3f}（AE 指导）/ {ps["_holm"]:.3f}（S2 指导），与 STRATEGY_FINDINGS ②"最小 p=0.198"一致；'
             f'0.44 实为预测通路 AE − S2（+{pp_["value"]:.3f}）在 14 项次要结果族中的 Holm p（{pp_["_holm"]:.3f}；CONCLUSIONS 第 103、585 行与 G_RESULTS 第 103 行'
             '把 0.44 用在预测通路上，是对的）。上列 REPOSITION 各处都应改为 0.198（或写"AE 指导 0.198、S2 指导 0.822"）。')
    lv, lm = _row('H3.labelvalue_vs_distance'), _row('H3.labelvalue_vs_min_dlat')
    N.append(f'  2. [H3] "标签价值与距离无关（ρ 在 −0.31 至 +0.33，p≥0.33）"只覆盖"平均纬度差"和"平均公里"两种距离（重算范围 {lv["value"]}，'
             f'最小 p={lv["p"]:.3f}）。同一脚本还算了"到外国池的最小纬度差"，结果 {lm["value"]}（{lm["p_note"]}），在原日志中未打印。'
             '这是分叉路径：写"与距离无关"之前需要预先说明为什么用平均距离，或把最小纬度差一并报告。')
    cr = _row('H6.shift_criterion.lgb')
    same, inside, lo, hi = cr['_crit']
    N.append(f'  3. [H6] rev3 B1"整体平移"判据要求变化量 95% CI 落在 (−1, +1) pp 内。LightGBM 的变化量 95% CI 为 [{lo:+.3f}, {hi:+.3f}]，'
             f'{"落在" if inside else "上界超出 +1，不落在"}区间内，所以 LightGBM {"满足" if (same and inside) else "本身不满足"}该判据。'
             '若其余学习器表现相近，"≥3/4 学习器"的条件更难满足；H6 的"整体平移"措辞需等 rev2 判定。')
    f3 = _row('H5.F3b_a0.4.k200')
    bh = _row('H5.balhead_only.vs_retrain.k200')
    c_mis = doc_lines('CONCLUSIONS', '只用 k 个目标标签拟合一个类均衡的决策层')
    r94 = doc_lines('REPOSITION', '0.94', '决策头')
    N.append(f'  4. [H5] "类均衡决策头相对重训 +0.94"（重算 {f3["value"]:+.3f} [{f3["ci95_lo"]:+.3f}, {f3["ci95_hi"]:+.3f}]）是"重训 ⊕ 类均衡头"'
             f'的对数线性混合 F3b（α=0.4，frontier2.py：(1−α)·log P_重训 + α·log softmax(头)），不是只用头；α 与 λ 都在同一 G 测试集上选定。'
             f'只用头相对重训为 {bh["value"]:+.2f} pp BA，OA {bh["_oa"]:+.2f}（calib_rows，k=200）。'
             f'CONCLUSIONS {_ln(c_mis)} 把它写成"冻结外国模型，只用 k 个目标标签拟合一个类均衡的决策层"，应改为"重训与类均衡头的对数线性混合（α=0.4）"；'
             f'REPOSITION 中把 +0.94 称为"决策头"的位置：{_ln(r94)}。'
             'CONCLUSIONS V1 的 λ 敏感性（+2.46 / +0.25）是在 α=0.5 下算的，与 headline 的 α=0.4 不同（见 H5.F3b_lambda_sens）。')
    fr, fa = _row('H4.geom_F.rep0'), _row('H4.geom_F.all')
    N.append(f'  5. [H4] "4 种几何方法间 F 检验 p=0.40"只用重复 0（22 单元）；而"几何法相对 SRS −0.156/−0.102"用全部 44 单元。'
             f'用 44 单元重算：F={fa["value"]:.2f}，p={fa["p"]:.3f}（{fa["p_note"]}）。两处的单元集合应统一或注明。')
    n0, n4 = _row('H6.dfeat_lgb.ba_ge30.k0'), _row('H6.dfeat_lgb.ba_ge30.k400')
    negsets = [set(c for c, v in json.loads(_row(f'H6.dfeat_lgb.ba_ge30.k{k}')['per_country']).items() if v < 0)
               for k in (0, 50, 100, 200, 400)]
    allneg = sorted(set.intersection(*negsets))
    N.append(f'  6. [H6] "PT 在每档 k 下都为负"属实，但不只 PT：k=0 时 Δ_feat（ba_ge30）为负的国家是 {", ".join(n0["_neg"])}；'
             f'k=400 时是 {", ".join(n4["_neg"])}；五档 k 全为负的国家是 {", ".join(allneg)}。正文若只点 PT 会遗漏其余国家。')
    ss = _row('S.single_source_slope')
    fe1, fe2, perm, l90, h90 = ss['_fe']
    N.append(f'  7. [补充] 单源斜率（{ss["value"]:+.3f} pp/°，国家块 bootstrap 95% CI [{ss["ci95_lo"]:+.3f}, {ss["ci95_hi"]:+.3f}]，'
             f'overlap_audit/posthoc_summary.json）：正文只用国家块推断。'
             f'(a) 另一脚本（rse_reviewer/crossover_proxy.log）报的 90% CI [{l90}, {h90}] 来自不同 bootstrap，不能当作同一分布的另一分位，不引用；'
             f'(b) posthoc_summary.json 的 perm_p={perm} 是按"对"置换得到的，忽略了国家聚类，不引用；'
             f'(c) 两个脚本的固定效应斜率不同（posthoc_summary.json fe_slope={fe1}；crossover_proxy.log 目标国固定效应下的差 {fe2}），'
             '只报其中一种并写明脚本。')
    m9 = [r for r in ROWS if r['id'].startswith('H6.M9.')]
    m9b = _row('H6.M9b.lr_k0')
    N.append('  8. [H6/M9] M9"LR（k=0）C=1 时 −1.48"来自模型诊断 5 国、划分 1 的试点（verify_fit/alt_long.csv，v2 已重算一致）；'
             f'同为 LR 的 11 国试点（altmodels/part1_rows.csv，划分 0 重复 0）给出 {m9b["value"]:+.2f}。'
             'rev2 的 LR 用 11 国、按外国留国家组选 C，口径不同；rev3 A2 披露过的 LR k=0 数字本表不引用（待 analyze_ML 正式运行）。'
             '建议删去"符号不稳"或改写为"依设置而定（n=5 与 n=11 不同）"。'
             + (f'（本表重算：{"；".join(f"{r["id"]} {r["value"]:+.2f}" for r in m9)}）' if m9 else '')
             + '这些都是 MODEL_DIAGNOSIS 试点，不属于 rev2 估计量。')
    t0 = [_row(f'H1.tgtonly.s0.ba_ge30.{f}') for f in ('ae', 's2')]
    t1 = [_row(f'H1.tgtonly.s1.ba.{f}') for f in ('ae', 's2')]
    N.append(f'  9. [H1] "只用目标池 3 万标签比只用外国高 13–14 pp BA，22/22"：划分 0 为 {t0[0]["value"]:+.2f}/{t0[1]["value"]:+.2f}，'
             f'划分 1 为 {t1[0]["value"]:+.2f}/{t1[1]["value"]:+.2f}；"22/22"来自划分 1（划分 0 也全部为正）。MLP 15、LR 19–23 pp 只有 5 国（划分 1），应注明 n=5。'
             '这组单个数字是事后、缓冲格网、复用测试集上的结果（C），支持的是 H1 的 B 级方向结论；REPOSITION H1 表格引用时不要把它们当 A/B 级数字。')
    p3 = _row('S.P3_protocol')
    N.append(f'  10. [补充] P3 协议效应（ID→LOCO {p3["value"]} pp）v2 已从原论文结果文件重算一致（S.P3_protocol）；出处应写 PROGRESS_REVIEW_2026-09-23 §3.2 第 183 行'
             '（该行注明"未写入"手稿）与 CONCLUSIONS P3。CONCLUSIONS P3 的"+15.4"仍没有结果文件，引用前需补出处或降为"原稿数字"。')
    r1 = _row('H1.resid_ae_s2.oa.es')
    N.append(f'  11. [H1] 检验 3 主要问题 2（AE − S2 域残差）的预注册检验是 Wilcoxon（OA p={r1["_wilcoxon"]:.3f}）。本表另给 t 区间以统一为 95% CI，'
             '正文如用 t 区间需注明检验方式改变。')
    ub = _row('H1附.dec_upper.dec_ba.ae')
    ub2 = _row('H1附.dec_upper.dec_ba.s2')
    bf = _row('H5.balhead_full.vs_k0')
    d1f = _row('D1.ba_ge30.dhead.full')
    N.append(f'  12. [H1附/H5/D1 术语] "决策层上界"有三套实现，定义表必须写明用哪一套：'
             f'(a) H1附（oracle_G_buffered：无 λ 正则、自然先验拟合后重加权到均匀），BA 上界 AE {ub["value"]:+.2f} / S2 {ub2["value"]:+.2f}，'
             f'平均 {(ub["value"] + ub2["value"]) / 2:+.2f}；(b) PIVOT 的全池类均衡头（calib_diag.py：类均衡权重，λ=1e-4），BA {bf["value"]:+.2f}；'
             f'(c) rev4 D1 的全池头（同 calib 的 fit_bcts，λ=0.1），BA {d1f["value"]:+.2f}。v1 把 (b) 误写成 λ=0.1。'
             'rev4 D1 的"决策层可挽回部分"是 (c)，D1 的曲线只能与 (c) 对照。')
    N.append('  13. [全表] 有 C 级的正向结果（predstrat、决策头、剂量反应、ORACLE2、H1附、D1）全部来自已复用的 G s0/s1 测试集；本表"测试集复用"列已逐行标出。')
    rc = _row('H5.rank_corr_retrain_vs_bias')['_corr']
    N.append(f'  14. [H5] "重训与偏置头之间的方法排序相关只有 0.12（OA 0.09）"：BA 用的是 Pearson（重算 {rc["ba"]:.2f}，Spearman {rc["ba_sp"]:.2f}），'
             f'OA 用的是 Spearman（重算 {rc["oa_sp"]:.2f}）；OA 的 Pearson 为 {rc["oa"]:.2f}。两个指标换了相关系数，OA 的"0.09"不宜与 BA 的"0.12"并列引用；'
             '建议统一用一种系数并两者都报。')
    s9b = _row('H3.B2_criterion.lgb')
    N.append(f'  15. [H3] rev3 B2 判据（LightGBM 部分）：去 RO/PT 后 G(LightGBM,200)（两种特征平均）的 95% CI 上界为 {s9b["value"]:+.2f} pp（{s9b["note"]}）。'
             'H3 中 G_lgb 各行的"B"要等四学习器判据。')
    shares = {f: _row(f'H1.ba_set.es.{f}') for f in ('ae', 's2')}
    N.append(f'  16. [H1] BA 的"类别集合占 11–15%"几乎全部来自 RO（及 LT、LV）：去掉 RO/PT 后，早停设置下类别集合份额为 '
             f'{shares["ae"]["noROPT"]:.1f}% / {shares["s2"]["noROPT"]:.1f}%（AE / S2）。正文写 11–15% 时应附去 RO/PT 的值。')
    # ---- v2 新增条目
    hk = {k: _row(f'H5.balhead_only.vs_k0.k{k}') for k in (50, 200, 400)}
    dh = {k: _row(f'D1.ba_ge30.dhead.k{k}') for k in (50, 200, 400)}
    ln279 = doc_lines('REPOSITION', '上界（约 3 万标签）与可兑现量')
    ln_mag = doc_lines('REPOSITION', '差一个数量级')
    ln1114 = doc_lines('REPOSITION', 'k≤400 的真实预算下，决策头相对重训只有 +0.94')
    N.append(f'  17. [H1附/H5] "决策层上界与可兑现量差一个数量级"口径不一致（REPOSITION {_ln(ln279)}把"上界与可兑现量（决策头 +0.94）之差"定为主结果；'
             f'{_ln(ln1114)}把 +0.94 当作 k≤400 的可兑现量；{_ln(ln_mag)}写"差一个数量级"）。+0.94 是 F3b 混合头相对同标签重训的增益，'
             f'+{bf["value"]:.2f} 是全池只用头相对 k=0 的增益（λ=1e-4），两者基线和估计量都不同，不能相除。'
             f'同口径（只用头、相对 k=0、同一 λ=0.1）比较：rev4 D1 的 Δhead(200) = {dh[200]["value"]:+.2f} [{dh[200]["ci95_lo"]:+.2f}, {dh[200]["ci95_hi"]:+.2f}]，'
             f'Δhead(全池) = {d1f["value"]:+.2f}，比值 {dh[200]["value"] / d1f["value"]:.2f}；k=50 为 {dh[50]["value"]:+.2f}（比值 {dh[50]["value"] / d1f["value"]:.2f}）。'
             f'calib_rows 的只用头（λ=0.1，G 的 SRS 选样，22 单元）相对 k=0 为 k=50 {hk[50]["value"]:+.2f}、k=200 {hk[200]["value"]:+.2f} '
             f'[{hk[200]["ci95_lo"]:+.2f}, {hk[200]["ci95_hi"]:+.2f}]、k=400 {hk[400]["value"]:+.2f}；复核意见用它除以 +{bf["value"]:.2f} 得"约 64%"，'
             '但分母是 λ=1e-4，仍混用了 λ。结论：远不到一个数量级；以 rev4 D1 的 Δhead(k)/Δhead(全池)（D1.ba_ge30.ratio）为准，'
             '"差一个数量级"在正文中不引用。代价：只用头时 OA 相对重训为 '
             f'{bh["_oa"]:+.2f}（k=200，calib_rows），D1 的 OA Δhead − Δretrain 见 D1.oa.diff.* 行。')
    his, byk = _row('H2.ae_adv_hi90_max')['_his'], _row('H2.ae_adv_hi90_max.byk')['_byk']
    c27 = doc_lines('CONCLUSIONS', '0.27', '上界', '所有')
    r27 = doc_lines('REPOSITION', '0.27', '上界', '所有')
    o27 = [f'CONCLUSIONS 第 {i} 行' for i in doc_lines('CONCLUSIONS', '0.27', '上界') if i not in c27] + \
          [f'REPOSITION 第 {i} 行' for i in doc_lines('REPOSITION', '0.27', '上界') if i not in r27]
    over = sorted([(m_, k_, v_) for (m_, k_), v_ in byk.items() if v_ > 0.27], key=lambda t: -t[2])
    N.append(f'  18. [H2] "AE 优势的 90% 上界 ≤ +0.27"只在 k=50 与 k=200 两档合并时成立（7 个已报指标合并 k 的最大值 {max(his.values()):+.3f}，'
             f'为 {max(his, key=his.get)}）。按 k 拆开，超过 +0.27 的有：'
             + '；'.join(f'{m_} k={k_} {v_:+.3f}' for m_, k_, v_ in over)
             + f'；ba_ge30 在 k=50 / 200 分别为 {byk[("ba_ge30", 50)]:+.3f} / {byk[("ba_ge30", 200)]:+.3f}。'
             f'CONCLUSIONS {_ln(c27)}与 REPOSITION {_ln(r27)}写"在所有已报指标（和两档预算）上"或"所有指标上"，应改为'
             '"两档 k 合并时所有已报指标 ≤ +0.27；按 k 拆开只对 ba_ge30 成立（f1_ge30 在 k=50 为 +0.36）"。'
             f'其余只写"只报上界 ≤+0.27"的位置（{"、".join(o27)}）也应注明"两档 k 合并"。')
    reg = H5_REG
    c988 = doc_lines('CONCLUSIONS', '标签只来自 1 个区域时约 80%')
    c3e = doc_lines('CONCLUSIONS', 'C3e｜')
    N.append(f'  19. [H5] 区域恢复率（1 个区域 {reg[(1, "ae")][0]:.1f}% / {reg[(1, "s2")][0]:.1f}%，3 个区域 {reg[(3, "ae")][0]:.1f}% / {reg[(3, "s2")][0]:.1f}%，AE / S2）'
             '中的"区域"是候选池 20 个 KMeans 簇之一，结果依赖这一定义：改用 50 簇时 1 个区域为 '
             f'{reg[(1, "ae")][1]:.1f}% / {reg[(1, "s2")][1]:.1f}%，3 个区域为 {reg[(3, "ae")][1]:.1f}% / {reg[(3, "s2")][1]:.1f}%（AE 低于"91–94%"的下沿）。'
             f'CONCLUSIONS C3e（{_ln(c3e)}）把"标签的空间分布效应"定为 C，但 {_ln(c988)}把这组数字列在 B 级条目下（〔检验 3；B〕、〔C3；B〕）；'
             'REPOSITION H5 把"1 个区域约 80%，3 个区域 91–94%"列在"先验估计 B（摘要）"的关键数字里。'
             '这组数字不进摘要，正文写明"区域 = 20 簇之一"并附 50 簇敏感性。')
    lq = _row('H5.F3b_labelequiv')
    a5, a59 = lq['_a5']
    N.append(f'  20. [H5] PIVOT_EVALUATION §2"标签当量"一段混用了 α：9 国"50 个标签加 F3b 为 48.81 ≈ 400 个重训 48.80"用的是 α=0.4（重算 '
             f'{lq["_lq"]["v9_f3b"]:.2f} / {lq["_lq"]["v9_rt400"]:.2f}）；11 国"50 个标签加 F3b 为 46.83 ≈ 200 个重训 46.84"中的 46.83 是 α=0.5 的值'
             f'（重算 {a5:.2f}；α=0.4 为 {lq["_lq"]["v11_f3b"]:.2f}）。CONCLUSIONS 第 992 行只引了 9 国版，数值无误，但应注明 α=0.4、只在 BA 上相当。')
    CH = CHANGES
    CH += ['  1. H1.tgtonly 的 8 行（s0/s1 × BA/OA × AE/S2）证据等级由"A（M3：方向）"改为 C，并注"支持 H1 的 B 级方向结论"；MLP/LR 两行标 C（n=5）。',
           '  2. 检验 3 预注册各行（H1 的 es/total/resid/bias_oracle，H5 的 pk50/pem/prior_hurts）的"测试集复用"改写为：PT、SE、SK、LV 的测试集与检验 2 逐位相同，'
           '检验 2 已在其上评估过；之后又用于选开发国、headroom 与 G 设计审查（TESTSET_AUDIT §3–4）。空间分块（test_sp）各行另写，注明审计未单列 test_sp。',
           '  3. H2.knn10 / auroc_margin / guide_acc 的"测试集复用"改为"不用测试标签（重复 0 候选池）"；auroc 的预注册改为"是（冻结方案 §7 中介变量，描述）"；'
           'knn10 注明冻结前看过类似的几何纯度诊断；guide_acc 注明 NLL 属 §7 中介变量、准确率为事后描述。',
           '  4. θ_E1 的 ba_P2、ba_far 两行：预注册改为"是（冻结方案 §8 次要，不校正）；逐指标 TOST 判定为事后"，复用改为"层级 1+2 首次使用"；按 k 的两行注明 §8 次要。',
           '  5. H4.dose_slope 拆成两行：合并斜率（CI 为本表新算的国家 cluster bootstrap）与逐国均值（t 区间，11/11）。',
           '  6. 补算去 RO/PT：H1.ba_resid.spatial、H1附.dec_share_c3、θ_E1 的 TOST 最小界值与分 k 两行、AE 优势上界、标签当量、AUROC、指导准确率、'
           '几何 F、predstrat 收缩、区域恢复率、F3b λ 敏感性、排序相关、整体平移判据、M9/M9b、单源斜率（点估计）、P3；其余写"不适用/未算 + 原因"。',
           '  7. 新增 REPOSITION §1/§3 讨论 Cheng 要引用的行：S.cheng.pairwise_domestic（55.54/53.35）、S.cheng_pred_slope（+1.35，文献换算）、'
           'S.cheng.PT_resid_flip（+4.9/−11.9）、S.P3_loco（42.71/43.17，p=0.413）、H5.F3b_labelequiv（48.81 ≈ 48.80）。',
           '  8. 区域恢复率 4 行注"区域 = 20 簇之一"并附 50 簇敏感性，等级标"C（不进摘要）"。',
           '  9. H4.geom_vs_srs 两行：预注册改"否（rev1 B3 事后；R3 预注册的是胜率）"，复用改"是：G s0/s1 已复用（事后）"。H3 的 G_lgb 各行等级改"B（待 rev3 B2 四学习器判据）"。',
           '  10. S.P3_protocol：v1 为"读取（文档）"，v2 在 Alpha_Crops/results/experiments 找到结果文件并重算（出处改 PROGRESS_REVIEW §3.2 第 183 行 + CONCLUSIONS P3）。',
           '  11. H6.M9b 补 mlp3，并在注中列出 part1_rows.csv 中 k=0 的全部学习器；H1.tgtonly.s1.ba.mlp/lr、H6.M9.* 由"读取"改为从 alt_long.csv 重算；'
           'H5.F3b_lambda_sens 由"读取"改为从 frontier3_rows.csv 重算。',
           '  12. H5.balhead_full.vs_k0 的描述由"λ=0.1"更正为"λ=1e-4"（calib_diag.py 全池头的实际设置；rev4 D1 核对 (2) 也按 λ=1e-4 复现）。',
           '  13. S.single_source_slope 去掉另一脚本的 90% CI 列（移入注，标"不引用"），注明 perm_p 不引用、两种 FE 斜率只报其一。',
           '  14. rev4 D1（decision_curve/）与 D2（cheng_comparable/）的已有输出填入：D1.*（ba_ge30 与 OA 的 Δhead、Δretrain、差值、全池、比值）、'
           'D2.lgbm.*（全类 BA、4 类 OA、4 类 BA，外加方案外的 8 国行）；rev2 学习器行仍为"待 rev2"。',
           '  15. 计数更正：doc_comparison 中 labelequiv 的 inf 一行现计入"一致（≤0.01）"；discrepancies 抬头按实际类别计数并校验合计。'
           'v1 实际为重算 160、读取 11、占位 23（194 行），不是 v1 报告中的 157/9/28。',
           '  16. NUMBERS.md 读法说明写明旧 LR/MLP 试点"不属于 rev2 估计量"，并补术语定义与未收录内容清单。',
           '  17. discrepancies 第三节新增第 17–20 条（"差一个数量级"口径、"≤+0.27"只在 k 合并时成立、区域定义、PIVOT 标签当量混用 α），'
           '第 1、4、7、12 条补全位置与细节。']


def main():
    R = A.load_rows()
    assert len(R) == 3872, len(R)
    Rr, res, Cc = h1_check3()
    h1_target_only()
    h1_decision_upper()
    h2_guidance(R)
    h2_check1()
    b2 = h3_label_value(R)
    h4_strategy(R)
    h5_usage(Rr, Cc)
    h6_feature(R)
    d1_rows()
    supp(res)
    build_notes()
    C = compare_all()
    C.to_csv(os.path.join(HERE, 'doc_comparison.csv'), index=False, encoding='utf-8-sig')
    df = write_outputs(C)
    print(f'行数 {len(df)}；核对字段 {len(C)}；真实差异 {int((C.cls == "真实差异").sum())}')
    print(C[C.cls == '真实差异'].to_string())


if __name__ == '__main__':
    main()
