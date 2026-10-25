"""
确认运行分析（方案 label_expansion/CONFIRM_PLAN_v5.md，sha256[:16]=5384fb7efa67f495，冻结；实现 §3–§5 与 §10 的填充表）。

读统一接口（--root 为结果根目录，须为绝对路径）
  splits/{unit}.npz                                   test（全局下标）、big；正式模式另用 pool、foreign、feasible；unit = f'{国家}_F{1|2}'
  selection/{unit}.npz                                正式模式：pi_r、S0_r、S1_{策略}_k{k1}_r{r}（§7 第 6 步 sel_hash 核对）
  split_manifest.csv                                  正式模式：各单元划分与选样的内容哈希（splits_hash、selection_hash）
  probs/{unit}/{learner}_{arm}_{feat}_r{r}.npz        'P_test'（float32）；learner ∈ {lgb, rf}；Z0 与 TGT 只有 r0
  runs/{unit}.csv                                     元数据（learner, arm, feat, r, k, sel_hash, seed, n_train, t_fit, status, code_hash；
                                                      正式模式另需 train_sha、threads、splits_hash、selection_hash）

完整性检查（§7 第 4、6 步；先对全部单元做完、全部通过后才读概率计算）
  全部模式  runs 的键 ⊂ 方案拟合计划（本文件独立写出：每单元 lgb 70、rf 8）且不重复；status ∈ {ok, failed}；
            计划中的拟合在 runs 中没有（absent）时：正式分析拒绝运行，--structure-check 计为问题
  正式模式  划分与选样的内容哈希 = split_manifest.csv = runs 每行；存档的可行性与重算一致；runs 线程数 = 2；
            code_hash 只有一种且等于当前 confirm_split.py + confirm_run.py 按 confirm_split.code_hash 规则算出的值；
            sel_hash、train_sha、n_train 与选样清单重建的训练下标一致
  正式根目录（results/confirm_v1）另核：split_manifest.csv、confirm_split.py、confirm_run.py 的 sha256[:16] 已登记在
            plan_hashes.txt；非结构检查的正式分析另要求本文件的 sha256[:16] 已登记（只运行冻结版本）
输出 {root}/analysis/：unit_rows.csv、country_values.csv、estimates.csv、tests.csv、a2_ratio.csv、
  x2_position.csv、x3_equiv.csv、x3_cost.csv、fill_table.csv、summary.txt（--retro 另有 retro_reconcile.txt）

度量（§3；import 复用已冻结的 step0B / step0B2，不改其文件）
  τ 族      ℓ = log P − τ·log π̂，π̂ = 该臂测试集平均预测；τ = −2..2 步长 0.05；精确 log（step0B.log_or_neginf）；
            π̂_c = 0 的类偏移取 0、argmax 并列取第一个（step0B2.tau_preds_grid = step0B.tau_preds）
  指标      OA、ba_ge30（step0B.Scorer，与 guidance_G._scores 相同）
  包络 / FA step0B.envelope、step0B.fa_of；OA★ = 同一学习器、同一特征的 Z0 在 τ=0 的 OA（两个场景共用）；
            窗口 OA★ + [−4, 0] 步长 0.1（41 点）
  两臂之差  共同可达格点上逐点 BA 差的平均；共同可达 < 50% 记缺失
  D         锚点 = min(OA★_ae, OA★_s2)（同一学习器的 Z0）；两特征该臂的包络在共同可达格点上平均 BA_ae − BA_s2
  国家值    抽样内配对求差（Z0、TGT 只有 r0，与各 r 配对）→ 对 r 平均 → 对 F1、F2 平均（每层跳过缺失）；
            both = AE、S2 国家值的平均（任一缺失则缺失）；交互量（E2）= AE 国家值 − S2 国家值
  推断      国家为单位，单样本 t（df = n − 1）；9 国 = 去掉 RO、PT；可判定：9 国至少 7 国、11 国至少 9 国有值
  Holm      H1、H2 单侧 p，族 α = 0.025；不可判定的一条按 p = 1（不拒绝）
  X2、X3    只用全部输入量都有值的国家（9 国内）；不足 7 国记"不可判定"。集中与随机样本曲线之差的 90% 区间覆盖 0
            （或不可判定、不可计算）时，交换率（conc_keq、conc_rate）的点估计与区间一律写"不报"

模式
  （默认）  正式分析；输出目录已存在时拒绝覆盖（--overwrite 只允许非正式根目录）。
  --retro   用适配器把 G s0/s1 的已存概率写成同一接口（F1 = s0、F2 = s1、r = 重复号，R = 2），只算场景 A 与 RQ2，
            并与 PC-1 的 pc1_rows.csv 逐单元核对（容差 1e-9），核对方案 §6 登记的 H1、H2、RF 数字。
  --structure-check  只检查结构（行数、键、sel_hash 与哈希、概率合法、拟合状态），不写指标文件、不打印任何指标；
            正式模式不打印依赖结果的计数（缺失值、可判定行）。
  删除保护  --retro 与 --overwrite 只允许 --root 在 verify/confirm_code_scratch/ 之下；只删除带本脚本标记文件的目录
            （回溯根目录的 .analyze_confirm_retro、输出目录的 .analyze_confirm_output）。

用法：/Users/perng_mini/AlphaCrops_local/.venv/bin/python -B analyze_confirm.py --root <根目录> [--retro] [--structure-check] [--overwrite]
"""
import os
import sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True  # 不在项目目录写 __pycache__
import argparse
import glob
import hashlib
import shutil
import time

import numpy as np
import pandas as pd
from scipy import stats

SELF = os.path.abspath(__file__)   # 在导入 step0B（会 chdir）之前解析
HERE = os.path.dirname(SELF)
L = os.path.dirname(HERE)
PARTB = '/Users/perng_mini/AlphaCrops_local/verify/step0/partB'
PARTB2 = '/Users/perng_mini/AlphaCrops_local/verify/step0/partB2'
PC1_DIR = '/Users/perng_mini/AlphaCrops_local/verify/confirm_plan/pc1'
FORMAL_ROOT = '/Users/perng_mini/AlphaCrops_local/results/confirm_v1'
SCRATCH = '/Users/perng_mini/AlphaCrops_local/verify/confirm_code_scratch'
PLAN_HASHES = os.path.join(L, 'plan_hashes.txt')
RUNNER_FILES = ('confirm_split.py', 'confirm_run.py')   # runs 的 code_hash = sha256(二者依次拼接)[:16]
RETRO_MARK = '.analyze_confirm_retro'
ANALYSIS_MARK = '.analyze_confirm_output'
R_G = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
R_ML = '/Users/perng_mini/AlphaCrops_local/results/guidance_G_ml'
PLAN = os.path.join(L, 'CONFIRM_PLAN_v5.md')


def sha256(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# 启动时断言：方案与复用的冻结模块（完整 sha256）
FROZEN = {
    PLAN: '5384fb7efa67f49543012ade5f8c577058d225b8289f8384d691ac40cfaaeba0',
    os.path.join(L, 'guidance_G.py'): '47d729be62d5d2c9280a8a0e2f5f0f85a88e315255cf181321982d7b00e650b5',
    os.path.join(L, 'stageA.py'): '5db47e2196d169d41217f72c36eca364761a993246e741ec611f5a23e9e2e6bb',
    os.path.join(L, 'check3_priors.py'): 'b57282b95d5fc954de2a8f7ee2a9e78293e48ece50bb6c495eec5a5d9ef5cd73',
    os.path.join(L, 'multilearner_G.py'): 'f32183af651d337a0809ffa0c678e65a9e6c6e30d0104eb904f8f1b8dc1150a4',
    os.path.join(PARTB, 'step0B.py'): '21a8e5d770fce2b2037a4060cceddf1dfd49460b47e2ab94ac1ee08d64b3a7c1',
    os.path.join(PARTB2, 'step0B2.py'): 'd7f37ca5f11c2e9ec630c9884fc3f675e50959bc3f8aa1830fa1e91dcbbbe46c',
}
for _p, _h in FROZEN.items():
    assert sha256(_p) == _h, f'冻结文件哈希不符：{_p}'

sys.path.insert(0, PARTB2)
import step0B2 as B2  # noqa: E402  导入链：step0B（chdir 到 label_expansion，只读）→ guidance_G、check3_priors

B = B2.B
G = B.G

# ---------------------------------------------------------------- 常量（方案 §1–§5）
C = 10
FEATS = ('ae', 's2')
COUNTRIES = tuple(B.COUNTRIES)
DROP = ('RO', 'PT')
NINE = tuple(c for c in COUNTRIES if c not in DROP)
SETS = {'9': NINE, '11': COUNTRIES}
MIN_DECIDE = {'9': 7, '11': 9}
TAUS = np.round(np.arange(-2, 2 + 1e-9, 0.05), 2)
assert np.array_equal(TAUS, B2.TAUS_NEW) and len(TAUS) == 81
I_T0 = int(np.flatnonzero(TAUS == 0.0)[0])
OFFS = B.OFFS
assert len(OFFS) == 41 and OFFS[0] == -4.0 and OFFS[-1] == 0.0
MIN_COV = B.MIN_COV
assert MIN_COV == 0.5
N_TEST_MIN, BIG_MIN = 3000, 2
ROWSUM_TOL = 1e-3
R_DRAWS = 3
KS = (50, 100, 200, 400)
K1S = (100, 300)
STRATS = ('SRS', 'EXT', 'BA')
LEARNERS = ('lgb', 'rf')
RT_ARMS = [f'RT{k}' for k in KS]
B_ARMS = ['B_S0'] + [f'B_{s}{k1}' for s in STRATS for k1 in K1S]
ARMS = {'lgb': ['Z0'] + RT_ARMS + ['TGT'] + B_ARMS, 'rf': ['Z0', 'RT200']}
RETRO_ARMS = {'lgb': ['Z0'] + RT_ARMS, 'rf': ['Z0', 'RT200']}
NO_R = ('Z0', 'TGT')  # 只拟合一次（写 r0）
RUN_COLS = ['learner', 'arm', 'feat', 'r', 'k', 'sel_hash', 'seed', 'n_train', 't_fit', 'status', 'code_hash']
FORMAL_COLS = ['train_sha', 'threads', 'splits_hash', 'selection_hash']
HASH_COLS = ('sel_hash', 'train_sha', 'code_hash', 'splits_hash', 'selection_hash')   # 一律按字符串读
FIT_KEY = ['learner', 'arm', 'feat', 'r']
N_FITS = {'lgb': 70, 'rf': 8}   # 每个可行单元的拟合次数（§2、§9）
THREADS = 2                     # §2.3 每进程 2 线程
# 标签构成（集中, 分散）：集中 = S0 与 EXT；分散 = SRS、BA 与场景 A 的全部标签（§5 X3）
COMPOSITION = {'Z0': (0, 0), **{f'RT{k}': (0, k) for k in KS}, 'B_S0': (100, 0),
               **{f'B_EXT{k1}': (100 + k1, 0) for k1 in K1S},
               **{f'B_{s}{k1}': (100, k1) for s in ('SRS', 'BA') for k1 in K1S}}
N_TOTAL = {a: sum(v) for a, v in COMPOSITION.items()}

# 两臂之差：名称 → (臂 a, 臂 b, 场景)；值 = FA(a) − FA(b)
DIFFS = {'lgb': {**{f'dFA_RT{k}': (f'RT{k}', 'Z0', 'A') for k in KS},
                 'dFA_TGT': ('TGT', 'Z0', 'A'),
                 **{f'dFA_{a}': (a, 'Z0', 'B') for a in B_ARMS},
                 **{f'B1_SRS-EXT{k1}': (f'B_SRS{k1}', f'B_EXT{k1}', 'B') for k1 in K1S},
                 **{f'B1_BA-SRS{k1}': (f'B_BA{k1}', f'B_SRS{k1}', 'B') for k1 in K1S},
                 **{f'B2_gain_{s}{k1}': (f'B_{s}{k1}', 'B_S0', 'B') for s in STRATS for k1 in K1S},
                 **{f'B2_gainA{k1}': (f'RT{100 + k1}', 'RT100', 'B') for k1 in K1S},
                 'X1_S0-RT100': ('B_S0', 'RT100', 'B'),
                 **{f'X1_{s}{k1}-RT{100 + k1}': (f'B_{s}{k1}', f'RT{100 + k1}', 'B') for s in STRATS for k1 in K1S}},
         'rf': {'dFA_RT200': ('RT200', 'Z0', 'A')}}
D_ARMS = {'lgb': ['Z0', 'RT200', 'B_SRS100'], 'rf': ['Z0']}

# 汇总量：(编号, 族, 学习器, 单元量名, 类型, 主国家集)；类型 feat → both/ae/s2，D → ae-s2，inter → AE−S2 交互
EST_SPECS = [
    ('H1', 'H1', 'lgb', 'dFA_RT200', 'feat', '9'),
    ('H2', 'H2', 'lgb', 'D_Z0', 'D', '11'),
    *[(f'A1_RT{k}', 'A1', 'lgb', f'dFA_RT{k}', 'feat', '9') for k in KS],
    ('A2_TGT', 'A2', 'lgb', 'dFA_TGT', 'feat', '9'),
    ('A3_RF_RT200', 'A3', 'rf', 'dFA_RT200', 'feat', '9'),
    *[(f'B1_SRS-EXT{k1}', 'B1', 'lgb', f'B1_SRS-EXT{k1}', 'feat', '9') for k1 in K1S],
    *[(f'B1_BA-SRS{k1}', 'B1', 'lgb', f'B1_BA-SRS{k1}', 'feat', '9') for k1 in K1S],
    ('B2_S0-Z0', 'B2', 'lgb', 'dFA_B_S0', 'feat', '9'),
    *[(f'B2_gain_{s}{k1}', 'B2', 'lgb', f'B2_gain_{s}{k1}', 'feat', '9') for s in STRATS for k1 in K1S],
    *[(f'B2_gainA{k1}', 'B2', 'lgb', f'B2_gainA{k1}', 'feat', '9') for k1 in K1S],
    ('X1_S0-RT100', 'X1', 'lgb', 'X1_S0-RT100', 'feat', '9'),
    *[(f'X1_{s}{k1}-RT{100 + k1}', 'X1', 'lgb', f'X1_{s}{k1}-RT{100 + k1}', 'feat', '9')
      for s in STRATS for k1 in K1S],
    *[(f'X2_{a}', 'X2', 'lgb', f'dFA_{a}', 'feat', '9') for a in B_ARMS],
    ('E1_D_RT200', 'E1', 'lgb', 'D_RT200', 'D', '11'),
    ('E1_D_S0+SRS100', 'E1', 'lgb', 'D_B_SRS100', 'D', '11'),
    ('E1_RF_D_Z0', 'E1', 'rf', 'D_Z0', 'D', '11'),
    ('E2_RT200', 'E2', 'lgb', 'dFA_RT200', 'inter', ''),
    *[(f'E2_B1_SRS-EXT{k1}', 'E2', 'lgb', f'B1_SRS-EXT{k1}', 'inter', '') for k1 in K1S],
    *[(f'E2_B1_BA-SRS{k1}', 'E2', 'lgb', f'B1_BA-SRS{k1}', 'inter', '') for k1 in K1S],
    ('M1_RT200-Z0_BA', 'M1', 'lgb', 't0_RT200-Z0_BA', 'feat', '9'),
    ('M1_RT200-Z0_OA', 'M1', 'lgb', 't0_RT200-Z0_OA', 'feat', '9'),
    ('M1_Z0_AE-S2_BA', 'M1', 'lgb', 't0_Z0_AE-S2_BA', 'D', '11'),
    ('M1_Z0_AE-S2_OA', 'M1', 'lgb', 't0_Z0_AE-S2_OA', 'D', '11'),
]
RETRO_FAMS = ('H1', 'H2', 'A1', 'A3', 'E1', 'E2', 'M1')
KIND_FEATS = {'feat': ('both', 'ae', 's2'), 'D': ('ae-s2',), 'inter': ('inter',)}

# X3（§5）
X3_SEED = 'confirm_v1_x3'
N_BOOT = 2000
KN_R = np.array([0.0, 50, 100, 200, 400])   # 随机样本曲线的档位
KN_C = np.array([0.0, 100, 200, 400])       # 集中样本曲线：Z0、S0 only、S0+EXT100、S0+EXT300
COSTS = (1, 2, 3)
BUDGETS = (100, 200, 400)                   # 同一成本的比较点（单位：集中样本单价）

# PC-1 登记值（方案 §6）
PC1_REGISTERED = {'H1': '+2.40 [+1.53, +3.27]', 'H2': '+4.95 [+2.15, +7.74]', 'A3_RF_RT200': '+2.30 [+1.12, +3.48]'}
TOL = 1e-9


def draws_of(arm, R):
    return [0] if arm in NO_R else list(range(R))


def expected_units():
    return [f'{c}_F{h}' for c in COUNTRIES for h in (1, 2)]


def arms_of(mode):
    return RETRO_ARMS if mode == 'retro' else ARMS


def diffs_of(mode, lrn):
    """本模式下要算的两臂之差；回溯只算场景 A 且两臂都有已存概率的。"""
    arms = arms_of(mode)[lrn]
    return {n: v for n, v in DIFFS[lrn].items()
            if mode != 'retro' or (v[2] == 'A' and v[0] in arms and v[1] in arms)}


def d_arms_of(mode, lrn):
    return [a for a in D_ARMS[lrn] if a in arms_of(mode)[lrn]]


def fit_keys(mode, R):
    """方案的拟合计划（§2；在此独立写出，不导入 confirm_run）：{(learner, arm, feat, r)}。"""
    return {(lrn, a, f, r) for lrn, arms in arms_of(mode).items() for a in arms for f in FEATS for r in draws_of(a, R)}


assert {lrn: sum(k[0] == lrn for k in fit_keys('formal', R_DRAWS)) for lrn in LEARNERS} == N_FITS


# ---------------------------------------------------------------- 路径保护、内容哈希与登记（§7 第 4、6 步）
def under(path, base):
    """path 是否为 base 本身或其下的路径（真实路径；前缀后加分隔符）。"""
    p, b = os.path.realpath(path), os.path.realpath(base)
    return p == b or p.startswith(b + os.sep)


def assert_scratch(root):
    """会删除目录的模式（--retro、--overwrite）只允许 root 在开发 scratch 之下（不含 scratch 本身）。"""
    assert os.path.realpath(root).startswith(os.path.realpath(SCRATCH) + os.sep), \
        f'{root} 不在 {SCRATCH}/ 之下：--retro 与 --overwrite 只用于开发目录'


def clear_own(path, marker):
    """删除本脚本生成过的目录：不存在或为空时直接返回；有内容时须含标记文件，否则拒绝删除。"""
    if os.path.isdir(path) and os.listdir(path):
        assert os.path.exists(os.path.join(path, marker)), f'{path} 没有标记文件 {marker}：不是本脚本生成的，拒绝删除'
        shutil.rmtree(path)


def arr_sha(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(f'{a.dtype.str}|{a.shape}|'.encode() + a.tobytes()).hexdigest()


def content_hash(d):
    """字典全部数组的内容哈希（算法与 confirm_split.content_hash 相同；在此自写，不导入运行器）。"""
    h = hashlib.sha256()
    for k in sorted(d):
        h.update(f'{k}|{arr_sha(d[k])}|'.encode())
    return h.hexdigest()


def npz(path):
    with np.load(path) as z:
        return dict(z)


def read_manifest(root):
    return pd.read_csv(os.path.join(root, 'split_manifest.csv'), dtype=str).set_index('unit')


def runner_code_hash():
    """runs 中 code_hash 的规则（同 confirm_split.code_hash）：sha256(confirm_split.py ‖ confirm_run.py)[:16]。"""
    h = hashlib.sha256()
    for n in RUNNER_FILES:
        with open(os.path.join(HERE, n), 'rb') as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def check_registration(root, self_too):
    """正式根目录：split_manifest.csv、confirm_split.py、confirm_run.py（self_too 时另加本文件）的 sha256[:16]
    已按 "{文件名} sha256[:16]={哈希}" 登记在 plan_hashes.txt。"""
    with open(PLAN_HASHES, encoding='utf-8') as f:
        rec = f.read()
    files = [os.path.join(root, 'split_manifest.csv')] + [os.path.join(HERE, n) for n in RUNNER_FILES]
    for p in files + ([SELF] if self_too else []):
        n, h = os.path.basename(p), sha256(p)[:16]
        assert f'{n} sha256[:16]={h}' in rec, f'{n} 的 sha256[:16]={h} 未登记在 plan_hashes.txt'


# ---------------------------------------------------------------- 读接口与完整性检查
def target_rows(arm, r, S, Sel):
    """由划分（pool）与选样清单（selection/{unit}.npz）重建该臂的目标训练下标（方案 §2；不导入 confirm_run）。"""
    pool = np.asarray(S['pool'], dtype=np.int64)
    if arm == 'Z0':
        return np.zeros(0, np.int64)
    if arm == 'TGT':
        return pool
    if arm.startswith('RT'):
        return pool[Sel[f'pi_r{r}'][:int(arm[2:])]]
    S0 = np.asarray(Sel[f'S0_r{r}'], dtype=np.int64)
    if arm == 'B_S0':
        return S0
    return np.r_[S0, Sel[f'S1_{arm[2:-3]}_k{arm[-3:]}_r{r}']].astype(np.int64)   # B_{SRS,EXT,BA}{100,300}


def check_sel_manifest(unit, S, Sel, runs):
    """§7 第 6 步：runs 每一行（含 failed）的 k、sel_hash、n_train、train_sha 与清单重建的训练下标一致。
    约定与 confirm_run 相同：sel_hash = G.sel_hash(目标下标)（Z0 记 'k0'）；train_sha = sha256(外国 ‖ 目标)[:16]（TGT 无外国）。"""
    fr = np.asarray(S['foreign'], dtype=np.int64)
    for q in runs.itertuples(index=False):
        ti = target_rows(q.arm, int(q.r), S, Sel)
        tr = np.r_[fr if q.arm != 'TGT' else np.zeros(0, np.int64), ti].astype(np.int64)
        key = (unit, q.learner, q.arm, q.feat, q.r)
        assert int(q.k) == len(ti) and str(q.sel_hash) == (G.sel_hash(ti) if len(ti) else 'k0'), (key, 'sel_hash 与选样清单不一致')
        assert int(q.n_train) == len(tr), (key, 'n_train 与清单不一致')
        assert str(q.train_sha) == hashlib.sha256(tr.tobytes()).hexdigest()[:16], (key, 'train_sha 与清单不一致')


def check_unit(root, unit, y, mode, R, M=None, strict=True):
    """一个单元的完整性检查（不读概率）。返回 (信息 dict, None) 或 (None, 剔除原因)。
    strict（正式分析）时计划中的拟合不得缺；否则缺的记 'absent'（结构检查计为问题）。"""
    sp = os.path.join(root, 'splits', f'{unit}.npz')
    assert os.path.exists(sp), (unit, '划分文件不存在（划分构建对全部单元都写文件，含不可行单元）')
    S = npz(sp)
    test = np.asarray(S['test'], dtype=np.int64)
    yT = np.asarray(y)[test]
    big = np.sort(np.asarray(S['big'], dtype=np.int64))
    assert np.array_equal(big, np.flatnonzero(np.bincount(yT, minlength=C) >= G.MIN_N)), (unit, 'big 与测试集重算不一致')
    if mode == 'formal':
        M = read_manifest(root) if M is None else M
        Sel = npz(os.path.join(root, 'selection', f'{unit}.npz'))
        hs, hl = content_hash(S), content_hash(Sel)
        assert unit in M.index and M.loc[unit, 'splits_hash'] == hs and M.loc[unit, 'selection_hash'] == hl, \
            (unit, '划分或选样的内容哈希与 split_manifest.csv 不一致')
        feas = len(test) >= N_TEST_MIN and len(big) >= BIG_MIN
        assert S['feasible'].dtype == np.bool_ and bool(S['feasible']) == feas, (unit, '存档的可行性与重算不一致')
        if not feas:
            return None, f'不可行（n_test {len(test)}，|big| {len(big)}）'
    runs = pd.read_csv(os.path.join(root, 'runs', f'{unit}.csv'), dtype={c: str for c in HASH_COLS})
    must = RUN_COLS + (FORMAL_COLS if mode == 'formal' else [])
    assert set(must) <= set(runs.columns), (unit, 'runs 缺列', sorted(set(must) - set(runs.columns)))
    assert not {'oa', 'ba', 'ba_ge30', 'fa', 'FA', 'acc', 'f1'} & set(runs.columns), (unit, 'runs 不得记指标')
    runs['r'] = runs.r.astype(int)
    assert not runs.duplicated(FIT_KEY).any(), (unit, 'runs 键重复')
    have, plan = set(runs[FIT_KEY].itertuples(index=False, name=None)), fit_keys(mode, R)
    assert have <= plan, (unit, '方案外的拟合', sorted(have - plan)[:8])
    assert runs.status.isin(['ok', 'failed']).all(), (unit, 'status 只能是 ok 或 failed', sorted(set(runs.status)))
    absent = plan - have
    assert not (strict and absent), (unit, f'runs 缺 {len(absent)} 次计划拟合，拒绝分析', sorted(absent)[:8])
    if mode == 'formal':
        assert (runs.splits_hash == hs).all() and (runs.selection_hash == hl).all(), \
            (unit, 'runs 的 splits_hash / selection_hash 与划分清单不一致')
        assert (runs.threads.astype(int) == THREADS).all(), (unit, f'runs 的线程数不是 {THREADS}')
        check_sel_manifest(unit, S, Sel, runs)
    # 配对：同一 (学习器, 臂, r) 的 AE 与 S2 用同一批标签；RF 的 RT200 与 LightGBM 的 RT200 用同一批标签
    ok = runs[runs.status == 'ok'].assign(sel_hash=lambda d: d.sel_hash.fillna(''))
    for key, g in ok.groupby(['learner', 'arm', 'r']):
        assert g.sel_hash.nunique() == 1, (unit, key, 'AE 与 S2 的 sel_hash 不同')
    rt = ok[ok.arm == 'RT200'].groupby(['r', 'learner']).sel_hash.first().unstack('learner')
    if {'lgb', 'rf'} <= set(rt.columns):
        both = rt.dropna()
        assert (both['lgb'] == both['rf']).all(), (unit, 'RF 与 LightGBM 的 RT200 选样不同')
    status = {(lrn, f, a, r): 'absent' for lrn, a, f, r in absent}
    status.update({(q.learner, q.feat, q.arm, q.r): q.status for q in runs.itertuples(index=False)})
    return dict(unit=unit, yT=yT, big=big, status=status, n_test=len(test),
                code_hash=set(runs.code_hash.astype(str))), None


def load_probs(root, d):
    """读 status = ok 的测试行概率：须为 float32、形状 (n_test, C)、有限、非负、行和为 1。"""
    P = {}
    for (lrn, f, arm, r), st in d['status'].items():
        if st != 'ok':
            continue
        Pt = np.load(os.path.join(root, 'probs', d['unit'], f'{lrn}_{arm}_{f}_r{r}.npz'))['P_test']
        key = (d['unit'], lrn, arm, f, r)
        assert Pt.dtype == np.float32 and Pt.shape == (d['n_test'], C), (key, Pt.dtype, Pt.shape)
        assert np.isfinite(Pt).all() and (Pt >= 0).all(), (key, '概率非法')
        assert np.abs(Pt.sum(1, dtype=np.float64) - 1).max() < ROWSUM_TOL, (key, '行和')
        P[(lrn, f, arm, r)] = Pt
    return P


# ---------------------------------------------------------------- 一个单元：τ 曲线、包络、差值、D
def pihat_of(Z):
    """π̂：该臂测试集平均预测概率（exp(Z − 行最大) 归一化后取列均值；与 step0B / PC-1 相同，不用标签）。"""
    Pa = np.exp(Z - Z.max(1, keepdims=True))
    Pa /= Pa.sum(1, keepdims=True)
    return Pa.mean(0)


def unit_curves(yT, big, P):
    """P：{(learner, feat, arm, r): (n, C) 概率} → {同键: (OA (81,), BA (81,))}。"""
    sc = B.Scorer(yT, big)
    CV = {}
    for key, Pa in P.items():
        Z = B.log_or_neginf(B.norm64(Pa))
        CV[key] = sc.curve(B2.tau_preds_grid(Z, pihat_of(Z), TAUS))
    return CV


def paired(ea, eb):
    """两条上包络在共同可达格点上的逐点差平均；共同可达 < 50% 记缺失。返回 (值, 共同可达率)。"""
    if ea is None or eb is None:
        return np.nan, np.nan
    ok = np.isfinite(ea) & np.isfinite(eb)
    cov = float(ok.mean())
    return (float(np.mean(ea[ok] - eb[ok])) if cov >= MIN_COV else np.nan), cov


def unit_rows(unit, CV, mode, R):
    """一个单元的全部逐单元量（level ∈ arm、diff、D、t0）。缺失的臂给出 NaN 行，行数只由方案决定。"""
    country, half = unit.split('_', 1)   # 正式 {国家}_F{1|2}；冒烟 {国家}_s{划分}
    base = dict(country=country, half=half, unit=unit)
    rows = []
    arms = arms_of(mode)
    for lrn in LEARNERS:
        star = {}
        for f in FEATS:
            z0 = CV.get((lrn, f, 'Z0', 0))
            star[f] = z0[0][I_T0] if z0 is not None else np.nan
            grid = star[f] + OFFS
            env = {}
            for arm in arms[lrn]:
                for r in draws_of(arm, R):
                    cv = CV.get((lrn, f, arm, r))
                    row = dict(level='arm', learner=lrn, **base, r=r, feat=f, name=arm, arm_a=arm, arm_b='',
                               OA_star=star[f])
                    if cv is None or not np.isfinite(star[f]):
                        env[(arm, r)] = None
                        rows.append(dict(row, value=np.nan, cov=np.nan, missing=True))
                        continue
                    best = B.envelope([cv], grid)[0]
                    env[(arm, r)] = best
                    fa, cov = B.fa_of(best)
                    oa, ba = cv
                    rows.append(dict(row, value=fa, cov=cov, missing=bool(cov < MIN_COV), oa_t0=oa[I_T0], ba_t0=ba[I_T0],
                                     oa_min=float(oa.min()), oa_max=float(oa.max())))
            for name, (a, b, _) in diffs_of(mode, lrn).items():
                for r in ([0] if a in NO_R and b in NO_R else range(R)):   # 抽样内配对；Z0、TGT 只有 r0
                    v, cov = paired(env[(a, 0 if a in NO_R else r)], env[(b, 0 if b in NO_R else r)])
                    rows.append(dict(level='diff', learner=lrn, **base, r=r, feat=f, name=name, arm_a=a, arm_b=b,
                                     OA_star=star[f], value=v, cov=cov, missing=bool(not cov >= MIN_COV)))
            if lrn == 'lgb':  # M1：τ = 0 的 RT200 − Z0
                z0c = CV.get(('lgb', f, 'Z0', 0))
                for r in range(R):
                    rc = CV.get(('lgb', f, 'RT200', r))
                    for j, met in ((1, 'BA'), (0, 'OA')):
                        v = rc[j][I_T0] - z0c[j][I_T0] if rc is not None and z0c is not None else np.nan
                        rows.append(dict(level='t0', learner=lrn, **base, r=r, feat=f, name=f't0_RT200-Z0_{met}',
                                         arm_a='RT200', arm_b='Z0', value=v, missing=bool(np.isnan(v))))
        # D（AE − S2；锚点 = min(OA★_ae, OA★_s2)）
        anc = min(star.values()) if all(np.isfinite(v) for v in star.values()) else np.nan
        grid = anc + OFFS
        for arm in d_arms_of(mode, lrn):
            for r in draws_of(arm, R):
                ca, cs = CV.get((lrn, 'ae', arm, r)), CV.get((lrn, 's2', arm, r))
                ea = B.envelope([ca], grid)[0] if ca is not None and np.isfinite(anc) else None
                es = B.envelope([cs], grid)[0] if cs is not None and np.isfinite(anc) else None
                v, cov = paired(ea, es)
                rows.append(dict(level='D', learner=lrn, **base, r=r, feat='ae-s2', name=f'D_{arm}', arm_a=arm, arm_b=arm,
                                 OA_star=anc, value=v, cov=cov, missing=bool(not cov >= MIN_COV),
                                 reach_ae=float(np.isfinite(ea).mean()) if ea is not None else np.nan,
                                 reach_s2=float(np.isfinite(es).mean()) if es is not None else np.nan))
        if lrn == 'lgb':  # M1：τ = 0 的 Z0 AE − S2
            ca, cs = CV.get(('lgb', 'ae', 'Z0', 0)), CV.get(('lgb', 's2', 'Z0', 0))
            for j, met in ((1, 'BA'), (0, 'OA')):
                v = ca[j][I_T0] - cs[j][I_T0] if ca is not None and cs is not None else np.nan
                rows.append(dict(level='t0', learner=lrn, **base, r=0, feat='ae-s2', name=f't0_Z0_AE-S2_{met}',
                                 arm_a='Z0', arm_b='Z0', value=v, missing=bool(np.isnan(v))))
    return rows


def quantity_names(mode, lrn):
    """本模式下逐单元量（diff、D、t0）的名称集合。"""
    q = set(diffs_of(mode, lrn)) | {f'D_{a}' for a in d_arms_of(mode, lrn)}
    if lrn == 'lgb':
        q |= {f't0_{p}_{m}' for p in ('RT200-Z0', 'Z0_AE-S2') for m in ('BA', 'OA')}
    return q


def est_specs(mode):
    return [s for s in EST_SPECS if s[3] in quantity_names(mode, s[2]) and (mode != 'retro' or s[1] in RETRO_FAMS)]


def expected_counts(mode, R):
    """每个单元应有的行数（只由方案决定）。"""
    n = {'arm': 0, 'diff': 0, 'D': 0, 't0': 0}
    for lrn in LEARNERS:
        n['arm'] += len(FEATS) * sum(len(draws_of(a, R)) for a in arms_of(mode)[lrn])
        n['diff'] += len(FEATS) * sum(1 if a in NO_R and b in NO_R else R for a, b, _ in diffs_of(mode, lrn).values())
        n['D'] += sum(len(draws_of(a, R)) for a in d_arms_of(mode, lrn))
    n['t0'] = len(FEATS) * 2 * R + 2
    return n


# ---------------------------------------------------------------- 国家值与汇总
def country_table(U):
    """逐单元量 → 国家值（长表：learner, name, feat, country, value, n_half）。
    对 r 平均 → 对 F1、F2 平均（跳过缺失）；both = ae、s2 国家值平均（任一缺失则缺失）；inter = ae − s2。"""
    x = U[U.level != 'arm']
    h = x.groupby(['learner', 'name', 'feat', 'country', 'half']).value.mean().rename('value').reset_index()
    c = h.groupby(['learner', 'name', 'feat', 'country']).agg(value=('value', 'mean'),
                                                             n_half=('value', 'count')).reset_index()
    fe = c[c.feat.isin(FEATS)].set_index(['learner', 'name', 'country', 'feat']).value.unstack('feat')
    extra = [fe[list(FEATS)].mean(axis=1, skipna=False).rename('value').reset_index().assign(feat='both'),
             (fe['ae'] - fe['s2']).rename('value').reset_index().assign(feat='inter')]
    return pd.concat([c] + extra, ignore_index=True)[['learner', 'name', 'feat', 'country', 'value', 'n_half']]


def cv_lookup(CT):
    return {k: g.set_index('country').value for k, g in CT.groupby(['learner', 'name', 'feat'])}


def tsum(v):
    """国家层单样本 t：均值、SD、90/95/97.5% 双侧 CI、单侧 p（H0：≤ 0）、>0 国家数。"""
    v = np.asarray(pd.Series(v, dtype=float).dropna(), dtype=float)
    n = len(v)
    out = dict(n=n, mean=float(v.mean()) if n else np.nan, sd=np.nan, t=np.nan, p_one=np.nan,
               n_pos=int((v > 0).sum()))
    for tag in ('90', '95', '975'):
        out[f'lo{tag}'] = out[f'hi{tag}'] = np.nan
    if n >= 2:
        m, sd = v.mean(), v.std(ddof=1)
        out.update(mean=float(m), sd=float(sd))
        for lev, tag in ((0.90, '90'), (0.95, '95'), (0.975, '975')):
            h = stats.t.ppf(1 - (1 - lev) / 2, n - 1) * sd / np.sqrt(n)
            out[f'lo{tag}'], out[f'hi{tag}'] = m - h, m + h
        if sd > 0:
            t = m / (sd / np.sqrt(n))
            out.update(t=float(t), p_one=float(stats.t.sf(t, n - 1)))
    return out


def b1_reading(lo, hi, mean):
    """§5 B1 读法，按顺序取第一条成立的。"""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return '不可计算'
    if lo > -1 and hi < 1:
        return '补采位置的影响小于 1 pp'
    if lo > 0 or hi < 0:
        return f'方向{"为正" if lo > 0 else "为负"}，大小 {mean:+.2f} pp'
    return f'未能区分，区间为 [{lo:.2f}, {hi:.2f}]'


def estimates(CVT, mode):
    rows = []
    for eid, fam, lrn, name, kind, prim in est_specs(mode):
        for f in KIND_FEATS[kind]:
            s = CVT[(lrn, name, f)]
            for cs in ('9', '11'):
                st = tsum(s.reindex(list(SETS[cs])))
                row = dict(est_id=eid, family=fam, learner=lrn, quantity=name, feat=f, set=cs,
                           primary=bool(cs == prim and f in ('both', 'ae-s2')), **st,
                           decidable=bool(st['n'] >= MIN_DECIDE[cs]))
                if fam == 'B1':   # §3：不满足可判定条件时记"不可判定"
                    row['reading'] = b1_reading(st['lo95'], st['hi95'], st['mean']) if row['decidable'] else '不可判定'
                rows.append(row)
    return pd.DataFrame(rows)


def est_row(E, eid, feat, cs):
    x = E[(E.est_id == eid) & (E.feat == feat) & (E.set == cs)]
    assert len(x) == 1, (eid, feat, cs, len(x))
    return x.iloc[0]


def holm(pv, alpha=0.025):
    """pv：{名: 单侧 p}（NaN = 不可判定，按 p = 1）。Holm：从小到大依次在 α/(m−i) 下检验，一旦不拒绝即停止。"""
    items = sorted(pv.items(), key=lambda kv: kv[1] if np.isfinite(kv[1]) else 1.0)
    out, go, m = {}, True, len(items)
    for i, (h, p) in enumerate(items):
        a = alpha / (m - i)
        rej = bool(go and np.isfinite(p) and p <= a)
        out[h] = dict(p_one=p, step=i + 1, alpha_step=a, reject=rej)
        go = rej
    return out


def tests_table(E):
    H = {'H1': est_row(E, 'H1', 'both', '9'), 'H2': est_row(E, 'H2', 'ae-s2', '11')}
    pv = {h: (r.p_one if r.decidable else np.nan) for h, r in H.items()}
    res = holm(pv)
    rows = []
    for h, r in H.items():
        rows.append(dict(hyp=h, quantity=r.quantity, feat=r.feat, set=r.set, n=r.n, decidable=r.decidable,
                         mean=r['mean'], sd=r.sd, lo95=r.lo95, hi95=r.hi95, lo975=r.lo975, hi975=r.hi975, t=r.t,
                         df=r.n - 1, **res[h]))
    return pd.DataFrame(rows)


def a2_ratio(E):
    """A2：ΔFA(RT-k) 占 ΔFA(TGT) 的比例（国家均值之比）；只在分母 95% CI 下界 > 0 时报告。"""
    rows = []
    for cs in ('9', '11'):
        d = est_row(E, 'A2_TGT', 'both', cs)
        for k in KS:
            nmr = est_row(E, f'A1_RT{k}', 'both', cs)
            ok = bool(np.isfinite(d.lo95) and d.lo95 > 0)
            rows.append(dict(k=k, set=cs, primary=cs == '9', num=nmr['mean'], den=d['mean'], den_lo95=d.lo95,
                             reported=ok, ratio=nmr['mean'] / d['mean'] if ok else np.nan))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- X2、X3
def pav(y):
    """等权保序回归（单调不减），池相邻违例者。"""
    blocks = []  # [均值, 个数]
    for v in np.asarray(y, dtype=float):
        blocks.append([v, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            (m1, n1), (m2, n2) = blocks[-2], blocks[-1]
            blocks[-2:] = [[(m1 * n1 + m2 * n2) / (n1 + n2), n1 + n2]]
    return np.concatenate([np.full(n, m) for m, n in blocks])


def k_eq(v, kn, fv):
    """单调不减折线（kn, fv）上达到 v 的最小 k；v 高于末档 → +inf（写 "> 末档"），低于首档 → −inf（写 "< 首档"）；不外推。"""
    if not np.isfinite(v):
        return np.nan
    if v > fv[-1]:
        return np.inf
    if v < fv[0]:
        return -np.inf
    j = int(np.flatnonzero(fv >= v)[0])
    if j == 0:
        return float(kn[0])
    return float(kn[j - 1] + (v - fv[j - 1]) / (fv[j] - fv[j - 1]) * (kn[j] - kn[j - 1]))


def fmt_k(k, lo=0, hi=400):
    return f'> {hi}' if k == np.inf else f'< {lo}' if k == -np.inf else ('—' if not np.isfinite(k) else f'{k:.0f}')


def fmt_rate(v, n, hi=400):
    """交换率 = k_eq ÷ n：k_eq > 400 时写 "> 400/n"，k_eq < 0 时写 "< 0"，不外推。"""
    return f'> {hi / n:.2f}' if v == np.inf else '< 0' if v == -np.inf else ('—' if not np.isfinite(v) else f'{v:.2f}')


def x3_fmt(kind, item, v):
    """X3 各项的文字写法。"""
    if kind.endswith('keq'):
        return fmt_k(v)
    if kind.endswith('rate'):
        return fmt_rate(v, item if kind == 'conc_rate' else N_TOTAL[item])
    return f'{v:+.3f}' if np.isfinite(v) else '—'


X3_DISTINCT = '可区分，报交换率'


def x3_point(m):
    """m：{量名: 国家均值}。返回 (派生量 dict, 随机样本曲线 PAV 值, 集中样本曲线 PAV 值)。"""
    fr = pav([0.0] + [m[f'dFA_RT{k}'] for k in KS])
    fc = pav([0.0, m['dFA_B_S0'], m['dFA_B_EXT100'], m['dFA_B_EXT300']])
    out = {}
    for a in B_ARMS:
        k = k_eq(m[f'dFA_{a}'], KN_R, fr)
        out[('keq', a)], out[('rate', a)] = k, k / N_TOTAL[a]
    for j, n in enumerate(KN_C[1:], start=1):
        k = k_eq(fc[j], KN_R, fr)
        out[('conc_diff', int(n))] = fc[j] - np.interp(n, KN_R, fr)
        out[('conc_keq', int(n))], out[('conc_rate', int(n))] = k, k / n
    return out, fr, fc


def q_interval(x, level=0.90):
    """自助分位区间（逆经验分布，不插值，可含 ±inf）。"""
    s = np.sort(np.asarray(x, dtype=float)[~np.isnan(np.asarray(x, dtype=float))])
    nb = len(s)
    if nb == 0:
        return np.nan, np.nan
    a = (1 - level) / 2
    return s[int(np.ceil(round(a * nb, 6))) - 1], s[int(np.ceil(round((1 - a) * nb, 6))) - 1]


def interp_noextrap(x, xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    return float(np.interp(x, xs, ys)) if xs[0] - 1e-9 <= x <= xs[-1] + 1e-9 else np.nan


def x_blocks(CVT):
    """X2 定位、X3 交换率与成本效益（both，9 国；只用全部输入量都有值的国家，不足 7 国记"不可判定"）。"""
    names = [f'dFA_RT{k}' for k in KS] + [f'dFA_{a}' for a in B_ARMS]
    M = pd.concat({n: CVT[('lgb', n, 'both')].reindex(list(NINE)) for n in names}, axis=1).dropna()
    nC = len(M)
    decid = nC >= MIN_DECIDE['9']      # §3 可判定条件，X2、X3 同样适用
    who = ' '.join(M.index)
    m = M.mean().to_dict()
    # X2：B 配置的 ΔFA 在场景 A 剂量曲线（国家均值，k = 0 为 0，未单调化）上落在哪两档之间
    ka = [0] + list(KS)
    va = [0.0] + [m[f'dFA_RT{k}'] for k in KS]
    x2 = []
    for a in B_ARMS:
        v = m[f'dFA_{a}']
        br = [f'RT{ka[i]}–RT{ka[i + 1]}'.replace('RT0', 'Z0') for i in range(len(ka) - 1)
              if min(va[i], va[i + 1]) <= v <= max(va[i], va[i + 1])]
        pos = '；'.join(br) if br else ('高于全部档位' if v > max(va) else '低于全部档位')
        x2.append(dict(config=a, n_labels=N_TOTAL[a], value=v, position=pos, n_countries=nC, countries=who,
                       decidable=decid))
    # X3：点估计与国家层自助
    pt, fr, fc = x3_point(m)
    rng = np.random.default_rng(G.stable_hash(X3_SEED) % 2 ** 31)
    V = M.to_numpy()
    boot = ([x3_point(dict(zip(M.columns, V[i].mean(0))))[0] for i in rng.integers(0, nC, size=(N_BOOT, nC))]
            if nC else [])
    rows = []
    for key, val in pt.items():
        lo, hi = q_interval([b[key] for b in boot])
        rows.append(dict(kind=key[0], item=key[1], point=val, lo90=lo, hi90=hi))
    X3 = pd.DataFrame(rows)
    X3['text'] = [x3_fmt(k, i, p) for k, i, p in zip(X3.kind, X3['item'], X3.point)]
    # 读法：集中样本曲线与随机样本曲线之差的 90% 区间覆盖 0 → 未能区分，不报交换率
    for n in KN_C[1:].astype(int):
        d = X3[(X3.kind == 'conc_diff') & (X3.item == n)].iloc[0]
        X3.loc[(X3.kind.isin(['conc_diff', 'conc_keq', 'conc_rate'])) & (X3.item == n), 'reading'] = (
            f'不可判定（全部输入有值的国家 {nC} < {MIN_DECIDE["9"]}）' if not decid else
            '不可计算' if not (np.isfinite(d.lo90) and np.isfinite(d.hi90)) else
            '集中与分散样本的价值未能区分，选便宜的方式（不报交换率）' if d.lo90 <= 0 <= d.hi90 else X3_DISTINCT)
    hide = X3.kind.isin(['conc_keq', 'conc_rate']) & (X3.reading != X3_DISTINCT)
    X3.loc[hide, ['point', 'lo90', 'hi90']] = np.nan
    X3.loc[hide, 'text'] = '不报'
    X3 = pd.concat([X3, pd.DataFrame([dict(kind='curve_random', item=int(k), point=v) for k, v in zip(KN_R, fr)] +
                                     [dict(kind='curve_conc', item=int(k), point=v) for k, v in zip(KN_C, fc)])],
                   ignore_index=True)
    X3['n_countries'], X3['countries'], X3['decidable'] = nC, who, decid
    # 成本效益（点估计；c = 分散样本单价 ÷ 集中样本单价）
    cost = []
    for c in COSTS:
        for a, (nc, nd) in COMPOSITION.items():
            K = nc + c * nd
            v = 0.0 if a == 'Z0' else m[f'dFA_{a}']
            rs = interp_noextrap(K / c, KN_R, fr)
            cost.append(dict(kind='config', c=c, name=a, n_conc=nc, n_disp=nd, cost=K, value=v,
                             random_same_cost=rs, minus_random=v - rs))
        fam = {'random': (c * KN_R, fr), 'concentrated': (KN_C, fc),
               'S0+SRS': ([100, 100 + 100 * c, 100 + 300 * c], [m['dFA_B_S0'], m['dFA_B_SRS100'], m['dFA_B_SRS300']]),
               'S0+BA': ([100, 100 + 100 * c, 100 + 300 * c], [m['dFA_B_S0'], m['dFA_B_BA100'], m['dFA_B_BA300']])}
        for K in BUDGETS:
            vals = {nm: interp_noextrap(K, xs, ys) for nm, (xs, ys) in fam.items()}
            fin = [v for v in vals.values() if np.isfinite(v)]
            top = max(fin) if fin else np.nan
            cost += [dict(kind='budget', c=c, name=nm, cost=K, value=v, best=bool(np.isfinite(v) and v == top))
                     for nm, v in vals.items()]
    return pd.DataFrame(x2), X3, pd.DataFrame(cost)


# ---------------------------------------------------------------- §10 填充表
def fill_table(E, T, X3):
    rows = []

    def add(sentence, variant, ph, value, text, applies, note=''):
        rows.append(dict(sentence=sentence, variant=variant, placeholder=ph, value=value, text=text,
                         applies=bool(applies), note=note))

    def f2(v):
        return f'{v:+.2f}' if np.isfinite(v) else '—'

    t = T.set_index('hyp')
    h1, tg = est_row(E, 'H1', 'both', '9'), est_row(E, 'A2_TGT', 'both', '9')
    rho_ok = bool(np.isfinite(tg.lo95) and tg.lo95 > 0)
    rho = h1['mean'] / tg['mean'] if rho_ok else np.nan
    for s, ap in (('H1', t.loc['H1', 'reject']), ('H1_fail', not t.loc['H1', 'reject'])):
        add(s, '', 'x', h1['mean'], f2(h1['mean']), ap)
        add(s, '', 'L', h1.lo95, f2(h1.lo95), ap)
        add(s, '', 'U', h1.hi95, f2(h1.hi95), ap)
    add('H1', '', 'ρ', rho, f'{100 * rho:.0f}%' if rho_ok else '不报告（ΔFA(TGT) 的 95% CI 下界 ≤ 0）',
        t.loc['H1', 'reject'], 'ΔFA(RT200) ÷ ΔFA(TGT)，both，9 国')
    for k1 in K1S:
        y = est_row(E, f'B1_SRS-EXT{k1}', 'both', '9')
        for s in STRATS:
            v = f'k1={k1}/{s}'
            z = est_row(E, f'X1_{s}{k1}-RT{100 + k1}', 'both', '9')
            ap = s == 'SRS'
            verb = '—' if not np.isfinite(z.hi95) else 'fell short of' if z.hi95 < 0 else 'matched'
            nt = '95% CI 整体 > 0：模板无"超过"选项，需作者决定措辞' if z.lo95 > 0 else ''
            add('B', v, 'k1', k1, str(k1), ap)
            add('B', v, 'y', y['mean'], f2(y['mean']), ap, 'FA(S0+SRS) − FA(S0+EXT)')
            add('B', v, 'y_L', y.lo95, f2(y.lo95), ap)
            add('B', v, 'y_U', y.hi95, f2(y.hi95), ap)
            add('B', v, 'matched/fell short of', np.nan, verb, ap, nt or '95% CI 上界 < 0 → fell short of；否则 matched')
            add('B', v, '100+k1', 100 + k1, str(100 + k1), ap)
            add('B', v, 'z', z['mean'], f2(z['mean']), ap, f'FA(S0+{s}{k1}) − FA(RT{100 + k1})；SRS 为主句，EXT、BA 备选')
            add('B', v, 'z_L', z.lo95, f2(z.lo95), ap)
            add('B', v, 'z_U', z.hi95, f2(z.hi95), ap)
    for n in KN_C[1:].astype(int):
        d = X3[(X3.kind == 'conc_diff') & (X3.item == n)].iloc[0]
        r = X3[(X3.kind == 'conc_rate') & (X3.item == n)].iloc[0]
        ap = d.reading == X3_DISTINCT
        v = f'集中样本 {n} 个'
        nt = ('只在集中与随机样本曲线之差的 90% 区间不覆盖 0（且可判定）时写；n=100 即 S0 only（已有调查本身）；'
              '交换率 = k_eq ÷ n，k_eq 超出 [0, 400] 时写 "> 400/n" 或 "< 0"')
        for ph, x in (('n', r.point), ('L', r.lo90), ('U', r.hi90)):
            add('X3', v, ph, x if ap and np.isfinite(x) else np.nan, fmt_rate(x, n) if ap else '不报', ap,
                nt if ph == 'n' else '')
    h2 = est_row(E, 'H2', 'ae-s2', '11')
    e_rt, e_b = est_row(E, 'E1_D_RT200', 'ae-s2', '11'), est_row(E, 'E1_D_S0+SRS100', 'ae-s2', '11')
    pers = [bool(np.isfinite(e.lo95) and e.lo95 > 0) for e in (e_rt, e_b)]
    verb = 'persisted' if all(pers) else 'was not detected' if not any(pers) else (
        f'混合：RT200 {"persisted" if pers[0] else "not detected"}，S0+SRS100 {"persisted" if pers[1] else "not detected"}'
        '（需作者拆句）')
    for s, ap in (('H2', t.loc['H2', 'reject']), ('H2_fail', not t.loc['H2', 'reject'])):
        add(s, '', 'd', h2['mean'], f2(h2['mean']), ap)
        add(s, '', 'L', h2.lo95, f2(h2.lo95), ap)
        add(s, '', 'U', h2.hi95, f2(h2.hi95), ap)
    add('H2', '', 'persisted/was not detected', np.nan, verb, t.loc['H2', 'reject'],
        'E1：D(RT200) 与 D(S0+SRS100)，11 国，95% CI 下界 > 0 为 persisted')
    return pd.DataFrame(rows)


FILL_KEYS = ([('H1', '', p) for p in ('x', 'L', 'U', 'ρ')] + [('H1_fail', '', p) for p in ('x', 'L', 'U')] +
             [('B', f'k1={k1}/{s}', p) for k1 in K1S for s in STRATS
              for p in ('k1', 'y', 'y_L', 'y_U', 'matched/fell short of', '100+k1', 'z', 'z_L', 'z_U')] +
             [('X3', f'集中样本 {n} 个', p) for n in (100, 200, 400) for p in ('n', 'L', 'U')] +
             [('H2', '', p) for p in ('d', 'L', 'U', 'persisted/was not detected')] +
             [('H2_fail', '', p) for p in ('d', 'L', 'U')])


# ---------------------------------------------------------------- 主流程
def analyze(root, y, mode, R, log=print, units=None, strict=True):
    """第 1 遍：全部单元的完整性检查（不读概率）；全部通过后，第 2 遍读概率、算逐单元量与汇总。
    strict = False 只用于 --structure-check（缺的拟合记 absent，由 structure_check 计为问题）。"""
    full, units, excluded, info = units is None, units or expected_units(), {}, {}
    M = read_manifest(root) if mode == 'formal' else None
    assert not (full and mode == 'formal') or sorted(M.index) == sorted(units), 'split_manifest.csv 的单元不是全部 22 个'
    for unit in units:
        d, why = check_unit(root, unit, y, mode, R, M, strict)
        if d is None:
            excluded[unit] = why
        else:
            info[unit] = d
    codes = sorted(set().union(*(d['code_hash'] for d in info.values())))
    if mode == 'formal':
        assert codes == [runner_code_hash()], \
            f'runs 的 code_hash {codes} ≠ 当前 confirm_split.py + confirm_run.py 的 {runner_code_hash()}'
    UR, status = [], {}
    for unit, d in info.items():
        status.update({(unit,) + k: v for k, v in d['status'].items()})
        UR += unit_rows(unit, unit_curves(d['yT'], d['big'], load_probs(root, d)), mode, R)
        log(f'{unit} 完成（n_test {d["n_test"]}）')
    U = pd.DataFrame(UR)
    CT = country_table(U)
    CVT = cv_lookup(CT)
    E = estimates(CVT, mode)
    out = dict(unit_rows=U, country_values=CT, estimates=E, excluded=excluded, status=status, codes=codes)
    if mode == 'retro':
        return out
    T = tests_table(E)
    x2, X3, cost = x_blocks(CVT)
    out.update(tests=T, a2_ratio=a2_ratio(E), x2_position=x2, x3_equiv=X3, x3_cost=cost, fill_table=fill_table(E, T, X3))
    return out


def structure_check(res, mode, R):
    """只检查结构，不含任何指标值。返回 (文本行, 问题数)。"""
    lines, bad = [], 0
    P = lines.append
    U = res['unit_rows']
    exp = expected_counts(mode, R)
    P(f'模式 {mode}，R = {R}；单元 {U.unit.nunique()} 个纳入，{len(res["excluded"])} 个剔除：'
      + ('、'.join(f'{u}（{w}）' for u, w in res['excluded'].items()) or '无'))
    cnt = U.groupby(['unit', 'level']).size().unstack('level').reindex(columns=list(exp), fill_value=0)
    wrong = cnt[(cnt != pd.Series(exp)).any(axis=1)]
    bad += len(wrong)
    P(f'每单元行数应为 {exp}：{"全部相符" if wrong.empty else f"不符单元 {list(wrong.index)}"}')
    P(f'逐单元行键唯一：{not U.duplicated(["level", "learner", "unit", "r", "feat", "name"]).any()}')
    bad += int(U.duplicated(['level', 'learner', 'unit', 'r', 'feat', 'name']).any())
    st = pd.Series(res['status'], dtype=object)
    n_abs = int((st == 'absent').sum())
    bad += n_abs
    P(f'拟合状态计数：{st.value_counts().to_dict()}（absent = 计划中的拟合在 runs 中没有，{n_abs} 个，计为问题）；'
      f'code_hash {res["codes"]}')
    if mode == 'formal':
        P(f'逐单元断言已通过：runs 键 ⊂ 拟合计划（每单元 lgb {N_FITS["lgb"]}、rf {N_FITS["rf"]}）且不重复，status ∈ {{ok, failed}}；'
          'sel_hash、train_sha、n_train 与选样清单一致；划分与选样内容哈希 = split_manifest.csv = runs；'
          f'可行性与重算一致；线程 {THREADS}；code_hash 唯一且等于当前运行器代码；概率 float32、有限、非负、行和为 1')
    else:
        P(f'缺失值计数（按层级）：{U.groupby("level").value.apply(lambda s: int(s.isna().sum())).to_dict()}')
    need = {'unit_rows': ['level', 'learner', 'country', 'half', 'unit', 'r', 'feat', 'name', 'arm_a', 'arm_b',
                          'OA_star', 'value', 'cov', 'missing'],
            'country_values': ['learner', 'name', 'feat', 'country', 'value', 'n_half'],
            'estimates': ['est_id', 'family', 'learner', 'quantity', 'feat', 'set', 'primary', 'n', 'mean', 'sd',
                          'lo90', 'hi90', 'lo95', 'hi95', 'lo975', 'hi975', 't', 'p_one', 'n_pos', 'decidable']}
    if mode != 'retro':
        need.update(tests=['hyp', 'n', 'mean', 'lo95', 'hi95', 'lo975', 'hi975', 'p_one', 'alpha_step', 'reject'],
                    a2_ratio=['k', 'set', 'ratio', 'reported'], x2_position=['config', 'value', 'position'],
                    x3_equiv=['kind', 'item', 'point', 'lo90', 'hi90'], x3_cost=['kind', 'c', 'name', 'cost', 'value'],
                    fill_table=['sentence', 'variant', 'placeholder', 'value', 'text', 'applies'])
    for k, cols in need.items():
        miss = [c for c in cols if c not in res[k].columns]
        bad += len(miss)
        P(f'{k}：{len(res[k])} 行，缺列 {miss or "无"}')
    E = res['estimates']
    n_exp = sum(len(KIND_FEATS[s[4]]) * 2 for s in est_specs(mode))
    bad += int(len(E) != n_exp)
    P(f'汇总行 {len(E)}（应 {n_exp}）' + ('' if mode == 'formal' else
                                      f'；可判定的主集行 {int(E[E.primary].decidable.sum())}/{int(E.primary.sum())}'))
    if mode != 'retro':
        F = res['fill_table']
        have = set(zip(F.sentence, F.variant, F.placeholder))
        lack = [k for k in FILL_KEYS if k not in have]
        bad += len(lack) + int(len(F) != len(FILL_KEYS))
        P(f'填充表占位符 {len(F)} 个（应 {len(FILL_KEYS)}），缺 {lack or "无"}（哪句适用取决于结果，不打印）')
        P(f'X3 自助次数 {N_BOOT}，项目 {len(res["x3_equiv"])} 行；成本表 {len(res["x3_cost"])} 行')
    P(f'结构检查：{"通过" if bad == 0 else f"{bad} 个问题"}')
    return lines, bad


# ---------------------------------------------------------------- 回溯适配器（G s0/s1 → 统一接口）
def build_retro(root, y):
    """F{split+1} ← G 划分 s{split}；r ← 重复号 rep。Z0 = stage0 的 P0test_{f}_i300；RT-k = G 的 SRS(k) i300；
    RF Z0 = guidance_G_ml k0，RF RT200 = SRS k=200（与 G 的 SRS k=200 同一批选样）。
    删除保护：root 须在 scratch 之下；已有 splits/probs/runs 时须有本函数写的标记文件，否则拒绝（不删 G 的数据源）。"""
    assert_scratch(root)
    mark, subs = os.path.join(root, RETRO_MARK), [os.path.join(root, s) for s in ('splits', 'probs', 'runs')]
    assert os.path.exists(mark) or not any(os.path.exists(x) for x in subs), \
        f'{root} 已有 splits/probs/runs 但没有标记文件 {RETRO_MARK}：不是本脚本生成的回溯目录，拒绝删除'
    os.makedirs(root, exist_ok=True)
    open(mark, 'w', encoding='utf-8').write('analyze_confirm.py --retro 生成的回溯目录；--retro 会删除并重建 splits、probs、runs\n')
    for x in subs:
        shutil.rmtree(x, ignore_errors=True)
        os.makedirs(x)
    GR = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(R_G, 'rows_t1_*.csv')))], ignore_index=True)
    GR = GR[(GR.cond == 'none') & (GR.method == 'srs')]
    MR = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(R_ML, 'rows_rf_*.csv')))], ignore_index=True)
    MR = MR[(MR.learner == 'rf') & (MR.cond == 'none') & (MR.method == 'srs') & (MR.k == 200)]
    assert (MR.status == 'ok').all()
    empty = np.zeros((0, C), np.float16)
    n_hash = 0
    for tgt in COUNTRIES:
        for split in (0, 1):
            unit = f'{tgt}_F{split + 1}'
            S = np.load(os.path.join(R_G, 'stage0', f'{tgt}_s{split}.npz'))
            test, allT = S['test'], S['allT']
            assert np.array_equal(allT[len(S['cand']):], test)
            big = np.flatnonzero(np.bincount(np.asarray(y)[test], minlength=C) >= G.MIN_N)
            np.savez(os.path.join(root, 'splits', f'{unit}.npz'), test=test, cand=S['cand'], foreign=S['foreign'],
                     big=big, lat0=S['lat0'], note=f'retro: guidance_G stage0 {tgt}_s{split}')
            pdir = os.path.join(root, 'probs', unit)
            os.makedirs(pdir)
            runs = []

            def put(lrn, arm, f, r, Pt, k, h, t_fit, src):
                Pt = np.asarray(Pt)
                assert Pt.dtype in (np.float16, np.float32)  # float16/32 → float32 无损
                np.savez(os.path.join(pdir, f'{lrn}_{arm}_{f}_r{r}.npz'), P_test=Pt.astype(np.float32), P_pool=empty)
                runs.append(dict(learner=lrn, arm=arm, feat=f, r=r, k=k, sel_hash=h, seed=np.nan, n_train=50000 + k,
                                 t_fit=t_fit, status='ok', code_hash=src))

            for f in FEATS:
                put('lgb', 'Z0', f, 0, S[f'P0test_{f}_i300'], 0, '', np.nan, 'retro:guidance_G/stage0')
                put('rf', 'Z0', f, 0, np.load(os.path.join(R_ML, 'probs', 'rf', f'{tgt}_s{split}_{f}_k0.npz'))['P'],
                    0, '', np.nan, 'retro:guidance_G_ml/rf_k0')
                for rep in (0, 1):
                    gu = f'{tgt}_s{split}_r{rep}'
                    pool_pos = G.pool_positions(S, split, rep)
                    hk = {}
                    for k in KS:
                        z = np.load(os.path.join(R_G, 'sel', f'{gu}_none_srs_k{k}.npz'))
                        assert np.array_equal(allT[pool_pos][z['sel_pos']], z['sel']) and len(z['sel']) == k
                        g = GR[(GR.target == tgt) & (GR.split == split) & (GR.rep == rep) & (GR.clf == f) & (GR.k == k)]
                        assert len(g) == 1 and g.iloc[0].sel_hash == G.sel_hash(z['sel']), (gu, f, k)
                        hk[k] = g.iloc[0].sel_hash
                        n_hash += 1
                        put('lgb', f'RT{k}', f, rep,
                            np.load(os.path.join(R_G, 'probs', f'{gu}_{f}_k{k}_{hk[k]}.npz'))['i300'], k, hk[k],
                            g.iloc[0].t_train, 'retro:guidance_G/probs_i300')
                    m2 = MR[(MR.target == tgt) & (MR.split == split) & (MR.rep == rep) & (MR.clf == f)]
                    assert len(m2) == 1 and m2.iloc[0].sel_hash == hk[200], (gu, f, 'rf')
                    n_hash += 1
                    put('rf', 'RT200', f, rep,
                        np.load(os.path.join(R_ML, 'probs', 'rf', f'{gu}_{f}_k200_{hk[200]}.npz'))['P'], 200, hk[200],
                        m2.iloc[0].t_fit, 'retro:guidance_G_ml/rf_srs_k200')
            pd.DataFrame(runs)[RUN_COLS].to_csv(os.path.join(root, 'runs', f'{unit}.csv'), index=False)
    return n_hash


def reconcile_pc1(U, E):
    """与 PC-1 逐单元核对（arm、diff、D 行），并核对 §6 登记的三个数字。返回 (文本行, 是否全部通过)。"""
    lines, ok_all = [], True
    P = lines.append
    R1 = pd.read_csv(os.path.join(PC1_DIR, 'pc1_rows.csv'), float_precision='round_trip')
    names = {('arm', 'lgb'): ['Z0'] + RT_ARMS, ('arm', 'rf'): ['Z0', 'RT200'],
             ('diff', 'lgb'): [f'dFA_RT{k}' for k in KS], ('diff', 'rf'): ['dFA_RT200'],
             ('D', 'lgb'): ['D_Z0', 'D_RT200'], ('D', 'rf'): ['D_Z0']}
    for (lev, lrn), nm in names.items():
        x = R1[(R1.level == lev) & (R1.learner == lrn) & R1.name.isin(nm)].copy()
        x['unitF'] = x.target + '_F' + (x.split + 1).astype(str)
        x['rr'] = np.where(x.name.isin(['Z0', 'D_Z0']), 0, x.rep)
        mine = U[(U.level == lev) & (U.learner == lrn) & U.name.isin(nm)]
        j = x.merge(mine, left_on=['unitF', 'rr', 'clf', 'name'], right_on=['unit', 'r', 'feat', 'name'],
                    how='left', suffixes=('_pc1', ''), indicator=True)
        found = bool((j._merge == 'both').all())
        cols = ['value', 'cov', 'OA_star'] + (['oa_t0', 'ba_t0'] if lev == 'arm' else [])
        dmax, nan_ok = 0.0, True
        for c in cols:
            a, b = j[f'{c}_pc1'].to_numpy(float), j[c].to_numpy(float)
            nan_ok &= bool(np.array_equal(np.isnan(a), np.isnan(b)))
            if np.isfinite(a).any():
                dmax = max(dmax, float(np.nanmax(np.abs(a - b))))
        ok = found and nan_ok and dmax <= TOL
        ok_all &= ok
        P(f'  {lev:4s} {lrn:3s} {",".join(nm)}：PC-1 {len(x)} 行，全部找到 {found}，NaN 位置一致 {nan_ok}，'
          f'最大 |Δ|（{"/".join(cols)}）{dmax:.2e} → {"通过" if ok else "不通过"}')
    # 登记数字：用 PC-1 的口径（国家值 = 4 个单元平扁平均）从 pc1_rows 重算，与本脚本（分层平均）比较
    DF, DD = R1[R1.level == 'diff'], R1[R1.level == 'D']

    def flat_both(lrn, name):
        x = DF[(DF.learner == lrn) & (DF.name == name)]
        cv = x.groupby(['clf', 'target']).value.mean().unstack('clf')
        return cv[list(FEATS)].mean(axis=1, skipna=False)

    ref = {'H1': (flat_both('lgb', 'dFA_RT200'), '9', 'both'),
           'H2': (DD[(DD.learner == 'lgb') & (DD.name == 'D_Z0')].groupby('target').value.mean(), '11', 'ae-s2'),
           'A3_RF_RT200': (flat_both('rf', 'dFA_RT200'), '9', 'both')}
    for eid, (cv, cs, feat) in ref.items():
        s0 = tsum(cv.reindex(list(SETS[cs])))
        s1 = est_row(E, eid, feat, cs)
        d = max(abs(s0[k] - s1[k]) for k in ('mean', 'lo95', 'hi95'))
        txt = f'{s1["mean"]:+.2f} [{s1.lo95:+.2f}, {s1.hi95:+.2f}]'
        ok = d <= TOL and txt == PC1_REGISTERED[eid] and s1.n == s0['n']
        ok_all &= ok
        P(f'  {eid:12s} {cs} 国：本脚本 {txt}（n={s1.n}），PC-1 登记 {PC1_REGISTERED[eid]}，'
          f'与 pc1_rows 重算的最大 |Δ|（均值/CI） {d:.2e} → {"通过" if ok else "不通过"}')
    return lines, ok_all


# ---------------------------------------------------------------- 输出
def fmt_est(r, lev='95'):
    if r.n < 2:
        return f'n={r.n}（不足 2 国）'
    return f'{r["mean"]:+.2f} [{r["lo" + lev]:+.2f}, {r["hi" + lev]:+.2f}]，>0 {r.n_pos}/{r.n}' + (
        '' if r.decidable else '（不可判定）')


def summary_lines(res, mode, R):
    lines = []
    P = lines.append
    E = res['estimates']
    P(f'确认运行分析（方案 CONFIRM_PLAN_v5.md sha256[:16]={sha256(PLAN)[:16]}；analyze_confirm.py '
      f'sha256[:16]={sha256(SELF)[:16]}）。模式 {mode}，R = {R}。')
    P('复用（未改）：' + '、'.join(f'{os.path.basename(p)} {h[:16]}' for p, h in FROZEN.items() if p != PLAN))
    P(f'纳入单元 {res["unit_rows"].unit.nunique()}；剔除 ' +
      ('、'.join(f'{u}（{w}）' for u, w in res['excluded'].items()) or '无'))
    st = pd.Series(res['status'])
    bad = st[st != 'ok']
    P(f'拟合状态：{st.value_counts().to_dict()}' + (f'；非 ok：{list(bad.index)}' if len(bad) else ''))
    P('单位 pp；格式：均值 [CI]，>0 国家数 / 有值国家数；9 国 = 去掉 RO、PT。')
    P('')
    if mode != 'retro':
        P('== 确认性检验（Holm，族 α = 0.025 单侧）==')
        for _, t in res['tests'].iterrows():
            P(f'  {t.hyp} {t.quantity}（{t.feat}，{t.set} 国）：{t["mean"]:+.2f}，95% CI [{t.lo95:+.2f}, {t.hi95:+.2f}]，'
              f'97.5% CI [{t.lo975:+.2f}, {t.hi975:+.2f}]，t = {t.t:.3f}，df = {t.df}，单侧 p = {t.p_one:.3g}，'
              f'Holm 第 {t.step} 步 α = {t.alpha_step:g} → {"拒绝 H0" if t.reject else "不拒绝"}'
              + ('' if t.decidable else '（不可判定）'))
        P('')
    for fam in ('H1', 'H2', 'A1', 'A2', 'A3', 'B1', 'B2', 'X1', 'X2', 'E1', 'E2', 'M1'):
        x = E[E.family == fam]
        if x.empty:
            continue
        P(f'== {fam} ==')
        lev = '90' if fam == 'E2' else '95'
        for eid in dict.fromkeys(x.est_id):
            y = x[x.est_id == eid]
            order = {'both': 0, 'ae-s2': 0, 'inter': 0, 'ae': 1, 's2': 2}
            y = y.iloc[sorted(range(len(y)), key=lambda i: (not y.primary.iloc[i], order[y.feat.iloc[i]],
                                                            y.set.iloc[i] != '9'))]
            for _, r in y.iterrows():
                P(f'  {eid:22s} {r.feat:6s} {r.set:>2s} 国{"（主）" if r.primary else "     "} {fmt_est(r, lev)}'
                  + (f'；97.5% CI [{r.lo975:+.2f}, {r.hi975:+.2f}]' if fam in ('H1', 'H2') and r.primary else '')
                  + (f'；读法：{r.reading}' if fam == 'B1' and r.feat == 'both' else '')
                  + ('（90% CI）' if fam == 'E2' else ''))
        P('')
    if mode != 'retro':
        P('== A2 比例（ΔFA(RT-k) ÷ ΔFA(TGT)，both）==')
        for _, r in res['a2_ratio'].iterrows():
            P(f'  k={r.k:3d} {r.set:>2s} 国：' + (f'{r.ratio:.3f}' if r.reported else '不报告（分母 95% CI 下界 ≤ 0）'))
        X3 = res['x3_equiv']
        x0 = X3.iloc[0]
        cset = (f'9 国中全部输入有值的 {int(x0.n_countries)} 国（{x0.countries or "无"}）'
                + ('' if x0.decidable else f'，不足 {MIN_DECIDE["9"]} 国：不可判定'))
        P(f'== X2 定位（both；{cset}；场景 A 剂量曲线未单调化）==')
        for _, r in res['x2_position'].iterrows():
            P(f'  {r.config:10s}（{r.n_labels} 个标签）ΔFA {r.value:+.2f} → {r.position}')
        P(f'== X3（both；{cset}；PAV + 线性插值；国家层自助 {N_BOOT} 次，种子 {X3_SEED}，90% 区间）==')
        for _, r in X3.iterrows():
            if r.kind.startswith('curve'):
                P(f'  {r.kind} 档 {r["item"]}: {r.point:+.3f}')
            elif r.text == '不报':
                P(f'  {r.kind:10s} {str(r["item"]):9s} 不报（{r.reading}）')
            else:
                P(f'  {r.kind:10s} {str(r["item"]):9s} {r.text}，90% [{x3_fmt(r.kind, r["item"], r.lo90)}, '
                  f'{x3_fmt(r.kind, r["item"], r.hi90)}]'
                  + (f'；{r.reading}' if isinstance(r.get('reading'), str) and r.kind == 'conc_diff' else ''))
        P('== X3 成本效益（c = 分散样本单价 ÷ 集中样本单价；预算内各方案族按成本线性插值，不外推）==')
        cost = res['x3_cost']
        for c in COSTS:
            for K in BUDGETS:
                x = cost[(cost.kind == 'budget') & (cost.c == c) & (cost.cost == K)]
                P(f'  c={c} 预算 {K}：' + '；'.join(f'{r["name"]} {r.value:+.2f}{"★" if r.best else ""}'
                                                  if np.isfinite(r.value) else f'{r["name"]} —' for _, r in x.iterrows()))
        P('== §10 填充表（applies = 按结局适用）==')
        for _, r in res['fill_table'].iterrows():
            P(f'  [{r.sentence}{"/" + r.variant if r.variant else ""}] ⟦{r.placeholder}⟧ = {r.text}'
              f'{"" if r.applies else "（不适用）"}{("；" + r.note) if r.note else ""}')
    return lines


def write_outputs(res, out, mode, extra_lines=()):
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, ANALYSIS_MARK), 'w', encoding='utf-8').write('analyze_confirm.py 的输出目录\n')
    keys = ['unit_rows', 'country_values', 'estimates'] + ([] if mode == 'retro' else
                                                           ['tests', 'a2_ratio', 'x2_position', 'x3_equiv',
                                                            'x3_cost', 'fill_table'])
    paths = []
    for k in keys:
        p = os.path.join(out, f'{k}.csv')
        res[k].to_csv(p, index=False)
        paths.append(p)
    lines = summary_lines(res, mode, R_DRAWS if mode != 'retro' else 2) + list(extra_lines)
    p = os.path.join(out, 'summary.txt')
    open(p, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    paths.append(p)
    return lines, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--retro', action='store_true')
    ap.add_argument('--structure-check', action='store_true')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--units', default=None, help='只用于开发期的结构检查（如冒烟单元 AT_s0,AT_s1）；默认全部 22 个单元')
    a = ap.parse_args()
    assert a.units is None or a.structure_check, '--units 只用于 --structure-check'
    assert os.path.isabs(a.root), '--root 须为绝对路径（导入 step0B 时工作目录已切到 label_expansion）'
    root = os.path.realpath(a.root)
    if a.retro or a.overwrite:
        assert_scratch(root)          # 会删除目录：只允许开发 scratch（正式根目录、G 结果目录一律拒绝）
    mode = 'retro' if a.retro else 'formal'
    R = 2 if a.retro else R_DRAWS
    if mode == 'formal' and under(root, FORMAL_ROOT):
        check_registration(root, self_too=not a.structure_check)   # §7 第 4、6 步：清单与代码已登记；只运行冻结版本
    out = os.path.join(root, 'analysis')
    if not a.structure_check and not a.retro and os.path.isdir(out) and os.listdir(out) and not a.overwrite:
        raise SystemExit(f'{out} 已有输出；正式分析只运行一次')
    t0 = time.time()
    y = np.asarray(G.GData().y)
    if a.retro:
        n_hash = build_retro(root, y)
        print(f'回溯适配器：写入 {root}（sel_hash 断言 {n_hash} 次通过）', flush=True)
    res = analyze(root, y, mode, R, log=lambda s: print(f'[{time.time() - t0:.0f}s] {s}', flush=True),
                  units=a.units.split(',') if a.units else None, strict=not a.structure_check)
    if a.structure_check:
        lines, bad = structure_check(res, mode, R)
        print('\n'.join(lines))
        raise SystemExit(1 if bad else 0)
    extra = []
    if a.retro:
        rl, ok = reconcile_pc1(res['unit_rows'], res['estimates'])
        extra = ['', '== 回溯对账（PC-1，容差 1e-9）=='] + rl + [f'对账结论：{"全部通过" if ok else "有不通过项"}']
    if a.retro or a.overwrite:
        clear_own(out, ANALYSIS_MARK)
    lines, paths = write_outputs(res, out, mode, extra)
    print('\n'.join(lines))
    print('== 输出 sha256 ==')
    for p in paths:
        print(f'  {os.path.basename(p):20s} {sha256(p)}')
    print(f'总耗时 {time.time() - t0:.0f}s')
    if a.retro and not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
