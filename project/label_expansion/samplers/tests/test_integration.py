"""
选样方法的集成测试：一次导入 designs / uncertainty / coverage / transfer 四个文件，遍历 REGISTRY 中的全部方法。

运行（pytest 风格，也可直接运行）：
    OMP_NUM_THREADS=2 LGBM_THREADS=1 python samplers/tests/test_integration.py
环境变量：
    INTEG_SKIP_TIMING=1     跳过 N=30000、k=1000 的耗时测试（其余测试约 3–6 分钟）
    INTEG_TIMING_JSON=路径  把耗时表写成 JSON
    INTEG_TIME_LIMIT=秒     单个方法的耗时上限（默认 180）
    INTEG_ONLY=a,b,c        只测指定的注册名（调试用）

检查内容：
1. 登记：METHODS（下方的方法清单，供实验运行器使用）与 REGISTRY 完全一致，无重名；每个函数的签名为
   select(ctx, k, seed, **hp)，超参数全部有默认值；各文件自带的名单（USES_ORACLE、PROB_DESIGNS）与清单一致。
2. 契约（make_toy_ctx(N=3000)，k ∈ {50, 200}）：int64、长度 k、无重复、升序、在 [0, N) 内；同种子可复现；
   随机方法不同种子结果不同，确定性方法不同种子结果相同。
3. 不读标签：不允许用 oracle 的方法在“oracle 抛异常、删除 _true_y”的 ctx 上结果不变；允许用 oracle 的方法
   （margin_r2、margin_r4、hybrid*）用记录型 oracle：只揭示最终入选的下标、不重复揭示、调用次数符合规格，
   oracle=None 时报错。
4. 不改 ctx：每次调用前后 ctx 的全部数组（T_*、F_*、P_T、P_T_loco、dom、P_F_oof）逐字节不变。
5. meta：概率设计写 ctx.meta[注册名]，pi ∈ [0,1]、和 ≈ k（Hybrid 为 k0）、prior_idx ⊆ 样本且 pi[prior_idx] > 0；
   非概率设计不写 ctx.meta。
6. 可选字段：需要 P_T_loco / dom / P_F_oof / oracle 的方法在缺字段时给出明确报错；其余方法在只有必需字段的
   ctx 上能运行（验证清单中的 ctx 需求）。
7. 规格公式抽查（独立重写后比对）：LPM2 不等概率入样频率、GIST 阈值序列与目标函数（与朴素实现逐一比对）、
   BADGE 幻觉梯度与 D² 抽样概率（显式 Kronecker 向量）、TypiClust 典型性、FASS 配额（用 designs.alloc 独立重算）、
   TQS 的 Q 分量与候选范围、BAIT 贪心（显式 Kronecker Fisher + 直接求逆）、UHerding 核截断的影响（只报告）。
8. 合理性汇总（k=200）：各方法样本的平均 margin、覆盖的预测类数、AE 空间 MSSSD，与 SRS 比较。
9. 耗时：make_toy_ctx(N=30000, n_F=50000)，k=1000，默认超参数（多轮 margin 与 Hybrid 为完整训练配置），
   超过上限的方法标出并判失败。
"""
import hashlib
import inspect
import json
import os
import sys
import time
import unittest

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '2')
os.environ.setdefault('LGBM_THREADS', '1')

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(os.path.dirname(HERE))
if CODE not in sys.path:
    sys.path.insert(0, CODE)

from samplers.base import REGISTRY, make_toy_ctx, X_of, X_of_foreign, unc_margin  # noqa: E402
from samplers import designs, uncertainty, coverage, transfer  # noqa: E402,F401

# =====================================================================
# 方法清单（供实验运行器使用）
# =====================================================================
# 字段：
#   mid     规格编号（M1–M20）
#   prob    概率设计（写 ctx.meta[注册名]['pi']，可做设计无偏先验估计）
#   oracle  选样时调用 ctx.oracle（只有 M4 多轮版与 M8 Hybrid）
#   needs   除必需字段外需要的 ctx 字段 / 超参数（'P_T_loco'、'dom'、'P_F_oof'、'oracle'、'hp:p_hat'）
#   seed    'random'：不同种子结果不同；'det'：与种子无关；
#           'weak'：种子只影响候选子采样、σ/s_c 的随机点对或重训的 LightGBM 随机性，小池上可能相同
M = dict
METHODS = {
    # ---------------- designs.py ----------------
    'srs':              M(mid='M1',  prob=True,  oracle=False, needs=(), seed='random'),
    'srs_nuts3':        M(mid='M1',  prob=True,  oracle=False, needs=(), seed='random'),
    'lpm2':             M(mid='M2',  prob=True,  oracle=False, needs=(), seed='random'),
    'lpm2_nogeo':       M(mid='M2',  prob=True,  oracle=False, needs=(), seed='random'),
    'lpm2_aes2':        M(mid='M2',  prob=True,  oracle=False, needs=(), seed='random'),
    'predstrat':        M(mid='M3',  prob=True,  oracle=False, needs=(), seed='random'),
    'predstrat_prop':   M(mid='M3',  prob=True,  oracle=False, needs=(), seed='random'),
    'predstrat_equal':  M(mid='M3',  prob=True,  oracle=False, needs=(), seed='random'),
    'predstrat_neyman': M(mid='M3',  prob=True,  oracle=False, needs=(), seed='random'),
    'predstrat_conf':   M(mid='M3',  prob=True,  oracle=False, needs=(), seed='random'),
    'predstrat_srs':    M(mid='M3',  prob=True,  oracle=False, needs=(), seed='random'),
    'pmht':             M(mid='M5',  prob=True,  oracle=False, needs=(), seed='random'),
    'pmht_b2':          M(mid='M5',  prob=True,  oracle=False, needs=(), seed='random'),
    'pmht_b4':          M(mid='M5',  prob=True,  oracle=False, needs=(), seed='random'),
    'pmht_t02':         M(mid='M5',  prob=True,  oracle=False, needs=(), seed='random'),
    'pmht_t05':         M(mid='M5',  prob=True,  oracle=False, needs=(), seed='random'),
    'pmht_gumbel':      M(mid='M5',  prob=False, oracle=False, needs=(), seed='random'),
    'cube':             M(mid='M9',  prob=True,  oracle=False, needs=(), seed='random'),
    'cube_ae3':         M(mid='M9',  prob=True,  oracle=False, needs=(), seed='random'),
    'cube_nuts2':       M(mid='M9',  prob=True,  oracle=False, needs=(), seed='random'),
    'clhs':             M(mid='M19', prob=False, oracle=False, needs=(), seed='random'),
    # ---------------- uncertainty.py ----------------
    'margin':           M(mid='M4',  prob=False, oracle=False, needs=(), seed='det'),
    'entropy':          M(mid='M4',  prob=False, oracle=False, needs=(), seed='det'),
    'margin_prior':     M(mid='M4',  prob=False, oracle=False, needs=(), seed='det'),
    'margin_r2':        M(mid='M4',  prob=False, oracle=True,  needs=('oracle',), seed='weak'),
    'margin_r4':        M(mid='M4',  prob=False, oracle=True,  needs=('oracle',), seed='weak'),
    'gist_margin':      M(mid='M6',  prob=False, oracle=False, needs=(), seed='det'),
    'gist_maxmin':      M(mid='M6',  prob=False, oracle=False, needs=(), seed='det'),
    'gist_geo':         M(mid='M6',  prob=False, oracle=False, needs=(), seed='det'),
    'kcenter':          M(mid='M6',  prob=False, oracle=False, needs=(), seed='random'),
    'kcenter_foreign':  M(mid='M6',  prob=False, oracle=False, needs=(), seed='det'),
    'badge':            M(mid='M7',  prob=False, oracle=False, needs=(), seed='random'),
    'badge_ae_s2':      M(mid='M7',  prob=False, oracle=False, needs=(), seed='random'),
    'kdpp':             M(mid='M16', prob=False, oracle=False, needs=(), seed='random'),
    'kdpp_div':         M(mid='M16', prob=False, oracle=False, needs=(), seed='random'),
    # ---------------- coverage.py ----------------
    'clue':             M(mid='M10', prob=False, oracle=False, needs=(), seed='random'),
    'clue_aes2':        M(mid='M10', prob=False, oracle=False, needs=(), seed='random'),
    'uherding':         M(mid='M11', prob=False, oracle=False, needs=(), seed='weak'),
    'maxherding':       M(mid='M11', prob=False, oracle=False, needs=(), seed='weak'),
    'uherding_foreign': M(mid='M11', prob=False, oracle=False, needs=(), seed='weak'),
    'typiclust':        M(mid='M12', prob=False, oracle=False, needs=(), seed='random'),
    'typiclust_strat':  M(mid='M12', prob=False, oracle=False, needs=(), seed='random'),
    'fass_quota':       M(mid='M14', prob=False, oracle=False, needs=(), seed='weak'),
    'fass_quota_prior': M(mid='M14', prob=False, oracle=False, needs=(), seed='weak'),
    'fass_quota_equal': M(mid='M14', prob=False, oracle=False, needs=(), seed='weak'),
    'fass_quota_prior_est': M(mid='M14', prob=False, oracle=False, needs=('hp:p_hat',), seed='weak'),
    'probcover':        M(mid='M15', prob=False, oracle=False, needs=(), seed='random'),
    'probcover_km':     M(mid='M15', prob=False, oracle=False, needs=(), seed='random'),
    'kherding':         M(mid='M17', prob=False, oracle=False, needs=(), seed='weak'),
    'kherding_foreign': M(mid='M17', prob=False, oracle=False, needs=(), seed='weak'),
    'fscs':             M(mid='M20', prob=False, oracle=False, needs=(), seed='random'),
    'fscs_geo':         M(mid='M20', prob=False, oracle=False, needs=(), seed='random'),
    # ---------------- transfer.py ----------------
    'tqs':              M(mid='M13', prob=False, oracle=False, needs=('P_T_loco', 'dom'), seed='random'),
    'tqs_uc':           M(mid='M13', prob=False, oracle=False, needs=('P_T_loco',), seed='random'),
    'tqs_u':            M(mid='M13', prob=False, oracle=False, needs=('P_T_loco',), seed='det'),
    'bait':             M(mid='M18', prob=False, oracle=False, needs=('P_F_oof',), seed='det'),
    'bait_wk10':        M(mid='M18', prob=False, oracle=False, needs=('P_F_oof',), seed='det'),
    'bait_wk':          M(mid='M18', prob=False, oracle=False, needs=('P_F_oof',), seed='det'),
    'bait_backward':    M(mid='M18', prob=False, oracle=False, needs=('P_F_oof',), seed='det'),
    'hybrid':           M(mid='M8',  prob=True,  oracle=True,  needs=('oracle',), seed='random'),
    'hybrid_margin':    M(mid='M8',  prob=True,  oracle=True,  needs=('oracle',), seed='random'),
    'hybrid_gist':      M(mid='M8',  prob=True,  oracle=True,  needs=('oracle',), seed='random'),
}
# 正确性检查时对重训方法用轻量 LightGBM（耗时测试用默认完整配置）
_FAST_FIT = {'n_trees': 20, 'calib_folds': 2}


def _names():
    only = os.environ.get('INTEG_ONLY')
    names = sorted(METHODS, key=lambda n: (METHODS[n]['mid'][1:].zfill(2), n))
    if only:
        keep = set(only.split(','))
        names = [n for n in names if n in keep]
    return names


def hp_for(name, ctx, fast=True):
    """各方法在集成测试中的额外超参数。"""
    if name == 'fass_quota_prior_est':
        return {'p_hat': np.bincount(ctx.P_T.argmax(1), minlength=ctx.C) / ctx.N}
    if fast and name in ('margin_r2', 'margin_r4'):
        return dict(_FAST_FIT)
    if fast and name.startswith('hybrid'):
        return {'fit_kw': dict(_FAST_FIT)}
    return {}


def augment(ctx, seed=0):
    """给 make_toy_ctx 加上可选字段：P_T_loco（10 个留一国成员）、dom（多数接近 1 的域判别器输出）、P_F_oof。"""
    r = np.random.default_rng(seed + 100)
    N, C = ctx.N, ctx.C
    lp = np.log(ctx.P_T)
    loco = []
    for _ in range(10):
        z = lp + r.normal(scale=0.3, size=C)[None] + r.normal(scale=0.4, size=lp.shape)
        z = np.exp(z - z.max(1, keepdims=True))
        loco.append(z / z.sum(1, keepdims=True))
    ctx.P_T_loco = np.stack(loco)
    dist = np.linalg.norm(ctx.T_ae[:, :8] - ctx.F_ae[:, :8].mean(0), axis=1)
    ctx.dom = 1 / (1 + np.exp(-(3 + 0.5 * (dist - np.median(dist)) + r.normal(scale=0.5, size=N))))
    n_F = len(ctx.F_y)
    zF = 2.5 * np.eye(C)[ctx.F_y] + r.normal(size=(n_F, C))
    zF = np.exp(zF - zF.max(1, keepdims=True))
    ctx.P_F_oof = zF / zF.sum(1, keepdims=True)
    return ctx


def toy(N=3000, n_F=5000, seed=0, full=True):
    ctx = make_toy_ctx(N=N, n_F=n_F, seed=seed)
    return augment(ctx, seed) if full else ctx


_ARRAY_FIELDS = ('T_ae', 'T_s2', 'T_xy', 'T_nuts3', 'F_ae', 'F_s2', 'F_xy', 'F_y', 'F_country', 'P_T',
                 'P_T_loco', 'dom', 'P_F_oof')


def ctx_digest(ctx):
    """ctx 全部数组字段的字节摘要（检查选样函数是否原地修改了 ctx）。"""
    h = {}
    for f in _ARRAY_FIELDS:
        a = getattr(ctx, f, None)
        if a is None:
            h[f] = None
        else:
            a = np.ascontiguousarray(a)
            h[f] = (a.shape, str(a.dtype), hashlib.sha1(a.tobytes()).hexdigest())
    h['N'] = ctx.N
    return h


class RecordingOracle:
    """记录每次调用揭示了哪些下标；标签来自测试自己保存的数组，不经 ctx._true_y。"""

    def __init__(self, y):
        self.y = np.asarray(y)
        self.calls = []

    def __call__(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        self.calls.append(idx.copy())
        return self.y[idx]


def _raise_oracle(*a, **kw):
    raise RuntimeError('选样函数调用了 ctx.oracle')


def no_label_ctx(ctx_factory):
    ctx = ctx_factory()
    ctx.oracle = _raise_oracle
    if hasattr(ctx, '_true_y'):
        del ctx._true_y
    return ctx


def run(name, ctx, k, seed, fast=True):
    ctx.meta = {}
    return REGISTRY[name](ctx, k, seed, **hp_for(name, ctx, fast))


def check_output(idx, k, N):
    """返回违反契约的描述列表（空表示通过）。"""
    bad = []
    if not isinstance(idx, np.ndarray):
        bad.append(f'返回类型 {type(idx)}')
        idx = np.asarray(idx)
    if idx.dtype != np.int64:
        bad.append(f'dtype {idx.dtype}')
    if idx.ndim != 1 or len(idx) != k:
        bad.append(f'长度 {len(idx)} ≠ k={k}')
    if len(np.unique(idx)) != len(idx):
        bad.append('有重复下标')
    if len(idx) > 1 and not (np.diff(idx) > 0).all():
        bad.append('未升序')
    if len(idx) and (idx.min() < 0 or idx.max() >= N):
        bad.append('下标越界')
    return bad


# =====================================================================
# 1. 登记与签名
# =====================================================================

def test_catalog_matches_registry():
    reg = set(REGISTRY)
    cat = set(METHODS)
    assert reg == cat, f'REGISTRY 多出 {sorted(reg - cat)}，清单多出 {sorted(cat - reg)}'
    mods = {n: REGISTRY[n].__module__.rsplit('.', 1)[-1] for n in REGISTRY}
    assert set(mods.values()) == {'designs', 'uncertainty', 'coverage', 'transfer'}, set(mods.values())
    # 各文件自带名单与清单一致
    assert set(designs.DESIGN_METHODS) == {n for n in mods if mods[n] == 'designs'}
    assert set(designs.PROB_DESIGNS) == {n for n in designs.DESIGN_METHODS if METHODS[n]['prob']}
    assert set(uncertainty.USES_ORACLE) == {n for n in mods if mods[n] == 'uncertainty' and METHODS[n]['oracle']}
    assert set(uncertainty.DETERMINISTIC) <= {n for n in METHODS if METHODS[n]['seed'] == 'det'}
    assert {n for n in METHODS if METHODS[n]['oracle']} == {'margin_r2', 'margin_r4', 'hybrid', 'hybrid_margin',
                                                            'hybrid_gist'}


def test_no_duplicate_registration():
    """静态扫描四个源文件中的 @register('名字')，确认没有重名（register 会静默覆盖先登记的函数）。"""
    import re
    seen = {}
    for m in (designs, uncertainty, coverage, transfer):
        with open(m.__file__, encoding='utf-8') as fh:
            for name in re.findall(r"@register\(\s*['\"]([^'\"]+)['\"]\s*\)", fh.read()):
                seen.setdefault(name, []).append(m.__name__)
    dups = {n: v for n, v in seen.items() if len(v) > 1}
    assert not dups, dups
    assert set(seen) == set(METHODS), (set(seen) ^ set(METHODS))


def test_signatures():
    """select(ctx, k, seed, **hp)：前三个位置参数为 ctx、k、seed，其余超参数全部有默认值。"""
    bad = []
    for n, fn in REGISTRY.items():
        ps = list(inspect.signature(fn).parameters.values())
        if [p.name for p in ps[:3]] != ['ctx', 'k', 'seed']:
            bad.append((n, [p.name for p in ps[:3]]))
        for p in ps[3:]:
            if p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
                continue
            if p.default is inspect.Parameter.empty:
                bad.append((n, p.name, '无默认值'))
    assert not bad, bad


# =====================================================================
# 2–5. 契约、可复现、种子、不读标签、不改 ctx、meta
# =====================================================================

def _check_meta(name, ctx, idx, k):
    bad = []
    info = METHODS[name]
    if not info['prob']:
        if name in ctx.meta:
            bad.append(f'非概率设计写了 ctx.meta[{name!r}]')
        return bad
    m = ctx.meta.get(name)
    if m is None or 'pi' not in m or 'prior_idx' not in m:
        return [f'概率设计没有写 ctx.meta[{name!r}] 的 pi / prior_idx']
    pi = np.asarray(m['pi'], dtype=np.float64)
    pidx = np.asarray(m['prior_idx'], dtype=np.int64)
    if pi.shape != (ctx.N,) or not np.isfinite(pi).all():
        bad.append(f'pi 形状 {pi.shape} 或含非有限值')
        return bad
    if pi.min() < -1e-12 or pi.max() > 1 + 1e-12:
        bad.append(f'pi 超出 [0,1]：[{pi.min()}, {pi.max()}]')
    target = m.get('k0', k) if name.startswith('hybrid') else k
    if abs(pi.sum() - target) > 1e-6 * max(1, target):
        bad.append(f'Σpi = {pi.sum():.6f} ≠ {target}')
    if not np.isin(pidx, idx).all():
        bad.append('prior_idx 不在样本内')
    if (pi[pidx] <= 0).any():
        bad.append('有入样单元 pi = 0')
    if name.startswith('hybrid'):
        if len(pidx) != target:
            bad.append(f'Hybrid 的 prior_idx 长度 {len(pidx)} ≠ k0={target}')
    elif not np.array_equal(np.sort(pidx), idx):
        bad.append('prior_idx 与样本不一致')
    return bad


def test_contract_reproducible_meta_immutable():
    """k ∈ {50, 200}：契约、同种子可复现、meta、不改 ctx。"""
    ctx = toy()
    y_true = ctx._true_y.copy()
    fails = []
    for name in _names():
        for k in (50, 200):
            ctx.oracle = RecordingOracle(y_true)
            before = ctx_digest(ctx)
            try:
                a = run(name, ctx, k, 0)
            except Exception as e:  # noqa: BLE001
                fails.append((name, k, f'异常 {e!r}'))
                continue
            if ctx_digest(ctx) != before:
                fails.append((name, k, '修改了 ctx 的数组字段'))
            for msg in check_output(a, k, ctx.N):
                fails.append((name, k, msg))
            for msg in _check_meta(name, ctx, a, k):
                fails.append((name, k, msg))
            ctx.oracle = RecordingOracle(y_true)
            b = run(name, ctx, k, 0)
            if not np.array_equal(a, b):
                fails.append((name, k, '同种子结果不同'))
    assert not fails, '\n'.join(map(str, fails))


def test_seed_behavior():
    """随机方法：3 个种子结果两两不同；确定性方法：结果相同；'weak' 只记录。"""
    ctx = toy()
    y_true = ctx._true_y.copy()
    fails, weak = [], {}
    for name in _names():
        outs = []
        for s in (0, 1, 2):
            ctx.oracle = RecordingOracle(y_true)
            outs.append(run(name, ctx, 50, s))
        same = [np.array_equal(outs[i], outs[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
        kind = METHODS[name]['seed']
        if kind == 'random' and any(same):
            fails.append((name, '随机方法在不同种子下结果相同'))
        elif kind == 'det' and not all(same):
            fails.append((name, '确定性方法在不同种子下结果不同'))
        elif kind == 'weak':
            weak[name] = sum(same)
    print('    [weak] 3 个种子两两相同的对数：' + '，'.join(f'{n} {v}' for n, v in weak.items()))
    assert not fails, fails


def test_no_label_access():
    """不允许用 oracle 的方法：oracle 抛异常、删除 _true_y 后结果不变。"""
    ref_ctx = toy()
    fails = []
    for name in _names():
        if METHODS[name]['oracle']:
            continue
        for k in (50, 200):
            ref = run(name, ref_ctx, k, 0)
            ctx = no_label_ctx(toy)
            try:
                out = run(name, ctx, k, 0)
            except Exception as e:  # noqa: BLE001
                fails.append((name, k, f'无标签 ctx 上失败：{e!r}'))
                continue
            if not np.array_equal(ref, out):
                fails.append((name, k, '有无 oracle 时结果不同'))
    assert not fails, fails


# 规格：M4 多轮版每轮之间调用一次 oracle（rounds−1 次）；Hybrid 只在第一批后调用一次
_ORACLE_CALLS = {'margin_r2': 1, 'margin_r4': 3, 'hybrid': 1, 'hybrid_margin': 1, 'hybrid_gist': 1}


def test_oracle_methods_reveal_only_selected():
    fails = []
    for name in _names():
        if not METHODS[name]['oracle']:
            continue
        for k in (50, 200):
            ctx = toy()
            y = ctx._true_y.copy()
            del ctx._true_y
            orc = RecordingOracle(y)
            ctx.oracle = orc
            out = run(name, ctx, k, 0)
            revealed = np.concatenate(orc.calls) if orc.calls else np.zeros(0, np.int64)
            if len(orc.calls) != _ORACLE_CALLS[name]:
                fails.append((name, k, f'oracle 调用 {len(orc.calls)} 次，应为 {_ORACLE_CALLS[name]}'))
            if len(np.unique(revealed)) != len(revealed):
                fails.append((name, k, '重复揭示'))
            if not np.isin(revealed, out).all():
                fails.append((name, k, '揭示了未入选的下标'))
            if name.startswith('hybrid'):
                k0 = ctx.meta[name]['k0']
                if len(revealed) != k0 or not np.array_equal(np.sort(revealed), np.sort(ctx.meta[name]['prior_idx'])):
                    fails.append((name, k, 'Hybrid 揭示的不是第一批 S1'))
            # oracle=None 时必须报错，而不是静默退化
            ctx2 = toy()
            ctx2.oracle = None
            del ctx2._true_y
            try:
                run(name, ctx2, k, 0)
                fails.append((name, k, 'oracle=None 时没有报错'))
            except (RuntimeError, ValueError):
                pass
    assert not fails, fails


def test_optional_fields():
    """缺少可选字段时：需要它的方法给出 ValueError / RuntimeError；不需要它的方法照常运行。"""
    fails = []
    for name in _names():
        needs = [f for f in METHODS[name]['needs'] if not f.startswith('hp:')]
        ctx = toy(full=False)              # 只有必需字段 + oracle
        if METHODS[name]['oracle']:
            if needs == ['oracle']:
                continue                   # 已在 test_oracle_methods_reveal_only_selected 中检查
        try:
            out = run(name, ctx, 50, 0)
            if needs:
                fails.append((name, f'缺 {needs} 时没有报错'))
            else:
                for msg in check_output(out, 50, ctx.N):
                    fails.append((name, msg))
        except (ValueError, RuntimeError) as e:
            if not needs:
                fails.append((name, f'只有必需字段时失败：{e!r}'))
        except Exception as e:  # noqa: BLE001
            fails.append((name, f'缺字段时报了非预期的异常 {e!r}'))
    # fass_quota_prior_est 没有 p_hat 时应报错
    ctx = toy()
    try:
        REGISTRY['fass_quota_prior_est'](ctx, 50, 0)
        fails.append(('fass_quota_prior_est', '没有 p_hat 时没有报错'))
    except ValueError:
        pass
    assert not fails, fails


def test_string_nuts_codes():
    """T_nuts3 为 NUTS 字符串（如 'BE211'）时所有方法仍可运行（designs 会用字符串前缀取 NUTS2）。"""
    ctx = toy()
    codes = np.unique(ctx.T_nuts3)
    lut = {c: f'BE{2 + (i % 3)}{1 + (i // 3) % 5}{i % 7}' for i, c in enumerate(codes)}
    ctx.T_nuts3 = np.array([lut[c] for c in ctx.T_nuts3])
    fails = []
    for name in _names():
        try:
            out = run(name, ctx, 50, 0)
            for msg in check_output(out, 50, ctx.N):
                fails.append((name, msg))
        except Exception as e:  # noqa: BLE001
            fails.append((name, repr(e)))
    assert not fails, fails
    assert 'T_nuts3[:4]' in str(designs._nuts2_of(ctx)[1])


# =====================================================================
# 7. 规格公式抽查（独立重写）
# =====================================================================

def test_spec_lpm2_unequal_inclusion():
    """LPM2 在不等概率下保持一阶包含概率（独立的 z 检验，N=80、3000 次）。"""
    rng = np.random.default_rng(7)
    N, reps = 80, 3000
    X = rng.normal(size=(N, 3))
    pi = rng.uniform(0.05, 0.6, N)
    pi *= 20 / pi.sum()
    cnt = np.zeros(N)
    for r in range(reps):
        s = designs.lpm2(pi, X, np.random.default_rng(1000 + r))
        assert len(s) == 20
        cnt[s] += 1
    z = (cnt / reps - pi) / np.sqrt(pi * (1 - pi) / reps)
    print(f'    [LPM2] max|z| = {np.abs(z).max():.2f}，mean z² = {np.mean(z ** 2):.2f}')
    assert np.abs(z).max() < 4.5 and 0.6 < np.mean(z ** 2) < 1.5


def test_spec_lpm2_pivot_rule():
    """两点 LPM2 的枢轴更新：s<1 与 s≥1 两种情形下，入样概率等于 pi（精确可算）。"""
    X = np.array([[0.0], [1.0]])
    for pi in ([0.3, 0.7], [0.6, 0.4], [0.2, 0.8]):
        # Σpi = 1：一个入样，i 入样的频率应为 pi_i
        hits = np.zeros(2)
        for r in range(4000):
            hits[designs.lpm2(np.array(pi), X, np.random.default_rng(r))] += 1
        z = (hits / 4000 - pi) / np.sqrt(np.array(pi) * (1 - np.array(pi)) / 4000)
        assert np.abs(z).max() < 4, (pi, hits)
    # s ≥ 1 的情形：3 个点 Σpi = 2
    X3 = np.array([[0.0], [1.0], [5.0]])
    pi3 = np.array([0.8, 0.7, 0.5])
    hits = np.zeros(3)
    for r in range(4000):
        s = designs.lpm2(pi3, X3, np.random.default_rng(r))
        assert len(s) == 2
        hits[s] += 1
    z = (hits / 4000 - pi3) / np.sqrt(pi3 * (1 - pi3) / 4000)
    assert np.abs(z).max() < 4, hits


def _naive_gist(u, X, k, alpha, eps):
    """规格 M6 伪代码的逐行朴素实现（全距离矩阵）。"""
    N = len(u)
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    d_max = D.max()
    a, b = np.unravel_index(np.argmax(D), D.shape)
    order = np.argsort(-u, kind='stable')

    def minpair(S):
        if len(S) < 2:
            return 0.0
        sub = D[np.ix_(S, S)]
        return sub[~np.eye(len(S), dtype=bool)].min()

    def f(S):
        return alpha * u[S].sum() / k + (1 - alpha) * minpair(S) / d_max

    def pad(S):
        S = list(S)
        for v in order:
            if len(S) >= k:
                break
            if v not in S:
                S.append(v)
        return np.array(S[:k])

    cands = [order[:k], pad([a, b])]
    i, n_thr = 0, 0
    while (1 + eps) ** i <= 2 / eps:
        d = (1 + eps) ** i * eps * d_max / 2
        i += 1
        n_thr += 1
        S, mind = [], np.full(N, np.inf)
        for v in order:
            if mind[v] >= d:
                S.append(v)
                mind = np.minimum(mind, D[v])
                if len(S) == k:
                    break
        cands.append(pad(S))
    vals = [f(np.asarray(S)) for S in cands]
    return np.sort(cands[int(np.argmax(vals))]), max(vals), n_thr


def test_spec_gist_vs_naive():
    """GIST：阈值个数（ε=0.05 → 76）、目标函数归一化与选集，与规格伪代码的朴素实现一致。"""
    assert len(uncertainty.gist_thresholds(1.0, 0.05)) == 76
    for seed, alpha in ((0, 0.9), (1, 0.5), (2, 0.0)):
        ctx = make_toy_ctx(N=400, n_F=500, seed=seed)
        u = unc_margin(ctx.P_T)
        X = X_of(ctx, 'ae', 32)
        for k in (5, 20):
            ours = REGISTRY['gist_margin'](ctx, k, 0, alpha=alpha)
            ref, fref, n_thr = _naive_gist(u, X, k, alpha, 0.05)
            assert n_thr == 76
            M_ = uncertainty.GistMetric(X)
            _, info = uncertainty.gist_core(u, M_, k, alpha=alpha, eps=0.05, return_info=True)
            assert abs(info['f'] - fref) < 1e-9, (seed, k, info['f'], fref)
            assert np.array_equal(ours, ref), (seed, alpha, k)


def test_spec_badge_explicit_kronecker():
    """BADGE：首点 = argmax‖g‖²；之后每步的抽样概率 ∝ 到已选点的最小‖g_i − g_s‖²（显式 Kronecker 向量）。"""
    ctx = make_toy_ctx(N=200, n_F=300, seed=3)
    P = ctx.P_T
    h = X_of(ctx, 'ae', 32)
    e = np.eye(ctx.C)[P.argmax(1)]
    G = np.stack([np.kron(P[i] - e[i], h[i]) for i in range(ctx.N)])     # (N, C·d)
    g2 = (G ** 2).sum(1)
    rng = np.random.default_rng(0)
    probs = []

    def chooser(p):
        probs.append(p.copy())
        return int(rng.choice(len(p), p=p))

    S = uncertainty.badge_core(P, h, 12, chooser=chooser)
    assert S[0] == int(np.argmax(g2))
    for t, p in enumerate(probs):
        chosen = S[:t + 1]
        dmin = ((G[:, None, :] - G[None, chosen, :]) ** 2).sum(-1).min(1)
        dmin[chosen] = 0.0
        np.testing.assert_allclose(p, dmin / dmin.sum(), rtol=1e-7, atol=1e-12)
    # 注册函数与 badge_core 一致
    out = REGISTRY['badge'](ctx, 12, 5)
    ref = uncertainty.badge_core(P, h, 12, rng=np.random.default_rng(5))
    assert np.array_equal(out, np.sort(ref))


def test_spec_typicality():
    """TypiClust 典型性 = 1 /（m = min(20, n−1) 个近邻的平均距离），两个计算分支都与 sklearn 近邻一致。"""
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(0)
    for n in (7, 300, 2600):
        Xc = rng.normal(size=(n, 6))
        m = min(20, n - 1)
        dist = NearestNeighbors(n_neighbors=m + 1).fit(Xc).kneighbors(Xc)[0][:, 1:]
        ref = 1.0 / (dist.mean(1) + 1e-7)
        np.testing.assert_allclose(coverage._typicality(Xc, 20), ref, rtol=1e-6)


def test_spec_fass_quota_counts():
    """FASS-quota：各预测类实际入选数 = designs.alloc(各类候选数, 权重, k, 1)（跨文件独立重算），且全部在 top-β 内。"""
    ctx = toy()
    yhat_pool = ctx.P_T.argmax(1)
    Npool = np.bincount(yhat_pool, minlength=ctx.C).astype(float)
    u = unc_margin(ctx.P_T)
    for name, w_of in (('fass_quota', np.sqrt), ('fass_quota_prior', lambda x: x),
                       ('fass_quota_equal', lambda x: np.ones_like(x))):
        for k in (50, 200):
            beta = min(ctx.N, max(3000, 10 * k))
            cand = np.argsort(-u, kind='stable')[:beta]
            cap = np.bincount(ctx.P_T[cand].argmax(1), minlength=ctx.C)
            expect = designs.alloc(cap, w_of(Npool), k, 1)
            out = run(name, ctx, k, 0)
            got = np.bincount(ctx.P_T[out].argmax(1), minlength=ctx.C)
            assert np.array_equal(got, expect), (name, k, got, expect)
            assert np.isin(out, cand).all()


def test_spec_tqs_components():
    """TQS：独立重算 Q_u、Q_c（广义 JS 的分位秩）、Q_d（域判别器 logit 分位秩上的高斯）；
    tqs_u = 委员会平均 margin 的 top-k；tqs 的样本都在 Q 最高的 3k 个候选内。"""
    from scipy.special import rel_entr
    from scipy.stats import rankdata
    ctx = toy()
    P = ctx.P_T_loco
    N = ctx.N
    Qu = np.mean([unc_margin(p) for p in P], axis=0)
    pbar = P.mean(0)
    js = rel_entr(P, pbar[None]).sum(-1).mean(0)
    Qc = rankdata(js) / N
    rr = rankdata(np.clip(ctx.dom, 1e-6, 1 - 1e-6)) / N        # logit 单调，秩与 dom 本身的秩相同
    Qd = np.exp(-(rr - 0.5) ** 2 / (2 * 0.25 ** 2))
    Q, parts = transfer.tqs_scores(ctx)
    np.testing.assert_allclose(parts['u'], Qu, rtol=1e-10)
    np.testing.assert_allclose(parts['js'], js, rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(parts['c'], Qc, atol=2.0 / N)     # 允许数值近似并列时秩互换
    np.testing.assert_allclose(parts['d'], Qd, atol=1e-6)
    np.testing.assert_allclose(Q, parts['u'] + parts['c'] + parts['d'], rtol=1e-12)
    for k in (50, 200):
        assert np.array_equal(REGISTRY['tqs_u'](ctx, k, 0), np.sort(np.argsort(-Qu, kind='stable')[:k]))
        thr = np.sort(Q)[::-1][3 * k - 1]
        assert (Q[REGISTRY['tqs'](ctx, k, 0)] >= thr - 1e-9).all()
        assert (Q[REGISTRY['tqs_uc'](ctx, k, 0)] >= 0).all()


def test_spec_bait_direct():
    """BAIT：用显式 Kronecker Fisher I(x) = A ⊗ hhᵀ 与直接求解重写前向贪心，所选集合与目标值一致。"""
    ctx = augment(make_toy_ctx(N=240, n_F=400, seed=5), 5)
    d, k, w = 8, 6, 1.0
    h = X_of(ctx, 'ae', d)
    hF = X_of_foreign(ctx, 'ae', d)

    def fisher(p, x):
        A = np.diag(p) - np.outer(p, p)
        return np.kron(A, np.outer(x, x))

    I_U = np.mean([fisher(ctx.P_T[i], h[i]) for i in range(ctx.N)], axis=0)
    I_F = np.mean([fisher(ctx.P_F_oof[i], hF[i]) for i in range(len(hF))], axis=0)
    D = I_U.shape[0]
    Mx = 1e-2 * np.trace(I_F) / D * np.eye(D) + w * I_F
    Ii = [fisher(ctx.P_T[i], h[i]) for i in range(ctx.N)]
    S = []
    for _ in range(k):
        base_ = np.trace(np.linalg.solve(Mx, I_U))
        gains = np.array([-np.inf if i in S else base_ - np.trace(np.linalg.solve(Mx + Ii[i], I_U))
                          for i in range(ctx.N)])
        i = int(np.argmax(gains))
        S.append(i)
        Mx = Mx + Ii[i]
    out = REGISTRY['bait'](ctx, k, 0, d=d)
    assert np.array_equal(out, np.sort(S)), (out, np.sort(S))
    obj = transfer.bait_objective(ctx, out, d=d, w=w)
    assert abs(obj - np.trace(np.linalg.solve(Mx, I_U))) < 1e-8 * max(1.0, obj)


def test_spec_uherding_truncation_report():
    """UHerding：规格的 M_nn = max(200, 3m) 近邻截断会丢掉多少核质量（只报告，不判定）。
    比较默认截断版与不截断版（M_nn = N）在精确 UC(S) 上的差距。"""
    ctx = toy()
    X = X_of(ctx, 'ae', 32)
    U = unc_margin(ctx.P_T)
    k = 50
    run('uherding', ctx, k, 0)
    sigma = coverage.DIAG['uherding']['sigma']
    M_nn = coverage.DIAG['uherding']['M_nn']
    from sklearn.neighbors import NearestNeighbors
    dist = NearestNeighbors(n_neighbors=M_nn).fit(X).kneighbors(X[:500])[0]
    k_at_cut = float(np.median(np.exp(-dist[:, -1] ** 2 / sigma ** 2)))

    def uc(S):
        D2 = ((X[:, None, :] - X[None, S, :]) ** 2).sum(-1)
        return float((U * np.exp(-D2 / sigma ** 2).max(1)).sum() / len(X))

    a = run('uherding', ctx, k, 0)
    b = REGISTRY['uherding'](ctx, k, 0, M_nn=ctx.N, sigma=sigma)
    rnd = [uc(np.random.default_rng(s).choice(ctx.N, k, replace=False)) for s in range(10)]
    print(f'    [UHerding] σ={sigma:.3f}，M_nn={M_nn}，第 M_nn 近邻处核值中位数 {k_at_cut:.3f}；'
          f'精确 UC：默认截断 {uc(a):.4f}，不截断 {uc(b):.4f}，随机最好 {max(rnd):.4f}')
    assert uc(a) > max(rnd)


# =====================================================================
# 8. 合理性汇总（k=200）
# =====================================================================

def test_sanity_summary():
    ctx = toy()
    N, k = ctx.N, 200
    u = unc_margin(ctx.P_T)
    yhat = ctx.P_T.argmax(1)
    n_pred = len(np.unique(yhat))
    X = X_of(ctx, 'ae', 32)
    sqX = (X ** 2).sum(1)

    def msssd(S):
        D2 = np.maximum(sqX[:, None] + sqX[None, S] - 2 * X @ X[S].T, 0)
        return float(D2.min(1).mean())

    srs_u, srs_m = [], []
    for s in range(10):
        S = np.random.default_rng(s).choice(N, k, replace=False)
        srs_u.append(u[S].mean())
        srs_m.append(msssd(S))
    rows = {}
    for name in _names():
        S = run(name, ctx, k, 0)
        rows[name] = (u[S].mean(), len(np.unique(yhat[S])), msssd(S), len(np.unique(ctx._true_y[S])))
    print(f'    SRS（10 次）：平均 margin {np.mean(srs_u):.3f}（最大 {np.max(srs_u):.3f}），'
          f'MSSSD {np.mean(srs_m):.2f}（最小 {np.min(srs_m):.2f}）；池中预测类 {n_pred}')
    print('    方法                  平均margin  预测类数  MSSSD   真实类数')
    for name, (mu, npc, ms, ntc) in rows.items():
        print(f'    {name:22s} {mu:8.3f}  {npc:6d}   {ms:7.2f}  {ntc:5d}')
    fails = []
    # 以不确定性为主的方法：样本平均 margin 高于全部 10 个 SRS 样本
    # （FASS 在 N=3000 时 β = N，不做不确定性过滤，因此不在此列；margin_prior 的 margin 是在先验校正后的
    #  概率上算的，玩具数据的目标先验很偏，按未校正的 P_T 看反而低于 SRS，这里改为检查校正后的 margin）
    for name in ('margin', 'entropy', 'gist_margin', 'pmht_b4', 'badge', 'uherding', 'tqs_u', 'margin_r2'):
        if name in rows and rows[name][0] <= np.max(srs_u):
            fails.append((name, '平均 margin 不高于 SRS'))
    if 'margin_prior' in rows:
        pt = uncertainty._resolve_p_train(ctx, 'foreign')
        Pa = uncertainty.prior_adjust(ctx.P_T, uncertainty.em_prior(ctx.P_T, pt), pt)
        ua = unc_margin(Pa)
        S = run('margin_prior', ctx, k, 0)
        if not np.array_equal(S, np.sort(np.argsort(-ua, kind='stable')[:k])):
            fails.append(('margin_prior', '不等于校正后 margin 的 top-k'))
    # 按预测类分层 / 配额的方法：覆盖全部非空预测类
    for name in ('predstrat', 'predstrat_equal', 'predstrat_neyman', 'typiclust_strat', 'fass_quota_equal'):
        if name in rows and rows[name][1] < n_pred:
            fails.append((name, f'只覆盖 {rows[name][1]}/{n_pred} 个预测类'))
    # 几何代表性方法：MSSSD 低于全部 10 个 SRS 样本
    for name in ('fscs', 'maxherding'):
        if name in rows and rows[name][2] >= np.min(srs_m):
            fails.append((name, 'MSSSD 不低于 SRS'))
    assert not fails, fails


# =====================================================================
# 9. 耗时：N=30000、n_F=50000、k=1000
# =====================================================================

def test_timing_N30000_k1000():
    if os.environ.get('INTEG_SKIP_TIMING'):
        raise unittest.SkipTest('INTEG_SKIP_TIMING=1')
    limit = float(os.environ.get('INTEG_TIME_LIMIT', 180))
    ctx = toy(N=30000, n_F=50000, seed=11)
    times, fails = {}, []
    for name in _names():
        ctx.oracle = RecordingOracle(ctx._true_y)
        t = time.time()
        try:
            out = run(name, ctx, 1000, 0, fast=False)       # 默认超参数（重训方法为完整训练配置）
            dt = time.time() - t
            bad = check_output(out, 1000, ctx.N)
            if bad:
                fails.append((name, bad))
        except Exception as e:  # noqa: BLE001
            dt = time.time() - t
            fails.append((name, repr(e)))
        times[name] = dt
        flag = '  <-- 超过上限' if dt > limit else ''
        print(f'    [耗时 N=30000 k=1000] {name:22s} {dt:7.1f} s{flag}', flush=True)
    slow = {n: round(t, 1) for n, t in times.items() if t > limit}
    out_path = os.environ.get('INTEG_TIMING_JSON')
    if out_path:
        with open(out_path, 'w') as fh:
            json.dump({'N': 30000, 'n_F': 50000, 'k': 1000, 'limit_s': limit,
                       'times_s': {n: round(t, 2) for n, t in times.items()},
                       'catalog': {n: {**METHODS[n], 'needs': list(METHODS[n]['needs'])} for n in times}},
                      fh, ensure_ascii=False, indent=1)
    test_timing_N30000_k1000.times = times
    assert not fails, fails
    assert not slow, f'超过 {limit:.0f} s：{slow}'


# =====================================================================

def _main():
    tests = [(n, f) for n, f in sorted(globals().items(), key=lambda kv: getattr(kv[1], '__code__', None)
                                         and kv[1].__code__.co_firstlineno or 0)
             if n.startswith('test_') and callable(f)]
    n_ok = n_fail = n_skip = 0
    for name, f in tests:
        t = time.time()
        try:
            f()
        except unittest.SkipTest as e:
            n_skip += 1
            print(f'SKIP {name}: {e}', flush=True)
            continue
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            import traceback
            traceback.print_exc()
            print(f'FAIL {name} ({time.time() - t:.1f}s): {e!r}', flush=True)
            continue
        n_ok += 1
        print(f'PASS {name} ({time.time() - t:.1f}s)', flush=True)
    print(f'\n通过 {n_ok}，失败 {n_fail}，跳过 {n_skip}')
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(_main())
