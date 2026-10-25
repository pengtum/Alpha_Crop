"""
选样方法的统一接口（见 SAMPLING_SURVEY_2026-09-24.md 第 3.0 节）。

每个方法实现 select(ctx, k, seed) -> np.ndarray[int64]：长度 k、无重复、升序的目标候选池下标。
概率设计另外写 ctx.meta[name] = {'pi': (N,) 一阶包含概率, 'prior_idx': 可用于设计无偏先验估计的下标}。
选样函数不得读取候选池标签；只有明确允许的方法（多轮 margin、Hybrid、神谕）可以调用 ctx.oracle(idx)。
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

REGISTRY = {}


def register(name):
    """把选样函数登记到 REGISTRY[name]。"""
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


@dataclass
class Ctx:
    N: int                        # 目标候选池大小
    C: int                        # 类别数（10）
    T_ae: np.ndarray              # (N, 32) AE PCA（在外国 ∪ 目标上拟合，按方差降序，未白化）
    T_s2: np.ndarray              # (N, 32) S2 PCA
    T_xy: np.ndarray              # (N, 2) 公里坐标
    T_nuts3: np.ndarray           # (N,) NUTS3 编码
    F_ae: np.ndarray              # (n_F, 32) 外国池 AE PCA（同一 PCA）
    F_s2: np.ndarray              # (n_F, 32)
    F_xy: np.ndarray              # (n_F, 2) 外国池公里坐标（与目标同一投影；跨国距离只作参考）
    F_y: np.ndarray               # (n_F,) 外国池标签
    F_country: np.ndarray         # (n_F,)
    P_T: np.ndarray               # (N, C) 外国 LightGBM 对目标池的概率（已校准）
    P_T_loco: Optional[np.ndarray] = None   # (10, N, C) 留一国委员会的概率（TQS 用）
    dom: Optional[np.ndarray] = None        # (N,) 域判别器 P(目标 | x)，交叉拟合
    P_F_oof: Optional[np.ndarray] = None    # (n_F, C) 外国池的袋外概率（BAIT 等用）
    oracle: Optional[Callable] = None       # oracle(idx) -> 真实标签；只供明确允许的方法使用
    meta: dict = field(default_factory=dict)


def X_of(ctx, space='ae', d=16, geo_w=0.0):
    """选样用的特征矩阵：取前 d 个主成分，整体缩放（保留主成分相对方差），可拼接加权的标准化坐标。"""
    def part(A):
        A = A[:, :d].astype(np.float64)
        return A / np.sqrt(A.var(0).sum() / d)
    if space == 'ae':
        X = part(ctx.T_ae)
    elif space == 's2':
        X = part(ctx.T_s2)
    elif space == 'ae+s2':
        X = np.hstack([part(ctx.T_ae), part(ctx.T_s2)])
    elif space == 'geo':
        X = np.zeros((ctx.N, 0))
    else:
        raise ValueError(space)
    if geo_w > 0 or space == 'geo':
        xy = ctx.T_xy - ctx.T_xy.mean(0)
        xy = xy / np.sqrt(xy.var(0).sum() / 2)
        X = np.hstack([X, (geo_w if geo_w > 0 else 1.0) * xy])
    return X


def X_of_foreign(ctx, space='ae', d=16):
    """与 X_of 同一缩放的外国池特征（缩放系数取自目标池，保证两者在同一空间）。"""
    T = {'ae': ctx.T_ae, 's2': ctx.T_s2}[space][:, :d].astype(np.float64)
    F = {'ae': ctx.F_ae, 's2': ctx.F_s2}[space][:, :d].astype(np.float64)
    s = np.sqrt(T.var(0).sum() / d)
    return F / s


def unc_margin(P):
    """1 −（最大概率 − 第二大概率），越大越不确定。"""
    s = np.sort(P, axis=1)
    return 1.0 - (s[:, -1] - s[:, -2])


def unc_ent(P):
    """归一化熵。"""
    C = P.shape[1]
    return -(P * np.log(np.clip(P, 1e-12, None))).sum(1) / np.log(C)


def finalize(idx, k, N):
    """检查并返回排序后的下标：长度 k、无重复、在 [0, N) 内。"""
    idx = np.asarray(idx, dtype=np.int64)
    assert idx.ndim == 1 and len(idx) == k, (len(idx), k)
    assert len(np.unique(idx)) == k, 'duplicate indices'
    assert idx.min() >= 0 and idx.max() < N
    return np.sort(idx)


def make_toy_ctx(N=3000, n_F=5000, C=10, seed=0):
    """合成的测试用上下文：目标与外国各有簇结构，外国概率带噪声。"""
    r = np.random.default_rng(seed)
    centers = r.normal(size=(C, 32)) * 3
    yT = r.choice(C, N, p=np.r_[0.4, 0.2, 0.1, 0.08, 0.07, 0.05, 0.04, 0.03, 0.02, 0.01])
    yF = r.choice(C, n_F)
    T = centers[yT] + r.normal(size=(N, 32)) + 0.8  # 目标整体平移（域偏移）
    F = centers[yF] + r.normal(size=(n_F, 32))
    logits = -((T[:, None, :8] - centers[None, :, :8]) ** 2).sum(-1) / 20 + r.normal(scale=0.5, size=(N, C))
    P = np.exp(logits - logits.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    xy = r.uniform(0, 300, size=(N, 2))
    ctx = Ctx(N=N, C=C, T_ae=T.astype(np.float32), T_s2=(T[:, ::-1] + r.normal(size=(N, 32))).astype(np.float32),
              T_xy=xy, T_nuts3=(xy[:, 0] // 60).astype(int) * 10 + (xy[:, 1] // 60).astype(int),
              F_ae=F.astype(np.float32), F_s2=(F[:, ::-1] + r.normal(size=(n_F, 32))).astype(np.float32),
              F_xy=r.uniform(-600, 0, size=(n_F, 2)), F_y=yF, F_country=r.choice(10, n_F), P_T=P)
    ctx._true_y = yT  # 只供测试与神谕使用
    ctx.oracle = lambda idx: yT[np.asarray(idx)]
    return ctx
