"""
G 实验运行器改动的原型（只放在 verify/，不改 CODE）。
- guidance_models(...)：阶段 0 新增，AE 与 S2 走同一个函数，唯一的参数差别是 feat。
- build_ctx_guided(...)：按 (几何来源, 预测来源) 构造 ctx；与指导无关的字段一律置 None（防止方法偷读）。
- G_HP / FAMILY：G 实验的超参数覆盖与方法分组。
"""
import inspect
import numpy as np

import os
import sys
CODE = "/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion"
sys.path.insert(0, CODE)
import samplers.base as B  # noqa: E402

# 条件名 → (几何来源, 预测来源)。'ref' = 各条件共用的参照概率（只在"几何类 + 需要 P_T 做次序"的情形使用）
GUIDANCE = {
    'AE': ('ae', 'ae'),
    'S2': ('s2', 's2'),
    'XY': ('xy', None),          # 只跑纯几何方法；P_T=None，任何读 P_T 的方法会直接报错
    'AEg_S2p': ('ae', 's2'),     # 通路交叉
    'S2g_AEp': ('s2', 'ae'),
}
FAMILY = {
    'geom': ['lpm2', 'fscs', 'kcenter', 'typiclust', 'kherding'],
    'pred': ['margin', 'predstrat_srs'],
    # gist_maxmin：α=0 时 P_T 的 margin 仍决定贪心扫描顺序与补足顺序（实测换 P_T 后重合 2–20%），归入混合组
    'both': ['gist_maxmin', 'gist_margin', 'predstrat', 'badge', 'fass_quota'],
}
G_HP = {'lpm2': {'geo_w': 0.0}, 'predstrat': {'geo_w': 0.0}}


def check_geo_w(methods, hp=G_HP):
    """凡签名里有 geo_w 且会用到坐标的方法，G 实验必须显式传 geo_w=0。"""
    bad = []
    for m in methods:
        sig = inspect.signature(B.REGISTRY[m])
        if 'geo_w' in sig.parameters and sig.parameters['geo_w'].default != 0.0 and hp.get(m, {}).get('geo_w') != 0.0:
            if m != 'predstrat_srs':   # 层内 SRS，不构造 X，geo_w 无效
                bad.append(m)
    return bad


def build_ctx_guided(S, pool_pos, guidance, F_y, xy_pool=None):
    """S：阶段 0 的 npz 字典，需含 T_ae/F_ae/T_s2/F_s2/P_T_ae/P_T_s2（均为 allT 顺序）。"""
    geo, pred = GUIDANCE[guidance]
    if geo in ('ae', 's2'):
        T_g = np.ascontiguousarray(S[f'T_{geo}'][pool_pos])
        F_g = S[f'F_{geo}']
    elif geo == 'xy':
        T_g = xy_pool - xy_pool.mean(0)          # 居中的公里坐标（2 列），放在 T_ae 槽位
        F_g = None
    else:
        raise ValueError(geo)
    P = None if pred is None else np.asarray(S[f'P_T_{pred}'][pool_pos], dtype=np.float64)
    return B.Ctx(N=len(pool_pos), C=10, T_ae=T_g, T_s2=None, T_xy=None, T_nuts3=None, F_ae=F_g, F_s2=None,
                 F_xy=None, F_y=F_y, F_country=None, P_T=P, P_T_loco=None, dom=None, P_F_oof=None)


# ---------------------------------------------------------------- 阶段 0 新增（与 stageA.stage0_one 同一套外国池、划分与 PCA）
def fold_groups(others, tgt, split, n_groups=5):
    """外国国家分 5 组；只依赖 (tgt, split)，AE 与 S2 共用同一分组。"""
    import hashlib
    h = int(hashlib.md5(f'{tgt}_s{split}_groups'.encode()).hexdigest(), 16) % (2 ** 32)
    perm = np.random.default_rng(h).permutation(np.asarray(others))
    return [list(g) for g in np.array_split(perm, n_groups)]


def fit_temp(P_oof, y):
    from scipy.optimize import minimize_scalar
    z = np.log(np.clip(P_oof.astype(np.float64), 1e-12, None))

    def nll(s):
        l_ = s * z
        l_ -= l_.max(1, keepdims=True)
        q = np.exp(l_)
        q /= q.sum(1, keepdims=True)
        return -np.log(np.clip(q[np.arange(len(y)), y], 1e-300, None)).mean()
    return float(minimize_scalar(nll, bounds=(0.05, 20), method='bounded').x)


def apply_temp(P, s):
    z = s * np.log(np.clip(P.astype(np.float64), 1e-12, None))
    z -= z.max(1, keepdims=True)
    Q = np.exp(z)
    return Q / Q.sum(1, keepdims=True)


def guidance_models(Xfeat, y, ctry, foreign, allT, groups, fit_predict, class_weights_from, threads, trees):
    """一种特征的外国全量模型 + 分组留出温度。AE 与 S2 调用同一函数，只换 Xfeat。"""
    yF = y[foreign]
    cw = class_weights_from(yF)
    P_raw, _ = fit_predict(Xfeat[foreign], yF, cw[yF], Xfeat[allT], threads, 0, trees)
    fc = ctry[foreign]
    P_oof = np.zeros((len(foreign), 10), np.float32)
    for g in groups:
        te = np.isin(fc, g)
        P_oof[te], _ = fit_predict(Xfeat[foreign[~te]], yF[~te], cw[yF[~te]], Xfeat[foreign[te]], threads, 0, trees)
    s = fit_temp(P_oof, yF)
    return apply_temp(P_raw, s).astype(np.float32), s, P_oof
