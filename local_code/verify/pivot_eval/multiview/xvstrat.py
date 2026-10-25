# XV-Strat：两视图联合预测格 (ŷ_AE, ŷ_S2) 分层选样（不读标签）。可直接改写成 samplers 接口（ctx 需要两份 P_T）。
import numpy as np
import sys
sys.dont_write_bytecode = True
from samplers.designs import alloc

def xv_strata(Pa, Ps, n_min=None):
    """层编码：一致格 (a,a) 与分歧格 (a,b)；小于 n_min 的一致格并入"一致-其他"，分歧格并入"分歧-其他"。"""
    N, C = Pa.shape
    n_min = n_min or max(30, N // 1000)
    ya, ys = Pa.argmax(1), Ps.argmax(1)
    cell = ya * C + ys
    cnt = np.bincount(cell, minlength=C * C)
    agree = ya == ys
    code = cell.copy()
    small = cnt[cell] < n_min
    code[small & agree] = C * C
    code[small & ~agree] = C * C + 1
    uniq, h = np.unique(code, return_inverse=True)
    dis = np.array([(u < C * C and (u // C) != (u % C)) or u == C * C + 1 for u in uniq])
    return h, len(uniq), dis

def select_xvstrat(Pa, Ps, k, seed, lam=0.0, m_min=2):
    rng = np.random.default_rng(seed)
    h, H, dis = xv_strata(Pa, Ps)
    N_h = np.bincount(h, minlength=H)
    w = np.sqrt(N_h) * (1.0 + lam * dis)
    n_h = alloc(N_h, w, k, m_min)
    parts = [rng.choice(np.flatnonzero(h == g), n_h[g], replace=False) for g in range(H) if n_h[g] > 0]
    return np.sort(np.concatenate(parts))

def select_predstrat(P, k, seed, m_min=2):
    rng = np.random.default_rng(seed)
    h = P.argmax(1); H = P.shape[1]
    N_h = np.bincount(h, minlength=H)
    n_h = alloc(N_h, np.sqrt(N_h), k, m_min)
    parts = [rng.choice(np.flatnonzero(h == g), n_h[g], replace=False) for g in range(H) if n_h[g] > 0]
    return np.sort(np.concatenate(parts))

def select_oracle_sqrt(y, k, seed, m_min=2, C=10):
    return select_predstrat(np.eye(C)[y], k, seed, m_min)
