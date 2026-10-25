"""EM / reweight / BBSE 的边界情况单元测试。"""
import numpy as np
from prior_shift_ref import *
rng = np.random.default_rng(0); C = 10
def dirichlet_p(n, k_idx, conc=0.3):
    p = np.zeros((n, C)); p[:, k_idx] = rng.dirichlet(np.full(len(k_idx), conc), n); return p

# 1) 未训练类列为 0 + float32 存储
k = np.array([0, 1, 2, 3, 5, 7, 9]); p32 = dirichlet_p(5000, k).astype(np.float32)
p = as64(p32); tr = trained_mask(p); pu = uniform_prior(tr)
print('1) trained 推断 =', np.flatnonzero(tr).tolist(), ' 行和误差(float32→64 归一化后) =', float(np.abs(p.sum(1) - 1).max()))
pi, it, conv, ll = em_prior(p, pu)
print('   EM: 未训练类 π̂ =', pi[~tr].tolist(), ' 有 NaN:', bool(np.isnan(pi).any()), ' Σπ̂ =', round(pi.sum(), 12),
      ' 迭代', it, '收敛', conv, ' loglik 单调:', bool(np.all(np.diff(ll) >= -1e-8)))

# 2) 目标中不存在的训练类：π̂ 衰减但不下溢
true = np.zeros(C); true[[0, 1, 9]] = [0.7, 0.2, 0.1]
n = 20000; yy = rng.choice(C, n, p=true)
lik = np.exp(rng.normal(size=(n, C)) * 1.5); lik[np.arange(n), yy] *= 8; lik[:, ~tr] = 0
p2 = lik / lik.sum(1, keepdims=True)
pi2, it2, conv2, _ = em_prior(p2, pu, max_iter=200)
print('2) 目标缺失类的 π̂ 最小值 =', f'{pi2[tr].min():.2e}', ' 迭代', it2, '收敛', conv2,
      ' 最小正规化 float64 =', f'{np.finfo(float).tiny:.1e}')
pi3, it3, conv3, _ = em_prior(p2, pu, max_iter=5000, tol=1e-12)
print('   max_iter=5000,tol=1e-12: π̂ 最小值 =', f'{pi3[tr].min():.2e}', ' 迭代', it3, ' 有 0:', bool((pi3[tr] == 0).any()))

# 3) reweight 回退：某行全部概率在 π_new=0 的类上
p4 = np.zeros((3, C)); p4[0, 5] = 1.0; p4[1, [0, 5]] = [.5, .5]; p4[2, 7] = 1e-45; p4[2, 5] = 1 - 1e-45
q, nbad = reweight(p4, (np.arange(C) != 5).astype(float), pu)
print('3) 回退行数 =', nbad, ' q 有 NaN:', bool(np.isnan(q).any()), ' 第 1 行 =', np.round(q[1], 3).tolist())
p4_32 = p4.astype(np.float32); print('   float32 下 1e-45 →', p4_32[2, 7], '（subnormal 保留）; 1e-46 →', np.float32(1e-46))

# 4) BBSE：某类从未被预测（混淆矩阵奇异）
n_s = 3000; ys = rng.choice(k, n_s); ps = dirichlet_p(n_s, k, 0.2); ps[np.arange(n_s), ys] += 2; ps /= ps.sum(1, keepdims=True)
ps[:, 3] = 0; ps /= ps.sum(1, keepdims=True)          # 类 3 永远不被预测（但训练中有）
pt_ = dirichlet_p(4000, k, 0.2); pt_[:, 3] = 0; pt_ /= pt_.sum(1, keepdims=True)
tr4 = tr.copy()
pb, cond, w = bbse_prior(ps, ys, np.ones(n_s), pt_, tr4, soft=False)
print(f'4) BBSE 奇异混淆矩阵：cond={cond:.1e}  π̂ 有 NaN: {bool(np.isnan(pb).any())}  Σ={pb.sum():.6f}  w 负值个数={int((w < 0).sum())}')
