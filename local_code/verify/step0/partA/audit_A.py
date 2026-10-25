# -*- coding: utf-8 -*-
"""
第 0 步 A 部分的自查（只读 rows_A.csv 与 draws/）：从保存的 S0、S1 全局索引独立重算先验与 OA/BA，与 rows_A.csv 逐行比较；
并独立核对 draws 的结构（不重叠、S1 在 S0 所在区、SPR 为 SPR= 前缀、簇内地块数、成本不超预算）。
另：重建底座（重新拟合区与 Img 的 KMeans），按 seed 方案对每单元每场景 1 次抽样重放 step0A 的抽样函数，与保存的索引逐位比较。
"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import step0A as A  # noqa: E402  （step0A 本身不在 import 时运行任何东西）
from check3_priors import reweight, laplace_prior, em_prior  # noqa: E402
import guidance_G as G  # noqa: E402

rng_pick = np.random.default_rng(20260929)
rows = pd.read_csv(os.path.join(HERE, 'rows_A.csv'))
D = G.GData()
y = np.asarray(D.y)
out = []
n_cmp = 0
max_d = dict(oa=0.0, ba=0.0, L1=0.0)
struct_fail = []
n_replay = [0]
for tgt in A.COUNTRIES:
    for split in A.SPLITS:
        S = np.load(os.path.join(A.STAGE0, f'{tgt}_s{split}.npz'))
        cand, test, fr = S['cand'], S['test'], S['foreign']
        n_cand = len(cand)
        yT = y[test]
        big = np.flatnonzero(np.bincount(yT, minlength=10) >= 30)
        cw = A.class_weights_from(y[fr])
        pit = np.bincount(y[fr], weights=cw[y[fr]], minlength=10) / cw[y[fr]].sum()
        pi_test = np.bincount(yT, minlength=10) / len(yT)
        P = {}
        for f in ('ae', 's2'):
            p = S[f'P_T_{f}'][n_cand:].astype(np.float64)
            P[f] = p / p.sum(1, keepdims=True)
        Z = np.load(os.path.join(HERE, 'draws', f'{tgt}_s{split}.npz'))
        FR = np.load(os.path.join(HERE, 'draws', f'{tgt}_s{split}_frame.npz'))
        assert np.array_equal(FR['cand'], cand)
        pos_of = {int(v): i for i, v in enumerate(cand)}
        reg = FR['reg']
        grp = {'Betti-Admin': reg, 'Betti-Img-ae': FR['img_ae'], 'Betti-Img-s2': FR['img_s2']}
        cellv = FR['cell']
        R_ = rows[(rows.target == tgt) & (rows.split == split)]
        # 参照行
        for f in ('ae', 's2'):
            for est, pin in (('E_none', pit), ('E_oracle', pi_test), ('E_EM', em_prior(P[f], pit)[0])):
                q, _ = reweight(P[f], pin, pit)
                oa, ba, _, _ = G._scores(yT, q.argmax(1), big)
                r = R_[(R_.s0_type == 'ref') & (R_.feat == f) & (R_.estimator == est)].iloc[0]
                max_d['oa'] = max(max_d['oa'], abs(oa - r.oa)); max_d['ba'] = max(max_d['ba'], abs(ba - r.ba_ge30))
                n_cmp += 1
        for s0type, n0 in A.SCENARIOS:
            for r_ in rng_pick.choice(20, 3, replace=False):   # 每单元每场景随机抽 3 次抽样逐行重算
                pre = f'{s0type}_{n0}_r{r_:02d}'
                s0 = np.array([pos_of[int(v)] for v in Z[f'{pre}_S0']], np.int64)
                regions = Z[f'{pre}_regions']
                sub = R_[(R_.s0_type == s0type) & (R_.n0 == n0) & (R_.draw == r_)]
                todo = []
                if n0:
                    todo.append(('none', 0, 'E_S0', laplace_prior(y[cand][s0], 10)))
                arms = ['SRS'] + (['EXT-p'] if s0type == 'clus' else []) + A.BETTI_ARMS
                for arm in arms:
                    for k in A.KS:
                        s1 = np.array([pos_of[int(v)] for v in Z[f'{pre}_{A.KEY_SAFE[arm]}_k{k}']], np.int64)
                        if set(s1.tolist()) & set(s0.tolist()) or len(s1) != k:
                            struct_fail.append((tgt, split, pre, arm, k, 'overlap/len'))
                        if arm == 'EXT-p' and not np.isin(reg[s1], regions).all():
                            struct_fail.append((tgt, split, pre, arm, k, 'EXT-p 区外'))
                        todo.append((arm, k, 'E_pool', laplace_prior(y[cand][np.r_[s0, s1]], 10)))
                        if arm.startswith('Betti'):
                            g = grp[arm]
                            s0m = np.zeros(n_cand, bool); s0m[s0] = True
                            Ng = np.bincount(g[~s0m], minlength=8).astype(float)
                            ng = np.bincount(g[s1], minlength=8).astype(float)
                            w = Ng[g[s1]] / ng[g[s1]]
                            h = np.bincount(y[cand][s1], weights=w, minlength=10) / w.sum()
                            todo.append((arm, k, 'E_S1', (k * h + 1) / (k + 10)))
                        else:
                            todo.append((arm, k, 'E_S1', laplace_prior(y[cand][s1], 10)))
                if s0type == 'clus':
                    s0cells = set(Z[f'{pre}_s0cells'].tolist())
                    for arm, key in (('EXT', 'EXT'), ('SPR', 'SPR'), ('SPR=', 'SPReq')):
                        s1 = np.array([pos_of[int(v)] for v in Z[f'{pre}_{key}_B20']], np.int64)
                        cells = Z[f'{pre}_{key}_cells']
                        if set(s1.tolist()) & set(s0.tolist()) or set(cells.tolist()) & s0cells:
                            struct_fail.append((tgt, split, pre, arm, 'overlap'))
                        if not (np.bincount(cellv[s1], minlength=len(FR['cell_cnt']))[cells] == 10).all():
                            struct_fail.append((tgt, split, pre, arm, 'per-cell'))
                        in_reg = np.isin(FR['cell_reg'][cells], regions)
                        if arm == 'EXT' and not in_reg.all():
                            struct_fail.append((tgt, split, pre, arm, 'EXT 区外'))
                        if arm == 'SPR' and np.where(in_reg, 1, 2).sum() > 20:
                            struct_fail.append((tgt, split, pre, arm, 'cost'))
                        todo.append((arm, 20, 'E_pool', laplace_prior(y[cand][np.r_[s0, s1]], 10)))
                        todo.append((arm, 20, 'E_S1', laplace_prior(y[cand][s1], 10)))
                    spr = Z[f'{pre}_SPR_B20']; eq = Z[f'{pre}_SPReq_B20']
                    if not np.array_equal(spr, eq[:len(spr)]):
                        struct_fail.append((tgt, split, pre, 'SPR', 'prefix'))
                for arm, kb, est, pin in todo:
                    L1 = np.abs(pin - pi_test).sum()
                    for f in ('ae', 's2'):
                        q, _ = reweight(P[f], pin, pit)
                        oa, ba, _, _ = G._scores(yT, q.argmax(1), big)
                        rr = sub[(sub.arm == arm) & (sub.k_or_B == kb) & (sub.estimator == est) & (sub.feat == f)]
                        assert len(rr) == 1, (tgt, split, pre, arm, kb, est, f, len(rr))
                        rr = rr.iloc[0]
                        max_d['oa'] = max(max_d['oa'], abs(oa - rr.oa))
                        max_d['ba'] = max(max_d['ba'], abs(ba - rr.ba_ge30))
                        max_d['L1'] = max(max_d['L1'], abs(L1 - rr.prior_L1_test))
                        n_cmp += 1
        # 重放抽样（逐位）
        F = A.build_frame(D, tgt, split)
        for key in ('cell', 'cell_reg', 'reg'):
            if not np.array_equal(F[key], FR[key]):
                struct_fail.append((tgt, split, 'frame', key))
        for f in ('ae', 's2'):
            if not np.array_equal(F['img'][f], FR[f'img_{f}']):
                struct_fail.append((tgt, split, 'frame', f'img_{f}'))
        for s0type, n0 in A.SCENARIOS:
            r_ = int(rng_pick.integers(20))
            pre = f'{s0type}_{n0}_r{r_:02d}'
            seed = A.seed_of(tgt, split, s0type, n0, r_)
            n_replay[0] += 1
            if seed != int(Z[f'{pre}_seed']):
                struct_fail.append((tgt, split, pre, 'seed'))
            S0, info = A.draw_S0(F, s0type, n0, np.random.default_rng(seed))
            ok = np.array_equal(cand[S0], Z[f'{pre}_S0'])
            for arm in ['SRS'] + (['EXT-p'] if s0type == 'clus' else []) + A.BETTI_ARMS:
                res = A.arm_k(F, arm, S0, info, seed)
                ok &= all(np.array_equal(cand[res[k][0]], Z[f'{pre}_{A.KEY_SAFE[arm]}_k{k}']) for k in A.KS)
            if s0type == 'clus':
                cres = A.arms_cost(F, S0, info, seed)
                for arm in ('EXT', 'SPR', 'SPR='):
                    ok &= np.array_equal(cand[cres[arm][0]], Z[f'{pre}_{A.KEY_SAFE[arm]}_B20'])
            if not ok:
                struct_fail.append((tgt, split, pre, 'replay'))
        print(tgt, split, flush=True)
msg = (f'自查：逐行独立重算 {n_cmp} 行（每单元每场景随机 3 次抽样 + 参照行）；最大 |差| OA {max_d["oa"]:.2e}、'
       f'BA {max_d["ba"]:.2e}、先验 L1 {max_d["L1"]:.2e}；重放抽样 {n_replay[0]} 次（逐位比较 S0 与全部臂的 S1）；结构/重放失败 {len(struct_fail)} 条 {struct_fail[:5]}')
print(msg)
open(os.path.join(HERE, 'audit_A.txt'), 'w').write(msg + '\n')
