# -*- coding: utf-8 -*-
"""审计：不调用 step0A 的任何函数，按规格与 step0A 文档化的随机数方案独立实现：
  (1) 底座重建（格、区 KMeans、Img KMeans）与 frame npz 逐位比较；另测 KMeans 对格顺序的敏感性；
  (2) 结构核对（全部 22 单元、全部抽样）：S0/S1 规则、Betti 贪心分配、成本臂停止规则；
  (3) 逐位重放抽样（REPLAY_UNITS 中的单元，全部场景 × 20 次）；
  (4) 从 draws 逐位重算全部估计量的 oa、ba_ge30、prior_L1_test 等并与 rows_A.csv 逐行比较。
只读；输出写在本脚本目录。"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '2'
sys.dont_write_bytecode = True
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

LE = ('/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/'
      'Alpha_Crops/label_expansion')
sys.path.insert(0, LE)
import guidance_G as G  # noqa: E402  只用 GData（数据读取）
from check3_priors import em_prior  # noqa: E402  规格指定的 EM

A_DIR = '/Users/perng_mini/AlphaCrops_local/verify/step0/partA'
ST0 = '/Users/perng_mini/AlphaCrops_local/results/guidance_G/stage0'
HERE = os.path.dirname(os.path.abspath(__file__))
CTR = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']
N0M = {100: 10, 500: 20, 2000: 25}
SCEN = [('none', 0)] + [('clus', n) for n in N0M] + [('rand', n) for n in N0M]
CODE = {'SRS': 1, 'EXT-p': 2, 'Betti-Admin': 3, 'Betti-Img-ae': 4, 'Betti-Img-s2': 5, 'EXT': 6, 'SPR': 7}
KEY = {'SRS': 'SRS', 'EXT-p': 'EXTp', 'Betti-Admin': 'BAdmin', 'Betti-Img-ae': 'BImgae', 'Betti-Img-s2': 'BImgs2'}
REPLAY_UNITS = [('AT', 0), ('RO', 1), ('SI', 1), ('PT', 0), ('BE', 1)]


def seed(t, s, sc, n0, r):
    return int(hashlib.md5(f'step0A_{t}_{s}_{sc}_{n0}_{r}'.encode()).hexdigest()[:8], 16)


def es(rng, w):
    u = rng.random(len(w))
    return np.argsort(-(np.log(u) / np.asarray(w, float)), kind='stable')


def lap(y):
    n = np.bincount(y, minlength=10).astype(float)
    return (n + 1) / (n.sum() + 10)


def greedy(gamma, n0g, cap, k):
    n = [0] * 8
    for _ in range(k):
        best, bg = None, None
        for g in range(8):
            if n[g] >= cap[g]:
                continue
            m = n0g[g] + n[g]
            v = gamma[g] * ((m + 1) ** -0.5 - (m + 2) ** -0.5)
            if best is None or v > best:
                best, bg = v, g
        n[bg] += 1
    return np.array(n)


def build(D, t, s):
    S = np.load(os.path.join(ST0, f'{t}_s{s}.npz'))
    cand, test, fr = S['cand'], S['test'], S['foreign']
    n = len(cand)
    lat0 = float(S['lat0'])
    lon, lat = np.asarray(D.lon[cand], float), np.asarray(D.lat[cand], float)
    x = lon * 111.32 * np.cos(np.radians(lat0))
    yk = lat * 110.57
    ix, iy = np.floor(x / 10.0).astype(np.int64), np.floor(yk / 10.0).astype(np.int64)
    ids, cell = {}, np.empty(n, np.int64)
    for i, key in enumerate(zip(ix.tolist(), iy.tolist())):      # 首次出现顺序编号（= make_split 的 factorize）
        cell[i] = ids.setdefault(key, len(ids))
    nc = len(ids)
    cnt = np.bincount(cell, minlength=nc)
    cen = np.c_[np.bincount(cell, x, nc), np.bincount(cell, yk, nc)] / cnt[:, None]
    fit = cnt >= 10

    def regions(cen_):
        km = KMeans(8, n_init=10, random_state=0).fit(cen_[fit])
        d = ((cen_[:, None] - km.cluster_centers_[None]) ** 2).sum(-1)
        rc = d.argmin(1)
        rc[fit] = km.labels_
        return rc, km.inertia_
    creg, inert = regions(cen)
    # 敏感性：同一批格按 (ix,iy) 字典序排列后再拟合（= 独立实现的 np.unique 顺序）
    keys = np.array(list(ids.keys()))
    perm = np.lexsort((keys[:, 1], keys[:, 0]))
    fit_perm = fit[perm]
    km2 = KMeans(8, n_init=10, random_state=0).fit(cen[perm][fit_perm])
    d2 = ((cen[:, None] - km2.cluster_centers_[None]) ** 2).sum(-1)
    creg2 = d2.argmin(1)
    tmp = creg2[perm]; tmp[fit_perm] = km2.labels_; creg2[perm] = tmp
    img = {f: KMeans(8, n_init=10, random_state=0).fit(np.asarray(S[f'G_{f}'][:n], np.float64)).labels_ for f in ('ae', 's2')}
    y = np.asarray(D.y)
    yc, yT = y[cand], y[test]
    yF = y[fr]
    nF = np.bincount(yF, minlength=10)
    cw = np.where(nF > 0, len(yF) / (np.count_nonzero(nF) * np.maximum(nF, 1)), 1.0)
    pit = np.bincount(yF, weights=cw[yF], minlength=10) / cw[yF].sum()
    P = {}
    for f in ('ae', 's2'):
        p = S[f'P_T_{f}'][n:].astype(np.float64)
        P[f] = p / p.sum(1, keepdims=True)
    members = [np.flatnonzero(cell == c) for c in range(nc)]
    return dict(S=S, cand=cand, n=n, cell=cell, cnt=cnt, cen=cen, creg=creg, reg=creg[cell], inert=inert,
                creg2=creg2, inert2=km2.inertia_, img=img, yc=yc, yT=yT, pit=pit, P=P, members=members,
                pi_test=np.bincount(yT, minlength=10) / len(yT), pi_frame=np.bincount(yc, minlength=10) / n,
                nT=np.bincount(yT, minlength=10))


def replay(U, sc, n0, sd):
    """按 step0A 文档化的随机数方案独立重放；返回 {键后缀: 候选位置}。"""
    out = {}
    rng = np.random.default_rng(sd)
    regs, cells0 = np.array([], np.int64), np.array([], np.int64)
    if sc == 'rand':
        S0 = rng.choice(U['n'], n0, replace=False)
    elif sc == 'none':
        S0 = np.array([], np.int64)
    else:
        M = N0M[n0]; ncl = n0 // M
        rs = np.bincount(U['reg'], minlength=8)
        ro = es(rng, rs)
        m = 2
        while m < 8 and ((U['cnt'] >= M) & np.isin(U['creg'], ro[:m])).sum() < ncl + 20:
            m += 1
        regs = ro[:m]
        pool = np.flatnonzero((U['cnt'] >= M) & np.isin(U['creg'], regs))
        cells0 = pool[es(rng, U['cnt'][pool])[:ncl]]
        S0 = np.concatenate([rng.choice(U['members'][c], M, replace=False) for c in cells0])
    out['S0'] = S0
    s0m = np.zeros(U['n'], bool); s0m[S0] = True
    arms = ['SRS'] + (['EXT-p'] if sc == 'clus' else []) + ['Betti-Admin', 'Betti-Img-ae', 'Betti-Img-s2']
    for a in arms:
        r = np.random.default_rng([sd, CODE[a]])
        if a in ('SRS', 'EXT-p'):
            el = np.flatnonzero(~s0m if a == 'SRS' else (~s0m & np.isin(U['reg'], regs)))
            tk = r.choice(el, 200, replace=False)
            out[f'{KEY[a]}_k200'], out[f'{KEY[a]}_k50'] = tk, tk[:50]
        else:
            g = U['reg'] if a == 'Betti-Admin' else U['img'][a[-2:]]
            gam = np.bincount(g, minlength=8) / U['n']
            n0g = np.bincount(g[S0], minlength=8)
            cap = np.bincount(g[~s0m], minlength=8)
            a200, a50 = greedy(gam, n0g, cap, 200), greedy(gam, n0g, cap, 50)
            dr = [r.choice(np.flatnonzero((g == gg) & ~s0m), a200[gg], replace=False) if a200[gg] else
                  np.array([], np.int64) for gg in range(8)]
            out[f'{KEY[a]}_k200'] = np.concatenate(dr)
            out[f'{KEY[a]}_k50'] = np.concatenate([dr[gg][:a50[gg]] for gg in range(8)])
    if sc == 'clus':
        inreg = np.isin(U['creg'], regs)
        avail = (U['cnt'] >= 10) & ~np.isin(np.arange(len(U['cnt'])), cells0)
        r = np.random.default_rng([sd, CODE['EXT']])
        pool = np.flatnonzero(avail & inreg)
        ec = pool[es(r, U['cnt'][pool])[:20]]
        out['EXT_B20'] = np.concatenate([r.choice(U['members'][c], 10, replace=False) for c in ec])
        out['EXT_cells'] = ec
        r = np.random.default_rng([sd, CODE['SPR']])
        pool = np.flatnonzero(avail)
        order = pool[es(r, U['cnt'][pool])]
        parts = [r.choice(U['members'][c], 10, replace=False) for c in order[:20]]
        out['SPReq_B20'] = np.concatenate(parts); out['SPReq_cells'] = order[:20]
        rem, m = 20, 0
        for c in order:
            cst = 1 if inreg[c] else 2
            if cst > rem:
                break
            rem -= cst; m += 1
        out['SPR_B20'] = np.concatenate(parts[:m]); out['SPR_cells'] = order[:m]
    out['regions'], out['s0cells'] = regs, cells0
    return out


def score(U, f, pin):
    w = np.where(U['pit'] > 0, pin / np.where(U['pit'] > 0, U['pit'], 1), 0.0)
    ph = (U['P'][f] * w).argmax(1)
    ok = ph == U['yT']
    big = np.flatnonzero(U['nT'] >= 30)
    rec = np.bincount(U['yT'][ok], minlength=10)[big] / U['nT'][big]
    return 100 * ok.mean(), 100 * rec.mean()


def main():
    t00 = time.time()
    D = G.GData()
    rows = pd.read_csv(os.path.join(A_DIR, 'rows_A.csv'))
    rows['k_or_B'] = rows['k_or_B'].astype(int)
    key_cols = ['target', 'split', 'feat', 's0_type', 'n0', 'draw', 'arm', 'k_or_B', 'estimator']
    rows = rows.set_index(key_cols).sort_index()
    assert rows.index.is_unique
    mine, frame_log, struct_fail, replay_log = [], [], [], []
    for t in CTR:
        for s in (0, 1):
            t0 = time.time()
            U = build(D, t, s)
            FR = np.load(os.path.join(A_DIR, 'draws', f'{t}_s{s}_frame.npz'))
            Z = np.load(os.path.join(A_DIR, 'draws', f'{t}_s{s}.npz'))
            fl = dict(target=t, split=s, cell_eq=bool(np.array_equal(U['cell'], FR['cell'])),
                      reg_eq=bool(np.array_equal(U['reg'], FR['reg'])),
                      img_ae_eq=bool(np.array_equal(U['img']['ae'], FR['img_ae'])),
                      img_s2_eq=bool(np.array_equal(U['img']['s2'], FR['img_s2'])),
                      n_cells=len(U['cnt']), n_fit=int((U['cnt'] >= 10).sum()),
                      inertia_main_order=U['inert'], inertia_lex_order=U['inert2'],
                      ari_reg_main_vs_lex=adjusted_rand_score(U['creg'][U['cell']], U['creg2'][U['cell']]),
                      reg_sizes_main=sorted(np.bincount(U['reg'], minlength=8).tolist()),
                      reg_sizes_lex=sorted(np.bincount(U['creg2'][U['cell']], minlength=8).tolist()))
            frame_log.append(fl)
            pos = {int(v): i for i, v in enumerate(U['cand'])}
            yc = U['yc']
            refs = {}
            for f in ('ae', 's2'):
                pem = em_prior(U['P'][f], U['pit'])[0]
                for est, pin in (('E_none', U['pit']), ('E_oracle', U['pi_test']), ('E_EM', pem)):
                    oa, ba = score(U, f, pin)
                    mine.append(dict(target=t, split=s, feat=f, s0_type='ref', n0=0, draw=-1, arm='none', k_or_B=0,
                                     estimator=est, oa=oa, ba_ge30=ba, prior_L1_test=np.abs(pin - U['pi_test']).sum(),
                                     n_labels_S1=0, s0_L1_frame=np.nan, n_regions=np.nan))
            do_replay = (t, s) in REPLAY_UNITS
            for sc, n0 in SCEN:
                for r in range(20):
                    pre = f'{sc}_{n0}_r{r:02d}'
                    sd = seed(t, s, sc, n0, r)
                    if int(Z[f'{pre}_seed']) != sd:
                        struct_fail.append((t, s, pre, 'seed'))
                    loc = lambda k: np.array([pos[int(v)] for v in Z[f'{pre}_{k}']], np.int64)  # noqa: E731
                    S0 = loc('S0')
                    regs = Z[f'{pre}_regions']; c0 = Z[f'{pre}_s0cells']
                    s0m = np.zeros(U['n'], bool); s0m[S0] = True
                    # ---- S0 结构
                    if len(S0) != n0 or len(np.unique(S0)) != n0:
                        struct_fail.append((t, s, pre, 'S0 size'))
                    if sc == 'clus':
                        M = N0M[n0]; ncl = n0 // M
                        cc = np.bincount(U['cell'][S0], minlength=len(U['cnt']))
                        el = U['cnt'] >= M
                        nel = lambda rg: int((el & np.isin(U['creg'], rg)).sum())  # noqa: E731
                        ok = (len(c0) == ncl == len(np.unique(c0)) and (cc[c0] == M).all() and cc.sum() == n0
                              and (U['cnt'][c0] >= M).all() and np.isin(U['creg'][c0], regs).all()
                              and len(regs) >= 2 and len(np.unique(regs)) == len(regs)
                              and nel(regs) >= ncl + 20 and (len(regs) == 2 or nel(regs[:-1]) < ncl + 20))
                        if not ok:
                            struct_fail.append((t, s, pre, 'S0 clus rule'))
                    s0L1 = np.abs(np.bincount(yc[S0], minlength=10) / n0 - U['pi_frame']).sum() if n0 else np.nan
                    nreg = len(regs) if sc == 'clus' else np.nan
                    ests = []
                    if n0:
                        ests.append(('none', 0, 'E_S0', lap(yc[S0]), 0))
                    arms = ['SRS'] + (['EXT-p'] if sc == 'clus' else []) + ['Betti-Admin', 'Betti-Img-ae', 'Betti-Img-s2']
                    for a in arms:
                        S1_200 = loc(f'{KEY[a]}_k200')
                        for k in (50, 200):
                            S1 = loc(f'{KEY[a]}_k{k}')
                            if len(S1) != k or len(np.unique(S1)) != k or s0m[S1].any():
                                struct_fail.append((t, s, pre, a, k, 'S1 size/overlap'))
                            if a == 'EXT-p' and not np.isin(U['reg'][S1], regs).all():
                                struct_fail.append((t, s, pre, a, k, 'EXT-p out'))
                            if a.startswith('Betti'):
                                g = U['reg'] if a == 'Betti-Admin' else U['img'][a[-2:]]
                                gam = np.bincount(g, minlength=8) / U['n']
                                n0g = np.bincount(g[S0], minlength=8)
                                Ng = np.bincount(g[~s0m], minlength=8)
                                ng = greedy(gam, n0g, Ng, k)
                                if not np.array_equal(np.bincount(g[S1], minlength=8), ng):
                                    struct_fail.append((t, s, pre, a, k, 'alloc'))
                                if k == 50 and not set(S1.tolist()) <= set(S1_200.tolist()):
                                    struct_fail.append((t, s, pre, a, k, 'k50 not subset'))
                                # Hájek：权重 N_g/n_g，逐地块循环
                                wsum = np.zeros(10)
                                for i in S1:
                                    wsum[yc[i]] += Ng[g[i]] / ng[g[i]]
                                h = wsum / wsum.sum()
                                e1 = (k * h + 1) / (k + 10)
                            else:
                                if k == 50 and not np.array_equal(S1, S1_200[:50]):
                                    struct_fail.append((t, s, pre, a, k, 'k50 prefix'))
                                e1 = lap(yc[S1])
                            ests.append((a, k, 'E_pool', lap(yc[np.r_[S0, S1]]), k))
                            ests.append((a, k, 'E_S1', e1, k))
                    if sc == 'clus':
                        inreg = np.isin(U['creg'], regs)
                        cells = {}
                        for a, kk in (('EXT', 'EXT'), ('SPR', 'SPR'), ('SPR=', 'SPReq')):
                            S1 = loc(f'{kk}_B20'); cl = Z[f'{pre}_{kk}_cells']; cells[a] = (S1, cl)
                            cc = np.bincount(U['cell'][S1], minlength=len(U['cnt']))
                            ok = (not s0m[S1].any() and len(np.unique(S1)) == len(S1) and (cc[cl] == 10).all()
                                  and cc.sum() == 10 * len(cl) and (U['cnt'][cl] >= 10).all()
                                  and not np.isin(cl, c0).any() and len(np.unique(cl)) == len(cl))
                            if a == 'EXT':
                                ok &= len(cl) == 20 and bool(inreg[cl].all())
                            if a == 'SPR=':
                                ok &= len(cl) == 20
                            if not ok:
                                struct_fail.append((t, s, pre, a, 'cost-arm struct'))
                            ests.append((a, 20, 'E_pool', lap(yc[np.r_[S0, S1]]), len(S1)))
                            ests.append((a, 20, 'E_S1', lap(yc[S1]), len(S1)))
                        spr, sprc = cells['SPR']; eq, eqc = cells['SPR=']
                        cost = np.where(inreg[sprc], 1, 2).sum()
                        nxt_cost = (1 if inreg[eqc[len(sprc)]] else 2) if len(sprc) < 20 else None
                        if not (np.array_equal(sprc, eqc[:len(sprc)]) and np.array_equal(spr, eq[:len(spr)])
                                and cost <= 20 and (nxt_cost is None or cost + nxt_cost > 20)):
                            struct_fail.append((t, s, pre, 'SPR rule'))
                    if do_replay:
                        rp = replay(U, sc, n0, sd)
                        bad = [k for k, v in rp.items() if not np.array_equal(
                            U['cand'][v] if k not in ('regions', 's0cells') and not k.endswith('_cells') else v,
                            Z[f'{pre}_{k}'])]
                        replay_log.append((t, s, pre, len(rp), bad))
                    for a, kb, est, pin, n1 in ests:
                        L1 = np.abs(pin - U['pi_test']).sum()
                        for f in ('ae', 's2'):
                            oa, ba = score(U, f, pin)
                            mine.append(dict(target=t, split=s, feat=f, s0_type=sc, n0=n0, draw=r, arm=a, k_or_B=kb,
                                             estimator=est, oa=oa, ba_ge30=ba, prior_L1_test=L1, n_labels_S1=n1,
                                             s0_L1_frame=s0L1, n_regions=nreg))
            print(t, s, f'{time.time() - t0:.0f}s', flush=True)
    M_ = pd.DataFrame(mine).set_index(key_cols).sort_index()
    M_.to_csv(os.path.join(HERE, 'mine_rows.csv'))
    pd.DataFrame(frame_log).to_csv(os.path.join(HERE, 'frame_check.csv'), index=False)
    miss = M_.index.difference(rows.index); extra = rows.index.difference(M_.index)
    J = M_.join(rows, rsuffix='_A', how='inner')
    res = dict(n_mine=len(M_), n_rows=len(rows), n_join=len(J), n_mine_not_in_rows=len(miss),
               n_rows_not_in_mine=len(extra))
    for c in ('oa', 'ba_ge30', 'prior_L1_test', 's0_L1_frame', 'n_regions', 'n_labels_S1'):
        d = (J[c] - J[c + '_A']).abs()
        res[f'max_abs_{c}'] = float(np.nanmax(d.to_numpy())) if d.notna().any() else 0.0
        res[f'n_diff_{c}'] = int((d > 1e-9).sum())
        res[f'nan_mismatch_{c}'] = int((J[c].isna() != J[c + '_A'].isna()).sum())
    res['struct_fail'] = struct_fail[:20]; res['n_struct_fail'] = len(struct_fail)
    res['replay_n'] = len(replay_log); res['replay_bad'] = [x for x in replay_log if x[4]][:10]
    res['replay_keys_checked'] = int(sum(x[3] for x in replay_log))
    res['time_s'] = round(time.time() - t00)
    json.dump(res, open(os.path.join(HERE, 'recompute_result.json'), 'w'), ensure_ascii=False, indent=1, default=str)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    main()
