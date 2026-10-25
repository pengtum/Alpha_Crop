"""
D3：不训练，只看"选出样本的构成"。比较 SRS、predstrat_srs（G 已存选样）、oracle_sqrt（按真实类 sqrt 配额）
与"两阶段种子扩展"（ACE，本提案）的若干变体：
  ACE：第 1 批 SRS k1=ρk 揭示标签 → 估计 π̂ → sqrt 配额 → 对欠额类，从该类种子出发在邻域中逐个扩展取标
       （round-robin 于种子之间；新取到的同类地块成为新种子 = 自适应群抽样规则）。
  邻域：'ae'（AE 16 维特征 kNN）、'ae_geo5'（种子 5 km 内按 AE 距离排序）、'geo'（纯地理 kNN）。
评价（只用于诊断）：blind 类 = 测试集 ≥30 个地块且外国 AE 模型在候选池召回 <0.2 的类。
成本代理：n_sites = 标签落入的不同 2 km 格数（目视解译的"开图/定位"次数）。
"""
import os, glob, sys
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
sys.path.insert(0, '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion')
from samplers.designs import alloc

G = '/Users/perng_mini/AlphaCrops_local/results/guidance_G'
Cd = '/Users/perng_mini/AlphaCrops_local/data/cache_guidance_G'
y = np.load(f'{Cd}/y.npy'); lon = np.load(f'{Cd}/lon.npy'); lat = np.load(f'{Cd}/lat.npy')

def km_xy(lo, la, lat0):
    return np.c_[lo * 111.32 * np.cos(np.radians(lat0)), la * 110.57]

def pool_positions(S, split, rep):
    n_cand = len(S['cand'])
    rng = np.random.default_rng(10_000 * split + 100 * rep + 7)
    return np.sort(rng.choice(n_cand, min(30_000, n_cand), replace=False))

class Nbr:
    def __init__(self, Z, xy):
        self.Z, self.xy = Z, xy
        self.tz, self.tg = cKDTree(Z), cKDTree(xy)
    def order(self, i, kind, m=400, R=5.0):
        if kind == 'ae':
            return self.tz.query(self.Z[i], k=m)[1]
        if kind == 'geo':
            return self.tg.query(self.xy[i], k=m)[1]
        if kind.startswith('ae_geo'):
            cand = np.array(self.tg.query_ball_point(self.xy[i], r=R))
            if len(cand) == 0:
                return np.array([], int)
            d = np.linalg.norm(self.Z[cand] - self.Z[i], axis=1)
            return cand[np.argsort(d)][:m]
        raise ValueError(kind)

def ace(yP, nb, k, rho, kind, rng, R=5.0, seed_rule='all'):
    N = len(yP)
    k1 = int(round(rho * k))
    s1 = rng.choice(N, k1, replace=False)
    lab = np.zeros(N, bool); lab[s1] = True
    order = list(s1)
    cnt = np.bincount(yP[s1], minlength=10)
    pi_hat = cnt / k1
    cap = np.where(cnt > 0, np.maximum(np.ceil(pi_hat * N), cnt), 0).astype(int)
    quota = alloc(cap, np.sqrt(pi_hat), k)
    deficit = np.maximum(quota - cnt, 0)
    rest = k - k1
    d_alloc = alloc(np.where(deficit > 0, rest, 0), deficit.astype(float), min(rest, int((deficit > 0).sum()) * rest)) \
        if deficit.sum() > 0 else np.zeros(10, int)
    # 按欠额比例把第 2 批 rest 个名额分给欠额类
    if deficit.sum() > 0:
        d_alloc = alloc(np.where(deficit > 0, rest, 0), deficit.astype(float), rest)
    hits, spent_exp = 0, 0
    for c in np.argsort(-d_alloc):
        b = int(d_alloc[c])
        if b == 0:
            continue
        seeds = [i for i in order if yP[i] == c]
        if not seeds:
            continue
        lists = {s: iter(nb.order(s, kind, R=R)) for s in seeds}
        queue = list(seeds)
        used = 0; qi = 0
        while used < b and queue:
            s = queue[qi % len(queue)]
            nxt = None
            for j in lists[s]:
                if not lab[j]:
                    nxt = j; break
            if nxt is None:
                queue.remove(s)
                continue
            lab[nxt] = True; order.append(nxt); used += 1; spent_exp += 1
            if yP[nxt] == c:
                hits += 1
                lists[nxt] = iter(nb.order(nxt, kind, R=R)); queue.append(nxt)
            qi += 1
    # 剩余名额（没有种子的类、邻域耗尽）用 SRS 补齐
    left = k - lab.sum()
    if left > 0:
        free = np.flatnonzero(~lab)
        add = rng.choice(free, left, replace=False)
        lab[add] = True; order += list(add)
    return np.flatnonzero(lab), hits / max(spent_exp, 1), spent_exp

def metrics(sel, yP, pi, blind, big, xy):
    comp = np.bincount(yP[sel], minlength=10) / len(sel)
    sq = np.sqrt(pi); sq = sq / sq.sum()
    cell = np.floor(xy[sel] / 2.0).astype(int)
    return dict(L1_sqrt=np.abs(comp - sq).sum(), L1_prior=np.abs(comp - pi).sum(),
                n_blind=int(np.isin(yP[sel], blind).sum()),
                n_big_ge5=int(sum((yP[sel] == c).sum() >= 5 for c in big)),
                min_big=int(min((yP[sel] == c).sum() for c in big)),
                n_sites2km=len({tuple(r) for r in cell}))

rows = []
for f0 in sorted(glob.glob(f'{G}/stage0/*.npz')):
    tgt, sp = os.path.basename(f0)[:-4].split('_s'); sp = int(sp)
    S = np.load(f0)
    test = S['test']; yT = y[test]
    big = np.flatnonzero(np.bincount(yT, minlength=10) >= 30)
    rep = 0
    pp = pool_positions(S, sp, rep)
    idx = S['allT'][pp]; yP = y[idx]
    xy = km_xy(lon[idx], lat[idx], float(S['lat0']))
    Z = S['G_ae'][pp][:, :16].astype(np.float64); Z /= np.sqrt(Z.var(0).sum() / 16)
    P = S['P_T_ae'][pp]; yh = P.argmax(1)
    pi = np.bincount(yP, minlength=10) / len(yP)
    rec = np.array([(yh[yP == c] == c).mean() if (yP == c).any() else np.nan for c in range(10)])
    blind = np.array([c for c in big if rec[c] < 0.2])
    nb = Nbr(Z, xy)
    for k in (50, 200, 400):
        # G 已存选样
        for cond, meth in (('none', 'srs'), ('AE', 'predstrat_srs'), ('S2', 'predstrat_srs'), ('AE', 'typiclust')):
            fz = f'{G}/sel/{tgt}_s{sp}_r{rep}_{cond}_{meth}_k{k}.npz'
            if os.path.exists(fz):
                sel = np.load(fz)['sel_pos']
                rows.append(dict(target=tgt, split=sp, k=k, method=f'G_{cond}_{meth}', draw=0, hit=np.nan,
                                 n_blind_classes=len(blind), **metrics(sel, yP, pi, blind, big, xy)))
        for d in range(5):
            rng = np.random.default_rng(777 + 31 * d + k + 1000 * sp)
            sel = rng.choice(len(yP), k, replace=False)
            rows.append(dict(target=tgt, split=sp, k=k, method='srs', draw=d, hit=np.nan,
                             n_blind_classes=len(blind), **metrics(sel, yP, pi, blind, big, xy)))
            Nc = np.bincount(yP, minlength=10)
            nq = alloc(Nc, np.sqrt(Nc), k)
            sel = np.concatenate([rng.choice(np.flatnonzero(yP == c), nq[c], replace=False) for c in range(10) if nq[c]])
            rows.append(dict(target=tgt, split=sp, k=k, method='oracle_sqrt', draw=d, hit=np.nan,
                             n_blind_classes=len(blind), **metrics(sel, yP, pi, blind, big, xy)))
            for kind in ('ae', 'ae_geo5', 'geo'):
                for rho in (0.5,):
                    sel, hit, ne = ace(yP, nb, k, rho, kind, np.random.default_rng(777 + 31 * d + k + 1000 * sp))
                    rows.append(dict(target=tgt, split=sp, k=k, method=f'ace_{kind}_rho{rho}', draw=d, hit=hit,
                                     n_exp=ne, n_blind_classes=len(blind), **metrics(sel, yP, pi, blind, big, xy)))
    print(tgt, sp, 'blind', blind, flush=True)
df = pd.DataFrame(rows)
df.to_csv('/Users/perng_mini/AlphaCrops_local/verify/pivot_eval/geo_cost/d3_ace_composition.csv', index=False)
pd.set_option('display.width', 250)
for k in (50, 200, 400):
    print(f'\n=== k={k}：11 国 × 2 划分平均（随机方法 5 次抽样平均）===')
    g = df[df.k == k].groupby('method')[['L1_sqrt', 'L1_prior', 'n_blind', 'n_big_ge5', 'min_big', 'n_sites2km', 'hit']].mean()
    print(g.round(3).to_string())
