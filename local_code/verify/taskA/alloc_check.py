import sys, itertools
CODE = "/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion"
sys.path.insert(0, CODE)
import numpy as np, pandas as pd
import pilot_betti as pb
rng = np.random.default_rng(0)
# 1) 小规模穷举：贪心结果 == pb.utility 在 {sum n = b, 0<=n<=cap} 上的最大值
worst = 0; nt = 0
for trial in range(300):
    G = rng.integers(2, 5); cap = rng.integers(1, 8, G); gamma = cap / cap.sum()
    b = int(rng.integers(1, cap.sum() + 3))
    n = pb.allocate_parcels(gamma, cap, b)
    best = -np.inf
    for comb in itertools.product(*[range(c + 1) for c in cap]):
        comb = np.array(comb)
        if comb.sum() == min(b, cap.sum()):
            best = max(best, pb.utility(comb.astype(float), gamma))
    worst = max(worst, best - pb.utility(n.astype(float), gamma)); nt += 1
    assert n.sum() == min(b, cap.sum()) and np.all(n <= cap)
print('brute-force trials', nt, 'max utility gap (best - greedy):', worst)
# 2) 真实分组规模（PT/SE/SK/LV 的 NUTS3 候选池占比）上：交换条件 + 与连续解 n_g+1 = c*gamma^(2/3) 对比
from sklearn.model_selection import train_test_split
import common
df = pd.read_parquet(common.DATASET_V2, columns=['parcel_id', 'country', 'has_ae', 'has_s2'])
df = df[df.has_ae & df.has_s2].reset_index(drop=True)
nuts = pd.read_parquet(common.DATA_DIR + '/parcel_nuts.parquet', columns=['parcel_id', 'nuts3'])
df = df.merge(nuts, on='parcel_id', how='left')
def cont(gamma, cap, b):   # 连续松弛 + 上下界：n_g = clip(c*gamma^(2/3) - 1, 0, cap)，二分 c
    lo, hi = 0, 1e9
    for _ in range(200):
        c = (lo + hi) / 2; n = np.clip(c * gamma ** (2 / 3) - 1, 0, cap)
        lo, hi = (c, hi) if n.sum() < b else (lo, c)
    return n
for tgt in ['PT', 'SE', 'SK', 'LV']:
    t_idx = np.flatnonzero(df.country.to_numpy() == tgt)
    pool, _ = train_test_split(t_idx, test_size=0.3, random_state=42)
    g = pd.factorize(df.nuts3.to_numpy()[pool])[0]; cap = np.bincount(g); gamma = cap / len(g)
    for b in [50, 200, 1000]:
        n = pb.allocate_parcels(gamma, cap, b)
        # 交换条件：任一已分配组去掉 1 个的损失 >= 任一未满组加 1 个的增益
        rem = np.where(n > 0, gamma * (n ** -0.5 - (n + 1) ** -0.5), np.inf)
        add = np.where(n < cap, gamma * ((n + 1) ** -0.5 - (n + 2) ** -0.5), -np.inf)
        kkt = rem.min() >= add.max() - 1e-15
        c = cont(gamma, cap, b)
        print(f'{tgt} b={b:4d} sum={n.sum()} cap_ok={np.all(n<=cap)} exchange_opt={kkt} '
              f'max|n-cont|={np.abs(n-c).max():.2f} zero_groups={int((n==0).sum())}/{len(n)} '
              f'min_cap={cap.min()} n_at_mincap={n[cap.argmin()]}')
# 3) 容量为上限的情形：预算 > 总容量、个别组容量很小
cap = np.array([1, 2, 1000]); gamma = cap / cap.sum()
print('small caps b=500:', pb.allocate_parcels(gamma, cap, 500), ' b=2000 (> total cap):', pb.allocate_parcels(gamma, cap, 2000))
