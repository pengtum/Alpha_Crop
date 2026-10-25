"""冒烟测试：真实数据，只跑 SI 划分 0 重复 0，每个模型 5 棵树，层级 1 与 2 的全部条件和方法各一次。
只检查能否运行、耗时，以及划分是否与阶段 A 一致；不看、不输出任何精度指标。"""
import os, sys, time
os.environ['GUIDE_OUT'] = os.path.expanduser('~/AlphaCrops_local/verify/guidance_smoke/out')
os.environ['GUIDE_TREES'] = '5'
sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd
import guidance_G as G
t0 = time.time()
G.GData.build_cache(); print(f"缓存 {time.time() - t0:.0f}s", flush=True)
D = G.GData()
from stageA import Data
D0 = Data()
assert np.array_equal(D0.y, D.y) and np.array_equal(D0.ctry, D.ctry) and np.array_equal(D0.lon, D.lon) and np.array_equal(D0.area, D.area)
assert all(np.array_equal(D0.X[f], D.X[f]) for f in ("ae", "s2")) and D0.countries == D.countries and list(D0.classes) == D.classes
print("缓存与 stageA.Data 逐位一致", flush=True); del D0
print(f'数据 {time.time() - t0:.0f}s', flush=True)
# 1. 划分与阶段 A 逐位一致（PT、SI 划分 0/1）
for tgt, s in (('PT', 0), ('SI', 1)):
    A = np.load(os.path.expanduser(f'~/AlphaCrops_local/results/stageA/stage0/{tgt}_s{s}.npz'))
    sp = G.make_split(D, tgt, s)
    for key in ('foreign', 'cand', 'test', 'allT'):
        assert np.array_equal(A[key], sp[key]), (tgt, s, key)
    assert np.isclose(float(A['lat0']), sp['lat0'])
    print(f'划分一致：{tgt} s{s}（cand {len(sp["cand"]):,}，test {len(sp["test"]):,}，E_far {sp["efar"].sum():,}）', flush=True)
# 2. 阶段 0（5 棵树）
os.makedirs(os.path.join(G.OUT, 'stage0'), exist_ok=True)
t = time.time(); G.stage0_one(D, 'SI', 0, threads=8); print(f'阶段 0：{time.time() - t:.0f}s', flush=True)
S = dict(np.load(os.path.join(G.OUT, 'stage0', 'SI_s0.npz')))
print('阶段 0 键：', sorted(S), flush=True)
print('几何形状：', {k: S[k].shape for k in ('G_ae', 'G_s2', 'RZ_ae', 'NT_ae', 'NT_s2', 'GF_ae')}, flush=True)
# 3. 层级 1、2、3 各跑一遍（只记录耗时与行数）
for tier in (1, 2, 3):
    t = time.time()
    R = pd.DataFrame(G.run_unit(D, S, 'SI', 0, 0, tier, threads=8, trees=5, log=lambda s: None))
    tsel = R.groupby(['cond', 'method']).t_select.max()
    print(f'层级 {tier}：{len(R)} 行，{time.time() - t:.0f}s；缓存命中 {int(R.cache_hit.sum())}；'
          f'选样最慢 {tsel.sort_values().tail(3).round(1).to_dict()}', flush=True)
    assert R.n_far.nunique() == 1
print(f'总计 {time.time() - t0:.0f}s')
