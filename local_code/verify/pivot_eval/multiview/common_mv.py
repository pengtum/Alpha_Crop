# 多视图（AE × S2）诊断的公共读取函数。只读 results/guidance_G/stage0/*.npz 与 data/cache_guidance_G/y.npy。
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
import numpy as np
R = '/Users/perng_mini/AlphaCrops_local'
PROJ = '/Users/perng_mini/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/我的云端硬盘/Research/Alpha_Crops/label_expansion'
sys.path.insert(0, PROJ)
CLASSES = ["grassland", "maize", "orchards", "other_crops", "potatoes", "root_crops", "strawberries",
           "tree_wood_forest", "vegetables", "winter_cereals"]
COUNTRIES = 'AT,BE,DE,EE,LT,LV,PT,RO,SE,SI,SK'.split(',')
Y = np.load(f'{R}/data/cache_guidance_G/y.npy')
N_POOL = 30_000

def pool_positions(n_cand, split, rep):
    rng = np.random.default_rng(10_000 * split + 100 * rep + 7)
    return np.sort(rng.choice(n_cand, min(N_POOL, n_cand), replace=False))

def unit(tgt, split, rep):
    """返回候选池（rep）上的 y、两视图概率，测试集 y 与 big 类。"""
    z = np.load(f'{R}/results/guidance_G/stage0/{tgt}_s{split}.npz')
    nc = len(z['cand'])
    pp = pool_positions(nc, split, rep)
    allT = z['allT']
    yP = Y[allT[pp]]
    yT = Y[z['test']]
    big = np.flatnonzero(np.bincount(yT, minlength=10) >= 30)
    return dict(pp=pp, idx_pool=allT[pp], y=yP, Pa=z['P_T_ae'][pp].astype(np.float64),
                Ps=z['P_T_s2'][pp].astype(np.float64), yT=yT, big=big,
                P0a=z['P0test_ae_i300'].astype(np.float64), P0s=z['P0test_s2_i300'].astype(np.float64))
