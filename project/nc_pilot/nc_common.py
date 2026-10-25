"""
NC 试点公共定义（方案 nc_pilot/PILOT_PLAN_v4.md，冻结后以 plan_hashes.txt 登记的哈希为准）。

只放常量与纯函数：种子、路径、目标域、类别映射、LightGBM 参数。
"""
import hashlib
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.expanduser('~/AlphaCrops_local/nc_pilot')          # 数据与结果放本地盘（不经 Drive 同步）
SPLITS = os.environ.get('NC_SPLITS', os.path.join(DATA, 'splits'))   # 环境变量只用于试跑
FEATS_DIR = os.path.join(DATA, 'features')
PLAN_HASHES = os.path.join(HERE, 'plan_hashes.txt')

NPZ = os.path.join(RESEARCH, '0-Xiayin_research/15_RobustGeoCP/real_world_study/results_8ds/so2sat_scores.npz')
LUCAS_ASSET = 'JRC/LUCAS_HARMO/COPERNICUS_POLYGONS/V1/2018'
AE_COLL = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
GEE_PROJECT = 'myproject-alpha-earth'
ASSET_ROOT = f'projects/{GEE_PROJECT}/assets/nc_pilot'

TASKS = ('T1', 'T2')
FEATS = ('ae', 's2')
YEAR = {'T1': 2017, 'T2': 2018}
KS = (50, 100, 200, 400)
R = 3
TARGET_W = 50.0
THREADS = 2

# T1：可用城市与分组（方案第 3 节）
T1_MIN_N, T1_MIN_C = 500, 5
T1_GROUPS = (
    ('欧洲', ('amsterdam', 'berlin', 'cologne', 'istanbul', 'lisbon', 'london', 'madrid', 'milan', 'paris', 'rome', 'zurich')),
    ('亚洲', ('beijing', 'changsha', 'dongying', 'hongkong', 'islamabad', 'kyoto', 'nanjing', 'qingdao', 'shanghai',
             'shenzhen', 'tokyo', 'wuhan')),
    ('其他', ('cairo', 'capetown', 'losangeles', 'melbourne', 'newyork', 'riodejaneiro', 'saopaulo', 'vancouver',
             'washingtondc')),
)
T1_C = 17
T1_GRID, T1_GRID_OFF, T1_BUF, T1_HALF = 2000.0, 0.0, 500.0, 160.0
T1_CLUSTER = 330.0

# T2：LUCAS Copernicus 多边形自身的一级土地覆盖（cprn_lc 首字母）；G 水体全欧仅 10 个，剔除；A..F、H → 0..6
T2_CLASSES = 'ABCDEFH'
T2_DROP = 'G'
T2_C = 7
T2_MIN_N = 2000
T2_GROUPS = (('南欧', ('EL', 'ES', 'IT')), ('中西欧', ('DE', 'FR', 'PL', 'RO', 'UK')), ('北欧', ('FI', 'SE')))
T2_GRID, T2_GRID_OFF, T2_BUF = 10000.0, 5000.0, 1000.0   # LUCAS 点在 3035 的 2 km 格网上；网格平移半格，格线离点位 ≥ 1 km
T2_EPSG = 3035

TEST_FRAC = 0.3
BA_MIN = 10
BA_MIN_CLUSTERS = 3   # 只用于 T1


def seed_of(s):
    """与 RSE confirm_split.seed_of 相同：int(md5(utf-8).hexdigest(), 16) mod 2^31。"""
    return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16) % 2 ** 31


def rng_of(s):
    return np.random.default_rng(seed_of(s))


def utm_epsg(lon_mean, lat_mean):
    zone = int(np.floor((lon_mean + 180.0) / 6.0)) + 1
    return (32600 if lat_mean >= 0 else 32700) + zone


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def lgb_params(seed, num_class):
    """与 RSE stageA.lgb_params 相同（只把 num_class 改为本任务类别数）。"""
    return dict(objective='multiclass', num_class=num_class, learning_rate=0.05, num_leaves=63, min_data_in_leaf=20,
                bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, max_bin=63, force_col_wise=True,
                num_threads=THREADS, seed=seed, deterministic=True, verbose=-1)


def class_weights_from(y_ref, C):
    """与 RSE stageA.class_weights_from 相同：只按源池类别频率；源池没有的类权重取 1。"""
    n = np.bincount(y_ref, minlength=C).astype(float)
    return np.where(n > 0, len(y_ref) / (np.count_nonzero(n) * np.maximum(n, 1)), 1.0)


def register(line):
    """在 plan_hashes.txt 末尾追加一行（只追加，不改已有行）。"""
    import time
    with open(PLAN_HASHES, 'a', encoding='utf-8') as fh:
        fh.write(time.strftime('%Y-%m-%d %H:%M ') + line.rstrip() + '\n')
