"""
标签扩展实验（label expansion）公共定义。

数据与结果放在本地盘（Google Drive 上读大文件太慢），代码放在项目内。
路径可用环境变量 ALPHA_LOCAL 覆盖。
"""
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from config.class_definitions import TIER_B_MAPPING  # noqa: E402

LOCAL_ROOT = os.environ.get('ALPHA_LOCAL', os.path.expanduser('~/AlphaCrops_local'))
DATA_DIR = os.path.join(LOCAL_ROOT, 'data')
RESULTS_DIR = os.path.join(LOCAL_ROOT, 'results')
DATASET_V2 = os.path.join(DATA_DIR, 'dataset_v2.parquet')

RANDOM_STATE = 42

# 11 国 2021 年严格对齐子集；区域 → 国家
COUNTRIES = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']
REGIONS = ['AT', 'BE_VLG', 'BE_WAL', 'DE_LS', 'DE_NRW', 'EE', 'LT', 'LV',
           'PT', 'RO', 'SE', 'SI', 'SK']

# ------------------------------------------------------------
# Tier-B 映射 v2：原映射表对未收录的 HCAT3 类名一律兜底为 other_crops，
# 11 国中有 63,144 个地块（9.7%）因此被错归。以下补全遵循原表已有的归类惯例
# （例如原表 olive_olives→orchards，则 olive_plantations→orchards）。
# ------------------------------------------------------------
TIER_B_ADDITIONS = {
    # winter_cereals（原表：spelt / winter_durum_wheat / durum_wheat → winter_cereals）
    'winter_unspecified_cereals': 'winter_cereals',
    'winter_spelt': 'winter_cereals',
    'winter_durum_hard_wheat': 'winter_cereals',
    'durum_hard_wheat': 'winter_cereals',
    'winter_meslin': 'winter_cereals',
    'winter_emmer': 'winter_cereals',
    # orchards（原表：olive_olives / chestnuts / almonds / apricot / berries / other_permanent_crops）
    'olive_plantations': 'orchards',
    'sweet_chestnuts': 'orchards',
    'almond': 'orchards',
    'apricots': 'orchards',
    'quinces': 'orchards',
    'nectarine': 'orchards',
    'avocado': 'orchards',
    'pecan': 'orchards',
    'medlar_loquat': 'orchards',
    'pawpaw': 'orchards',
    'currants': 'orchards',
    'cranberry': 'orchards',
    'aronia_chokeberries': 'orchards',
    'hippophae_sea_buckthorns_seaberry': 'orchards',
    'chaenomeles_cathayensis': 'orchards',
    'elder_elderberry': 'orchards',
    'honeysuckle': 'orchards',
    'rowan_rowanberries': 'orchards',
    'rose_hip_rosehip': 'orchards',
    'unspecified_orchards_fruits': 'orchards',
    'other_permanent_crops_plantations': 'orchards',
    # grassland（原表：other_grasses / ryegrass / fescue / sainfoin / forage_legumes）
    'poaceae_grasses': 'grassland',
    'lolium_ryegrass': 'grassland',
    'festuca_fescue': 'grassland',
    'melilot': 'grassland',
    'galega': 'grassland',
    'onobrychis_sainfoins': 'grassland',
    'esparsette_onobrychis': 'grassland',
    'serradella': 'grassland',
    'unspecified_legumes_harvested_green': 'grassland',
    # tree_wood_forest（原表：poplar / willow / nurseries）
    'oak': 'tree_wood_forest',
    'eucalyptus': 'tree_wood_forest',
    'populus': 'tree_wood_forest',
    'aspen': 'tree_wood_forest',
    'willows_osiers': 'tree_wood_forest',
    'nurseries_nursery': 'tree_wood_forest',
    'ginko': 'tree_wood_forest',
    # root_crops（原表：fodder_beet / turnip_rape_swede）
    'fodder_roots': 'root_crops',
    'swede_rutabaga': 'root_crops',
    'topinambur_jerusalem_artichoke': 'root_crops',
    # vegetables
    'alliums': 'vegetables',
    'cucurbits': 'vegetables',
    'scallion': 'vegetables',
    'capsicum': 'vegetables',
    'horseradish': 'vegetables',
    'other_brassica_oleracea_cabbage': 'vegetables',
    'cress': 'vegetables',
    'sorrel': 'vegetables',
}
# 其余未收录类名（春/夏季谷物、香料药草、花卉、kitchen_gardens、greenhouse 等）
# 按原表惯例仍归 other_crops。
TIER_B_MAPPING_V2 = {**TIER_B_MAPPING, **TIER_B_ADDITIONS}

# 休耕地不是作物类型，v2 中排除（待作者确认）
EXCLUDE_V2 = {'fallow_land_not_crop', 'not_known_and_other', 'unmaintained'}


def tier_b_v2(hcat_name):
    return TIER_B_MAPPING_V2.get(hcat_name, 'other_crops')


def load_dataset(require=('ae', 's2')):
    """读取 v2 数据集；默认只保留 AE 与 S2 都有的地块（三种特征集用同一批地块）。"""
    df = pd.read_parquet(DATASET_V2)
    for r in require:
        df = df[df[f'has_{r}']]
    return df.reset_index(drop=True)


def feature_cols(df, feature_set):
    ae = [c for c in df.columns if c.startswith('ae_')]
    s2 = [c for c in df.columns if c.startswith('s2_')]
    return {'ae': ae, 's2': s2, 'fusion': ae + s2}[feature_set]


LGBM_PARAMS = dict(
    n_estimators=500, learning_rate=0.05, num_leaves=63, max_depth=-1,
    min_child_samples=20, class_weight='balanced',
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    # 多个实验并行时用环境变量 LGBM_THREADS 限制线程数（-1 = 全部核心）
    n_jobs=int(os.environ.get('LGBM_THREADS', -1)), random_state=RANDOM_STATE, verbose=-1,
)
EARLY_STOPPING = 50


def balanced_accuracy_present(y_true, y_pred):
    """只在测试集中出现的类别上计算平衡精度（与原实验一致）。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    recalls = [(y_pred[y_true == c] == c).mean() for c in np.unique(y_true)]
    return float(np.mean(recalls))
