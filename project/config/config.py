"""
Central configuration for the AlphaEarth crop prediction pipeline.
"""
import os

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- 原始比利时流水线路径（保留兼容）---
SHAPEFILE_PATH = os.path.join(BASE_DIR, 'BE_VLG_2021', 'BE_VLG_2021_EC21.shp')
CSV_LOOKUP_PATH = os.path.join(BASE_DIR, 'be_vlg_2021.csv')

# --- 原始比利时处理数据 ---
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
PARCELS_FILTERED_PATH = os.path.join(PROCESSED_DIR, 'parcels_filtered.parquet')
EMBEDDINGS_CENTROID_PATH = os.path.join(PROCESSED_DIR, 'embeddings_centroid.parquet')
EMBEDDINGS_POLYGON_PATH = os.path.join(PROCESSED_DIR, 'embeddings_polygon.parquet')
FEATURES_TRAIN_PATH = os.path.join(PROCESSED_DIR, 'features_train.parquet')
FEATURES_VAL_PATH = os.path.join(PROCESSED_DIR, 'features_val.parquet')
FEATURES_TEST_PATH = os.path.join(PROCESSED_DIR, 'features_test.parquet')

# --- 泛欧洲流水线路径 ---
DATA_CROP_DIR = os.path.join(BASE_DIR, 'Data_Crop')
EU_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'eu_processed')
EU_PARCELS_PATH = os.path.join(EU_PROCESSED_DIR, 'parcels_eu.parquet')
EU_EMBEDDINGS_DIR = os.path.join(EU_PROCESSED_DIR, 'embeddings')
EU_FEATURES_DIR = os.path.join(EU_PROCESSED_DIR, 'features')
EU_FEATURES_TRAIN_PATH = os.path.join(EU_FEATURES_DIR, 'features_train.parquet')
EU_FEATURES_VAL_PATH = os.path.join(EU_FEATURES_DIR, 'features_val.parquet')
EU_FEATURES_TEST_PATH = os.path.join(EU_FEATURES_DIR, 'features_test.parquet')

# Model outputs
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')

# ============================================================
# GOOGLE EARTH ENGINE
# ============================================================
GEE_PROJECT = 'myproject-alpha-earth'
AE_COLLECTION = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
AE_SCALE = 10  # meters per pixel
AE_BANDS = [f'A{i:02d}' for i in range(64)]
AE_YEAR = 2021  # 比利时流水线默认年份
AE_NEIGHBORHOOD_RADIUS = 3  # pixels (30m at 10m scale)

# GEE extraction settings
GEE_BATCH_SIZE = 5000  # points per sampleRegions call
GEE_EXPORT_FOLDER = 'alpha_crops_embeddings'  # Google Drive folder
GEE_EU_EXPORT_PREFIX = 'alpha_crops_eu'  # 泛欧导出前缀

# ============================================================
# SENTINEL-2 BASELINE
# ============================================================
S2_COLLECTION = 'COPERNICUS/S2_SR_HARMONIZED'
S2_CLOUD_THRESHOLD = 20  # 场景云覆盖百分比
S2_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
S2_SCALE = 10  # meters per pixel
S2_NEIGHBORHOOD_RADIUS = 3  # pixels (与 AlphaEarth 一致)
GEE_S2_EU_EXPORT_PREFIX = 's2_eu'  # S2 泛欧导出前缀

# S2 嵌入输出路径
EU_S2_EMBEDDINGS_DIR = os.path.join(EU_PROCESSED_DIR, 's2_embeddings')

# ============================================================
# EXPERIMENT RESULTS
# ============================================================
EXPERIMENTS_DIR = os.path.join(RESULTS_DIR, 'experiments')

# ============================================================
# DATA FILTERING
# ============================================================
SOURCE_CRS = 'EPSG:31370'  # Belgian Lambert 72（比利时流水线用）
TARGET_CRS = 'EPSG:4326'   # WGS84
EQUAL_AREA_CRS = 'EPSG:3035'  # ETRS89-LAEA（欧洲等面积投影，用于计算面积）

# Classes to exclude (non-agricultural)
EXCLUDE_CLASSES = ['not_known_and_other', 'unmaintained']

# Only keep Flemish NUTS3 regions (BE2xx) — 比利时流水线用
FLEMISH_NUTS3_PREFIX = 'BE2'

# Minimum parcel area (ha)
MIN_AREA_HA = 0.005

# ============================================================
# 统一列名常量（泛欧流水线）
# ============================================================
COL_PARCEL_ID = 'parcel_id'
COL_COUNTRY = 'country'
COL_AREA = 'area_ha'
COL_LON = 'longitude'
COL_LAT = 'latitude'
COL_HCAT3_NAME = 'EC_hcat_n'
COL_HCAT3_CODE = 'EC_hcat_c'
COL_TRANS_NAME = 'EC_trans_n'

# ============================================================
# MODEL TRAINING
# ============================================================
RANDOM_STATE = 42

# Subsampling for quick experiments
QUICK_SAMPLES_PER_CLASS = 500
FULL_SAMPLES_PER_CLASS = 5000  # 0 = use all

# LightGBM hyperparameters
LGBM_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': -1,
    'min_child_samples': 20,
    'class_weight': 'balanced',
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_jobs': -1,
    'random_state': RANDOM_STATE,
    'verbose': -1,
}
LGBM_EARLY_STOPPING = 50

# XGBoost 超参数
XGB_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'n_jobs': -1,
    'random_state': RANDOM_STATE,
    'verbosity': 0,
}
XGB_EARLY_STOPPING = 50

# TabPFN 参数
TABPFN_MAX_TRAIN_SAMPLES = 10000   # CPU 模式下的最大训练样本数
TABPFN_PARAMS = {
    'random_state': RANDOM_STATE,
    'ignore_pretraining_limits': True,
}

# Random Forest hyperparameters
RF_PARAMS = {
    'n_estimators': 300,
    'max_depth': None,
    'min_samples_leaf': 5,
    'class_weight': 'balanced',
    'n_jobs': -1,
    'random_state': RANDOM_STATE,
}
