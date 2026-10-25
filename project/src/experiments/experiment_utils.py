"""
实验通用工具：数据加载、模型训练、结果保存。

所有实验运行器共享的核心逻辑。
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    balanced_accuracy_score, accuracy_score, cohen_kappa_score,
    f1_score, classification_report
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from config.config import (
    EU_FEATURES_DIR, EU_PARCELS_PATH, MODELS_DIR,
    EXPERIMENTS_DIR, LGBM_PARAMS, LGBM_EARLY_STOPPING,
    XGB_PARAMS, XGB_EARLY_STOPPING,
    TABPFN_MAX_TRAIN_SAMPLES, TABPFN_PARAMS,
    RANDOM_STATE, COL_PARCEL_ID, COL_COUNTRY, COL_AREA,
)


# ============================================================
# 气候带定义
# ============================================================
CLIMATE_ZONES = {
    'mediterranean': ['ES', 'PT', 'HR', 'SI'],
    'atlantic': ['FR', 'IE', 'NL', 'BE'],
    'continental': ['DE', 'AT', 'CZ', 'SK'],
    'nordic_baltic': ['DK', 'SE', 'FI', 'EE', 'LV', 'LT'],
    'southeast': ['RO'],
}

# 反向映射：国家 -> 气候带
COUNTRY_TO_ZONE = {}
for zone, countries in CLIMATE_ZONES.items():
    for c in countries:
        COUNTRY_TO_ZONE[c] = zone


# ============================================================
# 2021 年时间对齐国家列表（主实验使用）
# ============================================================
COUNTRIES_2021 = ['AT', 'BE', 'DE', 'EE', 'LT', 'LV', 'PT', 'RO', 'SE', 'SI', 'SK']


# ============================================================
# 数据加载
# ============================================================

def load_eu_features(feature_type='ae', tier='b', countries=None):
    """加载指定 feature_type 的完整 EU 数据集（三个 split 合并）。

    Args:
        feature_type: 'ae', 's2', 'fusion'
        tier: 'a', 'b', 'c'
        countries: 限制国家列表（None=全部）。传入 COUNTRIES_2021 可获取时间对齐子集。

    Returns:
        df: 完整 DataFrame (包含特征 + 元数据)
        feature_cols: 特征列名列表
        class_names: 类别名数组
    """
    # 确定路径（所有 feature_type 统一使用子目录）
    feat_dir = os.path.join(EU_FEATURES_DIR, feature_type)
    models_dir = os.path.join(MODELS_DIR, 'eu', feature_type)

    # 加载三个 split
    dfs = []
    for split in ['train', 'val', 'test']:
        path = os.path.join(feat_dir, f'features_{split}.parquet')
        if not os.path.exists(path):
            print(f"WARNING: {path} not found")
            continue
        dfs.append(pd.read_parquet(path))

    if not dfs:
        raise FileNotFoundError(f"No feature files found in {feat_dir}")

    df = pd.concat(dfs, ignore_index=True)

    # 加载特征列和标签编码器
    feature_cols = joblib.load(os.path.join(models_dir, 'feature_cols.joblib'))
    le_dict = joblib.load(os.path.join(models_dir, 'label_encoders.joblib'))

    label_col = f'label_tier_{tier}'
    class_names = le_dict[label_col].classes_

    # 确保编码列存在
    encoded_col = f'{label_col}_encoded'
    if encoded_col not in df.columns:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.classes_ = class_names
        df[encoded_col] = le.transform(df[label_col])

    # 按国家过滤
    if countries is not None:
        n_before = len(df)
        df = df[df[COL_COUNTRY].isin(countries)].reset_index(drop=True)
        # 剔除非 2021 年的 DE_BB 子区域（2023 年数据）
        if 'DE' in countries and COL_PARCEL_ID in df.columns:
            bb_mask = df[COL_PARCEL_ID].str.startswith('DE_BB_')
            n_bb = bb_mask.sum()
            if n_bb > 0:
                df = df[~bb_mask].reset_index(drop=True)
                print(f"  [year filter] Removed {n_bb} DE_BB (2023) parcels")
        print(f"  [country filter] {n_before} -> {len(df)} samples "
              f"({len(countries)} countries: {sorted(countries)})")

    print(f"Loaded {feature_type.upper()} features: {len(df)} samples, "
          f"{len(feature_cols)} features, {len(class_names)} classes (tier {tier})")

    return df, feature_cols, class_names


AUX_COORD_COLS = {'longitude', 'latitude', 'log_area'}


def filter_coords(feature_cols):
    """移除地理坐标和面积辅助特征，仅保留嵌入/光谱特征。"""
    filtered = [c for c in feature_cols if c not in AUX_COORD_COLS]
    removed = [c for c in feature_cols if c in AUX_COORD_COLS]
    if removed:
        print(f"  [no-coords] Removed {removed}, {len(filtered)} features remain")
    return filtered


def prepare_xy(df, feature_cols, tier='b'):
    """从 DataFrame 提取 X, y 数组。"""
    encoded_col = f'label_tier_{tier}_encoded'
    X = df[feature_cols].fillna(0).values
    y = df[encoded_col].values
    return X, y


def split_by_countries(df, feature_cols, tier,
                       train_countries, test_countries,
                       val_countries=None):
    """按国家列表划分数据集。

    Args:
        df: 完整数据集
        feature_cols: 特征列
        tier: 分类层级
        train_countries: 训练集国家 ISO 列表
        test_countries: 测试集国家 ISO 列表
        val_countries: 验证集国家 ISO 列表（None 则从 train 中随机分 10%）

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    train_mask = df[COL_COUNTRY].isin(train_countries)
    test_mask = df[COL_COUNTRY].isin(test_countries)

    if val_countries:
        val_mask = df[COL_COUNTRY].isin(val_countries)
    else:
        # 从训练集中随机分 10% 作验证
        train_indices = df[train_mask].index.values
        rng = np.random.RandomState(RANDOM_STATE)
        n_val = max(1, int(len(train_indices) * 0.1))
        val_idx = rng.choice(train_indices, size=n_val, replace=False)
        val_mask = df.index.isin(val_idx)
        train_mask = train_mask & ~val_mask

    X_train, y_train = prepare_xy(df[train_mask], feature_cols, tier)
    X_val, y_val = prepare_xy(df[val_mask], feature_cols, tier)
    X_test, y_test = prepare_xy(df[test_mask], feature_cols, tier)

    return X_train, y_train, X_val, y_val, X_test, y_test


# ============================================================
# 模型训练
# ============================================================

# ============================================================
# 分类器常量
# ============================================================
CLASSIFIER_LGBM = 'lgbm'
CLASSIFIER_XGB = 'xgb'
CLASSIFIER_TABPFN = 'tabpfn'
VALID_CLASSIFIERS = [CLASSIFIER_LGBM, CLASSIFIER_XGB, CLASSIFIER_TABPFN]


# ============================================================
# 训练集子采样（保证跨分类器一致性）
# ============================================================
def subsample_train(X, y, max_samples, seed=RANDOM_STATE):
    """对训练集进行分层子采样，保证不同分类器使用完全相同的样本。

    Args:
        X: 训练特征
        y: 训练标签
        max_samples: 最大样本数（0 或 None 表示不采样）
        seed: 随机种子

    Returns:
        X_sub, y_sub
    """
    if not max_samples or len(X) <= max_samples:
        return X, y

    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, train_size=max_samples,
                                 random_state=seed)
    idx, _ = next(sss.split(X, y))
    print(f"  [subsample] {len(X)} -> {len(idx)} samples (seed={seed})")
    return X[idx], y[idx]


# ============================================================
# 模型训练
# ============================================================

def train_lgbm(X_train, y_train, X_val, y_val, n_classes, verbose=False):
    """训练 LightGBM 分类器（使用统一超参数）。"""
    import lightgbm as lgb

    params = LGBM_PARAMS.copy()
    if n_classes > 2:
        params['objective'] = 'multiclass'
        params['num_class'] = n_classes
        params['metric'] = 'multi_logloss'

    if not verbose:
        params['verbose'] = -1

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(LGBM_EARLY_STOPPING),
            lgb.log_evaluation(0 if not verbose else 50)
        ]
    )

    return model


class _XGBWrapper:
    """XGBoost wrapper：处理非连续标签（如 [0,1,3,5] -> [0,1,2,3]）。

    XGBoost 要求 y 为连续 0~N-1，但单国训练时某些类别可能缺失。
    此 wrapper 在 fit 时重编码，predict 时映射回原始标签。
    """

    def __init__(self, model, label_map, inv_map):
        self.model = model
        self.label_map = label_map    # original -> compact
        self.inv_map = inv_map        # compact -> original

    def predict(self, X):
        compact_pred = self.model.predict(X)
        return np.array([self.inv_map.get(int(p), int(p)) for p in compact_pred])

    @property
    def feature_importances_(self):
        return self.model.feature_importances_


def train_xgboost(X_train, y_train, X_val, y_val, n_classes, verbose=False):
    """训练 XGBoost 分类器（使用统一超参数）。

    自动处理非连续标签（单国训练时某些全局类别可能缺失）。
    """
    from xgboost import XGBClassifier
    from sklearn.utils.class_weight import compute_sample_weight

    # 检查标签是否连续 0~N-1，不连续则重编码
    unique_labels = sorted(set(y_train) | set(y_val))
    needs_remap = (unique_labels != list(range(len(unique_labels))))

    if needs_remap:
        label_map = {orig: compact for compact, orig in enumerate(unique_labels)}
        inv_map = {compact: orig for orig, compact in label_map.items()}
        y_train_enc = np.array([label_map[y] for y in y_train])
        y_val_enc = np.array([label_map[y] for y in y_val])
        actual_n_classes = len(unique_labels)
    else:
        y_train_enc, y_val_enc = y_train, y_val
        actual_n_classes = n_classes
        label_map, inv_map = None, None

    params = XGB_PARAMS.copy()
    if actual_n_classes > 2:
        params['objective'] = 'multi:softmax'
        params['num_class'] = actual_n_classes
        params['eval_metric'] = 'mlogloss'

    params['early_stopping_rounds'] = XGB_EARLY_STOPPING

    model = XGBClassifier(**params)
    sw = compute_sample_weight('balanced', y_train_enc)

    model.fit(
        X_train, y_train_enc,
        sample_weight=sw,
        eval_set=[(X_val, y_val_enc)],
        verbose=verbose,
    )

    if needs_remap:
        return _XGBWrapper(model, label_map, inv_map)
    return model


def train_tabpfn(X_train, y_train, X_val, y_val, n_classes, verbose=False):
    """训练 TabPFN 分类器。

    注意：TabPFN 不使用验证集（非迭代训练），X_val/y_val 被忽略。
    训练数据应在调用前通过 subsample_train() 子采样到合理大小。
    """
    import os
    os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET'] = '1'
    from tabpfn import TabPFNClassifier

    # 安全检查：若训练集仍超出限制，强制子采样
    if len(X_train) > TABPFN_MAX_TRAIN_SAMPLES:
        print(f"  [TabPFN] WARNING: {len(X_train)} samples > {TABPFN_MAX_TRAIN_SAMPLES}, "
              f"auto-subsampling (应在 runner 层统一子采样)")
        X_train, y_train = subsample_train(X_train, y_train, TABPFN_MAX_TRAIN_SAMPLES)

    params = TABPFN_PARAMS.copy()
    model = TabPFNClassifier(**params)
    model.fit(X_train, y_train)
    return model


def get_trainer(classifier_type):
    """返回对应分类器的训练函数。

    所有训练函数签名: (X_train, y_train, X_val, y_val, n_classes, verbose) -> model
    """
    trainers = {
        CLASSIFIER_LGBM: train_lgbm,
        CLASSIFIER_XGB: train_xgboost,
        CLASSIFIER_TABPFN: train_tabpfn,
    }
    if classifier_type not in trainers:
        raise ValueError(f"Unknown classifier: {classifier_type}. "
                         f"Choose from {VALID_CLASSIFIERS}")
    return trainers[classifier_type]


def build_suffix(no_coords=False, max_samples=0, classifier_type='lgbm',
                  year_aligned=False):
    """构建统一的文件后缀。

    Examples:
        lgbm, no_coords=True                    -> '_nocoords'
        lgbm, no_coords=True, year_aligned=True -> '_nocoords_2021'
        xgb, no_coords=True, year_aligned=True  -> '_nocoords_2021_xgb'
    """
    parts = []
    if no_coords:
        parts.append('nocoords')
    if year_aligned:
        parts.append('2021')
    if max_samples:
        parts.append(f'sub{max_samples // 1000}k')
    if classifier_type != CLASSIFIER_LGBM:
        parts.append(classifier_type)
    return ('_' + '_'.join(parts)) if parts else ''


def evaluate_predictions(y_true, y_pred, class_names=None):
    """计算标准评估指标。"""
    results = {
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'overall_accuracy': float(accuracy_score(y_true, y_pred)),
        'cohen_kappa': float(cohen_kappa_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted')),
        'n_samples': int(len(y_true)),
    }

    if class_names is not None:
        # 确保 labels 参数覆盖所有可能的类别索引
        all_labels = list(range(len(class_names)))
        report = classification_report(
            y_true, y_pred, labels=all_labels,
            target_names=class_names,
            output_dict=True, zero_division=0)
        results['per_class'] = {
            k: {m: float(v) for m, v in metrics.items()}
            for k, metrics in report.items()
            if k not in ('accuracy', 'macro avg', 'weighted avg')
        }

    return results


# ============================================================
# 结果保存
# ============================================================

def save_experiment_results(exp_name, results, sub_name=None):
    """保存实验结果为 JSON。

    Args:
        exp_name: 实验名（如 'exp1_id_baseline'）
        results: 结果字典
        sub_name: 子实验名（可选，如 'ae_tier_b'）
    """
    out_dir = os.path.join(EXPERIMENTS_DIR, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    if sub_name:
        filename = f'{sub_name}.json'
    else:
        filename = 'results.json'

    out_path = os.path.join(out_dir, filename)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"Results saved to {out_path}")
    return out_path


def save_results_csv(exp_name, rows, filename='summary.csv'):
    """保存表格结果为 CSV。"""
    out_dir = os.path.join(EXPERIMENTS_DIR, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"CSV saved to {out_path}")
    return out_path


def get_all_country_isos(df):
    """从 DataFrame 获取所有唯一国家 ISO 码。"""
    return sorted(df[COL_COUNTRY].unique())


def print_metrics(metrics, label=''):
    """打印评估指标。"""
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}BA={metrics['balanced_accuracy']:.4f}, "
          f"F1={metrics['macro_f1']:.4f}, "
          f"Kappa={metrics['cohen_kappa']:.4f}, "
          f"n={metrics['n_samples']}")
