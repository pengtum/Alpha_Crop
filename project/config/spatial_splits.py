"""
Spatial train/validation/test split strategies.

Supports two modes:
1. Belgian-only: Split by NUTS3 regions (original pipeline)
2. EU-wide: Split by country (pan-European OOD experiments)
"""

# ============================================================
# 比利时流水线（保留兼容）
# ============================================================
TEST_REGIONS = ['BE253', 'BE232', 'BE258']
VAL_REGIONS = ['BE225', 'BE236']

ALL_FLEMISH_REGIONS = [
    'BE211', 'BE212', 'BE213',
    'BE223', 'BE224', 'BE225',
    'BE231', 'BE232', 'BE233', 'BE234', 'BE235', 'BE236',
    'BE241', 'BE242',
    'BE251', 'BE252', 'BE253', 'BE254', 'BE255', 'BE256', 'BE257', 'BE258',
]

TRAIN_REGIONS = [r for r in ALL_FLEMISH_REGIONS
                 if r not in TEST_REGIONS and r not in VAL_REGIONS]


def assign_split(nuts3_code):
    """Assign a parcel to train/val/test based on its NUTS3 region."""
    if nuts3_code in TEST_REGIONS:
        return 'test'
    elif nuts3_code in VAL_REGIONS:
        return 'val'
    elif nuts3_code in TRAIN_REGIONS:
        return 'train'
    else:
        return None


# ============================================================
# 泛欧洲流水线：基于国家的 OOD 划分
# ============================================================
# 测试国家：南欧（ES）、北欧（SE）、西欧（PT）— 地理多样性
EU_TEST_COUNTRIES = ['ES', 'SE', 'PT']

# 验证国家：中欧
EU_VAL_COUNTRIES = ['AT', 'HR']

# 训练国家：其余所有
_EU_HELD_OUT = set(EU_TEST_COUNTRIES + EU_VAL_COUNTRIES)


def assign_split_eu(country_iso):
    """基于 ISO 国家码分配 split（OOD 策略）。

    Args:
        country_iso: 2字母 ISO 国家码（如 'DE', 'FR'）

    Returns:
        'train', 'val', 'test'
    """
    if country_iso in EU_TEST_COUNTRIES:
        return 'test'
    elif country_iso in EU_VAL_COUNTRIES:
        return 'val'
    else:
        return 'train'


def get_split_summary():
    """返回 EU 划分概览。"""
    from config.country_registry import COUNTRY_REGISTRY
    summary = {'train': [], 'val': [], 'test': []}
    for code, cfg in COUNTRY_REGISTRY.items():
        split = assign_split_eu(cfg.country_iso)
        summary[split].append(code)
    return summary
