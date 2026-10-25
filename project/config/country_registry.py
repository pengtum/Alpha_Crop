"""
欧洲各国 EuroCrops 数据集注册表。

每个国家/地区定义其 ZIP 文件名、数据年份、ISO 国家码和对应的 CSV 查找表。
所有 EuroCrops shapefile 均包含统一的 EC_hcat_n / EC_hcat_c / EC_trans_n 列。
"""
from dataclasses import dataclass


@dataclass
class CountryConfig:
    code: str          # 唯一标识，如 'BE_VLG'
    zip_name: str      # Data_Crop 中的 ZIP 文件名
    year: int          # 数据年份（匹配 AlphaEarth 嵌入年份）
    country_iso: str   # ISO 3166-1 alpha-2 国家码
    csv_lookup: str    # 对应的 HCAT 查找 CSV（参考用）


# ============================================================
# 国家注册表
# ============================================================
COUNTRY_REGISTRY = {
    # --- 比利时 ---
    'BE_VLG': CountryConfig(
        code='BE_VLG', zip_name='BE_VLG_2021.zip', year=2021,
        country_iso='BE', csv_lookup='be_vlg_2021.csv'),
    'BE_WAL': CountryConfig(
        code='BE_WAL', zip_name='BE_WAL_2021.zip', year=2021,
        country_iso='BE', csv_lookup='be_wal_all_years.csv'),

    # --- 德国 ---
    'DE_BB': CountryConfig(
        code='DE_BB', zip_name='DE_BB_2023.zip', year=2023,
        country_iso='DE', csv_lookup='de_bb_2023.csv'),
    'DE_NRW': CountryConfig(
        code='DE_NRW', zip_name='DE_NRW_2021.zip', year=2021,
        country_iso='DE', csv_lookup='de_nrw_2021.csv'),
    'DE_LS': CountryConfig(
        code='DE_LS', zip_name='DE_LS_2021.zip', year=2021,
        country_iso='DE', csv_lookup='de_ls_2021.csv'),

    # --- 法国 ---
    'FR': CountryConfig(
        code='FR', zip_name='FR_2018.zip', year=2018,
        country_iso='FR', csv_lookup='fr_2018.csv'),

    # --- 荷兰 ---
    'NL': CountryConfig(
        code='NL', zip_name='NL_2020.zip', year=2020,
        country_iso='NL', csv_lookup='nl_2020.csv'),

    # --- 奥地利 ---
    'AT': CountryConfig(
        code='AT', zip_name='AT_2021.zip', year=2021,
        country_iso='AT', csv_lookup='at_2021.csv'),

    # --- 丹麦 ---
    'DK': CountryConfig(
        code='DK', zip_name='DK_2019.zip', year=2019,
        country_iso='DK', csv_lookup='dk_2019.csv'),

    # --- 瑞典 ---
    'SE': CountryConfig(
        code='SE', zip_name='SE_2021.zip', year=2021,
        country_iso='SE', csv_lookup='se_2021.csv'),

    # --- 西班牙 ---
    'ES': CountryConfig(
        code='ES', zip_name='ES_2023.zip', year=2023,
        country_iso='ES', csv_lookup='es_2023.csv'),
    'ES_NA': CountryConfig(
        code='ES_NA', zip_name='ES_NA_2020.zip', year=2020,
        country_iso='ES', csv_lookup='es_ne_2021.csv'),

    # --- 捷克 ---
    'CZ': CountryConfig(
        code='CZ', zip_name='CZ_2023.zip', year=2023,
        country_iso='CZ', csv_lookup='cz_2023.csv'),

    # --- 芬兰 ---
    'FI': CountryConfig(
        code='FI', zip_name='FI_2020.zip', year=2020,
        country_iso='FI', csv_lookup='fi_2020.csv'),

    # --- 爱尔兰 ---
    'IE': CountryConfig(
        code='IE', zip_name='IE_2023.zip', year=2023,
        country_iso='IE', csv_lookup='ie_2023.csv'),

    # --- 葡萄牙 ---
    'PT': CountryConfig(
        code='PT', zip_name='PT_2021.zip', year=2021,
        country_iso='PT', csv_lookup='pt_2021.csv'),

    # --- 斯洛文尼亚 ---
    'SI': CountryConfig(
        code='SI', zip_name='SI_2021.zip', year=2021,
        country_iso='SI', csv_lookup='si_2021.csv'),

    # --- 斯洛伐克 ---
    'SK': CountryConfig(
        code='SK', zip_name='SK_2021.zip', year=2021,
        country_iso='SK', csv_lookup='sk_2021.csv'),

    # --- 爱沙尼亚 ---
    'EE': CountryConfig(
        code='EE', zip_name='EE_2021.zip', year=2021,
        country_iso='EE', csv_lookup='ee_2021.csv'),

    # --- 拉脱维亚 ---
    'LV': CountryConfig(
        code='LV', zip_name='LV_2021.zip', year=2021,
        country_iso='LV', csv_lookup='lv_2021.csv'),

    # --- 立陶宛 ---
    'LT': CountryConfig(
        code='LT', zip_name='LT_2021.zip', year=2021,
        country_iso='LT', csv_lookup='lt_2021.csv'),

    # --- 克罗地亚 ---
    'HR': CountryConfig(
        code='HR', zip_name='HR_2020.zip', year=2020,
        country_iso='HR', csv_lookup='hr_2020.csv'),

    # --- 罗马尼亚（无明确年份，默认 2021）---
    'RO': CountryConfig(
        code='RO', zip_name='RO_ny.zip', year=2021,
        country_iso='RO', csv_lookup='ro_no_year.csv'),
}


def get_all_country_codes():
    """返回所有已注册的国家/地区代码。"""
    return list(COUNTRY_REGISTRY.keys())


def get_country(code):
    """根据代码获取国家配置。"""
    if code not in COUNTRY_REGISTRY:
        raise ValueError(f"未注册的国家代码: {code}. "
                         f"可用: {get_all_country_codes()}")
    return COUNTRY_REGISTRY[code]


def get_countries_by_iso(iso):
    """根据 ISO 国家码获取所有地区配置（如 'DE' 返回 DE_BB, DE_NRW, DE_LS）。"""
    return [c for c in COUNTRY_REGISTRY.values() if c.country_iso == iso]


def get_unique_isos():
    """返回所有不重复的 ISO 国家码。"""
    return sorted(set(c.country_iso for c in COUNTRY_REGISTRY.values()))
