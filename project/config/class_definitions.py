"""
Three-tier crop classification definitions.

Tier B: 10 coarse super-classes (rapid prototyping)
Tier A: ~21 classes (top 20 EU-wide HCAT3 + other_agriculture)
Tier C: All HCAT3 classes present in data (full granularity)
"""

# ============================================================
# TIER B: 10 Super-Classes
# ============================================================
# Maps HCAT3 class names -> Tier B super-class
TIER_B_MAPPING = {
    # 1. grassland
    'pasture_meadow_grassland_grass': 'grassland',
    'clover': 'grassland',
    'plants_harvested_green': 'grassland',
    'alfalfa_lucerne': 'grassland',
    'temporary_grass': 'grassland',
    'sod_turf': 'grassland',
    'festulolium': 'grassland',
    'grass_clover_mixture': 'grassland',
    'ryegrass': 'grassland',
    'timothy': 'grassland',
    'fescue': 'grassland',
    'sainfoin': 'grassland',
    'forage_legumes': 'grassland',
    'other_grasses': 'grassland',
    'grassland': 'grassland',
    'permanent_grassland': 'grassland',

    # 2. maize
    'green_silo_maize': 'maize',
    'grain_maize_corn_popcorn': 'maize',
    'maize': 'maize',

    # 3. winter_cereals
    'winter_common_soft_wheat': 'winter_cereals',
    'winter_barley': 'winter_cereals',
    'winter_rye': 'winter_cereals',
    'winter_oats': 'winter_cereals',
    'spelt': 'winter_cereals',
    'triticale': 'winter_cereals',
    'winter_durum_wheat': 'winter_cereals',
    'winter_triticale': 'winter_cereals',
    'common_soft_wheat': 'winter_cereals',
    'durum_wheat': 'winter_cereals',
    'wheat': 'winter_cereals',

    # 4. potatoes
    'potatoes': 'potatoes',
    'sweet_potatoes': 'potatoes',
    'seed_potatoes': 'potatoes',
    'starch_potatoes': 'potatoes',

    # 5. root_crops
    'sugar_beet': 'root_crops',
    'mangelwurzel_fodder_beet': 'root_crops',
    'carrots_daucus': 'root_crops',
    'chicory_chicories': 'root_crops',
    'fodder_beet': 'root_crops',
    'turnip_rape_swede': 'root_crops',

    # 6. orchards
    'pears': 'orchards',
    'apples': 'orchards',
    'cherry_cherries': 'orchards',
    'plums': 'orchards',
    'orchards_fruits': 'orchards',
    'vineyards_wine_vine_rebland_grapes': 'orchards',
    'walnuts': 'orchards',
    'hazelnuts_hazel': 'orchards',
    'blueberry': 'orchards',
    'raspberry_raspberries': 'orchards',
    'redcurrant': 'orchards',
    'blackberry': 'orchards',
    'kiwi': 'orchards',
    'peach': 'orchards',
    'berries_berry_species': 'orchards',
    'gooseberry_gooseberries_cranberries': 'orchards',
    'blackcurrant_cassis': 'orchards',
    'citrus_plantations': 'orchards',
    'olive_olives': 'orchards',
    'almonds': 'orchards',
    'apricot': 'orchards',
    'fig': 'orchards',
    'table_grapes': 'orchards',
    'chestnuts': 'orchards',
    'pistachios': 'orchards',
    'pomegranate': 'orchards',
    'nuts': 'orchards',
    'stone_fruits': 'orchards',
    'tropical_subtropical_fruits': 'orchards',
    'other_permanent_crops': 'orchards',

    # 7. vegetables
    'cauliflower': 'vegetables',
    'leek': 'vegetables',
    'beans': 'vegetables',
    'brussels_sprouts': 'vegetables',
    'onions': 'vegetables',
    'spinach': 'vegetables',
    'peas': 'vegetables',
    'asparagus': 'vegetables',
    'white_cabbage': 'vegetables',
    'broccoli': 'vegetables',
    'savoy_cabbage': 'vegetables',
    'celeriac': 'vegetables',
    'celery': 'vegetables',
    'tomato': 'vegetables',
    'zucchini_courgette': 'vegetables',
    'pumpkin_squash_gourd': 'vegetables',
    'salsify': 'vegetables',
    'fresh_vegetables': 'vegetables',
    'bell_pepper_paprika': 'vegetables',
    'cucumber_pickle': 'vegetables',
    'salads_lettuce_leaf_vegetables': 'vegetables',
    'red_cabbage': 'vegetables',
    'endive': 'vegetables',
    'iceberg': 'vegetables',
    'parsnips': 'vegetables',
    'fennel': 'vegetables',
    'kohlrabi': 'vegetables',
    'kale': 'vegetables',
    'chinese_cabbage': 'vegetables',
    'turnips': 'vegetables',
    'brassica_oleracea_cabbage': 'vegetables',
    'lambs_lettuce_rapunzel': 'vegetables',
    'beetroot_beets': 'vegetables',
    'radish': 'vegetables',
    'rhubarb': 'vegetables',
    'aubergine_eggplant': 'vegetables',
    'shallot': 'vegetables',
    'rocket_arugula': 'vegetables',
    'leaf_celery': 'vegetables',
    'artichoke': 'vegetables',
    'garlic': 'vegetables',
    'melon': 'vegetables',
    'watermelon': 'vegetables',
    'green_beans': 'vegetables',
    'dried_beans': 'vegetables',
    'sweet_corn': 'vegetables',

    # 8. strawberries
    'strawberries': 'strawberries',

    # 9. tree_wood_forest
    'tree_wood_forest': 'tree_wood_forest',
    'other_tree_wood_forest': 'tree_wood_forest',
    'afforestation_reforestation': 'tree_wood_forest',
    'short_rotation_coppice': 'tree_wood_forest',
    'christmas_trees': 'tree_wood_forest',
    'nurseries': 'tree_wood_forest',
    'poplar': 'tree_wood_forest',
    'willow': 'tree_wood_forest',

    # 10. other_crops (everything else agricultural)
    'flax_linen': 'other_crops',
    'other_arable_land_crops': 'other_crops',
    'flowers_ornamental_plants': 'other_crops',
    'chrysanthemum': 'other_crops',
    'azaleas': 'other_crops',
    'shrubberries_shrubs': 'other_crops',
    'permanent_crops_perennial': 'other_crops',
    'spring_barley': 'other_crops',
    'spring_common_soft_wheat': 'other_crops',
    'spring_oats': 'other_crops',
    'spring_rye': 'other_crops',
    'summer_barley': 'other_crops',
    'summer_common_soft_wheat': 'other_crops',
    'summer_oats': 'other_crops',
    'summer_rye': 'other_crops',
    'summer_rapeseed_rape': 'other_crops',
    'winter_rapeseed_rape': 'other_crops',
    'spring_rapeseed_rape': 'other_crops',
    'rapeseed_rape': 'other_crops',
    'fibre_crops': 'other_crops',
    'hemp_cannabis': 'other_crops',
    'hops': 'other_crops',
    'tobacco': 'other_crops',
    'tagetes': 'other_crops',
    'roses': 'other_crops',
    'miscanthus_silvergrass': 'other_crops',
    'millet_sorghum': 'other_crops',
    'oats': 'other_crops',
    'rye': 'other_crops',
    'barley': 'other_crops',
    'cereal': 'other_crops',
    'unspecified_cereals': 'other_crops',
    'other_cereals': 'other_crops',
    'other_industrial_crops': 'other_crops',
    'oilseed_crops': 'other_crops',
    'sunflower': 'other_crops',
    'soy_soybeans': 'other_crops',
    'buckwheat': 'other_crops',
    'quinoa': 'other_crops',
    'phacelia': 'other_crops',
    'mustard': 'other_crops',
    'aromatic_medicinal_culinary_plants_spices_herbs': 'other_crops',
    'unspecified_aromatic_medicinal_culinary_plants_spices_herbs': 'other_crops',
    'parsly': 'other_crops',
    'chives': 'other_crops',
    'basil': 'other_crops',
    'chervil': 'other_crops',
    'angelica': 'other_crops',
    'arable_land_seed_seedlings': 'other_crops',
    'flax_linseed_oil': 'other_crops',
    'legumes_dried_pulses_protein_crops': 'other_crops',
    'legumes_harvested_green': 'other_crops',
    'sweet_lupins': 'other_crops',
    'vetches': 'other_crops',
    'other_mushrooms_energy_crops_genetically_modified_crops': 'other_crops',
    'cotton': 'other_crops',
    'rice': 'other_crops',
    'lavender': 'other_crops',
    'poppy': 'other_crops',
    'lentils': 'other_crops',
    'chickpeas': 'other_crops',
    'faba_beans': 'other_crops',
    'field_beans': 'other_crops',
    'lupins': 'other_crops',
    'sugar_cane': 'other_crops',
    'energy_crops': 'other_crops',
    'cover_crops_catch_crops': 'other_crops',
    'fallow_land': 'other_crops',
    'set_aside': 'other_crops',
    'arable_crops': 'other_crops',
}

TIER_B_CLASSES = [
    'grassland', 'maize', 'winter_cereals', 'potatoes', 'root_crops',
    'orchards', 'vegetables', 'strawberries', 'tree_wood_forest', 'other_crops'
]

# ============================================================
# TIER A: Top 20 EU-wide HCAT3 + other_agriculture
# ============================================================
# 泛欧洲 Top 20 作物（按面积和频率排序）
TIER_A_TOP_CLASSES = [
    'pasture_meadow_grassland_grass',   # 草地/牧场
    'green_silo_maize',                 # 青贮玉米
    'grain_maize_corn_popcorn',         # 籽粒玉米
    'winter_common_soft_wheat',         # 冬小麦
    'winter_barley',                    # 冬大麦
    'spring_barley',                    # 春大麦
    'winter_rapeseed_rape',             # 冬油菜
    'potatoes',                         # 马铃薯
    'sugar_beet',                       # 甜菜
    'sunflower',                        # 向日葵
    'triticale',                        # 小黑麦
    'winter_rye',                       # 冬黑麦
    'clover',                           # 三叶草
    'alfalfa_lucerne',                  # 苜蓿
    'soy_soybeans',                     # 大豆
    'temporary_grass',                  # 临时草地
    'plants_harvested_green',           # 青刈作物
    'peas',                             # 豌豆
    'vineyards_wine_vine_rebland_grapes', # 葡萄园
    'olive_olives',                     # 橄榄
]

TIER_A_OTHER_LABEL = 'other_agriculture'


def get_tier_a_label(hcat3_name):
    """Map HCAT3 class to Tier A label."""
    if hcat3_name in TIER_A_TOP_CLASSES:
        return hcat3_name
    return TIER_A_OTHER_LABEL


def get_tier_b_label(hcat3_name):
    """Map HCAT3 class to Tier B super-class."""
    return TIER_B_MAPPING.get(hcat3_name, 'other_crops')


def get_tier_c_label(hcat3_name):
    """Tier C uses the raw HCAT3 class name directly."""
    return hcat3_name
