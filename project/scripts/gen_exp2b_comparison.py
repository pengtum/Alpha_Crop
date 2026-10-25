"""生成 Exp 2b XGBoost 热力图 + LightGBM vs XGBoost 对比图。"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
import os

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'Arial',
    'font.size': 10, 'figure.dpi': 150, 'savefig.dpi': 150,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})

out_dir = 'results/experiments/exp2b_country_transfer'
COUNTRIES = ['AT','BE','CZ','DE','DK','EE','ES','FR',
             'HR','IE','LT','LV','NL','PT','RO','SE','SI','SK']

ZONE_COLORS = {
    'atlantic': '#4CAF50', 'continental': '#2196F3',
    'mediterranean': '#FF9800', 'nordic_baltic': '#9C27B0',
    'southeast': '#F44336',
}
COUNTRY_ZONE = {
    'AT':'continental','BE':'atlantic','CZ':'continental','DE':'atlantic',
    'DK':'atlantic','EE':'nordic_baltic','ES':'mediterranean','FR':'atlantic',
    'HR':'mediterranean','IE':'atlantic','LT':'nordic_baltic','LV':'nordic_baltic',
    'NL':'atlantic','PT':'mediterranean','RO':'southeast','SE':'nordic_baltic',
    'SI':'mediterranean','SK':'continental',
}

cmap = LinearSegmentedColormap.from_list('custom',
    ['#FFFFFF','#E3F2FD','#64B5F6','#1976D2','#0D47A1'], N=256)
cmap_div = LinearSegmentedColormap.from_list('div',
    ['#E53935','#FFCDD2','#FFFFFF','#C8E6C9','#2E7D32'], N=256)
colors_ft = {'ae':'#2196F3', 's2':'#FF9800', 'fusion':'#4CAF50'}
labels_ft = {'ae':'AlphaEarth', 's2':'Sentinel-2', 'fusion':'Fusion'}


def build_matrix(csv_path):
    df = pd.read_csv(csv_path)
    matrix = np.full((18, 18), np.nan)
    for _, row in df.iterrows():
        i = COUNTRIES.index(row['train_country'])
        j = COUNTRIES.index(row['test_country'])
        matrix[i, j] = row['balanced_accuracy']
    return matrix


def add_zone_bars(ax):
    for idx, c in enumerate(COUNTRIES):
        color = ZONE_COLORS[COUNTRY_ZONE[c]]
        ax.add_patch(plt.Rectangle((-1.4, idx-0.5), 0.4, 1,
                                    facecolor=color, edgecolor='none', clip_on=False))
        ax.add_patch(plt.Rectangle((idx-0.5, -1.4), 1, 0.4,
                                    facecolor=color, edgecolor='none', clip_on=False))


def add_zone_legend(ax):
    elements = [Patch(facecolor=v, label=k.replace('_',' ').title())
                for k, v in ZONE_COLORS.items()]
    ax.legend(handles=elements, loc='upper left', bbox_to_anchor=(1.15, 0.5),
              fontsize=8, title='Climate Zone', title_fontsize=9)


def savefig(fig, name):
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(out_dir, f'{name}.{ext}'))
    plt.close(fig)
    print(f'  [OK] {name}')


# ─── 1. XGBoost 18x18 heatmaps ──────────────────────────
print('1. XGBoost heatmaps...')
for ft in ['ae', 's2', 'fusion']:
    matrix = build_matrix(os.path.join(out_dir, f'transfer_{ft}_nocoords_xgb_tier_b.csv'))
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(matrix, cmap=cmap, aspect='equal', vmin=0, vmax=0.65)
    ax.set_xticks(range(18)); ax.set_xticklabels(COUNTRIES, fontsize=8, rotation=45, ha='right')
    ax.set_yticks(range(18)); ax.set_yticklabels(COUNTRIES, fontsize=8)
    ax.set_xlabel('Test Country'); ax.set_ylabel('Train Country')
    ax.set_title(f'Country Transfer — {labels_ft[ft]} (XGBoost, Pure Spectral)')
    for i in range(18):
        for j in range(18):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6,
                        color='white' if val > 0.35 else 'black')
            elif i == j:
                ax.text(j, i, '--', ha='center', va='center', fontsize=7, color='gray')
    add_zone_bars(ax)
    plt.colorbar(im, ax=ax, shrink=0.8, label='Balanced Accuracy')
    add_zone_legend(ax)
    fig.tight_layout()
    savefig(fig, f'transfer_matrix_{ft}_nocoords_xgb')

# ─── 2. Delta heatmaps (XGB - LGBM) ─────────────────────
print('2. Delta heatmaps...')
for ft in ['ae', 's2', 'fusion']:
    m_lgbm = build_matrix(os.path.join(out_dir, f'transfer_{ft}_nocoords_tier_b.csv'))
    m_xgb = build_matrix(os.path.join(out_dir, f'transfer_{ft}_nocoords_xgb_tier_b.csv'))
    delta = m_xgb - m_lgbm

    fig, ax = plt.subplots(figsize=(10, 9))
    vmax = max(abs(np.nanmin(delta)), abs(np.nanmax(delta)), 0.01)
    im = ax.imshow(delta, cmap=cmap_div, aspect='equal', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(18)); ax.set_xticklabels(COUNTRIES, fontsize=8, rotation=45, ha='right')
    ax.set_yticks(range(18)); ax.set_yticklabels(COUNTRIES, fontsize=8)
    ax.set_xlabel('Test Country'); ax.set_ylabel('Train Country')
    ax.set_title(f'XGBoost - LightGBM Delta BA — {labels_ft[ft]}')
    for i in range(18):
        for j in range(18):
            val = delta[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:+.02f}', ha='center', va='center', fontsize=5.5,
                        color='black' if abs(val) < vmax*0.6 else 'white')
            elif i == j:
                ax.text(j, i, '--', ha='center', va='center', fontsize=7, color='gray')
    add_zone_bars(ax)
    plt.colorbar(im, ax=ax, shrink=0.8, label='Delta BA (XGB - LGBM)')
    add_zone_legend(ax)
    fig.tight_layout()
    savefig(fig, f'delta_xgb_lgbm_{ft}_nocoords')

# ─── 3. Train-country generalization bars ─────────────────
print('3. Train-country generalization...')
fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(18)
w = 0.18
for i, ft in enumerate(['ae', 's2', 'fusion']):
    lgbm = pd.read_csv(os.path.join(out_dir, f'transfer_{ft}_nocoords_tier_b.csv'))
    xgb = pd.read_csv(os.path.join(out_dir, f'transfer_{ft}_nocoords_xgb_tier_b.csv'))
    l_m = lgbm.groupby('train_country')['balanced_accuracy'].mean()
    x_m = xgb.groupby('train_country')['balanced_accuracy'].mean()
    l_v = [l_m.get(c, 0) for c in COUNTRIES]
    x_v = [x_m.get(c, 0) for c in COUNTRIES]
    offset = (i - 1) * w * 2
    ax.bar(x + offset - w/2, l_v, w, label=f'{ft.upper()} LGBM',
           color=colors_ft[ft], alpha=0.5, edgecolor='white', linewidth=0.3)
    ax.bar(x + offset + w/2, x_v, w, label=f'{ft.upper()} XGB',
           color=colors_ft[ft], alpha=1.0, edgecolor='white', linewidth=0.3, hatch='//')
ax.set_xticks(x); ax.set_xticklabels(COUNTRIES, fontsize=9)
ax.set_ylabel('Mean OOD Balanced Accuracy')
ax.set_title('Train-Country Generalization: LightGBM vs XGBoost')
ax.legend(frameon=True, fancybox=False, edgecolor='gray', fontsize=7, ncol=3)
ax.grid(axis='y', alpha=0.3, linestyle='--'); ax.set_ylim(0, 0.45)
fig.tight_layout()
savefig(fig, 'train_generalization_lgbm_vs_xgb')

# ─── 4. Scatter: LGBM BA vs XGB BA ───────────────────────
print('4. Scatter comparison...')
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, ft in enumerate(['ae', 's2', 'fusion']):
    lgbm = pd.read_csv(os.path.join(out_dir, f'transfer_{ft}_nocoords_tier_b.csv'))
    xgb = pd.read_csv(os.path.join(out_dir, f'transfer_{ft}_nocoords_xgb_tier_b.csv'))
    merged = lgbm.merge(xgb, on=['train_country','test_country'], suffixes=('_lgbm','_xgb'))
    ax = axes[idx]
    ax.scatter(merged['balanced_accuracy_lgbm'], merged['balanced_accuracy_xgb'],
               alpha=0.5, s=20, color=colors_ft[ft])
    lim = max(merged['balanced_accuracy_lgbm'].max(), merged['balanced_accuracy_xgb'].max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, linewidth=1)
    ax.set_xlabel('LightGBM BA'); ax.set_ylabel('XGBoost BA')
    ax.set_title(labels_ft[ft]); ax.set_aspect('equal')
    xgb_wins = (merged['balanced_accuracy_xgb'] > merged['balanced_accuracy_lgbm']).sum()
    corr = np.corrcoef(merged['balanced_accuracy_lgbm'], merged['balanced_accuracy_xgb'])[0,1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}\nXGB wins: {xgb_wins}/306',
            transform=ax.transAxes, fontsize=9, va='top', fontweight='bold')
fig.suptitle('LightGBM vs XGBoost: Per-Pair BA (306 pairs)', fontsize=13, y=1.02)
fig.tight_layout()
savefig(fig, 'lgbm_vs_xgb_scatter')

# ─── 5. Winner map ───────────────────────────────────────
print('5. Winner map...')
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
cmap3 = ListedColormap(['#FFCDD2', '#F5F5F5', '#C8E6C9'])
for idx, ft in enumerate(['ae', 's2', 'fusion']):
    m_lgbm = build_matrix(os.path.join(out_dir, f'transfer_{ft}_nocoords_tier_b.csv'))
    m_xgb = build_matrix(os.path.join(out_dir, f'transfer_{ft}_nocoords_xgb_tier_b.csv'))
    delta = m_xgb - m_lgbm
    win_map = np.where(np.isnan(delta), np.nan, np.sign(delta))
    ax = axes[idx]
    ax.imshow(win_map, cmap=cmap3, aspect='equal', vmin=-1, vmax=1)
    ax.set_xticks(range(18)); ax.set_xticklabels(COUNTRIES, fontsize=6, rotation=45, ha='right')
    ax.set_yticks(range(18)); ax.set_yticklabels(COUNTRIES, fontsize=6)
    n_xgb = int(np.nansum(win_map > 0))
    n_lgbm = int(np.nansum(win_map < 0))
    ax.set_title(f'{labels_ft[ft]}\nXGB wins {n_xgb}, LGBM wins {n_lgbm}', fontsize=10)
    for i in range(18):
        for j in range(18):
            if i == j:
                ax.text(j, i, '--', ha='center', va='center', fontsize=5, color='gray')
fig.suptitle('Winner Map: Green = XGBoost Better, Red = LightGBM Better', fontsize=12, y=1.02)
fig.tight_layout()
savefig(fig, 'winner_map_xgb_vs_lgbm')

print('\nAll figures done!')
