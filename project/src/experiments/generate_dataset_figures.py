#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Generate dataset overview figures for Alpha_Crops

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

BASE_DIR = os.path.normpath('G:/我的云端硬盘/Research/Alpha_Crops')
FEATURE_DIR = os.path.join(BASE_DIR, 'data', 'eu_processed', 'features', 'ae')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results', 'experiments', 'dataset_overview')
os.makedirs(OUTPUT_DIR, exist_ok=True)

COUNTRY_NAMES = {
    'AT': 'Austria', 'BE': 'Belgium', 'CZ': 'Czechia', 'DE': 'Germany',
    'DK': 'Denmark', 'EE': 'Estonia', 'ES': 'Spain', 'FR': 'France',
    'HR': 'Croatia', 'IE': 'Ireland', 'LT': 'Lithuania', 'LV': 'Latvia',
    'NL': 'Netherlands', 'PT': 'Portugal', 'RO': 'Romania', 'SE': 'Sweden',
    'SI': 'Slovenia', 'SK': 'Slovakia', 'FI': 'Finland',
}

CLIMATE_ZONES = {
    'Mediterranean':   ['ES', 'PT', 'HR', 'SI'],
    'Atlantic':        ['FR', 'IE', 'NL', 'BE'],
    'Continental':     ['DE', 'AT', 'CZ', 'SK'],
    'Nordic & Baltic': ['DK', 'SE', 'FI', 'EE', 'LV', 'LT'],
    'Southeast':       ['RO'],
}
ZONE_COLORS = {
    'Mediterranean':   '#E07B39',
    'Atlantic':        '#4C8C2B',
    'Continental':     '#3B7DD8',
    'Nordic & Baltic': '#7B68AE',
    'Southeast':       '#C94C4C',
}
COUNTRY_ZONE = {}
for zone, codes in CLIMATE_ZONES.items():
    for c in codes:
        COUNTRY_ZONE[c] = zone


def load_data():
    dfs = []
    for split in ['train', 'val', 'test']:
        fp = os.path.join(FEATURE_DIR, f'features_{split}.parquet')
        if not os.path.exists(fp):
            print(f'[WARNING] File not found: {fp}')
            continue
        df = pd.read_parquet(fp, columns=['country', 'label_tier_b'])
        df['split'] = split
        dfs.append(df)
        print(f'  Loaded {split}: {len(df):,} records')
    if not dfs:
        print('[ERROR] No data files loaded.')
        sys.exit(1)
    return pd.concat(dfs, ignore_index=True)


def print_statistics(df):
    print()
    print('=' * 60)
    total = len(df)
    train_n = int((df['split'] == 'train').sum())
    val_n = int((df['split'] == 'val').sum())
    test_n = int((df['split'] == 'test').sum())
    print(f'Total samples: {total:,}')
    print(f'  Train: {train_n:,}')
    print(f'  Val:   {val_n:,}')
    print(f'  Test:  {test_n:,}')
    print()
    print('-- Samples per country --')
    country_counts = df['country'].value_counts().sort_values(ascending=False)
    for code, cnt in country_counts.items():
        name = COUNTRY_NAMES.get(code, code)
        zone = COUNTRY_ZONE.get(code, '?')
        print(f'  {code} ({name:>15s}) [{zone:>16s}]: {cnt:>8,}')
    print()
    print('-- Tier B class distribution --')
    class_counts = df['label_tier_b'].value_counts().sort_values(ascending=False)
    for cls, cnt in class_counts.items():
        pct = cnt / total * 100
        print(f'  {cls:>25s}: {cnt:>8,}  ({pct:5.1f}%)')
    return country_counts, class_counts


def save_fig(fig, name):
    png_path = os.path.join(OUTPUT_DIR, f'{name}.png')
    pdf_path = os.path.join(OUTPUT_DIR, f'{name}.pdf')
    fig.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f'  Saved: {png_path}')
    print(f'  Saved: {pdf_path}')
    plt.close(fig)


def fig_samples_per_country(df):
    print()
    print('-- Generating Figure 1: samples_per_country --')
    counts = df['country'].value_counts().sort_values(ascending=True)
    labels = [f'{COUNTRY_NAMES.get(c, c)} ({c})' for c in counts.index]
    colors = [ZONE_COLORS.get(COUNTRY_ZONE.get(c, '?'), '#888888') for c in counts.index]

    fig, ax = plt.subplots(figsize=(8, max(5, len(counts) * 0.42)))
    bars = ax.barh(range(len(counts)), counts.values, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Number of samples', fontsize=12)
    ax.set_title('Sample Count per Country', fontsize=14, fontweight='bold', pad=12)

    max_val = counts.max()
    for i, (bar, val) in enumerate(zip(bars, counts.values)):
        if val > max_val * 0.6:
            ax.text(val - max_val * 0.02, i, f'{val:,}', va='center', ha='right',
                    fontsize=9, color='white', fontweight='bold')
        else:
            ax.text(val + max_val * 0.01, i, f'{val:,}', va='center', ha='left',
                    fontsize=9, color='#333333')

    legend_elements = [Patch(facecolor=ZONE_COLORS[z], label=z) for z in CLIMATE_ZONES]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
              frameon=True, edgecolor='#cccccc', title='Climate zone', title_fontsize=10)
    ax.set_xlim(0, max_val * 1.15)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    save_fig(fig, 'samples_per_country')


def fig_class_distribution(df):
    print()
    print('-- Generating Figure 2: class_distribution --')
    counts = df['label_tier_b'].value_counts().sort_values(ascending=True)
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(9, max(4, len(counts) * 0.45)))
    cmap = plt.cm.get_cmap('tab20', len(counts))
    colors = [cmap(i) for i in range(len(counts))]

    bars = ax.barh(range(len(counts)), counts.values, color=colors[::-1],
                   edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index, fontsize=10)
    ax.set_xlabel('Number of samples', fontsize=12)
    ax.set_title('Tier B Class Distribution (All Countries)', fontsize=14,
                 fontweight='bold', pad=12)

    max_val = counts.max()
    for i, (bar, val) in enumerate(zip(bars, counts.values)):
        pct = val / total * 100
        label_text = f'{val:,}  ({pct:.1f}%)'
        if val > max_val * 0.5:
            ax.text(val - max_val * 0.01, i, label_text, va='center', ha='right',
                    fontsize=8.5, color='white', fontweight='bold')
        else:
            ax.text(val + max_val * 0.01, i, label_text, va='center', ha='left',
                    fontsize=8.5, color='#333333')

    ax.set_xlim(0, max_val * 1.22)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    save_fig(fig, 'class_distribution')


def fig_country_class_heatmap(df):
    print()
    print('-- Generating Figure 3: country_class_heatmap --')
    ct = pd.crosstab(df['country'], df['label_tier_b'])
    country_order = ct.sum(axis=1).sort_values(ascending=False).index.tolist()
    class_order = ct.sum(axis=0).sort_values(ascending=False).index.tolist()
    ct = ct.loc[country_order, class_order]
    y_labels = [f'{COUNTRY_NAMES.get(c, c)} ({c})' for c in ct.index]

    fig, ax = plt.subplots(figsize=(max(8, len(class_order) * 0.9),
                                     max(5, len(country_order) * 0.52)))
    data = ct.values.astype(float)
    data_log = np.where(data > 0, np.log10(data), 0)

    im = ax.imshow(data_log, aspect='auto', cmap='YlOrRd', interpolation='nearest')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    tick_vals = [0, 1, 2, 3, 4, 5]
    tick_labels_list = ['0', '10', '100', '1k', '10k', '100k']
    max_log = data_log.max()
    valid_ticks = [v for v in tick_vals if v <= max_log + 0.5]
    valid_labels = tick_labels_list[:len(valid_ticks)]
    cbar.set_ticks(valid_ticks)
    cbar.set_ticklabels(valid_labels)
    cbar.set_label('Sample count', fontsize=11)

    ax.set_xticks(range(len(class_order)))
    ax.set_xticklabels(class_order, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(country_order)))
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_title('Sample Count: Country x Tier B Class', fontsize=14,
                 fontweight='bold', pad=14)

    for i in range(len(country_order)):
        for j in range(len(class_order)):
            val = int(data[i, j])
            if val == 0:
                text = '-'
                color = '#aaaaaa'
            elif val < 100:
                text = str(val)
                color = '#333333'
            elif val < 1000:
                text = str(val)
                color = '#333333' if data_log[i, j] < (max_log * 0.6) else 'white'
            else:
                text = f'{val / 1000:.1f}k' if val < 10000 else f'{val // 1000}k'
                color = 'white' if data_log[i, j] > (max_log * 0.5) else '#333333'
            ax.text(j, i, text, ha='center', va='center', fontsize=7.5, color=color)

    ax.tick_params(axis='both', length=0)
    save_fig(fig, 'country_class_heatmap')


def main():
    print('=' * 60)
    print('Alpha_Crops -- Dataset Statistics Figure Generation')
    print('=' * 60)
    df = load_data()
    country_counts, class_counts = print_statistics(df)
    fig_samples_per_country(df)
    fig_class_distribution(df)
    fig_country_class_heatmap(df)
    print()
    print('=' * 60)
    print(f'All figures saved to: {OUTPUT_DIR}')
    print('=' * 60)


if __name__ == '__main__':
    main()

