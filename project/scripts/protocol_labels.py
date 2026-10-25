"""Single source of truth for how the evaluation protocols are named.

Four distinct train/test protocols are used across this project, and two of them
are both loosely called "in-distribution". They differ by 13 percentage points,
so every figure must state which one it shows.

    pooled_cv          AE 68.5 / S2 67.7 / Fusion 71.5   (headline ID number)
    within_country_cv  AE 55.3 / S2 55.0 / Fusion 59.6   (per-country ID ceiling)
    loco               AE 42.7 / S2 43.2 / Fusion 45.2   (headline OOD number)
    pairwise           AE 27.9 / S2 29.0 / Fusion 30.7   (110 ordered pairs)

Use NAME[key] in titles and stamp(fig, key) for the figure footer. Never write a
bare "in-distribution" or "ID" on a figure: say which protocol produced it.
"""

PROTOCOL = {
    'pooled_cv': dict(
        name='Pooled cross-validation',
        short='pooled CV',
        regime='in-distribution',
        defn='all eleven countries pooled, 5-fold stratified CV; every fold contains '
             'parcels from every country',
        source='results/experiments/exp1_id_baseline/'),
    'within_country_cv': dict(
        name='Within-country cross-validation',
        short='within-country CV',
        regime='in-distribution',
        defn='one model per country, 5-fold CV inside that country; train and test '
             'parcels are from the same country',
        source='results/experiments/exp0_per_country_cv/'),
    'loco': dict(
        name='Leave-one-country-out',
        short='LOCO',
        regime='out-of-distribution',
        defn='trained on the other ten countries, tested on the held-out country, '
             'whose parcels never appear in training',
        source='results/experiments/exp2_loco/'),
    'pairwise': dict(
        name='Pairwise country-to-country transfer',
        short='pairwise 1→1',
        regime='out-of-distribution',
        defn='trained on one source country, tested on one target country; '
             '110 ordered (source, target) pairs',
        source='results/experiments/exp2b_country_transfer/'),
    'spatial_block_cv': dict(
        name='Spatial-block cross-validation',
        short='spatial-block CV',
        regime='in-distribution (robustness check)',
        defn='one model per country, folds are disjoint k-means spatial blocks so no '
             'test parcel has a near training neighbour',
        source='results/experiments/exp0_per_country_cv/ (block variant)'),
    'probe': dict(
        name='Country-predictability probe',
        short='country probe',
        regime='diagnostic, not a crop-accuracy protocol',
        defn='pooled random split stratified by country, 80/10/10; the prediction target '
             'is country identity, not crop class',
        source='results/experiments/exp8_debias_test/'),
    'diagnostic': dict(
        name='Representation diagnostic',
        short='diagnostic',
        regime='no train/test split',
        defn='computed directly on a balanced subsample of features and labels; no model '
             'is trained and no accuracy is measured',
        source='results/experiments/exp5_feature_analysis/'),
    'dataset': dict(
        name='Dataset composition',
        short='dataset',
        regime='no model evaluated',
        defn='counts and covariates of the parcel pool itself; no classifier is fitted',
        source='results/dataset_summary/'),
}

NAME  = {k: v['name'] for k, v in PROTOCOL.items()}
SHORT = {k: v['short'] for k, v in PROTOCOL.items()}


def line(key, extra=''):
    """One-sentence protocol statement for a figure footer."""
    p = PROTOCOL[key]
    s = f"Evaluation protocol — {p['name']} ({p['regime']}): {p['defn']}."
    return f'{s} {extra}'.strip()


def stamp(fig, key, extra='', x=0.0, y=-0.012, fontsize=7.6, color='#555', ha='left'):
    """Write the protocol statement under a figure."""
    fig.text(x, y, line(key, extra), fontsize=fontsize, color=color,
             style='italic', ha=ha, va='top')


def stamp_multi(fig, keys, extra='', x=0.0, y=-0.012, fontsize=7.6, color='#555',
                ha='left', panel_letters=True):
    """Figures that show more than one protocol side by side.

    One protocol per line: a single joined line stretches the tight bounding box
    far beyond the figure width.
    """
    seen, lines = set(), []
    for i, k in enumerate(keys):
        tag = f'{chr(97 + i)} — ' if panel_letters else ''
        if not panel_letters and k in seen:
            continue
        seen.add(k)
        lines.append(f"{tag}{PROTOCOL[k]['name']} ({PROTOCOL[k]['regime']}): {PROTOCOL[k]['defn']}.")
    body = 'Evaluation protocols.\n' + '\n'.join(lines)
    if extra:
        body += '\n' + extra.lstrip('\n')
    fig.text(x, y, body, fontsize=fontsize, color=color, style='italic',
             ha=ha, va='top', linespacing=1.45)
