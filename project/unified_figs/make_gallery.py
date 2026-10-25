"""把 out/ 下全部统一图（PNG）与英文图例汇编成一份 PDF：按图类排序，每类依次为 crop、lcz、lucas。
用法：~/AlphaCrops_local/.venv/bin/python -B make_gallery.py
"""
import os
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

import ustyle as us  # noqa: E402

FIGS = ['uf0_overview', 'uf1_design', 'uf2_evaluation', 'uf3_labels_representation', 'uf4_complementarity',
        'uf5_domain_heatmap', 'uf6_ae_drivers']
TITLE = {'uf0_overview': 'Overview across tasks', 'uf1_design': 'Design', 'uf2_evaluation': 'Evaluation',
         'uf3_labels_representation': 'Labels and representation on one scale',
         'uf4_complementarity': 'Complementarity', 'uf5_domain_heatmap': 'Per-domain results',
         'uf6_ae_drivers': 'Where the AE advantage comes from'}


def pages():
    for f in FIGS:
        names = [f] if f == 'uf0_overview' else [f'{f}_{t}' for t in us.TASKS]
        for n in names:
            if os.path.exists(os.path.join(us.OUT, n + '.png')):
                yield f, n


def main():
    us.setup()
    out = os.path.join(us.OUT, 'unified_figures_gallery.pdf')
    with PdfPages(out) as pdf:
        for i, (f, n) in enumerate(pages(), 1):
            task = n.split('_')[-1] if n != f else None
            img = mpimg.imread(os.path.join(us.OUT, n + '.png'))
            leg = open(os.path.join(us.OUT, n + '_legend.txt'), encoding='utf-8').read().strip()
            fig = plt.figure(figsize=(8.27, 11.69))
            head = f"{i}. {TITLE[f]}" + (f" — {us.META[task]['title']}" if task else '')
            fig.text(0.06, 0.965, head, fontsize=10, fontweight='bold', va='top')
            h, w = img.shape[:2]
            aw = 0.88
            ah = min(aw * h / w * 8.27 / 11.69, 0.52)
            aw = ah * w / h * 11.69 / 8.27
            ax = fig.add_axes([0.5 - aw / 2, 0.94 - ah, aw, ah])
            ax.imshow(img)
            ax.axis('off')
            wrapped = '\n'.join(textwrap.fill(p, 128) for p in leg.split('\n') if p.strip())
            fig.text(0.06, 0.92 - ah, wrapped, fontsize=6, va='top', ha='left', wrap=False, linespacing=1.35,
                     fontfamily='DejaVu Sans')
            pdf.savefig(fig)
            plt.close(fig)
    print(out, i, 'pages')


if __name__ == '__main__':
    main()
