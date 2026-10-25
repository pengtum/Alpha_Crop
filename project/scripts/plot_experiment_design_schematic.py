"""
Experimental-design schematic — column-aligned version.

Layout:
  1. Research question (top)
  2. Three protocol cards (A / B / C) horizontally aligned
  3. Findings column-aligned UNDER their producing protocol:
       Column A: Symptom (top) + Footprint (below)
       Column B: Mechanism
       Column C: Mitigation
  4. Diagnostic-arc arrows weave between findings:
       Symptom → motivates → Mechanism → predicts → Footprint
              → operationalises → Mitigation

Output: 0001_draft_v1/figures/fig_experiment_design.{pdf,png}
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'Arial',
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1, 'figure.dpi': 150,
})

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'Manuscript', 'Paper_Alpha_Earth', '0001_draft_v1', 'figures'
)
os.makedirs(OUT_DIR, exist_ok=True)

COL_RQ = '#1F3A93'
COL_A = '#1976D2'
COL_B = '#2E7D32'
COL_C = '#E65100'
COL_GREY = '#6A6A6A'


def box(ax, x, y, w, h, fc, ec, lw=1.4, pad=0.02):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f'round,pad={pad},rounding_size=0.8',
        facecolor=fc, edgecolor=ec, linewidth=lw,
    ))


def arrow(ax, x1, y1, x2, y2, color=COL_GREY, lw=1.5, mutation=14,
          style='-|>', curve=0):
    cs = f'arc3,rad={curve}'
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=color, linewidth=lw,
        mutation_scale=mutation, connectionstyle=cs,
    ))


fig, ax = plt.subplots(figsize=(13, 8))
ax.set_xlim(0, 100); ax.set_ylim(0, 75)
ax.set_aspect('equal'); ax.axis('off')

# ====================================================================
# 1) Research question
# ====================================================================
box(ax, 6, 66, 88, 7.5, '#E8EEF7', COL_RQ, lw=1.6)
ax.text(50, 71, 'Research question',
        ha='center', va='center', fontsize=10, color=COL_RQ,
        fontweight='bold', style='italic')
ax.text(50, 68.5,
        "Does AlphaEarth co-encode crop semantics and a geographic confound?",
        ha='center', va='center', fontsize=11, color='#1A1A1A')

# ====================================================================
# 2) Three protocol cards
# ====================================================================
PROT_Y, PROT_H = 36, 24
PROT_W, PROT_X = 28, [4, 36, 68]

# --- Protocol A ---
x = PROT_X[0]
box(ax, x, PROT_Y, PROT_W, PROT_H, '#E3F2FD', COL_A, lw=1.5)
ax.text(x + PROT_W/2, PROT_Y + PROT_H - 2.5,
        'Protocol A',
        ha='center', va='top', fontsize=12, color=COL_A, fontweight='bold')
ax.text(x + PROT_W/2, PROT_Y + PROT_H - 5.5,
        'Classification across\nthe ID–OOD spectrum',
        ha='center', va='top', fontsize=9.5, color=COL_A)
ax.text(x + PROT_W/2, PROT_Y + PROT_H - 12,
        'A1   Pooled CV\n'
        'A2   Leave-one-country-out\n'
        'A3   Pairwise 1$\\to$1',
        ha='center', va='center', fontsize=9.5, color='#1A1A1A')
ax.text(x + PROT_W/2, PROT_Y + 2.5,
        'produces:  Symptom + Footprint',
        ha='center', va='center', fontsize=8.5, color=COL_A,
        fontweight='bold', style='italic')

# --- Protocol B ---
x = PROT_X[1]
box(ax, x, PROT_Y, PROT_W, PROT_H, '#E8F5E9', COL_B, lw=1.5)
ax.text(x + PROT_W/2, PROT_Y + PROT_H - 2.5,
        'Protocol B',
        ha='center', va='top', fontsize=12, color=COL_B, fontweight='bold')
ax.text(x + PROT_W/2, PROT_Y + PROT_H - 5.5,
        'Representation\ndiagnostics',
        ha='center', va='top', fontsize=9.5, color=COL_B)
ax.text(x + PROT_W/2, PROT_Y + PROT_H - 12,
        'B1   CKA alignment\n'
        'B2   Country-pred probe\n'
        'B3   UMAP geometry\n'
        '+ Coord ablation',
        ha='center', va='center', fontsize=9.5, color='#1A1A1A')
ax.text(x + PROT_W/2, PROT_Y + 2.5,
        'produces:  Mechanism',
        ha='center', va='center', fontsize=8.5, color=COL_B,
        fontweight='bold', style='italic')

# --- Protocol C ---
x = PROT_X[2]
box(ax, x, PROT_Y, PROT_W, PROT_H, '#FFF3E0', COL_C, lw=1.5)
ax.text(x + PROT_W/2, PROT_Y + PROT_H - 2.5,
        'Protocol C',
        ha='center', va='top', fontsize=12, color=COL_C, fontweight='bold')
ax.text(x + PROT_W/2, PROT_Y + PROT_H - 5.5,
        'Label efficiency',
        ha='center', va='top', fontsize=9.5, color=COL_C)
ax.text(x + PROT_W/2, PROT_Y + PROT_H - 12,
        'Iso-accuracy budget\n'
        'curves across budgets,\n'
        'sources, and seeds',
        ha='center', va='center', fontsize=9.5, color='#1A1A1A')
ax.text(x + PROT_W/2, PROT_Y + 2.5,
        'produces:  Mitigation',
        ha='center', va='center', fontsize=8.5, color=COL_C,
        fontweight='bold', style='italic')

# ====================================================================
# 3) RQ-to-protocol arrows
# ====================================================================
for x in PROT_X:
    arrow(ax, x + PROT_W/2, 66,
          x + PROT_W/2, PROT_Y + PROT_H + 0.5,
          color=COL_GREY, lw=1.3)

# ====================================================================
# 4) Findings — COLUMN-ALIGNED under their producing protocol
# ====================================================================
FIND_W, FIND_H = 22, 6.5

cxA = PROT_X[0] + PROT_W/2   # 18
cxB = PROT_X[1] + PROT_W/2   # 50
cxC = PROT_X[2] + PROT_W/2   # 82

# Symptom (column A, TOP)
y_top = 24.5
box(ax, cxA - FIND_W/2, y_top, FIND_W, FIND_H, '#FAFAFA', COL_A, lw=1.3)
ax.text(cxA, y_top + FIND_H - 1.6, 'Symptom',
        ha='center', va='top', fontsize=11, fontweight='bold', color=COL_A)
ax.text(cxA, y_top + 1.5, '§5.1.1   (from A1 + A2)',
        ha='center', va='center', fontsize=8.5, color=COL_A, style='italic')

# Mechanism (column B, MIDDLE) and Mitigation (column C, MIDDLE)
y_mid = 16.5
box(ax, cxB - FIND_W/2, y_mid, FIND_W, FIND_H, '#FAFAFA', COL_B, lw=1.3)
ax.text(cxB, y_mid + FIND_H - 1.6, 'Mechanism',
        ha='center', va='top', fontsize=11, fontweight='bold', color=COL_B)
ax.text(cxB, y_mid + 1.5, '§5.2   (from B)',
        ha='center', va='center', fontsize=8.5, color=COL_B, style='italic')

box(ax, cxC - FIND_W/2, y_mid, FIND_W, FIND_H, '#FAFAFA', COL_C, lw=1.3)
ax.text(cxC, y_mid + FIND_H - 1.6, 'Mitigation',
        ha='center', va='top', fontsize=11, fontweight='bold', color=COL_C)
ax.text(cxC, y_mid + 1.5, '§5.3   (from C)',
        ha='center', va='center', fontsize=8.5, color=COL_C, style='italic')

# Footprint (column A, BOTTOM) — pushed well down to give arrow room
y_bot = 7.5
box(ax, cxA - FIND_W/2, y_bot, FIND_W, FIND_H, '#FAFAFA', COL_A, lw=1.3)
ax.text(cxA, y_bot + FIND_H - 1.6, 'Footprint',
        ha='center', va='top', fontsize=11, fontweight='bold', color=COL_A)
ax.text(cxA, y_bot + 1.5, '§5.1.2   (from A3)',
        ha='center', va='center', fontsize=8.5, color=COL_A, style='italic')

# ====================================================================
# 5) Protocol → Finding arrows (one arrow per protocol)
# ====================================================================
# A → Symptom (and the chain implies Footprint is also from A)
arrow(ax, cxA, PROT_Y - 0.5, cxA, y_top + FIND_H + 0.3,
      color=COL_A, lw=1.4)
# Faint vertical line linking Symptom box (bottom) to Footprint box (top),
# in A's column colour: signals "both findings are in A's column"
ax.plot([cxA, cxA], [y_top - 0.3, y_bot + FIND_H + 0.3],
        color=COL_A, lw=1.2, linestyle=':', alpha=0.55)

# B → Mechanism
arrow(ax, cxB, PROT_Y - 0.5, cxB, y_mid + FIND_H + 0.3,
      color=COL_B, lw=1.4)
# C → Mitigation
arrow(ax, cxC, PROT_Y - 0.5, cxC, y_mid + FIND_H + 0.3,
      color=COL_C, lw=1.4)

# ====================================================================
# 6) Diagnostic-arc arrows: Symptom + Footprint -> Mechanism -> Mitigation
# ====================================================================
# Symptom (top-A) -> Mechanism (mid-B): rightward, slight downward angle
arrow(ax, cxA + FIND_W/2 + 0.5, y_top + FIND_H/2 - 1.0,
      cxB - FIND_W/2 - 0.5, y_mid + FIND_H/2 + 1.0,
      color=COL_GREY, lw=1.1, mutation=11, curve=0.04)

# Footprint (bottom-A) -> Mechanism (mid-B): rightward, slight upward angle
arrow(ax, cxA + FIND_W/2 + 0.5, y_bot + FIND_H/2 + 1.0,
      cxB - FIND_W/2 - 0.5, y_mid + FIND_H/2 - 1.0,
      color=COL_GREY, lw=1.1, mutation=11, curve=-0.04)

# A single 'motivate' label between the two A-findings and Mechanism
ax.text((cxA + FIND_W/2 + cxB - FIND_W/2) / 2, y_mid + FIND_H/2,
        'motivate', ha='center', va='center',
        fontsize=8.5, color=COL_GREY, style='italic')

# Mechanism (mid-B) -> Mitigation (mid-C): rightward
arrow(ax, cxB + FIND_W/2 + 0.5, y_mid + FIND_H/2,
      cxC - FIND_W/2 - 0.5, y_mid + FIND_H/2,
      color=COL_GREY, lw=1.1, mutation=11)
ax.text((cxB + FIND_W/2 + cxC - FIND_W/2) / 2, y_mid + FIND_H + 0.6,
        'operationalises', ha='center', va='center',
        fontsize=8.5, color=COL_GREY, style='italic')

# ====================================================================
# 7) Footer
# ====================================================================
ax.text(50, 3.5,
        'Findings are grouped by producing protocol. The diagnostic arc is '
        'three-stage: empirical observations from A motivate the mechanism '
        'from B, which operationalises into the mitigation from C.',
        ha='center', va='center', fontsize=8.5, color=COL_GREY,
        style='italic')

for ext in ['png', 'pdf']:
    fig.savefig(os.path.join(OUT_DIR, f'fig_experiment_design.{ext}'))
plt.close(fig)
print('Saved fig_experiment_design.{png,pdf} to', OUT_DIR)
