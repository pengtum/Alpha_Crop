"""UF2 评估图 uf2_evaluation_{task}（双栏 183 × 62 mm，三个任务同一段代码循环生成）。

a｜示例单元：标签怎样移动前沿（AE 特征；零标签迁移、+200 / +400 随机目标标签、参照臂 REF（点线）的 τ 族 OA–BA 曲线）
b｜同一单元：表征怎样移动前沿（零标签迁移下 AE 与 S2 的 τ 族曲线）
c｜两种读法对比：τ = 0 读数与同 OA 读数（任务层；两组各一个轴，各调一次 us.forest(cols=True)，
   右侧两列与角注 us.forest_note 由共享组件画；域数 ≥ 5 画 95% t CI，否则画各域的值）

曲线用 us.oa_ba_curve，域名用 us.dname，术语用 us.TERM，图例文字的设计常量取 us.DESIGN / us.META。
只读 source_data/（us.read / us.est / us.row）；图中印出的数字全部来自源文件，不重算统计量。
箭头位置、文字位置只是几何摆放：用曲线的上包络（与登记口径同一算法的局部副本）确定箭头放在窗口内哪一点，
文字位置按“与曲线、已放文字不重叠”的规则在候选位置中自动选择，三个任务用同一规则。
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '2')

import matplotlib

matplotlib.use('Agg')
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.legend import Legend  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator, MultipleLocator  # noqa: E402

import ustyle as us  # noqa: E402

us.setup()
C, FS, MM = us.C, us.FS, us.MM
fmt = us.fmt_pp

W, H = 183, 62                              # 图幅（mm）
BOT, AH = 9.0, 43.0                         # a、b 轴底与轴高（mm）
TOP = BOT + AH                              # 轴顶
HEAD_Y = TOP + 4.6                          # 面板标号与小标题的基线
OFFS = np.round(np.linspace(-4.0, 0.0, 41), 1)   # 同 OA 窗口格点（与登记口径相同）
EPS = 1e-9
X_SPAN = 22.0                               # a、b 的 OA 轴跨度（pp），三个任务相同，保证尺度可比
LW_REF = 0.8                                # 参照臂 REF：细点线（us.REF_LS）
ARROW_MS, ARROW_LW = 4.5, 0.6               # 差值箭头：mutation_scale（pt）与线宽（pt）
HEAD_MM = 0.95                              # 渲染后的箭头头长：几何 0.4 × 4.5 pt ≈ 0.64 mm，加 0.6 pt 轮廓（尖端斜接）
ARROW_MIN_MM = 2 * HEAD_MM + 0.5            # 竖向间距小于两个头长 + 0.5 mm 可见箭杆时，改画无头短竖线
DPI = matplotlib.rcParams['figure.dpi']
PX_MM = DPI / 25.4                          # 显示坐标中每毫米的像素数


# ================================================================== 数据辅助
def unit_val(ur, unit, name, feat, r=0):
    """unit_rows 中唯一一行的值。"""
    s = ur[(ur.unit == unit) & (ur.name == name) & (ur.feat == feat) & (ur.r == r)]
    assert len(s) == 1, (unit, name, feat, r, len(s))
    return float(s.value.iloc[0])


def curve(cur, feat, arm):
    """示例单元的一条 τ 族曲线（r = 0），按 τ 排序。"""
    g = cur[(cur.feat == feat) & (cur.arm == arm) & (cur.r == 0)].sort_values('tau')
    assert len(g) == 81, (feat, arm, len(g))
    return g


def t0_point(g):
    p = g[np.isclose(g.tau, 0.0)]
    assert len(p) == 1
    return float(p.OA.iloc[0]), float(p.BA.iloc[0])


def envelope(oa, ba, grid):
    """上包络：每个 OA 格点取覆盖它的相邻 τ 段上插值 BA 的最大值（与登记口径同一算法的局部副本，只用于摆放箭头）。"""
    best = np.full(len(grid), -np.inf)
    for j in range(len(oa) - 1):
        o0, o1, b0, b1 = oa[j], oa[j + 1], ba[j], ba[j + 1]
        ins = (grid >= min(o0, o1) - EPS) & (grid <= max(o0, o1) + EPS)
        if not ins.any():
            continue
        if abs(o1 - o0) < EPS:
            v = np.full(len(grid), max(b0, b1))
        else:
            v = b0 + np.clip((grid - o0) / (o1 - o0), 0.0, 1.0) * (b1 - b0)
        best[ins] = np.maximum(best[ins], v[ins])
    best[np.isneginf(best)] = np.nan
    return best


def arrow_site(g_hi, g_lo, anchor, value):
    """箭头位置：窗口格点中，两条上包络之差最接近本单元登记值的一点（偏好窗口中部）。
    同时返回窗口内逐点差的平均，用来与登记值核对（只打印，不画）。"""
    grid = anchor + OFFS
    e1 = envelope(g_hi.OA.values, g_hi.BA.values, grid)
    e0 = envelope(g_lo.OA.values, g_lo.BA.values, grid)
    ok = np.isfinite(e1) & np.isfinite(e0)
    d = e1 - e0
    cost = np.where(ok, np.abs(d - value) + 0.05 * np.abs(grid - (anchor - 2.0)), np.inf)
    i = int(np.argmin(cost))
    return grid[i], e0[i], e1[i], float(np.mean(d[ok]))


# ================================================================== 文字自动摆放（避开曲线与已放文字）
class Placer:
    """在候选位置中选第一个不与曲线、已放文字重叠且不出轴的位置；都不行就取重叠最少的。"""

    def __init__(self, fig, ax):
        self.fig, self.ax = fig, ax
        self.rend = fig.canvas.get_renderer()
        self.pts = np.empty((0, 2))
        self.boxes = []

    def add_line(self, x, y):
        """把一条折线（数据坐标）加密成显示坐标点，作为障碍；只保留轴内的点。"""
        P = self.ax.transData.transform(np.column_stack([np.asarray(x, float), np.asarray(y, float)]))
        out = [P[:1]]
        for p0, p1 in zip(P[:-1], P[1:]):
            n = max(2, int(np.hypot(*(p1 - p0)) / 0.8) + 2)
            t = np.linspace(0.0, 1.0, n)[1:, None]
            out.append(p0 + t * (p1 - p0))
        P = np.vstack(out)
        bb = self.ax.bbox
        keep = (P[:, 0] >= bb.x0) & (P[:, 0] <= bb.x1) & (P[:, 1] >= bb.y0) & (P[:, 1] <= bb.y1)
        self.pts = np.vstack([self.pts, P[keep]])

    def add_box(self, bb):
        self.boxes.append(bb)

    def score(self, bb, pad_mm=0.5, inside=True):
        p = pad_mm * PX_MM
        x0, x1, y0, y1 = bb.x0 - p, bb.x1 + p, bb.y0 - p, bb.y1 + p
        n = int(np.count_nonzero((self.pts[:, 0] >= x0) & (self.pts[:, 0] <= x1) &
                                 (self.pts[:, 1] >= y0) & (self.pts[:, 1] <= y1)))
        for b in self.boxes:
            if not (x1 < b.x0 or b.x1 < x0 or y1 < b.y0 or b.y1 < y0):
                n += 1000
        if inside:
            ab = self.ax.bbox
            if bb.x0 < ab.x0 + 1 or bb.x1 > ab.x1 - 1 or bb.y0 < ab.y0 + 1 or bb.y1 > ab.y1 - 1:
                n += 5000
        return n

    def leader_hits(self, p_txt, p_tgt, skip_mm=1.8, tol_mm=0.35):
        """引线（文字端 → 目标点）穿过曲线的点数；目标点附近 skip_mm 内不计。"""
        p0, p1 = np.asarray(p_txt, float), np.asarray(p_tgt, float)
        v = p1 - p0
        L = float(np.hypot(*v))
        if L < 1e-6 or not len(self.pts):
            return 0
        t = ((self.pts - p0) @ v) / L ** 2
        proj = p0 + np.clip(t, 0, 1)[:, None] * v
        d = np.hypot(*(self.pts - proj).T)
        m = (d < tol_mm * PX_MM) & (t > 0) & (t * L < L - skip_mm * PX_MM)
        hits = int(np.count_nonzero(m))
        for b in self.boxes:                                   # 引线也不穿过已放文字
            for s in np.linspace(0, 1, 25)[1:-1]:
                q = p0 + s * v
                if b.x0 <= q[0] <= b.x1 and b.y0 <= q[1] <= b.y1:
                    hits += 1000
                    break
        return hits

    def place(self, make, cands, pad_mm=0.5, leader=None):
        """make(cand) 生成一个 artist；按候选顺序打分，留下最好的一个。
        leader(cand, bb) 可选：返回 (文字端, 目标点) 的显示坐标，引线穿过曲线也计分。"""
        best = None
        for c in cands:
            art = make(c)
            bb = art.get_window_extent(self.rend)
            s = self.score(bb, pad_mm)
            if leader is not None:
                lp = leader(c, bb)
                if lp is not None:
                    s += self.leader_hits(*lp)
            if best is None or s < best[0]:
                if best is not None:
                    best[1].remove()
                best = (s, art, bb)
            else:
                art.remove()
            if s == 0:
                break
        self.add_box(best[2])
        return best[1], best[0]


def mm2data(ax, dx_mm, dy_mm):
    """毫米偏移换算成数据单位（线性轴）。"""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bb = ax.get_position()
    return dx_mm * (x1 - x0) / (bb.width * W), dy_mm * (y1 - y0) / (bb.height * H)


# ================================================================== 版式辅助
def ax_mm(fig, l, b, w, h):
    return fig.add_axes([l / W, b / H, w / W, h / H])


def head(ax, letter, left_mm, sub=None):
    """面板标号（毫米定位，a、b、c 同一基线）与紧随其后的灰色小标题。"""
    l, b, w, h = ax.get_position().bounds
    us.panel(ax, letter, x=(left_mm / W - l) / w, y=(HEAD_Y / H - b) / h)
    if sub:
        ax.text((left_mm + 4.2) / W, HEAD_Y / H, sub, transform=ax.figure.transFigure, ha='left', va='bottom',
                fontsize=FS['small'], color=C['MUTED'])


def window(ax, lo, hi):
    """同 OA 窗口：浅灰竖带；文字放在轴顶之上，避开曲线。"""
    ax.axvspan(lo, hi, color=C['BAND'], lw=0, zorder=0)
    ax.text((lo + hi) / 2, ax.get_ylim()[1], 'matched-OA window', ha='center', va='bottom',
            fontsize=FS['small'], color=C['MUTED'], clip_on=False,
            transform=ax.transData)


def draw_curve(ax, pl, cur, feat, arm, color, lw, ls='-'):
    """共享组件画曲线与 τ = 0 实心点；把曲线登记为文字摆放的障碍，返回 τ = 0 点。"""
    g = us.oa_ba_curve(ax, cur, feat, arm, color, ls=ls, lw=lw)
    pl.add_line(g.OA.values, g.BA.values)
    return t0_point(g)


DIRS = ((1, 1), (-1, 1), (1, -1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1))


def radial(ax, xy, radii, dirs=DIRS):
    """以 xy 为中心、按毫米半径与方向生成候选 (x, y, ha, va, xy)。"""
    out = []
    for r in radii:
        for sx, sy in dirs:
            k = 1 / np.sqrt(2) if (sx and sy) else 1.0
            dx, dy = mm2data(ax, sx * r * k, sy * r * k)
            ha = 'left' if sx > 0 else ('right' if sx < 0 else 'center')
            va = 'bottom' if sy > 0 else ('top' if sy < 0 else 'center')
            out.append((xy[0] + dx, xy[1] + dy, ha, va, xy))
    return out


def anchor_px(ax, c):
    return ax.transData.transform((c[0], c[1]))


def gap_arrow(ax, pl, x, y0, y1, value, color):
    """两条上包络之间的双向箭头 + 本单元登记值。间距放不下两个箭头头加一段箭杆时画无头短竖线。
    文字先试紧贴箭头的位置；都被曲线挡住时退到稍远处并加引线。"""
    gap_mm = abs(y1 - y0) / mm2data(ax, 0.0, 1.0)[1]
    an = ax.annotate('', xy=(x, y1), xytext=(x, y0),
                     arrowprops=dict(arrowstyle='<|-|>' if gap_mm >= ARROW_MIN_MM else '-',
                                     lw=ARROW_LW if gap_mm >= ARROW_MIN_MM else 1.4,
                                     color=color, shrinkA=0, shrinkB=0, mutation_scale=ARROW_MS), zorder=8)
    an.arrow_patch.set_path_effects([pe.Stroke(linewidth=1.8, foreground='white'), pe.Normal()])
    pl.add_line([x, x], [y0, y1])
    ylo, yhi, ym = min(y0, y1), max(y0, y1), (y0 + y1) / 2
    dx, dy = mm2data(ax, 1.0, 0.8)
    near = [(x + dx, ym, 'left', 'center', None), (x - dx, ym, 'right', 'center', None),
            (x, yhi + dy, 'center', 'bottom', None), (x, ylo - dy, 'center', 'top', None),
            (x + dx, yhi + dy, 'left', 'bottom', None), (x - dx, yhi + dy, 'right', 'bottom', None),
            (x + dx, ylo - dy, 'left', 'top', None), (x - dx, ylo - dy, 'right', 'top', None)]
    far = radial(ax, (x, ym), (4.0, 5.5, 7.0, 9.0, 11.0))
    s = fmt(value) + ' pp'

    def mk(c):
        kw = dict(ha=c[2], va=c[3], fontsize=FS['tick'], color=color, fontweight='bold', zorder=7)
        if c[4] is None:
            return ax.text(c[0], c[1], s, **kw)
        return ax.annotate(s, xy=c[4], xytext=(c[0], c[1]), arrowprops=dict(
            arrowstyle='-', lw=0.4, color=color, shrinkA=0.5, shrinkB=1.0), **kw)
    pl.place(mk, near + far, pad_mm=0.4,
             leader=lambda c, bb: None if c[4] is None else (anchor_px(ax, c), ax.transData.transform(c[4])))
    return gap_mm >= ARROW_MIN_MM


def t0_label(ax, pl, pts):
    """标出一个 τ = 0 工作点（短引线）：按偏好顺序尝试各条曲线的 τ = 0 点与不同方向、距离。"""
    cands = []
    for r in (3.0, 4.5, 6.0, 8.0):
        for xy in pts:
            cands += radial(ax, xy, (r,))
    pl.place(lambda c: ax.annotate('τ = 0', xy=c[4], xytext=(c[0], c[1]), ha=c[2], va=c[3],
                                   fontsize=FS['small'], color=C['TEXT'], zorder=7,
                                   arrowprops=dict(arrowstyle='-', lw=0.4, color=C['MUTED'], shrinkA=0.5,
                                                   shrinkB=2.0)),
             cands, pad_mm=0.4, leader=lambda c, bb: (anchor_px(ax, c), ax.transData.transform(c[4])))


CORNERS = [((0.97, 0.03), 'right', 'bottom'), ((0.03, 0.97), 'left', 'top'), ((0.03, 0.03), 'left', 'bottom'),
           ((0.97, 0.97), 'right', 'top'), ((0.97, 0.5), 'right', 'center'), ((0.03, 0.5), 'left', 'center'),
           ((0.5, 0.03), 'center', 'bottom'), ((0.5, 0.97), 'center', 'top')]


def t0_readout(ax, pl, d_oa, d_ba, color):
    """τ = 0 读数（两点之差），放在不与曲线重叠的角落。"""
    s = f'at τ = 0:\nΔOA {fmt(d_oa)} pp\nΔBA {fmt(d_ba)} pp'
    pl.place(lambda c: ax.text(*c[0], s, transform=ax.transAxes, ha=c[1], va=c[2], fontsize=FS['small'],
                               color=color, linespacing=1.2, zorder=7), CORNERS)


def legend(ax, pl, handles, labels):
    """曲线图例放在轴内不与曲线重叠的角落（三个任务同一规则）。"""
    locs = ['lower left', 'upper left', 'lower right', 'upper right', 'center left', 'center right']

    def mk(loc):
        # 用 Legend 对象直接加到轴上（ax.legend 会替换已有图例，不能逐个试位置）
        leg = Legend(ax, handles, labels, loc=loc, fontsize=FS['small'], handlelength=1.8, handletextpad=0.5,
                     labelspacing=0.35, borderaxespad=0.4, borderpad=0.2, frameon=False)
        ax.add_artist(leg)
        return leg
    leg, s = pl.place(mk, locs, pad_mm=0.8)
    return leg


def frame(ax, xl, yl):
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    ax.set_xlabel('OA (%)', labelpad=1.5)


# ================================================================== c｜两种读法对比（us.forest，两组上下两个轴）
ROWS_C = [
    (f"{us.TERM['rk'].format(k=200)} vs {us.TERM['zero']}", 'LABEL', [
        ('t0_RT200-Z0_BA', 'both', 'ΔBA at τ = 0', False),
        ('t0_RT200-Z0_OA', 'both', 'ΔOA at τ = 0', False),
        ('dFA_RT200', 'both', 'ΔBA at matched OA', True)]),
    (us.TERM['ae0'], 'AE', [
        ('t0_Z0_AE-S2_BA', 'ae-s2', 'ΔBA at τ = 0', False),
        ('t0_Z0_AE-S2_OA', 'ae-s2', 'ΔOA at τ = 0', False),
        ('D_Z0', 'ae-s2', 'ΔBA at matched OA', True)]),
]
CL, CW = 131.0, 24.0                         # c 轴左缘与宽度（mm）
C_ROW_H = 15.0                               # 每组 3 行的轴高（mm）
C_AX_BOT = (BOT, BOT + C_ROW_H + 6.0)        # 下组、上组的轴底
X_COL = (158.6, 178.6)                       # forest 右侧两列（均值、> 0）的左端（mm）；离轴右缘 3.6 mm
COL_X = tuple((x - CL) / CW for x in X_COL)  # 换算成轴坐标，传给 us.forest(col_x=...)
X_HEAD = 108.0                               # 组标题左端（mm）
TITLE_GAP = 0.6                              # 组标题底边与其下方最高文字（上组为角注，下组为列头）的间距（mm）
MS_MAIN = 4.6                                # matched-OA 行的均值点（统一规则：行级 ms 放大）


def panel_c(fig, task):
    built = []
    for title, ckey, items in ROWS_C:
        rows = [us.row(task, name, feat, lab, C[ckey], filled=main, ms=MS_MAIN if main else None)
                for name, feat, lab, main in items]
        built.append((title, ckey, items, rows))
    # 共同 x 范围：n ≥ 5 用 CI 端点，否则用各域的值；含 0
    vals = [0.0]
    for _, _, _, rows in built:
        for r in rows:
            e = r['e']
            vals.append(e['mean'])
            if e['n'] >= us.CI_MIN_N and np.isfinite(e['lo95']):
                vals += [e['lo95'], e['hi95']]
            else:
                vals += [v for v in r['dv'] if np.isfinite(v)]
    lo, hi = min(vals), max(vals)
    pad = 0.07 * (hi - lo)
    xl = (lo - pad, hi + pad)

    axes = []
    for gi, (title, ckey, items, rows) in enumerate(built):
        ax = ax_mm(fig, CL, C_AX_BOT[1 - gi], CW, C_ROW_H)
        # 任务层面板：作物的 RO、PT 不进均值，也不在此画（其 τ = 0 读数远超主集范围，会压扁区间）
        us.forest(ax, rows, level=95, xlabel='Difference (pp)', col_x=COL_X, show_others=False)
        ax.set_xlim(*xl)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10], integer=True))
        for t, (_, _, _, main) in zip(ax.get_yticklabels(), items):
            t.set_fontweight('bold' if main else 'normal')
        if gi == 0:                              # 上组：不画 x 轴；角注只写一次
            ax.set_xlabel('')
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', bottom=False, labelbottom=False)
            us.forest_note(ax, task, 95)
        else:
            ax.xaxis.labelpad = 1.5
        # 组标题：放在该轴上方已有文字（列头基线 = 轴顶；上组另有角注在列头上方一行）的顶边之上
        rend = fig.canvas.get_renderer()
        ytop = max(t.get_window_extent(rend).y1 for t in ax.texts) / PX_MM + TITLE_GAP
        fig.text(X_HEAD / W, ytop / H, title, ha='left', va='bottom', fontsize=FS['tick'], color=C[ckey],
                 fontweight='bold')
        axes.append(ax)
    return axes, built


# ================================================================== 一个任务一张图
def make(task):
    mj = us.meta_json()[task]
    M = us.META[task]
    unit = mj['example_unit']
    cur = us.read(task, 'example_curves')
    cur = cur[cur.unit == unit]
    ur = us.read(task, 'unit_rows')
    arm = us.read(task, 'arm_rows')
    domain = ur[ur.unit == unit].domain.iloc[0]

    g = {(f, a): curve(cur, f, a) for f in ('ae', 's2') for a in ('Z0', 'RT200', 'RT400', 'REF')}
    star_ae = t0_point(g[('ae', 'Z0')])[0]
    star_s2 = t0_point(g[('s2', 'Z0')])[0]
    # 与 arm_rows 的 OA★ 对账
    for f, s in (('ae', star_ae), ('s2', star_s2)):
        a = arm[(arm.unit == unit) & (arm.feat == f) & (arm.arm == 'Z0')].OA_star
        assert len(a) == 1 and abs(float(a.iloc[0]) - s) < 1e-9, (task, f)
    anchor_b = min(star_ae, star_s2)

    # 本单元的登记值（unit_rows）
    d_rt200 = unit_val(ur, unit, 'dFA_RT200', 'ae', 0)
    d_z0 = unit_val(ur, unit, 'D_Z0', 'ae-s2', 0)
    t0a_ba = unit_val(ur, unit, 't0_RT200-Z0_BA', 'ae', 0)
    t0a_oa = unit_val(ur, unit, 't0_RT200-Z0_OA', 'ae', 0)
    t0b_ba = unit_val(ur, unit, 't0_Z0_AE-S2_BA', 'ae-s2', 0)
    t0b_oa = unit_val(ur, unit, 't0_Z0_AE-S2_OA', 'ae-s2', 0)
    # τ = 0 读数与曲线上的 τ = 0 点对账
    (oz, bz), (orr, brr), (os2, bs2) = t0_point(g[('ae', 'Z0')]), t0_point(g[('ae', 'RT200')]), \
        t0_point(g[('s2', 'Z0')])
    assert abs((orr - oz) - t0a_oa) < 1e-9 and abs((brr - bz) - t0a_ba) < 1e-9, task
    assert abs((oz - os2) - t0b_oa) < 1e-9 and abs((bz - bs2) - t0b_ba) < 1e-9, task

    # a、b 共用坐标范围：OA 跨度固定 22 pp，右端 = 曲线最大 OA + 1 取整；BA 范围按可见点自定
    shown = [g[k] for k in (('ae', 'Z0'), ('ae', 'RT200'), ('ae', 'RT400'), ('ae', 'REF'), ('s2', 'Z0'))]
    xhi = float(np.ceil(max(s.OA.max() for s in shown) + 1.0))
    xl = (xhi - X_SPAN, xhi)
    vis = np.concatenate([s.BA[(s.OA >= xl[0]) & (s.OA <= xl[1])].values for s in shown])
    span = vis.max() - vis.min()
    yl = (vis.min() - 0.04 * span, vis.max() + 0.07 * span)

    fig = plt.figure(figsize=(W * MM, H * MM))
    desc = us.dname(task, domain, unit)          # 示例单元显示名（作物写 "Belgium (one spatial half)"）
    LS_R4 = (0, (3, 1.6))                        # +400 随机标签：虚线

    # ---------------------------------------------------------------- a｜标签怎样移动前沿
    axa = ax_mm(fig, 10.5, BOT, 44.5, AH)
    frame(axa, xl, yl)
    axa.set_ylabel('BA (%)', labelpad=1.5)
    window(axa, star_ae - 4, star_ae)
    pa = Placer(fig, axa)
    xy_ref = draw_curve(axa, pa, cur, 'ae', 'REF', us.REF_COLOR, LW_REF, ls=us.REF_LS)
    xy_z0 = draw_curve(axa, pa, cur, 'ae', 'Z0', C['REF_Z0'], 1.1)
    xy_r4 = draw_curve(axa, pa, cur, 'ae', 'RT400', C['LABEL'], 1.0, ls=LS_R4)
    xy_rt = draw_curve(axa, pa, cur, 'ae', 'RT200', C['LABEL'], 1.2)
    hd = [Line2D([], [], color=C['REF_Z0'], lw=1.1), Line2D([], [], color=C['LABEL'], lw=1.2),
          Line2D([], [], color=C['LABEL'], lw=1.0, ls=LS_R4),
          Line2D([], [], color=us.REF_COLOR, lw=LW_REF, ls=us.REF_LS)]
    legend(axa, pa, hd, [us.TERM['zero'], us.TERM['rk'].format(k=200), us.TERM['rk'].format(k=400),
                         M['ref_short']])
    xa, y0, y1, chk_a = arrow_site(g[('ae', 'RT200')], g[('ae', 'Z0')], star_ae, d_rt200)
    t0_readout(axa, pa, t0a_oa, t0a_ba, C['LABEL'])
    heads = {'a': gap_arrow(axa, pa, xa, y0, y1, d_rt200, C['LABEL'])}
    t0_label(axa, pa, [xy_rt, xy_z0, xy_r4, xy_ref])
    head(axa, 'a', 0.0, f'Example target: {desc} · AE features')

    # ---------------------------------------------------------------- b｜表征怎样移动前沿
    axb = ax_mm(fig, 62.0, BOT, 39.0, AH)
    frame(axb, xl, yl)
    axb.tick_params(labelleft=False)
    window(axb, anchor_b - 4, anchor_b)
    pb = Placer(fig, axb)
    xy_s2 = draw_curve(axb, pb, cur, 's2', 'Z0', C['S2'], 1.0)
    xy_ae = draw_curve(axb, pb, cur, 'ae', 'Z0', C['AE'], 1.2)
    legend(axb, pb, [Line2D([], [], color=C['AE'], lw=1.2), Line2D([], [], color=C['S2'], lw=1.0)],
           ['AE embeddings', 'S2 statistics'])
    xb, y0b, y1b, chk_b = arrow_site(g[('ae', 'Z0')], g[('s2', 'Z0')], anchor_b, d_z0)
    t0_readout(axb, pb, t0b_oa, t0b_ba, C['AE'])
    heads['b'] = gap_arrow(axb, pb, xb, y0b, y1b, d_z0, C['AE'])
    t0_label(axb, pb, [xy_ae, xy_s2])
    head(axb, 'b', 58.0, 'Same unit · zero-label transfer')

    # ---------------------------------------------------------------- c｜两种读法对比
    axc, built = panel_c(fig, task)
    n = int(built[0][3][0]['e']['n'])
    n_all = len(mj['all'])
    sub_c = (f"Task level · {us.TERM['primary']}: {n} of {n_all} target {M['domains']}" if n < n_all
             else f"Task level · {n} target {M['domains']}")
    head(axc[0], 'c', 104.0, sub_c)

    name = f'uf2_evaluation_{task}'
    us.save(fig, name)
    n_draws = ur[(ur.unit == unit) & (ur.name == 'dFA_RT200')].r.nunique()
    us.write_legend(name, legend_text(task, M, desc, star_ae, anchor_b, n, n_draws, heads))
    print(f'{task}: unit {unit}  OA*ae {star_ae:.2f} OA*s2 {star_s2:.2f}  '
          f'dFA_RT200 {d_rt200:+.4f} (local envelope {chk_a:+.4f})  D_Z0 {d_z0:+.4f} (local {chk_b:+.4f})  '
          f'xl {xl} yl ({yl[0]:.1f}, {yl[1]:.1f})')
    return built


# ================================================================== 图例文字（英文，投稿用）
NUM_WORD = {2: 'two', 3: 'three', 4: 'four', 5: 'five'}


def pool_phrase(M):
    """源池写法（统一规则：图例写 "source pool"；作物写 "foreign source pool"；不嵌套括号）。"""
    kind, body = M['pool'].split(': ', 1)
    body = body.replace(' (', ', ').replace(')', '')
    word = ('foreign ' if kind.lower().startswith('foreign') else '') + 'source pool'
    return word, f'{word} ({body})'


def legend_text(task, M, desc, star_ae, anchor_b, n, n_draws, heads):
    D, T = us.DESIGN, us.TERM
    doms = M['domains']
    pool_word, pool_full = pool_phrase(M)
    n_all = len(us.all_domains(task))
    if n >= us.CI_MIN_N:
        ci_rule = (f'Points are means over the n = {n} target {doms} of the {T["primary"]}; bars are 95% t confidence '
                   f'intervals (df = {n - 1}).')
    else:
        ci_rule = (f'Means over the n = {n} target {doms} are the larger circles; because n < {us.CI_MIN_N}, no interval is '
                   f'drawn: small open circles, drawn over the mean, show the single target {doms} and the thin line '
                   f'spans their range.')
    oth = us.others(task)
    prim_note = ''
    if oth:
        names = ' and '.join(f'{us.dname(task, d)} ({d})' for d in oth)
        prim_note = (f' The {T["primary"]} comprises {n} of the {n_all} target {doms}; {names} are outside it and '
                     f'are not shown in c.')
    if M['ref'].startswith('target-only'):
        ref_desc = f"{M['ref']}, trained on the target {M['domain']} alone without the source pool"
    else:
        ref_desc = f"the source pool plus {M['ref']} of the target {M['domain']}"
    hdr = 'mean [95% CI]' if n >= us.CI_MIN_N else 'mean'
    draws = NUM_WORD.get(n_draws, str(n_draws))
    rk200 = T['rk'].format(k=200)
    bar = [p for p in ('a', 'b') if not heads[p]]          # 间距太小、画成无头短竖线的面板
    arrow = ('The double arrow' if not bar else
             f"The double arrow (a short plain bar in {' and '.join(bar)}, where the gap is too small for arrowheads)")
    return f"""
{M['long']} — evaluating label and representation effects at matched overall accuracy.

LightGBM models are trained on the {pool_full}, alone ({T['zero']}) or with k random target labels (labelled target {M['samples_short']}, sample weight {D['target_weight']}). Features are {D['ae_desc']} or {D['s2_desc']}, with {M['support']}. {M['test']}. Operating points (a, b): the {D['tau']} shifts each model's class scores; each τ gives one (OA, BA) point and points are joined in τ order. Filled dots mark τ = 0 (the unadjusted model); curves may leave the plotted range. Grey band, the {D['window']} with the same features (for b, see below). "ΔBA at matched OA" is the mean, over the window, of the pointwise difference between the upper BA envelopes of two curves. {arrow} is placed at one window point for orientation only; the printed value is this unit's window mean.

a, Example target unit: {desc}; AE features. Dark grey, {T['zero']} from the {pool_word}; blue, the same pool plus 200 (solid) or 400 (dashed) random target labels (first of {draws} random draws); dotted black line, reference arm: {ref_desc}. The printed value is this unit's ΔBA at matched OA for {rk200} (OA★ = {star_ae:.2f}%). "at τ = 0" gives the differences between the τ = 0 points of the two models ({rk200} minus {T['zero']}; ΔOA and ΔBA).
b, Same unit, {T['zero']} with AE (teal) or S2 (grey) features. The window is anchored at min(OA★_AE, OA★_S2) = {anchor_b:.2f}%. The printed value is this unit's AE − S2 difference in BA at matched OA; "at τ = 0" gives the AE − S2 differences between the two τ = 0 points.
c, Both readings at task level. Top (blue), {rk200} vs {T['zero']}; {M['domain']} values average the AE and S2 runs. Bottom (teal), {T['ae0']}. Open circles, readings at the fixed operating point τ = 0 (ΔBA and ΔOA); large filled circles, ΔBA at matched OA. {ci_rule}{prim_note} Right-hand columns: {hdr}, and the number of target {doms} with a difference > 0 out of n. Each {M['domain']} value averages {M['units_note']} (draws apply to the label arm).

OA, overall accuracy; BA, balanced accuracy (mean per-class recall over the unit's BA classes); differences in percentage points (pp).
"""


if __name__ == '__main__':
    for t in us.TASKS:
        make(t)
