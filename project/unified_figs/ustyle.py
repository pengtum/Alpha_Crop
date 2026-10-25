"""三任务统一配图的共享模块：样式沿用 RSE 的 nstyle（字体、字号、颜色语义、导出），并提供
任务元数据、源数据读取与通用绘图组件。所有统一图只 import 本模块，保证三个任务的版式与风格一致。

源数据：unified_figs/source_data/（prep_unified.py 生成，已与登记结果对账）。
输出：unified_figs/out/{图名}_{任务}.{svg,pdf,tiff,png}；跨任务图为 {图名}.{…}。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'label_expansion', 'storyline', 'rse_figs'))
import nstyle as ns  # noqa: E402  只 import，不修改

C, FS, W1, W15, W2, MM = ns.C, ns.FS, ns.W1, ns.W15, ns.W2, ns.MM
panel, zero_line, fmt_pp = ns.panel, ns.zero_line, ns.fmt_pp
OUT = os.path.join(HERE, 'out')
SRC = os.path.join(HERE, 'source_data')
TASKS = ('crop', 'lcz', 'lucas')
KS = (50, 100, 200, 400)
REF_COLOR = C['REF_TGT']
CI_MIN_N = 5        # 统一规则：域数 ≥ 5 才画 t 置信区间；否则直接画各域的值（df 太小时区间无信息）

# ---------------------------------------------------------------- 任务元数据（图中文字一律英文）
META = {
    'crop': dict(
        title='Crop type', long='Crop-type mapping (EuroCrops, 2021)', domain='country', domains='countries',
        sample='parcel', samples='parcels', n_classes=10, year=2021, primary_note='9 countries (RO, PT shown open)',
        pool='Foreign source pool: other 10 countries × 5,000 parcels',
        test='Test: 10-km cells of the target country, 1-km buffer',
        ref='target-only model (~30k labels)', ref_short='target-only (~30k)',
        units_note='two spatial halves × 3 random draws per country'),
    'lcz': dict(
        title='Urban form (LCZ)', long='Local climate zones (So2Sat LCZ42, 2017)', domain='city', domains='cities',
        sample='patch', samples='320-m patches', n_classes=17, year=2017, primary_note='3 target cities',
        pool='Source pool: other 29 cities (28,884 patches)',
        test='Test: 2-km cells of the target city, 500-m buffer',
        ref='all candidate labels (n = 508–713)', ref_short='all candidates (≤713)',
        units_note='3 random draws per city'),
    'lucas': dict(
        title='Land cover (LUCAS)', long='Land cover (LUCAS Copernicus survey, 2018)', domain='country',
        domains='countries', sample='polygon', samples='survey polygons', n_classes=7, year=2018,
        primary_note='3 target countries',
        pool='Source pool: other 25 countries (53,662 polygons)',
        test='Test: 10-km cells of the target country, 1-km buffer',
        ref='all candidate labels (n = 1,826–2,503)', ref_short='all candidates (≤2,503)',
        units_note='3 random draws per country'),
}
DOMAIN_LABEL = {'amsterdam': 'Amsterdam', 'qingdao': 'Qingdao', 'melbourne': 'Melbourne',
                'IT': 'IT', 'UK': 'UK', 'SE': 'SE'}
for _t, _short, _support in (('crop', 'parcels', 'features averaged within a 30-m radius of the parcel centroid'),
                             ('lcz', 'patches', 'features averaged over the 320-m patch'),
                             ('lucas', 'polygons', 'features averaged over the survey polygon')):
    META[_t]['samples_short'] = _short
    META[_t]['support'] = _support
# 图例用的设计常量（避免各脚本硬编码，措辞一致）
DESIGN = dict(target_weight=50, ae_dim=64, s2_n=84,
              s2_desc='Sentinel-2 statistics (S2; 84 annual and quarterly band and index statistics)',
              ae_desc='AlphaEarth embeddings (AE; 64 dimensions)',
              tau='τ-family ℓ = log P − τ·log π̂, τ = −2…2 in steps of 0.05 (π̂ = mean predicted class share on the test set)',
              window='matched-OA window [OA★ − 4, OA★] in steps of 0.1, OA★ = OA of zero-label transfer at τ = 0')
# 统一术语（图中文字）
TERM = dict(rk='+{k} random labels', ae0='AE instead of S2 (0 labels)', primary='primary set',
            zero='zero-label transfer', axis_k='Random target labels (k)', dba='ΔBA at matched OA (pp)',
            inter='AE − S2 difference in label gain (pp)', d_ba='AE − S2 BA at matched OA (pp)')
# 剂量面板统一的 x 范围与 AE 逐域刻度位置（UF0、UF3、UF4 共用）
DOSE_XLIM = (-14, 440)
DOSE_XDOM = 425
# OA–BA 曲线：参照臂 REF 用点线，以与零标签迁移（实线深灰）区分
REF_LS = (0, (1, 1.2))


def setup():
    ns.setup()


def save(fig, name):
    return ns.save(fig, name, out_dir=OUT)


def src(name):
    return os.path.join(SRC, name)


def read(task, what):
    """what ∈ estimates、domain_values、unit_rows、arm_rows、class_rows、class_summary、size_rows、size_summary、
    example_curves、example_layout、example_test_cells、domains。"""
    return pd.read_csv(src(f'{task}_{what}.csv'))


def meta_json():
    with open(src('tasks_meta.json')) as fh:
        return {m['task']: m for m in json.load(fh)}


def dname(task, d, unit=None):
    """域的显示名（全文统一）：作物、LUCAS 用国家全名，LCZ 用城市名；作物示例单元加 "(one spatial half)"。"""
    if task == 'lcz':
        n = dlabel(d)
    else:
        t = read(task, 'domains').set_index('domain')
        n = t.loc[d, 'name'] if d in t.index else d
    return n + (' (one spatial half)' if unit and unit != d else '')


def others(task):
    """非主集域（作物的 RO、PT），按 RSE 的顺序。"""
    o = [d for d in all_domains(task) if d not in primary(task)]
    return [d for d in ('RO', 'PT') if d in o] + [d for d in o if d not in ('RO', 'PT')]


def primary(task):
    return meta_json()[task]['primary']


def all_domains(task):
    return meta_json()[task]['all']


def dlabel(d):
    return DOMAIN_LABEL.get(d, d)


def est(task, name, feat, set_='primary'):
    """任务层估计：dict(mean, lo95, hi95, lo90, hi90, n, n_pos)。"""
    e = read(task, 'estimates')
    r = e[(e.name == name) & (e.feat == feat) & (e.set == set_)]
    assert len(r) == 1, (task, name, feat, set_)
    return r.iloc[0].to_dict()


def dom(task, name, feat):
    """域值 Series（index = 域）。"""
    d = read(task, 'domain_values')
    return d[(d.name == name) & (d.feat == feat)].set_index('domain').value


def ci_text(e, level=95, nd=2):
    lo, hi = (e['lo95'], e['hi95']) if level == 95 else (e['lo90'], e['hi90'])
    cnt = f"{int(e['n_pos'])}/{int(e['n'])} > 0"
    if e['n'] < CI_MIN_N or not np.isfinite(lo):
        return f"{fmt_pp(e['mean'], nd=nd)}, {cnt}"
    return f"{fmt_pp(e['mean'], nd=nd)} [{fmt_pp(lo, signed=False, nd=nd)}, {fmt_pp(hi, signed=False, nd=nd)}], {cnt}"


# ---------------------------------------------------------------- 通用绘图组件
def row(task, name, feat, label, color, set_='primary', **kw):
    """forest 的一行：任务层估计 + 主集各域的值（域数 < CI_MIN_N 时画各域点）。"""
    v = dom(task, name, feat)
    return dict(label=label, e=est(task, name, feat, set_), dv=v.reindex(primary(task)).values, color=color,
                others=v.reindex(others(task)).dropna(), **kw)


def forest(ax, rows, level=95, xlabel='ΔBA at matched OA (pp)', ms=3.2, cols=True, col_x=(1.04, 1.30),
           show_others=True):
    """横向点图（全文统一的森林图）。rows：list of dict(label, e, dv, color, filled=True, grey=False, ms=None,
    others=Series)，自上而下排列。域数 ≥ CI_MIN_N：均值 + t CI（level）；否则均值 + 各域小空心点（画在均值之上）。
    cols=True 时在轴右侧写两列：均值（n ≥ 5 时附 CI）与 "> 0" 计数，列头在顶行之上。非主集域（作物 RO、PT）
    画灰色空心圈并标代码。返回 y 位置。"""
    ys = np.arange(len(rows))[::-1]
    for y, r in zip(ys, rows):
        if show_others and r.get('others') is not None and len(r['others']):
            for d, v in r['others'].items():
                ax.plot(v, y, 'o', ms=ms * 0.8, color=C['MUTED'], mfc='white', mew=0.6, zorder=2)
                ax.text(v, y + 0.28, d, fontsize=FS['small'], color=C['MUTED'], ha='center', va='bottom')
        e, col = r['e'], (C['MUTED'] if r.get('grey') else r['color'])
        lo, hi = (e['lo95'], e['hi95']) if level == 95 else (e['lo90'], e['hi90'])
        if e['n'] >= CI_MIN_N and np.isfinite(lo):
            ax.plot([lo, hi], [y, y], color=col, lw=0.9, solid_capstyle='butt')
        elif r.get('dv') is not None:
            dv = np.asarray(r['dv'], float)
            dv = dv[np.isfinite(dv)]
            if len(dv):
                ax.plot([dv.min(), dv.max()], [y, y], color=col, lw=0.5, alpha=0.5)
                ax.plot(dv, np.full(len(dv), y), 'o', ms=ms * 0.6, color=col, mfc='white', mew=0.6, zorder=4)
        m_ = r.get('ms') or ms
        ax.plot(e['mean'], y, 'o', ms=m_, color=col, mfc=col if r.get('filled', True) else 'white', mew=0.8, zorder=3)
        if cols:
            lo_, hi_ = (e['lo95'], e['hi95']) if level == 95 else (e['lo90'], e['hi90'])
            txt = fmt_pp(e['mean'])
            if e['n'] >= CI_MIN_N and np.isfinite(lo_):
                txt += f" [{fmt_pp(lo_, signed=False)}, {fmt_pp(hi_, signed=False)}]"
            tr = ax.get_yaxis_transform()
            ax.text(col_x[0], y, txt, transform=tr, fontsize=FS['small'], color=col, va='center', ha='left',
                    clip_on=False)
            ax.text(col_x[1], y, f"{int(e['n_pos'])}/{int(e['n'])}", transform=tr, fontsize=FS['small'],
                    color=C['MUTED'], va='center', ha='left', clip_on=False)
    ax.set_yticks(ys)
    ax.set_yticklabels([r['label'] for r in rows])
    for t, r in zip(ax.get_yticklabels(), rows):
        if r.get('grey'):
            t.set_color(C['MUTED'])
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    zero_line(ax, axis='x')
    ax.set_xlabel(xlabel)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    if cols:
        import matplotlib.transforms as mtrans
        tr = mtrans.blended_transform_factory(ax.transAxes, ax.transAxes)    # 列头基线 = 轴顶
        hdr = 'mean' + (f' [{level}% CI]' if rows and rows[0]['e']['n'] >= CI_MIN_N else '')
        ax.text(col_x[0], 1.0, hdr, transform=tr, fontsize=FS['small'], color=C['MUTED'], ha='left',
                va='bottom', clip_on=False)
        ax.text(col_x[1], 1.0, '> 0', transform=tr, fontsize=FS['small'], color=C['MUTED'], ha='left',
                va='bottom', clip_on=False)
    return ys


def forest_note(ax, task, level=95):
    """森林图右上角的统一角注："9 countries; mean and 95% CI" 或 "3 cities; mean and single cities"。"""
    n = len(primary(task))
    dm = META[task]['domains']
    txt = f'{n} {dm}; mean and {level}% CI' if n >= CI_MIN_N else f'{n} {dm}; mean and single {dm}'
    ax.annotate(txt, (1.0, 1.0), xycoords='axes fraction', xytext=(0, 8), textcoords='offset points',
                fontsize=FS['small'], color=C['MUTED'], ha='right', va='bottom', annotation_clip=False)


def _end_label(ax, x, y, txt, color):
    """线末端标签：固定 4 pt 右移（与轴宽无关），白底遮断零线等水平参考线，不盖均值线与点（zorder 2.5）。"""
    ax.annotate(txt, (x, y), xytext=(4, 0), textcoords='offset points', fontsize=FS['small'], color=color,
                va='center', ha='left', zorder=2.5, annotation_clip=False,
                bbox=dict(fc='white', ec='none', pad=0.4))


def dose(ax, task, feat='both', color=None, show_domains=True, label=None, offset=0.0, ci=True):
    """剂量曲线：x = 0、50、100、200、400；主集域细线（非主集灰色虚线），均值粗线 + 95% CI。"""
    color = color or C['LABEL']
    ks = np.array((0,) + KS, float)
    prim = primary(task)
    if show_domains:
        for d in all_domains(task):
            v = [0.0] + [dom(task, f'dFA_RT{k}', feat).get(d, np.nan) for k in KS]
            if d in prim:
                ax.plot(ks, v, color=color, lw=0.6, alpha=0.3, zorder=1)
                if len(prim) < CI_MIN_N and np.isfinite(v[-1]):        # 域数少：线末端标域名（与 RO、PT 一致）
                    _end_label(ax, ks[-1], v[-1], dlabel(d), color)
            else:
                ax.plot(ks, v, color=C['MUTED'], lw=0.6, ls=(0, (2, 1.5)), zorder=1)
                _end_label(ax, ks[-1], v[-1], d, C['MUTED'])
    m = [0.0] + [est(task, f'dFA_RT{k}', feat)['mean'] for k in KS]
    lo = [0.0] + [est(task, f'dFA_RT{k}', feat)['lo95'] for k in KS]
    hi = [0.0] + [est(task, f'dFA_RT{k}', feat)['hi95'] for k in KS]
    ax.plot(ks + offset, m, color=color, lw=1.4, marker='o', ms=3, zorder=3, label=label)
    n = est(task, 'dFA_RT50', feat)['n']
    if ci and n >= CI_MIN_N:
        for x, a, b in zip(ks[1:] + offset, lo[1:], hi[1:]):
            if np.isfinite(a):
                ax.plot([x, x], [a, b], color=color, lw=0.8, zorder=2)
    ax.set_xticks(ks)
    ax.set_xlabel('Random target labels (k)')
    return m


def ae_band(ax, task, xmax=DOSE_XLIM[1], label=True, x_dom=DOSE_XDOM):
    """AE − S2（零标签）的均值线；n ≥ 5 画 95% CI 浅带，否则在 x_dom 处逐域短刻度并标域名（k = 400 右侧）。"""
    e = est(task, 'D_Z0', 'ae-s2')
    if e['n'] >= CI_MIN_N and np.isfinite(e['lo95']):
        ax.axhspan(e['lo95'], e['hi95'], color=C['AE'], alpha=0.10, lw=0, zorder=0)
    # 域数少（n < CI_MIN_N）：只画均值线；各域的 AE − S2 见同图的森林图，不在剂量面板上逐域标注，
    # 以免与单域剂量线末端的域名混淆
    ax.axhline(e['mean'], color=C['AE'], lw=1.1, ls=(0, (4, 2)), zorder=2)
    if label:
        ax.annotate(f"AE instead of S2\n(0 labels) {fmt_pp(e['mean'])}", (xmax, e['mean']), xytext=(0, 1.5),
                    textcoords='offset points', color=C['AE'], fontsize=FS['small'], ha='right', va='bottom')
    return e


def oa_ba_curve(ax, curves, feat, arm, color, ls='-', lw=1.0, label=None, t0=True):
    g = curves[(curves.feat == feat) & (curves.arm == arm) & (curves.r == 0)].sort_values('tau')
    ax.plot(g.OA, g.BA, color=color, ls=ls, lw=lw, label=label)
    if t0:
        p = g[np.isclose(g.tau, 0.0)].iloc[0]
        ax.plot(p.OA, p.BA, 'o', ms=3, color=color, zorder=4)
    return g


def domain_dots(ax, task, name, feat, color, order=None, filled=True, ms=3.2, grey_markers=True):
    """逐域点图（横向），非主集域行标签灰色。返回排序后的域列表。"""
    v = dom(task, name, feat).reindex(all_domains(task))
    order = order or list(v.sort_values().index)
    prim = primary(task)
    ys = np.arange(len(order))
    for y, d in zip(ys, order):
        c = color if (d in prim or not grey_markers) else C['MUTED']
        ax.plot(v[d], y, 'o', ms=ms, color=c, mfc=c if (filled and (d in prim or not grey_markers)) else 'white', mew=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([dlabel(d) for d in order])
    for t, d in zip(ax.get_yticklabels(), order):
        if d not in prim:
            t.set_color(C['MUTED'])
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    zero_line(ax, axis='x')
    return order


def write_legend(name, text):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f'{name}_legend.txt'), 'w', encoding='utf-8') as fh:
        fh.write(text.strip() + '\n')
