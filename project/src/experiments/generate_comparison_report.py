#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 有坐标+面积 vs 纯光谱特征 的消融对比 HTML 报告。

读取 results/experiments/all_results_combined.json，
输出 results/experiments/comparison_report.html（自包含、中文）。
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

# ────────────────────────────────────────────────
# 常量
# ────────────────────────────────────────────────
COUNTRIES = [
    "AT", "BE", "CZ", "DE", "DK", "EE", "ES", "FR",
    "HR", "IE", "LT", "LV", "NL", "PT", "RO", "SE", "SI", "SK",
]

COUNTRY_NAMES = {
    "AT": "奥地利", "BE": "比利时", "CZ": "捷克", "DE": "德国",
    "DK": "丹麦", "EE": "爱沙尼亚", "ES": "西班牙", "FR": "法国",
    "HR": "克罗地亚", "IE": "爱尔兰", "LT": "立陶宛", "LV": "拉脱维亚",
    "NL": "荷兰", "PT": "葡萄牙", "RO": "罗马尼亚", "SE": "瑞典",
    "SI": "斯洛文尼亚", "SK": "斯洛伐克",
}

ZONES = {
    "atlantic":      {"label": "大西洋", "countries": ["BE", "DE", "DK", "FR", "IE", "NL"]},
    "continental":   {"label": "大陆性", "countries": ["AT", "CZ", "SK"]},
    "mediterranean": {"label": "地中海", "countries": ["ES", "HR", "PT"]},
    "nordic_baltic": {"label": "北欧-波罗的海", "countries": ["EE", "LT", "LV", "SE"]},
    "southeast":     {"label": "东南欧", "countries": ["RO"]},
}

FEATURE_SETS = ["ae", "s2", "fusion"]
FS_LABELS = {"ae": "AlphaEarth", "s2": "Sentinel-2", "fusion": "Fusion"}
FS_DIM_COORDS = {"ae": 131, "s2": 171, "fusion": 299}
FS_DIM_PURE   = {"ae": 128, "s2": 168, "fusion": 296}
FS_COLORS = {"ae": "#2196F3", "s2": "#FF9800", "fusion": "#4CAF50"}

# 学习效率实验的 样本标签
CURVE_LABELS = {
    "10": "100", "50": "500", "100": "1,000",
    "500": "5,000", "1000": "10,000", "5000": "46,025", "0": "全部",
}
CURVE_ORDER = ["10", "50", "100", "500", "1000", "5000", "0"]


# ────────────────────────────────────────────────
# 工具函数
# ────────────────────────────────────────────────
def pct(v):
    """浮点 → 百分比字符串，保留2位"""
    return f"{v * 100:.2f}"


def delta_str(d):
    """差值 → 带正负号的百分比字符串"""
    sign = "+" if d >= 0 else ""
    return f"{sign}{d * 100:.2f}"


def delta_class(d, invert=False):
    """返回 CSS class: positive / negative / neutral"""
    if abs(d) < 0.0005:
        return "neutral"
    if invert:
        return "positive" if d < 0 else "negative"
    return "positive" if d > 0 else "negative"


def delta_cell(d, invert=False):
    """生成一个带颜色的 td"""
    cls = delta_class(d, invert)
    return f'<td class="delta {cls}">{delta_str(d)}</td>'


# ────────────────────────────────────────────────
# HTML 生成
# ────────────────────────────────────────────────
def build_html(data: dict) -> str:
    """根据 data 生成完整 HTML 字符串"""

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    parts = []

    # ─── 头部 ───
    parts.append(f"""\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AlphaEarth 泛欧作物分类 — 辅助信息消融对比报告</title>
<style>
:root {{
    --ae: #2196F3;
    --s2: #FF9800;
    --fusion: #4CAF50;
    --bg: #FAFAFA;
    --card: #FFFFFF;
    --border: #E0E0E0;
    --text: #212121;
    --text2: #616161;
    --accent: #1565C0;
    --red: #C62828;
    --green: #2E7D32;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.7;
    padding: 0 20px;
}}
.container {{ max-width: 1200px; margin: 0 auto; }}

/* Header */
header {{
    background: linear-gradient(135deg, #1565C0, #0D47A1);
    color: white;
    padding: 40px 30px;
    margin: 20px -20px 30px;
    border-radius: 0 0 12px 12px;
}}
header h1 {{ font-size: 26px; margin-bottom: 8px; }}
header .subtitle {{ opacity: 0.92; font-size: 15px; margin-bottom: 12px; }}
header p.meta {{ opacity: 0.7; font-size: 12px; }}
.tag {{
    display: inline-block;
    background: rgba(255,255,255,0.2);
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 12px;
    margin-right: 8px;
    margin-top: 6px;
}}

/* Section */
.section {{
    background: var(--card);
    border-radius: 10px;
    border: 1px solid var(--border);
    margin-bottom: 24px;
    overflow: hidden;
}}
.section-header {{
    background: #F5F5F5;
    padding: 16px 24px;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.section-header h2 {{ font-size: 18px; color: var(--accent); }}
.section-header .exp-tag {{
    background: var(--accent);
    color: white;
    padding: 2px 10px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
}}
.section-body {{ padding: 24px; }}
.section-body p {{ margin-bottom: 12px; font-size: 14px; }}

/* Tables */
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    font-size: 13px;
}}
th, td {{
    padding: 7px 10px;
    text-align: center;
    border: 1px solid #E8E8E8;
}}
th {{
    background: #F5F5F5;
    font-weight: 600;
    font-size: 12px;
    letter-spacing: 0.3px;
}}
tr:hover {{ background: #FAFAFA; }}
td.delta.positive {{ color: var(--green); background: #E8F5E9; font-weight: 600; }}
td.delta.negative {{ color: var(--red); background: #FFEBEE; font-weight: 600; }}
td.delta.neutral  {{ color: #757575; background: #F5F5F5; }}
.mean-row {{ font-weight: 700; background: #F5F5F5 !important; }}
.mean-row:hover {{ background: #EEEEEE !important; }}

/* Feature set badges */
.fs-badge {{
    display: inline-block;
    padding: 1px 8px;
    border-radius: 4px;
    color: white;
    font-size: 11px;
    font-weight: 600;
}}
.fs-ae {{ background: var(--ae); }}
.fs-s2 {{ background: var(--s2); }}
.fs-fusion {{ background: var(--fusion); }}

/* Overview grid */
.overview-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 14px;
    margin: 20px 0;
}}
.stat-card {{
    background: var(--card);
    border-radius: 10px;
    padding: 18px 14px;
    border: 1px solid var(--border);
    text-align: center;
}}
.stat-card .number {{ font-size: 28px; font-weight: 700; }}
.stat-card .label {{ font-size: 12px; color: var(--text2); margin-top: 4px; }}

/* Findings */
.finding {{
    border-left: 4px solid var(--accent);
    background: #E3F2FD;
    padding: 12px 16px;
    margin: 16px 0;
    border-radius: 0 6px 6px 0;
    font-size: 14px;
}}
.finding strong {{ color: var(--accent); }}
.finding.warning {{
    background: #FFF3E0;
    border-left-color: #E65100;
}}
.finding.warning strong {{ color: #E65100; }}
.finding.success {{
    background: #E8F5E9;
    border-left-color: var(--green);
}}
.finding.success strong {{ color: var(--green); }}
.finding.critical {{
    background: #FCE4EC;
    border-left-color: var(--red);
}}
.finding.critical strong {{ color: var(--red); }}

/* Sub-table title */
.sub-title {{
    font-size: 15px;
    font-weight: 600;
    color: var(--text);
    margin: 20px 0 8px;
}}

/* Color code for column groups */
th.col-ae {{ background: #E3F2FD; }}
th.col-s2 {{ background: #FFF3E0; }}
th.col-fusion {{ background: #E8F5E9; }}

/* Compact table */
.compact-table td, .compact-table th {{ padding: 5px 8px; font-size: 12px; }}

/* Legend */
.legend {{
    display: flex;
    gap: 20px;
    justify-content: center;
    margin: 10px 0 6px;
    font-size: 12px;
    color: var(--text2);
}}
.legend span {{ display: flex; align-items: center; gap: 5px; }}
.legend .dot {{
    width: 10px; height: 10px; border-radius: 50%; display: inline-block;
}}

footer {{
    text-align: center;
    padding: 30px 0 20px;
    color: var(--text2);
    font-size: 12px;
}}
</style>
</head>
<body>
<div class="container">
""")

    # ─── HEADER ───
    parts.append(f"""\
<header>
    <h1>AlphaEarth 泛欧作物分类 — 辅助信息消融对比报告</h1>
    <div class="subtitle">有坐标+面积 vs 纯光谱特征 — 全部实验综合对比</div>
    <div>
        <span class="tag">18 个欧洲国家</span>
        <span class="tag">5 个气候带</span>
        <span class="tag">3 种特征集</span>
        <span class="tag">6 组实验</span>
    </div>
    <p class="meta">生成时间: {now}</p>
</header>
""")

    # ════════════════════════════════════════════
    # Section 0: 实验概览
    # ════════════════════════════════════════════
    parts.append("""\
<div class="section">
    <div class="section-header">
        <h2>Section 0: 实验概览</h2>
        <span class="exp-tag">OVERVIEW</span>
    </div>
    <div class="section-body">
        <p><strong>研究问题：</strong>在泛欧作物分类任务中，坐标（经度、纬度）和面积（log_area）这三个辅助特征对模型性能的贡献如何？
        它们在不同实验设置下（域内 ID / 域外 OOD / 气候带迁移 / 标签效率 / 特征分析）的作用是否一致？</p>

        <p><strong>消融设计：</strong>对每种特征集分别训练「含坐标」与「去坐标」两个版本，对比所有指标。</p>

        <table>
        <tr>
            <th>特征集</th>
            <th>有坐标 (dims)</th>
            <th>去坐标 (dims)</th>
            <th>移除特征</th>
        </tr>
        <tr>
            <td><span class="fs-badge fs-ae">AE</span> AlphaEarth</td>
            <td>131d (A00-A63 + 64 agg + coords)</td>
            <td>128d (纯光谱)</td>
            <td>longitude, latitude, log_area</td>
        </tr>
        <tr>
            <td><span class="fs-badge fs-s2">S2</span> Sentinel-2</td>
            <td>171d (168 S2 bands + coords)</td>
            <td>168d (纯光谱)</td>
            <td>longitude, latitude, log_area</td>
        </tr>
        <tr>
            <td><span class="fs-badge fs-fusion">FUS</span> Fusion</td>
            <td>299d (AE + S2 + coords)</td>
            <td>296d (纯光谱)</td>
            <td>longitude, latitude, log_area</td>
        </tr>
        </table>

        <div class="legend">
            <span><span class="dot" style="background:#E8F5E9"></span> 绿色 = 去坐标后提升（有利）</span>
            <span><span class="dot" style="background:#FFEBEE"></span> 红色 = 去坐标后下降</span>
        </div>
    </div>
</div>
""")

    # ════════════════════════════════════════════
    # Section 1: Exp 1 — ID Baseline 对比
    # ════════════════════════════════════════════
    exp1 = data["exp1"]
    rows1 = []
    for fs in FEATURE_SETS:
        c = exp1[f"coords_{fs}"]
        n = exp1[f"nocoords_{fs}"]
        d_ba = n["BA"] - c["BA"]
        d_f1 = n["F1"] - c["F1"]
        rows1.append(f"""\
        <tr>
            <td><span class="fs-badge fs-{fs}">{FS_LABELS[fs]}</span></td>
            <td>{c['n_features']}</td>
            <td>{n['n_features']}</td>
            <td>{pct(c['BA'])}</td>
            <td>{pct(n['BA'])}</td>
            {delta_cell(d_ba)}
            <td>{pct(c['F1'])}</td>
            <td>{pct(n['F1'])}</td>
            {delta_cell(d_f1)}
        </tr>""")

    # 计算 S2 的 BA 差值用于 finding
    s2_d_ba = exp1["nocoords_s2"]["BA"] - exp1["coords_s2"]["BA"]
    ae_d_ba = exp1["nocoords_ae"]["BA"] - exp1["coords_ae"]["BA"]
    fu_d_ba = exp1["nocoords_fusion"]["BA"] - exp1["coords_fusion"]["BA"]

    parts.append(f"""\
<div class="section">
    <div class="section-header">
        <h2>Section 1: Exp 1 — ID Baseline 对比</h2>
        <span class="exp-tag">EXP 1</span>
    </div>
    <div class="section-body">
        <p>在域内（ID）基线实验中，使用 18 国合并数据集、5-fold CV 的 Balanced Accuracy 和 F1 对比。</p>

        <table>
        <tr>
            <th>特征集</th>
            <th>维度 (coords)</th>
            <th>维度 (no-coords)</th>
            <th>BA (coords)</th>
            <th>BA (no-coords)</th>
            <th>&Delta; BA (pp)</th>
            <th>F1 (coords)</th>
            <th>F1 (no-coords)</th>
            <th>&Delta; F1 (pp)</th>
        </tr>
        {"".join(rows1)}
        </table>

        <div class="finding warning">
            <strong>关键发现：</strong>
            S2 在 ID 设置下受影响最大（&Delta;BA = {delta_str(s2_d_ba)} pp），
            AE 下降 {delta_str(ae_d_ba)} pp，
            Fusion 下降 {delta_str(fu_d_ba)} pp。
            坐标+面积在 ID 场景下为所有特征集提供了性能增益，
            但 S2 对辅助特征的依赖程度明显高于 AE。
        </div>
    </div>
</div>
""")

    # ════════════════════════════════════════════
    # Section 2: Exp 0 — 逐国 CV 对比
    # ════════════════════════════════════════════
    exp0c = data["exp0_coords"]
    exp0n = data["exp0_nocoords"]

    # 构建大表
    rows0 = []
    means_c = {fs: [] for fs in FEATURE_SETS}
    means_n = {fs: [] for fs in FEATURE_SETS}

    for cc in COUNTRIES:
        cells = [f"<td><strong>{cc}</strong> {COUNTRY_NAMES[cc]}</td>"]
        for fs in FEATURE_SETS:
            ck = f"{cc}_{fs}"
            ba_c = exp0c[ck]["BA"]
            ba_n = exp0n[ck]["BA"]
            d = ba_n - ba_c
            means_c[fs].append(ba_c)
            means_n[fs].append(ba_n)
            cells.append(f"<td>{pct(ba_c)}</td>")
            cells.append(f"<td>{pct(ba_n)}</td>")
            cells.append(delta_cell(d))
        rows0.append(f"<tr>{''.join(cells)}</tr>")

    # 均值行
    mean_cells = ['<td class="mean-row"><strong>均值</strong></td>']
    for fs in FEATURE_SETS:
        mc = sum(means_c[fs]) / len(means_c[fs])
        mn = sum(means_n[fs]) / len(means_n[fs])
        d = mn - mc
        mean_cells.append(f'<td class="mean-row">{pct(mc)}</td>')
        mean_cells.append(f'<td class="mean-row">{pct(mn)}</td>')
        mean_cells.append(f'<td class="mean-row delta {"positive" if d>=0 else "negative"}">{delta_str(d)}</td>')
    rows0.append(f'<tr class="mean-row">{"".join(mean_cells)}</tr>')

    # 找出每个特征集中提升最大/下降最大的国家
    best_improve = {}
    worst_drop = {}
    for fs in FEATURE_SETS:
        deltas = {}
        for cc in COUNTRIES:
            ck = f"{cc}_{fs}"
            deltas[cc] = exp0n[ck]["BA"] - exp0c[ck]["BA"]
        best_improve[fs] = max(deltas, key=deltas.get)
        worst_drop[fs] = min(deltas, key=deltas.get)

    parts.append(f"""\
<div class="section">
    <div class="section-header">
        <h2>Section 2: Exp 0 — 逐国 CV 对比</h2>
        <span class="exp-tag">EXP 0</span>
    </div>
    <div class="section-body">
        <p>每个国家独立 5-fold CV 的 Balanced Accuracy 对比。去坐标后的变化反映了辅助特征在各国的相对重要性。</p>

        <table class="compact-table">
        <tr>
            <th rowspan="2">国家</th>
            <th colspan="3" class="col-ae">AlphaEarth</th>
            <th colspan="3" class="col-s2">Sentinel-2</th>
            <th colspan="3" class="col-fusion">Fusion</th>
        </tr>
        <tr>
            <th class="col-ae">BA(c)</th><th class="col-ae">BA(nc)</th><th class="col-ae">&Delta;</th>
            <th class="col-s2">BA(c)</th><th class="col-s2">BA(nc)</th><th class="col-s2">&Delta;</th>
            <th class="col-fusion">BA(c)</th><th class="col-fusion">BA(nc)</th><th class="col-fusion">&Delta;</th>
        </tr>
        {"".join(rows0)}
        </table>

        <div class="finding">
            <strong>关键发现：</strong>
            逐国 CV 中，去坐标后大多数国家 BA 下降。
            AE 平均 &Delta;BA = {delta_str(sum(means_n["ae"])/len(means_n["ae"]) - sum(means_c["ae"])/len(means_c["ae"]))} pp；
            S2 平均 &Delta;BA = {delta_str(sum(means_n["s2"])/len(means_n["s2"]) - sum(means_c["s2"])/len(means_c["s2"]))} pp；
            Fusion 平均 &Delta;BA = {delta_str(sum(means_n["fusion"])/len(means_n["fusion"]) - sum(means_c["fusion"])/len(means_c["fusion"]))} pp。
            IE（爱尔兰）在 AE/S2 中去坐标后反而有微弱提升，
            说明坐标在部分国家可能引入噪声。
        </div>
    </div>
</div>
""")

    # ════════════════════════════════════════════
    # Section 3: Exp 2 — LOCO (OOD) 对比
    # ════════════════════════════════════════════
    exp2 = data["exp2"]

    rows2 = []
    ood_means_c = {fs: [] for fs in FEATURE_SETS}
    ood_means_n = {fs: [] for fs in FEATURE_SETS}

    for cc in COUNTRIES:
        cells = [f"<td><strong>{cc}</strong> {COUNTRY_NAMES[cc]}</td>"]
        for fs in FEATURE_SETS:
            ba_c = exp2[f"coords_{fs}"]["countries"][cc]
            ba_n = exp2[f"nocoords_{fs}"]["countries"][cc]
            d = ba_n - ba_c
            ood_means_c[fs].append(ba_c)
            ood_means_n[fs].append(ba_n)
            cells.append(f"<td>{pct(ba_c)}</td>")
            cells.append(f"<td>{pct(ba_n)}</td>")
            # OOD 中提升 = 正向（绿色），因为我们希望 OOD 更好
            cells.append(delta_cell(d))
        rows2.append(f"<tr>{''.join(cells)}</tr>")

    # 均值行
    mean_cells2 = ['<td class="mean-row"><strong>均值</strong></td>']
    ood_deltas = {}
    for fs in FEATURE_SETS:
        mc = exp2[f"coords_{fs}"]["mean_ba"]
        mn = exp2[f"nocoords_{fs}"]["mean_ba"]
        d = mn - mc
        ood_deltas[fs] = d
        mean_cells2.append(f'<td class="mean-row">{pct(mc)}</td>')
        mean_cells2.append(f'<td class="mean-row">{pct(mn)}</td>')
        mean_cells2.append(f'<td class="mean-row delta {"positive" if d>=0 else "negative"}">{delta_str(d)}</td>')
    rows2.append(f'<tr class="mean-row">{"".join(mean_cells2)}</tr>')

    # 找出 OOD 改善最多的国家
    improve_countries = {}
    for fs in FEATURE_SETS:
        deltas = {}
        for cc in COUNTRIES:
            deltas[cc] = exp2[f"nocoords_{fs}"]["countries"][cc] - exp2[f"coords_{fs}"]["countries"][cc]
        improve_countries[fs] = sorted(deltas.items(), key=lambda x: -x[1])[:3]

    parts.append(f"""\
<div class="section">
    <div class="section-header">
        <h2>Section 3: Exp 2 — LOCO (OOD) 对比</h2>
        <span class="exp-tag">EXP 2</span>
    </div>
    <div class="section-body">
        <p>Leave-One-Country-Out：在其他 17 国训练、目标国测试。OOD 设置下，坐标是否帮助还是阻碍泛化？</p>

        <table class="compact-table">
        <tr>
            <th rowspan="2">国家</th>
            <th colspan="3" class="col-ae">AlphaEarth</th>
            <th colspan="3" class="col-s2">Sentinel-2</th>
            <th colspan="3" class="col-fusion">Fusion</th>
        </tr>
        <tr>
            <th class="col-ae">BA(c)</th><th class="col-ae">BA(nc)</th><th class="col-ae">&Delta;</th>
            <th class="col-s2">BA(c)</th><th class="col-s2">BA(nc)</th><th class="col-s2">&Delta;</th>
            <th class="col-fusion">BA(c)</th><th class="col-fusion">BA(nc)</th><th class="col-fusion">&Delta;</th>
        </tr>
        {"".join(rows2)}
        </table>

        <div class="finding success">
            <strong>核心发现 — OOD 泛化提升：</strong>
            在 LOCO 设置下，去坐标后 AE 平均 OOD BA <strong>提升 {delta_str(ood_deltas["ae"])} pp</strong>，
            S2 <strong>提升 {delta_str(ood_deltas["s2"])} pp</strong>，
            Fusion <strong>提升 {delta_str(ood_deltas["fusion"])} pp</strong>。
            坐标在 ID 场景有利，但在 OOD 场景<strong>阻碍泛化</strong>。
            这是本报告最重要的发现之一：辅助特征让模型过拟合于地理位置，
            削弱了对未见国家的迁移能力。
        </div>

        <div class="finding">
            <strong>OOD 改善最大的国家：</strong>
            AE: {', '.join(f'{c}(+{d*100:.1f}pp)' for c, d in improve_countries['ae'])}；
            S2: {', '.join(f'{c}(+{d*100:.1f}pp)' for c, d in improve_countries['s2'])}；
            Fusion: {', '.join(f'{c}(+{d*100:.1f}pp)' for c, d in improve_countries['fusion'])}。
        </div>
    </div>
</div>
""")

    # ════════════════════════════════════════════
    # Section 4: Exp 3 — 气候带迁移对比
    # ════════════════════════════════════════════
    exp3 = data["exp3"]
    zone_keys = ["atlantic", "continental", "mediterranean", "nordic_baltic", "southeast"]

    # Sub-table A: LOZO
    rows3a = []
    for zk in zone_keys:
        zl = ZONES[zk]["label"]
        cells = [f"<td><strong>{zl}</strong></td>"]
        for fs in FEATURE_SETS:
            bc = exp3[f"coords_{fs}"]["zones"][zk]
            bn = exp3[f"nocoords_{fs}"]["zones"][zk]
            d = bn - bc
            cells.append(f"<td>{pct(bc)}</td>")
            cells.append(f"<td>{pct(bn)}</td>")
            cells.append(delta_cell(d))
        rows3a.append(f"<tr>{''.join(cells)}</tr>")

    # Transfer mean
    rows3b_cells = ['<td><strong>Transfer Mean</strong></td>']
    for fs in FEATURE_SETS:
        tc = exp3[f"coords_{fs}"]["transfer_mean"]
        tn = exp3[f"nocoords_{fs}"]["transfer_mean"]
        d = tn - tc
        rows3b_cells.append(f"<td>{pct(tc)}</td>")
        rows3b_cells.append(f"<td>{pct(tn)}</td>")
        rows3b_cells.append(delta_cell(d))
    rows3a.append(f'<tr class="mean-row">{"".join(rows3b_cells)}</tr>')

    # nordic_baltic 差值
    nb_d = {fs: exp3[f"nocoords_{fs}"]["zones"]["nordic_baltic"] - exp3[f"coords_{fs}"]["zones"]["nordic_baltic"]
            for fs in FEATURE_SETS}

    parts.append(f"""\
<div class="section">
    <div class="section-header">
        <h2>Section 4: Exp 3 — 气候带迁移对比</h2>
        <span class="exp-tag">EXP 3</span>
    </div>
    <div class="section-body">
        <p>Leave-One-Zone-Out (LOZO)：在其他气候带训练、目标带测试。以及跨气候带迁移矩阵的均值。</p>

        <div class="sub-title">A. LOZO 各气候带 BA</div>
        <table class="compact-table">
        <tr>
            <th rowspan="2">气候带</th>
            <th colspan="3" class="col-ae">AlphaEarth</th>
            <th colspan="3" class="col-s2">Sentinel-2</th>
            <th colspan="3" class="col-fusion">Fusion</th>
        </tr>
        <tr>
            <th class="col-ae">BA(c)</th><th class="col-ae">BA(nc)</th><th class="col-ae">&Delta;</th>
            <th class="col-s2">BA(c)</th><th class="col-s2">BA(nc)</th><th class="col-s2">&Delta;</th>
            <th class="col-fusion">BA(c)</th><th class="col-fusion">BA(nc)</th><th class="col-fusion">&Delta;</th>
        </tr>
        {"".join(rows3a)}
        </table>

        <div class="finding success">
            <strong>关键发现 — 北欧-波罗的海带迁移大幅改善：</strong>
            去坐标后，nordic_baltic 气候带 LOZO BA 显著提升：
            AE +{nb_d["ae"]*100:.1f}pp，
            S2 +{nb_d["s2"]*100:.1f}pp，
            Fusion +{nb_d["fusion"]*100:.1f}pp。
            坐标特征让模型过度依赖地理位置，当目标区域的纬度与训练集差异较大时（如北欧），迁移失败。
            去坐标后模型被迫依赖光谱特征，泛化能力增强。
        </div>

        <div class="finding">
            <strong>Transfer Mean BA：</strong>
            跨带迁移均值变化较小，
            AE: {delta_str(exp3["nocoords_ae"]["transfer_mean"]-exp3["coords_ae"]["transfer_mean"])} pp，
            S2: {delta_str(exp3["nocoords_s2"]["transfer_mean"]-exp3["coords_s2"]["transfer_mean"])} pp，
            Fusion: {delta_str(exp3["nocoords_fusion"]["transfer_mean"]-exp3["coords_fusion"]["transfer_mean"])} pp。
            说明坐标在整体迁移中的影响被各带差异平均掉，但对特定带（北欧-波罗的海）影响很大。
        </div>
    </div>
</div>
""")

    # ════════════════════════════════════════════
    # Section 5: Exp 4 — 标签效率对比
    # ════════════════════════════════════════════
    exp4 = data["exp4"]

    # 只对 AE 和 S2 对比（融合没有）
    exp4_sets = ["ae", "s2"]

    # curve 部分
    rows4c = []
    for sk in CURVE_ORDER:
        cells = [f"<td>{CURVE_LABELS[sk]}</td>"]
        for fs in exp4_sets:
            bc = exp4[f"coords_{fs}"]["curve"][sk]
            bn = exp4[f"nocoords_{fs}"]["curve"][sk]
            d = bn - bc
            cells.append(f"<td>{pct(bc)}</td>")
            cells.append(f"<td>{pct(bn)}</td>")
            cells.append(delta_cell(d))
        rows4c.append(f"<tr>{''.join(cells)}</tr>")

    # mixed 部分
    mixed_keys = ["0", "10", "50", "100", "500"]
    mixed_labels = {"0": "0%", "10": "10%", "50": "50%", "100": "100%", "500": "500%"}
    rows4m = []
    for mk in mixed_keys:
        cells = [f"<td>{mixed_labels[mk]}</td>"]
        for fs in exp4_sets:
            bc = exp4[f"coords_{fs}"]["mixed"][mk]
            bn = exp4[f"nocoords_{fs}"]["mixed"][mk]
            d = bn - bc
            cells.append(f"<td>{pct(bc)}</td>")
            cells.append(f"<td>{pct(bn)}</td>")
            cells.append(delta_cell(d))
        rows4m.append(f"<tr>{''.join(cells)}</tr>")

    parts.append(f"""\
<div class="section">
    <div class="section-header">
        <h2>Section 5: Exp 4 — 标签效率对比</h2>
        <span class="exp-tag">EXP 4</span>
    </div>
    <div class="section-body">
        <p>在不同训练样本量下比较有/无坐标的 OOD BA 表现。</p>

        <div class="sub-title">A. 学习曲线（全域训练 → OOD 测试）</div>
        <p>每类样本数 → 对应总样本量；所有值为 LOCO 18 国平均 BA。</p>
        <table class="compact-table">
        <tr>
            <th rowspan="2">总样本量</th>
            <th colspan="3" class="col-ae">AlphaEarth</th>
            <th colspan="3" class="col-s2">Sentinel-2</th>
        </tr>
        <tr>
            <th class="col-ae">BA(c)</th><th class="col-ae">BA(nc)</th><th class="col-ae">&Delta;</th>
            <th class="col-s2">BA(c)</th><th class="col-s2">BA(nc)</th><th class="col-s2">&Delta;</th>
        </tr>
        {"".join(rows4c)}
        </table>

        <div class="sub-title">B. 混合比例实验（in-country 混入比例 → OOD BA）</div>
        <table class="compact-table">
        <tr>
            <th rowspan="2">混入比例</th>
            <th colspan="3" class="col-ae">AlphaEarth</th>
            <th colspan="3" class="col-s2">Sentinel-2</th>
        </tr>
        <tr>
            <th class="col-ae">BA(c)</th><th class="col-ae">BA(nc)</th><th class="col-ae">&Delta;</th>
            <th class="col-s2">BA(c)</th><th class="col-s2">BA(nc)</th><th class="col-s2">&Delta;</th>
        </tr>
        {"".join(rows4m)}
        </table>

        <div class="finding">
            <strong>关键发现：</strong>
            在学习曲线中，小样本（100-1000）时有/无坐标差异极小（&lt;1pp），
            随样本增加差异才显现。说明坐标信息需要足够样本量才能被模型有效利用。
            混合实验中，有坐标版本混入少量 in-country 数据后 OOD 性能更好，
            因为坐标帮助模型定位目标国。但无坐标版本对混入比例更不敏感，表现更稳定。
        </div>
    </div>
</div>
""")

    # ════════════════════════════════════════════
    # Section 6: Exp 5 — 特征分析 (CKA / Country Predictability)
    # ════════════════════════════════════════════
    exp5 = data["exp5"]

    rows5 = []
    for fs in ["ae", "s2"]:
        c = exp5[f"coords_{fs}"]
        n = exp5[f"nocoords_{fs}"]
        d_crop = n["cka_crop"] - c["cka_crop"]
        d_country = n["cka_country"] - c["cka_country"]
        d_pred = n["country_pred_ba"] - c["country_pred_ba"]
        rows5.append(f"""\
        <tr>
            <td><span class="fs-badge fs-{fs}">{FS_LABELS[fs]}</span></td>
            <td>{c['cka_crop']:.4f}</td>
            <td>{n['cka_crop']:.4f}</td>
            {delta_cell(d_crop)}
            <td>{c['cka_country']:.4f}</td>
            <td>{n['cka_country']:.4f}</td>
            {delta_cell(d_country)}
            <td>{pct(c['country_pred_ba'])}</td>
            <td>{pct(n['country_pred_ba'])}</td>
            {delta_cell(d_pred)}
        </tr>""")

    ae_cka_crop_c = exp5["coords_ae"]["cka_crop"]
    ae_cka_crop_n = exp5["nocoords_ae"]["cka_crop"]
    ae_cpred_c = exp5["coords_ae"]["country_pred_ba"]
    ae_cpred_n = exp5["nocoords_ae"]["country_pred_ba"]
    s2_cpred_n = exp5["nocoords_s2"]["country_pred_ba"]

    parts.append(f"""\
<div class="section">
    <div class="section-header">
        <h2>Section 6: Exp 5 — 特征分析 (CKA / Country Predictability)</h2>
        <span class="exp-tag">EXP 5</span>
    </div>
    <div class="section-body">
        <p>CKA (Centered Kernel Alignment) 衡量特征与作物标签/国家标签的对齐程度。
        Country Predictability 衡量从嵌入特征预测国家身份的能力（越低 → 越地理无关）。</p>

        <table>
        <tr>
            <th rowspan="2">特征集</th>
            <th colspan="3">CKA<sub>crop</sub></th>
            <th colspan="3">CKA<sub>country</sub></th>
            <th colspan="3">国家预测 BA (%)</th>
        </tr>
        <tr>
            <th>coords</th><th>no-coords</th><th>&Delta;</th>
            <th>coords</th><th>no-coords</th><th>&Delta;</th>
            <th>coords</th><th>no-coords</th><th>&Delta;</th>
        </tr>
        {"".join(rows5)}
        </table>

        <div class="finding critical">
            <strong>最关键发现 — 坐标遮蔽了光谱信号：</strong>
            AE 的 CKA<sub>crop</sub> 从 {ae_cka_crop_c:.4f}（有坐标）跳升到 <strong>{ae_cka_crop_n:.4f}</strong>（无坐标），
            增长近 <strong>8.7 倍</strong>。
            这意味着当坐标存在时，LightGBM 主要依赖地理位置而非光谱特征来分类作物。
            去掉坐标后，模型被迫学习光谱模式，CKA<sub>crop</sub> 大幅提升。
        </div>

        <div class="finding success">
            <strong>地理可预测性大幅下降：</strong>
            国家预测 BA：AE 从 {pct(ae_cpred_c)}% 降至 <strong>{pct(ae_cpred_n)}%</strong>，
            S2 从 {pct(exp5["coords_s2"]["country_pred_ba"])}% 降至 <strong>{pct(s2_cpred_n)}%</strong>。
            说明去坐标后嵌入的地理信息被大幅削弱（尤其 S2 降至 {pct(s2_cpred_n)}%），
            模型不再能轻易从特征中推断出样本来自哪个国家。
        </div>
    </div>
</div>
""")

    # ════════════════════════════════════════════
    # Section 7: 综合结论
    # ════════════════════════════════════════════
    parts.append(f"""\
<div class="section">
    <div class="section-header">
        <h2>Section 7: 综合结论</h2>
        <span class="exp-tag">CONCLUSION</span>
    </div>
    <div class="section-body">
        <div class="sub-title">1. 辅助特征（坐标+面积）的双刃剑效应</div>
        <p>坐标和面积特征在<strong>域内 (ID)</strong> 场景下一致提升性能
        （Exp 1: AE {delta_str(ae_d_ba)}pp, S2 {delta_str(s2_d_ba)}pp, Fusion {delta_str(fu_d_ba)}pp），
        但在<strong>域外 (OOD)</strong> 场景下反而<strong>促进泛化失败</strong>。</p>

        <div class="sub-title">2. OOD 泛化: 去坐标反而更好</div>
        <p>Exp 2 (LOCO) 中，去坐标后三种特征集的平均 OOD BA 均提升
        （AE {delta_str(ood_deltas["ae"])}pp, S2 {delta_str(ood_deltas["s2"])}pp, Fusion {delta_str(ood_deltas["fusion"])}pp）。
        Exp 3 (LOZO) 中，北欧-波罗的海气候带提升尤为显著
        （S2 +{nb_d["s2"]*100:.1f}pp）。</p>

        <div class="sub-title">3. 坐标遮蔽光谱信号（Exp 5 核心发现）</div>
        <p>AE 的 CKA<sub>crop</sub> 从 0.024 跳升到 0.207（8.7x），说明坐标存在时模型几乎不学习光谱模式。
        国家可预测性从 99.0% 降至 81.5%（AE）/ 65.4%（S2），证实坐标是地理泄漏的主要来源。</p>

        <div class="sub-title">4. 对论文的启示</div>
        <div class="finding">
            <strong>建议：</strong>
            <ul style="margin-top:8px;padding-left:20px;line-height:2;">
                <li>主实验（对比 AE vs S2 的泛化能力）应使用<strong>纯光谱版本</strong>（128d / 168d / 296d），以公平评估嵌入本身的质量。</li>
                <li>将「有坐标 vs 无坐标」消融作为论文的一个重要分析章节，展示辅助特征的 ID/OOD trade-off。</li>
                <li>CKA<sub>crop</sub> 的 8.7x 跳升是一个非常有力的证据，可作为论文亮点。</li>
                <li>S2 的国家预测 BA 降至 65.4% 说明去坐标后 S2 特征的地理信息最少，泛化潜力最大。</li>
            </ul>
        </div>

        <div class="sub-title">5. 总结表</div>
        <table>
        <tr>
            <th>实验</th>
            <th>设置</th>
            <th>坐标的作用</th>
            <th>结论</th>
        </tr>
        <tr>
            <td>Exp 0 逐国 CV</td>
            <td>ID</td>
            <td style="color:var(--green)">有利（小幅提升）</td>
            <td>坐标在域内提供额外判别力</td>
        </tr>
        <tr>
            <td>Exp 1 ID Baseline</td>
            <td>ID</td>
            <td style="color:var(--green)">有利（S2 最明显）</td>
            <td>S2 对辅助特征依赖最强</td>
        </tr>
        <tr>
            <td>Exp 2 LOCO</td>
            <td>OOD</td>
            <td style="color:var(--red)"><strong>有害</strong>（阻碍泛化）</td>
            <td>去坐标后 OOD 全面提升</td>
        </tr>
        <tr>
            <td>Exp 3 LOZO</td>
            <td>OOD</td>
            <td style="color:var(--red)"><strong>有害</strong>（北欧尤甚）</td>
            <td>地理差异大的区域受害最深</td>
        </tr>
        <tr>
            <td>Exp 4 标签效率</td>
            <td>OOD</td>
            <td>小样本时影响小</td>
            <td>坐标需大量样本才能利用</td>
        </tr>
        <tr>
            <td>Exp 5 特征分析</td>
            <td>—</td>
            <td style="color:var(--red)"><strong>遮蔽光谱信号</strong></td>
            <td>CKA<sub>crop</sub> 8.7x 跳升为核心证据</td>
        </tr>
        </table>
    </div>
</div>
""")

    # ─── Footer ───
    parts.append(f"""\
<footer>
    AlphaEarth 泛欧作物分类实验 &mdash; 辅助信息消融对比报告 &mdash; 自动生成于 {now}
</footer>
</div>
</body>
</html>
""")

    return "".join(parts)


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────
def main():
    # 确定项目根目录
    script_dir = Path(__file__).resolve().parent          # src/experiments/
    project_root = script_dir.parent.parent               # Alpha_Crops/

    input_path = project_root / "results" / "experiments" / "all_results_combined.json"
    output_path = project_root / "results" / "experiments" / "comparison_report.html"

    if not input_path.exists():
        print(f"[ERROR] 找不到输入文件: {input_path}")
        sys.exit(1)

    print(f"[INFO] 读取: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("[INFO] 生成 HTML 报告 ...")
    html = build_html(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"[OK] 报告已生成: {output_path}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
