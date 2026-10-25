"""
将 comparison/ 目录下的 12 张 PNG 图片以 base64 data URI 嵌入到 comparison_report.html 中，
使 HTML 成为自包含文件，便于分享。
"""

import base64
from pathlib import Path

# ── 路径 ──────────────────────────────────────────────
BASE = Path(r"G:\我的云端硬盘\Research\Alpha_Crops\results\experiments")
HTML_PATH = BASE / "comparison_report.html"
FIG_DIR = BASE / "comparison"

# ── 各 Section 对应的图片 ─────────────────────────────
SECTION_FIGURES = {
    "Section 1": ["fig1_exp1_id_comparison.png"],
    "Section 2": ["fig2_exp0_country_ba_delta.png", "fig3_exp0_country_comparison.png"],
    "Section 3": ["fig4_exp2_loco_comparison.png", "fig5_exp2_loco_delta_per_country.png", "fig6_exp2_id_ood_gap.png"],
    "Section 4": ["fig7_exp3_lozo_comparison.png", "fig8_exp3_transfer_comparison.png"],
    "Section 5": ["fig9_exp4_learning_curve_comparison.png"],
    "Section 6": ["fig10_exp5_cka_comparison.png", "fig11_exp5_country_pred.png"],
    "Section 7": ["fig12_summary_all_experiments.png"],
}


def png_to_data_uri(filepath: Path) -> str:
    """将 PNG 文件编码为 base64 data URI。"""
    with open(filepath, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_img_block(filepaths: list[Path]) -> str:
    """为一组图片生成 HTML <img> 标签块。"""
    parts = []
    for fp in filepaths:
        data_uri = png_to_data_uri(fp)
        alt_text = fp.stem.replace("_", " ")
        parts.append(
            f'<div style="margin:18px 0; text-align:center;">'
            f'<img src="{data_uri}" alt="{alt_text}" '
            f'style="max-width:100%; display:block; margin:0 auto; '
            f'border:1px solid #E0E0E0; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.08);" />'
            f'<div style="font-size:12px; color:#757575; margin-top:6px;">{alt_text}</div>'
            f'</div>'
        )
    return "\n".join(parts)


def main():
    html = HTML_PATH.read_text(encoding="utf-8")

    # 逐 section 插入图片
    for section_key, fig_names in SECTION_FIGURES.items():
        fig_paths = [FIG_DIR / name for name in fig_names]

        # 确认所有图片存在
        for fp in fig_paths:
            if not fp.exists():
                print(f"[WARNING] 图片不存在: {fp}")

        img_html = build_img_block(fig_paths)

        # 找到 section 标题所在的 <h2> 行，然后找到该 section-body 的末尾 </div>
        # 策略：找到 "Section N" 对应的 <h2>，然后找该 section-body 末尾的 </div>
        # 即找到 "Section N" 后面最近的 "<!-- /section-body -->" 或倒着数的 </div>

        # 更可靠的方式：找 section-header 中的 h2 包含 section_key 的位置，
        # 然后找后续第一个 class="finding" 的最后一个 </div> 之后，
        # 或者找该 section 的 </div></div> 结尾。

        # 最简洁的方法：在 section-body 的最后一个 </div> 前插入，
        # 即在 "</div>\n</div>" (section-body close + section close) 之前。

        # 我们使用精准定位：找到包含 section_key 的 <h2> 行，
        # 然后找该 <div class="section"> 块末尾的 section-body 关闭标签。

        # 由于 HTML 结构规律：每个 section 都是
        #   <div class="section">
        #     <div class="section-header"><h2>Section N: ...</h2>...</div>
        #     <div class="section-body">
        #       ... content ...
        #     </div>     ← 这是 section-body 的关闭
        #   </div>       ← 这是 section 的关闭

        # 找 section_key 在 html 中的位置
        idx = html.find(f"<h2>{section_key}")
        if idx == -1:
            print(f"[WARNING] 未找到 section: {section_key}")
            continue

        # 从这个位置向后找 section-body 的关闭 </div>
        # section-body 关闭后紧接着是 section 关闭的 </div>
        # 我们需要找到该 section 的结束标记 "</div>\n</div>"

        # 找该 section 的末尾：从 idx 开始找 "</div>\n</div>"
        # 这对应于 section-body 关闭 + section 关闭

        close_marker = "    </div>\n</div>"
        close_idx = html.find(close_marker, idx)
        if close_idx == -1:
            print(f"[WARNING] 未找到 section 关闭标签: {section_key}")
            continue

        # 在 section-body 关闭标签前插入图片
        insert_point = close_idx
        html = html[:insert_point] + "\n" + img_html + "\n" + html[insert_point:]

    # 写入更新后的 HTML
    HTML_PATH.write_text(html, encoding="utf-8")

    size_kb = HTML_PATH.stat().st_size / 1024
    size_mb = size_kb / 1024
    print(f"[OK] 已更新: {HTML_PATH}")
    print(f"[OK] 文件大小: {size_kb:.0f} KB ({size_mb:.2f} MB)")
    print(f"[OK] 共嵌入 {sum(len(v) for v in SECTION_FIGURES.values())} 张图片")


if __name__ == "__main__":
    main()
