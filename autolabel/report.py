"""
report.py - one self-contained HTML file: score table on top, then every test
image with gold boxes and each model's boxes drawn on it, one colour per model,
and a checkbox per model to show or hide its boxes.

Images are downscaled and embedded as base64 JPEG so the file opens anywhere
without the image folder.
"""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

from PIL import Image

PALETTE = ["#2563eb", "#ea580c", "#c026d3", "#ca8a04", "#0891b2", "#dc2626"]
GOLD_COLOUR = "#16a34a"


def _embed(path: Path, width: int) -> tuple[str, float, int, int]:
    with Image.open(path) as im:
        im = im.convert("RGB")
        scale = min(1.0, width / im.width)
        small = im.resize((round(im.width * scale), round(im.height * scale)))
    buf = io.BytesIO()
    small.save(buf, "JPEG", quality=72)
    return base64.b64encode(buf.getvalue()).decode(), scale, small.width, small.height


def _boxes_svg(boxes, scale: float, colour: str, css_class: str, dashed: bool = False) -> str:
    parts = []
    for b in boxes:
        cls, (x, y, w, h) = b[0], b[1]
        label = cls if len(b) < 3 else f"{cls} {b[2]:.2f}"
        x, y, w, h = x * scale, y * scale, w * scale, h * scale
        dash = ' stroke-dasharray="6 3"' if dashed else ""
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="none" '
            f'stroke="{colour}" stroke-width="2"{dash}/>'
            f'<text x="{x + 2:.1f}" y="{max(10.0, y - 3):.1f}" fill="{colour}">{html.escape(label)}</text>'
        )
    return f'<g class="{css_class}">{"".join(parts)}</g>'


def build_report(
    out_path: Path,
    title: str,
    notes: list[str],
    table_html: str,
    images: list[tuple[str, Path]],
    gold: dict[str, list],
    models: dict[str, dict[str, list]],
    width: int = 960,
) -> None:
    """
    images: (file_name, path) in display order. gold[file] = [(class, bbox)].
    models[name][file] = [(class, bbox, score)] already filtered to the shown threshold.
    """
    colours = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(models)}
    keys = {name: f"m{i}" for i, name in enumerate(models)}

    toggles = [f'<label><input type="checkbox" checked data-k="gold"> '
               f'<span class="sw" style="border-color:{GOLD_COLOUR};border-style:dashed"></span>gold (provisional)</label>']
    for name in models:
        toggles.append(f'<label><input type="checkbox" checked data-k="{keys[name]}"> '
                       f'<span class="sw" style="border-color:{colours[name]}"></span>{html.escape(name)}</label>')

    cards = []
    for file_name, path in images:
        b64, scale, w, h = _embed(path, width)
        layers = [_boxes_svg(gold.get(file_name, []), scale, GOLD_COLOUR, "gold", dashed=True)]
        for name in models:
            layers.append(_boxes_svg(models[name].get(file_name, []), scale, colours[name], keys[name]))
        cards.append(
            f'<figure><figcaption>{html.escape(file_name)}</figcaption>'
            f'<div class="frame"><img src="data:image/jpeg;base64,{b64}" width="{w}" height="{h}" alt="">'
            f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none">{"".join(layers)}</svg></div></figure>'
        )

    hide_rules = "\n".join(f"body.hide-{k} .{k} {{ display: none; }}" for k in ["gold", *keys.values()])
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ --bg: #ffffff; --fg: #111827; --muted: #6b7280; --line: #e5e7eb; --panel: #f9fafb; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #111827; --fg: #f3f4f6; --muted: #9ca3af; --line: #374151; --panel: #1f2937; }} }}
body {{ background: var(--bg); color: var(--fg); font: 14px/1.45 system-ui, sans-serif; margin: 0 auto; padding: 16px; max-width: 1000px; }}
h1 {{ font-size: 22px; margin: 0 0 8px; }}
.notes {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 8px 16px; }}
.notes li {{ margin: 4px 0; }}
.tablewrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; margin: 12px 0; font-variant-numeric: tabular-nums; }}
th, td {{ border: 1px solid var(--line); padding: 4px 8px; text-align: right; white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; }}
.controls {{ position: sticky; top: 0; background: var(--bg); padding: 8px 0; border-bottom: 1px solid var(--line); z-index: 2; display: flex; flex-wrap: wrap; gap: 12px; }}
.sw {{ display: inline-block; width: 14px; height: 10px; border: 2px solid; margin: 0 4px; vertical-align: middle; }}
figure {{ margin: 16px 0; }}
figcaption {{ color: var(--muted); font-size: 12px; margin-bottom: 4px; word-break: break-all; }}
.frame {{ position: relative; max-width: 100%; }}
.frame img {{ display: block; max-width: 100%; height: auto; }}
.frame svg {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
svg text {{ font: 11px system-ui, sans-serif; paint-order: stroke; stroke: #000; stroke-width: 3px; }}
{hide_rules}
</style></head><body>
<h1>{html.escape(title)}</h1>
<ul class="notes">{"".join(f"<li>{n}</li>" for n in notes)}</ul>
<div class="tablewrap">{table_html}</div>
<div class="controls">{"".join(toggles)}</div>
{"".join(cards)}
<script>
document.querySelectorAll('.controls input').forEach(function (cb) {{
  cb.addEventListener('change', function () {{ document.body.classList.toggle('hide-' + cb.dataset.k, !cb.checked); }});
}});
</script>
</body></html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
