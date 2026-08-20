"""Deterministic public-safe summaries and dependency-free SVG diagnostics."""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from .config import write_stable_json


def write_summary(path: str | Path, summary: dict[str, object]) -> None:
    write_stable_json(path, summary)


def write_pitch_profile_svg(path: str | Path, points: list[tuple[str, float, float]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    width, height = 760, 90 + 52 * len(points)
    max_pitch = max([truth for _, truth, _ in points] + [value for _, _, value in points] + [1.0])
    scale = 500.0 / max_pitch
    rows = []
    for index, (label, truth, recovered) in enumerate(points):
        y = 55 + index * 52
        rows.append(
            f'<text x="10" y="{y + 14}" font-size="13">{escape(label)}</text>'
            f'<line x1="170" y1="{y}" x2="{170 + truth * scale:.3f}" y2="{y}" '
            'stroke="#2563eb" stroke-width="8"/>'
            f'<circle cx="{170 + recovered * scale:.3f}" cy="{y}" r="6" fill="#dc2626"/>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        '<text x="10" y="25" font-size="18" font-family="sans-serif">Synthetic truth vs recovered pitch (x/12)</text>\n'
        '<g font-family="sans-serif">\n' + "\n".join(rows) + "\n</g>\n"
        '<text x="170" y="45" font-size="12" fill="#2563eb">blue = analytic truth</text>\n'
        '<text x="350" y="45" font-size="12" fill="#dc2626">red = recovered</text>\n'
        '</svg>\n'
    )
    destination.write_text(svg, encoding="utf-8")
