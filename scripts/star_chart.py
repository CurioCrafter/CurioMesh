# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025-2026 Andrew Rainsberger and CurioMesh contributors
"""Draw docs/stars.svg from GitHub's star dates: stars over time, drawn as a quad mesh whose top edge loop is the
star count. Needs the gh CLI. Run before a release:
python scripts/star_chart.py
"""

import datetime as dt
import json
import subprocess
from pathlib import Path

REPO = "CurioCrafter/CurioMesh"
OUT = Path(__file__).resolve().parents[1] / "docs" / "stars.svg"
W, H, L, R, T, B = 760, 320, 52, 28, 52, 40
COLS, ROWS = 40, 6  # quads across, and edge loops under the curve


def star_dates():
    raw = subprocess.run(
        ["gh", "api", "--paginate", "-H", "Accept: application/vnd.github.star+json", f"repos/{REPO}/stargazers", "--jq", ".[].starred_at"],
        check=True, capture_output=True, text=True,
    ).stdout.split()
    return sorted(dt.datetime.fromisoformat(s.replace("Z", "+00:00")) for s in raw)


def svg(dates, today):
    start = min(dates[0], today - dt.timedelta(days=1)) if dates else today - dt.timedelta(days=30)
    span = max((today - start).total_seconds(), 1)
    top = max(5, -(-len(dates) // 5) * 5)  # the next multiple of 5
    px = lambda t: L + (W - L - R) * t  # noqa: E731
    py = lambda n: H - B - (H - T - B) * n / top  # noqa: E731
    # Stars at each column of vertices: the top edge loop.
    times = [start + dt.timedelta(seconds=span * i / COLS) for i in range(COLS + 1)]
    count = [sum(d <= t for d in dates) for t in times]
    count[-1] = len(dates)
    vx = [px(i / COLS) for i in range(COLS + 1)]
    vy = [[py(count[i] * j / ROWS) for i in range(COLS + 1)] for j in range(ROWS + 1)]  # loops follow the curve

    faces = []
    for j in range(ROWS):
        for i in range(COLS):
            if count[i] == 0 and count[i + 1] == 0:
                continue
            quad = f"{vx[i]:.1f},{vy[j][i]:.1f} {vx[i + 1]:.1f},{vy[j][i + 1]:.1f} {vx[i + 1]:.1f},{vy[j + 1][i + 1]:.1f} {vx[i]:.1f},{vy[j + 1][i]:.1f}"
            faces.append(f'<polygon points="{quad}" class="{"f1" if (i + j) % 2 else "f0"}"/>')
    loops = "".join(
        f'<polyline points="{" ".join(f"{vx[i]:.1f},{vy[j][i]:.1f}" for i in range(COLS + 1))}" class="{"top" if j == ROWS else "e"}"/>'
        for j in range(ROWS + 1)
    )
    rings = "".join(f'<line x1="{vx[i]:.1f}" y1="{vy[0][i]:.1f}" x2="{vx[i]:.1f}" y2="{vy[ROWS][i]:.1f}" class="e"/>' for i in range(COLS + 1) if count[i])
    verts = "".join(
        f'<circle cx="{vx[i]:.1f}" cy="{vy[j][i]:.1f}" r="{2.6 if j == ROWS else 1.3}" class="{"vt" if j == ROWS else "v"}"/>'
        for j in range(ROWS + 1) for i in range(COLS + 1) if count[i] or j == 0
    )
    grid = "".join(
        f'<line x1="{L}" x2="{W - R}" y1="{py(n):.1f}" y2="{py(n):.1f}" class="g"/><text x="{L - 8}" y="{py(n) + 4:.1f}" text-anchor="end">{n}</text>'
        for n in range(0, top + 1, max(1, top // 5))
    )
    ticks = "".join(f'<text x="{px(k / 4):.1f}" y="{H - B + 22}" text-anchor="middle">{start + (today - start) * k / 4:%b %Y}</text>' for k in range(5))
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Segoe UI,Helvetica,Arial,sans-serif" font-size="12">
<style>
text{{fill:#768390}} .g{{stroke:#768390;stroke-opacity:.18;stroke-dasharray:2 4}}
.f0{{fill:#f5a623;fill-opacity:.10}} .f1{{fill:#f5a623;fill-opacity:.20}}
.e{{fill:none;stroke:#768390;stroke-opacity:.55;stroke-width:.8}} .v{{fill:#768390}}
.top{{fill:none;stroke:#f5a623;stroke-width:2.6;stroke-linejoin:round}} .vt{{fill:#f5a623}}
.t{{fill:#f5a623;font-size:17px;font-weight:700}} .s{{font-size:12px}}
</style>
<text x="{L}" y="26" class="t">★ {len(dates)} stars</text>
<text x="{L}" y="42" class="s">{REPO} on GitHub, drawn as quads: the top edge loop is the star count</text>
{grid}{"".join(faces)}{rings}{loops}{verts}{ticks}
<text x="{W - R}" y="{H - 4}" text-anchor="end" font-size="10">updated {today:%Y-%m-%d}</text>
</svg>
"""


if __name__ == "__main__":
    dates = star_dates()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(svg(dates, dt.datetime.now(dt.timezone.utc)), encoding="utf-8")
    print(json.dumps({"stars": len(dates), "out": str(OUT)}))
