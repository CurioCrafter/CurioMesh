"""Draw docs/stars.svg, stars over time, from GitHub's star dates. Needs the gh CLI. Run before a release:
python scripts/star_chart.py
"""

import datetime as dt
import json
import subprocess
from pathlib import Path

REPO = "CurioCrafter/CurioMesh"
OUT = Path(__file__).resolve().parents[1] / "docs" / "stars.svg"
W, H, L, R, T, B = 720, 300, 56, 24, 40, 44


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
    x = lambda d: L + (W - L - R) * (d - start).total_seconds() / span  # noqa: E731
    y = lambda n: H - B - (H - T - B) * n / top  # noqa: E731
    pts = [(x(start), y(0))]
    for i, d in enumerate(dates, 1):
        pts += [(x(d), y(i - 1)), (x(d), y(i))]
    pts.append((x(today), y(len(dates))))
    line = " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
    area = f"{L},{y(0):.1f} {line} {x(today):.1f},{y(0):.1f}"
    grid = "".join(
        f'<line x1="{L}" x2="{W - R}" y1="{y(n):.1f}" y2="{y(n):.1f}" class="g"/><text x="{L - 8}" y="{y(n) + 4:.1f}" text-anchor="end">{n}</text>'
        for n in range(0, top + 1, max(1, top // 5))
    )
    ticks = "".join(
        f'<text x="{x(d):.1f}" y="{H - B + 20}" text-anchor="middle">{d:%b %Y}</text>'
        for d in (start + (today - start) * k / 4 for k in range(5))
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Segoe UI,Helvetica,Arial,sans-serif" font-size="12">
<style>text{{fill:#768390}} .g{{stroke:#768390;stroke-opacity:.25}} .t{{fill:#e3b341;font-size:15px;font-weight:600}}</style>
<text x="{L}" y="24" class="t">★ {len(dates)} stars on GitHub · {REPO}</text>
{grid}{ticks}
<polygon points="{area}" fill="#e3b341" fill-opacity=".18"/>
<polyline points="{line}" fill="none" stroke="#e3b341" stroke-width="2.5" stroke-linejoin="round"/>
<text x="{W - R}" y="{H - 6}" text-anchor="end" font-size="10">updated {today:%Y-%m-%d}</text>
</svg>
"""


if __name__ == "__main__":
    dates = star_dates()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(svg(dates, dt.datetime.now(dt.timezone.utc)), encoding="utf-8")
    print(json.dumps({"stars": len(dates), "out": str(OUT)}))
