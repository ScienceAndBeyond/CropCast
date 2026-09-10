"""Create SVG figures for the irrigation contrast."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "irrigation_results"
FIGURES = ROOT / "irrigation_weather_yield" / "figures"

BLUE = "#0072B2"
ORANGE = "#D55E00"
INK = "#222222"
GRID = "#D9D9D9"
BG = "#FFFFFF"


def scale(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if hi == lo:
        return (out_lo + out_hi) / 2
    return out_lo + (value - lo) * (out_hi - out_lo) / (hi - lo)


def svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "middle",
             weight: str = "400", rotate: int | None = None) -> str:
    transform = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{INK}"{transform}>{escape(text)}</text>'
    )


def line(x1: float, y1: float, x2: float, y2: float, color: str = GRID, width: float = 1) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"/>'


def axis_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    return [lo + i * (hi - lo) / (n - 1) for i in range(n)]


def panel_axes(x0: float, y0: float, w: float, h: float, xlim: tuple[float, float],
               ylim: tuple[float, float], xlabel: str, ylabel: str | None = None,
               xticks: list[float] | None = None, yticks: list[float] | None = None,
               xfmt: str = "{:.1f}", yfmt: str = "{:.0f}") -> list[str]:
    parts = [
        f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{BG}" stroke="{GRID}"/>'
    ]
    for t in xticks if xticks is not None else axis_ticks(*xlim):
        x = scale(t, xlim[0], xlim[1], x0, x0 + w)
        parts.append(line(x, y0, x, y0 + h, GRID, 0.8))
        parts.append(svg_text(x, y0 + h + 18, xfmt.format(t), 10))
    for t in yticks if yticks is not None else axis_ticks(*ylim):
        y = scale(t, ylim[0], ylim[1], y0 + h, y0)
        parts.append(line(x0, y, x0 + w, y, GRID, 0.8))
        parts.append(svg_text(x0 - 8, y + 4, yfmt.format(t), 10, anchor="end"))
    parts.append(line(x0, y0 + h, x0 + w, y0 + h, INK, 1.2))
    parts.append(line(x0, y0, x0, y0 + h, INK, 1.2))
    parts.append(svg_text(x0 + w / 2, y0 + h + 42, xlabel, 12))
    if ylabel:
        parts.append(svg_text(x0 - 52, y0 + h / 2, ylabel, 12, rotate=-90))
    return parts


def fig1_prcp_scatter() -> None:
    df = pd.read_csv(RESULTS / "irrigation_pairs.csv")
    x = df["PRCP_a"]
    ys = {"Non-irrigated": df["non_irrigated_a"], "Irrigated": df["irrigated_a"]}
    xlim = (float(x.min()) - 0.1, float(x.max()) + 0.1)
    all_y = pd.concat([ys["Non-irrigated"], ys["Irrigated"]])
    ylim = (float(all_y.min()) - 8, float(all_y.max()) + 8)

    width, height = 980, 430
    panel_w, panel_h = 380, 275
    y0 = 50
    panels = [("Non-irrigated", 92, BLUE), ("Irrigated", 555, ORANGE)]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{BG}"/>',
    ]

    for label, x0, color in panels:
        parts.extend(panel_axes(x0, y0, panel_w, panel_h, xlim, ylim, "PRCP anomaly (mm/day)", "Yield anomaly (BU/AC)" if label == "Non-irrigated" else None))
        parts.append(svg_text(x0 + panel_w / 2, y0 - 18, label, 14, weight="700"))
        y = ys[label]
        for xv, yv in zip(x, y):
            cx = scale(float(xv), *xlim, x0, x0 + panel_w)
            cy = scale(float(yv), *ylim, y0 + panel_h, y0)
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.1" fill="{color}" opacity="0.26"/>')

        binned = pd.DataFrame({"x": x, "y": y})
        binned["bin"] = pd.qcut(binned["x"], 10, duplicates="drop")
        means = binned.groupby("bin", observed=True).agg(x=("x", "mean"), y=("y", "mean")).reset_index()
        pts = [
            (
                scale(float(row.x), *xlim, x0, x0 + panel_w),
                scale(float(row.y), *ylim, y0 + panel_h, y0),
            )
            for row in means.itertuples()
        ]
        path = " ".join(("M" if i == 0 else "L") + f" {px:.1f} {py:.1f}" for i, (px, py) in enumerate(pts))
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3"/>')
        for px, py in pts:
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.2" fill="{color}" stroke="{BG}" stroke-width="1"/>')

    parts.append(svg_text(width / 2, height - 22, "Points are paired county-years; lines connect 10-bin averages.", 11))
    parts.append("</svg>")
    (FIGURES / "figure1_prcp_yield_anomalies.svg").write_text("\n".join(parts) + "\n", encoding="utf-8")


def fig2_leave_one_year_out() -> None:
    df = pd.read_csv(RESULTS / "irrigation_leave_one_year_out.csv")
    df = df[(df["trend"] == "linear") & (df["dropped_year"] != "none")].copy()
    df["dropped_year"] = df["dropped_year"].astype(int)

    width, height = 820, 420
    x0, y0, w, h = 86, 42, 670, 275
    xlim = (2007.5, 2018.5)
    ylim = (0.0, max(0.65, float(df["r2_gap"].max()) + 0.05))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{BG}"/>',
    ]
    parts.extend(panel_axes(
        x0,
        y0,
        w,
        h,
        xlim,
        ylim,
        "Dropped year",
        "R² gap",
        xticks=list(range(2008, 2019, 2)),
        yticks=[0.0, 0.2, 0.4, 0.6],
        xfmt="{:.0f}",
        yfmt="{:.1f}",
    ))
    pts = []
    for row in df.sort_values("dropped_year").itertuples():
        px = scale(row.dropped_year, *xlim, x0, x0 + w)
        py = scale(row.r2_gap, *ylim, y0 + h, y0)
        pts.append((px, py, row))
    path = " ".join(("M" if i == 0 else "L") + f" {px:.1f} {py:.1f}" for i, (px, py, _) in enumerate(pts))
    parts.append(f'<path d="{path}" fill="none" stroke="{INK}" stroke-width="2.2"/>')
    for px, py, row in pts:
        color = ORANGE if row.dropped_year == 2012 else BLUE
        radius = 7 if row.dropped_year == 2012 else 4.8
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius}" fill="{color}" stroke="{BG}" stroke-width="1.4"/>')
        if row.dropped_year == 2012:
            parts.append(svg_text(px + 46, py - 10, "2012", 12, anchor="start", weight="700"))
            parts.append(svg_text(px + 46, py + 7, f"gap {row.r2_gap:.3f}", 11, anchor="start"))
    parts.append(svg_text(width / 2, height - 22, "Gap is non-irrigated R² minus irrigated R²; anomalies and trends are recomputed after each drop.", 11))
    parts.append("</svg>")
    (FIGURES / "figure2_leave_one_year_out_gap.svg").write_text("\n".join(parts) + "\n", encoding="utf-8")


def fig3_state_r2() -> None:
    df = pd.read_csv(RESULTS / "irrigation_by_state.csv")
    df = df[df["trend"] == "linear"].copy()

    width, height = 760, 420
    x0, y0, w, h = 90, 42, 560, 275
    ylim = (0.0, 0.8)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{BG}"/>',
    ]
    parts.extend(panel_axes(
        x0,
        y0,
        w,
        h,
        (0, 1),
        ylim,
        "",
        "Weather R²",
        xticks=[],
        yticks=[0.0, 0.2, 0.4, 0.6, 0.8],
        yfmt="{:.1f}",
    ))

    groups = list(df["state"])
    group_x = [x0 + w * 0.32, x0 + w * 0.68]
    bar_w = 54
    for gx, state in zip(group_x, groups):
        row = df[df["state"] == state].iloc[0]
        vals = [("Non-irrigated", row["r2_non_irrigated"], BLUE, -bar_w / 2 - 5),
                ("Irrigated", row["r2_irrigated"], ORANGE, bar_w / 2 + 5)]
        for label, val, color, dx in vals:
            bh = scale(0, *ylim, y0 + h, y0) - scale(float(val), *ylim, y0 + h, y0)
            bx = gx + dx - bar_w / 2
            by = y0 + h - bh
            parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w}" height="{bh:.1f}" fill="{color}"/>')
            parts.append(svg_text(bx + bar_w / 2, by - 8, f"{val:.3f}", 11, weight="700"))
        parts.append(svg_text(gx, y0 + h + 34, f"{state} ({int(row['n_pairs'])} pairs)", 12, weight="700"))

    lx, ly = x0 + w - 120, y0 + 14
    parts.append(f'<rect x="{lx}" y="{ly}" width="12" height="12" fill="{BLUE}"/>')
    parts.append(svg_text(lx + 18, ly + 11, "Non-irrigated", 11, anchor="start"))
    parts.append(f'<rect x="{lx}" y="{ly + 22}" width="12" height="12" fill="{ORANGE}"/>')
    parts.append(svg_text(lx + 18, ly + 33, "Irrigated", 11, anchor="start"))
    parts.append(svg_text(width / 2, height - 22, "After removing the year trend; weather predictors fitted jointly.", 11))
    parts.append("</svg>")
    (FIGURES / "figure3_state_weather_r2.svg").write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig1_prcp_scatter()
    fig2_leave_one_year_out()
    fig3_state_r2()


if __name__ == "__main__":
    main()
