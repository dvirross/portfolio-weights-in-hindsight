"""Publication figures for "The Best Portfolio, Had We Known the Future".

Reads the tables in ``results/`` written by scripts/run_analysis.py and writes
  * Plotly specs   public/charts/portfolio-hindsight/NN-name.json   ({data, layout, narrow}, same format as the Monte Carlo note)
  * WebP fallbacks public/images/writing/portfolio-hindsight/NN-name.webp   (2200 px wide, lossless)
  * copies (PNG and WebP) in research/backtest-to-strategy/figures/, and CAPTIONS.md there.

Static export uses kaleido; it needs a Chromium: set BROWSER_PATH (kaleido 1.x), e.g.
  BROWSER_PATH=/opt/pw-browsers/chromium-1243/chrome-linux64/chrome python3 scripts/make_figures.py
Every plotted or quoted number is read from results/ or recomputed from the raw data; nothing is typed in except
the labels of the episodes (dates of the crashes) and the design constants.
Usage: python3 scripts/make_figures.py [--no-static]
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))




RES = ROOT / "results"
FIGS = ROOT / "figures"
# Inside the dvirross.com repository the site copies go to its public/ folder; elsewhere (the companion repository)
# set SITE_DIR to a site checkout, or the copies are written under figures/site/.
import os  # noqa: E402
_site = Path(os.environ["SITE_DIR"]) if os.environ.get("SITE_DIR") else (REPO if (REPO / "astro.config.mjs").exists() else FIGS / "site")
CHARTS = _site / "public/charts/portfolio-hindsight"
IMAGES = _site / "public/images/writing/portfolio-hindsight"
NO_STATIC = "--no-static" in sys.argv

# ---- design constants shared with the Monte Carlo figures -------------------------------------------------------
BLUE, GOLD, TEAL, PURPLE, ORANGE = "#3758d7", "#b8860b", "#0e9488", "#7a4fb3", "#c96f33"
GREY, INK, MUTE, GRID, AXLINE = "#b9bfca", "#142033", "#5d6678", "#eceef3", "#c9ced8"
FONT = "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
W = 1100          # CSS width of the static export; scale 2 gives 2200 px

CAPTIONS: list[str] = []


# ---- small helpers ---------------------------------------------------------------------------------------------
def sig(x, n=4):
    """Round to n significant digits (keeps the JSON lean)."""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    x = float(x)
    return 0.0 if x == 0 else float(f"{x:.{n}g}")


def sigs(a, n=4):
    return [sig(v, n) for v in a]


def dstr(d) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def fdate(d, year=True) -> str:
    t = pd.Timestamp(d)
    return f"{t.day} {t.strftime('%b')} {t.year}" if year else f"{t.day} {t.strftime('%b')}"


def pct(x, d=0) -> str:
    return f"{100 * x:.{d}f}%"


def downsample(x, y, n=2000):
    """Keep at most ~n points: first, last and the min and max of every bucket (so extremes survive)."""
    x, y = list(x), np.asarray(y, float)
    if len(y) <= n:
        return x, y
    nb = n // 2 - 1
    edges = np.linspace(0, len(y), nb + 1).astype(int)
    keep = {0, len(y) - 1}
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            seg = y[a:b]
            keep.add(a + int(np.argmin(seg)))
            keep.add(a + int(np.argmax(seg)))
    idx = sorted(keep)
    return [x[i] for i in idx], y[idx]


def title(head: str, sub: str, size_head=16, size_sub=13) -> dict:
    return {"text": f"<b>{head}</b><br><span style='font-size:{size_sub}px;color:{MUTE}'>{sub}</span>",
            "x": 0, "xref": "paper", "xanchor": "left", "y": 1, "yref": "container", "yanchor": "top",
            "pad": {"t": 14}, "font": {"size": size_head, "color": INK}}


def ax(**kw) -> dict:
    d = {"tickfont": {"color": MUTE, "size": 12}, "gridcolor": GRID, "gridwidth": 1, "linecolor": AXLINE,
         "zeroline": False, "ticks": "", "showline": True}
    d.update(kw)
    return d


def ax_title(text: str, size=12) -> dict:
    return {"text": text, "font": {"size": size, "color": MUTE}}


def base_layout(head, sub, height, margin, showlegend=False, narrow=False, **extra) -> dict:
    lay = {
        "font": {"family": FONT, "size": 13, "color": INK},
        "title": title(head, sub, 14 if narrow else 16, 11 if narrow else 13),
        "margin": margin,
        "showlegend": showlegend,
        "legend": {"font": {"size": 12, "color": INK}, "orientation": "h", "x": 0, "xanchor": "left", "y": 1.0,
                   "yanchor": "bottom", "bgcolor": "rgba(255,255,255,0)", "itemwidth": 30},
        "hoverlabel": {"font": {"family": FONT, "color": INK, "size": 12}, "bgcolor": "white", "bordercolor": AXLINE},
        "paper_bgcolor": "white", "plot_bgcolor": "white", "height": height, "hovermode": "closest",
        "annotations": [],
    }
    lay.update(extra)
    return lay


def _lg(v, on):
    return math.log10(v) if on else v


def note(text, x, y, xref="x", yref="y", anchor="left", yanchor="middle", color=INK, size=12, xlog=False, ylog=False, **kw) -> dict:
    x, y = _lg(x, xlog), _lg(y, ylog)
    d = {"x": x, "y": y, "xref": xref, "yref": yref, "text": text, "showarrow": False, "xanchor": anchor,
         "yanchor": yanchor, "align": "left" if anchor == "left" else ("right" if anchor == "right" else "center"),
         "font": {"size": size, "color": color}}
    d.update(kw)
    return d


def arrow(text, x, y, ax_, ay, xref="x", yref="y", axref="pixel", ayref="pixel", anchor="left", color=INK,
          size=12, xlog=False, ylog=False, **kw) -> dict:
    x, y = _lg(x, xlog), _lg(y, ylog)
    d = {"x": x, "y": y, "xref": xref, "yref": yref, "ax": ax_, "ay": ay, "axref": axref, "ayref": ayref,
         "showarrow": True, "arrowhead": 0, "arrowwidth": 1, "arrowcolor": AXLINE, "standoff": 4, "text": text,
         "xanchor": anchor, "yanchor": "middle", "align": "left" if anchor == "left" else "center",
         "font": {"size": size, "color": color}}
    d.update(kw)
    return d


def hline(y, color=INK, dash="dot", width=1, xref="x", yref="y", x0=0, x1=1, xtype="paper") -> dict:
    return {"type": "line", "xref": "paper" if xtype == "paper" else xref, "yref": yref, "x0": x0, "x1": x1,
            "y0": y, "y1": y, "line": {"color": color, "width": width, "dash": dash}}


def vline(x, color=INK, dash="dot", width=1, xref="x", y0=0, y1=1, yref="paper") -> dict:
    return {"type": "line", "xref": xref, "yref": yref, "x0": x, "x1": x, "y0": y0, "y1": y1,
            "line": {"color": color, "width": width, "dash": dash}}


def narrow_part(wide: dict, narrow: dict, max_width=560) -> dict:
    """Top-level layout keys that differ in the narrow layout (the page replaces keys wholesale)."""
    diff = {k: v for k, v in narrow.items() if wide.get(k) != v and k != "height"}
    return {"maxWidth": max_width, "layout": diff}


def finish(name, data, wide, narrow, ratio_h, caption, alt, source):
    wide = copy.deepcopy(wide)
    wide["height"] = ratio_h // 2          # CSS px; the static export is 2x
    spec = {"data": data, "layout": wide, "narrow": narrow_part(wide, narrow)}
    CHARTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    path = CHARTS / f"{name}.json"
    path.write_text(json.dumps(spec, separators=(",", ":"), ensure_ascii=False))
    status = "json only"
    if not NO_STATIC:
        try:
            import plotly.io as pio
            from PIL import Image
            png = FIGS / f"{name}.png"
            pio.write_image({"data": data, "layout": wide}, str(png), width=W, height=ratio_h // 2, scale=2)
            im = Image.open(png).convert("RGB")
            assert im.size == (2200, ratio_h), im.size
            im.save(FIGS / f"{name}.webp", "WEBP", lossless=True, quality=100, method=6)
            im.save(IMAGES / f"{name}.webp", "WEBP", lossless=True, quality=100, method=6)
            status = f"{im.size[0]}x{im.size[1]} webp {os.path.getsize(IMAGES / f'{name}.webp') // 1024} KB"
            prev = os.environ.get("NARROW_PREVIEW_DIR")
            if prev:   # 360 px preview of the phone layout for visual checks (not shipped)
                nl = copy.deepcopy(wide)
                nl.update(narrow)
                nl["height"] = narrow.get("height", 700)
                pio.write_image({"data": data, "layout": nl}, str(Path(prev) / f"{name}-narrow.png"), width=360, height=nl["height"], scale=2)
        except Exception as e:  # noqa: BLE001
            status = f"STATIC EXPORT FAILED: {e!r}"
    print(f"{name}: json {path.stat().st_size // 1024} KB; {status}")
    CAPTIONS.append((name, caption, alt, source, ratio_h))


def finish_wrap(name, data, wide, narrow_layout, ratio_h, caption, alt, source, preview_narrow=True):
    """finish() plus a 360 px preview PNG of the narrow layout in scratch (checked by eye, not shipped)."""
    finish(name, data, wide, narrow_layout, ratio_h, caption, alt, source)
# ---- data -----------------------------------------------------------------------------------------------------
SRC = "Kenneth R. French Data Library, 10 value-weighted US industry portfolios, daily returns 1926–2026 (including dividends)."
SRC3 = SRC + " 3× series built from them: 3 × daily return − 2 × risk-free rate − 0.95% a year, reset daily."
NAMES = {"NoDur": "Consumer staples", "Durbl": "Consumer durables", "Manuf": "Manufacturing", "Enrgy": "Energy",
         "HiTec": "Technology", "Telcm": "Telecom", "Shops": "Retail", "Hlth": "Health", "Utils": "Utilities",
         "Other": "Other"}
COLS = list(NAMES)
PAL10 = ["#4e79a7", "#f28e2b", "#9c755f", "#e15759", "#59a14f", "#76b7b2", "#edc948", "#b07aa1", "#ff9da7", "#bab0ac"]
OBJ = {"end wealth": ("Maximize end wealth", ORANGE),
       "time-average wealth (notebook default)": ("Maximize average wealth (the 2025 notebook)", PURPLE),
       "Sharpe ratio": ("Maximize Sharpe ratio", TEAL),
       "minimum variance": ("Minimize variance", BLUE),
       "equal weights": ("Equal weights, no optimization", INK),
       "half end wealth, half equal": ("Half optimized, half equal", GOLD)}


def P(x, d=1):
    return f"{100 * x:.{d}f}%".replace("-", "−")


# =================================================================================================================
# Figure 1: the hill with two assets
# =================================================================================================================
def fig1():
    h = pd.read_csv(RES / "hill_pairs.csv")
    pairs = [("Hlth-Enrgy", "Health", "Energy"), ("HiTec-Utils", "Technology", "Utilities")]
    data, ann = [], []
    for i, (key, a, b) in enumerate(pairs):
        g = h[h.pair == key]
        xa, ya = ("x" if i == 0 else "x2"), ("y" if i == 0 else "y2")
        top = g.loc[g.growth.idxmax()]
        data += [
            {"type": "scatter", "mode": "lines", "name": "Rebalanced every month", "legendgroup": "r", "showlegend": i == 0,
             "x": sigs(100 * g.w_first), "y": sigs(100 * g.growth), "xaxis": xa, "yaxis": ya, "line": {"color": BLUE, "width": 2.5},
             "hovertemplate": f"{a} %{{x:.0f}}%, {b} %{{customdata:.0f}}%<br>growth %{{y:.2f}}% a year<extra>rebalanced</extra>",
             "customdata": sigs(100 - 100 * g.w_first)},
            {"type": "scatter", "mode": "lines", "name": "Same starting weights, never rebalanced", "legendgroup": "n", "showlegend": i == 0,
             "x": sigs(100 * g.w_first), "y": sigs(100 * g.growth_never_rebalanced), "xaxis": xa, "yaxis": ya,
             "line": {"color": GREY, "width": 2, "dash": "dash"},
             "hovertemplate": f"start {a} %{{x:.0f}}%<br>growth %{{y:.2f}}% a year<extra>never rebalanced</extra>"},
            {"type": "scatter", "mode": "markers", "showlegend": False, "x": [sig(100 * top.w_first)], "y": [sig(100 * top.growth)],
             "xaxis": xa, "yaxis": ya, "marker": {"size": 10, "color": BLUE, "line": {"color": "white", "width": 2}},
             "hovertemplate": f"peak: {a} %{{x:.0f}}%, growth %{{y:.2f}}%<extra></extra>"},
        ]
    h1 = h[h.pair == "Hlth-Enrgy"]
    t1 = h1.loc[h1.growth.idxmax()]
    h2 = h[h.pair == "HiTec-Utils"]

    def lay(narrow):
        ann = [
            note(f"Peak inside: {t1.w_first:.0%} Health,<br>{P(t1.growth, 2)} a year, above either industry alone", 100 * t1.w_first, 100 * t1.growth,
                 anchor="right", yanchor="bottom", color=BLUE, xshift=-6, yshift=8, size=11),
            note(f"Peak at the edge:<br>all Technology, {P(h2.growth.iloc[-1], 2)}", 100, 10.0, xref="x2", yref="y2",
                 anchor="right", yanchor="middle", color=BLUE, size=11),
            note("<b>Health and Energy</b>", 0, 1.0, xref="x domain", yref="y domain", anchor="left", yanchor="bottom", yshift=4),
            note("<b>Technology and Utilities</b>", 0, 1.0, xref="x2 domain", yref="y2 domain", anchor="left", yanchor="bottom", yshift=4),
        ]
        if narrow:
            ann[0] = note(f"Peak: {t1.w_first:.0%} Health,<br>{P(t1.growth, 2)}", 100 * t1.w_first, 100 * t1.growth,
                          anchor="center", yanchor="top", color=BLUE, yshift=-10, size=10)
            ann[1] = note(f"Peak at the edge:<br>{P(h2.growth.iloc[-1], 2)}", 100, 10.0, xref="x2", yref="y2", anchor="right", color=BLUE, size=10)
        dom = ([0, 0.46], [0.54, 1]) if not narrow else ([0, 1], [0, 1])
        ydom = ([0, 1], [0, 1]) if not narrow else ([0.58, 1], [0, 0.42])
        return base_layout("Rebalancing to fixed weights: one hill per pair, sometimes with the peak inside" if not narrow else
                           "Rebalancing to fixed weights:<br>one hill per pair",
                           "Average yearly growth 1927–2026 for every split between two industries, rebalanced every month" if not narrow else
                           "Average yearly growth 1927–2026,<br>rebalanced every month",
                           600, {"l": 60, "r": 24, "t": 112, "b": 60} if not narrow else {"l": 50, "r": 14, "t": 150, "b": 56},
                           showlegend=True, narrow=narrow,
                           xaxis=ax(domain=dom[0], range=[-2, 102], ticksuffix="%", title=ax_title("Weight in Health")),
                           xaxis2=ax(domain=dom[1], range=[-2, 102], ticksuffix="%", anchor="y2", title=ax_title("Weight in Technology")),
                           yaxis=ax(domain=ydom[0], ticksuffix="%", title=ax_title("Growth per year" if not narrow else ""), range=[10.4, 11.9]),
                           yaxis2=ax(domain=ydom[1], ticksuffix="%", anchor="x2", range=[8.8, 11.9]),
                           annotations=ann,
                           legend={"font": {"size": 11 if narrow else 12, "color": INK}, "orientation": "h", "x": 0, "xanchor": "left",
                                   "y": 1.0, "yanchor": "bottom", "yref": "container" if False else "paper", "bgcolor": "rgba(255,255,255,0)"} | ({"y": 1.07} if not narrow else {"y": 1.02}))
    wide, nar = lay(False), lay(True)
    nar["height"] = 900
    finish_wrap("01-two-asset-hill", data, wide, nar, 1200,
                f"With monthly rebalancing, every split between two industries has one growth rate, and the curve has a single peak. For Health and Energy the peak is inside, at {t1.w_first:.0%} Health, and the mixture grows faster ({P(t1.growth, 2)} a year) than either industry alone ({P(h1.growth.iloc[-1], 2)} and {P(h1.growth.iloc[0], 2)}). For Technology and Utilities the peak is at the edge, all Technology.",
                f"Two line charts of average yearly growth from 1927 to 2026 against the weight of one industry, from 0% to 100%. Left, Health and Energy: the rebalanced curve rises from {P(h1.growth.iloc[0], 2)} at 0% Health to a peak of {P(t1.growth, 2)} at {t1.w_first:.0%} and ends at {P(h1.growth.iloc[-1], 2)} at 100%; the never-rebalanced curve lies below it in the middle. Right, Technology and Utilities: the curve rises from {P(h2.growth.iloc[0], 2)} to {P(h2.growth.iloc[-1], 2)} with its peak at 100% Technology.",
                SRC + " Monthly returns, no costs.")


# =================================================================================================================
# Figure 2: the hill with three assets, seen from above, and the notebook's search paths
# =================================================================================================================
def tri_xy(w_e, w_u):
    return w_e + w_u / 2, w_u * math.sqrt(3) / 2


def fig2():
    G = pd.read_csv(RES / "simplex_grid.csv")
    Pth = pd.read_csv(RES / "simplex_paths.csv")
    x, y = tri_xy(G.w_Enrgy.to_numpy(), G.w_Utils.to_numpy())
    top = G.loc[G.growth.idxmax()]
    data = [{"type": "scatter", "mode": "markers", "x": sigs(x), "y": sigs(y), "showlegend": False,
             "marker": {"symbol": "hexagon", "size": 9, "color": sigs(100 * G.growth), "colorscale": "Viridis", "line": {"width": 0},
                        "colorbar": {"title": {"text": "Growth<br>per year", "font": {"size": 11, "color": MUTE}}, "ticksuffix": "%",
                                     "thickness": 12, "len": 0.7, "tickfont": {"size": 11, "color": MUTE}}},
             "customdata": np.round(100 * G[["w_Hlth", "w_Enrgy", "w_Utils"]].to_numpy()).astype(int).tolist(),
             "hovertemplate": "Health %{customdata[0]}%, Energy %{customdata[1]}%, Utilities %{customdata[2]}%<br>growth %{marker.color:.2f}% a year<extra></extra>"}]
    starts = {0: "start: equal weights", 1: "start: 80% Health", 2: "start: 80% Utilities"}
    for s in (0, 1, 2):
        g = Pth[(Pth.kind == "simple") & (Pth["mode"] == "random 1 in 10") & (Pth.start == s)]
        px, py = tri_xy(g.w_Enrgy.to_numpy(), g.w_Utils.to_numpy())
        data.append({"type": "scatter", "mode": "lines", "x": sigs(px), "y": sigs(py), "name": "Search path (adds 1% to one or two weights; a random step 1 time in 10)" if s == 0 else None,
                     "showlegend": s == 0, "legendgroup": "p", "line": {"color": INK, "width": 2},
                     "hovertemplate": f"{starts[s]}<extra></extra>"})
        data.append({"type": "scatter", "mode": "markers", "x": [sig(px[0])], "y": [sig(py[0])], "showlegend": False,
                     "marker": {"size": 8, "color": "white", "line": {"color": INK, "width": 1.5}}, "hovertemplate": f"{starts[s]}<extra></extra>"})
    tx, ty = tri_xy(top.w_Enrgy, top.w_Utils)
    data.append({"type": "scatter", "mode": "markers", "x": [sig(tx)], "y": [sig(ty)], "name": "Peak", "showlegend": True,
                 "marker": {"symbol": "star", "size": 16, "color": GOLD, "line": {"color": INK, "width": 1}},
                 "hovertemplate": f"peak: Health {top.w_Hlth:.0%}, Energy {top.w_Enrgy:.0%}, Utilities {top.w_Utils:.0%}<br>{P(top.growth, 2)} a year<extra></extra>"})
    h3 = math.sqrt(3) / 2

    def lay(narrow):
        ann = [note("<b>100% Health</b>", 0, 0, anchor="center", yanchor="top", yshift=-6),
               note("<b>100% Energy</b>", 1, 0, anchor="center", yanchor="top", yshift=-6),
               note("<b>100% Utilities</b>", 0.5, h3, anchor="center", yanchor="bottom", yshift=6),
               note(f"Peak: {top.w_Hlth:.0%} Health,<br>{top.w_Enrgy:.0%} Energy, {P(top.growth, 2)}", tx, ty, anchor="left", yanchor="bottom", xshift=8, yshift=6, size=11, bgcolor="rgba(255,255,255,0.85)")]
        if narrow:
            ann = ann[:3] + [note(f"Peak {top.w_Hlth:.0%} / {top.w_Enrgy:.0%}", tx, ty, anchor="left", yanchor="bottom", xshift=8, yshift=6, size=10,
                                  bgcolor="rgba(255,255,255,0.85)")]
        return base_layout("Climbing a hill we cannot see" if not narrow else "Climbing a hill we cannot see",
                           "Growth per year 1927–2026 for every mix of three industries, rebalanced monthly, and three runs of the 2025 search" if not narrow else
                           "Growth 1927–2026 for every mix of<br>three industries, and three search runs",
                           700, {"l": 30, "r": 30, "t": 110, "b": 40} if not narrow else {"l": 6, "r": 6, "t": 110, "b": 20},
                           showlegend=not narrow, narrow=narrow,
                           xaxis=ax(visible=False, range=[-0.08, 1.08] if not narrow else [-0.2, 1.2], scaleanchor="y", scaleratio=1),
                           yaxis=ax(visible=False, range=[-0.08, h3 + 0.08]),
                           annotations=ann,
                           legend={"font": {"size": 11 if narrow else 12, "color": INK}, "orientation": "h", "x": 0, "xanchor": "left",
                                   "y": 1.0, "yanchor": "bottom", "bgcolor": "rgba(255,255,255,0)"})
    wide, nar = lay(False), lay(True)
    nar["height"] = 640
    finish_wrap("02-three-asset-hill", data, wide, nar, 1400,
                f"Every mixture of Health, Energy and Utilities is a point in the triangle; the colour is its growth per year from 1927 to 2026 with monthly rebalancing. The highest point is at {top.w_Hlth:.0%} Health and {top.w_Enrgy:.0%} Energy ({P(top.growth, 2)} a year). Dark lines: three runs of the search from the 2025 notebook, from three starting points, each step adding 1% to one or two weights, with a random step one time in ten.",
                f"Triangle heat map: corners are 100% Health (bottom left), 100% Energy (bottom right) and 100% Utilities (top). Colours run from dark (slow growth, near the Utilities corner) to yellow (fast growth) along the Health–Energy edge, with a single peak marked by a star at {top.w_Hlth:.0%} Health and {top.w_Enrgy:.0%} Energy. Three dark search paths start at equal weights, at 80% Health and at 80% Utilities and all wander toward the star.",
                SRC + " Monthly returns, no costs; search from bts/original.py with seeds 0–2, 400 steps.")


# =================================================================================================================
# Figure 3: rebalanced versus never rebalanced
# =================================================================================================================
def fig3():
    r = pd.read_csv(RES / "rebalanced_vs_never.csv", index_col=0, parse_dates=True)
    data = []
    for lab, col, name in (("1x", BLUE, "Ten industries, no leverage"), ("3x", ORANGE, "Ten 3× industry funds")):
        ratio = r[f"{lab}_monthly"] / r[f"{lab}_never"]
        data.append({"type": "scatter", "mode": "lines", "name": name, "x": [dstr(d) for d in r.index], "y": sigs(ratio),
                     "line": {"color": col, "width": 2}, "hovertemplate": "%{x|%b %Y}: rebalanced / never = %{y:.2f}<extra>" + name + "</extra>"})
    e1 = (r["1x_monthly"] / r["1x_never"]).iloc[-1]
    e3 = (r["3x_monthly"] / r["3x_never"]).iloc[-1]

    def lay(narrow):
        return base_layout("Rebalancing back to equal weights added up over a century" if not narrow else "Rebalancing added up<br>over a century",
                           "Value of a portfolio rebalanced every month divided by the same portfolio never rebalanced, both starting at equal weights in 1927" if not narrow else
                           "Rebalanced monthly ÷ never rebalanced,<br>both starting at equal weights in 1927",
                           500, {"l": 60, "r": 150, "t": 84, "b": 50} if not narrow else {"l": 46, "r": 14, "t": 150, "b": 46},
                           showlegend=narrow, narrow=narrow,
                           xaxis=ax(range=["1926-06-01", "2027-06-01"]), yaxis=ax(type="log", tickvals=[0.5, 0.7, 1, 1.5, 2, 3, 4, 6], ticktext=["0.5", "0.7", "1", "1.5", "2", "3", "4", "6"],
                                                title=ax_title("Ratio (log scale)" if not narrow else "")),
                           shapes=[hline(1, GREY, "dash")],
                           annotations=[] if narrow else [
                               note(f"No leverage: ×{e1:.2f}", dstr(r.index[-1]), math.log10(e1), anchor="left", color=BLUE, xshift=8),
                               note(f"3× funds: ×{e3:.2f}", dstr(r.index[-1]), math.log10(e3), anchor="left", color=ORANGE, xshift=8)],
                           legend={"font": {"size": 11, "color": INK}, "orientation": "h", "x": 0, "y": 1.0, "yanchor": "bottom"})
    wide, nar = lay(False), lay(True)
    nar["height"] = 560
    finish_wrap("03-rebalanced-vs-never", data, wide, nar, 1000,
                f"Starting from equal weights in the ten industries in 1927, the portfolio rebalanced every month ended with {e1:.2f} times the value of the one never rebalanced; with 3× funds, {e3:.2f} times. With 3× funds the ratio fell below 1 for long stretches, when a few industries kept winning.",
                f"Line chart from 1927 to 2026 on a log scale of the ratio between a monthly rebalanced and a never rebalanced equal-weight portfolio. Without leverage the ratio stays between {(r['1x_monthly'] / r['1x_never']).min():.2f} and {(r['1x_monthly'] / r['1x_never']).max():.2f} and ends at {e1:.2f}; with 3× funds it swings between {(r['3x_monthly'] / r['3x_never']).min():.2f} and {(r['3x_monthly'] / r['3x_never']).max():.2f} and ends at {e3:.2f}.",
                SRC3 + " Daily simulation; rebalancing at month ends, no costs.")


# =================================================================================================================
# Figure 4: the weights the optimizer chose each year (3x, maximize end wealth)
# =================================================================================================================
def fig4():
    ww = pd.read_csv(RES / "walkforward_weights.csv")
    e = ww[(ww.assets == "3x") & (ww.objective == "end wealth")].copy()
    years = [int(s[:4]) for s in e.start]
    data = []
    for c, col in zip(COLS, PAL10):
        data.append({"type": "bar", "name": NAMES[c], "x": years, "y": sigs(100 * e[c], 3), "marker": {"color": col, "line": {"width": 0}},
                     "hovertemplate": f"%{{x}}: {NAMES[c]} %{{y:.0f}}%<extra></extra>"})
    big = e[COLS].max(axis=1)

    def lay(narrow):
        return base_layout("Maximizing past growth puts almost everything on last decade’s winner" if not narrow else
                           "Maximizing past growth bets<br>on last decade’s winner",
                           "Weights chosen each year from the previous ten years by maximizing end wealth, ten 3× industry funds, 1937–2025" if not narrow else
                           "Weights chosen each year from the previous<br>ten years, 3× industry funds, 1937–2025",
                           600, {"l": 56, "r": 20, "t": 130, "b": 46} if not narrow else {"l": 44, "r": 10, "t": 290, "b": 40},
                           showlegend=True, narrow=narrow, barmode="stack", bargap=0.05,
                           xaxis=ax(range=[1936, 2026]), yaxis=ax(range=[0, 100], ticksuffix="%", title=ax_title("Weight" if not narrow else "")),
                           legend={"font": {"size": 11, "color": INK}, "orientation": "h", "x": 0, "xanchor": "left", "y": 1.0, "yanchor": "bottom",
                                   "traceorder": "normal"})
    wide, nar = lay(False), lay(True)
    nar["height"] = 760
    finish_wrap("04-chosen-weights", data, wide, nar, 1200,
                f"Each bar is the portfolio the search chose at the start of a year, using only the ten years before it. On average the largest weight was {big.mean():.0%}, and in {(big > 0.9).mean():.0%} of the years more than 90% went to a single industry.",
                f"Stacked bar chart, one bar per year from 1937 to 2025, showing the weights in ten 3× industry funds chosen by maximizing growth over the previous ten years. Most bars are dominated by one colour that changes every few years; the largest weight averages {big.mean():.0%}.",
                SRC3 + " Exponentiated-gradient ascent on mean log growth; monthly rebalancing.")


# =================================================================================================================
# Figure 5: in-sample growth versus the following year
# =================================================================================================================
def fig5():
    ww = pd.read_csv(RES / "walkforward_weights.csv")
    data = []
    for lab, col, name in (("1x", BLUE, "No leverage"), ("3x", ORANGE, "3× funds")):
        e = ww[(ww.assets == lab) & (ww.objective == "end wealth")]
        data.append({"type": "scatter", "mode": "markers", "name": name, "x": sigs(100 * e.in_sample_growth), "y": sigs(100 * e.next_year),
                     "text": [s[:4] for s in e.start], "marker": {"size": 8, "color": col, "opacity": 0.8, "line": {"color": "white", "width": 1}},
                     "hovertemplate": "%{text}: %{x:.0f}% a year over the previous ten years, %{y:.0f}% in the following year<extra>" + name + "</extra>"})
    e1 = ww[(ww.assets == "1x") & (ww.objective == "end wealth")]
    e3 = ww[(ww.assets == "3x") & (ww.objective == "end wealth")]
    c1, c3 = np.corrcoef(e1.in_sample_growth, e1.next_year)[0, 1], np.corrcoef(e3.in_sample_growth, e3.next_year)[0, 1]

    def lay(narrow):
        return base_layout("How well a portfolio did in the past says little about the next year" if not narrow else
                           "The past ten years say little<br>about the next one",
                           "Each point: weights that maximized growth over ten years (x) and what they returned in the following year (y), 1937–2025" if not narrow else
                           "Each point: growth of the chosen weights over<br>ten years (x) and in the next year (y)",
                           600, {"l": 64, "r": 24, "t": 100, "b": 60} if not narrow else {"l": 50, "r": 12, "t": 160, "b": 56},
                           showlegend=True, narrow=narrow,
                           xaxis=ax(ticksuffix="%", title=ax_title("Growth per year over the previous ten years")),
                           yaxis=ax(ticksuffix="%", title=ax_title("Return in the following year" if not narrow else "")),
                           shapes=[hline(0, GREY, "dash")],
                           annotations=[note(f"Correlation: {c1:.2f} without leverage, {c3:.2f} with 3× funds", 0, 1, xref="paper", yref="paper",
                                             anchor="left", yanchor="top", color=MUTE, size=11)],
                           legend={"font": {"size": 11, "color": INK}, "orientation": "h", "x": 0, "y": 1.0, "yanchor": "bottom"})
    wide, nar = lay(False), lay(True)
    nar["height"] = 640
    finish_wrap("05-past-vs-next-year", data, wide, nar, 1200,
                f"The weights that did best over the previous ten years did not do correspondingly well in the next year: the correlation between the two is {c1:.2f} without leverage and {c3:.2f} with 3× funds.",
                f"Scatter plot with 89 blue points (no leverage) and 89 orange points (3× funds). In-sample growth runs from {P(e1.in_sample_growth.min(), 0)} to {P(e1.in_sample_growth.max(), 0)} a year without leverage and from {P(e3.in_sample_growth.min(), 0)} to {P(e3.in_sample_growth.max(), 0)} with 3× funds; the next year's return runs from {P(e3.next_year.min(), 0)} to {P(e3.next_year.max(), 0)}, with no visible trend. Correlation {c1:.2f} and {c3:.2f}.",
                SRC3 + " Weights from exponentiated-gradient ascent on mean log growth over the previous 120 months.")


# =================================================================================================================
# Figure 6: out-of-sample wealth for each choice of what to maximize (3x)
# =================================================================================================================
def fig6():
    wf = pd.read_csv(RES / "walkforward_returns.csv")
    T = pd.read_csv(RES / "walkforward_summary.csv")
    data = []
    order = ["equal weights", "minimum variance", "half end wealth, half equal", "Sharpe ratio", "end wealth", "time-average wealth (notebook default)"]
    for k in order:
        g = wf[(wf.assets == "3x") & (wf.objective == k)]
        v = np.cumprod(1 + g.ret.to_numpy())
        name, col = OBJ[k]
        gr = float(T[(T.assets == "3x") & (T.objective == k)].growth.iloc[0])
        data.append({"type": "scatter", "mode": "lines", "name": f"{name}: {P(gr)} a year", "x": [m + "-15" for m in g.month], "y": sigs(v),
                     "line": {"color": col, "width": 2.4 if k in ("end wealth", "equal weights") else 1.6},
                     "hovertemplate": "%{x|%b %Y}: ×%{y:,.3s}<extra>" + name + "</extra>"})

    def lay(narrow):
        return base_layout("What we maximize decides what we get" if not narrow else "What we maximize<br>decides what we get",
                           "Value of 1 dollar in ten 3× industry funds, weights chosen each year from the previous ten years and held for one year, 1937–2025" if not narrow else
                           "1 dollar in ten 3× industry funds,<br>weights re-chosen every year, 1937–2025",
                           700, {"l": 64, "r": 24, "t": 150, "b": 46} if not narrow else {"l": 52, "r": 10, "t": 290, "b": 40},
                           showlegend=True, narrow=narrow,
                           xaxis=ax(), yaxis=ax(type="log", dtick=1, exponentformat="power", title=ax_title("Value (log scale)" if not narrow else "")),
                           legend={"font": {"size": 11, "color": INK}, "orientation": "h", "x": 0, "xanchor": "left", "y": 1.0, "yanchor": "bottom"})
    wide, nar = lay(False), lay(True)
    nar["height"] = 820
    t = T[T.assets == "3x"].set_index("objective")
    finish_wrap("06-what-we-maximize", data, wide, nar, 1400,
                f"Out of sample, the portfolio that maximized past growth grew {P(t.loc['end wealth', 'growth'])} a year and fell {P(-t.loc['end wealth', 'max_drawdown'], 0)} at its worst, while equal weights in the same funds grew {P(t.loc['equal weights', 'growth'])} a year and minimum variance {P(t.loc['minimum variance', 'growth'])}.",
                f"Line chart on a log scale of the value of one dollar from 1937 to 2025 for six ways of choosing weights among ten 3× industry funds. Equal weights end highest (about ×{t.loc['equal weights', 'end_wealth']:,.0f}), followed by minimum variance; maximizing end wealth ends near ×{t.loc['end wealth', 'end_wealth']:,.0f} and maximizing average wealth, the 2025 notebook's default, near ×{t.loc['time-average wealth (notebook default)', 'end_wealth']:,.0f}, after falls of nearly 100%.",
                SRC3 + " Monthly rebalancing, no trading costs; weights chosen at the start of each year from the previous 120 months.")


# =================================================================================================================
# Figure 7: rebalancing frequency and costs
# =================================================================================================================
def fig7():
    fr = pd.read_csv(RES / "frequency_costs.csv")
    lab = {"daily": "Daily", "monthly": "Monthly", "quarterly": "Quarterly", "annual": "Yearly", "band 25%": "25% band", "never": "Never"}
    data = []
    for i, a in enumerate(("1x", "3x")):
        for cost, col, nm in ((0.0, BLUE, "No trading cost"), (0.005, GOLD, "0.5% of the amount traded")):
            g = fr[(fr.assets == a) & (fr.cost == cost)]
            data.append({"type": "bar", "name": nm, "legendgroup": nm, "showlegend": i == 0, "x": [lab[s] for s in g.schedule], "y": sigs(100 * g.growth),
                         "xaxis": "x" if i == 0 else "x2", "yaxis": "y" if i == 0 else "y2", "marker": {"color": col},
                         "hovertemplate": "%{x}: %{y:.2f}% a year<extra>" + nm + "</extra>"})

    def lay(narrow):
        dom = ([0, 0.47], [0.53, 1]) if not narrow else ([0, 1], [0, 1])
        ydom = ([0, 1], [0, 1]) if not narrow else ([0.58, 1], [0, 0.42])
        return base_layout("How often to rebalance: less often did better here" if not narrow else "How often to rebalance",
                           "Growth per year 1927–2026 of an equal-weight portfolio of the ten industries, by rebalancing rule" if not narrow else
                           "Equal-weight portfolio of ten industries,<br>growth per year 1927–2026",
                           560, {"l": 60, "r": 20, "t": 120, "b": 70} if not narrow else {"l": 46, "r": 10, "t": 150, "b": 60},
                           showlegend=True, narrow=narrow, barmode="group",
                           xaxis=ax(domain=dom[0], tickangle=0 if not narrow else -40, tickfont={"size": 12 if not narrow else 10, "color": MUTE}),
                           xaxis2=ax(domain=dom[1], anchor="y2", tickangle=0 if not narrow else -40, tickfont={"size": 12 if not narrow else 10, "color": MUTE}),
                           yaxis=ax(domain=ydom[0], ticksuffix="%", range=[10, 11.4]), yaxis2=ax(domain=ydom[1], ticksuffix="%", anchor="x2", range=[14, 19.5]),
                           annotations=[note("<b>No leverage</b>", 0, 1, xref="x domain", yref="y domain", anchor="left", yanchor="bottom", yshift=4),
                                        note("<b>3× funds</b>", 0, 1, xref="x2 domain", yref="y2 domain", anchor="left", yanchor="bottom", yshift=4)],
                           legend={"font": {"size": 11, "color": INK}, "orientation": "h", "x": 0, "y": 1.08 if not narrow else 1.03, "yanchor": "bottom"})
    wide, nar = lay(False), lay(True)
    nar["height"] = 860
    g3 = fr[(fr.assets == "3x") & (fr.cost == 0)].set_index("schedule").growth
    finish_wrap("07-frequency-costs", data, wide, nar, 1120,
                f"Every rebalancing rule beat never rebalancing, but more trading was not better: with 3× funds, yearly rebalancing grew {P(g3['annual'])} a year, monthly {P(g3['monthly'])} and daily {P(g3['daily'])}. Trading costs widen the gap.",
                f"Two grouped bar charts of growth per year from 1927 to 2026 for six rebalancing rules (daily, monthly, quarterly, yearly, a 25% band that rebalances at a month end only when some weight is more than 25% away from its target, never), each with no cost and with a cost of 0.5% of the amount traded. Without leverage all bars lie between about 10.3% and 11.2%; with 3× funds they range from {P(g3['never'])} (never) to {P(g3['annual'])} (yearly).",
                SRC3 + " Daily simulation; costs charged on the value traded at each rebalance.")


# =================================================================================================================
# Figure 8: sudden events
# =================================================================================================================
def fig8():
    ev = pd.read_csv(RES / "events.csv", parse_dates=["date"], dtype={"event": str})
    data = []
    for i, (name, ttl) in enumerate((("1987", "October 1987"), ("2020", "March 2020"))):
        g = ev[ev.event == name]
        for col, c, nm in (("3x_monthly", ORANGE, "3× funds, rebalanced monthly"), ("3x_never", GOLD, "3× funds, never rebalanced"),
                           ("1x_monthly", BLUE, "No leverage, rebalanced monthly")):
            data.append({"type": "scatter", "mode": "lines", "name": nm, "legendgroup": nm, "showlegend": i == 0, "x": [dstr(d) for d in g.date],
                         "y": sigs(g[col]), "xaxis": "x" if i == 0 else "x2", "yaxis": "y" if i == 0 else "y2",
                         "line": {"color": c, "width": 2, "dash": "dot" if col == "3x_never" else "solid"},
                         "hovertemplate": "%{x|%d %b %Y}: %{y:.2f}<extra>" + nm + "</extra>"})

    def lay(narrow):
        dom = ([0, 0.47], [0.53, 1]) if not narrow else ([0, 1], [0, 1])
        ydom = ([0, 1], [0, 1]) if not narrow else ([0.58, 1], [0, 0.42])
        return base_layout("Rebalancing does not protect against a sudden crash" if not narrow else "Rebalancing does not protect<br>against a sudden crash",
                           "Equal-weight portfolios of the ten industries, value relative to the start of each panel" if not narrow else
                           "Equal-weight industry portfolios,<br>value relative to the start",
                           560, {"l": 56, "r": 20, "t": 120, "b": 50} if not narrow else {"l": 46, "r": 10, "t": 170, "b": 46},
                           showlegend=True, narrow=narrow,
                           xaxis=ax(domain=dom[0]), xaxis2=ax(domain=dom[1], anchor="y2"),
                           yaxis=ax(domain=ydom[0], range=[0, 1.8]), yaxis2=ax(domain=ydom[1], anchor="x2", range=[0, 1.8]),
                           shapes=[{"type": "line", "xref": "x domain", "yref": "y", "x0": 0, "x1": 1, "y0": 1, "y1": 1, "line": {"color": GREY, "width": 1, "dash": "dash"}},
                                   {"type": "line", "xref": "x2 domain", "yref": "y2", "x0": 0, "x1": 1, "y0": 1, "y1": 1, "line": {"color": GREY, "width": 1, "dash": "dash"}}],
                           annotations=[note("<b>August 1987 to December 1988</b>", 0, 1, xref="x domain", yref="y domain", anchor="left", yanchor="bottom", yshift=4),
                                        note("<b>2020</b>", 0, 1, xref="x2 domain", yref="y2 domain", anchor="left", yanchor="bottom", yshift=4)],
                           legend={"font": {"size": 11, "color": INK}, "orientation": "h", "x": 0, "y": 1.08 if not narrow else 1.03, "yanchor": "bottom"})
    wide, nar = lay(False), lay(True)
    nar["height"] = 860
    g87, g20 = ev[ev.event == "1987"], ev[ev.event == "2020"]
    finish_wrap("08-sudden-events", data, wide, nar, 1120,
                f"A sudden crash hits a rebalanced portfolio as hard as any other: on 19 October 1987 the equal-weight 3× portfolio lost about half its value in one day, with or without monthly rebalancing. Rebalancing acts only afterwards, by buying what fell; in 2020 that helped less than holding the winners ({g20['3x_monthly'].iloc[-1]:.2f} against {g20['3x_never'].iloc[-1]:.2f} by the end of the year).",
                f"Two line charts. Left, August 1987 to December 1988: both 3× portfolios fall from 1 to about {g87['3x_monthly'].min():.2f} in October 1987 and end near {g87['3x_monthly'].iloc[-1]:.2f}; the unleveraged portfolio falls to about {g87['1x_monthly'].min():.2f}. Right, 2020: the 3× portfolios fall to about {g20['3x_monthly'].min():.2f} in March and end at {g20['3x_monthly'].iloc[-1]:.2f} (rebalanced) and {g20['3x_never'].iloc[-1]:.2f} (never rebalanced).",
                SRC3 + " Daily simulation, rebalancing at month ends, no costs.")


# =================================================================================================================
def write_captions():
    lines = ["# Captions, alt text and sources: portfolio-hindsight figures", "",
             "Generated by `scripts/make_figures.py`; every number is read from `results/`.", ""]
    import re
    fix = lambda t: re.sub(r"(?<![\w.])-(?=\d)", "−", t)
    for name, cap, alt, src, ratio_h in CAPTIONS:
        cap, alt = fix(cap), fix(alt)
        lines += [f"## {name}", "", f"- **Caption:** {cap}", f"- **Alt text:** {alt}", f"- **Source line:** {src}",
                  f"- **Image size:** 2200 x {ratio_h} (suggested `--pf-ratio:2200 / {ratio_h}`)", ""]
    (FIGS / "CAPTIONS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8):
        if not only or any(f.__name__ == "fig" + o.lstrip("0") for o in only):
            f()
    write_captions()
