"""Per-theme model-ranking bar charts for the Bloomberg-overlay theme scores
(`score_by_theme_bloomberg.py`).

Label precision: full plots (`plots_bloomberg/`) keep 2-decimal value labels;
no-agent plots (`plots_bloomberg_no_agent/`) use 3 decimals — dropping
claude-code-agent collapses the inter-model gap into the 3rd decimal for
most (theme, variant) combinations.

Score-plot y-axis floor: full plots keep the metric-bound floor (−1 for
BMSC, −0.6 for BDRC). No-agent plots compute the floor dynamically per
(theme, variant) so the bottom-most CI whisker always fits inside the
axes (the previous fixed −0.2 BDRC floor was clipped by Inflation /
Housing CIs); see `_dynamic_score_lower_bound`.

Reads the Bloomberg-overlay aggregate (point estimates + parametric-bootstrap
90% CIs) and renders one figure tree per scoring variant. CI whiskers are
drawn whenever the corresponding `<variant>_<metric>_ci90_{lo,hi}` columns
are present in the aggregate.

Source:  ../scores_by_theme_bloomberg_aggregate.csv  (wide; one row per
         (model, theme) with `<variant>_<metric>` and
         `<variant>_<metric>_ci{90,95}_{lo,hi}` columns).

Variants rendered (filename stems used in titles and output paths):
  score_on_median_1d
  score_on_median_3d
  score_on_median_7d
  final_vs_final

Themes (4 in total): Production, Inflation_Consumption_Services,
Labor_Market, Housing.

Five metrics per (variant, theme):
  1. BMSC Score                         — axhline 0
  2. BMSC BP-RMSE                       — lower is better
  3. BMSC Weighted Directional Hit Rate — axhline 0.5
  4. BDRC Score                         — axhline 0
  5. BDRC BP-RMSE                       — lower is better

(The upstream pipeline does not emit DRC_WDH on these variants, so the
6th metric from the Investing.com flow is intentionally omitted.)

Output:
  plots_bloomberg/<variant>/<theme>/{bmsc_score, bmsc_bp_rmse, bmsc_wdhr,
                                     bdrc_score, bdrc_bp_rmse}.png
  plots_bloomberg/<variant>/<theme>/metric_ranking_table.csv

Add `--drop-claude-agent` to redirect output to
`plots_bloomberg_no_agent/...` (small-N at the theme level).

Conda env: livemacro.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "scores_by_theme_bloomberg_aggregate.csv"

# Theme labels match the new (4-theme) composition.
THEME_LABELS = {
    "Production":                     "Supply and Production",
    "Inflation_Consumption_Services": "Demand and Inflation",
    "Labor_Market":                   "Labor Market",
    "Housing":                        "Housing",
}
THEME_ORDER = list(THEME_LABELS)

# Variants and their pretty subtitles.
VARIANT_LABELS = {
    "score_on_median_1d": "score-on-median (d=1)",
    "score_on_median_3d": "score-on-median (d∈{1,2,3})",
    "score_on_median_7d": "score-on-median (d∈{1..7})",
    "final_vs_final":   "final-vs-final",
}
# Short median-window suffix appended to the BDRC score plot title.
VARIANT_MEDIAN_LABEL = {
    "score_on_median_1d": "1-day Median",
    "score_on_median_3d": "3-day Median",
    "score_on_median_7d": "7-day Median",
}
VARIANT_ORDER = list(VARIANT_LABELS)

# Identical palette to step 15.4 / Investing.com plot script.
MODEL_LABELS = {
    "arima_aic":                       "ARIMA",
    "claude-code-agent":               "Claude Code",
    "claude-sonnet-4.5-api":           "Sonnet 4.5",
    "gpt-5-search-api":                "GPT-5",
    "qwen3-235b-a22b-instruct-2507":   "Qwen3-235B",
    "qwen3-next-80b-a3b-instruct":     "Qwen3-80B",
}
MODEL_COLORS = {
    "gpt-5-search-api":                "#298c8c",  # teal
    "claude-sonnet-4.5-api":           "#d97757",  # Claude-like terracotta orange
    "qwen3-235b-a22b-instruct-2507":   "#8f63b8",  # Qwen-like purple approximation
    "qwen3-next-80b-a3b-instruct":     "#7488b8",  # muted purple-blue
    "arima_aic":                       "#b0b0b0",  # academic gray
    "claude-code-agent":               "#ffcd8e",  # light gold
}
DROPPED_AGENT_LABEL = "claude-code-agent"

# (filename_stem, metric_suffix, title, ylabel, hline, lower_better,
#  uses_drc_n)
PLOTS: list[tuple[str, str, str, str, float | None, bool, bool]] = [
    ("bmsc_score",   "BMSC",
     "Bounded Market-Surprise Capture Score",
     "",                  0.0,  False, False),
    ("bmsc_bp_rmse", "BP_RMSE",
     "Bounded Market-Surprise Capture · Basis-Point RMSE",
     "BP-RMSE (bp)",      None, True,  False),
    ("bmsc_wdhr",    "WDH",
     "Bounded Market-Surprise Capture · Weighted Directional Hit Rate",
     "",                  0.5,  False, False),
    ("bdrc_score",   "BDRC",
     "Market-Return Capture Score",
     "",                  0.0,  False, True),
    ("bdrc_bp_rmse", "DRC_RMSE",
     "Bounded Daily-Return Capture · Basis-Point RMSE",
     "BP-RMSE (bp)",      None, True,  True),
]

HLINE_COLOR = "#B23A48"

# Score plots: bars are bottom-aligned to a per-metric visual floor instead
# of 0, mirroring the step_15_4 dashboards. Both the y=0 benchmark and the
# floor reference lines are highlighted.
#   • BMSC floor = −1 (metric bound; observed scores can sit near −0.95).
#   • BDRC floor = −0.6 (visual cap; almost all observed BDRC scores are
#     above this, and below −0.6 the score saturates rapidly toward −1
#     — BDRC=−0.6 already corresponds to F/R=4, i.e. RMSE ≈ 2× the
#     realized-return scale; BDRC=−0.9 ⇒ F/R=19. Cropping the empty band
#     keeps the informative range visually expanded).
SCORE_ALIGNED_STEMS = {"bmsc_score", "bdrc_score"}
# Full-plot floors: BMSC bounded by its metric bound (−1); BDRC truncated
# at −0.6 because below that the score saturates rapidly toward −1 and the
# extra range is empty.
SCORE_LOWER_BOUNDS_FULL = {
    "bmsc_score": -1.0,
    "bdrc_score": -0.6,
}
# Hard floors used by the dynamic per-(theme, variant) floor in no-agent
# mode. The dynamic floor never drops below these.
SCORE_LOWER_HARD_FLOOR = {
    "bmsc_score": -1.0,
    "bdrc_score": -1.0,
}


def _dynamic_score_lower_bound(
    stem: str,
    values: np.ndarray,
    ci_lo: np.ndarray | None,
    pad: float = 0.02,
    step: float = 0.05,
) -> float | None:
    """Per-(theme, variant) floor for a score stem in no-agent mode.

    Floor = floor_to_nearest(min(values, ci_lo) − pad, step), clipped at
    the metric's hard floor (−1). Padding leaves a small visual buffer
    between the lowest CI whisker and the axes bottom.
    """
    hard = SCORE_LOWER_HARD_FLOOR.get(stem.split("_no_agent")[0])
    if hard is None:
        return None
    candidates: list[float] = []
    fv = values[np.isfinite(values)]
    if len(fv):
        candidates.append(float(fv.min()))
    if ci_lo is not None:
        fl = ci_lo[np.isfinite(ci_lo)]
        if len(fl):
            candidates.append(float(fl.min()))
    if not candidates:
        return hard
    lo = min(candidates) - pad
    bound = math.floor(lo / step) * step
    return max(bound, hard)


def _score_lower_bound(
    stem: str,
    drop_agent: bool,
    values: np.ndarray,
    ci_lo: np.ndarray | None,
) -> float | None:
    """Resolve the bottom-alignment floor for a score stem; None otherwise."""
    if drop_agent:
        return _dynamic_score_lower_bound(stem, values, ci_lo)
    for prefix, lb in SCORE_LOWER_BOUNDS_FULL.items():
        if stem.startswith(prefix):
            return lb
    return None


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        # TeX Gyre Heros is a free, metric-compatible Helvetica clone (the
        # closest open-source stand-in for Helvetica Neue) installed under
        # ~/.local/share/fonts/texgyreheros/.
        "font.sans-serif": ["TeX Gyre Heros", "Helvetica Neue", "Helvetica",
                            "Nimbus Sans", "Liberation Sans", "Arial",
                            "DejaVu Sans"],
        "font.weight": "normal",
        "font.size": 18,
        "axes.edgecolor": "#1a1a1a",
        "axes.linewidth": 3.1,
        "axes.labelcolor": "#1a1a1a",
        "axes.labelweight": "normal",
        "xtick.color": "#1a1a1a",
        "ytick.color": "#1a1a1a",
        "xtick.labelsize": 20,
        "ytick.labelsize": 19,
        "xtick.major.width": 3.1,
        "ytick.major.width": 3.1,
        "xtick.major.size": 10,
        "ytick.major.size": 10,
        "xtick.minor.width": 1.5,
        "ytick.minor.width": 1.5,
        "axes.labelsize": 18,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "normal",
        "axes.titlesize": 24,
        "axes.titlecolor": "#0e0e0e",
        "axes.titlepad": 16,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _format_value(v: float, precision: int = 2) -> str:
    if not np.isfinite(v):
        return "n/a"
    if abs(v) >= 100:
        return f"{v:,.1f}"
    return f"{v:,.{precision}f}"


def _hline_text(hline: float) -> str:
    if abs(hline) < 1e-12:
        return "Consensus"
    if abs(hline - 0.5) < 1e-12:
        return "Consensus"
    return f"Benchmark ({hline:g})"


def _sort_indices(values: np.ndarray, lower_better: bool) -> np.ndarray:
    finite = np.isfinite(values)
    fin_idx = np.where(finite)[0]
    nan_idx = np.where(~finite)[0]
    fin_sorted = fin_idx[np.argsort(values[fin_idx], kind="stable")]
    if not lower_better:
        fin_sorted = fin_sorted[::-1]
    return np.concatenate([fin_sorted, nan_idx])


def plot_metric(
    out_dir: Path,
    models: list[str],
    values: np.ndarray,
    n_events: np.ndarray,
    stem: str,
    title: str,
    ylabel: str,
    hline: float | None,
    lower_better: bool,
    ci_lo: np.ndarray | None,
    ci_hi: np.ndarray | None,
    drop_agent: bool = False,
    label_precision: int = 2,
) -> Path:
    score_aligned = any(stem.startswith(s) for s in SCORE_ALIGNED_STEMS)
    score_lb = _score_lower_bound(stem, drop_agent, values, ci_lo)

    order = _sort_indices(values, lower_better)
    models = [models[i] for i in order]
    values = values[order]
    n_events = n_events[order]
    if ci_lo is not None:
        ci_lo = ci_lo[order]
    if ci_hi is not None:
        ci_hi = ci_hi[order]

    labels = [MODEL_LABELS.get(m, m) for m in models]
    colors = [MODEL_COLORS.get(m, "#888888") for m in models]

    # Score plots are flatter — their floor is fixed (−1 for BMSC, −0.6 for
    # BDRC), so the chart doesn't need to grow proportionally to data range.
    # Unified figure size for EMNLP single-column inclusion (~2.6x column
    # width — include with \includegraphics[width=\columnwidth]). Same size
    # for score and non-score plots so a figure grid looks uniform.
    figsize = (8.5, 5.2)
    fig, ax = plt.subplots(figsize=figsize)

    finite_vals = values[np.isfinite(values)]
    base_bottom: float | None = None
    if score_aligned and len(finite_vals) and score_lb is not None:
        v_max = float(finite_vals.max())
        base_bottom = score_lb
        bar_heights = np.where(np.isfinite(values), values - base_bottom, 0.0)
        bars = ax.bar(
            labels, bar_heights, bottom=base_bottom,
            color=colors, edgecolor="none", linewidth=0, width=0.55,
            zorder=2,
        )
    else:
        plot_vals = np.where(np.isfinite(values), values, 0.0)
        bars = ax.bar(
            labels, plot_vals,
            color=colors, edgecolor="none", linewidth=0, width=0.55,
            zorder=2,
        )

    ax.set_axisbelow(True)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)

    if ci_lo is not None and ci_hi is not None:
        lower = np.where(np.isfinite(ci_lo), values - ci_lo, np.nan)
        upper = np.where(np.isfinite(ci_hi), ci_hi - values, np.nan)
        lower = np.where(lower >= 0, lower, 0.0)
        upper = np.where(upper >= 0, upper, 0.0)
        x_centers = [bar.get_x() + bar.get_width() / 2.0 for bar in bars]
        ax.errorbar(
            x_centers, values,
            yerr=[lower, upper],
            fmt="none", ecolor="#222222", elinewidth=1.2, capsize=4, capthick=1.2,
            zorder=3,
        )

    if hline is not None:
        ax.axhline(
            hline, color=HLINE_COLOR,
            linewidth=4.5 if score_aligned else 4.0,
            alpha=0.97, zorder=4,
        )
        # Label straddles the right end of the red line. ha="left" anchored at
        # the axes right edge, with a negative x-offset pulling the leftmost
        # character partway inside the axes. Offsets are tuned per-label
        # (labels are ~90pt at fontsize 19):
        #   "Consensus": -12pt → almost flush, only "C" sits inside the axes.
        #   "Coin flip": -20pt → ~1/5 inside, 4/5 outside.
        label = _hline_text(hline)
        x_offset = -12 if label == "Consensus" else -20
        ax.annotate(
            label,
            xy=(1.0, hline), xycoords=ax.get_yaxis_transform(),
            xytext=(x_offset, 6), textcoords="offset points",
            ha="left", va="bottom",
            color=HLINE_COLOR, fontsize=19,
        )

    ax.set_title(title, pad=14)
    if ylabel:
        ax.set_ylabel(ylabel)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    if score_aligned and base_bottom is not None:
        candidates = [v_max]
        if hline is not None:
            candidates.append(hline)
        if ci_hi is not None:
            fin_hi = ci_hi[np.isfinite(ci_hi)]
            if len(fin_hi):
                candidates.append(float(fin_hi.max()))
        if ci_lo is not None:
            fin_lo = ci_lo[np.isfinite(ci_lo)]
            if len(fin_lo):
                candidates.append(float(fin_lo.min()))
        v_top = float(max(candidates))
        ax.set_ylim(base_bottom, v_top + 0.20 * (v_top - base_bottom))
    elif len(finite_vals):
        candidates = [float(finite_vals.min()), float(finite_vals.max())]
        if hline is not None:
            candidates.append(hline)
        if ci_lo is not None:
            fin_lo = ci_lo[np.isfinite(ci_lo)]
            if len(fin_lo):
                candidates.append(float(fin_lo.min()))
        if ci_hi is not None:
            fin_hi = ci_hi[np.isfinite(ci_hi)]
            if len(fin_hi):
                candidates.append(float(fin_hi.max()))
        v_lo = float(min(candidates))
        v_hi = float(max(candidates))
        if v_lo == v_hi:
            v_lo, v_hi = v_lo - 1.0, v_hi + 1.0
        span = v_hi - v_lo
        ax.set_ylim(v_lo - 0.10 * span, v_hi + 0.22 * span)

    y_lo, y_hi = ax.get_ylim()
    pad = 0.05 * (y_hi - y_lo)
    for i, (bar, v) in enumerate(zip(bars, values)):
        if not np.isfinite(v):
            continue
        if score_aligned or v >= 0:
            y_text = v + pad
            if ci_hi is not None and np.isfinite(ci_hi[i]):
                y_text = max(y_text, ci_hi[i] + pad)
            # Push the label above the hline only when its natural position is
            # close to the line relative to the bar's reach. Bars whose ci_hi
            # sits deep below the hline keep the label directly on top.
            if hline is not None and v < hline:
                bar_bottom = base_bottom if base_bottom is not None else 0.0
                range_below = hline - bar_bottom
                target = hline + pad
                if (range_below > 0
                        and (hline - y_text) < 0.18 * range_below
                        and target > y_text):
                    y_text = target
            va = "bottom"
        else:
            y_text = v - pad
            if ci_lo is not None and np.isfinite(ci_lo[i]):
                y_text = min(y_text, ci_lo[i] - pad)
            va = "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_text,
            _format_value(v, precision=label_precision),
            ha="center", va=va, fontsize=19, color="#111111",
        )

    for s in ("left", "bottom"):
        ax.spines[s].set_color("#1a1a1a")
        ax.spines[s].set_linewidth(3.1)

    fig.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.16)
    out_path = out_dir / f"{stem}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def render_theme_variant(
    df: pd.DataFrame,
    variant: str,
    theme: str,
    out_root: Path,
    drop_agent: bool,
) -> None:
    sub = df[df["theme"] == theme].copy()
    if drop_agent:
        sub = sub[sub["model"] != DROPPED_AGENT_LABEL].copy()
    if sub.empty:
        print(f"  [{variant}/{theme}] no rows after filters; skipping")
        return

    bmsc_col = f"{variant}_BMSC"
    n_col = f"{variant}_n_events"
    n_drc_col = f"{variant}_n_drc_events"
    if bmsc_col not in sub.columns:
        print(f"  [{variant}/{theme}] missing column {bmsc_col}; skipping")
        return

    sub = sub.sort_values(
        bmsc_col, ascending=False, na_position="last"
    ).reset_index(drop=True)

    out_dir = out_root / variant / theme
    out_dir.mkdir(parents=True, exist_ok=True)

    models = sub["model"].tolist()
    n_events = sub[n_col].to_numpy(dtype=float) if n_col in sub.columns else np.full(len(sub), np.nan)
    n_drc = sub[n_drc_col].to_numpy(dtype=float) if n_drc_col in sub.columns else n_events

    label_str = THEME_LABELS.get(theme, theme)
    variant_str = VARIANT_LABELS.get(variant, variant)
    print(f"\n  [{variant} / {theme}] models={models}")

    for stem, suffix, title, ylabel, hline, lower_better, use_drc_n in PLOTS:
        col = f"{variant}_{suffix}"
        if col not in sub.columns:
            print(f"    skip {stem}: column {col} not present")
            continue
        values = sub[col].to_numpy(dtype=float)
        ne = n_drc if use_drc_n else n_events
        lo_col = f"{variant}_{suffix}_ci90_lo"
        hi_col = f"{variant}_{suffix}_ci90_hi"
        ci_lo = sub[lo_col].to_numpy(dtype=float) if lo_col in sub.columns else None
        ci_hi = sub[hi_col].to_numpy(dtype=float) if hi_col in sub.columns else None
        # BDRC score plots become "LiveMacro Score: <Theme>" so each
        # per-theme figure is self-identifying when included standalone;
        # other metrics keep the theme name as their title.
        if stem == "bdrc_score":
            full_title = f"LiveMacro Score: {label_str}"
        else:
            full_title = label_str
        # No-agent plots use 3-digit labels (compressed inter-model gaps);
        # full plots keep 2 digits.
        label_precision = 3 if drop_agent else 2
        p = plot_metric(out_dir, models, values, ne, stem, full_title,
                        ylabel, hline, lower_better=lower_better,
                        ci_lo=ci_lo, ci_hi=ci_hi,
                        drop_agent=drop_agent,
                        label_precision=label_precision)
        print(f"    wrote {p.relative_to(HERE)}")

    # Per-(variant, theme) ranking table.
    table_cols = [f"{variant}_{suffix}" for _, suffix, _, _, _, _, _ in PLOTS]
    table_cols = [c for c in table_cols if c in sub.columns]
    extra_cols = [c for c in (n_col, n_drc_col) if c in sub.columns]
    table = sub.set_index("model")[extra_cols + table_cols]
    rename = {f"{variant}_{suffix}": pretty for pretty, suffix in (
        ("BMSC", "BMSC"), ("BMSC_BP_RMSE", "BP_RMSE"), ("BMSC_WDHR", "WDH"),
        ("BDRC", "BDRC"), ("BDRC_BP_RMSE", "DRC_RMSE"),
    ) if f"{variant}_{suffix}" in table.columns}
    rename.update({n_col: "n_events", n_drc_col: "n_drc_events"})
    table = table.rename(columns=rename)
    table_path = out_dir / "metric_ranking_table.csv"
    table.to_csv(table_path)
    print(f"    wrote {table_path.relative_to(HERE)}")


def _outdir(drop_agent: bool) -> Path:
    name = "plots_bloomberg_no_agent" if drop_agent else "plots_bloomberg"
    out = HERE / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def run(drop_agent: bool, themes: list[str], variants: list[str]) -> None:
    out_root = _outdir(drop_agent)
    df = pd.read_csv(SRC)
    label = "drop claude-code-agent" if drop_agent else "default"
    print(
        f"\n=== Plotting Bloomberg theme scores ({label}) → "
        f"{out_root.relative_to(HERE)}/ ==="
    )
    print(f"  loaded {len(df)} rows from {SRC.name}")
    for variant in variants:
        for theme in themes:
            render_theme_variant(df, variant, theme, out_root, drop_agent)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--drop-claude-agent", action="store_true",
                   help="drop claude-code-agent (small-N at the theme level)")
    p.add_argument("--all-variants", action="store_true",
                   help="render both full and no-agent variants in one shot")
    p.add_argument("--themes", type=str, default=",".join(THEME_ORDER),
                   help="comma-separated theme keys (default: all)")
    p.add_argument("--variants", type=str, default=",".join(VARIANT_ORDER),
                   help="comma-separated variant keys "
                        "(score_on_median_1d, _3d, _7d, final_vs_final)")
    args = p.parse_args()

    themes = [t.strip() for t in args.themes.split(",") if t.strip()]
    bad_t = [t for t in themes if t not in THEME_LABELS]
    if bad_t:
        raise SystemExit(f"Unknown theme key(s): {bad_t}. "
                         f"Valid: {list(THEME_LABELS)}")

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad_v = [v for v in variants if v not in VARIANT_LABELS]
    if bad_v:
        raise SystemExit(f"Unknown variant key(s): {bad_v}. "
                         f"Valid: {list(VARIANT_LABELS)}")

    style()
    if args.all_variants:
        for da in (False, True):
            run(drop_agent=da, themes=themes, variants=variants)
    else:
        run(drop_agent=args.drop_claude_agent, themes=themes, variants=variants)


if __name__ == "__main__":
    main()
