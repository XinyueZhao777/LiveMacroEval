"""Plot the headline final_vs_final scores (with 90% bootstrap CI) for the
Bloomberg-overlay live-scoring pipeline.

Source : ../bloomberg_overlay/bloomberg_final_vs_final_ci.csv
         (one row per model, schema identical to the legacy
          huber_final/score_ci.csv).
Output : six PNGs per variant + a metric ranking CSV.

Two variants are emitted by --all-variants (default):
  plots_ci/final_vs_final/             — all models, with 90% CI whiskers.
  plots_no_agent_ci/final_vs_final/    — claude-code-agent dropped (n=11;
                                         small-N outlier).

The `final_vs_final/` subfolder mirrors the layout used in
step_15_5_scoring_by_theme/plots/plots_bloomberg/<variant>/, where each
scoring variant lives in its own subdirectory. Score-on-median (1d/3d/7d)
plots for the same dataset are written to the sibling subfolders by
`plot_score_on_median_scores.py`.

The six metrics, plotted as vertical bar charts:
  1. BMSC Score                         — axhline 0   (no improvement)
  2. BMSC BP-RMSE                       — lower is better
  3. BMSC Weighted Directional Hit Rate — axhline 0.5 (consensus)
  4. BDRC Score                         — axhline 0   (no better than zero)
  5. BDRC BP-RMSE                       — lower is better
  6. BDRC Weighted Directional Hit Rate — axhline 0.5 (consensus)

Each plot sorts bars best -> worst (lower-better for BP-RMSE, higher-
better for everything else). NaN models fall to the right.
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
SRC = HERE.parent / "bloomberg_overlay" / "bloomberg_final_vs_final_ci.csv"

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

# Plot stems whose bars should be bottom-aligned to the score's lower bound
# (rather than to 0). The y=0 benchmark and the per-metric score floor are
# both highlighted.
SCORE_ALIGNED_STEMS = {"bmsc_score", "bdrc_score"}
# BMSC and BDRC are bounded in [-1, 1] by construction.
#
# The formulas are
#   BMSC = (S - E) / (S + E)   with S = Σ Q²,  E = Σ (Q̂ - Q)²
#   BDRC = (R - F) / (R + F)   with R = Σ r²,  F = Σ (r - Q̂)²
# where Q is the market surprise relative to consensus and r is the realized
# daily return. So:
#   • Score = 0  ⇔  E = S  (resp. F = R), i.e. predictions are no better than
#                   the trivial "predict the benchmark" baseline (consensus
#                   for BMSC, zero return for BDRC).
#   • Score → -1 ⇔  E ≫ S  (resp. F ≫ R), i.e. the model's prediction error
#                   far exceeds the natural scale of the target — the model
#                   is much WORSE than the trivial baseline.
#
# Per-metric visual floors:
#   • BMSC: bars run all the way to the −1 metric floor; observed scores can
#           sit close to −0.9, so we keep the full range visible.
#   • BDRC: capped at −0.6 because all observed BDRC scores are well above
#           that, AND the score saturates rapidly past it (BDRC=−0.6 already
#           means F/R=4, i.e. RMSE ≈ 2× the realized-return scale; BDRC=−0.9
#           means F/R=19). Showing the dead empty band below −0.6 wastes
#           visual real estate and compresses the actually-informative range.
#   • BDRC, no-agent: dropping claude-code-agent (n=11 outlier at BDRC≈−0.36)
#           leaves all observed BDRC scores in [−0.12, +0.01] with CI95 lows
#           reaching only ~−0.18. The −0.6 floor would compress those bars to
#           a thin top strip, so we tighten to −0.2 just for the no-agent
#           variant. BDRC=−0.2 ⇔ F/R=1.5, i.e. error variance is 50% above
#           the realized-return scale — a clean "moderately worse than 'trust
#           consensus'" threshold (and RMSE ≈ 1.22× the realized-return
#           scale).
#
# Reference-line labels are metric-specific so a reader can grok the
# economic meaning without consulting the paper.
SCORE_LOWER_BOUNDS = {
    "bmsc_score": -1.0,
    "bdrc_score": -0.6,
}
# Hard floors used by the dynamic per-render floor in no-agent mode. The
# dynamic floor never drops below these.
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
    """Per-render floor for a score stem in no-agent mode.

    Floor = floor_to_nearest(min(values, ci_lo) − pad, step), clipped at
    the metric's hard floor (−1). Stems carry the ``_no_agent_ci{level}``
    suffix; the metric prefix is recovered by stripping ``_no_agent``.
    """
    base = stem.split("_no_agent")[0]
    hard = SCORE_LOWER_HARD_FLOOR.get(base)
    if hard is None:
        return None
    cand: list[float] = []
    fv = values[np.isfinite(values)]
    if len(fv):
        cand.append(float(fv.min()))
    if ci_lo is not None:
        fl = ci_lo[np.isfinite(ci_lo)]
        if len(fl):
            cand.append(float(fl.min()))
    if not cand:
        return hard
    lo = min(cand) - pad
    bound = math.floor(lo / step) * step
    return max(bound, hard)


def _score_lower_bound(
    stem: str,
    values: np.ndarray,
    ci_lo: np.ndarray | None,
) -> float | None:
    """Resolve the bottom-alignment floor for a score stem; None for non-score plots.

    No-agent renders (stem contains ``_no_agent_ci``) use a dynamic floor
    that always fits the lowest CI inside the axes. Full renders use the
    static SCORE_LOWER_BOUNDS table.
    """
    if "_no_agent_ci" in stem:
        return _dynamic_score_lower_bound(stem, values, ci_lo)
    for prefix, lb in SCORE_LOWER_BOUNDS.items():
        if stem.startswith(prefix):
            return lb
    return None

# (filename_stem, value_col, title, ylabel, hline, lower_better)
PLOTS: list[tuple[str, str, str, str, float | None, bool]] = [
    ("bmsc_score",    "BMSC_point",
     "LiveSurprise Score",
     "",                  0.0,  False),
    ("bmsc_bp_rmse",  "BP_RMSE_point",
     "Bounded Market-Surprise Capture · Basis-Point RMSE",
     "BP-RMSE (bp)",      None, True),
    ("bmsc_wdhr",     "WDH_point",
     "LiveSurprise Directional Score",
     "",                  0.5,  False),
    ("bdrc_score",    "BDRC_point",
     "LiveMacro Score",
     "",                  0.0,  False),
    ("bdrc_bp_rmse",  "DRC_RMSE_point",
     "Bounded Daily-Return Capture · Basis-Point RMSE",
     "BP-RMSE (bp)",      None, True),
    ("bdrc_wdhr",     "DRC_WDH_point",
     "LiveMacro Directional Score",
     "",                  0.5,  False),
]

DROPPED_AGENT_LABEL = "claude-code-agent"
HLINE_COLOR = "#B23A48"  # deep crimson — academic-muted but still attention-grabbing
CI_LEVELS = (90, 95)  # bootstrap CI levels to render (one PNG per level)


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
    av = abs(v)
    if av >= 100:
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
    n_events: np.ndarray,  # kept for signature compatibility; no longer rendered
    stem: str,
    title: str,
    ylabel: str,
    hline: float | None,
    lower_better: bool,
    ci_lo: np.ndarray,
    ci_hi: np.ndarray,
    label_precision: int = 2,
) -> Path:
    # `stem` here is the file stem (with suffix). The bottom-alignment switch
    # keys off the bare metric prefix; lower bound is metric-specific.
    score_aligned = any(stem.startswith(s) for s in SCORE_ALIGNED_STEMS)
    score_lb = _score_lower_bound(stem, values, ci_lo)

    order = _sort_indices(values, lower_better)
    models = [models[i] for i in order]
    values = values[order]
    ci_lo = ci_lo[order]
    ci_hi = ci_hi[order]

    labels = [MODEL_LABELS.get(m, m) for m in models]
    colors = [MODEL_COLORS.get(m, "#888888") for m in models]

    # Score plots are flatter — their floor is fixed at -1, the theoretical
    # bound, so the chart doesn't need to grow proportionally to data range.
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

    # CI whiskers anchored at the point estimate.
    lower = np.where(np.isfinite(ci_lo), values - ci_lo, np.nan)
    upper = np.where(np.isfinite(ci_hi), ci_hi - values, np.nan)
    lower = np.where(lower >= 0, lower, 0.0)
    upper = np.where(upper >= 0, upper, 0.0)
    x_centers = [bar.get_x() + bar.get_width() / 2.0 for bar in bars]
    ax.errorbar(
        x_centers, values, yerr=[lower, upper],
        fmt="none", ecolor="#1a1a1a", elinewidth=1.3, capsize=4, capthick=1.3,
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
        # character partway inside the axes. -12pt leaves only "C" inside the
        # axes for the ~90pt "Consensus" label at fontsize 19.
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

    # Y-axis limits.
    if score_aligned and base_bottom is not None:
        candidates = [v_max]
        if hline is not None:
            candidates.append(hline)
        fin_hi = ci_hi[np.isfinite(ci_hi)]
        if len(fin_hi):
            candidates.append(float(fin_hi.max()))
        fin_lo = ci_lo[np.isfinite(ci_lo)]
        if len(fin_lo):
            candidates.append(float(fin_lo.min()))
        v_top = float(max(candidates))
        ax.set_ylim(base_bottom, v_top + 0.20 * (v_top - base_bottom))
    elif len(finite_vals):
        candidates = [float(finite_vals.min()), float(finite_vals.max())]
        if hline is not None:
            candidates.append(hline)
        fin_lo = ci_lo[np.isfinite(ci_lo)]
        if len(fin_lo):
            candidates.append(float(fin_lo.min()))
        fin_hi = ci_hi[np.isfinite(ci_hi)]
        if len(fin_hi):
            candidates.append(float(fin_hi.max()))
        v_lo, v_hi = float(min(candidates)), float(max(candidates))
        if v_lo == v_hi:
            v_lo, v_hi = v_lo - 1.0, v_hi + 1.0
        span = v_hi - v_lo
        ax.set_ylim(v_lo - 0.10 * span, v_hi + 0.22 * span)

    # Value annotations. Place above the bar top (and above the CI whisker).
    # For bars whose natural label position would sit ON or just BELOW the 0
    # reference line, push the label above the line so it's legible. For bars
    # whose natural position is comfortably below the line, leave it there
    # (on top of the bar) instead of forcing every negative bar's label
    # to the line.
    y_lo, y_hi = ax.get_ylim()
    pad = 0.05 * (y_hi - y_lo)
    for i, (bar, v) in enumerate(zip(bars, values)):
        if not np.isfinite(v):
            continue
        if score_aligned or v >= 0:
            y_text = v + pad
            if np.isfinite(ci_hi[i]):
                y_text = max(y_text, ci_hi[i] + pad)
            # Push the label above the hline only when its natural position is
            # close to the line *relative to the bar's reach*: distance below
            # the hline must be under 18% of (hline − bar bottom). For bars
            # whose ci_hi sits deep below the hline (Qwen3-80B / ARIMA on the
            # BDRC plot) this leaves the label directly on top of the bar.
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
            if np.isfinite(ci_lo[i]):
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

    # Reference label sits inside the axes now, so the right margin is small.
    fig.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.16)
    out_path = out_dir / f"{stem}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def run(drop_agent: bool) -> None:
    suffix = "_no_agent_ci" if drop_agent else "_ci"
    # Drop-agent plots have a much tighter score range (the agent is the worst
    # outlier dragging the full-plot scale wide), so the inter-model gaps are
    # in the 3rd decimal and benefit from a 3-digit label. Full plots keep
    # 2 digits to avoid label crowding under the wide-range layout.
    label_precision = 3 if drop_agent else 2
    out_dir = HERE / f"plots{suffix}" / "final_vs_final"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SRC)
    if df.empty:
        raise SystemExit(f"No rows in {SRC}")
    if drop_agent:
        df = df[df["model"] != DROPPED_AGENT_LABEL].copy()
        if df.empty:
            raise SystemExit(f"All rows dropped after removing {DROPPED_AGENT_LABEL}")

    df = df.sort_values("BMSC_point", ascending=False).reset_index(drop=True)
    models = df["model"].tolist()
    n_events = df["n_events"].to_numpy()
    n_drc_events = df["n_drc_events"].to_numpy()

    label_base = "drop claude-code-agent" if drop_agent else "all models"
    print(f"\n=== Plotting ({label_base}) -> {out_dir} ===")
    print(f"  loaded {len(df)} rows from {SRC.name}")
    print(f"  model order: {models}")
    print(f"  CI levels: {CI_LEVELS}")

    for ci_level in CI_LEVELS:
        for stem, col, title, ylabel, hline, lower_better in PLOTS:
            values = df[col].to_numpy(dtype=float)
            ne = n_drc_events if col.startswith(("BDRC", "DRC")) else n_events
            base = col.replace("_point", "")
            lo_col = f"{base}_ci{ci_level}_lo"
            hi_col = f"{base}_ci{ci_level}_hi"
            if lo_col not in df.columns or hi_col not in df.columns:
                raise SystemExit(
                    f"CI columns {lo_col}/{hi_col} missing from {SRC.name}; "
                    f"re-run build_live_scoring_bloomberg.py."
                )
            ci_lo = df[lo_col].to_numpy(dtype=float)
            ci_hi = df[hi_col].to_numpy(dtype=float)
            out_stem = f"{stem}{suffix}{ci_level}"  # e.g. bmsc_score_ci90
            p = plot_metric(out_dir, models, values, ne, out_stem, title,
                            ylabel, hline, lower_better, ci_lo, ci_hi,
                            label_precision=label_precision)
            print(f"  wrote {p.relative_to(HERE)}")

    # Ranking table is written once per dir — it already carries every CI
    # level's columns, so re-writing per CI level would just duplicate.
    table_cols = [c for _, c, _, _, _, _ in PLOTS]
    table = df.set_index("model")[["n_events", "n_drc_events", *table_cols]]
    table = table.rename(columns={
        "BMSC_point": "BMSC", "BP_RMSE_point": "BMSC_BP_RMSE",
        "WDH_point": "BMSC_WDHR",
        "BDRC_point": "BDRC", "DRC_RMSE_point": "BDRC_BP_RMSE",
        "DRC_WDH_point": "BDRC_WDHR",
    })
    table_path = out_dir / f"metric_ranking_table{suffix}.csv"
    table.to_csv(table_path)
    print(f"  wrote {table_path.relative_to(HERE)}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--drop-claude-agent", action="store_true",
                   help="exclude claude-code-agent (small-N outlier; n=11)")
    p.add_argument("--all-variants", action="store_true",
                   help="render BOTH variants (with-CI all + with-CI no-agent)")
    args = p.parse_args()

    style()

    if args.all_variants:
        for da in (False, True):
            run(drop_agent=da)
    else:
        run(drop_agent=args.drop_claude_agent)


if __name__ == "__main__":
    main()
