"""Plot the score-on-median (1d / 3d / 7d) rankings for the Bloomberg-overlay
live-scoring pipeline, with parametric-bootstrap CI whiskers.

Source : ../bloomberg_overlay/bloomberg_score_on_median_<v>_ci.csv
         (one row per model — schema identical to
          bloomberg_final_vs_final_ci.csv; produced by
          build_live_scoring_bloomberg.py).

Output (matches the variant subfolder layout used by
        step_15_5_scoring_by_theme/plots/plots_bloomberg/<variant>/):

  plots_ci/score_on_median_1d/           — all models, CI whiskers
  plots_ci/score_on_median_3d/
  plots_ci/score_on_median_7d/
  plots_no_agent_ci/score_on_median_1d/  — claude-code-agent dropped (n=11)
  plots_no_agent_ci/score_on_median_3d/
  plots_no_agent_ci/score_on_median_7d/

Per (variant, drop-agent) directory: five PNGs × two CI levels (90, 95)
+ a metric ranking CSV.

CI methodology: parametric bootstrap of β from N(β̂, V_β_HC1), the same
β-bootstrap used for final_vs_final (`plot_final_scores.py` →
`plots_ci/final_vs_final/`). Same n_boot, seed, and β draws across
variants; only the per-event (S, S_hat) field rows differ, so bars
across variants share the same β-uncertainty model.

The five metrics, plotted as vertical bar charts:
  1. BMSC Score                         — axhline 0   (no improvement)
  2. BMSC BP-RMSE                       — lower is better
  3. BMSC Weighted Directional Hit Rate — axhline 0.5 (coin flip)
  4. BDRC Score                         — axhline 0   (no skill)
  5. BDRC BP-RMSE                       — lower is better

(BDRC Weighted Directional Hit Rate is intentionally omitted to keep
the score-on-median dashboard focused on the headline 5 metrics; the
CI CSV does carry DRC_WDH columns if you want to add it later.)

Each plot sorts bars best -> worst (lower-better for BP-RMSE, higher-
better for everything else). NaN models fall to the right.

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
SRC_DIR = HERE.parent / "bloomberg_overlay"

VARIANT_LABELS = {
    "score_on_median_1d": "score-on-median (d=1)",
    "score_on_median_3d": "score-on-median (d in {1,2,3})",
    "score_on_median_7d": "score-on-median (d in {1..7})",
}
# Short median-window suffix used for the BDRC score plot title.
VARIANT_MEDIAN_LABEL = {
    "score_on_median_1d": "1-day Median",
    "score_on_median_3d": "3-day Median",
    "score_on_median_7d": "7-day Median",
}
VARIANT_ORDER = list(VARIANT_LABELS)
VARIANT_CI_CSV = {
    v: SRC_DIR / f"bloomberg_{v}_ci.csv" for v in VARIANT_ORDER
}

# Palette + labels: kept in sync with plot_final_scores.py so a reader can
# move between the two without re-learning the colour scheme.
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
    "claude-sonnet-4.5-api":           "#f1a226",  # gold
    "qwen3-235b-a22b-instruct-2507":   "#800074",  # med purple
    "qwen3-next-80b-a3b-instruct":     "#c98ec2",  # soft mauve (lighter purple)
    "arima_aic":                       "#b8b8b8",  # light gray
    "claude-code-agent":               "#ffcd8e",  # light gold
}

# Score plots: bars are bottom-aligned to a per-metric visual floor instead
# of 0, and both the y=0 benchmark and the floor reference lines are
# highlighted. Same convention as plot_final_scores.py.
#   • BMSC floor = −1 (metric bound; observed scores can sit near −0.9).
#   • BDRC floor = −0.6 (visual cap; all observed BDRC scores are above this,
#     and below −0.6 the score saturates rapidly toward −1, so widening the
#     range past −0.6 only adds dead empty space).
#   • BDRC, no-agent: dropping claude-code-agent leaves all observed BDRC
#     scores in [−0.12, +0.01] with CI95 lows reaching only ~−0.18, so the
#     floor tightens to −0.2 for the no-agent variant. BDRC=−0.2 ⇔ F/R=1.5,
#     i.e. error variance 50% above the realized-return scale (RMSE ≈ 1.22×).
SCORE_ALIGNED_STEMS = {"bmsc_score", "bdrc_score"}
# Full-plot floors: BMSC bounded by its metric bound (−1); BDRC truncated at
# −0.6 because below that the score saturates rapidly toward −1 and the
# extra range is empty.
SCORE_LOWER_BOUNDS = {
    "bmsc_score": -1.0,
    "bdrc_score": -0.6,
}
# Hard floors used by the dynamic per-variant floor in no-agent mode. The
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
    """Per-variant floor for a score stem in no-agent mode.

    Floor = floor_to_nearest(min(values, ci_lo) − pad, step), clipped at
    the metric's hard floor. Picks up the metric prefix (the stem may be
    e.g. ``bdrc_score_no_agent_ci``).
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

# (filename_stem, value_col, title, ylabel, hline, lower_better, uses_drc_n)
# value_col always ends in "_point"; CI columns are derived by stripping
# "_point" and appending "_ci{level}_{lo,hi}", matching the schema of
# bloomberg_final_vs_final_ci.csv.
PLOTS: list[tuple[str, str, str, str, float | None, bool, bool]] = [
    ("bmsc_score",   "BMSC_point",
     "Bounded Market-Surprise Capture Score",
     "",                  0.0,  False, False),
    ("bmsc_bp_rmse", "BP_RMSE_point",
     "Bounded Market-Surprise Capture · Basis-Point RMSE",
     "BP-RMSE (bp)",      None, True,  False),
    ("bmsc_wdhr",    "WDH_point",
     "Bounded Market-Surprise Capture · Weighted Directional Hit Rate",
     "weighted hit rate", 0.5,  False, False),
    ("bdrc_score",   "BDRC_point",
     "LiveMacro Score",
     "",                  0.0,  False, True),
    ("bdrc_bp_rmse", "DRC_RMSE_point",
     "Bounded Daily-Return Capture · Basis-Point RMSE",
     "BP-RMSE (bp)",      None, True,  True),
]

DROPPED_AGENT_LABEL = "claude-code-agent"
HLINE_COLOR = "#B23A48"
CI_LEVELS = (90, 95)  # bootstrap CI levels to render (one PNG per level)


def style() -> None:
    # Style matches plot_final_scores.py so a reader can place the headline
    # and the median-variant plots side by side without visual jitter.
    mpl.rcParams.update({
        "font.family": "sans-serif",
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
    n_events: np.ndarray,
    stem: str,
    title: str,
    ylabel: str,
    hline: float | None,
    lower_better: bool,
    ci_lo: np.ndarray,
    ci_hi: np.ndarray,
    label_precision: int = 2,
) -> Path:
    score_aligned = any(stem.startswith(s) for s in SCORE_ALIGNED_STEMS)
    score_lb = _score_lower_bound(stem, values, ci_lo)

    order = _sort_indices(values, lower_better)
    models = [models[i] for i in order]
    values = values[order]
    n_events = n_events[order]
    ci_lo = ci_lo[order]
    ci_hi = ci_hi[order]

    labels = [MODEL_LABELS.get(m, m) for m in models]
    colors = [MODEL_COLORS.get(m, "#888888") for m in models]

    # Unified figsize matches plot_final_scores.py for visual consistency
    # across the headline and median-variant dashboards.
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

    y_lo, y_hi = ax.get_ylim()
    pad = 0.05 * (y_hi - y_lo)
    for i, (bar, v) in enumerate(zip(bars, values)):
        if not np.isfinite(v):
            continue
        if score_aligned or v >= 0:
            y_text = v + pad
            if np.isfinite(ci_hi[i]):
                y_text = max(y_text, ci_hi[i] + pad)
            # Keep the label clear of the 0 hline when v sits just below it.
            if hline is not None and v < hline and y_text < hline + pad:
                y_text = hline + pad
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

    fig.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.16)
    out_path = out_dir / f"{stem}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def render_variant(variant: str, drop_agent: bool) -> None:
    suffix = "_no_agent_ci" if drop_agent else "_ci"
    # Drop-agent plots have a much tighter score range (agent is the worst
    # outlier dragging the full-plot scale wide); 3-digit labels expose the
    # inter-model gaps that compress to ~0.01 there. Full plots keep 2 digits.
    label_precision = 3 if drop_agent else 2
    out_dir = HERE / f"plots{suffix}" / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    src = VARIANT_CI_CSV[variant]
    if not src.exists():
        raise SystemExit(
            f"CI CSV missing: {src}. Re-run build_live_scoring_bloomberg.py."
        )
    df = pd.read_csv(src)
    if df.empty:
        print(f"  skipping {variant}: {src.name} is empty")
        return

    if drop_agent:
        df = df[df["model"] != DROPPED_AGENT_LABEL].copy()
        if df.empty:
            raise SystemExit(
                f"All rows dropped after removing {DROPPED_AGENT_LABEL} for {variant}"
            )

    df = df.sort_values("BMSC_point", ascending=False).reset_index(drop=True)
    models = df["model"].tolist()
    n_events = df["n_events"].to_numpy(dtype=float)
    n_drc_events = (
        df["n_drc_events"].to_numpy(dtype=float)
        if "n_drc_events" in df.columns else n_events
    )

    label_base = "drop claude-code-agent" if drop_agent else "all models"
    print(f"\n=== Plotting {variant} ({label_base}) -> "
          f"{out_dir.relative_to(HERE)} ===")
    print(f"  loaded {len(df)} rows from {src.name}")
    print(f"  model order: {models}")
    print(f"  CI levels: {CI_LEVELS}")

    variant_str = VARIANT_LABELS.get(variant, variant)

    for ci_level in CI_LEVELS:
        for stem, col, title, ylabel, hline, lower_better, use_drc_n in PLOTS:
            if col not in df.columns:
                print(f"  skip {stem}: column {col} not present")
                continue
            values = df[col].to_numpy(dtype=float)
            ne = n_drc_events if use_drc_n else n_events
            base = col.replace("_point", "")
            lo_col = f"{base}_ci{ci_level}_lo"
            hi_col = f"{base}_ci{ci_level}_hi"
            if lo_col not in df.columns or hi_col not in df.columns:
                raise SystemExit(
                    f"CI columns {lo_col}/{hi_col} missing from {src.name}; "
                    f"re-run build_live_scoring_bloomberg.py."
                )
            ci_lo = df[lo_col].to_numpy(dtype=float)
            ci_hi = df[hi_col].to_numpy(dtype=float)
            # BDRC score gets a short single-line title with the median window
            # (e.g. "LiveMacro Score 3-day Median"); other metrics keep the
            # two-line "<title>\n[<variant>]" layout.
            if stem == "bdrc_score":
                med = VARIANT_MEDIAN_LABEL.get(variant)
                full_title = f"{title} {med}" if med else title
            else:
                full_title = f"{title}\n[{variant_str}]"
            out_stem = f"{stem}{suffix}{ci_level}"
            p = plot_metric(out_dir, models, values, ne, out_stem, full_title,
                            ylabel, hline, lower_better, ci_lo, ci_hi,
                            label_precision=label_precision)
            print(f"  wrote {p.relative_to(HERE)}")

    # Ranking table mirrors metric_ranking_table_ci.csv from plot_final_scores.py:
    # carries every CI level's columns so re-writing per CI level would just duplicate.
    table_cols = [col for _, col, _, _, _, _, _ in PLOTS if col in df.columns]
    extra_cols = [c for c in ("n_events", "n_drc_events") if c in df.columns]
    rename = {
        "BMSC_point": "BMSC", "BP_RMSE_point": "BMSC_BP_RMSE",
        "WDH_point": "BMSC_WDHR",
        "BDRC_point": "BDRC", "DRC_RMSE_point": "BDRC_BP_RMSE",
    }
    table = df.set_index("model")[extra_cols + table_cols]
    table = table.rename(
        columns={k: v for k, v in rename.items() if k in table.columns}
    )
    table_path = out_dir / f"metric_ranking_table{suffix}.csv"
    table.to_csv(table_path)
    print(f"  wrote {table_path.relative_to(HERE)}")


def run(drop_agent: bool, variants: list[str]) -> None:
    for variant in variants:
        render_variant(variant, drop_agent)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--drop-claude-agent", action="store_true",
                   help="exclude claude-code-agent (small-N outlier; n=11)")
    p.add_argument("--all-variants", action="store_true",
                   help="render BOTH agent variants (full + no-agent)")
    p.add_argument("--variants", type=str, default=",".join(VARIANT_ORDER),
                   help="comma-separated variant keys "
                        "(score_on_median_1d, _3d, _7d)")
    args = p.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in VARIANT_LABELS]
    if bad:
        raise SystemExit(
            f"Unknown variant key(s): {bad}. Valid: {list(VARIANT_LABELS)}"
        )

    style()

    if args.all_variants:
        for da in (False, True):
            run(drop_agent=da, variants=variants)
    else:
        run(drop_agent=args.drop_claude_agent, variants=variants)


if __name__ == "__main__":
    main()
