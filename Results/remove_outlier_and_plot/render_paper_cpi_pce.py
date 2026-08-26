#!/usr/bin/env python3
"""Re-render the spline-overlay CPI and PCE plots with paper-figure styling.

Targets only two PNGs (overwriting them in place):
  Results/remove_outlier_and_plot/generated_serverA_serverB/
    final_analysis_data_smoother_spline/spline_overlay/
    model_gpt-5-search-api/2026-03_core_macroeconomic_conditions/
      cpi_yoy.png
      pce_price_index_mom.png

Reuses the already-outlier-filtered + release-clipped rows produced by
generate_plots_serverA_serverB.py (mirrored under processed_final_analysis_data/),
so the underlying data is byte-identical to what the original plots used.
Smoothing is the same days-scale GCV cubic spline that the original script
applies in compute_smoothing_spline; raw points are likewise pre-summarized
in groups of 5 to dampen visual noise (raw_aggregate_n=5 in the source).

Styling departs from the original:
  - figsize set for a single-column paper figure, slightly wider than tall;
  - TeX Gyre Heros font (Helvetica clone), all weights normal;
  - thick axes / ticks matching plot_final_scores.py;
  - no legend (no "Raw" / "Smoothing Spline" labels);
  - no "Outliers removed" annotation;
  - 5 evenly-spaced x ticks rendered as short date labels;
  - axis titles: x="Nowcast Time"; y is variable-specific;
  - plot titles: "CPI" / "PCE Price Index".

Conda env: livemacro.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.interpolate import make_smoothing_spline

REPO_ROOT = Path(__file__).resolve().parents[1]

# Reuse the canonical outlier-removal logic so the upstream-loaded event-study
# variant is filtered identically to the way generate_plots_serverA_serverB.py
# filters the processed CSV (just without the release+3d clip).
sys.path.insert(0, str(REPO_ROOT / "remove_outlier_and_plot"))
from generate_plots_serverA_serverB import (  # noqa: E402
    POST_OUTLIER_Z_THRESHOLD,
    _robust_scale,
    remove_outliers,
)

PROCESSED_CSV = (
    REPO_ROOT
    / "remove_outlier_and_plot"
    / "processed_final_analysis_data"
    / "model_gpt-5-search-api"
    / "2026-03_core_macroeconomic_conditions.csv"
)

# Upstream pre-clip pre-outlier-removal data, sourced from the same CSV that
# generate_plots_serverA_serverB.py reads when --data-source=final_analysis_data.
# The default-variant plots continue to use PROCESSED_CSV (already clipped to
# release+3d) for byte-identical reproduction; the event-study variant uses
# UPSTREAM_CSV so it can extend past the release+3d boundary (to Apr 20).
UPSTREAM_CSV = (
    REPO_ROOT
    / "data_from_serverA_serverB"
    / "final_analysis_data"
    / "model_gpt-5-search-api"
    / "2026-03_core_macroeconomic_conditions.csv"
)

GT_CSV = REPO_ROOT / "ground_truth" / "data" / "field_releases_live.csv"

OUTPUT_DIR = (
    REPO_ROOT
    / "remove_outlier_and_plot"
    / "generated_serverA_serverB"
    / "final_analysis_data_smoother_spline"
    / "spline_overlay"
    / "model_gpt-5-search-api"
    / "2026-03_core_macroeconomic_conditions"
)

# (variable, title, ylabel, gt_field_id, gt_scale)
PANELS = [
    ("cpi_yoy",             "CPI",             "Year-over-year growth (%)", "cpi",             1.0),
    ("pce_price_index_mom", "PCE Price Index", "Month-over-month growth (%)", "pce_price_index", 1.0),
]

REF_PERIOD = "2026-03"
RAW_AGGREGATE_N = 5  # same as the source script

STEEL_GREY = "#708090"  # raw scatter, matches SLATE in source
SPLINE_RED = "#A52A2A"  # smoothed line, matches BRICK_RED
GT_RED     = "#8B0000"  # GT horizontal ref, matches GROUND_TRUTH_COLOR
RELEASE_BLACK = "#1C1C1C"
# GPT teal from Results/polymarket_return/plot_continuous_feb_mar.py
# MODEL_STYLES["gpt-5-search-api"]. Used as the spline-line color in the
# event-study variant.
GPT_TEAL = "#298c8c"

# Event-study variant: x-axis clipped to this upper bound and a single
# vertical line drawn at EVENT_TIME (no GT horizontal ref, no GT release
# vertical line). Times are naive datetimes treated in the same ET-ish
# convention as the prediction timestamp_local / GT release_datetime_et.
EVENT_TIME = pd.Timestamp("2026-04-08 10:30:00")
EVENT_XMAX = pd.Timestamp("2026-04-20 23:59:59")


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["TeX Gyre Heros", "Helvetica Neue", "Helvetica",
                            "Nimbus Sans", "Liberation Sans", "Arial",
                            "DejaVu Sans"],
        "font.weight": "normal",
        "font.size": 16,
        "axes.titlesize": 22,
        "axes.titleweight": "normal",
        "axes.titlecolor": "#0e0e0e",
        "axes.titlepad": 10,
        "axes.labelsize": 18,
        "axes.labelweight": "normal",
        "axes.labelcolor": "#1a1a1a",
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "xtick.color": "#1a1a1a",
        "ytick.color": "#1a1a1a",
        "xtick.major.width": 3.4,
        "ytick.major.width": 3.4,
        "xtick.major.size": 10,
        "ytick.major.size": 10,
        "xtick.minor.width": 1.5,
        "ytick.minor.width": 1.5,
        "axes.edgecolor": "#1a1a1a",
        "axes.linewidth": 3.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def load_upstream_filtered(
    variable: str,
    gt_field_id: str,
    gt_scale: float,
    t_max: pd.Timestamp,
) -> pd.DataFrame:
    """Load upstream (release-cutoff-free) data for one variable, clip to
    t_max, and apply the same outlier-removal logic that
    generate_plots_serverA_serverB.py uses on the processed CSV.

    Pre/post-release split mirrors prepare_variable_data: pre-release uses
    segment median + MAD; post-release is anchored on the GT value with the
    pre-release MAD as the noise scale (POST_OUTLIER_Z_THRESHOLD=5.0). When
    the release falls outside the window (PCE: release Apr 30 > Apr 20),
    the segment is treated as fully pre-release.
    """
    df = pd.read_csv(UPSTREAM_CSV)
    df["timestamp_local"] = pd.to_datetime(df["timestamp_local"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    sub = (
        df[df["variable"] == variable]
        .dropna(subset=["timestamp_local", "value"])
        .sort_values("timestamp_local")
    )
    sub = sub[sub["timestamp_local"] <= t_max]
    if sub.empty:
        return sub

    gt_value, release_dt = lookup_gt(gt_field_id, gt_scale)
    if release_dt is not None and release_dt <= t_max:
        pre = sub[sub["timestamp_local"] < release_dt]
        post = sub[sub["timestamp_local"] >= release_dt]
        pre_filt, _, _ = remove_outliers(pre)
        pre_scale = _robust_scale(pre["value"]) if not pre.empty else 0.0
        if pre_scale > 0:
            post_filt, _, _ = remove_outliers(
                post,
                anchor=float(gt_value),
                scale=pre_scale,
                z_threshold=POST_OUTLIER_Z_THRESHOLD,
                apply_relative_guard=False,
            )
        else:
            post_filt, _, _ = remove_outliers(post)
        return pd.concat([pre_filt, post_filt]).sort_values("timestamp_local")
    filtered, _, _ = remove_outliers(sub)
    return filtered


def lookup_gt(field_id: str, scale: float) -> tuple[float, pd.Timestamp]:
    gt = pd.read_csv(GT_CSV)
    rows = gt[(gt["field_id"] == field_id) & (gt["ref_period"] == REF_PERIOD)]
    if rows.empty:
        raise SystemExit(f"No GT row for field_id={field_id} ref={REF_PERIOD}")
    row = rows.sort_values("release_datetime_et").iloc[0]
    return float(row["A"]) * scale, pd.Timestamp(row["release_datetime_et"])


def compute_spline(df: pd.DataFrame) -> pd.Series:
    """Same algorithm as compute_smoothing_spline in generate_plots_serverA_serverB.py:
    days-scale x, GCV-chosen lambda, returns spline evaluated at unique timestamps."""
    series = df.set_index("timestamp_local")["value"].sort_index()
    series = series.groupby(level=0).mean()
    if series.size < 5:
        return pd.Series(dtype=float, index=series.index)
    x_ns = series.index.astype("int64").to_numpy()
    x_days = (x_ns - x_ns[0]).astype("float64") / (1e9 * 86400.0)
    y = series.to_numpy(dtype="float64")
    spline = make_smoothing_spline(x_days, y, lam=None)
    return pd.Series(spline(x_days), index=series.index)


def aggregate_raw(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Collapse consecutive rows into the mean of every N — matches source script."""
    if n < 2 or df.empty:
        return df
    s = df.sort_values("timestamp_local").reset_index(drop=True)
    bucket = s.index // n
    return s.groupby(bucket).agg(
        timestamp_local=("timestamp_local", "mean"),
        value=("value", "mean"),
    )


def set_n_ticks(ax: plt.Axes, t_min: pd.Timestamp, t_max: pd.Timestamp, n: int = 5) -> None:
    """Place exactly n evenly-spaced datetime ticks across [t_min, t_max]."""
    ticks = pd.date_range(t_min, t_max, periods=n)
    ax.set_xticks(ticks)
    span_days = (t_max - t_min).total_seconds() / 86400.0
    fmt = "%b %d" if span_days <= 60 else "%Y-%m-%d"
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))


def render_panel(
    df_all: pd.DataFrame,
    variable: str,
    title: str,
    ylabel: str,
    gt_field_id: str,
    gt_scale: float,
    out_path: Path,
    variant: str = "default",
    spline_linewidth: float = 2.4,
) -> None:
    """Render one panel.

    variant="default":
      Smoothed line in brick red, GT horizontal ref + GT release vertical
      line drawn, x extends to the last observation.

    variant="event_study":
      Smoothed line in GPT teal (matching plot_continuous_feb_mar.py's
      gpt-5-search-api color). NO GT horizontal ref, NO GT release line.
      A single black-dashed vertical line at EVENT_TIME marks the event-
      study event. x clipped to [t_min, EVENT_XMAX] (Apr 13 end-of-day).
    """
    if variant == "event_study":
        # Use upstream CSV (no release+3d cutoff) so the plot can extend
        # past Apr 13. Outlier removal still applied to keep the smoother
        # honest; spline is fit on the clipped window so the GCV lambda
        # isn't distorted by data outside the visible range.
        sub = load_upstream_filtered(variable, gt_field_id, gt_scale, EVENT_XMAX)
        if sub.empty:
            raise SystemExit(f"No upstream rows for {variable} ≤ {EVENT_XMAX:%Y-%m-%d}")
    else:
        sub = df_all[df_all["variable"] == variable].copy()
        sub["timestamp_local"] = pd.to_datetime(sub["timestamp_local"], errors="coerce")
        sub = sub.dropna(subset=["timestamp_local", "value"]).sort_values("timestamp_local")
        if sub.empty:
            raise SystemExit(f"No rows for variable {variable}")

    smoothed = compute_spline(sub)
    raw_display = aggregate_raw(sub, RAW_AGGREGATE_N)

    fig, ax = plt.subplots(figsize=(10, 4.0), facecolor="white")
    ax.set_facecolor("white")

    spline_color = GPT_TEAL if variant == "event_study" else SPLINE_RED

    ax.plot(
        raw_display["timestamp_local"], raw_display["value"],
        linestyle="none", marker="o", markersize=2.2, alpha=0.35, color=STEEL_GREY,
    )
    ax.plot(smoothed.index, smoothed.values, linewidth=spline_linewidth, color=spline_color)

    if variant == "default":
        gt_value, release_dt = lookup_gt(gt_field_id, gt_scale)
        ax.axhline(gt_value, color=GT_RED, linewidth=1.4, linestyle="--", alpha=0.9)
        ax.axvline(release_dt, color=RELEASE_BLACK, linewidth=1.4, linestyle="--", alpha=0.9)
    else:
        ax.axvline(EVENT_TIME, color=RELEASE_BLACK, linewidth=1.4, linestyle="--", alpha=0.9)

    t_min = sub["timestamp_local"].min()
    t_max = EVENT_XMAX if variant == "event_study" else sub["timestamp_local"].max()
    set_n_ticks(ax, t_min, t_max, n=5)
    ax.set_xlim(t_min, t_max)

    ax.set_title(title)
    ax.set_xlabel("Nowcast Time")
    ax.set_ylabel(ylabel)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> int:
    if not PROCESSED_CSV.exists():
        raise SystemExit(f"Missing processed CSV: {PROCESSED_CSV}")
    if not GT_CSV.exists():
        raise SystemExit(f"Missing GT CSV: {GT_CSV}")

    configure_style()
    df_all = pd.read_csv(PROCESSED_CSV)

    for variable, title, ylabel, gt_field, gt_scale in PANELS:
        # Default-variant render is intentionally skipped: render_paper_all_macro_2026_03.py
        # is the canonical source for cpi_yoy.png / pce_price_index_mom.png (GPT teal).
        # Event-study variant (no GT, single event line at Apr 8 10:30,
        # x clipped to Apr 13, spline in GPT teal).
        render_panel(
            df_all, variable, title, ylabel, gt_field, gt_scale,
            OUTPUT_DIR / f"{variable}_event_study.png",
            variant="event_study",
            spline_linewidth=3.5,
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
