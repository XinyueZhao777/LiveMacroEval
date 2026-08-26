#!/usr/bin/env python3
"""Re-render every 2026-03 core-macro spline-overlay plot with paper-figure styling.

Two destinations, same styling:

  Main folder (release + 3 days, byte-identical data to the existing PNGs):
    Results/remove_outlier_and_plot/generated_serverA_serverB/
      final_analysis_data_smoother_spline/spline_overlay/
      model_gpt-5-search-api/2026-03_core_macroeconomic_conditions/
        {variable}.png

  release_plus_5d/ subfolder (release + 5 days, rebuilt from upstream):
    .../2026-03_core_macroeconomic_conditions/release_plus_5d/{variable}.png

Main folder reuses the already-outlier-filtered + release-clipped rows
mirrored under processed_final_analysis_data/ — the same source the
existing CPI/PCE paper PNGs (render_paper_cpi_pce.py) draw from.

The 5-day variant sources upstream final_analysis_data and re-runs the
same outlier removal that generate_plots_serverA_serverB.py applies in
prepare_variable_data (pre-release: segment median+MAD; post-release:
GT-anchored with the pre-MAD scale; segment-MAD fallback when pre is
empty / pre_scale = 0), then clips to release + 5 days.

The two _event_study.png files (cpi_yoy, pce_price_index_mom) are left
untouched — render_paper_cpi_pce.py is the canonical source for those.

Conda env: livemacro.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from scipy.interpolate import make_smoothing_spline

REPO_ROOT = Path(__file__).resolve().parents[1]

# Reuse the canonical outlier-removal logic from the main pipeline so the
# 5-day variant is filtered identically to what the existing default plots
# use, just with a different release cutoff.
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
UPSTREAM_CSV = (
    REPO_ROOT
    / "data_from_serverA_serverB"
    / "final_analysis_data"
    / "model_gpt-5-search-api"
    / "2026-03_core_macroeconomic_conditions.csv"
)
GT_CSV = REPO_ROOT / "ground_truth" / "data" / "field_releases_live.csv"

MAIN_DIR = (
    REPO_ROOT
    / "remove_outlier_and_plot"
    / "generated_serverA_serverB"
    / "final_analysis_data_smoother_spline"
    / "spline_overlay"
    / "model_gpt-5-search-api"
    / "2026-03_core_macroeconomic_conditions"
)
SUBDIR_5D = MAIN_DIR / "release_plus_5d"

RELEASE_CUTOFF_DAYS_5D = 5.0
RAW_AGGREGATE_N = 5

# Palette: GPT teal (matches the event-study variant in render_paper_cpi_pce.py
# and MODEL_STYLES["gpt-5-search-api"] in plot_continuous_feb_mar.py) for the
# smoothed spline; muted grey scatter; dark-red GT ref line; near-black
# release line.
STEEL_GREY = "#708090"
GPT_TEAL = "#298c8c"
GT_RED = "#8B0000"
RELEASE_BLACK = "#1C1C1C"

# (variable, title, ylabel, gt_field_id, gt_scale, ref_period)
PANELS = [
    ("cpi_yoy",                  "CPI",                    "Year-over-year growth (%)",       "cpi",                  1.0,   "2026-03"),
    ("ppi_mom",                  "PPI",                    "Month-over-month growth (%)",     "ppi",                  1.0,   "2026-03"),
    ("industrial_production_mom","Industrial Production",  "Month-over-month growth (%)",     "industrial_production",1.0,   "2026-03"),
    ("pce_price_index_mom",      "PCE Price Index",        "Month-over-month growth (%)",     "pce_price_index",      1.0,   "2026-03"),
    ("durable_goods_mom",        "Durable Goods",          "Month-over-month growth (%)",     "durable_goods",        1.0,   "2026-03"),
    ("ism_manufacturing_index",  "ISM Manufacturing",      "Index",                           "ism_manufacturing",    1.0,   "2026-03"),
    ("nonfarm_payrolls_change",  "Nonfarm Payrolls",       "Change (thousands)",              "nonfarm_payrolls",     0.001, "2026-03"),
    ("unemployment_rate",        "Unemployment Rate",      "Rate (%)",                        "unemployment_rate",    1.0,   "2026-03"),
    ("real_gdp_qoq",             "Real GDP",               "Quarter-over-quarter growth (%)", "real_gdp_advance",     1.0,   "2026-Q1"),
]


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["TeX Gyre Heros", "Helvetica Neue", "Helvetica",
                            "Nimbus Sans", "Liberation Sans", "Arial",
                            "DejaVu Sans"],
        "font.weight": "normal",
        "font.size": 20,
        "axes.titlesize": 28,
        "axes.titleweight": "normal",
        "axes.titlecolor": "#0e0e0e",
        "axes.titlepad": 12,
        "axes.labelsize": 24,
        "axes.labelweight": "normal",
        "axes.labelcolor": "#1a1a1a",
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
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


def lookup_gt(field_id: str, ref_period: str, scale: float) -> tuple[float, pd.Timestamp]:
    gt = pd.read_csv(GT_CSV)
    rows = gt[(gt["field_id"] == field_id) & (gt["ref_period"] == ref_period)]
    if rows.empty:
        raise SystemExit(f"No GT row for field_id={field_id} ref={ref_period}")
    row = rows.sort_values("release_datetime_et").iloc[0]
    return float(row["A"]) * scale, pd.Timestamp(row["release_datetime_et"])


def load_filtered_upstream(
    variable: str,
    release_dt: pd.Timestamp,
    gt_value: float,
    cutoff_days: float,
) -> pd.DataFrame:
    """Pull upstream rows for one variable, clip to release + cutoff_days, and
    apply the same outlier filter used by generate_plots_serverA_serverB.py's
    prepare_variable_data. ``level``-form variables drop non-positive values
    to mirror the upstream guard in that function.
    """
    df = pd.read_csv(UPSTREAM_CSV)
    df["timestamp_local"] = pd.to_datetime(df["timestamp_local"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    sub = (
        df[df["variable"] == variable]
        .dropna(subset=["timestamp_local", "value"])
        .sort_values("timestamp_local")
    )
    if "level" in variable.lower():
        sub = sub[sub["value"] > 0]
    cutoff = release_dt + pd.Timedelta(days=cutoff_days)
    sub = sub[sub["timestamp_local"] <= cutoff]
    if sub.empty:
        return sub

    pre = sub[sub["timestamp_local"] < release_dt]
    post = sub[sub["timestamp_local"] >= release_dt]

    if pre.empty:
        pre_filt = pre
    else:
        pre_filt, _, _ = remove_outliers(pre)

    pre_scale = _robust_scale(pre["value"]) if not pre.empty else 0.0
    if post.empty:
        post_filt = post
    elif pre_scale > 0:
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


def compute_spline(df: pd.DataFrame) -> pd.Series:
    """Days-scale GCV cubic smoothing spline — same algorithm as
    compute_smoothing_spline in generate_plots_serverA_serverB.py."""
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
    if n < 2 or df.empty:
        return df
    s = df.sort_values("timestamp_local").reset_index(drop=True)
    bucket = s.index // n
    return s.groupby(bucket).agg(
        timestamp_local=("timestamp_local", "mean"),
        value=("value", "mean"),
    )


def set_n_ticks(ax: plt.Axes, t_min: pd.Timestamp, t_max: pd.Timestamp, n: int = 5) -> None:
    ticks = pd.date_range(t_min, t_max, periods=n)
    ax.set_xticks(ticks)
    span_days = (t_max - t_min).total_seconds() / 86400.0
    fmt = "%b %d" if span_days <= 60 else "%Y-%m-%d"
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))


def render_panel(
    sub: pd.DataFrame,
    title: str,
    ylabel: str,
    gt_value: float,
    release_dt: pd.Timestamp,
    out_path: Path,
    spline_linewidth: float = 2.4,
) -> None:
    sub = sub.copy()
    sub["timestamp_local"] = pd.to_datetime(sub["timestamp_local"], errors="coerce")
    sub = sub.dropna(subset=["timestamp_local", "value"]).sort_values("timestamp_local")
    if sub.empty:
        print(f"  skipping {out_path.name} (no rows)")
        return

    smoothed = compute_spline(sub)
    raw_display = aggregate_raw(sub, RAW_AGGREGATE_N)

    fig, ax = plt.subplots(figsize=(17, 5.5), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(
        raw_display["timestamp_local"], raw_display["value"],
        linestyle="none", marker="o", markersize=2.2, alpha=0.35, color=STEEL_GREY,
    )
    ax.plot(smoothed.index, smoothed.values, linewidth=spline_linewidth, color=GPT_TEAL)
    ax.axhline(gt_value, color=RELEASE_BLACK, linewidth=1.4, linestyle="--", alpha=0.9)
    ax.axvline(release_dt, color=RELEASE_BLACK, linewidth=1.4, linestyle="--", alpha=0.9)

    t_min = sub["timestamp_local"].min()
    t_max = sub["timestamp_local"].max()
    set_n_ticks(ax, t_min, t_max, n=5)
    ax.set_xlim(t_min, t_max)

    ax.set_title(title)
    ax.set_xlabel("Nowcast Time")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> int:
    for p, label in [(PROCESSED_CSV, "processed"), (UPSTREAM_CSV, "upstream"), (GT_CSV, "ground truth")]:
        if not p.exists():
            raise SystemExit(f"Missing {label} CSV: {p}")

    configure_style()

    # --- Main folder: use processed (release+3d) CSV as-is. ---
    processed_df = pd.read_csv(PROCESSED_CSV)
    print(f"Rendering main paper plots (release+3d) to {MAIN_DIR}")
    for variable, title, ylabel, gt_field, gt_scale, ref_period in PANELS:
        gt_value, release_dt = lookup_gt(gt_field, ref_period, gt_scale)
        sub = processed_df[processed_df["variable"] == variable]
        if sub.empty:
            print(f"  skipping {variable} (not in processed CSV)")
            continue
        render_panel(sub, title, ylabel, gt_value, release_dt,
                     MAIN_DIR / f"{variable}.png")

    # --- 5d subfolder: rebuild from upstream with release+5d cutoff. ---
    print(f"Rendering release+5d plots to {SUBDIR_5D}")
    for variable, title, ylabel, gt_field, gt_scale, ref_period in PANELS:
        gt_value, release_dt = lookup_gt(gt_field, ref_period, gt_scale)
        sub = load_filtered_upstream(
            variable, release_dt, gt_value, RELEASE_CUTOFF_DAYS_5D
        )
        if sub.empty:
            print(f"  skipping {variable} (no upstream rows)")
            continue
        render_panel(sub, title, ylabel, gt_value, release_dt,
                     SUBDIR_5D / f"{variable}.png",
                     spline_linewidth=3.5)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
