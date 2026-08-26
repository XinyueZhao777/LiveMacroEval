#!/usr/bin/env python3
"""Generate thin-line, small-dot plots for macro variables from serverA+serverB combined data.

This script can plot data from multiple sources under data_from_serverA_serverB/:
- final_analysis_data (default): Final cleaned analysis data
- concatenated_serverA_serverB: Raw concatenated data from both servers

Output is saved to remove_outlier_and_plot/generated_serverA_serverB/<data_source>/ and
remove_outlier_and_plot/outlier_diagnostics_serverA_serverB/<data_source>/.

When --data-source is final_analysis_data, the post-outlier-removal rows that
land in the plots are ALSO mirrored to
remove_outlier_and_plot/processed_final_analysis_data/<model>/<csv_stem>.csv
so downstream calculations can reuse the cleaned series without re-running the
filter. Only GT-mapped variables are written (the same set the plots emit);
each CSV preserves the original column schema and is clipped to release + 3 d.

Examples:

# Generate all plots from final_analysis_data (default)
python plots/generate_plots_serverA_serverB.py

# Generate plots from concatenated_serverA_serverB
python plots/generate_plots_serverA_serverB.py --data-source concatenated_serverA_serverB

# Generate plots from start to 12.15 23:34:26 (no filters)
python plots/generate_plots_serverA_serverB.py --time-start "start" --time-end "12.15 23:34:26"

# Generate plots from 12.18 15:23:11 to end
python plots/generate_plots_serverA_serverB.py --time-start "12.18 15:23:11" --time-end "end"

# Process only one folder
python plots/generate_plots_serverA_serverB.py --folders model_gpt-5-search-api

# Combine time range and folder filtering
python plots/generate_plots_serverA_serverB.py --time-start "12.06 09:00:00" --time-end "12.20 18:00:00" --folders model_gpt-5-search-api

# Generate raw and moving average overlay plots with 3-day window
python plots/generate_plots_serverA_serverB.py --plot-types raw ma_overlay --ma-window 3D --ma-min-periods 2

12.15 23:34:26 version 1 prompt
12.18 15:23:11 version 2 prompt
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
# Base directory for serverA+serverB data
DATA_BASE_DIR = REPO_ROOT / "data_from_serverA_serverB"
# Available data sources
DATA_SOURCE_CHOICES = ("final_analysis_data", "final_analysis_data_0709", "concatenated_serverA_serverB")
DEFAULT_DATA_SOURCE = "final_analysis_data"
# Output to separate subfolder for serverA_serverB plots
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "generated_serverA_serverB"
DEFAULT_OUTLIER_DIR = Path(__file__).resolve().parent / "outlier_diagnostics_serverA_serverB"
# Cleaned per-CSV mirror written only for data_source=final_analysis_data, so
# downstream code can reuse the same outlier-removed + release-clipped rows
# that feed the plots.
DEFAULT_PROCESSED_DIR = Path(__file__).resolve().parent / "processed_final_analysis_data"
PROCESSED_DATA_SOURCE = "final_analysis_data"
DEFAULT_GROUND_TRUTH_PATH = REPO_ROOT / "ground_truth" / "data" / "field_releases_live.csv"

OUTLIER_Z_THRESHOLD = 10.0
MIN_RELATIVE_DEVIATION = 0.5
# Post-release outlier detection: predictions are expected to sit at the
# (now-known) ground truth, so we anchor on GT and use the pre-release MAD as
# the natural-noise scale. The threshold is stricter than pre-release
# because we know the truth — any meaningful deviation is suspect.
POST_OUTLIER_Z_THRESHOLD = 5.0
# Fallback when the pre-release segment is empty or has zero MAD: flag any
# post-release value whose relative deviation from GT exceeds this fraction.
POST_RELEASE_REL_THRESHOLD = 0.20
DEFAULT_MA_WINDOWS = ("3D",)
PLOT_TYPE_CHOICES = ("raw", "ma", "ma_overlay")
PLOT_MODE_CHOICES = ("simple", "error")
DEFAULT_PLOT_MODES = list(PLOT_MODE_CHOICES)
DEFAULT_RELEASE_CUTOFF_DAYS = 3.0
ERROR_OUTPUT_SUFFIX = "_error"

# Color scheme constants used across plot routines.
STEEL_BLUE = "#4682B4"
BRICK_RED = "#A52A2A"
SLATE = "#708090"
GROUND_TRUTH_COLOR = "#8B0000"  # dark red, used for the horizontal GT ref line
RELEASE_LINE_COLOR = "#1C1C1C"  # near-black, used for the vertical release-time line

# When True, only emit plots for variables that have a ground-truth release
# mapping (see VARIABLE_TO_FIELD_ID). Other variants of the same indicator
# (e.g., cpi_index, cpi_mom when only cpi_yoy has GT) are skipped, matching
# what the error-plot path already does. Flip to False to restore plotting
# of every variant.
ONLY_PLOT_GT_VARIABLES = True

# Map prediction-variable name -> (ground_truth field_id, scale applied to A).
# scale * ground_truth.A produces a value in the same units as the prediction,
# so error = prediction - scale * A is meaningful.
#
# Only direct-form matches are listed: the prediction variable and the
# corresponding ground-truth field must both report the same quantity (YoY
# for cpi, MoM for ppi/industrial_production/pce_price_index/durable_goods/
# retail_sales/real_pce, raw level for ism_*/housing/sales, change-in-jobs
# for nonfarm_payrolls, rate for unemployment_rate, QoQ for real_gdp_advance).
# Variables whose form does not match any released field (e.g. cpi_mom,
# ppi_yoy, *_index where GT is MoM-only, real_dpi_*, nonfarm_payrolls_level)
# are intentionally omitted; those plots fall through with no GT ref line,
# no release cutoff and no error plot.
VARIABLE_TO_FIELD_ID: dict[str, tuple[str, float]] = {
    "cpi_yoy": ("cpi", 1.0),
    "ppi_mom": ("ppi", 1.0),
    "industrial_production_mom": ("industrial_production", 1.0),
    "pce_price_index_mom": ("pce_price_index", 1.0),
    "durable_goods_mom": ("durable_goods", 1.0),
    "retail_sales_mom": ("retail_sales", 1.0),
    "real_pce_mom": ("real_pce", 1.0),
    "ism_manufacturing_index": ("ism_manufacturing", 1.0),
    "ism_services_index": ("ism_services", 1.0),
    # Ground-truth nonfarm_payrolls is in raw counts; predictions are in thousands.
    "nonfarm_payrolls_change": ("nonfarm_payrolls", 0.001),
    "unemployment_rate": ("unemployment_rate", 1.0),
    "real_gdp_qoq": ("real_gdp_advance", 1.0),
    # Housing/sales levels: GT is in raw counts, predictions are in thousands.
    "building_permits_level": ("building_permits", 0.001),
    "existing_home_sales_level": ("existing_home_sales", 0.001),
    "housing_starts_level": ("housing_starts", 0.001),
    "new_home_sales_level": ("new_home_sales", 0.001),
}


def parse_time_range(time_str: str, year: int = 2026) -> datetime | None:
    """
    Parse time string in format 'MM.DD HH:MM:SS' or keywords 'start'/'end'.

    Args:
        time_str: Time string to parse (e.g., '12.15 23:34:26', 'start', 'end')
        year: Year to use for the timestamp (default: 2026)

    Returns:
        datetime object or None for 'start'/'end' keywords
    """
    time_str = time_str.strip().lower()
    if time_str == "start":
        return None
    if time_str == "end":
        return None

    try:
        parts = time_str.split()
        if len(parts) != 2:
            raise ValueError(f"Expected format 'MM.DD HH:MM:SS', got '{time_str}'")

        date_part = parts[0]
        time_part = parts[1]

        month, day = date_part.split(".")
        hour, minute, second = time_part.split(":")

        return datetime(
            year=year,
            month=int(month),
            day=int(day),
            hour=int(hour),
            minute=int(minute),
            second=int(second),
        )
    except Exception as e:
        raise ValueError(f"Failed to parse time '{time_str}': {e}")


def load_ground_truth(path: Path) -> pd.DataFrame:
    """Load the ground-truth field-release table.

    Returns a DataFrame with columns: field_id, release_datetime_et, ref_period, A.
    """
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"field_id", "release_datetime_et", "ref_period", "A"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Ground-truth CSV missing columns: {sorted(missing)}")
    df["field_id"] = df["field_id"].astype(str).str.strip()
    df["ref_period"] = df["ref_period"].astype(str).str.strip()
    df["release_datetime_et"] = pd.to_datetime(df["release_datetime_et"], errors="coerce")
    df["A"] = pd.to_numeric(df["A"], errors="coerce")
    df = df.dropna(subset=["field_id", "ref_period", "release_datetime_et", "A"])
    return df


def target_month_to_ref_period(variable: str, target_month: str) -> str:
    """Translate a prediction target_month (YYYY-MM) into a ground-truth ref_period.

    Quarterly variables map YYYY-MM to YYYY-Qn (e.g., 2025-12 -> 2025-Q4) using
    (month - 1) // 3 + 1. Monthly variables pass through unchanged.
    """
    quarterly_variables = {"real_gdp_qoq"}
    if variable not in quarterly_variables:
        return target_month
    try:
        year_str, month_str = target_month.split("-")
        month = int(month_str)
        quarter = (month - 1) // 3 + 1
        return f"{year_str}-Q{quarter}"
    except Exception:
        return target_month


def lookup_ground_truth(
    variable: str,
    target_month: str | None,
    gt_df: pd.DataFrame | None,
) -> tuple[float | None, pd.Timestamp | None]:
    """Return (gt_value, release_dt) for a (variable, target_month) or (None, None).

    The returned gt_value is already scaled to match the prediction unit using
    VARIABLE_TO_FIELD_ID's scale factor.
    """
    if gt_df is None or not target_month or variable not in VARIABLE_TO_FIELD_ID:
        return None, None
    field_id, scale = VARIABLE_TO_FIELD_ID[variable]
    ref_period = target_month_to_ref_period(variable, target_month)
    rows = gt_df[(gt_df["field_id"] == field_id) & (gt_df["ref_period"] == ref_period)]
    if rows.empty:
        return None, None
    row = rows.sort_values("release_datetime_et").iloc[0]
    return float(row["A"]) * scale, pd.Timestamp(row["release_datetime_et"])


def apply_release_cutoff(
    df: pd.DataFrame,
    release_dt: pd.Timestamp | None,
    cutoff_days: float = DEFAULT_RELEASE_CUTOFF_DAYS,
) -> pd.DataFrame:
    """Restrict df to timestamps <= release_dt + cutoff_days. Pass-through if no release."""
    if release_dt is None:
        return df
    cutoff = release_dt + pd.Timedelta(days=cutoff_days)
    return df[df["timestamp_local"] <= cutoff]


def resolve_target_month(filtered_df: pd.DataFrame, csv_stem: str) -> str | None:
    """Resolve the target_month for a variable's filtered DataFrame.

    Prefers a unique target_month from the data; otherwise falls back to the
    YYYY-MM prefix of the CSV file name.
    """
    if "target_month" in filtered_df.columns:
        values = filtered_df["target_month"].dropna().astype(str).unique().tolist()
        if len(values) == 1 and values[0]:
            return values[0]
    match = re.match(r"^(\d{4}-\d{2})_", csv_stem)
    if match:
        return match.group(1)
    return None


def sanitize_filename(value: str) -> str:
    """Return a filesystem-friendly representation of a variable name."""
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", value).strip("_")
    return cleaned or "variable"


def parse_window_token(token: str) -> str | int:
    """Parse a rolling window token into an int (points) or str (time offset)."""
    token = token.strip()
    if token.isdigit():
        return int(token)
    return token


def format_window_label(window: str | int, for_display: bool = False) -> str:
    """Create a human-friendly window label for titles and folders.

    Args:
        window: The window specification (e.g., "3D", 5)
        for_display: If True, return full human-readable label for plot legends
    """
    if isinstance(window, int):
        if for_display:
            return f"Moving Ave {window} points"
        return f"{window}pt"
    # Handle time-based windows like "3D", "7D"
    window_str = str(window)
    if for_display:
        # Convert "3D" -> "Moving Ave 3 days"
        match = re.match(r"(\d+)([DHMShms])", window_str)
        if match:
            num = match.group(1)
            unit = match.group(2).upper()
            unit_names = {"D": "days", "H": "hours", "M": "minutes", "S": "seconds"}
            unit_name = unit_names.get(unit, unit)
            return f"Moving Ave {num} {unit_name}"
        return f"Moving Ave {window_str}"
    return window_str


def humanize_variable_name(variable: str) -> str:
    """Convert variable names to human-readable format.

    Examples:
        cpi_yoy -> CPI Growth YoY
        nonfarm_payrolls_change -> Nonfarm Payrolls Change
        unemployment_rate -> Unemployment Rate
    """
    # Mapping for common variable names
    variable_mappings = {
        "cpi_yoy": "CPI Growth YoY",
        "cpi_mom": "CPI Growth MoM",
        "core_cpi_yoy": "Core CPI Growth YoY",
        "core_cpi_mom": "Core CPI Growth MoM",
        "nonfarm_payrolls_change": "Nonfarm Payrolls Change",
        "unemployment_rate": "Unemployment Rate",
        "gdp_growth": "GDP Growth",
        "pce_yoy": "PCE Growth YoY",
        "pce_mom": "PCE Growth MoM",
    }

    # Check for exact match (case-insensitive)
    var_lower = variable.lower().strip()
    if var_lower in variable_mappings:
        return variable_mappings[var_lower]

    # Fallback: convert underscores to spaces and title case
    # But preserve acronyms like YoY, MoM, CPI, GDP
    words = variable.replace("_", " ").split()
    acronyms = {"yoy", "mom", "cpi", "gdp", "pce", "ppi"}
    result_words = []
    for word in words:
        if word.lower() in acronyms:
            result_words.append(word.upper())
        else:
            result_words.append(word.capitalize())
    return " ".join(result_words)


def extract_month_from_suffix(title_suffix: str) -> str | None:
    """Extract month info (e.g., 'Dec 2025') from title suffix.

    The title_suffix typically contains patterns like 'Target Dec-2025' or 'Release Jan-2026'.
    We extract the target month for the title.
    """
    # Match "Target Month-Year" format (e.g., "Target Dec-2025")
    match = re.search(r"Target\s+([A-Za-z]{3})-?(\d{4})", title_suffix)
    if match:
        month = match.group(1).capitalize()
        year = match.group(2)
        return f"{month} {year}"

    # Match "Target Year-Month" format (e.g., "Target 2025-12")
    match = re.search(r"Target\s+(\d{4})-(\d{2})", title_suffix)
    if match:
        year = match.group(1)
        month_num = int(match.group(2))
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        if 1 <= month_num <= 12:
            return f"{month_names[month_num - 1]} {year}"

    return None


def build_plot_title(variable: str, title_suffix: str, extra: str | None = None) -> str:
    """Compose a plot title with human-readable variable name and month."""
    human_var = humanize_variable_name(variable)

    # Extract month info from title_suffix
    month_info = extract_month_from_suffix(title_suffix)

    # Build title: "CPI Growth YoY, Dec 2025" or "CPI Growth YoY, Dec 2025, Moving Ave 3 days"
    parts = [human_var]
    if month_info:
        parts.append(month_info)
    if extra:
        parts.append(extra)

    return ", ".join(parts)


def generate_folder_suffix(
    folders: list[str] | None = None,
    time_start_str: str | None = None,
    time_end_str: str | None = None,
) -> str:
    """
    Generate a subdirectory name suffix based on time range and model filters.

    Returns:
        Suffix string with leading underscore or empty string if no filters applied.
    """
    parts = []

    if folders:
        if len(folders) == 1:
            folder_name = folders[0].replace("model_", "")
            parts.append(folder_name)
        else:
            parts.append(f"{len(folders)}_models")

    time_parts = []
    if time_start_str and time_start_str.strip().lower() != "start":
        try:
            date_part = time_start_str.split()[0].replace(".", "")
            time_parts.append(date_part)
        except:
            time_parts.append("start")

    if time_end_str and time_end_str.strip().lower() != "end":
        try:
            date_part = time_end_str.split()[0].replace(".", "")
            if time_parts:
                time_parts.append(date_part)
            else:
                time_parts = ["start", date_part]
        except:
            if not time_parts:
                time_parts.append("end")

    if time_parts:
        if len(time_parts) == 2:
            parts.append(f"{time_parts[0]}-{time_parts[1]}")
        else:
            parts.append(time_parts[0])

    return "_" + "_".join(parts) if parts else ""


def load_csv(
    csv_path: Path,
    time_start: datetime | None = None,
    time_end: datetime | None = None,
) -> pd.DataFrame:
    """
    Load and clean a single CSV file with optional time filtering.
    """
    df = pd.read_csv(csv_path)

    if "timestamp_local" not in df.columns or "variable" not in df.columns:
        raise ValueError(f"{csv_path} missing required columns.")

    df["timestamp_local"] = pd.to_datetime(df["timestamp_local"], errors="coerce")
    df["variable"] = df["variable"].astype(str).str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.replace({"variable": {"": pd.NA}})
    df = df.dropna(subset=["timestamp_local", "value", "variable"])

    if time_start is not None:
        df = df[df["timestamp_local"] >= time_start]
    if time_end is not None:
        df = df[df["timestamp_local"] <= time_end]

    return df


def prepare_variable_data(
    df: pd.DataFrame,
    variable: str,
    release_dt: pd.Timestamp | None,
    cutoff_days: float,
    gt_value: float | None = None,
) -> tuple[pd.DataFrame, int, pd.DataFrame]:
    """Extract and clean the time series for a single variable.

    If a ground-truth release date is available, restrict the series to
    [.. release_dt + cutoff_days] and run outlier removal separately on the
    pre-release and post-release segments. Pre uses the standard segment-
    based MAD rule. Post is anchored at gt_value (since the model is
    supposed to be reporting the released number) and scaled by the pre-
    segment's MAD — pre's natural prediction noise. This catches small
    post-release deviations that would otherwise hide under a tight cluster
    at GT (where the segment's own MAD collapses to 0).

    With no release date, return the series as-is (no outlier removal).
    """
    var_df = df[df["variable"] == variable].sort_values("timestamp_local")
    if "level" in variable.lower():
        var_df = var_df[var_df["value"] > 0]
    empty = pd.DataFrame(columns=df.columns)
    if var_df.empty:
        return empty, 0, empty
    if release_dt is None:
        return var_df, 0, empty
    cutoff = release_dt + pd.Timedelta(days=cutoff_days)
    var_df = var_df[var_df["timestamp_local"] <= cutoff]
    if var_df.empty:
        return var_df, 0, empty
    pre = var_df[var_df["timestamp_local"] < release_dt]
    post = var_df[var_df["timestamp_local"] >= release_dt]
    pre_filt, pre_removed, pre_removed_rows = remove_outliers(pre)

    pre_scale = _robust_scale(pre["value"]) if not pre.empty else 0.0
    if gt_value is not None and pre_scale > 0:
        post_filt, post_removed, post_removed_rows = remove_outliers(
            post,
            anchor=float(gt_value),
            scale=pre_scale,
            z_threshold=POST_OUTLIER_Z_THRESHOLD,
            apply_relative_guard=False,
        )
    elif gt_value is not None:
        post_filt, post_removed, post_removed_rows = _relative_outlier_split(
            post, float(gt_value), POST_RELEASE_REL_THRESHOLD
        )
    else:
        post_filt, post_removed, post_removed_rows = remove_outliers(post)

    filtered_df = pd.concat([pre_filt, post_filt]).sort_values("timestamp_local")
    removed_rows = pd.concat([pre_removed_rows, post_removed_rows]).sort_values("timestamp_local")
    return filtered_df, pre_removed + post_removed, removed_rows


def finalize_plot(
    fig: plt.Figure,
    ax: plt.Axes,
    destination: Path,
    title: str,
    removed: int,
    ref_line_value: float | None = None,
    ref_line_label: str | None = None,
    ref_line_color: str | None = None,
    show_legend: bool = False,
    y_label: str = "Value",
    release_dt: pd.Timestamp | None = None,
) -> None:
    """Apply common styling, optionally draw a horizontal reference line and a
    vertical ground-truth-release line, and save."""
    if ref_line_value is not None and ref_line_color is not None:
        ax.axhline(
            ref_line_value,
            color=ref_line_color,
            linewidth=1.0,
            linestyle="--",
            alpha=0.85,
            label=ref_line_label,
        )
    if release_dt is not None:
        ax.axvline(
            release_dt,
            color=RELEASE_LINE_COLOR,
            linewidth=1.0,
            linestyle="--",
            alpha=0.85,
            label="GT release",
        )
    ax.set_title(title)
    ax.set_xlabel("Timestamp (local)")
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    if handles and (show_legend or ref_line_label or release_dt is not None):
        ax.legend(loc="best", fontsize=8, frameon=False)
    annotation = f"Outliers removed: {removed}"
    ax.text(
        0.99,
        0.02,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="dimgray",
    )
    fig.autofmt_xdate(rotation=25)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def plot_raw_series(
    filtered_df: pd.DataFrame,
    variable: str,
    destination: Path,
    title_suffix: str,
    removed: int,
    extra: str | None = None,
    ref_line_value: float | None = None,
    ref_line_label: str | None = None,
    ref_line_color: str | None = None,
    y_label: str = "Value",
    release_dt: pd.Timestamp | None = None,
) -> bool:
    """Plot the raw time series with thin lines and small dots."""
    if filtered_df.empty:
        return False
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(
        filtered_df["timestamp_local"],
        filtered_df["value"],
        linewidth=0.8,
        marker="o",
        markersize=2,
        alpha=0.9,
        color=STEEL_BLUE,
    )
    title = build_plot_title(variable, title_suffix, extra)
    finalize_plot(
        fig,
        ax,
        destination,
        title,
        removed,
        ref_line_value=ref_line_value,
        ref_line_label=ref_line_label,
        ref_line_color=ref_line_color,
        y_label=y_label,
        release_dt=release_dt,
    )
    return True


def compute_moving_average(
    filtered_df: pd.DataFrame,
    window: str | int,
    min_periods: int,
    center: bool,
) -> pd.Series:
    """Compute a rolling mean using a point or time-based window."""
    series = filtered_df.set_index("timestamp_local")["value"].sort_index()
    return series.rolling(window=window, min_periods=min_periods, center=center).mean()


def compute_smoothing_spline(
    filtered_df: pd.DataFrame,
    lam: float | None = None,
) -> pd.Series:
    """Fit a cubic smoothing spline and evaluate at the input timestamps.

    x is rescaled to *days* since the first sample. On a seconds-scale x,
    SciPy's ``make_smoothing_spline`` GCV picks a numerically tiny λ and
    effectively interpolates the noise; days-scale x lets GCV select a
    sensible λ that produces a real trend. See:
        https://docs.scipy.org/doc/scipy-1.16.0/tutorial/interpolate/smoothing_splines.html

    ``make_smoothing_spline`` requires a strictly increasing x, so values
    sharing a timestamp are averaged first. The resulting Series is indexed
    by *unique* timestamp. ``lam=None`` selects λ via generalized cross-
    validation; pass a positive float to override (larger λ → smoother).
    Returns an empty Series when fewer than 5 unique timestamps remain —
    SciPy's ``make_smoothing_spline`` requires ``len(x) >= 5``.
    """
    from scipy.interpolate import make_smoothing_spline

    series = filtered_df.set_index("timestamp_local")["value"].sort_index()
    series = series.groupby(level=0).mean()
    if series.size < 5:
        return pd.Series(dtype=float, index=series.index)
    x_ns = series.index.astype("int64").to_numpy()
    x_days = (x_ns - x_ns[0]).astype("float64") / (1e9 * 86400.0)
    y = series.to_numpy(dtype="float64")
    spline = make_smoothing_spline(x_days, y, lam=lam)
    return pd.Series(spline(x_days), index=series.index)


def plot_moving_average(
    filtered_df: pd.DataFrame,
    smoothed: pd.Series,
    variable: str,
    destination: Path,
    title_suffix: str,
    removed: int,
    window_label: str,
    window_display_label: str,
    overlay: bool,
    extra_suffix: str | None = None,
    ref_line_value: float | None = None,
    ref_line_label: str | None = None,
    ref_line_color: str | None = None,
    y_label: str = "Value",
    release_dt: pd.Timestamp | None = None,
    raw_as_dots: bool = False,
    raw_aggregate_n: int = 1,
) -> bool:
    """Plot a moving average series, optionally overlaid on raw data.

    ``raw_as_dots=True`` renders raw points as markers only (no connecting
    line) so the smoothed curve reads cleanly above a faint background of
    raw observations — used by the spline overlay.

    ``raw_aggregate_n`` (>= 2) collapses the displayed raw points into the
    mean of every N consecutive observations (mean of timestamp_local and
    mean of value) to dampen visual noise. The smoothing fit itself is
    unaffected — only the scatter underneath is summarized.
    """
    smoothed = smoothed.dropna()
    if smoothed.empty:
        return False
    if raw_aggregate_n >= 2 and not filtered_df.empty:
        raw_sorted = filtered_df.sort_values("timestamp_local").reset_index(drop=True)
        bucket = raw_sorted.index // raw_aggregate_n
        raw_display = raw_sorted.groupby(bucket).agg(
            timestamp_local=("timestamp_local", "mean"),
            value=("value", "mean"),
        )
    else:
        raw_display = filtered_df
    fig, ax = plt.subplots(figsize=(11, 4))
    if overlay:
        raw_label = (
            f"Raw (mean of {raw_aggregate_n})" if raw_aggregate_n >= 2 else "Raw"
        )
        ax.plot(
            raw_display["timestamp_local"],
            raw_display["value"],
            linestyle="none" if raw_as_dots else "-",
            linewidth=0.0 if raw_as_dots else 0.6,
            marker="o",
            markersize=2.2 if raw_as_dots else 2,
            alpha=0.35,
            color=SLATE,
            label=raw_label,
        )
        ax.plot(
            smoothed.index,
            smoothed.values,
            linewidth=2.0 if raw_as_dots else 1.8,
            color=BRICK_RED,
            label=window_display_label,
        )
    else:
        ax.plot(
            smoothed.index,
            smoothed.values,
            linewidth=1.6,
            color=BRICK_RED,
        )
    extra = window_display_label
    if extra_suffix:
        extra = f"{extra}, {extra_suffix}"
    title = build_plot_title(variable, title_suffix, extra)
    finalize_plot(
        fig,
        ax,
        destination,
        title,
        removed,
        ref_line_value=ref_line_value,
        ref_line_label=ref_line_label,
        ref_line_color=ref_line_color,
        show_legend=overlay,
        y_label=y_label,
        release_dt=release_dt,
    )
    return True


def build_title_suffix(df: pd.DataFrame, csv_name: str) -> str:
    """Construct a short descriptor for plot titles."""
    target_vals = ", ".join(sorted(df.get("target_month", pd.Series(dtype=str)).dropna().unique()))
    release_vals = ", ".join(sorted(df.get("release_month", pd.Series(dtype=str)).dropna().unique()))
    parts: list[str] = []
    if target_vals:
        parts.append(f"Target {target_vals}")
    if release_vals:
        parts.append(f"Release {release_vals}")
    parts.append(csv_name)
    return " | ".join(parts)


def iter_csvs(data_dir: Path, folders: list[str] | None = None) -> Iterable[Path]:
    """
    Yield CSV files under the provided directory, optionally filtered by folder names.
    """
    all_csvs = sorted(data_dir.rglob("*.csv"))

    # Only process model outputs, not auxiliary logs in the data root.
    all_csvs = [
        csv_path for csv_path in all_csvs
        if csv_path.relative_to(data_dir).parts
        and csv_path.relative_to(data_dir).parts[0].startswith("model_")
    ]

    # Filter out anything from the 'archived' folder
    all_csvs = [
        csv_path for csv_path in all_csvs
        if 'archived' not in csv_path.relative_to(data_dir).parts
    ]

    if folders is None:
        return all_csvs

    filtered_csvs = []
    for csv_path in all_csvs:
        rel_path = csv_path.relative_to(data_dir)
        if rel_path.parts and rel_path.parts[0] in folders:
            filtered_csvs.append(csv_path)

    return filtered_csvs


def _robust_scale(values: pd.Series) -> float:
    """Robust spread estimator: MAD with off-median fallback for tied data."""
    if values.empty:
        return 0.0
    abs_diff = (values - values.median()).abs()
    mad = abs_diff.median()
    if mad == 0:
        off = abs_diff[abs_diff > 0]
        return float(off.median()) if not off.empty else 0.0
    return float(mad)


def _relative_outlier_split(
    post_df: pd.DataFrame,
    gt_value: float,
    rel_threshold: float,
) -> tuple[pd.DataFrame, int, pd.DataFrame]:
    """Post-release fallback when no usable scale is available: flag values
    whose relative deviation from GT exceeds rel_threshold."""
    if post_df.empty:
        empty = post_df.iloc[0:0]
        return post_df, 0, empty
    abs_diff = (post_df["value"] - gt_value).abs()
    gt_abs = max(abs(gt_value), 1e-10)
    rel_dev = abs_diff / gt_abs
    mask = rel_dev > rel_threshold
    return post_df.loc[~mask], int(mask.sum()), post_df.loc[mask]


def remove_outliers(
    var_df: pd.DataFrame,
    *,
    anchor: float | None = None,
    scale: float | None = None,
    z_threshold: float = OUTLIER_Z_THRESHOLD,
    apply_relative_guard: bool = True,
) -> tuple[pd.DataFrame, int, pd.DataFrame]:
    """Remove extreme outliers using a robust MAD-based rule.

    Defaults reproduce the segment-based behaviour: center on the segment
    median, scale from the segment's MAD (with off-median fallback for heavy
    ties), z>OUTLIER_Z_THRESHOLD, and a relative-deviation guard against
    over-flagging when the scale is artificially small.

    Pass ``anchor`` and ``scale`` to override (e.g., post-release segment
    anchored on GT with the pre-release MAD as the noise scale). Disable
    ``apply_relative_guard`` when the anchor is a known truth so the rule is
    not blocked by a small relative deviation in absolute-scale variables.
    """
    values = var_df["value"]
    center = float(anchor) if anchor is not None else float(values.median())
    abs_diff = (values - center).abs()

    if scale is not None:
        mad = float(scale)
    else:
        mad = _robust_scale(values)

    if mad > 0:
        robust_z = 0.6745 * abs_diff / mad
        statistical_mask = robust_z > z_threshold
    else:
        statistical_mask = pd.Series(False, index=values.index)

    if apply_relative_guard:
        center_abs = max(abs(center), 1e-10)
        relative_deviation = abs_diff / center_abs
        mask = statistical_mask & (relative_deviation > MIN_RELATIVE_DEVIATION)
    else:
        mask = statistical_mask

    filtered_df = var_df.loc[~mask]
    removed_rows = var_df.loc[mask]
    removed = int(mask.sum())
    return filtered_df, removed, removed_rows


def _emit_plots_for_series(
    series_df: pd.DataFrame,
    variable: str,
    target_dir: Path,
    csv_output_dir: Path,
    title_suffix: str,
    removed_count: int,
    plot_types: list[str],
    ma_windows: list[str | int],
    ma_min_periods: int,
    ma_center: bool,
    extra_suffix: str | None,
    ref_line_value: float | None,
    ref_line_label: str | None,
    ref_line_color: str | None,
    y_label: str,
    release_dt: pd.Timestamp | None = None,
    smoother: str = "ma",
    spline_lam: float | None = None,
) -> list[Path]:
    """Generate raw / ma / ma_overlay plots for a single series under target_dir.

    With ``smoother="ma"`` (default), the smoothed plots use a rolling mean
    over each entry in ``ma_windows`` and land in
    ``moving_average_<label>/`` and ``moving_average_overlay_<label>/``.
    With ``smoother="spline"``, ``ma_windows``/``ma_min_periods``/``ma_center``
    and ``plot_types`` are ignored: only the overlay is emitted (raw points
    as small grey dots beneath a smoothed red line) into ``spline_overlay/``.
    """
    created: list[Path] = []
    if series_df.empty:
        return created
    if smoother == "spline":
        # Spline mode emits ONLY the overlay variant. raw-only and
        # spline-only outputs are intentionally suppressed; the overlay
        # already shows raw observations as faint grey dots.
        smoothed = compute_smoothing_spline(series_df, lam=spline_lam)
        outfile = target_dir / "spline_overlay" / csv_output_dir / f"{sanitize_filename(variable)}.png"
        if plot_moving_average(
            series_df,
            smoothed,
            variable,
            outfile,
            title_suffix,
            removed_count,
            "spline_overlay",
            "Smoothing Spline",
            overlay=True,
            extra_suffix=extra_suffix,
            ref_line_value=ref_line_value,
            ref_line_label=ref_line_label,
            ref_line_color=ref_line_color,
            y_label=y_label,
            release_dt=release_dt,
            raw_as_dots=True,
            raw_aggregate_n=5,
        ):
            created.append(outfile)
        return created
    if "raw" in plot_types:
        outfile = target_dir / "raw" / csv_output_dir / f"{sanitize_filename(variable)}.png"
        if plot_raw_series(
            series_df,
            variable,
            outfile,
            title_suffix,
            removed_count,
            extra=extra_suffix,
            ref_line_value=ref_line_value,
            ref_line_label=ref_line_label,
            ref_line_color=ref_line_color,
            y_label=y_label,
            release_dt=release_dt,
        ):
            created.append(outfile)
    if "ma" in plot_types or "ma_overlay" in plot_types:
        smoother_iters = []
        for window in ma_windows:
            window_label = format_window_label(window)
            window_display_label = format_window_label(window, for_display=True)
            smoothed = compute_moving_average(series_df, window, ma_min_periods, ma_center)
            ma_folder = f"moving_average_{sanitize_filename(window_label)}"
            overlay_folder = f"moving_average_overlay_{sanitize_filename(window_label)}"
            smoother_iters.append((ma_folder, overlay_folder, window_display_label, smoothed))
        for ma_folder, overlay_folder, display_label, smoothed in smoother_iters:
            if "ma" in plot_types:
                outfile = target_dir / ma_folder / csv_output_dir / f"{sanitize_filename(variable)}.png"
                if plot_moving_average(
                    series_df,
                    smoothed,
                    variable,
                    outfile,
                    title_suffix,
                    removed_count,
                    ma_folder,
                    display_label,
                    overlay=False,
                    extra_suffix=extra_suffix,
                    ref_line_value=ref_line_value,
                    ref_line_label=ref_line_label,
                    ref_line_color=ref_line_color,
                    y_label=y_label,
                    release_dt=release_dt,
                ):
                    created.append(outfile)
            if "ma_overlay" in plot_types:
                outfile = target_dir / overlay_folder / csv_output_dir / f"{sanitize_filename(variable)}.png"
                if plot_moving_average(
                    series_df,
                    smoothed,
                    variable,
                    outfile,
                    title_suffix,
                    removed_count,
                    overlay_folder,
                    display_label,
                    overlay=True,
                    extra_suffix=extra_suffix,
                    ref_line_value=ref_line_value,
                    ref_line_label=ref_line_label,
                    ref_line_color=ref_line_color,
                    y_label=y_label,
                    release_dt=release_dt,
                ):
                    created.append(outfile)
    return created


def generate_plots(
    data_dir: Path,
    output_dir: Path,
    outlier_dir: Path,
    folders: list[str] | None = None,
    time_start: datetime | None = None,
    time_end: datetime | None = None,
    time_start_str: str | None = None,
    time_end_str: str | None = None,
    plot_types: list[str] | None = None,
    ma_windows: list[str | int] | None = None,
    ma_min_periods: int = 1,
    ma_center: bool = False,
    plot_modes: list[str] | None = None,
    error_output_dir: Path | None = None,
    gt_df: pd.DataFrame | None = None,
    release_cutoff_days: float = DEFAULT_RELEASE_CUTOFF_DAYS,
    smoother: str = "ma",
    spline_lam: float | None = None,
    processed_output_dir: Path | None = None,
) -> tuple[list[Path], list[Path], list[Path]]:
    """
    Generate plots for CSV files in data_dir and store outlier diagnostics.

    Modes:
      simple : prediction series. If a ground-truth value is available for
               (variable, target_month), draw a dark-red horizontal reference
               line and clip the series to release + release_cutoff_days.
      error  : prediction-minus-ground-truth series. Only emitted for
               variables with a ground-truth mapping; clipped to
               release + release_cutoff_days; ref line at zero.
    """
    folder_suffix = generate_folder_suffix(folders, time_start_str, time_end_str)

    if folder_suffix:
        subdir_name = folder_suffix.lstrip("_")
        output_dir = output_dir / subdir_name
        outlier_dir = outlier_dir / subdir_name
        if error_output_dir is not None:
            error_output_dir = error_output_dir / subdir_name
    plot_types = plot_types or ["raw"]
    ma_windows = ma_windows or [DEFAULT_MA_WINDOWS[0]]
    plot_modes = plot_modes or list(DEFAULT_PLOT_MODES)

    do_simple = "simple" in plot_modes
    do_error = "error" in plot_modes and error_output_dir is not None and gt_df is not None

    created_files: list[Path] = []
    outlier_reports: list[Path] = []
    processed_reports: list[Path] = []
    processed_months: set[str] = set()
    error_skipped_no_gt = 0
    error_emitted = 0
    simple_with_gt = 0
    for csv_path in iter_csvs(data_dir, folders):
        try:
            df = load_csv(csv_path, time_start, time_end)
        except ValueError as exc:
            print(f"Skipping {csv_path.name}: {exc}")
            continue
        if df.empty:
            print(f"Skipping {csv_path.name} (no data after time filtering)")
            continue
        match = re.match(r"^(\d{4}-\d{2})_", csv_path.stem)
        if match:
            processed_months.add(match.group(1))
        title_suffix = build_title_suffix(df, csv_path.stem)
        csv_output_dir = csv_path.relative_to(data_dir).with_suffix("")
        csv_outliers: list[pd.DataFrame] = []
        csv_kept: list[pd.DataFrame] = []
        for variable in sorted(df["variable"].unique()):
            var_only = df[df["variable"] == variable]
            target_month = resolve_target_month(var_only, csv_path.stem)
            gt_value, release_dt = lookup_ground_truth(variable, target_month, gt_df)
            # Skip variables that don't have a ground-truth release: simple
            # plots now match what the error path already does. Flip
            # ONLY_PLOT_GT_VARIABLES to False to restore plotting all units.
            if ONLY_PLOT_GT_VARIABLES and (gt_value is None or release_dt is None):
                if do_error:
                    error_skipped_no_gt += 1
                continue
            filtered_df, removed_count, removed_rows = prepare_variable_data(
                df, variable, release_dt, release_cutoff_days, gt_value=gt_value
            )
            if removed_count > 0:
                csv_outliers.append(removed_rows.assign(variable=variable))
            if filtered_df.empty:
                continue
            if processed_output_dir is not None:
                csv_kept.append(filtered_df)

            simple_df = filtered_df

            if do_simple:
                if release_dt is not None and gt_value is not None:
                    simple_with_gt += 1
                created_files.extend(
                    _emit_plots_for_series(
                        simple_df,
                        variable,
                        output_dir,
                        csv_output_dir,
                        title_suffix,
                        removed_count,
                        plot_types,
                        ma_windows,
                        ma_min_periods,
                        ma_center,
                        extra_suffix=None,
                        ref_line_value=gt_value,
                        ref_line_label="Ground truth" if gt_value is not None else None,
                        ref_line_color=GROUND_TRUTH_COLOR if gt_value is not None else None,
                        y_label="Value",
                        release_dt=release_dt,
                        smoother=smoother,
                        spline_lam=spline_lam,
                    )
                )

            if do_error:
                if gt_value is None or release_dt is None:
                    error_skipped_no_gt += 1
                    continue
                error_df = simple_df.copy()
                error_df["value"] = error_df["value"] - gt_value
                created_for_error = _emit_plots_for_series(
                    error_df,
                    variable,
                    error_output_dir,
                    csv_output_dir,
                    title_suffix,
                    removed_count,
                    plot_types,
                    ma_windows,
                    ma_min_periods,
                    ma_center,
                    extra_suffix="Prediction Error",
                    ref_line_value=0.0,
                    ref_line_label="Zero error",
                    ref_line_color=GROUND_TRUTH_COLOR,
                    y_label="Prediction - Ground Truth",
                    release_dt=release_dt,
                    smoother=smoother,
                    spline_lam=spline_lam,
                )
                if created_for_error:
                    error_emitted += 1
                created_files.extend(created_for_error)
        if csv_outliers:
            outlier_df = pd.concat(csv_outliers, ignore_index=True)
            outlier_path = outlier_dir / csv_path.relative_to(data_dir)
            outlier_path.parent.mkdir(parents=True, exist_ok=True)
            outlier_df.to_csv(outlier_path, index=False)
            outlier_reports.append(outlier_path)
        if processed_output_dir is not None and csv_kept:
            kept_df = pd.concat(csv_kept, ignore_index=True).sort_values(
                ["timestamp_local", "variable"]
            )
            processed_path = processed_output_dir / csv_path.relative_to(data_dir)
            processed_path.parent.mkdir(parents=True, exist_ok=True)
            kept_df.to_csv(processed_path, index=False)
            processed_reports.append(processed_path)
    if processed_months:
        print(f"Processed target months: {', '.join(sorted(processed_months))}")
    else:
        print("Processed target months: none")
    if do_simple:
        print(f"Simple plots with ground-truth ref line: {simple_with_gt}")
    if do_error:
        print(f"Error plots emitted: {error_emitted} (skipped {error_skipped_no_gt} series without GT)")
    return created_files, outlier_reports, processed_reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data source examples:
  --data-source final_analysis_data
  --data-source concatenated_serverA_serverB

Time range examples:
  --time-start "start" --time-end "12.15 23:34:26"
  --time-start "12.18 15:23:11" --time-end "end"
  --time-start "12.06 09:00:00" --time-end "12.20 18:00:00"

Folder examples:
  --folders model_gpt-5-search-api
  --folders model_claude-sonnet-4.5-api model_gpt-5-search-api

Smoothing examples:
  --plot-types raw ma_overlay --ma-window 3D
  --plot-types raw ma ma_overlay --ma-window 3D 7D --ma-min-periods 2
  --plot-types raw ma ma_overlay --smoother spline
  --plot-types raw ma_overlay --smoother spline --spline-lam 1e3
        """,
    )
    parser.add_argument(
        "--data-source",
        default=DEFAULT_DATA_SOURCE,
        choices=DATA_SOURCE_CHOICES,
        help="Data source directory under data_from_serverA_serverB/ (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Path to store generated plots (default: %(default)s)",
    )
    parser.add_argument(
        "--outlier-dir",
        default=str(DEFAULT_OUTLIER_DIR),
        help="Path to store removed outlier rows (default: %(default)s)",
    )
    parser.add_argument(
        "--processed-dir",
        default=str(DEFAULT_PROCESSED_DIR),
        help="Path to mirror the cleaned (outlier-removed, release-clipped) "
        "rows that feed the plots. Only written when --data-source is "
        f"{PROCESSED_DATA_SOURCE}. Default: %(default)s",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        help="Specific folder names to process (e.g., model_gpt-5-search-api). If not specified, all folders are processed.",
    )
    parser.add_argument(
        "--time-start",
        help='Start time filter: "start" or "MM.DD HH:MM:SS" (e.g., "12.15 23:34:26")',
    )
    parser.add_argument(
        "--time-end",
        help='End time filter: "end" or "MM.DD HH:MM:SS" (e.g., "12.18 15:23:11")',
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2026,
        help="Year for time range parsing (default: %(default)s)",
    )
    parser.add_argument(
        "--plot-types",
        nargs="+",
        default=list(PLOT_TYPE_CHOICES),
        choices=PLOT_TYPE_CHOICES,
        help="Plot variants to generate (default: %(default)s)",
    )
    parser.add_argument(
        "--ma-window",
        nargs="+",
        default=list(DEFAULT_MA_WINDOWS),
        help="Moving average window(s), e.g. '3D' or '5' points (default: %(default)s)",
    )
    parser.add_argument(
        "--ma-min-periods",
        type=int,
        default=1,
        help="Minimum periods for moving averages (default: %(default)s)",
    )
    parser.add_argument(
        "--ma-center",
        action="store_true",
        help="Center the moving average window instead of right-aligning it",
    )
    parser.add_argument(
        "--smoother",
        choices=("ma", "spline"),
        default="ma",
        help="Smoother used for non-raw plots: 'ma' (rolling mean over --ma-window, "
        "default) or 'spline' (cubic smoothing spline; ignores --ma-window/--ma-min-periods/--ma-center). "
        "When 'spline', plots land in <data_source>_smoother_spline/ to keep MA outputs untouched.",
    )
    parser.add_argument(
        "--spline-lam",
        type=float,
        default=None,
        help="Smoothing parameter for --smoother spline (larger -> smoother). "
        "Default: chosen automatically by GCV.",
    )
    parser.add_argument(
        "--plot-modes",
        nargs="+",
        default=list(DEFAULT_PLOT_MODES),
        choices=PLOT_MODE_CHOICES,
        help="Plot modes to emit: 'simple' (prediction with optional GT ref line) "
        "and/or 'error' (prediction minus GT). Default: %(default)s",
    )
    parser.add_argument(
        "--ground-truth-csv",
        default=str(DEFAULT_GROUND_TRUTH_PATH),
        help="Path to ground-truth field-release CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--release-cutoff-days",
        type=float,
        default=DEFAULT_RELEASE_CUTOFF_DAYS,
        help="When ground truth is available, clip series to release + N days (default: %(default)s)",
    )
    args = parser.parse_args()

    data_dir = DATA_BASE_DIR / args.data_source
    smoother_suffix = f"_smoother_{args.smoother}" if args.smoother != "ma" else ""
    output_dir = Path(args.output_dir).expanduser().resolve() / f"{args.data_source}{smoother_suffix}"
    outlier_dir = Path(args.outlier_dir).expanduser().resolve() / f"{args.data_source}{smoother_suffix}"
    error_output_dir = (
        Path(args.output_dir).expanduser().resolve()
        / f"{args.data_source}{ERROR_OUTPUT_SUFFIX}{smoother_suffix}"
    )
    # Cleaned-row mirror lives outside the smoother/error split because the
    # outlier filter doesn't depend on either; every run with data-source
    # final_analysis_data refreshes the same canonical folder.
    processed_output_dir: Path | None = None
    if args.data_source == PROCESSED_DATA_SOURCE:
        processed_output_dir = Path(args.processed_dir).expanduser().resolve()

    print(f"Data source: {args.data_source} ({data_dir})")

    if not data_dir.exists():
        raise SystemExit(f"Data directory {data_dir} not found.")

    # Parse time range
    time_start = None
    time_end = None
    if args.time_start:
        time_start = parse_time_range(args.time_start, args.year)
        print(f"Time start: {time_start or 'beginning'}")
    if args.time_end:
        time_end = parse_time_range(args.time_end, args.year)
        print(f"Time end: {time_end or 'end'}")

    # Display folder filter
    if args.folders:
        print(f"Processing folders: {', '.join(args.folders)}")
    else:
        print("Processing all folders")

    plot_types = args.plot_types
    ma_windows = [parse_window_token(token) for token in args.ma_window]
    if "ma" in plot_types or "ma_overlay" in plot_types:
        window_labels = ", ".join(format_window_label(window) for window in ma_windows)
        print(f"Moving average windows: {window_labels}")
    print(f"Plot types: {', '.join(plot_types)}")
    print(f"Plot modes: {', '.join(args.plot_modes)}")

    plot_modes = list(args.plot_modes)
    needs_gt = bool(plot_modes) and ("simple" in plot_modes or "error" in plot_modes)
    gt_df: pd.DataFrame | None = None
    if needs_gt:
        gt_path = Path(args.ground_truth_csv).expanduser().resolve()
        try:
            gt_df = load_ground_truth(gt_path)
            print(f"Loaded ground truth: {gt_path} ({len(gt_df)} releases)")
        except FileNotFoundError as exc:
            if "error" in plot_modes:
                raise SystemExit(str(exc))
            print(f"Warning: {exc}. Simple plots will skip GT ref line / cutoff.")
            gt_df = None

    # Generate folder suffix preview
    folder_suffix = generate_folder_suffix(args.folders, args.time_start, args.time_end)
    if folder_suffix:
        print(f"Output folder suffix: {folder_suffix}")

    created, outlier_reports, processed_reports = generate_plots(
        data_dir,
        output_dir,
        outlier_dir,
        args.folders,
        time_start,
        time_end,
        args.time_start,
        args.time_end,
        plot_types,
        ma_windows,
        args.ma_min_periods,
        args.ma_center,
        plot_modes=plot_modes,
        error_output_dir=error_output_dir,
        gt_df=gt_df,
        release_cutoff_days=args.release_cutoff_days,
        smoother=args.smoother,
        spline_lam=args.spline_lam,
        processed_output_dir=processed_output_dir,
    )
    print(f"Generated {len(created)} plots (simple -> {output_dir}; error -> {error_output_dir})")
    print(f"Saved {len(outlier_reports)} outlier reports in {outlier_dir}")
    if processed_output_dir is not None:
        print(
            f"Saved {len(processed_reports)} cleaned CSVs in {processed_output_dir}"
        )


if __name__ == "__main__":
    main()
