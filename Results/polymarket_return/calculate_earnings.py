#!/usr/bin/env python3
"""
Calculate hypothetical earnings from betting on Polymarket based on model predictions.

This script:
1. Loads model predictions for CPI YoY (supports GPT model format and Fed format)
2. Loads Polymarket hourly price data for November 2025 CPI
3. For each hour/day, uses predictions to decide which bucket to bet on
4. Calculates final earnings when the market resolves
5. Saves summary and detailed results to CSV

Prediction aggregation methods:
- "latest": Use the most recent prediction before the betting time
- "daily_avg": Use the average prediction over the day (for daily betting)
- "12hour_avg": Use the average prediction over the 12 hours before EOD (for daily betting)

Daily betting price logic:
- Uses end-of-day price (beginning of next day, e.g., "11-15-2025 00:00" for day 11-14)
- This represents the closing price at 23:59 of the betting day

Polymarket mechanics:
- Buy shares at price P (0 to 1)
- Winning shares pay $1 each at resolution
- Losing shares pay $0
- Profit = (shares * $1) - cost if won, -cost if lost

example:
python polymarket_return/calculate_earnings.py \
  --frequency hourly \
  --aggregation latest \
  --output-dir polymarket_return/results/bet_hourly_latest
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Variable / bucket configuration. The release datetime, the released value
# (ground truth), and the resulting winning bucket are all derived at runtime
# from the scraped tables in Results/ground_truth/data/ — see
# lookup_ground_truth() and derive_winning_bucket() below. Fed predictions /
# nowcasts are auto-discovered from polymarket_return/fed/ at runtime — see
# resolve_fed_data_paths(), discover_fed_cpi_file(), and friends. Everything
# that remains in this table is what those auto-resolvers cannot give us: the
# polymarket CSV path and the per-market bucket schema.
VARIABLE_CONFIGS = {
    "cpi_yoy": {
        "name": "CPI YoY",
        "buckets": ["≤2.8%", "2.9%", "3.0%", "3.1%", "≥3.2%"],
        "months": {
            "2025-11": {"polymarket_data": "polymarket_return/polymarket/cpi-nov-polymarket-price-data.csv"},
            "2025-12": {"polymarket_data": "polymarket_return/polymarket/cpi-dec-polymarket-price-data.csv"},
            "2026-01": {"polymarket_data": "polymarket_return/polymarket/cpi-jan-polymarket-price-data.csv"},
            "2026-02": {
                "polymarket_data": "polymarket_return/polymarket/cpi-feb-polymarket-price-data.csv",
                "buckets": ["≤2.1%", "2.2%", "2.3%", "2.4%", "2.5%", "2.6%", "≥2.7%"],
            },
            "2026-03": {
                "polymarket_data": "polymarket_return/polymarket/cpi-mar-polymarket-price-data.csv",
                "buckets": ["≤2.0%", "2.1%", "2.2%", "2.3%", "2.4%", "2.5%", "2.6%", "2.7%", "≥2.8%"],
            },
        },
    },
    "nonfarm_payrolls_change": {
        "name": "Jobs Added",
        "buckets": ["<0", "0-25k", "25k–50k", "50k–75k", "75k–100k", "100k–125k", ">125k"],
        "months": {
            "2025-11": {"polymarket_data": "polymarket_return/polymarket/jobs-nov-polymarket-price-data.csv"},
            "2025-12": {"polymarket_data": "polymarket_return/polymarket/jobs-dec-polymarket-price-data.csv"},
            "2026-01": {"polymarket_data": "polymarket_return/polymarket/jobs-jan-polymarket-price-data.csv"},
            "2026-02": {
                "polymarket_data": "polymarket_return/polymarket/jobs-feb-polymarket-price-data.csv",
                "buckets": ["<25k", "25k–50k", "50k–75k", "75k–100k", "100k–125k", "125k–150k", "150k–175k", "175k+"],
            },
            "2026-03": {
                "polymarket_data": "polymarket_return/polymarket/jobs-mar-polymarket-price-data.csv",
                "buckets": ["<-150k", "-150k – -100k", "-100k – -50k", "-50k – 0", "0 – 50k", "50k – 100k", "100k+"],
            },
        },
    },
    "unemployment_rate": {
        "name": "Unemployment Rate",
        # Buckets differ per month.
        "months": {
            "2025-11": {
                "polymarket_data": "polymarket_return/polymarket/unemp-nov-polymarket-price-data.csv",
                "buckets": ["≤4.1%", "4.2%", "4.3%", "4.4%", "4.5%", "≥4.6%"],
            },
            "2025-12": {
                "polymarket_data": "polymarket_return/polymarket/unemp-dec-polymarket-price-data.csv",
                "buckets": ["≤4.4%", "4.5%", "4.6%", "4.7%", "4.8%", "≥4.9%"],
            },
            "2026-01": {
                "polymarket_data": "polymarket_return/polymarket/unemp-jan-polymarket-price-data.csv",
                "buckets": ["≤4.2%", "4.3%", "4.4%", "4.5%", "4.6%", "≥4.7%"],
            },
            "2026-02": {
                "polymarket_data": "polymarket_return/polymarket/unemp-feb-polymarket-price-data.csv",
                "buckets": ["≤4.0%", "4.1%", "4.2%", "4.3%", "4.4%", "4.5%", "≥4.6%"],
            },
            "2026-03": {
                "polymarket_data": "polymarket_return/polymarket/unemp-mar-polymarket-price-data.csv",
                "buckets": ["≤3.9%", "4.0%", "4.1%", "4.2%", "4.3%", "4.4%", "4.5%", "4.6%", "≥4.7%"],
            },
        },
    },
    "real_gdp_qoq": {
        "name": "GDP QoQ (Annualized)",
        "buckets": ["<1.0%", "1.0–1.5%", "1.5–2.0%", "2.0–2.5%", "2.5–3.0%", "3.0–3.5%", ">3.5%"],
        "months": {
            "2025-Q4": {"polymarket_data": "polymarket_return/polymarket/gdp-q4-2025-polymarket-price-data.csv"},
            # Q1 2026 uses a slightly different top bucket label ("≥3.5%" vs Q4's ">3.5%").
            "2026-Q1": {
                "polymarket_data": "polymarket_return/polymarket/gdp-q1-2026-polymarket-price-data.csv",
                "buckets": ["<1.0%", "1.0–1.5%", "1.5–2.0%", "2.0–2.5%", "2.5–3.0%", "3.0–3.5%", "≥3.5%"],
            },
        },
    },
}

# Model prediction file paths by month
MODEL_PREDICTIONS = {
    "2025-11": {
        "name": "November 2025",
        "gpt_predictions": "data_from_serverA_serverB/final_analysis_data/model_gpt-5-search-api/2025-11_core_macroeconomic_conditions.csv",
        "claude_predictions": None,  # Claude not available for November
        "qwen_next_predictions": None,
        "qwen_235_predictions": None,
        "claude_code_agent_predictions": None,
    },
    "2025-12": {
        "name": "December 2025",
        "gpt_predictions": "data_from_serverA_serverB/final_analysis_data/model_gpt-5-search-api/2025-12_core_macroeconomic_conditions.csv",
        "claude_predictions": "data_from_serverA_serverB/final_analysis_data/model_claude-sonnet-4.5-api/2025-12_core_macroeconomic_conditions.csv",
        "qwen_next_predictions": "data_from_serverA_serverB/final_analysis_data/model_qwen3-next-80b-a3b-instruct/2025-12_core_macroeconomic_conditions.csv",
        "qwen_235_predictions": "data_from_serverA_serverB/final_analysis_data/model_qwen3-235b-a22b-instruct-2507/2025-12_core_macroeconomic_conditions.csv",
        "claude_code_agent_predictions": None,
    },
    "2026-01": {
        "name": "January 2026",
        "gpt_predictions": "data_from_serverA_serverB/final_analysis_data/model_gpt-5-search-api/2026-01_core_macroeconomic_conditions.csv",
        "claude_predictions": "data_from_serverA_serverB/final_analysis_data/model_claude-sonnet-4.5-api/2026-01_core_macroeconomic_conditions.csv",
        "qwen_next_predictions": "data_from_serverA_serverB/final_analysis_data/model_qwen3-next-80b-a3b-instruct/2026-01_core_macroeconomic_conditions.csv",
        "qwen_235_predictions": "data_from_serverA_serverB/final_analysis_data/model_qwen3-235b-a22b-instruct-2507/2026-01_core_macroeconomic_conditions.csv",
        "claude_code_agent_predictions": None,
    },
    "2026-02": {
        "name": "February 2026",
        "gpt_predictions": "data_from_serverA_serverB/final_analysis_data/model_gpt-5-search-api/2026-02_core_macroeconomic_conditions.csv",
        "claude_predictions": "data_from_serverA_serverB/final_analysis_data/model_claude-sonnet-4.5-api/2026-02_core_macroeconomic_conditions.csv",
        "qwen_next_predictions": "data_from_serverA_serverB/final_analysis_data/model_qwen3-next-80b-a3b-instruct/2026-02_core_macroeconomic_conditions.csv",
        "qwen_235_predictions": "data_from_serverA_serverB/final_analysis_data/model_qwen3-235b-a22b-instruct-2507/2026-02_core_macroeconomic_conditions.csv",
        "claude_code_agent_predictions": "data_from_serverA_serverB/final_analysis_data/model_claude-code-agent/2026-02_core_macroeconomic_conditions.csv",
    },
    "2026-03": {
        "name": "March 2026",
        "gpt_predictions": "data_from_serverA_serverB/final_analysis_data/model_gpt-5-search-api/2026-03_core_macroeconomic_conditions.csv",
        "claude_predictions": "data_from_serverA_serverB/final_analysis_data/model_claude-sonnet-4.5-api/2026-03_core_macroeconomic_conditions.csv",
        "qwen_next_predictions": "data_from_serverA_serverB/final_analysis_data/model_qwen3-next-80b-a3b-instruct/2026-03_core_macroeconomic_conditions.csv",
        "qwen_235_predictions": "data_from_serverA_serverB/final_analysis_data/model_qwen3-235b-a22b-instruct-2507/2026-03_core_macroeconomic_conditions.csv",
        "claude_code_agent_predictions": "data_from_serverA_serverB/final_analysis_data/model_claude-code-agent/2026-03_core_macroeconomic_conditions.csv",
    },
    # GDP Q4 2025 uses predictions from the 2025-12 file (target_month=2025-12 = last month of Q4)
    "2025-Q4": {
        "name": "Q4 2025 (GDP)",
        "gpt_predictions": "data_from_serverA_serverB/final_analysis_data/model_gpt-5-search-api/2025-12_core_macroeconomic_conditions.csv",
        "claude_predictions": "data_from_serverA_serverB/final_analysis_data/model_claude-sonnet-4.5-api/2025-12_core_macroeconomic_conditions.csv",
        "qwen_next_predictions": "data_from_serverA_serverB/final_analysis_data/model_qwen3-next-80b-a3b-instruct/2025-12_core_macroeconomic_conditions.csv",
        "qwen_235_predictions": "data_from_serverA_serverB/final_analysis_data/model_qwen3-235b-a22b-instruct-2507/2025-12_core_macroeconomic_conditions.csv",
        "claude_code_agent_predictions": None,  # No claude-code-agent file for Q4 2025
    },
    # GDP Q1 2026 combines the 2026-02 and 2026-03 prediction files because
    # Q1 spans Jan–Mar and predictions of Q1 GDP are scattered across both
    # target_month files (e.g. claude-sonnet's 2026-03 file only starts in
    # April, but its 2026-02 file already has Q1 predictions from late Feb).
    "2026-Q1": {
        "name": "Q1 2026 (GDP)",
        "gpt_predictions": [
            "data_from_serverA_serverB/final_analysis_data/model_gpt-5-search-api/2026-02_core_macroeconomic_conditions.csv",
            "data_from_serverA_serverB/final_analysis_data/model_gpt-5-search-api/2026-03_core_macroeconomic_conditions.csv",
        ],
        "claude_predictions": [
            "data_from_serverA_serverB/final_analysis_data/model_claude-sonnet-4.5-api/2026-02_core_macroeconomic_conditions.csv",
            "data_from_serverA_serverB/final_analysis_data/model_claude-sonnet-4.5-api/2026-03_core_macroeconomic_conditions.csv",
        ],
        "qwen_next_predictions": [
            "data_from_serverA_serverB/final_analysis_data/model_qwen3-next-80b-a3b-instruct/2026-02_core_macroeconomic_conditions.csv",
            "data_from_serverA_serverB/final_analysis_data/model_qwen3-next-80b-a3b-instruct/2026-03_core_macroeconomic_conditions.csv",
        ],
        "qwen_235_predictions": [
            "data_from_serverA_serverB/final_analysis_data/model_qwen3-235b-a22b-instruct-2507/2026-02_core_macroeconomic_conditions.csv",
            "data_from_serverA_serverB/final_analysis_data/model_qwen3-235b-a22b-instruct-2507/2026-03_core_macroeconomic_conditions.csv",
        ],
        "claude_code_agent_predictions": [
            "data_from_serverA_serverB/final_analysis_data/model_claude-code-agent/2026-02_core_macroeconomic_conditions.csv",
            "data_from_serverA_serverB/final_analysis_data/model_claude-code-agent/2026-03_core_macroeconomic_conditions.csv",
        ],
    },
}

PREDICTION_RANGE_LOG = Path("data_from_serverA_serverB/final_analysis_data/variable_prediction_ranges.csv")

# Legacy MONTH_CONFIGS for backward compatibility
MONTH_CONFIGS = {
    "2025-11": {"name": "November 2025"},
    "2025-12": {"name": "December 2025"},
    "2026-01": {"name": "January 2026"},
    "2026-02": {"name": "February 2026"},
    "2026-03": {"name": "March 2026"},
    "2025-Q4": {"name": "Q4 2025"},
    "2026-Q1": {"name": "Q1 2026"},
}


# ---------------------------------------------------------------------------
# Scraped ground-truth lookup. Replaces the hand-maintained release_datetime_utc
# / winning_bucket / ground_truth_value entries that used to live in
# VARIABLE_CONFIGS. Source: Results/ground_truth/data/field_releases_live.csv
# (produced by scrape_incremental.py + parse_ground_truth.py).
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
GROUND_TRUTH_LIVE_CSV = _SCRIPT_DIR.parent / "ground_truth" / "data" / "field_releases_live.csv"

# Benchmark variable -> (field_id in field_releases_live.csv, value scaler).
# scaler converts the calendar's stored unit into the unit the benchmark
# buckets use (e.g. NFP calendar stores raw headcount; benchmark buckets are
# in thousands -> scale by 1e-3).
# NOTE: field_id "cpi" in field_releases_live.csv stores CPI YoY (the user
# updated it from CPI MoM to YoY so cpi_yoy can be looked up directly here).
_FIELD_RELEASES_LIVE_FIELD = {
    "cpi_yoy":                 ("cpi",               1.0),
    "unemployment_rate":       ("unemployment_rate", 1.0),
    "nonfarm_payrolls_change": ("nonfarm_payrolls",  1e-3),
    "real_gdp_qoq":            ("real_gdp_advance",  1.0),
}

_ET_TZ = "America/New_York"


def _load_ground_truth_table() -> pd.DataFrame:
    """Cached load of the parsed live ground-truth table."""
    cache = getattr(_load_ground_truth_table, "_cache", None)
    if cache is None:
        cache = pd.read_csv(GROUND_TRUTH_LIVE_CSV, parse_dates=["release_datetime_et"])
        _load_ground_truth_table._cache = cache  # type: ignore[attr-defined]
    return cache


def lookup_ground_truth(variable: str, target_month: str) -> tuple[datetime | None, float | None]:
    """Return (release_datetime_utc, value) for (variable, target_month).

    Reads from Results/ground_truth/data/field_releases_live.csv. target_month
    is 'YYYY-MM' for monthly variables and 'YYYY-Qn' for real_gdp_qoq. Returns
    (None, None) if the release has not been recorded yet.

    The ET -> UTC conversion uses pandas tz-aware localization so DST cutovers
    (e.g. early Mar 2026) are handled correctly instead of a fixed +5h offset.
    """
    if variable not in _FIELD_RELEASES_LIVE_FIELD:
        return None, None
    field_id, scale = _FIELD_RELEASES_LIVE_FIELD[variable]
    gt = _load_ground_truth_table()
    cand = gt[(gt["field_id"] == field_id) & (gt["ref_period"] == target_month)]
    if cand.empty:
        return None, None
    # Some fields have a near-duplicate row at HH:29 (preliminary); take the
    # later row — the canonical HH:30 release.
    row = cand.sort_values("release_datetime_et", ascending=False).iloc[0]
    et_naive = pd.Timestamp(row["release_datetime_et"])
    utc = (
        et_naive.tz_localize(_ET_TZ).tz_convert("UTC").tz_localize(None).to_pydatetime()
    )
    value = None if pd.isna(row["A"]) else float(row["A"]) * scale
    return utc, value


def derive_winning_bucket(
    variable: str,
    value: float,
    buckets: list,
    polymarket_df: pd.DataFrame,
) -> str:
    """Map a ground-truth value to a bucket and cross-check against polymarket.

    Polymarket's winning bucket prints ~0.9995 once the market resolves. If the
    polymarket data shows that resolution (max bucket prob >= 0.99) we require
    the highest-probability bucket to equal the value-derived bucket; otherwise
    we raise. If the market never reaches 0.99 (e.g. low-volume market that
    didn't fully converge) we accept the value-derived bucket and warn.
    """
    bucket = map_prediction_to_bucket(value, variable, buckets)
    if bucket not in buckets:
        raise ValueError(
            f"Derived bucket {bucket!r} not in configured buckets {buckets} "
            f"(variable={variable}, value={value})"
        )
    cols_in_df = [b for b in buckets if b in polymarket_df.columns]
    if not cols_in_df:
        return bucket
    max_per_bucket = polymarket_df[cols_in_df].apply(pd.to_numeric, errors="coerce").max()
    market_max = float(max_per_bucket.max())
    market_winner = max_per_bucket.idxmax()
    if market_max < 0.99:
        print(
            f"  [warn] polymarket never resolved (max bucket prob {market_max:.4f}); "
            f"using value-derived winning_bucket={bucket!r} without market verification."
        )
    elif market_winner != bucket:
        raise ValueError(
            f"Inconsistency for {variable}: ground-truth value {value} maps to bucket "
            f"{bucket!r}, but polymarket's resolved bucket is {market_winner!r} "
            f"(max prob {max_per_bucket[market_winner]:.4f}). Check the bucket scheme "
            f"or the ground-truth value."
        )
    return bucket


def parse_month_year_month(month: str) -> tuple[int, int] | None:
    """Parse month token in YYYY-MM format."""
    match = re.fullmatch(r"(\d{4})-(\d{2})", month)
    if not match:
        return None
    year = int(match.group(1))
    month_num = int(match.group(2))
    if not 1 <= month_num <= 12:
        return None
    return year, month_num


def discover_fed_cpi_file(month: str, base_dir: Path) -> Path | None:
    """Auto-discover Cleveland Fed CPI file for a month if present."""
    parsed = parse_month_year_month(month)
    if not parsed:
        return None
    year, month_num = parsed
    fed_dir = base_dir / "polymarket_return/fed"
    if not fed_dir.exists():
        return None

    prioritized_names = [
        f"cpi-Year-Over-YearPercentChange-{year}-{month_num:02d}.csv",
        f"cpi-Year-Over-YearPercentChange-{year}-{month_num}.csv",
    ]
    for name in prioritized_names:
        candidate = fed_dir / name
        if candidate.exists():
            return candidate

    fallback_matches = sorted(fed_dir.glob(f"cpi-Year-Over-YearPercentChange-{year}-*.csv"))
    for path in fallback_matches:
        suffix = path.stem.rsplit("-", 1)[-1]
        if suffix.isdigit() and int(suffix) == month_num:
            return path

    return None


def discover_fed_unemp_nowcast_file(month: str, base_dir: Path) -> Path | None:
    """Auto-discover Chicago Fed unemployment nowcast file for a month if present."""
    parsed = parse_month_year_month(month)
    if not parsed:
        return None
    _, month_num = parsed
    fed_dir = base_dir / "polymarket_return/fed"
    if not fed_dir.exists():
        return None

    month_to_abbr = {
        1: "jan",
        2: "feb",
        3: "mar",
        4: "apr",
        5: "may",
        6: "jun",
        7: "jul",
        8: "aug",
        9: "sep",
        10: "oct",
        11: "nov",
        12: "dec",
    }
    abbr = month_to_abbr[month_num]
    candidate = fed_dir / f"unemp-{abbr}.csv"
    if candidate.exists():
        return candidate
    return None


def discover_fed_gdp_files(base_dir: Path, target_month: str | None = None) -> list[tuple[str, Path]]:
    """
    Discover GDP nowcast/forecast files and map each to a model name.

    When `target_month` is a quarter token (YYYY-Qn), per-quarter files such
    as `gdp-atlantafed-2026q1.csv` are preferred over the generic files (which
    may contain stale/other-quarter data).

    Returns:
        List of (model_name, path), e.g.
        [("fed-atlanta", ...), ("fed-newyork", ...), ("fed-stlouis", ...)]
    """
    fed_dir = base_dir / "polymarket_return/fed"
    if not fed_dir.exists():
        return []

    discovered: dict[str, Path] = {}

    # Build the per-quarter suffix (e.g. "2026q1") if target_month is quarterly.
    quarter_suffix: str | None = None
    if target_month:
        m = re.fullmatch(r"(\d{4})-Q([1-4])", target_month)
        if m:
            quarter_suffix = f"{m.group(1)}q{m.group(2)}".lower()

    # Per-quarter overrides: prefer when target is quarterly, since the
    # generic gdp-<src>fed.csv files often pin to a single quarter (e.g.
    # Q4 2025) and would parse to zero rows for other quarters.
    if quarter_suffix:
        per_quarter = {
            "fed-atlanta": fed_dir / f"gdp-atlantafed-{quarter_suffix}.csv",
            "fed-newyork": fed_dir / f"gdp-newyorkfed-{quarter_suffix}.csv",
            "fed-stlouis": fed_dir / f"gdp-stlouisfed-{quarter_suffix}.csv",
        }
        for model_name, path in per_quarter.items():
            if path.exists():
                discovered[model_name] = path

    # Generic fallbacks for sources that don't have a per-quarter override.
    preferred = {
        "fed-atlanta": fed_dir / "gdp-atlantafed.csv",
        "fed-newyork": fed_dir / "gdp-newyorkfed.csv",
        "fed-stlouis": fed_dir / "gdp-stlouisfed.csv",
    }
    for model_name, path in preferred.items():
        if model_name not in discovered and path.exists():
            discovered[model_name] = path

    for path in sorted(fed_dir.glob("gdp-*.csv")):
        name = path.name.lower()
        # Skip per-quarter files for quarters other than the target so they
        # don't accidentally get assigned via the loose substring match below.
        if re.search(r"\d{4}q[1-4]", name) and (quarter_suffix is None or quarter_suffix not in name):
            continue
        if "atlanta" in name and "fed-atlanta" not in discovered:
            discovered["fed-atlanta"] = path
        elif ("newyork" in name or "new-york" in name or "nyfed" in name) and "fed-newyork" not in discovered:
            discovered["fed-newyork"] = path
        elif ("stlouis" in name or "st-louis" in name or "st_louis" in name) and "fed-stlouis" not in discovered:
            discovered["fed-stlouis"] = path

    ordered_model_names = ["fed-atlanta", "fed-newyork", "fed-stlouis"]
    return [(model_name, discovered[model_name]) for model_name in ordered_model_names if model_name in discovered]


def resolve_fed_data_paths(
    variable: str,
    month: str,
    month_var_config: dict,
    base_dir: Path,
) -> tuple[Path | None, Path | None]:
    """
    Resolve Fed file paths using config first, then auto-discovery in polymarket_return/fed.

    Returns:
        (fed_predictions_path, fed_nowcast_path)
    """
    fed_predictions_path: Path | None = None
    fed_nowcast_path: Path | None = None

    configured_fed_predictions = month_var_config.get("fed_predictions")
    if configured_fed_predictions:
        candidate = base_dir / configured_fed_predictions
        if candidate.exists():
            fed_predictions_path = candidate
        else:
            print(f"  Configured fed_predictions not found: {candidate}")

    configured_fed_nowcast = month_var_config.get("fed_nowcast")
    if configured_fed_nowcast:
        candidate = base_dir / configured_fed_nowcast
        if candidate.exists():
            fed_nowcast_path = candidate
        else:
            print(f"  Configured fed_nowcast not found: {candidate}")

    if fed_predictions_path is None and variable == "cpi_yoy":
        discovered = discover_fed_cpi_file(month, base_dir)
        if discovered is not None:
            fed_predictions_path = discovered
            print(f"  Auto-discovered fed_predictions: {fed_predictions_path}")

    if fed_nowcast_path is None and variable == "unemployment_rate":
        discovered = discover_fed_unemp_nowcast_file(month, base_dir)
        if discovered is not None:
            fed_nowcast_path = discovered
            print(f"  Auto-discovered fed_nowcast: {fed_nowcast_path}")

    return fed_predictions_path, fed_nowcast_path


def to_utc_from_et(local_series: pd.Series) -> pd.Series:
    """Convert ET naive timestamps to UTC naive timestamps, DST-aware."""
    et = local_series.dt.tz_localize("America/New_York", ambiguous="infer", nonexistent="shift_forward")
    return et.dt.tz_convert("UTC").dt.tz_localize(None)


def quarter_end_for_token(month: str) -> datetime | None:
    """Convert YYYY-QX token to quarter-end datetime at midnight."""
    match = re.fullmatch(r"(\d{4})-Q([1-4])", month)
    if not match:
        return None
    year = int(match.group(1))
    quarter = int(match.group(2))
    end_month_day = {
        1: (3, 31),
        2: (6, 30),
        3: (9, 30),
        4: (12, 31),
    }
    month_num, day = end_month_day[quarter]
    return datetime(year, month_num, day)


def quarter_token_for_month(month: str) -> str | None:
    """Convert YYYY-QX token to YYYYQX."""
    match = re.fullmatch(r"(\d{4})-Q([1-4])", month)
    if not match:
        return None
    return f"{match.group(1)}Q{match.group(2)}"


def clean_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names by trimming whitespace/BOM."""
    normalized = []
    for col in df.columns:
        normalized.append(str(col).replace("\ufeff", "").strip())
    df = df.copy()
    df.columns = normalized
    return df


def load_fed_gdp_predictions(filepath: Path, target_month: str, model_name: str) -> pd.DataFrame:
    """
    Load GDP nowcast/forecast CSVs from different Fed sources into common schema.

    Expected output columns:
      datetime_local (ET), datetime_utc, value
    """
    df = pd.read_csv(filepath)
    df = clean_dataframe_columns(df)
    lower_map = {c.lower(): c for c in df.columns}
    filename = filepath.name.lower()

    date_col: str | None = None
    value_col: str | None = None

    def first_numeric_like_column(exclude: set[str]) -> str | None:
        for c in df.columns:
            if c in exclude:
                continue
            candidate = pd.to_numeric(df[c], errors="coerce")
            if candidate.notna().any():
                return c
        return None

    parse_date_format: str | None = None
    parse_quarter_format: str | None = None
    default_release_hour_et = 10
    default_release_minute_et = 0

    if "atlanta" in filename:
        date_col = lower_map.get("forecast date")
        quarter_col = lower_map.get("quarter being forecasted")
        value_col = lower_map.get("gdp nowcast")
        parse_date_format = "%m/%d/%y"
        parse_quarter_format = "%m/%d/%y"
        # Assumption: Atlanta Fed GDPNow release treated as 5:00 PM ET.
        default_release_hour_et = 17
        default_release_minute_et = 0
        if value_col is None:
            value_col = first_numeric_like_column({date_col or "", quarter_col or ""})
        if date_col is None or value_col is None:
            raise ValueError(f"Unable to parse Atlanta GDP file columns: {df.columns.tolist()}")

        if quarter_col is not None:
            target_q_end = quarter_end_for_token(target_month)
            if target_q_end is not None:
                quarter_dates = pd.to_datetime(df[quarter_col], format=parse_quarter_format, errors="coerce")
                if quarter_dates.isna().all():
                    quarter_dates = pd.to_datetime(df[quarter_col], errors="coerce")
                df = df[quarter_dates.dt.date == target_q_end.date()].copy()

    elif "newyork" in filename or "new-york" in filename or "nyfed" in filename:
        date_col = lower_map.get("forecast date")
        target_q_col = quarter_token_for_month(target_month)
        parse_date_format = "%d-%b-%y"
        # NY Fed updates at/shortly after 12:45 PM ET.
        default_release_hour_et = 12
        default_release_minute_et = 45
        if target_q_col and target_q_col in df.columns:
            value_col = target_q_col
        else:
            print(
                f"  Skipping {model_name}: target quarter column {target_q_col!r} "
                f"not found in {filepath.name}"
            )
            return pd.DataFrame(columns=["datetime_local", "datetime_utc", "value"])
        if date_col is None or value_col is None:
            raise ValueError(f"Unable to parse New York Fed GDP file columns: {df.columns.tolist()}")

    elif "stlouis" in filename or "st-louis" in filename or "st_louis" in filename:
        date_col = lower_map.get("observation_date")
        if date_col is None:
            date_col = lower_map.get("date")
        parse_date_format = "%m/%d/%y"
        # St. Louis release at 10:00 AM US Central = 11:00 AM ET.
        default_release_hour_et = 11
        default_release_minute_et = 0
        value_col = first_numeric_like_column({date_col or ""})
        if date_col is None or value_col is None:
            raise ValueError(f"Unable to parse St. Louis GDP file columns: {df.columns.tolist()}")

    else:
        for c in df.columns:
            if "date" in c.lower():
                date_col = c
                break
        value_col = first_numeric_like_column({date_col or ""})
        if date_col is None or value_col is None:
            raise ValueError(f"Unable to parse GDP fed file columns: {df.columns.tolist()}")

    result = pd.DataFrame()
    result["datetime_local"] = pd.to_datetime(df[date_col], format=parse_date_format, errors="coerce")
    if result["datetime_local"].isna().all():
        result["datetime_local"] = pd.to_datetime(df[date_col], errors="coerce")
    result["value"] = pd.to_numeric(df[value_col], errors="coerce")
    result = result.dropna(subset=["datetime_local", "value"]).copy()

    # When source dates are date-only, use source-specific ET publication time.
    if not result.empty and (result["datetime_local"].dt.hour.eq(0) & result["datetime_local"].dt.minute.eq(0)).all():
        result["datetime_local"] = (
            result["datetime_local"].dt.normalize()
            + pd.Timedelta(hours=default_release_hour_et, minutes=default_release_minute_et)
        )

    result = result.sort_values("datetime_local").drop_duplicates(subset=["datetime_local"], keep="last")
    result["datetime_utc"] = to_utc_from_et(result["datetime_local"])
    result = result.reset_index(drop=True)

    print(f"  Loaded {len(result)} GDP Fed predictions from {model_name}: {filepath}")
    if not result.empty:
        print(f"  Date range: {result['datetime_local'].min()} to {result['datetime_local'].max()} ET")
        print(f"  Value range: {result['value'].min():.3f} to {result['value'].max():.3f}")

    return result[["datetime_local", "datetime_utc", "value"]]


def map_prediction_to_bucket(value: float, variable: str, buckets: list) -> str:
    """
    Map a prediction value to a Polymarket bucket based on variable type.

    Args:
        value: The predicted value
        variable: Variable name (e.g., "cpi_yoy", "nonfarm_payrolls_change", "unemployment_rate")
        buckets: List of bucket names for this variable/month

    Returns:
        The bucket name that the prediction maps to
    """
    if variable == "cpi_yoy":
        if "≤2.0%" in buckets:  # Mar 2026: 0.1% steps from 2.0% to 2.8%
            if value <= 2.0:
                return "≤2.0%"
            elif value < 2.15:
                return "2.1%"
            elif value < 2.25:
                return "2.2%"
            elif value < 2.35:
                return "2.3%"
            elif value < 2.45:
                return "2.4%"
            elif value < 2.55:
                return "2.5%"
            elif value < 2.65:
                return "2.6%"
            elif value < 2.75:
                return "2.7%"
            else:
                return "≥2.8%"
        elif "≤2.1%" in buckets:  # Feb 2026 buckets
            if value <= 2.1:
                return "≤2.1%"
            elif value < 2.25:
                return "2.2%"
            elif value < 2.35:
                return "2.3%"
            elif value < 2.45:
                return "2.4%"
            elif value < 2.55:
                return "2.5%"
            elif value < 2.65:
                return "2.6%"
            else:
                return "≥2.7%"
        else:  # Nov/Dec/Jan buckets: ["≤2.8%", "2.9%", "3.0%", "3.1%", "≥3.2%"]
            if value <= 2.8:
                return "≤2.8%"
            elif value < 2.95:
                return "2.9%"
            elif value < 3.05:
                return "3.0%"
            elif value < 3.15:
                return "3.1%"
            else:
                return "≥3.2%"

    elif variable == "nonfarm_payrolls_change":
        # Jobs added: value is in thousands (e.g., 50 means 50k)
        # Buckets vary by month
        if "<-150k" in buckets:  # Mar 2026 buckets
            if value < -150:
                return "<-150k"
            elif value < -100:
                return "-150k – -100k"
            elif value < -50:
                return "-100k – -50k"
            elif value < 0:
                return "-50k – 0"
            elif value < 50:
                return "0 – 50k"
            elif value < 100:
                return "50k – 100k"
            else:
                return "100k+"
        elif "<25k" in buckets:  # Feb 2026 buckets
            if value < 25:
                return "<25k"
            elif value < 50:
                return "25k–50k"
            elif value < 75:
                return "50k–75k"
            elif value < 100:
                return "75k–100k"
            elif value < 125:
                return "100k–125k"
            elif value < 150:
                return "125k–150k"
            elif value < 175:
                return "150k–175k"
            else:
                return "175k+"
        else:  # Nov/Dec/Jan buckets: ["<0", "0-25k", "25k–50k", ...]
            if value < 0:
                return "<0"
            elif value < 25:
                return "0-25k"
            elif value < 50:
                return "25k–50k"
            elif value < 75:
                return "50k–75k"
            elif value < 100:
                return "75k–100k"
            elif value < 125:
                return "100k–125k"
            else:
                return ">125k"

    elif variable == "unemployment_rate":
        # Unemployment rate: buckets vary by month
        # Dynamically determine based on bucket structure
        if "≤3.9%" in buckets:  # March 2026 buckets (9 buckets)
            if value <= 3.9:
                return "≤3.9%"
            elif value < 4.05:
                return "4.0%"
            elif value < 4.15:
                return "4.1%"
            elif value < 4.25:
                return "4.2%"
            elif value < 4.35:
                return "4.3%"
            elif value < 4.45:
                return "4.4%"
            elif value < 4.55:
                return "4.5%"
            elif value < 4.65:
                return "4.6%"
            else:
                return "≥4.7%"
        elif "≤4.0%" in buckets:  # February 2026 buckets
            if value <= 4.0:
                return "≤4.0%"
            elif value < 4.15:
                return "4.1%"
            elif value < 4.25:
                return "4.2%"
            elif value < 4.35:
                return "4.3%"
            elif value < 4.45:
                return "4.4%"
            elif value < 4.55:
                return "4.5%"
            else:
                return "≥4.6%"
        elif "≤4.1%" in buckets:  # November buckets
            if value <= 4.1:
                return "≤4.1%"
            elif value < 4.25:
                return "4.2%"
            elif value < 4.35:
                return "4.3%"
            elif value < 4.45:
                return "4.4%"
            elif value < 4.55:
                return "4.5%"
            else:
                return "≥4.6%"
        elif "≤4.2%" in buckets:  # January buckets
            if value <= 4.2:
                return "≤4.2%"
            elif value < 4.35:
                return "4.3%"
            elif value < 4.45:
                return "4.4%"
            elif value < 4.55:
                return "4.5%"
            elif value < 4.65:
                return "4.6%"
            else:
                return "≥4.7%"
        else:  # December buckets (≤4.4%, 4.5%, etc.)
            if value <= 4.4:
                return "≤4.4%"
            elif value < 4.55:
                return "4.5%"
            elif value < 4.65:
                return "4.6%"
            elif value < 4.75:
                return "4.7%"
            elif value < 4.85:
                return "4.8%"
            else:
                return "≥4.9%"

    elif variable == "real_gdp_qoq":
        # GDP QoQ annualized. Q4 2025 uses ">3.5%" for the top bucket; Q1 2026
        # uses "≥3.5%". The numeric thresholds are identical so we just pick
        # whichever label is present in the schema.
        top_bucket = "≥3.5%" if "≥3.5%" in buckets else ">3.5%"
        if value < 1.0:
            return "<1.0%"
        elif value < 1.5:
            return "1.0–1.5%"
        elif value < 2.0:
            return "1.5–2.0%"
        elif value < 2.5:
            return "2.0–2.5%"
        elif value < 3.0:
            return "2.5–3.0%"
        elif value < 3.5:
            return "3.0–3.5%"
        else:
            return top_bucket

    else:
        raise ValueError(f"Unknown variable: {variable}")


def load_polymarket_data(filepath: Path) -> pd.DataFrame:
    """Load Polymarket price data."""
    df = pd.read_csv(filepath)
    # Parse date - format is "MM-DD-YYYY HH:MM"
    df["datetime_utc"] = pd.to_datetime(df["Date (UTC)"], format="%m-%d-%Y %H:%M")
    df = df.sort_values("datetime_utc").reset_index(drop=True)
    return df


def load_model_predictions(filepath, variable: str = "cpi_yoy") -> pd.DataFrame:
    """Load model predictions for a specific variable.

    `filepath` may be a single Path/str OR a list of Paths/strs. When a list
    is passed (used for quarterly markets that aggregate multiple monthly
    target_month files, e.g. Q1 2026 GDP combining 2026-02 + 2026-03), the
    files are concatenated and deduplicated by `timestamp_local` (keeping the
    last value for ties).
    """
    if isinstance(filepath, (list, tuple)):
        if not filepath:
            return pd.DataFrame()
        dfs = []
        for fp in filepath:
            sub = pd.read_csv(fp)
            sub = sub[sub["variable"] == variable].copy()
            dfs.append(sub)
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(filepath)
        df = df[df["variable"] == variable].copy()
    # Parse timestamp
    df["datetime_local"] = pd.to_datetime(df["timestamp_local"], format="mixed")
    df = (
        df.sort_values("datetime_local")
        .drop_duplicates(subset=["datetime_local"], keep="last")
        .reset_index(drop=True)
    )
    return df


def load_fed_predictions(filepath: Path, column: str = "CPI Inflation") -> pd.DataFrame:
    """
    Load Fed predictions from CSV.

    Fed CPI nowcast is released daily at 10:00 AM Eastern Time.
    This function adds the release time and converts to UTC.

    Returns:
        DataFrame with columns: datetime_local (ET), datetime_utc, value
    """
    df = pd.read_csv(filepath)

    # Parse date - format is "MM/DD" (need to add year and time)
    def parse_fed_date(label: str) -> datetime:
        month, day = label.split("/")
        month, day = int(month), int(day)
        # Determine year based on month
        if month >= 11:
            year = 2025
        else:
            year = 2026
        # Fed CPI nowcast is released at 10:00 AM ET daily
        return datetime(year, month, day, 10, 0, 0)

    df["datetime_local"] = df["Label"].apply(parse_fed_date)

    # Convert ET to UTC (ET is UTC-5 in winter/EST, UTC-4 in summer/EDT)
    # For Nov-Jan, use EST (UTC-5)
    df["datetime_utc"] = df["datetime_local"] + timedelta(hours=5)

    df["value"] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df.sort_values("datetime_local").reset_index(drop=True)
    return df[["datetime_local", "datetime_utc", "value"]]


def load_fed_nowcast(filepath: Path, value_column: str = "unemployment_rate_nowcast") -> pd.DataFrame:
    """
    Load Fed nowcast data from CSV.

    Expected format:
    target_month,release_type,release_datetime_et,unemployment_rate_nowcast
    2025-11,advance,2025-11-24 08:30,4.44
    2025-11,final,2025-12-04 08:30,4.44

    Args:
        filepath: Path to the Fed nowcast CSV
        value_column: Column name containing the nowcast value

    Returns:
        DataFrame with columns: release_datetime_et, release_datetime_utc, value, release_type
    """
    df = pd.read_csv(filepath)

    # Parse release datetime (format: "YYYY-MM-DD HH:MM")
    df["release_datetime_et"] = pd.to_datetime(df["release_datetime_et"])

    # Convert ET to UTC (ET is UTC-5 in winter, UTC-4 in summer)
    # For simplicity, assume EST (UTC-5) for Nov-Jan releases
    df["release_datetime_utc"] = df["release_datetime_et"] + timedelta(hours=5)

    df["value"] = pd.to_numeric(df[value_column], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df.sort_values("release_datetime_et").reset_index(drop=True)

    return df[["release_datetime_et", "release_datetime_utc", "value", "release_type"]]


def get_effective_cutoff_time(
    polymarket_df: pd.DataFrame,
    winning_bucket: str,
    official_release_utc: datetime | None = None,
) -> tuple[datetime, datetime]:
    """Return (cutoff_time, resolution_time).

    cutoff_time is the latest betting timestamp we allow. We use the first
    moment the winning bucket exceeds 0.99 (typically ~30 min after the
    official release; acceptable since Polymarket data is hourly). If the
    market never reaches 0.99 we fall back to Polymarket's max timestamp,
    clamped by the official release so an unresolved market still gets a
    release-time cutoff and post-release predictions don't leak into the
    betting loop.
    """
    resolution_mask = polymarket_df[winning_bucket] > 0.99
    if resolution_mask.any():
        cutoff_time = polymarket_df.loc[resolution_mask, "datetime_utc"].iloc[0]
        resolution_time = cutoff_time
    else:
        polymax = polymarket_df["datetime_utc"].max()
        if official_release_utc is not None:
            cutoff_time = min(polymax, pd.Timestamp(official_release_utc))
        else:
            cutoff_time = polymax
        resolution_time = polymax
    return cutoff_time, resolution_time


def load_prediction_range_log(range_log_path: Path) -> pd.DataFrame:
    """Load prediction availability ranges produced by concat_model_data.py."""
    ranges_df = pd.read_csv(range_log_path)
    required_columns = {
        "model",
        "target_month",
        "variable",
        "prediction_start",
        "prediction_end",
    }
    missing_columns = required_columns.difference(ranges_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Prediction range log missing required columns: {missing}")
    return ranges_df


def infer_target_month_from_prediction_path(predictions_path) -> str:
    """Infer target month token from a final-analysis CSV filename.

    Accepts a single Path or a list of Paths. When a list is provided (e.g.,
    for quarterly markets that combine multiple monthly prediction files),
    returns the EARLIEST target_month so the prediction-window eligibility
    check uses the earliest available prediction range.
    """
    if isinstance(predictions_path, (list, tuple)):
        tokens = [Path(p).name.split("_", 1)[0] for p in predictions_path]
        return min(tokens)
    return predictions_path.name.split("_", 1)[0]


def resolve_prediction_spec(spec, base_dir: Path) -> list[Path]:
    """Normalize a MODEL_PREDICTIONS value (None / str / list[str]) into a list
    of absolute Paths. Returns an empty list when the spec is None.

    Single-string specs become a 1-element list so downstream callers can
    uniformly treat the result as a list.
    """
    if spec is None:
        return []
    if isinstance(spec, str):
        return [base_dir / spec]
    return [base_dir / s for s in spec]


def model_has_usable_prediction_window(
    prediction_ranges_df: pd.DataFrame,
    model_name: str,
    target_month: str,
    variable: str,
    official_release_utc: datetime | None,
    local_tz_offset_hours: int,
) -> tuple[bool, str]:
    """Return whether the model has any prediction before the release cutoff.

    Used to gate sparse-data models (Qwen, claude-code-agent) whose series
    may start after the release in some (variable, target_month) combos.
    Continuous-coverage models (gpt, claude-sonnet) bypass this gate.
    """
    if official_release_utc is None:
        return True, "no official release cutoff configured"

    matches = prediction_ranges_df[
        (prediction_ranges_df["model"] == model_name)
        & (prediction_ranges_df["target_month"] == target_month)
        & (prediction_ranges_df["variable"] == variable)
    ]
    if matches.empty:
        return False, (
            f"no prediction-range entry for target_month={target_month}, "
            f"variable={variable}"
        )

    prediction_start_local = pd.to_datetime(matches["prediction_start"]).min().to_pydatetime()
    prediction_start_utc = prediction_start_local - timedelta(hours=local_tz_offset_hours)

    if prediction_start_utc >= official_release_utc:
        return False, (
            f"prediction_start={prediction_start_local} local "
            f"({prediction_start_utc} UTC) is at/after release cutoff "
            f"{official_release_utc}"
        )

    return True, (
        f"prediction_start={prediction_start_local} local "
        f"({prediction_start_utc} UTC) is before release cutoff "
        f"{official_release_utc}"
    )


def remove_existing_result_artifacts(
    output_dir: Path,
    summary_path: Path,
    model_name: str,
    month: str,
    frequency: str,
    aggregation_method: str,
) -> None:
    """Remove stale result files and summary rows for a skipped model/run."""
    subfolder_dir = output_dir / get_output_subfolder(frequency, aggregation_method)
    detail_path = subfolder_dir / f"betting_results_{model_name}_{month}_{frequency}_{aggregation_method}.csv"
    plot_path = subfolder_dir / f"returns_plot_{model_name}_{month}_{frequency}_{aggregation_method}.png"

    if detail_path.exists():
        detail_path.unlink()
        print(f"Removed stale detailed results for skipped model: {detail_path}")

    if plot_path.exists():
        plot_path.unlink()
        print(f"Removed stale returns plot for skipped model: {plot_path}")

    if not summary_path.exists():
        return

    existing_df = pd.read_csv(summary_path)
    required_columns = {"model_name", "month", "betting_frequency"}
    if not required_columns.issubset(existing_df.columns):
        return

    mask = ~(
        (existing_df["model_name"] == model_name)
        & (existing_df["month"] == month)
        & (existing_df["betting_frequency"] == frequency)
        & (
            existing_df["aggregation_method"] == aggregation_method
            if "aggregation_method" in existing_df.columns
            else True
        )
    )
    filtered_df = existing_df[mask].copy()
    if len(filtered_df) == len(existing_df):
        return

    filtered_df.to_csv(summary_path, index=False)
    print(
        "Removed stale summary rows for skipped model: "
        f"{model_name} [{month}] ({frequency}, {aggregation_method})"
    )


def get_latest_prediction_before(
    predictions: pd.DataFrame,
    cutoff_time: datetime,
    lookback_hours: int | None = None,
    time_col: str = "datetime_local",
) -> tuple[float | None, datetime | None]:
    """Get the latest prediction before a given time.

    Returns:
        Tuple of (prediction_value, prediction_timestamp) or (None, None) if not found.
    """
    if lookback_hours is None:
        mask = predictions[time_col] <= cutoff_time
    else:
        lookback_start = cutoff_time - timedelta(hours=lookback_hours)
        mask = (predictions[time_col] <= cutoff_time) & \
               (predictions[time_col] >= lookback_start)

    valid_predictions = predictions[mask]

    if valid_predictions.empty:
        return None, None

    # Get the latest one
    latest = valid_predictions.iloc[-1]
    return float(latest["value"]), latest[time_col]


def get_daily_avg_prediction(
    predictions: pd.DataFrame,
    target_date: datetime,
    time_col: str = "datetime_local",
) -> tuple[float | None, int]:
    """Get the average prediction for a specific day.

    Args:
        predictions: DataFrame with predictions
        target_date: The date to get average for (uses date part only)
        time_col: Column name for timestamp

    Returns:
        Tuple of (average_value, count_of_predictions) or (None, 0) if not found.
    """
    # Get all predictions from the target date
    target_day = target_date.date() if hasattr(target_date, 'date') else target_date
    mask = predictions[time_col].dt.date == target_day

    valid_predictions = predictions[mask]

    if valid_predictions.empty:
        return None, 0

    avg_value = valid_predictions["value"].mean()
    return float(avg_value), len(valid_predictions)


def get_3day_avg_prediction(
    predictions: pd.DataFrame,
    target_date: datetime,
    time_col: str = "datetime_local",
) -> tuple[float | None, int]:
    """Get the average prediction over the past 3 days (including target date).

    Args:
        predictions: DataFrame with predictions
        target_date: The end date of the 3-day window
        time_col: Column name for timestamp

    Returns:
        Tuple of (average_value, count_of_predictions) or (None, 0) if not found.
    """
    # Get the date part
    target_day = target_date.date() if hasattr(target_date, 'date') else target_date

    # Calculate 3-day window: target_date and 2 days before
    start_day = target_day - timedelta(days=2)

    mask = (predictions[time_col].dt.date >= start_day) & \
           (predictions[time_col].dt.date <= target_day)

    valid_predictions = predictions[mask]

    if valid_predictions.empty:
        return None, 0

    avg_value = valid_predictions["value"].mean()
    return float(avg_value), len(valid_predictions)


def get_12hour_avg_prediction(
    predictions: pd.DataFrame,
    eod_time: datetime,
    time_col: str = "datetime_local",
) -> tuple[float | None, int]:
    """Get the average prediction over the 12 hours before end-of-day.

    Args:
        predictions: DataFrame with predictions
        eod_time: The end-of-day time (typically 00:00 of next day in UTC)
        time_col: Column name for timestamp

    Returns:
        Tuple of (average_value, count_of_predictions) or (None, 0) if not found.
    """
    # Calculate the 12-hour window ending at EOD
    # EOD is typically 00:00 of next day, so we want 12:00 to 23:59:59 of previous day
    # In other words, from (eod_time - 12 hours) to (eod_time - 1 second)
    end_time = eod_time - timedelta(seconds=1)  # Just before midnight
    start_time = eod_time - timedelta(hours=12)

    mask = (predictions[time_col] >= start_time) & \
           (predictions[time_col] <= end_time)

    valid_predictions = predictions[mask]

    if valid_predictions.empty:
        return None, 0

    avg_value = valid_predictions["value"].mean()
    return float(avg_value), len(valid_predictions)


def calculate_earnings(
    polymarket_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    bet_amount: float = 1.0,
    betting_frequency: str = "hourly",
    aggregation_method: str = "latest",
    local_tz_offset_hours: int = -5,  # EST is UTC-5
    model_name: str = "unknown",
    winning_bucket: str = "",
    month: str = "",
    variable: str = "cpi_yoy",
    buckets: list = None,
    official_release_utc: datetime | None = None,
) -> dict:
    """
    Calculate earnings from betting based on model predictions.

    Args:
        polymarket_df: Polymarket price data (timestamps in UTC)
        predictions_df: Model predictions (timestamps in local time, e.g., EST)
        bet_amount: Amount to bet each period (default $1)
        betting_frequency: "hourly" or "daily"
        aggregation_method: How to aggregate predictions:
            - "latest": Use the most recent prediction (default)
            - "daily_avg": Use average prediction over the day (for daily betting)
            - "3day_avg": Use average prediction over the past 3 days
            - "12hour_avg": Use average prediction over the 12 hours before EOD (for daily betting)
        local_tz_offset_hours: Local timezone offset from UTC (e.g., -5 for EST)
        model_name: Name of the model for reporting
        winning_bucket: The bucket that won (based on actual outcome)
        month: Month identifier (e.g., "2025-11")
        variable: Variable name (e.g., "cpi_yoy", "nonfarm_payrolls_change", "unemployment_rate")
        buckets: List of bucket names for this variable

    Returns:
        Dictionary with earnings breakdown
    """
    if buckets is None:
        buckets = ["≤2.8%", "2.9%", "3.0%", "3.1%", "≥3.2%"]  # Default CPI buckets

    results = {
        "model_name": model_name,
        "month": month,
        "variable": variable,
        "betting_frequency": betting_frequency,
        "aggregation_method": aggregation_method,
        "bet_amount": bet_amount,
        "bets": [],
        "total_invested": 0.0,
        "total_shares": {bucket: 0.0 for bucket in buckets},
        "winning_shares": 0.0,
        "final_value": 0.0,
        "profit": 0.0,
        "return_pct": 0.0,
        "winning_bucket": winning_bucket,
    }

    # Convert predictions from local time to UTC for matching with Polymarket data
    # EST (UTC-5): To convert EST -> UTC, add 5 hours
    # Formula: UTC = local_time - offset (where offset is negative for west of UTC)
    # Example: 10:00 EST -> 10:00 - (-5 hours) = 15:00 UTC ✓
    predictions_df = predictions_df.copy()
    predictions_df["datetime_utc"] = predictions_df["datetime_local"] - timedelta(hours=local_tz_offset_hours)

    print(f"\nTimezone conversion: local_tz_offset = {local_tz_offset_hours} hours (EST=UTC-5)")
    print(f"Aggregation method: {aggregation_method}")
    if len(predictions_df) > 0:
        print(f"  Sample conversion: {predictions_df['datetime_local'].iloc[0]} (local) -> {predictions_df['datetime_utc'].iloc[0]} (UTC)")

    cutoff_time, resolution_time = get_effective_cutoff_time(
        polymarket_df,
        winning_bucket,
        official_release_utc=official_release_utc,
    )

    print(f"Market resolution detected at: {resolution_time}")
    print(f"Effective betting cutoff: {cutoff_time}")
    if not predictions_df.empty and predictions_df["datetime_utc"].min() >= cutoff_time:
        print(
            "  Note: earliest prediction arrives at or after the effective cutoff, "
            "so no bets can be placed."
        )

    # Store resolution time in results for plotting
    results["resolution_time"] = resolution_time
    results["cutoff_time"] = cutoff_time

    # Filter to only periods before the effective cutoff.
    trading_df = polymarket_df[polymarket_df["datetime_utc"] < cutoff_time].copy()

    if betting_frequency == "daily":
        # For daily betting, use end-of-day price (beginning of next day)
        # E.g., for day 11-14, use price at "11-15-2025 00:00"
        trading_df["date"] = trading_df["datetime_utc"].dt.date

        # Get unique dates and find the price at the START of the NEXT day
        unique_dates = sorted(trading_df["date"].unique())

        daily_rows = []
        for target_date in unique_dates:
            # The end-of-day price for target_date is at the beginning of next day (00:00)
            next_day = target_date + timedelta(days=1)

            # Find the row at next_day 00:00 (first hour of next day = end of target_date)
            next_day_rows = trading_df[trading_df["datetime_utc"].dt.date == next_day]

            if not next_day_rows.empty:
                # Use the first hour of next day (which is end of target_date)
                price_row = next_day_rows.iloc[0].copy()
                price_row["betting_date"] = target_date
                price_row["price_time_utc"] = price_row["datetime_utc"]
                daily_rows.append(price_row)
            else:
                # Fallback: use the last hour of target_date if next day not available
                target_day_rows = trading_df[trading_df["date"] == target_date]
                if not target_day_rows.empty:
                    price_row = target_day_rows.iloc[-1].copy()
                    price_row["betting_date"] = target_date
                    price_row["price_time_utc"] = price_row["datetime_utc"]
                    daily_rows.append(price_row)

        if daily_rows:
            trading_df = pd.DataFrame(daily_rows)
        else:
            trading_df = pd.DataFrame()

        print(f"Daily betting: using end-of-day prices (start of next day)")

    print(f"Total trading periods: {len(trading_df)}")

    # Track predictions used
    hours_without_prediction = 0

    for _, row in trading_df.iterrows():
        current_time = row["datetime_utc"]

        # Get prediction based on aggregation method
        pred_time_info = None  # For tracking prediction time in output

        if aggregation_method == "latest":
            # Use the most recent prediction before this hour
            latest_pred, pred_time_utc = get_latest_prediction_before(
                predictions_df,
                current_time,
                time_col="datetime_utc",
            )
            pred_time_info = pred_time_utc

        elif aggregation_method == "daily_avg":
            if betting_frequency == "daily":
                # Use the betting_date for daily average
                target_date = row.get("betting_date", current_time.date())
            else:
                target_date = current_time.date()

            # Use datetime_utc for consistency - all data uses UTC to determine days
            latest_pred, pred_count = get_daily_avg_prediction(
                predictions_df,
                target_date,
                time_col="datetime_utc",
            )
            pred_time_info = f"daily_avg({target_date}, n={pred_count})"

        elif aggregation_method == "3day_avg":
            if betting_frequency == "daily":
                target_date = row.get("betting_date", current_time.date())
            else:
                target_date = current_time.date()

            # Use datetime_utc for consistency - all data uses UTC to determine days
            latest_pred, pred_count = get_3day_avg_prediction(
                predictions_df,
                target_date,
                time_col="datetime_utc",
            )
            pred_time_info = f"3day_avg({target_date}, n={pred_count})"

        elif aggregation_method == "12hour_avg":
            # Use the EOD time (price_time_utc) to calculate 12-hour window
            # For daily betting, this is typically 00:00 of the next day
            eod_time = row.get("price_time_utc", current_time)

            # Use datetime_utc for consistency
            latest_pred, pred_count = get_12hour_avg_prediction(
                predictions_df,
                eod_time,
                time_col="datetime_utc",
            )
            pred_time_info = f"12hour_avg(EOD={eod_time}, n={pred_count})"

        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")

        if latest_pred is None:
            hours_without_prediction += 1
            continue

        # Map prediction to bucket
        predicted_bucket = map_prediction_to_bucket(latest_pred, variable, buckets)

        # Get current price for predicted bucket
        share_price = float(row[predicted_bucket])

        if share_price <= 0 or share_price >= 1:
            continue  # Skip invalid prices

        # Calculate shares bought
        shares_bought = bet_amount / share_price

        # Find corresponding local time for the prediction (only for "latest" method)
        pred_time_local = None
        if aggregation_method == "latest" and pred_time_info is not None:
            pred_row = predictions_df[predictions_df["datetime_utc"] == pred_time_info]
            pred_time_local = pred_row["datetime_local"].iloc[0] if len(pred_row) > 0 else None

        # Record bet
        bet_info = {
            "datetime_utc": current_time,
            "betting_date": row.get("betting_date") if betting_frequency == "daily" else current_time.date(),
            "price_time_utc": row.get("price_time_utc", current_time),
            "prediction_time_local": pred_time_local,
            "prediction_info": str(pred_time_info),
            "prediction": latest_pred,
            "predicted_bucket": predicted_bucket,
            "share_price": share_price,
            "bet_amount": bet_amount,
            "shares_bought": shares_bought,
            "is_winning_bet": predicted_bucket == winning_bucket,
        }
        results["bets"].append(bet_info)

        # Update totals
        results["total_invested"] += bet_amount
        results["total_shares"][predicted_bucket] += shares_bought

        if predicted_bucket == winning_bucket:
            results["winning_shares"] += shares_bought

    # Calculate final value (winning shares pay $1 each)
    results["final_value"] = results["winning_shares"] * 1.0
    results["profit"] = results["final_value"] - results["total_invested"]

    if results["total_invested"] > 0:
        results["return_pct"] = (results["profit"] / results["total_invested"]) * 100

    results["hours_without_prediction"] = hours_without_prediction
    results["total_bets"] = len(results["bets"])

    return results


def calculate_fed_nowcast_earnings(
    polymarket_df: pd.DataFrame,
    nowcast_df: pd.DataFrame,
    bet_amount: float = 1.0,
    betting_mode: str = "hourly",  # "hourly" or "daily"
    model_name: str = "fed-nowcast",
    winning_bucket: str = "",
    month: str = "",
    variable: str = "unemployment_rate",
    buckets: list = None,
    official_release_utc: datetime | None = None,
) -> dict:
    """
    Calculate earnings from betting based on Fed nowcast releases.

    Two betting modes:
    1. "hourly": Bet every available Polymarket hour using the latest nowcast available
    2. "daily": Bet daily using end-of-day prices, carrying forward the latest nowcast value

    Args:
        polymarket_df: Polymarket price data (timestamps in UTC)
        nowcast_df: Fed nowcast data with release times
        bet_amount: Amount to bet each period (default $1)
        betting_mode: "hourly" (bet each hour with latest nowcast) or "daily" (bet daily with EOD prices)
        model_name: Name of the model for reporting
        winning_bucket: The bucket that won (based on actual outcome)
        month: Month identifier (e.g., "2025-11")
        variable: Variable name (e.g., "unemployment_rate")
        buckets: List of bucket names for this variable

    Returns:
        Dictionary with earnings breakdown
    """
    if buckets is None:
        buckets = ["≤4.4%", "4.5%", "4.6%", "4.7%", "4.8%", "≥4.9%"]

    results = {
        "model_name": model_name,
        "month": month,
        "variable": variable,
        "betting_frequency": betting_mode,
        "aggregation_method": "latest",  # Both modes use latest available nowcast
        "bet_amount": bet_amount,
        "bets": [],
        "total_invested": 0.0,
        "total_shares": {bucket: 0.0 for bucket in buckets},
        "winning_shares": 0.0,
        "final_value": 0.0,
        "profit": 0.0,
        "return_pct": 0.0,
        "winning_bucket": winning_bucket,
    }

    print(f"\nFed Nowcast betting mode: {betting_mode}")
    print(f"Nowcast releases: {len(nowcast_df)}")
    for _, row in nowcast_df.iterrows():
        print(f"  {row['release_type']}: {row['release_datetime_et']} ET -> {row['value']}")

    cutoff_time, resolution_time = get_effective_cutoff_time(
        polymarket_df,
        winning_bucket,
        official_release_utc=official_release_utc,
    )

    print(f"Market resolution detected at: {resolution_time}")
    print(f"Effective betting cutoff: {cutoff_time}")

    # Store resolution time in results for plotting
    results["resolution_time"] = resolution_time
    results["cutoff_time"] = cutoff_time

    # Filter to only periods before the effective cutoff.
    trading_df = polymarket_df[polymarket_df["datetime_utc"] < cutoff_time].copy()

    periods_without_prediction = 0

    if betting_mode == "hourly":
        # Hourly mode: bet every hour with the latest nowcast available at that time.
        for _, hour_row in trading_df.iterrows():
            bet_time = hour_row["datetime_utc"]
            valid_nowcasts = nowcast_df[nowcast_df["release_datetime_utc"] <= bet_time]
            if valid_nowcasts.empty:
                periods_without_prediction += 1
                continue

            latest_nowcast = valid_nowcasts.iloc[-1]
            nowcast_value = latest_nowcast["value"]
            pred_time_local = latest_nowcast["release_datetime_et"]
            release_type = latest_nowcast["release_type"]

            predicted_bucket = map_prediction_to_bucket(nowcast_value, variable, buckets)
            share_price = float(hour_row[predicted_bucket])

            if share_price <= 0 or share_price >= 1:
                periods_without_prediction += 1
                continue

            shares_bought = bet_amount / share_price

            bet_info = {
                "datetime_utc": bet_time,
                "betting_date": bet_time.date(),
                "price_time_utc": bet_time,
                "prediction_time_local": pred_time_local,
                "prediction_info": f"latest_{release_type}_{pred_time_local}",
                "prediction": nowcast_value,
                "predicted_bucket": predicted_bucket,
                "share_price": share_price,
                "bet_amount": bet_amount,
                "shares_bought": shares_bought,
                "is_winning_bet": predicted_bucket == winning_bucket,
            }
            results["bets"].append(bet_info)
            results["total_invested"] += bet_amount
            results["total_shares"][predicted_bucket] += shares_bought

            if predicted_bucket == winning_bucket:
                results["winning_shares"] += shares_bought

    else:  # daily mode
        # Daily mode: bet every day using end-of-day price
        # Use the latest nowcast value available at that time
        trading_df["date"] = trading_df["datetime_utc"].dt.date
        unique_dates = sorted(trading_df["date"].unique())

        for target_date in unique_dates:
            # Find the latest nowcast available before end of this day
            day_end_utc = datetime.combine(target_date, datetime.max.time().replace(microsecond=0))

            valid_nowcasts = nowcast_df[nowcast_df["release_datetime_utc"] <= day_end_utc]
            if valid_nowcasts.empty:
                periods_without_prediction += 1
                continue  # No nowcast available yet

            latest_nowcast = valid_nowcasts.iloc[-1]
            nowcast_value = latest_nowcast["value"]

            # Get end-of-day price (first hour of next day)
            next_day = target_date + timedelta(days=1)
            next_day_rows = trading_df[trading_df["datetime_utc"].dt.date == next_day]

            if not next_day_rows.empty:
                price_row = next_day_rows.iloc[0]
            else:
                # Fallback: use last hour of target_date
                target_day_rows = trading_df[trading_df["date"] == target_date]
                if target_day_rows.empty:
                    continue
                price_row = target_day_rows.iloc[-1]

            bet_time = price_row["datetime_utc"]

            # Map prediction to bucket
            predicted_bucket = map_prediction_to_bucket(nowcast_value, variable, buckets)
            share_price = float(price_row[predicted_bucket])

            if share_price <= 0 or share_price >= 1:
                periods_without_prediction += 1
                continue

            shares_bought = bet_amount / share_price

            bet_info = {
                "datetime_utc": bet_time,
                "betting_date": target_date,
                "price_time_utc": bet_time,
                "prediction_time_local": latest_nowcast["release_datetime_et"],
                "prediction_info": f"daily_{latest_nowcast['release_type']}",
                "prediction": nowcast_value,
                "predicted_bucket": predicted_bucket,
                "share_price": share_price,
                "bet_amount": bet_amount,
                "shares_bought": shares_bought,
                "is_winning_bet": predicted_bucket == winning_bucket,
            }
            results["bets"].append(bet_info)
            results["total_invested"] += bet_amount
            results["total_shares"][predicted_bucket] += shares_bought

            if predicted_bucket == winning_bucket:
                results["winning_shares"] += shares_bought

    # Calculate final value (winning shares pay $1 each)
    results["final_value"] = results["winning_shares"] * 1.0
    results["profit"] = results["final_value"] - results["total_invested"]

    if results["total_invested"] > 0:
        results["return_pct"] = (results["profit"] / results["total_invested"]) * 100

    results["hours_without_prediction"] = periods_without_prediction
    results["total_bets"] = len(results["bets"])

    return results


def calculate_fed_cpi_earnings(
    polymarket_df: pd.DataFrame,
    fed_predictions_df: pd.DataFrame,
    bet_amount: float = 1.0,
    betting_mode: str = "hourly",  # "hourly" or "daily"
    model_name: str = "fed-forecast",
    winning_bucket: str = "",
    month: str = "",
    variable: str = "cpi_yoy",
    buckets: list = None,
    official_release_utc: datetime | None = None,
) -> dict:
    """
    Calculate earnings from betting based on Fed CPI predictions released at 10 AM ET daily.

    Two betting modes:
    1. "hourly": Bet every available Polymarket hour using the latest prediction
    2. "daily": Bet daily at end-of-day prices using the latest prediction available by day-end

    Args:
        polymarket_df: Polymarket price data (timestamps in UTC)
        fed_predictions_df: Fed predictions with datetime_utc column
        bet_amount: Amount to bet each period (default $1)
        betting_mode: "hourly" (bet every hour) or "daily" (bet daily with EOD prices)
        model_name: Name of the model for reporting
        winning_bucket: The bucket that won (based on actual outcome)
        month: Month identifier (e.g., "2025-11")
        variable: Variable name (e.g., "cpi_yoy")
        buckets: List of bucket names for this variable

    Returns:
        Dictionary with earnings breakdown
    """
    if buckets is None:
        buckets = ["<2.4%", "2.4%", "2.5%", "2.6%", "2.7%", "≥2.8%"]

    results = {
        "model_name": model_name,
        "month": month,
        "variable": variable,
        "betting_frequency": betting_mode,
        "aggregation_method": "latest",
        "bet_amount": bet_amount,
        "bets": [],
        "total_invested": 0.0,
        "total_shares": {bucket: 0.0 for bucket in buckets},
        "winning_shares": 0.0,
        "final_value": 0.0,
        "profit": 0.0,
        "return_pct": 0.0,
        "winning_bucket": winning_bucket,
    }

    print(f"\nFed CPI betting mode: {betting_mode}")
    print(f"Fed predictions: {len(fed_predictions_df)} daily releases")
    if len(fed_predictions_df) > 0:
        print(f"  First: {fed_predictions_df.iloc[0]['datetime_local']} ET -> {fed_predictions_df.iloc[0]['value']:.4f}%")
        print(f"  Last: {fed_predictions_df.iloc[-1]['datetime_local']} ET -> {fed_predictions_df.iloc[-1]['value']:.4f}%")

    cutoff_time, resolution_time = get_effective_cutoff_time(
        polymarket_df,
        winning_bucket,
        official_release_utc=official_release_utc,
    )

    print(f"Market resolution detected at: {resolution_time}")
    print(f"Effective betting cutoff: {cutoff_time}")

    # Store resolution time in results for plotting
    results["resolution_time"] = resolution_time
    results["cutoff_time"] = cutoff_time

    # Filter to only periods before the effective cutoff.
    trading_df = polymarket_df[polymarket_df["datetime_utc"] < cutoff_time].copy()

    hours_without_prediction = 0

    if betting_mode == "hourly":
        # Hourly mode: bet every hour using the latest available prediction
        for _, hour_row in trading_df.iterrows():
            bet_time = hour_row["datetime_utc"]

            # Find latest prediction before this hour
            valid_preds = fed_predictions_df[fed_predictions_df["datetime_utc"] <= bet_time]
            if valid_preds.empty:
                hours_without_prediction += 1
                continue

            latest_pred = valid_preds.iloc[-1]
            prediction_value = latest_pred["value"]
            pred_time_local = latest_pred["datetime_local"]

            # Map prediction to bucket
            predicted_bucket = map_prediction_to_bucket(prediction_value, variable, buckets)
            share_price = float(hour_row[predicted_bucket])

            if share_price <= 0 or share_price >= 1:
                hours_without_prediction += 1
                continue

            shares_bought = bet_amount / share_price

            bet_info = {
                "datetime_utc": bet_time,
                "betting_date": bet_time.date(),
                "price_time_utc": bet_time,
                "prediction_time_local": pred_time_local,
                "prediction_info": f"latest_{pred_time_local.date()}",
                "prediction": prediction_value,
                "predicted_bucket": predicted_bucket,
                "share_price": share_price,
                "bet_amount": bet_amount,
                "shares_bought": shares_bought,
                "is_winning_bet": predicted_bucket == winning_bucket,
            }
            results["bets"].append(bet_info)
            results["total_invested"] += bet_amount
            results["total_shares"][predicted_bucket] += shares_bought

            if predicted_bucket == winning_bucket:
                results["winning_shares"] += shares_bought

    else:  # daily mode
        # Daily mode: bet each day at end-of-day price with latest prediction available by day-end.
        trading_df["date"] = trading_df["datetime_utc"].dt.date
        unique_dates = sorted(trading_df["date"].unique())

        for target_date in unique_dates:
            day_end_utc = datetime.combine(target_date, datetime.max.time().replace(microsecond=0))

            valid_preds = fed_predictions_df[fed_predictions_df["datetime_utc"] <= day_end_utc]
            if valid_preds.empty:
                hours_without_prediction += 1
                continue

            latest_pred = valid_preds.iloc[-1]
            prediction_value = latest_pred["value"]
            pred_time_local = latest_pred["datetime_local"]

            next_day = target_date + timedelta(days=1)
            next_day_rows = trading_df[trading_df["datetime_utc"].dt.date == next_day]
            if not next_day_rows.empty:
                price_row = next_day_rows.iloc[0]
            else:
                target_day_rows = trading_df[trading_df["date"] == target_date]
                if target_day_rows.empty:
                    hours_without_prediction += 1
                    continue
                price_row = target_day_rows.iloc[-1]

            bet_time = price_row["datetime_utc"]
            predicted_bucket = map_prediction_to_bucket(prediction_value, variable, buckets)
            share_price = float(price_row[predicted_bucket])

            if share_price <= 0 or share_price >= 1:
                hours_without_prediction += 1
                continue

            shares_bought = bet_amount / share_price

            bet_info = {
                "datetime_utc": bet_time,
                "betting_date": target_date,
                "price_time_utc": bet_time,
                "prediction_time_local": pred_time_local,
                "prediction_info": f"daily_latest_{pred_time_local.date()}",
                "prediction": prediction_value,
                "predicted_bucket": predicted_bucket,
                "share_price": share_price,
                "bet_amount": bet_amount,
                "shares_bought": shares_bought,
                "is_winning_bet": predicted_bucket == winning_bucket,
            }
            results["bets"].append(bet_info)
            results["total_invested"] += bet_amount
            results["total_shares"][predicted_bucket] += shares_bought

            if predicted_bucket == winning_bucket:
                results["winning_shares"] += shares_bought

    # Calculate final value (winning shares pay $1 each)
    results["final_value"] = results["winning_shares"] * 1.0
    results["profit"] = results["final_value"] - results["total_invested"]

    if results["total_invested"] > 0:
        results["return_pct"] = (results["profit"] / results["total_invested"]) * 100

    results["hours_without_prediction"] = hours_without_prediction
    results["total_bets"] = len(results["bets"])

    return results


def print_results(results: dict) -> None:
    """Print earnings results summary."""
    winning_bucket = results.get('winning_bucket', '')
    month = results.get('month', '')
    variable = results.get('variable', 'cpi_yoy')
    var_name = VARIABLE_CONFIGS.get(variable, {}).get('name', variable)
    month_str = f" [{month}]" if month else ""

    print("\n" + "=" * 70)
    agg_method = results.get('aggregation_method', 'latest')
    print(f"POLYMARKET EARNINGS: {results['model_name']}{month_str} - {var_name} ({results['betting_frequency']}, {agg_method})")
    print("=" * 70)

    print(f"\n📊 BETTING SUMMARY")
    print(f"   Total bets placed: {results['total_bets']}")
    print(f"   Periods without prediction: {results['hours_without_prediction']}")
    print(f"   Total invested: ${results['total_invested']:.2f}")

    print(f"\n📈 SHARES ACQUIRED BY BUCKET")
    for bucket, shares in results["total_shares"].items():
        marker = " ✓ WINNER" if bucket == winning_bucket else ""
        print(f"   {bucket}: {shares:.2f} shares{marker}")

    print(f"\n💰 FINAL RESULTS")
    print(f"   Winning bucket: {winning_bucket}")
    print(f"   Winning shares: {results['winning_shares']:.2f}")
    print(f"   Final value: ${results['final_value']:.2f}")
    print(f"   Total invested: ${results['total_invested']:.2f}")
    print(f"   Profit/Loss: ${results['profit']:.2f}")
    print(f"   Return: {results['return_pct']:.1f}%")

    # Show prediction distribution
    if results["bets"]:
        bets_df = pd.DataFrame(results["bets"])
        print(f"\n📉 PREDICTION DISTRIBUTION")
        pred_dist = bets_df["predicted_bucket"].value_counts()
        for bucket, count in pred_dist.items():
            pct = count / len(bets_df) * 100
            print(f"   {bucket}: {count} bets ({pct:.1f}%)")

        print(f"\n📊 PREDICTION VALUE STATISTICS")
        print(f"   Min prediction: {bets_df['prediction'].min():.2f}%")
        print(f"   Max prediction: {bets_df['prediction'].max():.2f}%")
        print(f"   Mean prediction: {bets_df['prediction'].mean():.2f}%")
        print(f"   Median prediction: {bets_df['prediction'].median():.2f}%")

        # Calculate average price paid for winning vs losing bets
        winning_bets = bets_df[bets_df["is_winning_bet"]]
        losing_bets = bets_df[~bets_df["is_winning_bet"]]

        if len(winning_bets) > 0:
            print(f"\n   Winning bets: {len(winning_bets)} (avg price: ${winning_bets['share_price'].mean():.3f})")
        if len(losing_bets) > 0:
            print(f"   Losing bets: {len(losing_bets)} (avg price: ${losing_bets['share_price'].mean():.3f})")


def print_winning_slots(results: dict, max_slots: int = 50) -> pd.DataFrame | None:
    """Print and return the hour slots where the model predicted the winning bucket correctly."""
    winning_bucket = results.get('winning_bucket', '')

    if not results["bets"]:
        print("\nNo bets to analyze.")
        return None

    bets_df = pd.DataFrame(results["bets"])
    winning_bets = bets_df[bets_df["is_winning_bet"]].copy()

    print(f"\n{'=' * 70}")
    print(f"WINNING PREDICTIONS (betting on {winning_bucket})")
    print(f"{'=' * 70}")
    print(f"Total winning bets: {len(winning_bets)} / {len(bets_df)} ({len(winning_bets)/len(bets_df)*100:.1f}%)")

    if len(winning_bets) == 0:
        print("No winning bets found.")
        return None

    # Sort by datetime
    winning_bets = winning_bets.sort_values("datetime_utc").reset_index(drop=True)

    print(f"\n{'Bet Hour (UTC)':<22} {'Pred Time (EST)':<22} {'Prediction':<12} {'Price':<8} {'Shares':<10}")
    print("-" * 80)

    display_count = min(len(winning_bets), max_slots)
    for i, row in winning_bets.head(display_count).iterrows():
        bet_time = row["datetime_utc"].strftime("%Y-%m-%d %H:%M") if pd.notna(row["datetime_utc"]) else "N/A"
        pred_time = row["prediction_time_local"].strftime("%Y-%m-%d %H:%M") if pd.notna(row.get("prediction_time_local")) else "N/A"
        print(f"{bet_time:<22} {pred_time:<22} {row['prediction']:<12.2f} ${row['share_price']:<7.3f} {row['shares_bought']:<10.2f}")

    if len(winning_bets) > max_slots:
        print(f"... and {len(winning_bets) - max_slots} more winning bets")

    # Summary statistics
    print(f"\nWinning bet statistics:")
    print(f"  Avg prediction value: {winning_bets['prediction'].mean():.3f}%")
    print(f"  Avg share price paid: ${winning_bets['share_price'].mean():.3f}")
    print(f"  Total shares acquired: {winning_bets['shares_bought'].sum():.2f}")
    print(f"  Total invested in winning: ${winning_bets['bet_amount'].sum():.2f}")

    return winning_bets


def save_detailed_results(results: dict, output_path: Path) -> None:
    """Save detailed bet-by-bet results to CSV."""
    if not results["bets"]:
        print("No bets to save.")
        return

    bets_df = pd.DataFrame(results["bets"])
    bets_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")


def save_earnings_summary(results: dict, output_path: Path, append: bool = True) -> None:
    """Save earnings summary to CSV."""
    summary = {
        "model_name": results["model_name"],
        "month": results.get("month", ""),
        "variable": results.get("variable", "cpi_yoy"),
        "betting_frequency": results["betting_frequency"],
        "aggregation_method": results.get("aggregation_method", "latest"),
        "bet_amount": results["bet_amount"],
        "total_bets": results["total_bets"],
        "periods_without_prediction": results["hours_without_prediction"],
        "total_invested": results["total_invested"],
        "winning_shares": results["winning_shares"],
        "final_value": results["final_value"],
        "profit": results["profit"],
        "return_pct": results["return_pct"],
        "winning_bucket": results.get("winning_bucket", ""),
    }

    # Add shares by bucket (use buckets from results)
    for bucket, shares in results["total_shares"].items():
        col_name = f"shares_{bucket.replace('≤', 'lte').replace('≥', 'gte').replace('%', '').replace('<', 'lt').replace('>', 'gt').replace('–', '_').replace('-', '_')}"
        summary[col_name] = shares

    summary_df = pd.DataFrame([summary])

    if append and output_path.exists():
        existing_df = pd.read_csv(output_path)
        # Remove existing entry for same model/month/frequency/aggregation if exists
        agg_method = results.get("aggregation_method", "latest")
        month = results.get("month", "")
        # Handle case where columns might not exist in old files
        if "aggregation_method" in existing_df.columns and "month" in existing_df.columns:
            mask = ~((existing_df["model_name"] == results["model_name"]) &
                     (existing_df["month"] == month) &
                     (existing_df["betting_frequency"] == results["betting_frequency"]) &
                     (existing_df["aggregation_method"] == agg_method))
        elif "aggregation_method" in existing_df.columns:
            mask = ~((existing_df["model_name"] == results["model_name"]) &
                     (existing_df["betting_frequency"] == results["betting_frequency"]) &
                     (existing_df["aggregation_method"] == agg_method))
        else:
            mask = ~((existing_df["model_name"] == results["model_name"]) &
                     (existing_df["betting_frequency"] == results["betting_frequency"]))
        existing_df = existing_df[mask]
        summary_df = pd.concat([existing_df, summary_df], ignore_index=True)

    summary_df.to_csv(output_path, index=False)
    print(f"Earnings summary saved to: {output_path}")


def calculate_cumulative_returns_series(results: dict) -> pd.DataFrame:
    """
    Calculate cumulative returns time series from betting results.

    Returns DataFrame with columns: datetime_utc, cumulative_return_pct, cumulative_invested, cumulative_profit
    """
    if not results["bets"]:
        return pd.DataFrame()

    bets_df = pd.DataFrame(results["bets"])
    bets_df = bets_df.sort_values("datetime_utc").reset_index(drop=True)

    # Calculate cumulative metrics
    cumulative_invested = 0.0
    cumulative_value = 0.0
    cumulative_returns = []
    cumulative_profits = []
    cumulative_invested_list = []

    for _, row in bets_df.iterrows():
        cumulative_invested += row["bet_amount"]

        if row["is_winning_bet"]:
            # Winning shares pay $1 each at resolution
            cumulative_value += row["shares_bought"]
        # Losing bets add 0 to value

        cumulative_profit = cumulative_value - cumulative_invested

        # Calculate return percentage: (profit / invested) * 100
        if cumulative_invested > 0:
            return_pct = (cumulative_profit / cumulative_invested) * 100
        else:
            return_pct = 0.0

        cumulative_returns.append(return_pct)
        cumulative_profits.append(cumulative_profit)
        cumulative_invested_list.append(cumulative_invested)

    return pd.DataFrame({
        'datetime_utc': bets_df['datetime_utc'],
        'cumulative_return_pct': cumulative_returns,
        'cumulative_invested': cumulative_invested_list,
        'cumulative_profit': cumulative_profits,
    })


def plot_cumulative_returns(results: dict, output_path: Path) -> None:
    """Legacy single-model plot (kept for backward compatibility)."""
    plot_cumulative_returns_combined([results], output_path)


def plot_cumulative_returns_combined(
    all_results: list,
    output_path: Path,
    generated_at_label: str | None = None,
) -> None:
    """
    Generate a time series plot comparing cumulative returns across all models.

    Args:
        all_results: List of result dictionaries from different models
        output_path: Path to save the plot
    """
    if not all_results:
        print("No results to plot.")
        return

    # Filter out results with no bets; log which models were excluded so
    # missing lines in combined plots are not silent.
    with_bets = [r for r in all_results if r.get("bets")]
    without_bets = [r for r in all_results if not r.get("bets")]
    if without_bets:
        skipped = [
            f"{r.get('model_name', 'unknown')}[{r.get('month', '')}|{r.get('betting_frequency', '')}|{r.get('aggregation_method', '')}]"
            for r in without_bets
        ]
        print(f"Skipping models with zero bets in combined plot: {', '.join(skipped)}")

    all_results = with_bets
    if not all_results:
        print("No bets to plot.")
        return

    # Get metadata from first result
    month = all_results[0].get("month", "")
    variable = all_results[0].get("variable", "cpi_yoy")
    frequency = all_results[0].get("betting_frequency", "")
    agg_method = all_results[0].get("aggregation_method", "latest")

    var_name = VARIABLE_CONFIGS.get(variable, {}).get("name", variable)
    month_name = MONTH_CONFIGS.get(month, {}).get("name", month)

    # Define model styles - academic color palette (lighter, more distinguishable)
    # Colors: steel blue, brick red, teal, slate
    model_styles = {
        "gpt-5-search-api": {"color": "#4682B4", "linestyle": "-", "marker": "o", "label": "GPT-5"},
        "claude-sonnet-4.5": {"color": "#B45C5C", "linestyle": "-", "marker": "s", "label": "Claude 4.5"},
        "claude-code-agent": {"color": "#6A3D9A", "linestyle": "-", "marker": "*", "label": "Agent (Claude Code)"},
        "qwen3-next-80b-a3b-instruct": {"color": "#B8860B", "linestyle": "-", "marker": "P", "label": "Qwen 80B"},
        "qwen3-235b-a22b-instruct-2507": {"color": "#2E8B57", "linestyle": "-", "marker": "X", "label": "Qwen 235B"},
        "fed-forecast": {"color": "#3D9970", "linestyle": "--", "marker": "^", "label": "Clev. Fed Nowcast"},
        "fed-nowcast": {"color": "#7B7B7B", "linestyle": "--", "marker": "D", "label": "Chic. Fed Nowcast"},
        "fed-atlanta": {"color": "#1B9E77", "linestyle": "--", "marker": "^", "label": "Atlanta Fed GDPNow"},
        "fed-newyork": {"color": "#D95F02", "linestyle": "--", "marker": "v", "label": "NY Fed Nowcast"},
        "fed-stlouis": {"color": "#7570B3", "linestyle": "--", "marker": "P", "label": "St. Louis Fed"},
    }

    # Create the plot with white background. Sized to comfortably hold up to
    # 10 simultaneous model lines (gpt, claude-sonnet, claude-code-agent,
    # qwen-next, qwen-235, plus up to 5 fed series for GDP).
    fig, ax = plt.subplots(figsize=(17, 9), facecolor='white')
    ax.set_facecolor('white')

    # Find the latest resolution time across all results (ground truth release date)
    resolution_times = [r.get("resolution_time") for r in all_results if r.get("resolution_time") is not None]
    global_resolution_time = max(resolution_times) if resolution_times else None

    # Track stats for annotation and line end positions for labels
    stats_lines = []
    line_end_positions = []  # (y_position, label, color)

    # Plot each model
    for results in all_results:
        model_name = results.get("model_name", "Unknown")
        style = model_styles.get(model_name, {"color": "#666666", "linestyle": "-", "marker": "o", "label": model_name})

        # Calculate cumulative returns
        series_df = calculate_cumulative_returns_series(results)
        if series_df.empty:
            continue

        # Extend the line to the global resolution time if it ends earlier
        last_time = series_df["datetime_utc"].iloc[-1]
        final_return = series_df["cumulative_return_pct"].iloc[-1]

        if global_resolution_time is not None and last_time < global_resolution_time:
            # Add a point at the resolution time with the same final return value
            extension_df = pd.DataFrame({
                'datetime_utc': [global_resolution_time],
                'cumulative_return_pct': [final_return],
                'cumulative_invested': [series_df["cumulative_invested"].iloc[-1]],
                'cumulative_profit': [series_df["cumulative_profit"].iloc[-1]],
            })
            series_df = pd.concat([series_df, extension_df], ignore_index=True)

        # Plot the line (with markers at intervals to avoid clutter)
        marker_interval = max(1, len(series_df) // 15)  # Show ~15 markers max
        ax.plot(series_df["datetime_utc"], series_df["cumulative_return_pct"],
                linewidth=3, color=style["color"], linestyle=style["linestyle"],
                marker=style["marker"], markevery=marker_interval, markersize=8,
                label=style["label"])

        # Collect final stats and line end positions (use extended end time)
        extended_final_time = series_df["datetime_utc"].iloc[-1]
        total_invested = results.get("total_invested", 0)
        total_bets = results.get("total_bets", 0)
        stats_lines.append(f"{style['label']}: {final_return:.1f}% (${total_invested:.0f}, {total_bets} bets)")
        line_end_positions.append((extended_final_time, final_return, style["label"], style["color"]))

    # Add horizontal line at y=0
    ax.axhline(y=0, color='#333333', linestyle='-', linewidth=1, alpha=0.6)

    # Format x-axis as datetime (UTC) with vertical labels for readability.
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=14))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.xticks(rotation=90, ha='center', fontsize=9)

    # Labels and title
    title = f"{var_name}\nCumulative Returns"
    if month_name:
        title += f" - {month_name}"
    title += f" ({frequency}, {agg_method})"

    ax.set_title(title, fontsize=14, fontweight='bold', pad=16)
    ax.set_xlabel("Time (UTC)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Cumulative Return (%)", fontsize=12, fontweight='bold')
    ax.tick_params(axis='y', labelsize=10)

    # Add line labels at the end of each line (outside plot area)
    # Sort by y position to avoid overlap
    line_end_positions.sort(key=lambda x: x[1], reverse=True)
    min_spacing = 20  # Minimum spacing between labels in percentage points

    # First pass: calculate adjusted y positions to avoid overlap
    adjusted_positions = []
    for final_time, final_return, label, color in line_end_positions:
        adjusted_y = final_return
        # Check against all previously positioned labels
        for prev_y in [p[1] for p in adjusted_positions]:
            if abs(adjusted_y - prev_y) < min_spacing:
                # Move this label down (since we're sorted high to low)
                adjusted_y = prev_y - min_spacing
        adjusted_positions.append((final_time, adjusted_y, final_return, label, color))

    for final_time, adjusted_y, final_return, label, color in adjusted_positions:
        ax.annotate(f'{label}: {final_return:.1f}%',
                    xy=(final_time, final_return),
                    xytext=(10, adjusted_y - final_return), textcoords='offset points',
                    fontsize=9, fontweight='bold', color=color,
                    ha='left', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor=color, alpha=0.9),
                    arrowprops=dict(arrowstyle='-', color=color, alpha=0.5) if abs(adjusted_y - final_return) > 2 else None)

    # Legend at upper left (away from line labels on right)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95,
              edgecolor='black', fancybox=False)
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.5)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))

    # Expand x-axis slightly to make room for labels
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0], xlim[1] + (xlim[1] - xlim[0]) * 0.15)

    if generated_at_label:
        fig.text(0.99, 0.01, f"Generated {generated_at_label}", ha='right', va='bottom', fontsize=9, color='#555555')

    plt.tight_layout()
    plt.savefig(output_path, dpi=500, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Combined returns plot saved to: {output_path}")


def plot_final_returns_comparison(
    all_results: list,
    output_path: Path,
    target_month: str = "",
    generated_at_label: str | None = None,
) -> None:
    """
    Generate a bar plot comparing final returns across all models for a specific month.

    Args:
        all_results: List of result dictionaries
        output_path: Path to save the plot
        target_month: Month to filter by (e.g., "2025-11"). If empty, uses all results.
    """
    # Filter by month if specified
    if target_month:
        filtered_results = [r for r in all_results if r.get("month") == target_month]
    else:
        filtered_results = all_results

    filtered_results = [r for r in filtered_results if r.get("total_bets", 0) > 0]
    if not filtered_results:
        print(f"No results with bets to plot for month {target_month}.")
        return

    # Get variable name from first result
    variable = filtered_results[0].get("variable", "cpi_yoy")
    var_name = VARIABLE_CONFIGS.get(variable, {}).get("name", variable)

    # Prepare data for plotting
    labels = []
    returns = []
    colors = []
    models_present = set()

    # Define colors and display names for each model - academic color palette (lighter, more distinguishable)
    model_colors = {
        "gpt-5-search-api": "#4682B4",  # Steel blue
        "claude-sonnet-4.5": "#B45C5C",  # Brick red
        "claude-code-agent": "#6A3D9A",  # Deep purple
        "qwen3-next-80b-a3b-instruct": "#B8860B",  # Dark goldenrod
        "qwen3-235b-a22b-instruct-2507": "#2E8B57",  # Sea green
        "fed-forecast": "#3D9970",  # Teal
        "fed-nowcast": "#7B7B7B",  # Slate
        "fed-atlanta": "#1B9E77",
        "fed-newyork": "#D95F02",
        "fed-stlouis": "#7570B3",
    }

    # Display names based on variable
    def get_model_display_name(model: str, variable: str) -> str:
        if model == "fed-forecast":
            return "Clev. Fed Nowcast"
        elif model == "fed-nowcast":
            return "Chic. Fed Nowcast"
        elif model == "fed-atlanta":
            return "Atlanta Fed GDPNow"
        elif model == "fed-newyork":
            return "NY Fed Nowcast"
        elif model == "fed-stlouis":
            return "St. Louis Fed"
        elif model == "gpt-5-search-api":
            return "GPT-5"
        elif model == "claude-sonnet-4.5":
            return "Claude 4.5"
        elif model == "claude-code-agent":
            return "Agent (Claude Code)"
        elif model == "qwen3-next-80b-a3b-instruct":
            return "Qwen 80B"
        elif model == "qwen3-235b-a22b-instruct-2507":
            return "Qwen 235B"
        else:
            return model

    # Track duplicate labels so model names stay readable even when repeated.
    label_counts: dict[str, int] = {}

    for r in filtered_results:
        model = r.get("model_name", "Unknown")
        models_present.add(model)

        # Keep x-axis labels compact: model name only.
        model_display = get_model_display_name(model, variable)
        label_counts[model_display] = label_counts.get(model_display, 0) + 1
        if label_counts[model_display] == 1:
            label = model_display
        else:
            label = f"{model_display} ({label_counts[model_display]})"

        labels.append(label)

        returns.append(r.get("return_pct", 0))
        colors.append(model_colors.get(model, "#666666"))

    # Adaptive width: per-bar inches scale with the longest label so multi-word
    # entries like "Agent (Claude Code)" / "Clev. Fed Nowcast" don't collide.
    longest_label = max((len(l) for l in labels), default=1)
    per_bar_in = max(2.0, 0.16 * longest_label)
    fig_width = max(12.0, len(labels) * per_bar_in)
    if variable == "real_gdp_qoq":
        fig_width = max(fig_width, 15.0)
    fig, ax = plt.subplots(figsize=(fig_width, 8.5), facecolor='white')
    ax.set_facecolor('white')

    x = range(len(labels))
    bar_width = 0.5  # Slightly wider bars for visibility
    bars = ax.bar(x, returns, width=bar_width, color=colors, edgecolor='black', linewidth=1.0)

    # Add value labels on bars
    for bar, ret in zip(bars, returns):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        offset = 1 if height >= 0 else -1
        ax.annotate(f'{ret:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, offset * 4),
                    textcoords="offset points",
                    ha='center', va=va, fontsize=15, fontweight='bold')

    # Add horizontal line at y=0
    ax.axhline(y=0, color='#333333', linestyle='-', linewidth=1.0)

    # Customize plot. Tick-label fontsize scales down a touch when there are
    # 6+ bars so longer model names ("Agent (Claude Code)") don't collide.
    ax.set_xticks(x)
    xtick_fontsize = 16 if len(labels) >= 6 else 19
    ax.set_xticklabels(labels, fontsize=xtick_fontsize, rotation=0, ha='center')
    ax.set_ylabel("Final Return (%)", fontsize=22, fontweight='bold')
    ax.tick_params(axis='y', labelsize=17)

    # Title with month info and prominent variable name
    month_name = MONTH_CONFIGS.get(target_month, {}).get("name", target_month) if target_month else "All Months"
    ax.set_title(f"{var_name}\nFinal Returns Comparison - {month_name}",
                 fontsize=24, fontweight='bold', pad=18)
    ax.grid(True, axis='y', alpha=0.4, linestyle='--', linewidth=0.5)

    # Add legend only for models that are present
    from matplotlib.patches import Patch
    legend_elements = []
    for model in sorted(models_present):
        if model in model_colors:
            display_name = get_model_display_name(model, variable)
            legend_elements.append(
                Patch(facecolor=model_colors[model], edgecolor='black', label=display_name)
            )

    if legend_elements:
        legend_cols = min(len(legend_elements), 4)
        ax.legend(
            handles=legend_elements,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.09),
            ncol=legend_cols,
            fontsize=14,
            framealpha=0.95,
            edgecolor='black',
            fancybox=False,
        )

    # Add compact method note below the axis.
    method_pairs = sorted(
        {
            (
                r.get("betting_frequency", ""),
                r.get("aggregation_method", "latest"),
            )
            for r in filtered_results
        }
    )
    if len(method_pairs) == 1 and method_pairs[0] == ("hourly", "latest"):
        method_note = "Betting method: Hourly using the latest prediction from each source."
    elif len(method_pairs) == 1:
        freq, agg = method_pairs[0]
        method_note = f"Betting method: {freq.replace('_', ' ').title()} with {agg.replace('_', ' ').title()} predictions."
    else:
        parts = [f"{f}/{a}" for f, a in method_pairs]
        method_note = "Betting methods shown: " + ", ".join(parts)

    fig.text(0.5, 0.015, method_note, ha='center', va='bottom', fontsize=12, color='#444444')
    if generated_at_label:
        fig.text(0.99, 0.015, f"Generated {generated_at_label}", ha='right', va='bottom', fontsize=10, color='#555555')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.24)

    plt.savefig(output_path, dpi=500, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Final returns comparison plot saved to: {output_path}")


def get_output_subfolder(frequency: str, aggregation_method: str) -> str:
    """Get the subfolder name for results based on frequency and aggregation."""
    return f"{frequency}_{aggregation_method}"


def build_run_timestamp() -> tuple[str, str]:
    """Return filesystem-safe and human-readable UTC timestamps for this run."""
    now_utc = datetime.now(timezone.utc)
    return now_utc.strftime("%Y%m%d_%H%M%S_UTC"), now_utc.strftime("%Y-%m-%d %H:%M UTC")


def run_calculation(
    polymarket_path: Path,
    predictions_path: Path,
    model_name: str,
    prediction_source: str,  # "gpt", "claude", or "fed"
    frequency: str,
    aggregation_method: str,
    bet_amount: float,
    tz_offset: int,
    output_dir: Path,
    summary_path: Path,
    month: str = "",
    winning_bucket: str = "",
    variable: str = "cpi_yoy",
    buckets: list = None,
    official_release_utc: datetime | None = None,
    plot_timestamp_suffix: str | None = None,
    generated_at_label: str | None = None,
) -> dict:
    """Run earnings calculation for a specific model, frequency, aggregation method, and variable."""
    month_str = f" [{month}]" if month else ""
    var_name = VARIABLE_CONFIGS.get(variable, {}).get("name", variable)
    print(f"\n{'#' * 70}")
    print(f"# Running: {model_name}{month_str} - {var_name} ({frequency}, {aggregation_method})")
    print(f"{'#' * 70}")

    # Create subfolder for this frequency/aggregation combination
    subfolder = get_output_subfolder(frequency, aggregation_method)
    output_dir = output_dir / subfolder
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Polymarket data
    print("\nLoading Polymarket data...")
    polymarket_df = load_polymarket_data(polymarket_path)
    print(f"  Loaded {len(polymarket_df)} hourly price records")

    # Load predictions based on source
    print(f"\nLoading {prediction_source.upper()} predictions for {variable}...")
    if prediction_source == "fed":
        predictions_df = load_fed_predictions(predictions_path)
        print(f"  Loaded {len(predictions_df)} daily predictions")
    else:
        predictions_df = load_model_predictions(predictions_path, variable)
        print(f"  Loaded {len(predictions_df)} predictions")

    if len(predictions_df) > 0:
        print(f"  Date range: {predictions_df['datetime_local'].min()} to {predictions_df['datetime_local'].max()}")
        print(f"  Value range: {predictions_df['value'].min():.2f} to {predictions_df['value'].max():.2f}")

    # Calculate earnings
    print(f"\nCalculating earnings (betting ${bet_amount} {frequency}, aggregation: {aggregation_method})...")
    results = calculate_earnings(
        polymarket_df,
        predictions_df,
        bet_amount=bet_amount,
        betting_frequency=frequency,
        aggregation_method=aggregation_method,
        local_tz_offset_hours=tz_offset,
        model_name=model_name,
        winning_bucket=winning_bucket,
        month=month,
        variable=variable,
        buckets=buckets,
        official_release_utc=official_release_utc,
    )

    # Print results
    print_results(results)

    # Print winning slots (when model predicted correctly)
    print_winning_slots(results, max_slots=100)

    # Save detailed results
    detail_filename = f"betting_results_{model_name}_{month}_{frequency}_{aggregation_method}.csv"
    detail_path = output_dir / detail_filename
    if results["bets"]:
        save_detailed_results(results, detail_path)
    elif detail_path.exists():
        detail_path.unlink()
        print(f"Removed stale detailed results with zero bets: {detail_path}")

    # Save summary
    save_earnings_summary(results, summary_path)

    # Generate and save returns plot
    legacy_plot_path = output_dir / f"returns_plot_{model_name}_{month}_{frequency}_{aggregation_method}.png"
    if results["bets"]:
        plot_filename = f"returns_plot_{model_name}_{month}_{frequency}_{aggregation_method}"
        if plot_timestamp_suffix:
            plot_filename += f"_{plot_timestamp_suffix}"
        plot_filename += ".png"
        plot_cumulative_returns_combined(
            [results],
            output_dir / plot_filename,
            generated_at_label=generated_at_label,
        )
    elif legacy_plot_path.exists():
        legacy_plot_path.unlink()
        print(f"Removed stale returns plot with zero bets: {legacy_plot_path}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calculate Polymarket earnings based on model predictions"
    )
    parser.add_argument(
        "--variable",
        choices=["cpi_yoy", "nonfarm_payrolls_change", "unemployment_rate", "real_gdp_qoq", "all"],
        default="all",
        help="Which variable to run (default: all)",
    )
    parser.add_argument(
        "--month",
        choices=list(MODEL_PREDICTIONS.keys()) + ["all"],
        default="all",
        help="Which month to run (default: all)",
    )
    parser.add_argument(
        "--model",
        choices=["gpt", "claude", "qwen", "agent", "fed", "all"],
        default="all",
        help="Which model to run (default: all)",
    )
    parser.add_argument(
        "--frequency",
        choices=["hourly", "daily", "all"],
        default="all",
        help="Betting frequency (default: all)",
    )
    parser.add_argument(
        "--aggregation",
        choices=["latest", "daily_avg", "12hour_avg", "all"],
        default="all",
        help="Prediction aggregation method (default: all)",
    )
    parser.add_argument(
        "--bet-amount",
        type=float,
        default=1.0,
        help="Amount to bet each period (default: $1)",
    )
    parser.add_argument(
        "--tz-offset",
        type=int,
        default=-5,
        help="Local timezone offset from UTC in hours (default: -5 for EST)",
    )
    parser.add_argument(
        "--output-dir",
        default="polymarket_return/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--summary-file",
        default="earnings_summary.csv",
        help="Filename for earnings summary CSV",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Base directory for relative paths",
    )

    args = parser.parse_args()
    base_dir = Path(args.base_dir)
    output_base_dir = base_dir / args.output_dir

    # Create output directory if it doesn't exist
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Redirect stdout to output.log
    import sys
    log_file = output_base_dir / "output.log"
    log_handle = open(log_file, 'w', buffering=1)  # Line buffered
    original_stdout = sys.stdout
    sys.stdout = log_handle

    try:
        print(f"Output redirected to: {log_file}")
        print(f"Running with arguments: {args}\n")
        plot_timestamp_suffix = None
        generated_at_label = None

        # Determine which variables to run
        if args.variable == "all":
            variables_to_run = list(VARIABLE_CONFIGS.keys())
        else:
            variables_to_run = [args.variable]

        # Determine which months to run
        if args.month == "all":
            months_to_run = list(MODEL_PREDICTIONS.keys())
        else:
            months_to_run = [args.month]

        # Determine what to run
        run_gpt = args.model in ("gpt", "all")
        run_claude = args.model in ("claude", "all")
        run_qwen = args.model in ("qwen", "all")
        run_agent = args.model in ("agent", "all")
        run_fed = args.model in ("fed", "all")
        run_hourly = args.frequency in ("hourly", "all")
        run_daily = args.frequency in ("daily", "all")

        # Determine aggregation methods to run
        if args.aggregation == "all":
            aggregation_methods = ["latest", "daily_avg", "12hour_avg"]
        else:
            aggregation_methods = [args.aggregation]

        prediction_ranges_df = None
        if run_qwen or run_agent:
            range_log_path = base_dir / PREDICTION_RANGE_LOG
            if not range_log_path.exists():
                raise FileNotFoundError(f"Prediction range log not found: {range_log_path}")
            prediction_ranges_df = load_prediction_range_log(range_log_path)
            print(f"Loaded prediction range log: {range_log_path}")

        all_results = []

        # Iterate over variables
        for variable in variables_to_run:
            var_config = VARIABLE_CONFIGS[variable]
            var_name = var_config.get("name", variable)

            # Create variable-specific output directory
            var_output_dir = output_base_dir / variable
            var_output_dir.mkdir(parents=True, exist_ok=True)
            summary_path = var_output_dir / args.summary_file

            print(f"\n{'=' * 80}")
            print(f"PROCESSING VARIABLE: {var_name} ({variable})")
            print(f"{'=' * 80}")

            var_results = []

            for month in months_to_run:
                # Check if this variable has data for this month
                if month not in var_config["months"]:
                    print(f"  Skipping {month}: no data for {variable}")
                    continue

                month_var_config = var_config["months"][month]
                month_name = MODEL_PREDICTIONS[month]["name"]

                # Get buckets (may be at variable level or month level for unemployment)
                buckets = month_var_config.get("buckets", var_config.get("buckets"))
                polymarket_path = base_dir / month_var_config["polymarket_data"]

                print(f"\n{'=' * 70}")
                print(f"Processing {month_name} - {var_name}")
                print(f"{'=' * 70}")

                if not polymarket_path.exists():
                    raise FileNotFoundError(f"Polymarket data file not found: {polymarket_path}")

                # Look up release datetime + released value from the scraped truth.
                # Derive the winning bucket from the value and verify it matches the
                # bucket Polymarket converged on (>=0.99 prob).
                official_release_utc, gt_value = lookup_ground_truth(variable, month)
                if official_release_utc is None or gt_value is None:
                    print(f"  Skipping {variable} {month}: ground truth not yet available "
                          f"in field_releases_live.csv / investing_us_calendar_latest.csv")
                    continue

                polymarket_for_check = load_polymarket_data(polymarket_path)
                winning_bucket = derive_winning_bucket(
                    variable, gt_value, buckets, polymarket_for_check
                )
                print(f"  Released value: {gt_value}")
                print(f"  Release datetime (UTC): {official_release_utc}")
                print(f"  Winning bucket (derived): {winning_bucket}")
                if run_qwen:
                    print(f"  Qwen eligibility cutoff (UTC): {official_release_utc}")
                fed_predictions_path, fed_nowcast_path = resolve_fed_data_paths(
                    variable=variable,
                    month=month,
                    month_var_config=month_var_config,
                    base_dir=base_dir,
                )
                gdp_fed_sources: list[tuple[str, Path]] = []
                if variable == "real_gdp_qoq":
                    gdp_fed_sources = discover_fed_gdp_files(base_dir, target_month=month)
                    if gdp_fed_sources:
                        discovered_names = ", ".join(
                            [f"{name}={path.name}" for name, path in gdp_fed_sources]
                        )
                        print(f"  Auto-discovered GDP fed sources: {discovered_names}")

                # GPT model. predictions_path may be a list (for quarterly markets
                # like Q1 2026 GDP that combine multiple monthly target_month files).
                if run_gpt:
                    gpt_paths = resolve_prediction_spec(
                        MODEL_PREDICTIONS[month]["gpt_predictions"], base_dir
                    )
                    if not gpt_paths:
                        raise ValueError(f"GPT predictions not configured for {month}")
                    for p in gpt_paths:
                        if not p.exists():
                            raise FileNotFoundError(f"GPT predictions file not found: {p}")
                    gpt_path = gpt_paths if len(gpt_paths) > 1 else gpt_paths[0]

                    if run_hourly:
                        results = run_calculation(
                            polymarket_path=polymarket_path,
                            predictions_path=gpt_path,
                            model_name="gpt-5-search-api",
                            prediction_source="gpt",
                            frequency="hourly",
                            aggregation_method="latest",
                            bet_amount=args.bet_amount,
                            tz_offset=args.tz_offset,
                            output_dir=var_output_dir,
                            summary_path=summary_path,
                            month=month,
                            winning_bucket=winning_bucket,
                            variable=variable,
                            buckets=buckets,
                            official_release_utc=official_release_utc,
                            plot_timestamp_suffix=plot_timestamp_suffix,
                            generated_at_label=generated_at_label,
                        )
                        var_results.append(results)

                    if run_daily:
                        for agg_method in aggregation_methods:
                            results = run_calculation(
                                polymarket_path=polymarket_path,
                                predictions_path=gpt_path,
                                model_name="gpt-5-search-api",
                                prediction_source="gpt",
                                frequency="daily",
                                aggregation_method=agg_method,
                                bet_amount=args.bet_amount,
                                tz_offset=args.tz_offset,
                                output_dir=var_output_dir,
                                summary_path=summary_path,
                                month=month,
                                winning_bucket=winning_bucket,
                                variable=variable,
                                buckets=buckets,
                                official_release_utc=official_release_utc,
                                plot_timestamp_suffix=plot_timestamp_suffix,
                                generated_at_label=generated_at_label,
                            )
                            var_results.append(results)

                # Claude model. Same list-handling pattern as GPT.
                claude_predictions = MODEL_PREDICTIONS[month].get("claude_predictions")
                if run_claude and claude_predictions:
                    claude_paths = resolve_prediction_spec(claude_predictions, base_dir)
                    for p in claude_paths:
                        if not p.exists():
                            raise FileNotFoundError(f"Claude predictions file not found: {p}")
                    claude_path = claude_paths if len(claude_paths) > 1 else claude_paths[0]

                    if run_hourly:
                        results = run_calculation(
                            polymarket_path=polymarket_path,
                            predictions_path=claude_path,
                            model_name="claude-sonnet-4.5",
                            prediction_source="claude",
                            frequency="hourly",
                            aggregation_method="latest",
                            bet_amount=args.bet_amount,
                            tz_offset=args.tz_offset,
                            output_dir=var_output_dir,
                            summary_path=summary_path,
                            month=month,
                            winning_bucket=winning_bucket,
                            variable=variable,
                            buckets=buckets,
                            official_release_utc=official_release_utc,
                            plot_timestamp_suffix=plot_timestamp_suffix,
                            generated_at_label=generated_at_label,
                        )
                        var_results.append(results)

                    if run_daily:
                        for agg_method in aggregation_methods:
                            results = run_calculation(
                                polymarket_path=polymarket_path,
                                predictions_path=claude_path,
                                model_name="claude-sonnet-4.5",
                                prediction_source="claude",
                                frequency="daily",
                                aggregation_method=agg_method,
                                bet_amount=args.bet_amount,
                                tz_offset=args.tz_offset,
                                output_dir=var_output_dir,
                                summary_path=summary_path,
                                month=month,
                                winning_bucket=winning_bucket,
                                variable=variable,
                                buckets=buckets,
                                official_release_utc=official_release_utc,
                                plot_timestamp_suffix=plot_timestamp_suffix,
                                generated_at_label=generated_at_label,
                            )
                            var_results.append(results)
                elif run_claude and not claude_predictions:
                    print(f"  Claude skipped for {month}: no predictions file configured.")

                # Qwen models
                qwen_prediction_specs = [
                    (
                        "qwen3-next-80b-a3b-instruct",
                        MODEL_PREDICTIONS[month].get("qwen_next_predictions"),
                    ),
                    (
                        "qwen3-235b-a22b-instruct-2507",
                        MODEL_PREDICTIONS[month].get("qwen_235_predictions"),
                    ),
                ]
                if run_qwen:
                    any_qwen_configured = False
                    for qwen_model_name, qwen_predictions in qwen_prediction_specs:
                        if not qwen_predictions:
                            continue
                        any_qwen_configured = True
                        qwen_paths = resolve_prediction_spec(qwen_predictions, base_dir)
                        for p in qwen_paths:
                            if not p.exists():
                                raise FileNotFoundError(
                                    f"Qwen predictions file not found for {qwen_model_name}: {p}"
                                )
                        qwen_path = qwen_paths if len(qwen_paths) > 1 else qwen_paths[0]

                        qwen_target_month = infer_target_month_from_prediction_path(qwen_path)
                        usable_qwen, usable_reason = model_has_usable_prediction_window(
                            prediction_ranges_df=prediction_ranges_df,
                            model_name=qwen_model_name,
                            target_month=qwen_target_month,
                            variable=variable,
                            official_release_utc=official_release_utc,
                            local_tz_offset_hours=args.tz_offset,
                        )
                        if not usable_qwen:
                            print(
                                f"  Skipping {qwen_model_name} for {month} ({variable}): "
                                f"{usable_reason}"
                            )
                            if run_hourly:
                                remove_existing_result_artifacts(
                                    output_dir=var_output_dir,
                                    summary_path=summary_path,
                                    model_name=qwen_model_name,
                                    month=month,
                                    frequency="hourly",
                                    aggregation_method="latest",
                                )
                            if run_daily:
                                for agg_method in aggregation_methods:
                                    remove_existing_result_artifacts(
                                        output_dir=var_output_dir,
                                        summary_path=summary_path,
                                        model_name=qwen_model_name,
                                        month=month,
                                        frequency="daily",
                                        aggregation_method=agg_method,
                                    )
                            continue

                        print(f"  Using {qwen_model_name}: {usable_reason}")

                        if run_hourly:
                            results = run_calculation(
                                polymarket_path=polymarket_path,
                                predictions_path=qwen_path,
                                model_name=qwen_model_name,
                                prediction_source="qwen",
                                frequency="hourly",
                                aggregation_method="latest",
                                bet_amount=args.bet_amount,
                                tz_offset=args.tz_offset,
                                output_dir=var_output_dir,
                                summary_path=summary_path,
                                month=month,
                                winning_bucket=winning_bucket,
                                variable=variable,
                                buckets=buckets,
                                official_release_utc=official_release_utc,
                                plot_timestamp_suffix=plot_timestamp_suffix,
                                generated_at_label=generated_at_label,
                            )
                            var_results.append(results)

                        if run_daily:
                            for agg_method in aggregation_methods:
                                results = run_calculation(
                                    polymarket_path=polymarket_path,
                                    predictions_path=qwen_path,
                                    model_name=qwen_model_name,
                                    prediction_source="qwen",
                                    frequency="daily",
                                    aggregation_method=agg_method,
                                    bet_amount=args.bet_amount,
                                    tz_offset=args.tz_offset,
                                    output_dir=var_output_dir,
                                    summary_path=summary_path,
                                    month=month,
                                    winning_bucket=winning_bucket,
                                    variable=variable,
                                    buckets=buckets,
                                    official_release_utc=official_release_utc,
                                    plot_timestamp_suffix=plot_timestamp_suffix,
                                    generated_at_label=generated_at_label,
                                )
                                var_results.append(results)

                    if not any_qwen_configured:
                        print(f"  Qwen skipped for {month}: no predictions file configured.")

                # Claude Code Agent (sparse model, gated by prediction-window check)
                if run_agent:
                    agent_predictions = MODEL_PREDICTIONS[month].get("claude_code_agent_predictions")
                    if not agent_predictions:
                        print(f"  Agent skipped for {month}: no predictions file configured.")
                    else:
                        agent_paths = resolve_prediction_spec(agent_predictions, base_dir)
                        for p in agent_paths:
                            if not p.exists():
                                raise FileNotFoundError(
                                    f"Claude-code-agent predictions file not found: {p}"
                                )
                        agent_path = agent_paths if len(agent_paths) > 1 else agent_paths[0]
                        agent_target_month = infer_target_month_from_prediction_path(agent_path)
                        usable_agent, usable_reason = model_has_usable_prediction_window(
                            prediction_ranges_df=prediction_ranges_df,
                            model_name="claude-code-agent",
                            target_month=agent_target_month,
                            variable=variable,
                            official_release_utc=official_release_utc,
                            local_tz_offset_hours=args.tz_offset,
                        )
                        if not usable_agent:
                            print(
                                f"  Skipping claude-code-agent for {month} ({variable}): {usable_reason}"
                            )
                            if run_hourly:
                                remove_existing_result_artifacts(
                                    output_dir=var_output_dir,
                                    summary_path=summary_path,
                                    model_name="claude-code-agent",
                                    month=month,
                                    frequency="hourly",
                                    aggregation_method="latest",
                                )
                            if run_daily:
                                for agg_method in aggregation_methods:
                                    remove_existing_result_artifacts(
                                        output_dir=var_output_dir,
                                        summary_path=summary_path,
                                        model_name="claude-code-agent",
                                        month=month,
                                        frequency="daily",
                                        aggregation_method=agg_method,
                                    )
                        else:
                            print(f"  Using claude-code-agent: {usable_reason}")
                            if run_hourly:
                                results = run_calculation(
                                    polymarket_path=polymarket_path,
                                    predictions_path=agent_path,
                                    model_name="claude-code-agent",
                                    prediction_source="claude",
                                    frequency="hourly",
                                    aggregation_method="latest",
                                    bet_amount=args.bet_amount,
                                    tz_offset=args.tz_offset,
                                    output_dir=var_output_dir,
                                    summary_path=summary_path,
                                    month=month,
                                    winning_bucket=winning_bucket,
                                    variable=variable,
                                    buckets=buckets,
                                    official_release_utc=official_release_utc,
                                    plot_timestamp_suffix=plot_timestamp_suffix,
                                    generated_at_label=generated_at_label,
                                )
                                var_results.append(results)
                            if run_daily:
                                for agg_method in aggregation_methods:
                                    results = run_calculation(
                                        polymarket_path=polymarket_path,
                                        predictions_path=agent_path,
                                        model_name="claude-code-agent",
                                        prediction_source="claude",
                                        frequency="daily",
                                        aggregation_method=agg_method,
                                        bet_amount=args.bet_amount,
                                        tz_offset=args.tz_offset,
                                        output_dir=var_output_dir,
                                        summary_path=summary_path,
                                        month=month,
                                        winning_bucket=winning_bucket,
                                        variable=variable,
                                        buckets=buckets,
                                        official_release_utc=official_release_utc,
                                        plot_timestamp_suffix=plot_timestamp_suffix,
                                        generated_at_label=generated_at_label,
                                    )
                                    var_results.append(results)

                # Fed model forecasts (CPI + GDP variants)
                fed_prediction_sources_map: dict[str, Path] = {}
                if fed_predictions_path is not None:
                    fed_prediction_sources_map["fed-forecast"] = fed_predictions_path
                for model_name, model_path in gdp_fed_sources:
                    fed_prediction_sources_map[model_name] = model_path

                if run_fed and fed_prediction_sources_map:
                    polymarket_df = load_polymarket_data(polymarket_path)

                    for fed_model_name, fed_path in fed_prediction_sources_map.items():
                        print(f"\n{'#' * 70}")
                        print(f"# Running: {fed_model_name} [{month}] - {var_name}")
                        print(f"{'#' * 70}")

                        if variable == "cpi_yoy":
                            fed_predictions_df = load_fed_predictions(fed_path, column="CPI Inflation")
                        elif variable == "real_gdp_qoq":
                            fed_predictions_df = load_fed_gdp_predictions(
                                fed_path,
                                target_month=month,
                                model_name=fed_model_name,
                            )
                        else:
                            print(f"  Skipping {fed_model_name}: unsupported variable for fed forecast mode ({variable})")
                            continue

                        if fed_predictions_df.empty:
                            print(f"  Skipping {fed_model_name}: no usable predictions parsed from {fed_path}")
                            continue

                        # Run betting modes based on requested frequency.
                        betting_modes = []
                        if run_hourly:
                            betting_modes.append("hourly")
                        if run_daily:
                            betting_modes.append("daily")

                        for betting_mode in betting_modes:
                            results = calculate_fed_cpi_earnings(
                                polymarket_df=polymarket_df,
                                fed_predictions_df=fed_predictions_df,
                                bet_amount=args.bet_amount,
                                betting_mode=betting_mode,
                                model_name=fed_model_name,
                                winning_bucket=winning_bucket,
                                month=month,
                                variable=variable,
                                buckets=buckets,
                                official_release_utc=official_release_utc,
                            )

                            # Create subfolder based on results
                            freq = results["betting_frequency"]
                            agg = results["aggregation_method"]
                            subfolder = get_output_subfolder(freq, agg)
                            subfolder_dir = var_output_dir / subfolder
                            subfolder_dir.mkdir(parents=True, exist_ok=True)

                            # Print results
                            print_results(results)
                            print_winning_slots(results, max_slots=50)

                            # Save detailed results
                            detail_filename = f"betting_results_{fed_model_name}_{month}_{betting_mode}.csv"
                            save_detailed_results(results, subfolder_dir / detail_filename)

                            # Save summary
                            save_earnings_summary(results, summary_path)

                            # Generate and save returns plot
                            plot_filename = f"returns_plot_{fed_model_name}_{month}_{betting_mode}.png"
                            plot_cumulative_returns_combined(
                                [results],
                                subfolder_dir / plot_filename,
                            )

                            var_results.append(results)

                # Fed nowcast model (for unemployment_rate)
                if run_fed and fed_nowcast_path is not None:

                    print(f"\n{'#' * 70}")
                    print(f"# Running: fed-nowcast [{month}] - {var_name}")
                    print(f"{'#' * 70}")

                    # Load data
                    polymarket_df = load_polymarket_data(polymarket_path)
                    nowcast_df = load_fed_nowcast(fed_nowcast_path)

                    # Run betting modes based on frequency settings.
                    betting_modes = []
                    if run_hourly:
                        betting_modes.append("hourly")
                    if run_daily:
                        betting_modes.append("daily")

                    for betting_mode in betting_modes:
                        results = calculate_fed_nowcast_earnings(
                            polymarket_df=polymarket_df,
                            nowcast_df=nowcast_df,
                            bet_amount=args.bet_amount,
                            betting_mode=betting_mode,
                            model_name="fed-nowcast",
                            winning_bucket=winning_bucket,
                            month=month,
                            variable=variable,
                            buckets=buckets,
                            official_release_utc=official_release_utc,
                        )

                        # Create subfolder based on results
                        freq = results["betting_frequency"]
                        agg = results["aggregation_method"]
                        subfolder = get_output_subfolder(freq, agg)
                        subfolder_dir = var_output_dir / subfolder
                        subfolder_dir.mkdir(parents=True, exist_ok=True)

                        # Print results
                        print_results(results)
                        print_winning_slots(results, max_slots=50)

                        # Save detailed results
                        detail_filename = f"betting_results_fed-nowcast_{month}_{betting_mode}.csv"
                        save_detailed_results(results, subfolder_dir / detail_filename)

                        # Save summary
                        save_earnings_summary(results, summary_path)

                        # Generate and save returns plot
                        plot_filename = f"returns_plot_fed-nowcast_{month}_{betting_mode}.png"
                        plot_cumulative_returns_combined(
                            [results],
                            subfolder_dir / plot_filename,
                        )

                        var_results.append(results)

                if run_fed and fed_predictions_path is None and fed_nowcast_path is None and not gdp_fed_sources:
                    print(f"  Fed skipped for {month} ({variable}): no matching fed file found in polymarket_return/fed.")

            # Print variable summary and generate plots
            if len(var_results) > 1:
                print(f"\n{'=' * 100}")
                print(f"SUMMARY FOR {var_name.upper()}")
                print(f"{'=' * 100}")
                print(f"\n{'Model':<20} {'Month':<10} {'Frequency':<10} {'Aggregation':<12} {'Invested':>12} {'Profit':>12} {'Return':>10}")
                print("-" * 100)
                for r in var_results:
                    agg_method = r.get('aggregation_method', 'latest')
                    month = r.get('month', '')
                    print(f"{r['model_name']:<20} {month:<10} {r['betting_frequency']:<10} {agg_method:<12} ${r['total_invested']:>10.2f} ${r['profit']:>10.2f} {r['return_pct']:>9.1f}%")

                # Generate comparison bar plots - one per month for this variable
                for month_key in months_to_run:
                    if month_key in var_config["months"]:
                        plot_final_returns_comparison(
                            var_results,
                            var_output_dir / f"final_returns_comparison_{month_key}.png",
                            target_month=month_key,
                        )

                # Generate combined cumulative returns plots
                # Group results by month, frequency, and aggregation method
                from collections import defaultdict
                grouped_results = defaultdict(list)
                for r in var_results:
                    month = r.get('month', '')
                    freq = r.get('betting_frequency', '')
                    agg = r.get('aggregation_method', 'latest')
                    key = (month, freq, agg)
                    grouped_results[key].append(r)

                # Generate a combined plot for each group
                for (month, freq, agg), results_group in grouped_results.items():
                    if len(results_group) > 0:
                        subfolder = get_output_subfolder(freq, agg)
                        subfolder_dir = var_output_dir / subfolder
                        subfolder_dir.mkdir(parents=True, exist_ok=True)
                        plot_filename = f"returns_plot_combined_{month}_{freq}_{agg}.png"
                        plot_cumulative_returns_combined(
                            results_group,
                            subfolder_dir / plot_filename,
                        )

            all_results.extend(var_results)

        print(f"\n✅ All results saved to: {output_base_dir}")
        for var in variables_to_run:
            print(f"   {var}: {output_base_dir / var}")

        return 0

    finally:
        # Restore stdout and close log file
        sys.stdout = original_stdout
        log_handle.close()
        print(f"✅ Output saved to: {log_file}")


if __name__ == "__main__":
    raise SystemExit(main())
