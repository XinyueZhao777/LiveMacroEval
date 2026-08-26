"""Step 15.2 Historical preprocessing (Project Outline sections 15.1 + 15.2).

For each historical field release this script performs, per the outline:

  1. identify the scored field i using the mapping table,
  2. read A_it, C_it, T_it from the economic calendar,
  3. compute the raw surprise A_it - C_it,
  4. estimate sigma_i from the historical sample for field i
     (sample std of raw surprises; one scalar per field, held fixed),
  5. compute the standardized surprise S_it = (A_it - C_it) / sigma_i,
  6. assign the release to a timestamp-group event g (one g per unique T),
  7. compute r_g^HF from the precomputed ES 1-min event-window closes
     (close at minute containing T-5m, close at minute containing T+30m).

Inputs (not modified by this script):
  - Results/data_consensus/filtered_macro_events.csv
  - Results/data_sp500futures/event_window_coverage.csv

Outputs (written alongside this script):
  - field_mapping.csv              section 15.1 mapping table
  - field_releases.csv             per valid release: A, C, raw, sigma, S, cohort
  - sigma_by_field.csv             sigma_i with sample size and date range
  - timestamp_group_events.csv     one row per g (section 15.3 regression input):
                                   timestamp, cohort, hf_return, X_{i,g} for all fields
  - preprocessing_report.txt       QC summary

Historical sample default (long-history baseline):
  --hist-start 2010-01-01
  --hist-end   2025-10-31           live model forecasts begin 2025-11
  --events     filtered_macro_events_combined.csv
  --coverage   event_window_coverage_with_2010_2014.csv

Pass --hist-start 2015-01-01 --events filtered_macro_events.csv
  --coverage event_window_coverage.csv to use the legacy 2015-only inputs.

Conventions (per outline):
  - All timestamps are naive US/Eastern minute clock (matches ES bars and the
    calendar's release_datetime_et).
  - A field release is "valid" (section 5.1) only if A, C, and r_g are all
    observed. sigma_i is estimated on the weaker set of releases where A and C
    are observed (section 6 requires only A-C).
  - Release-time cohort is HH:MM ET.
  - Timestamp-group events with zero valid field releases are dropped
    (section 15.5).
  - No fallback logic is used silently; any deviation is flagged in the report.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent  # Results/market_surprise_capture_score
RESULTS = ROOT.parent  # Results
# Long-history (2010+) inputs are the canonical default. The 2010-2014
# extension is built by data_consensus/filter_macro_events_2010_2014.py and
# data_sp500futures/extract_event_windows_2010_2014.py; the *_combined.csv /
# event_window_coverage_with_2010_2014.csv files are the merged tables.
# Pass --events / --coverage explicitly to fall back to the 2015-only inputs.
DEFAULT_EVENTS = RESULTS / "data_consensus" / "filtered_macro_events_combined.csv"
DEFAULT_COVERAGE = RESULTS / "data_sp500futures" / "event_window_coverage_with_2010_2014.csv"

# ---------------------------------------------------------------------------
# Section 15.1 mapping table.
# 16 fields (real_disposable_personal_income intentionally dropped — not
# published on the Investing.com US calendar).
# "event_bases" is a priority list: at each release timestamp, the first
# matching pattern wins. This dedupes PCE Deflator (pre-2019-08) vs PCE price
# index (post-2019-08) on any overlapping timestamp.
# ---------------------------------------------------------------------------
MAPPING: list[dict] = [
    {
        "field_id": "real_gdp_advance",
        "package": "gdp",
        "event_bases": ["GDP (QoQ)"],
        "units": "% QoQ SAAR",
        "notes": "Keep only ADVANCE release (earliest of 3 same-Q# releases per reference quarter).",
    },
    {
        "field_id": "industrial_production",
        "package": "industrial_production",
        "event_bases": ["Industrial Production (MoM)"],
        "units": "% MoM",
        "notes": "",
    },
    {
        "field_id": "durable_goods",
        "package": "durable_goods",
        "event_bases": ["Durable Goods Orders (MoM)"],
        "units": "% MoM",
        "notes": "",
    },
    {
        "field_id": "ism_manufacturing",
        "package": "ism_manufacturing",
        "event_bases": ["ISM Manufacturing PMI"],
        "units": "index level",
        "notes": "",
    },
    {
        "field_id": "cpi",
        "package": "cpi",
        "event_bases": ["CPI (YoY)"],
        "units": "% YoY",
        "notes": "Headline YoY (using the YoY series rather than MoM SA).",
    },
    {
        "field_id": "ppi",
        "package": "ppi",
        "event_bases": ["PPI (MoM)"],
        "units": "% MoM",
        "notes": "",
    },
    {
        "field_id": "pce_price_index",
        "package": "personal_income_outlays",
        "event_bases": ["PCE price index (MoM)", "PCE Deflator (MoM)"],
        "units": "% MoM",
        "notes": "Concatenate PCE Deflator (MoM) pre-2019-08 and PCE price index (MoM) post-2019-08 (same series renamed; prefer PCE price index on overlap).",
    },
    {
        "field_id": "real_pce",
        "package": "personal_income_outlays",
        "event_bases": ["Real Personal Consumption (MoM)"],
        "units": "% MoM",
        "notes": "",
    },
    {
        "field_id": "retail_sales",
        "package": "retail_sales",
        "event_bases": ["Retail Sales (MoM)"],
        "units": "% MoM",
        "notes": "",
    },
    {
        "field_id": "nonfarm_payrolls",
        "package": "employment_situation",
        "event_bases": ["Nonfarm Payrolls"],
        "units": "K level change",
        "notes": "",
    },
    {
        "field_id": "unemployment_rate",
        "package": "employment_situation",
        "event_bases": ["Unemployment Rate"],
        "units": "% level",
        "notes": "",
    },
    {
        "field_id": "housing_starts",
        "package": "new_residential_construction",
        "event_bases": ["Housing Starts"],
        "units": "M SAAR level",
        "notes": "",
    },
    {
        "field_id": "building_permits",
        "package": "new_residential_construction",
        "event_bases": ["Building Permits"],
        "units": "M SAAR level",
        "notes": "All releases with valid A+C+r kept; outline does not restrict to 08:30. From 2023 onward the revised release is a separate timestamp-group event.",
    },
    {
        "field_id": "existing_home_sales",
        "package": "existing_home_sales",
        "event_bases": ["Existing Home Sales"],
        "units": "M SAAR level",
        "notes": "",
    },
    {
        "field_id": "new_home_sales",
        "package": "new_home_sales",
        "event_bases": ["New Home Sales"],
        "units": "K SAAR level",
        "notes": "",
    },
    {
        "field_id": "ism_services",
        "package": "ism_services",
        "event_bases": ["ISM Non-Manufacturing PMI"],
        "units": "index level",
        "notes": "Investing.com uses the legacy 'Non-Manufacturing' label for ISM Services.",
    },
]

FIELD_IDS: list[str] = [m["field_id"] for m in MAPPING]


# ---------------------------------------------------------------------------
# Mapping CSV writer (section 15.1 artifact).
# ---------------------------------------------------------------------------
def write_mapping_csv(out_dir: Path) -> Path:
    rows = []
    for m in MAPPING:
        rows.append(
            {
                "field_id": m["field_id"],
                "package_name": m["package"],
                "calendar_event_name": m["event_bases"][0],
                "calendar_field_name_fallbacks": " | ".join(m["event_bases"][1:]),
                "calendar_units": m["units"],
                "model_output_source": "",            # filled at section 15.4
                "model_transform_rule": "",           # filled at section 15.4
                "active_flag": True,
                "notes": m["notes"],
            }
        )
    p = out_dir / "field_mapping.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# First-release-only filter (applies to every field).
#
# The Investing.com calendar embeds the reference period in the `event` string.
# Rule: keep the EARLIEST release per (field, reference_period). That drops:
#   - GDP's 2nd and 3rd revisions (same '(Q#)' suffix),
#   - Building Permits' 5-10-day-later revised release (same '(Mon)' suffix
#     or no suffix but same implied reference month),
#   - Durable Goods' final-release revisions ~10 days after the advance,
#   - any other revision-style duplicate.
#
# Reference-period parsing priority on the `event` suffix:
#   1. '(Q[1-4])'                                  quarterly  -> 'YYYY-Q#'
#   2. '([A-Z][a-z]{2}/[A-Z][a-z]{2})'             dual-month -> 'YYYY-MM/MM'
#   3. '([A-Z][a-z]{2,3})'                         single mon -> 'YYYY-MM'
# Reference-year is inferred by "reference month cannot be after release
# month", with year roll-over.
#
# If the suffix is absent the row is a revision whose Investing.com tag was
# dropped. We infer its ref_period with a dual rule:
#   4a. Copy ref_period from the most-recent preceding same-field row with a
#       parseable suffix, if it is within LOOKBACK_DAYS. This handles:
#         - Building Permits revised release ~1 week after initial,
#         - Durable Goods final release ~1-2 weeks after advance.
#   4b. Otherwise fall back to 'YYYY-MM' with ref month = release_month - 1.
#       This handles legitimate initial releases that happen to be missing
#       the Investing.com suffix tag (e.g., Existing Home Sales 2023-03-21).
# ---------------------------------------------------------------------------
_Q_PAT = re.compile(r"\(Q([1-4])\)\s*$")
_MM_PAT = re.compile(r"\(([A-Z][a-z]{2})/([A-Z][a-z]{2})\)\s*$")
_M_PAT = re.compile(r"\(([A-Z][a-z]{2,3})\)\s*$")

_MONTH_ABBR = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

LOOKBACK_DAYS = pd.Timedelta(days=14)


def _parse_ref_from_suffix(event: str, release_dt: pd.Timestamp) -> str | None:
    """Return canonical ref_period key if event has a parseable suffix."""
    s = str(event).strip()

    m = _Q_PAT.search(s)
    if m:
        q = int(m.group(1))
        ref_end_month = 3 * q
        ry = release_dt.year - (1 if ref_end_month > release_dt.month else 0)
        return f"{ry}-Q{q}"

    m = _MM_PAT.search(s)
    if m:
        a, b = m.group(1), m.group(2)
        mo_a = _MONTH_ABBR.get(a)
        mo_b = _MONTH_ABBR.get(b)
        if mo_a and mo_b:
            later = mo_b if mo_b >= mo_a else mo_a
            ry = release_dt.year - (1 if later > release_dt.month else 0)
            return f"{ry}-{mo_a:02d}/{mo_b:02d}"

    m = _M_PAT.search(s)
    if m:
        a = m.group(1)
        mo = _MONTH_ABBR.get(a)
        if mo:
            ry = release_dt.year - (1 if mo > release_dt.month else 0)
            return f"{ry}-{mo:02d}"

    return None


def _compute_ref_periods(sub: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (ref_periods, ref_sources) for a sorted-by-time subset of one
    field's rows.

    ref_source in {'suffix', 'inherited', 'fallback'} for auditing.
    """
    ref_periods: list[str] = [""] * len(sub)
    sources: list[str] = [""] * len(sub)
    # Direct parse pass.
    parsed: list[str | None] = [
        _parse_ref_from_suffix(e, t)
        for e, t in zip(sub["event"].tolist(), sub["release_datetime_et"].tolist())
    ]
    # Fill unparsed via dual rule.
    last_parsed_i = -1
    ts = sub["release_datetime_et"].tolist()
    for i in range(len(sub)):
        if parsed[i] is not None:
            ref_periods[i] = parsed[i]
            sources[i] = "suffix"
            last_parsed_i = i
            continue
        # Unparsed row: try inherit from nearest preceding parsed row within
        # LOOKBACK_DAYS.
        if last_parsed_i >= 0 and ts[i] - ts[last_parsed_i] <= LOOKBACK_DAYS:
            ref_periods[i] = parsed[last_parsed_i]
            sources[i] = "inherited"
        else:
            # Fallback: release_month - 1 as ref month.
            t = ts[i]
            mo = t.month - 1
            ry = t.year if mo >= 1 else t.year - 1
            if mo < 1:
                mo = 12
            ref_periods[i] = f"{ry}-{mo:02d}"
            sources[i] = "fallback"
    return ref_periods, sources


def keep_first_release(sub: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (kept, dropped). kept = earliest row per ref_period."""
    d = sub.sort_values("release_datetime_et", kind="mergesort").copy()
    ref_periods, sources = _compute_ref_periods(d)
    d["_ref_period"] = ref_periods
    d["_ref_source"] = sources
    keep_mask = ~d.duplicated(subset=["_ref_period"], keep="first")
    kept = d[keep_mask].drop(columns=["_ref_period", "_ref_source"])
    dropped = d[~keep_mask].copy()
    return kept, dropped


# ---------------------------------------------------------------------------
# Build per-field release table (steps 1-3).
# ---------------------------------------------------------------------------
def build_field_releases(
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (field_releases, dropped_revisions).

    field_releases: long-format (field_id, release_datetime_et, A, C,
    raw_surprise, ...) — one row per (field_id, reference_period) keeping the
    earliest release. Only rows with A and C both observed are kept (section
    5.1 items 1-2; hf_return validity is joined later).

    dropped_revisions: rows dropped by the first-release-per-reference-period
    filter, for auditing.
    """
    ev = events.copy()
    ev["release_datetime_et"] = pd.to_datetime(ev["release_datetime_et"])
    kept_parts: list[pd.DataFrame] = []
    drop_parts: list[pd.DataFrame] = []
    for m in MAPPING:
        field_id = m["field_id"]
        priority = m["event_bases"]
        sub = ev[ev["event_base"].isin(priority)].copy()
        if sub.empty:
            continue
        # Priority rank collapses same-timestamp duplicates across fallback
        # names (e.g., PCE price index vs PCE Deflator on 2019-08-30 both at
        # 08:30) before the first-release-per-reference filter.
        rank_map = {name: i for i, name in enumerate(priority)}
        sub["_rank"] = sub["event_base"].map(rank_map)
        sub.sort_values(
            ["release_datetime_et", "_rank"], kind="mergesort", inplace=True
        )
        sub = sub.drop_duplicates(
            subset=["release_datetime_et"], keep="first"
        ).drop(columns="_rank")

        # Universal first-release-per-reference-period filter.
        kept, dropped = keep_first_release(sub)
        # A+C filter on kept rows only.
        kept = kept[kept["actual"].notna() & kept["forecast"].notna()].copy()
        kept["field_id"] = field_id
        kept["A"] = kept["actual"].astype(float)
        kept["C"] = kept["forecast"].astype(float)
        kept["raw_surprise"] = kept["A"] - kept["C"]
        kept["event_base_matched"] = kept["event_base"]
        kept_parts.append(
            kept[
                [
                    "field_id",
                    "release_datetime_et",
                    "event",
                    "event_base_matched",
                    "A",
                    "C",
                    "raw_surprise",
                ]
            ]
        )
        if len(dropped):
            dropped = dropped.copy()
            dropped["field_id"] = field_id
            drop_parts.append(
                dropped[
                    [
                        "field_id",
                        "release_datetime_et",
                        "event",
                        "event_base",
                        "actual",
                        "forecast",
                        "_ref_period",
                    ]
                ].rename(columns={"_ref_period": "ref_period", "event_base": "event_base_matched"})
            )
    if not kept_parts:
        raise RuntimeError("No field releases built — mapping or input empty.")
    kept_out = pd.concat(kept_parts, ignore_index=True)
    drop_out = (
        pd.concat(drop_parts, ignore_index=True)
        if drop_parts
        else pd.DataFrame(
            columns=[
                "field_id",
                "release_datetime_et",
                "event",
                "event_base_matched",
                "actual",
                "forecast",
                "ref_period",
            ]
        )
    )
    return kept_out, drop_out


# ---------------------------------------------------------------------------
# Step 4: sigma_i (section 6).
# ---------------------------------------------------------------------------
def compute_sigma_i(
    fr: pd.DataFrame, hist_start: pd.Timestamp, hist_end_inclusive: pd.Timestamp
) -> pd.DataFrame:
    """Sample std of (A - C) per field on historical sample.

    Per outline section 6, sigma_i is estimated only on the historical sample.
    Per section 15.2 step 4, sigma_i is estimated once and held fixed.
    """
    hist = fr[
        (fr["release_datetime_et"] >= hist_start)
        & (fr["release_datetime_et"] <= hist_end_inclusive)
    ].copy()
    g = hist.groupby("field_id", sort=False)
    out = pd.DataFrame(
        {
            "n_hist_releases": g.size(),
            "mean_raw_surprise": g["raw_surprise"].mean(),
            "sigma_i": g["raw_surprise"].std(ddof=1),
            "min_raw_surprise": g["raw_surprise"].min(),
            "max_raw_surprise": g["raw_surprise"].max(),
            "first_release": g["release_datetime_et"].min(),
            "last_release": g["release_datetime_et"].max(),
        }
    ).reset_index()
    # Preserve mapping order.
    order = {fid: i for i, fid in enumerate(FIELD_IDS)}
    out["__ord"] = out["field_id"].map(order)
    out.sort_values("__ord", inplace=True)
    out.drop(columns="__ord", inplace=True)
    out.reset_index(drop=True, inplace=True)

    # Sanity checks.
    if (out["sigma_i"] <= 0).any() or out["sigma_i"].isna().any():
        bad = out[(out["sigma_i"] <= 0) | out["sigma_i"].isna()]
        raise ValueError(
            f"sigma_i non-positive or NaN for:\n{bad.to_string(index=False)}"
        )
    missing_fields = set(FIELD_IDS) - set(out["field_id"])
    if missing_fields:
        raise ValueError(
            f"sigma_i missing for fields (no historical releases): {missing_fields}"
        )
    return out


# ---------------------------------------------------------------------------
# Steps 5-7: standardized surprise + join event-window return + build timestamp
# -group event table (section 15.3 regression input).
# ---------------------------------------------------------------------------
def build_timestamp_group_events(
    fr: pd.DataFrame,
    sigma: pd.DataFrame,
    coverage: pd.DataFrame,
    hist_start: pd.Timestamp,
    hist_end_inclusive: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (field_releases_valid, timestamp_group_events).

    field_releases_valid: per valid release with standardized surprise.
    timestamp_group_events: one row per valid timestamp-group event (section
    15.3 regression input) with X_{i,g} for each scored field.
    """
    # Attach sigma_i.
    sig_map = sigma.set_index("field_id")["sigma_i"]
    fr = fr.copy()
    fr["sigma_i"] = fr["field_id"].map(sig_map)
    if fr["sigma_i"].isna().any():
        raise ValueError("Release rows without sigma_i — bug in sigma mapping.")
    fr["S"] = fr["raw_surprise"] / fr["sigma_i"]

    # Restrict to historical window (section 14.1).
    fr = fr[
        (fr["release_datetime_et"] >= hist_start)
        & (fr["release_datetime_et"] <= hist_end_inclusive)
    ].copy()

    # Join with event-window coverage (section 8, r_g^HF).
    cov = coverage.copy()
    cov["release_datetime_et"] = pd.to_datetime(cov["release_datetime_et"])
    cov = cov[
        [
            "release_datetime_et",
            "contract_code",
            "start_bar_close",
            "end_bar_close",
            "valid",
        ]
    ].rename(columns={"valid": "futures_valid"})

    fr = fr.merge(cov, on="release_datetime_et", how="left")
    has_cov = fr["futures_valid"].fillna(False)
    fr["hf_return"] = np.where(
        has_cov & fr["start_bar_close"].notna() & fr["end_bar_close"].notna(),
        np.log(fr["end_bar_close"].astype(float))
        - np.log(fr["start_bar_close"].astype(float)),
        np.nan,
    )
    fr["valid_release"] = has_cov & fr["hf_return"].notna()

    # Release-time cohort HH:MM (ET).
    fr["release_time_cohort"] = fr["release_datetime_et"].dt.strftime("%H:%M")

    valid = fr[fr["valid_release"]].copy()
    if valid.empty:
        raise RuntimeError("No valid historical releases after joining futures returns.")

    # Build timestamp-group event table.
    # One row per unique release_datetime_et with any valid release.
    grouped = valid.groupby("release_datetime_et", sort=True)

    # Per-g metadata (cohort, hf_return, contract) — must be identical across
    # all field-rows at the same timestamp; assert this.
    per_g = grouped.agg(
        release_time_cohort=("release_time_cohort", "first"),
        hf_return=("hf_return", "first"),
        contract_code=("contract_code", "first"),
        n_valid_fields=("field_id", "nunique"),
    ).reset_index()

    # Assert consistency: all field-rows at the same T share the same hf_return,
    # cohort, and contract.
    bad = grouped.agg(
        _cohort_n=("release_time_cohort", "nunique"),
        _ret_n=("hf_return", "nunique"),
        _ctr_n=("contract_code", "nunique"),
    )
    inconsistent = bad[(bad["_cohort_n"] > 1) | (bad["_ret_n"] > 1) | (bad["_ctr_n"] > 1)]
    if len(inconsistent):
        raise RuntimeError(
            f"Timestamp-group consistency violation for {len(inconsistent)} "
            f"timestamps. Example:\n{inconsistent.head().to_string()}"
        )

    # Pivot X_{i,g} wide. Missing field at a timestamp -> 0 (section 7).
    wide = (
        valid.pivot_table(
            index="release_datetime_et",
            columns="field_id",
            values="S",
            aggfunc="first",
        )
        .reindex(columns=FIELD_IDS)
        .fillna(0.0)
    )
    wide.columns = [f"X_{c}" for c in wide.columns]
    wide = wide.reset_index()

    events = per_g.merge(wide, on="release_datetime_et", how="left")
    events.insert(0, "timestamp_group_id", np.arange(1, len(events) + 1))
    events.rename(columns={"release_datetime_et": "release_timestamp"}, inplace=True)

    # Final column order.
    x_cols = [f"X_{f}" for f in FIELD_IDS]
    events = events[
        [
            "timestamp_group_id",
            "release_timestamp",
            "release_time_cohort",
            "contract_code",
            "hf_return",
            "n_valid_fields",
            *x_cols,
        ]
    ]

    # Tidy valid-release table output columns.
    valid_out = valid[
        [
            "field_id",
            "release_datetime_et",
            "release_time_cohort",
            "contract_code",
            "event",
            "event_base_matched",
            "A",
            "C",
            "raw_surprise",
            "sigma_i",
            "S",
            "hf_return",
            "start_bar_close",
            "end_bar_close",
        ]
    ].sort_values(
        ["release_datetime_et", "field_id"], kind="mergesort"
    ).reset_index(drop=True)

    return valid_out, events


# ---------------------------------------------------------------------------
# Report writer.
# ---------------------------------------------------------------------------
def build_report(
    events: pd.DataFrame,
    coverage: pd.DataFrame,
    valid: pd.DataFrame,
    field_releases: pd.DataFrame,
    sigma: pd.DataFrame,
    timestamp_group_events: pd.DataFrame,
    dropped_revisions: pd.DataFrame,
    hist_start: pd.Timestamp,
    hist_end_inclusive: pd.Timestamp,
) -> str:
    lines: list[str] = []
    L = lines.append
    L("=" * 78)
    L("Step 15.2 Historical preprocessing — report")
    L("=" * 78)
    L(f"Historical sample:   {hist_start.date()} .. {hist_end_inclusive.date()} (inclusive)")
    L(f"Mapping file:        field_mapping.csv ({len(MAPPING)} fields)")
    L(f"Calendar rows in:    {len(events):,}")
    L(f"Coverage rows in:    {len(coverage):,} "
      f"({int(coverage['valid'].sum()):,} valid futures windows)")
    L("")

    # Per-field coverage inside historical window (A, C both observed).
    fr_hist = field_releases[
        (field_releases["release_datetime_et"] >= hist_start)
        & (field_releases["release_datetime_et"] <= hist_end_inclusive)
    ]
    L("-- per-field historical coverage (A, C both observed; pre-sigma) --")
    per_field = (
        fr_hist.groupby("field_id")
        .agg(
            n_AC=("release_datetime_et", "size"),
            first=("release_datetime_et", "min"),
            last=("release_datetime_et", "max"),
        )
        .reindex(FIELD_IDS)
    )
    L(per_field.to_string())
    L("")

    # Sigma table.
    L("-- sigma_i (section 6) --")
    show = sigma[
        [
            "field_id",
            "n_hist_releases",
            "mean_raw_surprise",
            "sigma_i",
            "min_raw_surprise",
            "max_raw_surprise",
            "first_release",
            "last_release",
        ]
    ]
    L(show.to_string(index=False))
    L("")

    # Per-field VALID (A+C+r) counts vs A+C counts.
    L("-- per-field VALID release counts inside historical window (A+C+r) --")
    vpf = (
        valid.groupby("field_id")
        .size()
        .reindex(FIELD_IDS, fill_value=0)
        .to_frame("n_valid_ACr")
    )
    vpf["n_AC"] = per_field["n_AC"].reindex(FIELD_IDS).fillna(0).astype(int)
    vpf["dropped_missing_r"] = vpf["n_AC"] - vpf["n_valid_ACr"]
    L(vpf.to_string())
    L("")

    # Cohort distribution.
    L("-- timestamp-group event count by release-time cohort (HH:MM ET) --")
    coh = timestamp_group_events["release_time_cohort"].value_counts().sort_index()
    L(coh.to_string())
    L(f"total timestamp-group events: {len(timestamp_group_events):,}")
    L("")

    # Sanity: hf_return distribution.
    r = timestamp_group_events["hf_return"]
    L("-- hf_return distribution over all timestamp-group events --")
    L(
        f"  n={len(r):,}  mean={r.mean():+.6f}  std={r.std():.6f}  "
        f"min={r.min():+.6f}  max={r.max():+.6f}  |r|>0.02: {(r.abs() > 0.02).sum()}"
    )
    L("")

    # Simultaneous-field distribution.
    L("-- n_valid_fields per timestamp-group event (distribution) --")
    nv = timestamp_group_events["n_valid_fields"].value_counts().sort_index()
    L(nv.to_string())
    L("")

    # Per-timestamp field co-occurrence (top cohorts).
    L("-- field co-release matrix counts (how often field pairs co-release) --")
    x_cols = [f"X_{f}" for f in FIELD_IDS]
    present = (timestamp_group_events[x_cols] != 0).astype(int)
    co = present.T.dot(present)
    co.index = [c.replace("X_", "") for c in co.index]
    co.columns = co.index
    L(co.to_string())
    L("")

    # Invalid/dropped events inside historical window (for audit).
    drop_ts = set(
        coverage.loc[
            (pd.to_datetime(coverage["release_datetime_et"]) >= hist_start)
            & (pd.to_datetime(coverage["release_datetime_et"]) <= hist_end_inclusive)
            & (~coverage["valid"].astype(bool)),
            "release_datetime_et",
        ]
    )
    L(f"-- timestamps in coverage marked invalid inside historical window: {len(drop_ts)} --")
    if drop_ts:
        for t in sorted(drop_ts):
            L(f"  dropped: {t}")
    L("")

    L("Outline deviations / flagged choices:")
    L("  - real_disposable_personal_income (in outline section 2) has no "
      "Investing.com calendar series; intentionally dropped. Final "
      "scored field count = 16.")
    L("  - UNIVERSAL FIRST-RELEASE FILTER: earliest release per "
      "(field, reference period) is kept. Revisions are dropped.")
    L("  - PCE price index: concatenated PCE Deflator (MoM) (pre-2019-08) "
      "with PCE price index (MoM) (post-2019-08); Investing.com renamed the "
      "same series. On any overlapping timestamp, PCE price index wins.")
    L("")

    L("-- dropped-as-revision rows by field (universal first-release filter) --")
    if len(dropped_revisions):
        drp = dropped_revisions.copy()
        drp["release_datetime_et"] = pd.to_datetime(drp["release_datetime_et"])
        # Restrict to historical window for the summary.
        drp_in = drp[
            (drp["release_datetime_et"] >= hist_start)
            & (drp["release_datetime_et"] <= hist_end_inclusive)
        ]
        by_field = drp_in.groupby("field_id").size().reindex(FIELD_IDS, fill_value=0)
        L(by_field.to_frame("n_dropped").to_string())
        L(f"total dropped inside historical window: {int(by_field.sum())}")
        L(f"total dropped all-time: {len(drp)}")
        # Sample listing.
        L("\n  sample (first 12 historical-window drops):")
        show_cols = [
            "field_id",
            "release_datetime_et",
            "event",
            "ref_period",
            "actual",
            "forecast",
        ]
        L(drp_in.head(12)[show_cols].to_string(index=False))
    else:
        L("  none.")
    L("")

    L("Data-quality flags (NOT addressed by this step; reported for §15.3):")
    # NFP COVID-era outliers.
    nfp_valid = valid[valid["field_id"] == "nonfarm_payrolls"].copy()
    if len(nfp_valid):
        ext = nfp_valid.reindex(
            nfp_valid["raw_surprise"].abs().sort_values(ascending=False).index
        ).head(5)
        L("  * Nonfarm Payrolls: COVID-era releases are record-magnitude "
          "surprises that inflate sigma_i (max |raw_surprise| = "
          f"{ext['raw_surprise'].abs().max():,.0f}). Top 5 by |raw|:")
        for _, r in ext.iterrows():
            L(f"      {r.release_datetime_et}  A={r.A:,.0f}  C={r.C:,.0f}  "
              f"raw={r.raw_surprise:+,.0f}  S={r.S:+.2f}")
        L("    These are real observations (May-2020 rebound etc.); outline "
          "does not prescribe winsorization so none is applied. Consider "
          "robust inference or outlier diagnostics at the regression step.")
    # PCE low-coverage flag.
    low_cov = {
        "real_pce": "Real Personal Consumption (MoM) — many Investing.com rows have no consensus",
        "pce_price_index": "PCE price index (MoM) / PCE Deflator (MoM) — many Investing.com rows have no consensus",
    }
    for fid, msg in low_cov.items():
        n_ac = int(fr_hist[fr_hist["field_id"] == fid].shape[0])
        n_valid_row = int(valid[valid["field_id"] == fid].shape[0])
        if n_ac > 0 and n_valid_row < 80:
            L(f"  * {fid}: only {n_valid_row} valid releases in the "
              f"historical window ({n_ac} have A+C observed). {msg}. "
              f"This reduces precision of beta_{fid} in §15.3.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entrypoint.
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events-csv", type=Path, default=DEFAULT_EVENTS)
    p.add_argument("--coverage-csv", type=Path, default=DEFAULT_COVERAGE)
    p.add_argument("--out-dir", type=Path, default=HERE)
    p.add_argument("--hist-start", type=str, default="2010-01-01")
    p.add_argument("--hist-end", type=str, default="2025-10-31",
                   help="inclusive historical sample end; live forecasts begin the month after")
    args = p.parse_args(argv)

    hist_start = pd.Timestamp(args.hist_start)
    # Inclusive end-of-day for --hist-end.
    hist_end_incl = pd.Timestamp(args.hist_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs.
    events = pd.read_csv(args.events_csv)
    coverage = pd.read_csv(args.coverage_csv, parse_dates=["release_datetime_et"])

    # 1) Emit section 15.1 mapping.
    mapping_p = write_mapping_csv(args.out_dir)

    # 2) Build per-field release table (steps 1-3) with first-release filter.
    fr_all, dropped_revisions = build_field_releases(events)

    # 3) sigma_i on historical sample (step 4).
    sigma = compute_sigma_i(fr_all, hist_start, hist_end_incl)

    # 4) Join futures return + build timestamp-group event table (steps 5-7).
    valid_releases, ts_events = build_timestamp_group_events(
        fr_all, sigma, coverage, hist_start, hist_end_incl
    )

    # Write outputs.
    fr_all_out = fr_all[
        (fr_all["release_datetime_et"] >= hist_start)
        & (fr_all["release_datetime_et"] <= hist_end_incl)
    ].copy().sort_values(
        ["release_datetime_et", "field_id"], kind="mergesort"
    ).reset_index(drop=True)
    fr_all_p = args.out_dir / "field_releases.csv"
    fr_all_out.to_csv(fr_all_p, index=False)

    valid_p = args.out_dir / "field_releases_valid.csv"
    valid_releases.to_csv(valid_p, index=False)

    sigma_p = args.out_dir / "sigma_by_field.csv"
    sigma.to_csv(sigma_p, index=False)

    ts_p = args.out_dir / "timestamp_group_events.csv"
    ts_events.to_csv(ts_p, index=False)

    dropped_p = args.out_dir / "dropped_revisions.csv"
    dropped_revisions.to_csv(dropped_p, index=False)

    report = build_report(
        events=events,
        coverage=coverage,
        valid=valid_releases,
        field_releases=fr_all,
        sigma=sigma,
        timestamp_group_events=ts_events,
        dropped_revisions=dropped_revisions,
        hist_start=hist_start,
        hist_end_inclusive=hist_end_incl,
    )
    report_p = args.out_dir / "preprocessing_report.txt"
    report_p.write_text(report)

    print(report)
    print(
        "Wrote:\n"
        f"  {mapping_p}\n  {fr_all_p}\n  {valid_p}\n  {sigma_p}\n"
        f"  {ts_p}\n  {dropped_p}\n  {report_p}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
