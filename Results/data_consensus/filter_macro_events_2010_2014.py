"""Apply the macro-variable filter to the 2010-2014 calendar scrape.

Reuses the MAPPING / normalize_event / filter_calendar / print_validation
logic from filter_macro_events.py without modifying that file.

Input:  investing_us_calendar_2010_2014.csv
Output: filtered_macro_events_2010_2014.csv

Note on validation expectations: the existing filter script's expectation
sheet hard-codes years 2016-2025. For 2010-2014 we override the full-year
window to 2010-2014. Coverage caveats inherited from the existing notes:

  * pce_price_index: Investing only began listing the monthly headline PCE
    price index in March 2018, so this variable will be empty for 2010-2014.
    Pre-2018 PCE inflation appeared on Investing only as Core PCE or as the
    quarterly "PCE Prices" line under the GDP release.
  * Other variables should hit ~12 distinct release timestamps per year
    (monthly), or ~12/yr for GDP (4 quarters x 3 revisions).

Pre-2014 dual-listing quirk (handled in this script):

  Investing's older calendar posted TWO rows per release for several
  monthly variables (cpi, ppi, housing_starts, building_permits,
  existing_home_sales, new_home_sales, real_pce) -- a placeholder/stub
  at midnight UTC (release_datetime_utc HH:MM:SS == 09:00:00 on day 1
  of the month, which renders as 04:00 ET in winter or 05:00 ET in
  summer), plus a real-time row at the actual release time on the
  actual release day. The placeholders carry the same release VALUES
  but a fake timestamp and are unusable for market-impact scoring.

  Two-step cleanup applied below:

    1) Re-stamp placeholders with the verified release datetime
       (_restamp_placeholders_via_companion). Some series were posted by
       Investing ONLY as placeholders in early years, so dropping them
       outright would erase the variable for those years. Instead we
       re-use a companion event from the same release that Investing
       always recorded with real timestamps -- per-year sort-and-zip
       gives an unambiguous 1-to-1 mapping (the Nth placeholder by date
       maps to the Nth companion release by date).

       Variables re-stamped here:

         * Real Personal Consumption (MoM) -- companion: "Personal
           Spending (MoM)" (BEA Personal Income & Outlays release,
           8:30 ET). Investing posted RPC only as placeholders for
           2010-2012 and most of 2013. Cross-validated against four
           BEA primary-source release dates (Jan 2010, Aug 2010,
           Mar 2012, Apr 2012); all four match exactly.

         * PPI (YoY) -- companion: "PPI (MoM)" (BLS Producer Price
           Index release, 8:30 ET). Investing posted PPI YoY only as
           placeholders in 2011 (12) and partly in 2012 (Jan-Apr).
           Cross-validated against three BLS primary-source release
           dates (Dec 2010 PPI, Sep 2011 PPI, Apr 2012 PPI); all
           three match exactly. PPI MoM and YoY are reported in the
           same BLS news release at the same instant.

       If a re-stamped placeholder collides with an existing actual-
       time row at the same datetime (same release captured twice by
       Investing), the placeholder is dropped and the actual-time row
       is kept.

    2) Drop all remaining placeholder rows (drop_remaining_placeholders).
       For other variables (cpi index level, housing/building/new/
       existing home sales MoM, etc.) the placeholders are duplicates
       of a non-placeholder row already in the scrape (the headline
       level release at 8:30/10:00 ET), so dropping them simply
       deduplicates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from filter_macro_events import (  # noqa: E402
    MAPPING,
    REVERSE,
    filter_calendar,
    normalize_event,
)

INPUT_CSV = HERE / "investing_us_calendar_2010_2014.csv"
OUTPUT_CSV = HERE / "filtered_macro_events_2010_2014.csv"

# (target_event_base, companion_event_name) pairs. Both companion events
# are released in the same BEA / BLS news release at the same instant
# as the target, and Investing always recorded the companion with a real
# timestamp -- so the companion's release_datetime is the true release
# datetime of the target. See module docstring for cross-validation.
RESTAMP_RULES: list[tuple[str, str]] = [
    ("Real Personal Consumption (MoM)", "Personal Spending (MoM)"),
    ("PPI (YoY)", "PPI (MoM)"),
]


def _placeholder_mask(df: pd.DataFrame) -> pd.Series:
    """Identify placeholder rows: UTC 09:00:00 on day 1 of any month
    (renders as 04:00 ET winter / 05:00 ET summer)."""
    utc = df["release_datetime_utc"].astype(str)
    return (utc.str[11:19] == "09:00:00") & (utc.str[8:10] == "01")


def _restamp_placeholders_via_companion(
    kept: pd.DataFrame,
    raw: pd.DataFrame,
    target_event_base: str,
    companion_event_name: str,
) -> tuple[pd.DataFrame, int, int]:
    """Re-stamp placeholders for `target_event_base` rows using the
    `companion_event_name` actual-time releases via per-year sort-and-zip.

    Returns (modified_df, n_restamped, n_dropped_due_to_collision).

    If a re-stamped placeholder collides with an existing actual-time
    row of the same event_base at the same UTC datetime, drop the
    placeholder (the actual-time row wins). This handles the 2013
    Real PCE case where Investing posted both forms for the same
    Sep-2013 release.
    """
    out = kept.copy()
    placeholder = _placeholder_mask(out)
    target_mask = (out["event_base"] == target_event_base)
    target_placeholder_idx = out.index[target_mask & placeholder]
    if len(target_placeholder_idx) == 0:
        return out, 0, 0

    # Companion comes from the raw scrape; require exact event-name match
    # and exclude any companion rows that themselves landed on a placeholder
    # timestamp (defensive -- empirically this is empty for both companions).
    raw_ph = _placeholder_mask(raw)
    companion = raw[
        (raw["event"] == companion_event_name) & ~raw_ph
    ].sort_values("release_datetime_utc").reset_index(drop=True)
    if companion.empty:
        raise RuntimeError(
            f"No companion {companion_event_name!r} rows found in raw "
            f"scrape; cannot re-stamp {target_event_base!r} placeholders."
        )

    rows_to_drop: list = []
    n_restamped = 0
    for year in sorted(out.loc[target_placeholder_idx, "year"].unique()):
        year_ph_idx = sorted(
            [i for i in target_placeholder_idx if out.at[i, "year"] == year],
            key=lambda i: out.at[i, "release_datetime_utc"],
        )
        comp_year = companion[
            companion["release_datetime_et"].str[:4] == str(year)
        ].reset_index(drop=True)
        if len(comp_year) < len(year_ph_idx):
            raise RuntimeError(
                f"Year {year}: only {len(comp_year)} companion "
                f"{companion_event_name!r} rows for {len(year_ph_idx)} "
                f"{target_event_base!r} placeholders -- cannot match 1-to-1."
            )
        # Existing actual-time same-event_base UTC timestamps in this year.
        existing_actual_ts = set(
            out.loc[
                target_mask
                & ~placeholder
                & out["release_datetime_et"].str.startswith(f"{year}-"),
                "release_datetime_utc",
            ].astype(str)
        )
        for n, ph_idx in enumerate(year_ph_idx):
            comp_row = comp_year.iloc[n]
            new_utc = str(comp_row["release_datetime_utc"])
            if new_utc in existing_actual_ts:
                rows_to_drop.append(ph_idx)
                continue
            out.at[ph_idx, "release_datetime_utc"] = new_utc
            out.at[ph_idx, "release_datetime_et"] = str(
                comp_row["release_datetime_et"]
            )
            out.at[ph_idx, "utc_offset_et"] = str(comp_row["utc_offset_et"])
            n_restamped += 1

    if rows_to_drop:
        out = out.drop(index=rows_to_drop)

    return out, n_restamped, len(rows_to_drop)


def drop_remaining_placeholders(kept: pd.DataFrame) -> pd.DataFrame:
    """Drop any remaining placeholder rows (after Real PCE re-stamping).
    Real PCE rows that have just been re-stamped no longer match the
    placeholder pattern (their UTC time is now the actual release time)
    and so are preserved automatically.
    """
    placeholder = _placeholder_mask(kept)
    return kept.loc[~placeholder].copy()


def print_validation_2010_2014(kept: pd.DataFrame) -> None:
    print("=" * 78)
    print("FILTERED MACRO EVENTS 2010-2014 -- validation")
    print("=" * 78)
    print(f"Total rows kept: {len(kept):,}")
    print(f"Unique event names kept: {kept['event_base'].nunique()}")
    print(f"Date range: {kept['release_datetime_et'].min()}"
          f" .. {kept['release_datetime_et'].max()}")
    print()

    print("-- per variable_key x base event --")
    piv = (
        kept.groupby(["package", "variable_key", "event_base"])
        .size()
        .reset_index(name="n")
        .sort_values(["package", "variable_key", "event_base"])
    )
    print(piv.to_string(index=False))
    print()

    print("-- per-year count per variable_key (any matching base event) --")
    yearly = (
        kept.groupby(["variable_key", "year"]).size().unstack(fill_value=0)
    )
    print(yearly.to_string())
    print()

    print("-- distinct release timestamps per variable_key --")
    per_ts = (
        kept.groupby(["variable_key", "year"])["release_datetime_et"]
        .nunique()
        .unstack(fill_value=0)
    )
    print(per_ts.to_string())
    print()

    EXPECTED_MONTHLY = 12
    EXPECTED_GDP = 12  # 4 quarters * 3 revisions
    expectations = {
        "real_gdp": EXPECTED_GDP,
        "industrial_production": EXPECTED_MONTHLY,
        "durable_goods": EXPECTED_MONTHLY,
        "ism_manufacturing": EXPECTED_MONTHLY,
        "cpi": EXPECTED_MONTHLY,
        "ppi": EXPECTED_MONTHLY,
        "pce_price_index": EXPECTED_MONTHLY,
        "real_pce": EXPECTED_MONTHLY,
        "retail_sales": EXPECTED_MONTHLY,
        "nonfarm_payrolls": EXPECTED_MONTHLY,
        "unemployment_rate": EXPECTED_MONTHLY,
        "housing_starts": EXPECTED_MONTHLY,
        "building_permits": EXPECTED_MONTHLY,
        "existing_home_sales": EXPECTED_MONTHLY,
        "new_home_sales": EXPECTED_MONTHLY,
        "ism_services": EXPECTED_MONTHLY,
    }

    print("-- expectation vs observed distinct release timestamps "
          "(years 2010-2014, all full years) --")
    full_years = [y for y in range(2010, 2015) if y in per_ts.columns]
    print(f"full years used: {full_years}")
    for vk, exp in expectations.items():
        if vk not in per_ts.index:
            print(f"  {vk:25s}  NO EVENTS FOUND")
            continue
        obs = per_ts.loc[vk, full_years]
        total_obs = int(obs.sum())
        total_exp = exp * len(full_years)
        diff = total_obs - total_exp
        flag = "OK" if abs(diff) <= 2 * len(full_years) else "CHECK"
        print(f"  {vk:25s}  expected={total_exp:3d}  "
              f"observed={total_obs:3d}  diff={diff:+d}  [{flag}]")
    print()

    print("-- coverage notes --")
    print("  pce_price_index: Investing began listing monthly headline PCE")
    print("    price index only in March 2018 -- expected 0 for 2010-2014.")
    print("  real_pce: 2010-2012 placeholders re-stamped to companion")
    print("    Personal Spending (MoM) datetimes (BEA Personal Income &")
    print("    Outlays release slot, 8:30 ET); see module docstring.")
    print("  ppi YoY: 2011 and Jan-Apr 2012 placeholders re-stamped to")
    print("    companion PPI (MoM) datetimes (BLS PPI release slot,")
    print("    8:30 ET); see module docstring.")


def main() -> None:
    df = pd.read_csv(INPUT_CSV, dtype={"event_id": str})
    print(f"Loaded {len(df):,} rows from {INPUT_CSV.name}")

    kept = filter_calendar(df)
    n_total = len(kept)
    n_placeholder = int(_placeholder_mask(kept).sum())
    print(f"After mapping filter: {n_total:,} rows "
          f"({n_placeholder} placeholder rows present)")

    for target, companion in RESTAMP_RULES:
        kept, n_re, n_drop = _restamp_placeholders_via_companion(
            kept, df, target, companion
        )
        print(f"  re-stamped {n_re} {target!r} placeholders to companion "
              f"{companion!r} datetimes; dropped {n_drop} that collided "
              "with existing actual-time rows")

    n_before_drop = len(kept)
    kept = drop_remaining_placeholders(kept)
    print(f"  dropped {n_before_drop - len(kept)} remaining placeholder "
          f"rows from other variables")

    out_cols = list(df.columns) + ["event_base", "package", "variable_key"]
    kept[out_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(kept):,} rows -> {OUTPUT_CSV.name}\n")

    print_validation_2010_2014(kept)


if __name__ == "__main__":
    main()
