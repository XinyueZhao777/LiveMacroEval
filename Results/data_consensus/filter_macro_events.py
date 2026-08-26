"""Filter the Investing.com calendar to the 17 macro variables in the scoring
framework, then validate per-variable release counts.

Input:  investing_us_calendar_2015_2026.csv (full scrape)
Output: filtered_macro_events.csv (rows tagged with `package` + `variable_key`)

Mapping choices (see Project Outline.md section 2 and variables.py):

  * Each 'variable_key' corresponds to one calendar headline field (level,
    MoM, or YoY). Releases can produce multiple calendar rows per release
    timestamp (e.g., CPI MoM + YoY + index on the same timestamp).
  * Real Disposable Personal Income is not published on Investing.com's
    calendar; that variable has no calendar event.
  * PCE price index was historically labeled 'PCE Deflator' on Investing
    (pre-2019) and 'PCE price index' (post-2019). Both are kept and map to
    the same variable.
  * For GDP, Investing lists advance/second/third releases under a single
    event name 'GDP (QoQ)'. All revisions are retained here; filtering to
    advance-only can be done downstream if needed.
  * 'ISM Services PMI' appears on Investing as 'ISM Non-Manufacturing PMI'
    throughout the sample.

Coverage (distinct release timestamps, 2016–2025 full years, expected 120):

  * 14 of 16 mapped variables match 120 ± 5 (OK).
  * pce_price_index: 91 (−29) — Investing did not list monthly headline PCE price index before 2018-03-29 (only Core PCE and quarterly "PCE Prices" from the GDP release existed). No in-source fix.
  * building_permits: 154 (+34) — from 2023 onward Investing posts both the preliminary 08:30 ET release (concurrent with Housing Starts) and a final/revised release ~1 week later. Both retained; downstream scoring can restrict to 08:30.
  * real_dpi: 0 — Real Disposable Personal Income is not published on the Investing.com calendar. This variable has no consensus data source.

16 of the 17 variables are mapped. Mapping explicitly excludes core measures (Core CPI/PPI/PCE), regional variants (Cleveland CPI, Dallas Fed PCE), derived Census Bureau breakdowns (Core/Control/ex-Gas retail sales), ADP private payrolls, and Investing's GDPNow nowcast, to avoid false positives.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


HERE = Path(__file__).parent
INPUT_CSV = HERE / "investing_us_calendar_2015_2026.csv"
OUTPUT_CSV = HERE / "filtered_macro_events.csv"


# package -> { variable_key -> [normalized base event names] }
MAPPING: dict[str, dict[str, list[str]]] = {
    "gdp": {
        # Advance + 2 revisions per quarter share the same calendar name.
        "real_gdp": ["GDP (QoQ)"],
    },
    "industrial_production": {
        "industrial_production": [
            "Industrial Production (MoM)",
            "Industrial Production (YoY)",
        ],
    },
    "durable_goods": {
        "durable_goods": ["Durable Goods Orders (MoM)"],
    },
    "ism_manufacturing": {
        "ism_manufacturing": ["ISM Manufacturing PMI"],
    },
    "cpi": {
        "cpi": [
            "CPI (MoM)",
            "CPI (YoY)",
            "CPI Index, n.s.a.",
            "CPI Index, s.a",
            "CPI, n.s.a (MoM)",
        ],
    },
    "ppi": {
        "ppi": ["PPI (MoM)", "PPI (YoY)"],
    },
    "personal_income_outlays": {
        # Monthly PCE price index (BEA). "PCE Deflator" (pre-Aug-2019) was
        # renamed to "PCE price index" thereafter; both represent the same
        # monthly series.
        "pce_price_index": [
            "PCE price index (MoM)",
            "PCE Price index (YoY)",
            "PCE Deflator (MoM)",
            "PCE Deflator (YoY)",
        ],
        "real_pce": ["Real Personal Consumption (MoM)"],
        # Real Disposable Personal Income is not on Investing's calendar.
    },
    "retail_sales": {
        "retail_sales": ["Retail Sales (MoM)", "Retail Sales (YoY)"],
    },
    "employment_situation": {
        "nonfarm_payrolls": ["Nonfarm Payrolls"],
        "unemployment_rate": ["Unemployment Rate"],
    },
    "new_residential_construction": {
        "housing_starts": ["Housing Starts", "Housing Starts (MoM)"],
        "building_permits": ["Building Permits", "Building Permits (MoM)"],
    },
    "existing_home_sales": {
        "existing_home_sales": [
            "Existing Home Sales",
            "Existing Home Sales (MoM)",
        ],
    },
    "new_home_sales": {
        "new_home_sales": ["New Home Sales", "New Home Sales (MoM)"],
    },
    "ism_services": {
        # Investing still uses the legacy "Non-Manufacturing" label.
        "ism_services": ["ISM Non-Manufacturing PMI"],
    },
}


# Build reverse index: base event name -> (package, variable_key)
REVERSE: dict[str, tuple[str, str]] = {}
for pkg, vars_ in MAPPING.items():
    for vk, names in vars_.items():
        for n in names:
            if n in REVERSE:
                raise RuntimeError(f"Duplicate event name in mapping: {n}")
            REVERSE[n] = (pkg, vk)


_SUFFIX_MONTH = re.compile(r"\s*\([A-Z][a-z]{2,3}\)\s*$")
_SUFFIX_MONTHS = re.compile(r"\s*\([A-Z][a-z]{2,3}/[A-Z][a-z]{2,3}\)\s*$")
_SUFFIX_QUARTER = re.compile(r"\s*\(Q[1-4]\)\s*$")


def normalize_event(name: str) -> str:
    """Strip trailing period tag like ' (Feb)', ' (Jan/Feb)', ' (Q3)' so
    events can be compared across release periods."""
    if not isinstance(name, str):
        return ""
    s = _SUFFIX_MONTH.sub("", name)
    s = _SUFFIX_MONTHS.sub("", s)
    s = _SUFFIX_QUARTER.sub("", s)
    return s.strip()


def filter_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["event_base"] = df["event"].map(normalize_event)
    df["package"] = df["event_base"].map(lambda b: REVERSE.get(b, (None, None))[0])
    df["variable_key"] = df["event_base"].map(
        lambda b: REVERSE.get(b, (None, None))[1]
    )
    kept = df[df["package"].notna()].copy()
    # Year for validation convenience.
    kept["year"] = kept["release_datetime_et"].str[:4].astype(int)
    return kept


def print_validation(kept: pd.DataFrame) -> None:
    print("=" * 78)
    print("FILTERED MACRO EVENTS — validation")
    print("=" * 78)
    print(f"Total rows kept: {len(kept):,}")
    print(f"Unique event names kept: {kept['event_base'].nunique()}")
    print(f"Date range: {kept['release_datetime_et'].min()}"
          f" .. {kept['release_datetime_et'].max()}")
    print()

    # Overall count per (package, variable_key, base event)
    print("-- per variable_key x base event --")
    piv = (
        kept.groupby(["package", "variable_key", "event_base"])
        .size()
        .reset_index(name="n")
        .sort_values(["package", "variable_key", "event_base"])
    )
    print(piv.to_string(index=False))
    print()

    # Per-year count per variable_key (should be ~12/yr monthly, ~4/yr GDP,
    # where GDP has 3 revisions each = ~12/yr events under same name).
    print("-- per-year count per variable_key (any matching base event) --")
    yearly = (
        kept.groupby(["variable_key", "year"]).size().unstack(fill_value=0)
    )
    print(yearly.to_string())
    print()

    # Distinct release timestamps per variable_key (collapse the same
    # timestamp with multiple headline fields to one release).
    print("-- distinct release timestamps per variable_key --")
    per_ts = (
        kept.groupby(["variable_key", "year"])["release_datetime_et"]
        .nunique()
        .unstack(fill_value=0)
    )
    print(per_ts.to_string())
    print()

    # Expectation sheet. Year 2015 and 2026 are partial and excluded from
    # pass/fail evaluation. Years 2016-2025 should be close to target.
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
          "(years 2016-2025 full years only) --")
    full_years = [y for y in range(2016, 2026) if y in per_ts.columns]
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

    missing = {"real_dpi": "Real Disposable Personal Income is not "
               "published on the Investing.com US calendar."}
    if missing:
        print("-- variables intentionally not mapped --")
        for k, v in missing.items():
            print(f"  {k}: {v}")
    print()

    print("-- notes on CHECK flags --")
    print("  pce_price_index: Investing did not track monthly headline PCE")
    print("    price index before March 2018 — only Core PCE and the")
    print("    quarterly 'PCE Prices' (from GDP release) existed. Coverage")
    print("    begins 2018-03-29.")
    print("  building_permits: from 2023 onward Investing lists two")
    print("    releases per month — the preliminary (08:30 ET, alongside")
    print("    Housing Starts) and the final/revised (~1 week later,")
    print("    irregular time). Both are retained; downstream MSC scoring")
    print("    can drop rows without consensus or restrict to the 08:30")
    print("    release.")


def main() -> None:
    df = pd.read_csv(INPUT_CSV, dtype={"event_id": str})
    print(f"Loaded {len(df):,} rows from {INPUT_CSV.name}")

    kept = filter_calendar(df)
    # Save with the extra labels at the end, keeping original columns first.
    out_cols = list(df.columns) + ["event_base", "package", "variable_key"]
    kept[out_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(kept):,} rows -> {OUTPUT_CSV.name}\n")

    print_validation(kept)


if __name__ == "__main__":
    main()
