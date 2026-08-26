"""Validate the 2010-2014 Investing.com calendar CSV.

Mirrors validate_investing_calendar.py but with a ground-truth set drawn from
the 2010-2014 window. Imports the helper functions (check_ground_truth,
live_respot_check, _approx_equal, _match_row) from validate_investing_calendar
so the comparison logic is identical -- only the GROUND_TRUTH list differs.

Usage:
    python validate_2010_2014.py --csv investing_us_calendar_2010_2014.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from validate_investing_calendar import (  # noqa: E402
    GroundTruth,
    check_ground_truth,
    live_respot_check,
)
import validate_investing_calendar as base  # noqa: E402


# Ground-truth releases in the 2010-2014 window. Actuals are taken from primary
# sources (BLS / BEA / ISM press archives). Forecasts are set to None where
# the contemporaneous consensus is not unambiguous in public records -- the
# validator skips the forecast check when forecast is None.
#
# Naming note: Investing began appending the "(Mon)" period suffix to
# headline event names only on 2014-10-01. Releases before that date use
# the bare event name (e.g., "Nonfarm Payrolls"); on/after that date the
# suffix appears with a double-space ("Nonfarm Payrolls  (Nov)"), which
# the validator's _norm collapses to single-space before comparison. The
# strings below match exactly what is observed in the saved CSV for each
# date so the validator's exact-match path succeeds.
GROUND_TRUTH_2010_2014: list[GroundTruth] = [
    # NFP -- always 8:30 ET, first Friday of the month (BLS)
    # Aug 2011 NFP, released 2011-09-02: famously "zero jobs" headline.
    GroundTruth("2011-09-02", "08:30", "Nonfarm Payrolls",
                actual=0, forecast=74_000, source="BLS"),
    # Jan 2014 NFP, released 2014-02-07: weak +113K (winter weather).
    # Pre-2014-10-01, bare event name (no month suffix).
    GroundTruth("2014-02-07", "08:30", "Nonfarm Payrolls",
                actual=113_000, forecast=185_000, source="BLS"),
    # Nov 2014 NFP, released 2014-12-05: initial +321K, very strong report.
    GroundTruth("2014-12-05", "08:30", "Nonfarm Payrolls (Nov)",
                actual=321_000, forecast=225_000, source="BLS"),
    # Unemployment rate (released with NFP, BLS)
    # Nov 2014 unemployment: 5.8%.
    GroundTruth("2014-12-05", "08:30", "Unemployment Rate (Nov)",
                actual=5.8, forecast=5.8, source="BLS"),
    # CPI YoY (BLS, 8:30 ET)
    # Sep 2011 CPI YoY released 2011-10-19: 3.9%.
    GroundTruth("2011-10-19", "08:30", "CPI (YoY)",
                actual=3.9, forecast=None, source="BLS"),
    # ISM Manufacturing PMI (10:00 ET)
    # Jan 2012 PMI released 2012-02-01: 54.1 (consensus 54.5).
    GroundTruth("2012-02-01", "10:00", "ISM Manufacturing PMI",
                actual=54.1, forecast=54.5, source="ISM"),
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to scraped CSV.")
    p.add_argument("--sample", type=int, default=8,
                   help="Random dates to re-verify against the live site.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-live", action="store_true",
                   help="Only run ground-truth checks, skip live re-scrape.")
    args = p.parse_args(argv)

    # Patch the GROUND_TRUTH used by check_ground_truth so we can reuse the
    # existing implementation unchanged.
    base.GROUND_TRUTH = GROUND_TRUTH_2010_2014

    df = pd.read_csv(args.csv, dtype={"event_id": str})
    print(f"Loaded {len(df):,} rows from {args.csv}")
    print(f"Date range: {df['release_datetime_et'].min()} .. "
          f"{df['release_datetime_et'].max()}")
    print(f"Distinct events: {df['event'].nunique():,}\n")

    print("=" * 70)
    print("GROUND-TRUTH CHECKS (BLS/BEA/ISM primary sources, 2010-2014)")
    print("=" * 70)
    gt_results = check_ground_truth(df)
    ok = sum(1 for r in gt_results if r["status"] == "OK")
    print(f"Passed {ok}/{len(gt_results)}\n")
    for r in gt_results:
        print(f"  [{r['status']:8}] {r['event']}")
        print(f"    time ET:  expected={r['expected_time_et']}  "
              f"got={r['got_time_et']}")
        print(f"    actual:   expected={r['expected_actual']}  "
              f"got={r['got_actual']}  raw={r.get('actual_raw')!r}")
        print(f"    forecast: expected={r['expected_forecast']}  "
              f"got={r['got_forecast']}  raw={r.get('forecast_raw')!r}")
    print()

    if not args.skip_live:
        print("=" * 70)
        print(f"LIVE RE-SCRAPE SPOT CHECK ({args.sample} random dates)")
        print("=" * 70)
        diffs = live_respot_check(df, n_dates=args.sample, seed=args.seed)
        if not diffs:
            print("No diffs. Saved CSV matches the live site for sampled dates.\n")
        else:
            print(f"Found {len(diffs)} diffs:\n")
            for d in diffs:
                print(" ", d)
            print()

    has_mismatch = any(r["status"] != "OK" for r in gt_results)
    return 1 if has_mismatch else 0


if __name__ == "__main__":
    sys.exit(main())
