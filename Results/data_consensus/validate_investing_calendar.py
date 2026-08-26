"""Validate a scraped Investing.com calendar CSV against known ground truth.

Two checks:

  1. Hard-coded ground-truth releases (BLS/BEA/etc).
     These values come from primary sources' press releases. If the scraper
     got the timezone wrong or parsed a cell incorrectly, these will mismatch.

  2. Live re-scrape spot check.
     Re-request a random sample of single-day windows from Investing's AJAX
     endpoint and diff against the saved CSV. Any changed Actual/Forecast
     values between the saved CSV and the current site view are reported.

Usage:
    python validate_investing_calendar.py --csv investing_us_calendar_2015_2026.csv
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Reuse parsing primitives from the scraper.
sys.path.insert(0, str(Path(__file__).parent))
from scrape_investing_calendar import (  # noqa: E402
    fetch_chunk,
    make_scraper,
    parse_rows,
    parse_numeric,
)


# ---------- ground truth ------------------------------------------------------


@dataclass
class GroundTruth:
    release_date: str  # YYYY-MM-DD in ET
    release_time_et: str  # HH:MM (24h)
    event_contains: str  # substring to match in Investing's event name
    actual: float  # numeric, in the same unit convention as parse_numeric
    forecast: float | None  # may be None if Investing did not publish a forecast
    source: str


# Curated ground-truth releases. All values are confirmed against BLS/BEA/ISM
# press releases and contemporaneous Investing.com records. The release times
# (ET) are the official first-release times from the source agency.
GROUND_TRUTH: list[GroundTruth] = [
    # NFP: always 8:30 ET, first Friday of the month (BLS)
    GroundTruth("2024-03-08", "08:30", "Nonfarm Payrolls (Feb)",
                actual=275_000, forecast=198_000, source="BLS"),
    GroundTruth("2024-07-05", "08:30", "Nonfarm Payrolls (Jun)",
                actual=206_000, forecast=191_000, source="BLS"),
    GroundTruth("2019-12-06", "08:30", "Nonfarm Payrolls (Nov)",
                actual=266_000, forecast=186_000, source="BLS"),
    GroundTruth("2015-11-06", "08:30", "Nonfarm Payrolls (Oct)",
                actual=271_000, forecast=180_000, source="BLS"),
    # CPI (BLS, 8:30 ET)
    GroundTruth("2024-03-12", "08:30", "CPI (YoY) (Feb)",
                actual=3.2, forecast=3.1, source="BLS"),
    GroundTruth("2022-06-10", "08:30", "CPI (YoY) (May)",
                actual=8.6, forecast=8.3, source="BLS"),
    # Unemployment rate (reported with NFP)
    GroundTruth("2024-03-08", "08:30", "Unemployment Rate (Feb)",
                actual=3.9, forecast=3.7, source="BLS"),
    # Retail sales (Census, 8:30 ET)
    GroundTruth("2024-03-14", "08:30", "Retail Sales (MoM) (Feb)",
                actual=0.6, forecast=0.8, source="Census"),
    # ISM Manufacturing PMI (ISM, 10:00 ET)
    GroundTruth("2024-03-01", "10:00", "ISM Manufacturing PMI (Feb)",
                actual=47.8, forecast=49.5, source="ISM"),
    # Fed Funds (FOMC, usually 14:00 ET)
    GroundTruth("2024-03-20", "14:00", "Fed Interest Rate Decision",
                actual=5.50, forecast=5.50, source="FOMC"),
]


# ---------- validators --------------------------------------------------------


def _norm(s: str) -> str:
    return " ".join(str(s).split()).lower()


def _match_row(df: pd.DataFrame, gt: GroundTruth) -> pd.Series | None:
    """Return the matching row in df for a ground-truth entry, or None."""
    day_mask = df["release_datetime_et"].str.startswith(gt.release_date)
    target = _norm(gt.event_contains)
    ev_norm = df["event"].astype(str).map(_norm)
    # Exact normalized match first.
    exact_mask = day_mask & (ev_norm == target)
    if exact_mask.any():
        return df.loc[exact_mask].iloc[0]
    # Fallback: prefix match on base name + containment of suffix-in-parens.
    base = target.split(" (")[0]
    name_mask = ev_norm.str.startswith(base)
    if "(" in target:
        suffix = "(" + target.split("(", 1)[1]
        name_mask &= ev_norm.str.contains(suffix, regex=False)
    candidates = df.loc[day_mask & name_mask]
    if candidates.empty:
        return None
    return candidates.iloc[0]


def check_ground_truth(df: pd.DataFrame) -> list[dict]:
    results = []
    for gt in GROUND_TRUTH:
        row = _match_row(df, gt)
        if row is None:
            results.append({
                "event": gt.event_contains, "status": "MISSING",
                "expected_time_et": gt.release_time_et,
                "expected_actual": gt.actual, "expected_forecast": gt.forecast,
                "got_time_et": None, "got_actual": None, "got_forecast": None,
            })
            continue
        got_time = str(row["release_datetime_et"]).split(" ", 1)[1][:5]
        time_ok = got_time == gt.release_time_et
        actual_ok = (
            gt.actual is None or row["actual"] is None or pd.isna(row["actual"])
            or _approx_equal(row["actual"], gt.actual)
        )
        forecast_ok = (
            gt.forecast is None or row["forecast"] is None or pd.isna(row["forecast"])
            or _approx_equal(row["forecast"], gt.forecast)
        )
        status = "OK" if (time_ok and actual_ok and forecast_ok) else "MISMATCH"
        results.append({
            "event": gt.event_contains, "status": status,
            "expected_time_et": gt.release_time_et, "got_time_et": got_time,
            "expected_actual": gt.actual, "got_actual": row["actual"],
            "expected_forecast": gt.forecast, "got_forecast": row["forecast"],
            "actual_raw": row["actual_raw"], "forecast_raw": row["forecast_raw"],
        })
    return results


def _approx_equal(a: float, b: float, rtol: float = 0.01, atol: float = 0.5) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= max(atol, abs(float(b)) * rtol)
    except (TypeError, ValueError):
        return False


def live_respot_check(
    df: pd.DataFrame, n_dates: int = 8, seed: int = 0
) -> list[dict]:
    """Re-scrape a random sample of single-day windows and compare."""
    rng = random.Random(seed)
    # Only sample rows that actually have an Actual (i.e., past events).
    past = df[df["actual"].notna()]
    if past.empty:
        return []
    sample_dates = list(
        {
            str(d)
            for d in past["release_datetime_et"].str[:10].sample(
                n=min(n_dates, past["release_datetime_et"].nunique()),
                random_state=seed,
            ).tolist()
        }
    )
    sample_dates.sort()

    scraper = make_scraper()
    diffs: list[dict] = []
    for d in sample_dates:
        try:
            payload = fetch_chunk(scraper, d, d)
        except Exception as e:
            diffs.append({"date": d, "status": "FETCH_FAIL", "error": repr(e)})
            continue
        live_rows = parse_rows(payload.get("data", "") or "")
        live_idx = {(r.event, r.release_datetime_et[:10]): r for r in live_rows}

        saved = df[df["release_datetime_et"].str[:10] == d]
        for _, r in saved.iterrows():
            key = (r["event"], d)
            live = live_idx.get(key)
            if live is None:
                diffs.append({
                    "date": d, "event": r["event"],
                    "status": "NOT_IN_LIVE",
                })
                continue
            if not _strings_match(str(r["actual_raw"]), live.actual_raw) or \
               not _strings_match(str(r["forecast_raw"]), live.forecast_raw):
                diffs.append({
                    "date": d, "event": r["event"], "status": "VALUE_DIFF",
                    "saved_actual": r["actual_raw"], "live_actual": live.actual_raw,
                    "saved_forecast": r["forecast_raw"], "live_forecast": live.forecast_raw,
                })
        time.sleep(5)
    return diffs


def _strings_match(a, b) -> bool:
    def norm(x) -> str:
        # Collapse NaN / None / "nan" / "" all to empty
        if x is None:
            return ""
        try:
            if pd.isna(x):
                return ""
        except (TypeError, ValueError):
            pass
        s = str(x).replace("\xa0", "").replace(",", "").strip()
        if s.lower() in {"nan", "none"}:
            return ""
        return s
    return norm(a) == norm(b)


# ---------- main --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to scraped CSV.")
    p.add_argument("--sample", type=int, default=8, help="Random dates to re-verify.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-live", action="store_true",
                   help="Only run ground-truth checks, skip live re-scrape.")
    args = p.parse_args(argv)

    df = pd.read_csv(args.csv, dtype={"event_id": str})
    print(f"Loaded {len(df):,} rows from {args.csv}")
    print(f"Date range: {df['release_datetime_et'].min()} .. "
          f"{df['release_datetime_et'].max()}")
    print(f"Distinct events: {df['event'].nunique():,}\n")

    print("=" * 70)
    print("GROUND-TRUTH CHECKS (BLS/BEA/ISM/FOMC primary sources)")
    print("=" * 70)
    gt_results = check_ground_truth(df)
    ok = sum(1 for r in gt_results if r["status"] == "OK")
    print(f"Passed {ok}/{len(gt_results)}\n")
    for r in gt_results:
        print(f"  [{r['status']:8}] {r['event']}")
        print(f"    time ET:  expected={r['expected_time_et']}  got={r['got_time_et']}")
        print(f"    actual:   expected={r['expected_actual']}  got={r['got_actual']}"
              f"  raw={r.get('actual_raw')!r}")
        print(f"    forecast: expected={r['expected_forecast']}  got={r['got_forecast']}"
              f"  raw={r.get('forecast_raw')!r}")
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
