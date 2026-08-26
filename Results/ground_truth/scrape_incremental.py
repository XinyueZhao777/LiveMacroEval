"""Incrementally extend the Investing.com calendar CSV up to --end (today by default).

Why a separate script:
  The baseline scraper `data_consensus/scrape_investing_calendar.py`
  scrapes the full date range every run, which is slow. This script reuses the
  same low-level scraper primitives but only requests the window
      [max_existing_release - overlap_days, end]
  then merges the newly scraped rows into the existing CSV with duplicate
  preference given to newly scraped rows (so any upstream revisions to
  actuals/consensus that Investing posted after the first scrape are picked up).

The baseline input CSV is NEVER modified — the combined output is always
written to a new file under `Results/ground_truth/data/`.

Dedupe key matches the upstream scraper: (event_id, release_datetime_utc, event).
Concatenation order places newly scraped rows first so `drop_duplicates(keep="first")`
retains the fresh values on an overlap row.

Conda env: livemacro.

Usage:
  python scrape_incremental.py                      # end defaults to today
  python scrape_incremental.py --end 2026-04-23
  python scrape_incremental.py --overlap-days 21

Outputs:
  data/investing_us_calendar_latest.csv  (combined: baseline + newly scraped)
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
REPO_RESULTS = HERE.parent  # livemacro/Results

DEFAULT_BASELINE_CSV = (
    REPO_RESULTS
    / "data_consensus"
    / "investing_us_calendar_2015_2026.csv"
)
DEFAULT_OUT_CSV = HERE / "data" / "investing_us_calendar_latest.csv"

# Reuse the upstream scraper's low-level primitives.
sys.path.insert(
    0,
    str(REPO_RESULTS / "data_consensus"),
)
from scrape_investing_calendar import scrape  # noqa: E402


DEDUPE_KEY = ["event_id", "release_datetime_utc", "event"]


def read_baseline(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline calendar CSV not found: {path}. Expected the initial "
            "2015–2026 scrape produced by "
            "`data_consensus/scrape_investing_calendar.py`."
        )
    return pd.read_csv(path, dtype={"event_id": str, "event_attr_id": str})


def resolve_incremental_window(
    baseline: pd.DataFrame, overlap_days: int, end: date
) -> tuple[date, date]:
    """Return (start, end) for the incremental request, using the ET date of
    the latest row in the baseline CSV as the anchor.

    A negative `overlap_days - 1` range (e.g. baseline already covers through
    today) still requests at least the single day `end`, to keep the flow
    uniform for reruns on the same day.
    """
    if baseline.empty:
        raise RuntimeError("Baseline CSV is empty; cannot derive incremental start.")
    max_et = pd.to_datetime(baseline["release_datetime_et"]).max()
    anchor = max_et.date()
    start = anchor - timedelta(days=max(0, overlap_days))
    if start > end:
        # baseline reaches past the requested end — still scrape at least `end`.
        start = end
    return start, end


def combine(baseline: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Merge fresh rows into the baseline, preferring fresh on dup keys.

    Column order follows the baseline. Any new columns in `fresh` are appended
    at the end. Rows are sorted by `release_datetime_et` then `event`.
    """
    if fresh.empty:
        return baseline.copy()

    base_cols = list(baseline.columns)
    extra_cols = [c for c in fresh.columns if c not in base_cols]
    all_cols = base_cols + extra_cols

    # Align columns (fresh first so keep='first' prefers fresh on dup).
    fresh = fresh.reindex(columns=all_cols)
    baseline = baseline.reindex(columns=all_cols)
    combined = pd.concat([fresh, baseline], ignore_index=True)
    missing = [c for c in DEDUPE_KEY if c not in combined.columns]
    if missing:
        raise RuntimeError(f"Dedupe key columns missing: {missing}")
    combined = combined.drop_duplicates(subset=DEDUPE_KEY, keep="first")
    combined = combined.sort_values(
        ["release_datetime_et", "event"], kind="mergesort"
    ).reset_index(drop=True)
    return combined


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--baseline-csv",
        type=Path,
        default=DEFAULT_BASELINE_CSV,
        help=(
            "Existing calendar CSV to extend. Not modified. "
            f"Default: {DEFAULT_BASELINE_CSV}"
        ),
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_OUT_CSV,
        help=(
            "Combined output CSV (baseline + newly scraped rows). "
            f"Default: {DEFAULT_OUT_CSV}"
        ),
    )
    p.add_argument(
        "--end",
        type=str,
        default=None,
        help="Inclusive end date YYYY-MM-DD for the incremental request. Default: today.",
    )
    p.add_argument(
        "--overlap-days",
        type=int,
        default=14,
        help=(
            "Re-scrape this many days before the baseline's latest row to pick "
            "up any upstream revisions Investing may have posted after the "
            "initial scrape. Default: 14."
        ),
    )
    p.add_argument(
        "--importance",
        default="1,2,3",
        help="Comma list of Investing importance levels (1=low,2=med,3=high). Default 1,2,3.",
    )
    p.add_argument("--delay", type=float, default=4.0)
    p.add_argument("--chunk-days", type=int, default=7)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    end = (
        datetime.strptime(args.end, "%Y-%m-%d").date()
        if args.end
        else date.today()
    )
    imp = tuple(int(x) for x in args.importance.split(",") if x.strip())

    baseline = read_baseline(args.baseline_csv)
    base_max = pd.to_datetime(baseline["release_datetime_et"]).max()
    print(f"Baseline: {len(baseline):,} rows, latest release_datetime_et = {base_max}")

    start, end = resolve_incremental_window(baseline, args.overlap_days, end)
    print(f"Incremental scrape window (ET dates): {start} .. {end}")

    fresh = scrape(
        start,
        end,
        importance=imp,
        delay_seconds=args.delay,
        chunk_days=args.chunk_days,
        verbose=not args.quiet,
    )
    print(f"Newly scraped: {len(fresh):,} rows")

    combined = combine(baseline, fresh)
    new_rows = len(combined) - len(baseline)
    print(
        f"Combined: {len(combined):,} rows  "
        f"(net new after dedupe: {new_rows:+d})"
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")
    if not combined.empty:
        print(
            "Combined date range (ET): "
            f"{combined['release_datetime_et'].min()} -> "
            f"{combined['release_datetime_et'].max()}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
