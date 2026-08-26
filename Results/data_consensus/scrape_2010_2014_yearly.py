"""Scrape 2010-2014 in five separate per-year passes, then concatenate.

Wraps scrape_investing_calendar.scrape() without modifying it. Each year
gets a fresh cloudscraper session (via the inner make_scraper() call),
which empirically clears Cloudflare flags between years. Per-year CSVs
are written immediately so a Cloudflare ban on one year does not lose
the others -- you can manually re-run any failed year.

Outputs:
  investing_us_calendar_2010.csv
  investing_us_calendar_2011.csv
  investing_us_calendar_2012.csv
  investing_us_calendar_2013.csv
  investing_us_calendar_2014.csv
  investing_us_calendar_2010_2014.csv  (concatenation of the above)
"""

from __future__ import annotations

import sys
import time
import traceback
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from scrape_investing_calendar import scrape  # noqa: E402


YEARS = [2010, 2011, 2012, 2013, 2014]
# A diagnostic showed fresh single requests succeed (200 OK), but bursts of
# ~12 chunks at 4 s/chunk trip Cloudflare's rate limit. Slowing to 15 s/chunk
# and doubling the chunk window halves the request count and keeps us well
# below the observed threshold. Total estimated runtime:
#   5 years * ~26 14-day chunks/year * ~17 s/chunk = ~37 min.
DELAY_SECONDS = 15.0
CHUNK_DAYS = 14
BETWEEN_YEAR_PAUSE = 60  # seconds; longer break to let Cloudflare fully cool

OUT_FINAL = HERE / "investing_us_calendar_2010_2014.csv"


def per_year_path(year: int) -> Path:
    return HERE / f"investing_us_calendar_{year}.csv"


def scrape_one_year(year: int) -> pd.DataFrame:
    out_path = per_year_path(year)
    if out_path.exists():
        # Idempotent: skip years already scraped (manual re-run support).
        df = pd.read_csv(out_path, dtype={"event_id": str})
        print(f"[year {year}] using cached {out_path.name} ({len(df):,} rows)")
        return df
    print(f"[year {year}] starting scrape", flush=True)
    df = scrape(
        date(year, 1, 1),
        date(year, 12, 31),
        importance=(1, 2, 3),
        delay_seconds=DELAY_SECONDS,
        chunk_days=CHUNK_DAYS,
        verbose=True,
    )
    df.to_csv(out_path, index=False)
    print(f"[year {year}] wrote {len(df):,} rows -> {out_path.name}",
          flush=True)
    return df


def main() -> int:
    failed_years: list[int] = []
    frames: list[pd.DataFrame] = []
    for i, y in enumerate(YEARS):
        try:
            frames.append(scrape_one_year(y))
        except Exception:
            print(f"[year {y}] FAILED:", flush=True)
            traceback.print_exc()
            failed_years.append(y)
        if i < len(YEARS) - 1:
            print(f"[pause {BETWEEN_YEAR_PAUSE}s before next year]",
                  flush=True)
            time.sleep(BETWEEN_YEAR_PAUSE)

    if failed_years:
        print(f"\nFAILED YEARS: {failed_years}. "
              "Re-run this script after diagnosing -- successfully written "
              "per-year CSVs will be reused.", flush=True)
        return 1

    combined = pd.concat(frames, ignore_index=True)
    # Same dedup + sort as the inner scrape() does (defensive).
    combined = combined.drop_duplicates(
        subset=["event_id", "release_datetime_utc", "event"], keep="first"
    )
    combined = combined.sort_values(
        ["release_datetime_et", "event"]
    ).reset_index(drop=True)
    combined.to_csv(OUT_FINAL, index=False)
    print(f"\nWrote concatenated CSV: {len(combined):,} rows -> "
          f"{OUT_FINAL.name}")
    print(f"Date range: {combined['release_datetime_et'].min()} .. "
          f"{combined['release_datetime_et'].max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
