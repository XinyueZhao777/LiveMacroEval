"""Scrape the Investing.com economic calendar for US events.

Uses the site's internal AJAX endpoint
  POST https://www.investing.com/economic-calendar/Service/getCalendarFilteredData

Investing returns `data-event-datetime` in **GMT** by default (the site's
`timeZone` POST parameter is unreliable for anonymous requests). We therefore
interpret the raw timestamp as UTC and convert to America/New_York using
`zoneinfo`, which handles DST correctly. The two columns saved are:
  - `release_datetime_utc` (tz-naive ISO string, UTC wall time as Investing sent it)
  - `release_datetime_et`  (tz-naive ISO string in America/New_York local time)

CLI:
  python scrape_investing_calendar.py \
      --start 2015-01-01 \
      --end   2026-04-20 \
      --out   investing_us_calendar_2015_2026.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import cloudscraper
import pandas as pd
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

ET = ZoneInfo("America/New_York")

ENDPOINT = "https://www.investing.com/economic-calendar/Service/getCalendarFilteredData"
CAL_HOME = "https://www.investing.com/economic-calendar/"

COUNTRY_UNITED_STATES = 5
# Empirically, Investing's `timeZone` IDs are FIXED offsets (no DST). ID=55 is
# the UTC/GMT option (verified: NFP on 2024-03-08 returns 13:30, which equals
# 08:30 EST + 5h). We therefore request UTC and convert to America/New_York in
# Python with `zoneinfo`, which handles DST correctly. Other IDs observed:
#   7  = UTC-5 fixed (EST year-round, no DST)
#   8  = UTC-4 fixed
#   56 = UTC (same as 55 on sampled dates)
#   57 = UTC+2
TIMEZONE_UTC = 55

# Max rows Investing returns per request.
MAX_ROWS_PER_REQUEST = 200

HEADERS_BASE = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.9",
    "X-Requested-With": "XMLHttpRequest",
    "Origin": "https://www.investing.com",
    "Referer": CAL_HOME,
}


# ---------- number parsing -----------------------------------------------------

_NUM_RE = re.compile(r"^\s*(-?)([\d,]+(?:\.\d+)?)\s*([KMBT%]?)\s*$", re.IGNORECASE)
_UNIT_MULT = {"": 1.0, "%": 1.0, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


def parse_numeric(raw: str) -> tuple[float | None, str | None]:
    """Return (value, unit) parsed from an Investing cell, or (None, None).

    Unit is one of {"", "%", "K", "M", "B", "T"}. Percent keeps numeric value
    as-is (5.4% -> 5.4); K/M/B/T are multiplied out so 275K -> 275000.
    """
    if raw is None:
        return None, None
    s = raw.strip()
    if not s or s in {"\xa0", "&nbsp;"}:
        return None, None
    s_clean = s.replace("\xa0", "").replace(",", "").strip()
    m = _NUM_RE.match(s_clean)
    if not m:
        return None, None
    sign = -1.0 if m.group(1) == "-" else 1.0
    number = float(m.group(2))
    unit = m.group(3).upper()
    return sign * number * _UNIT_MULT[unit], unit


# ---------- session & network -------------------------------------------------


def make_scraper() -> cloudscraper.CloudScraper:
    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "linux", "mobile": False},
        delay=5,
    )
    # Warm up cookies / Cloudflare clearance.
    scraper.get(CAL_HOME, headers={"User-Agent": HEADERS_BASE["User-Agent"]}, timeout=30)
    return scraper


@retry(stop=stop_after_attempt(8), wait=wait_exponential(multiplier=4, min=10, max=600),
       reraise=True)
def fetch_chunk(
    scraper: cloudscraper.CloudScraper,
    date_from: str,
    date_to: str,
    importance: tuple[int, ...] = (1, 2, 3),
    limit_from: int = 0,
) -> dict:
    data = [
        ("country[]", str(COUNTRY_UNITED_STATES)),
        *[("importance[]", str(i)) for i in importance],
        ("timeZone", str(TIMEZONE_UTC)),
        ("timeFilter", "timeRemain"),
        ("currentTab", "custom"),
        ("dateFrom", date_from),
        ("dateTo", date_to),
        ("limit_from", str(limit_from)),
    ]
    resp = scraper.post(ENDPOINT, headers=HEADERS_BASE, data=data, timeout=60)
    if resp.status_code == 429:
        retry_after = int(resp.headers.get("Retry-After", "180"))
        # Cap at 10 minutes.
        retry_after = min(retry_after, 600)
        print(
            f"[rate-limited] 429 on {date_from}..{date_to}; sleeping {retry_after}s",
            flush=True,
        )
        time.sleep(retry_after)
        resp.raise_for_status()
    if resp.status_code >= 400:
        print(
            f"[http-err] {resp.status_code} on {date_from}..{date_to}: "
            f"{resp.text[:200]!r}",
            flush=True,
        )
    resp.raise_for_status()
    try:
        return resp.json()
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Non-JSON response ({resp.status_code}) for {date_from}..{date_to}: "
            f"{resp.text[:400]!r}"
        ) from e


# ---------- row parsing -------------------------------------------------------


@dataclass
class CalendarRow:
    event_id: str
    event_attr_id: str
    release_datetime_utc: str  # "YYYY-MM-DD HH:MM:SS" (wall time as Investing sent)
    release_datetime_et: str  # "YYYY-MM-DD HH:MM:SS" (America/New_York)
    utc_offset_et: str  # e.g. "-04:00" or "-05:00"
    timezone: str  # always "America/New_York"
    event: str
    event_url: str | None
    country: str
    currency: str
    importance: int  # 1,2,3
    actual_raw: str
    forecast_raw: str
    previous_raw: str
    actual: float | None
    forecast: float | None
    previous: float | None
    unit: str | None
    notes: str


_IMPORTANCE_FROM_ATTR = {"bull1": 1, "bull2": 2, "bull3": 3}


def _importance_from_cell(cell) -> int:
    key = cell.get("data-img_key") if cell else None
    if key and key in _IMPORTANCE_FROM_ATTR:
        return _IMPORTANCE_FROM_ATTR[key]
    if cell is None:
        return 0
    # Fallback: count the filled bull icons.
    return len(cell.select("i.grayFullBullishIcon"))


def _text(node) -> str:
    if node is None:
        return ""
    return node.get_text(strip=True)


def parse_rows(html_fragment: str) -> list[CalendarRow]:
    """Parse the HTML fragment returned in the `data` field of the JSON response."""
    soup = BeautifulSoup(html_fragment, "lxml")
    out: list[CalendarRow] = []
    for tr in soup.select("tr.js-event-item"):
        dt_raw = tr.get("data-event-datetime", "").strip()
        if not dt_raw:
            continue
        # Investing format: "2024/03/08 13:30:00" in GMT.
        try:
            dt_utc_naive = datetime.strptime(dt_raw, "%Y/%m/%d %H:%M:%S")
        except ValueError:
            # skip rows we can't parse
            continue
        dt_utc = dt_utc_naive.replace(tzinfo=timezone.utc)
        dt_et = dt_utc.astimezone(ET)
        release_utc_iso = dt_utc_naive.strftime("%Y-%m-%d %H:%M:%S")
        release_et_iso = dt_et.strftime("%Y-%m-%d %H:%M:%S")
        et_offset = dt_et.strftime("%z")  # e.g. "-0400"
        et_offset_fmt = et_offset[:3] + ":" + et_offset[3:] if et_offset else ""

        event_id = tr.get("event_attr_id", "") or tr.get("data-event-id", "")
        event_attr_id = tr.get("event_attr_id", "")

        # currency cell
        flag_td = tr.select_one("td.flagCur")
        currency = ""
        country = "United States"
        if flag_td:
            # Remove flag span to read just the currency text.
            currency = flag_td.get_text(strip=True)
            flag_span = flag_td.select_one("span.ceFlags")
            if flag_span and flag_span.get("title"):
                country = flag_span.get("title")

        # sentiment / importance
        sent_td = tr.select_one("td.sentiment")
        importance = _importance_from_cell(sent_td)

        # event name + url
        event_a = tr.select_one("td.event a")
        if event_a is not None:
            event_name = event_a.get_text(strip=True)
            event_url = event_a.get("href")
            if event_url and event_url.startswith("/"):
                event_url = "https://www.investing.com" + event_url
        else:
            ev_td = tr.select_one("td.event")
            event_name = _text(ev_td)
            event_url = None

        # Preliminary / speech marker
        notes_bits: list[str] = []
        ev_td = tr.select_one("td.event")
        if ev_td is not None:
            for span in ev_td.select("span"):
                t = span.get_text(strip=True)
                if t and t not in notes_bits:
                    notes_bits.append(t)

        actual_raw = _text(tr.select_one("td.act"))
        forecast_raw = _text(tr.select_one("td.fore"))
        previous_raw = _text(tr.select_one("td.prev"))

        actual, unit_a = parse_numeric(actual_raw)
        forecast, unit_f = parse_numeric(forecast_raw)
        previous, unit_p = parse_numeric(previous_raw)
        # Prefer the unit we can observe; priority: actual > previous > forecast
        unit = unit_a or unit_p or unit_f

        out.append(
            CalendarRow(
                event_id=event_id,
                event_attr_id=event_attr_id,
                release_datetime_utc=release_utc_iso,
                release_datetime_et=release_et_iso,
                utc_offset_et=et_offset_fmt,
                timezone="America/New_York",
                event=event_name,
                event_url=event_url,
                country=country,
                currency=currency,
                importance=importance,
                actual_raw=actual_raw,
                forecast_raw=forecast_raw,
                previous_raw=previous_raw,
                actual=actual,
                forecast=forecast,
                previous=previous,
                unit=unit,
                notes="; ".join(notes_bits),
            )
        )
    return out


# ---------- date chunking -----------------------------------------------------


def week_chunks(start: date, end: date, days: int = 7) -> Iterable[tuple[date, date]]:
    """Yield (from, to) fixed-width chunks covering [start, end] inclusive.

    Investing's endpoint caps at ~200 rows/response and `bind_scroll_handler` is
    unreliable, so we split small enough that no single chunk hits the cap. With
    US-only and all importance levels, ~50 events/week leaves ample headroom.
    """
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=days - 1), end)
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)


# ---------- driver ------------------------------------------------------------


def scrape(
    start: date,
    end: date,
    importance: tuple[int, ...] = (1, 2, 3),
    delay_seconds: float = 2.0,
    chunk_days: int = 7,
    verbose: bool = True,
) -> pd.DataFrame:
    scraper = make_scraper()
    all_rows: list[CalendarRow] = []
    chunks = list(week_chunks(start, end, days=chunk_days))
    it = tqdm(chunks, disable=not verbose, desc="scraping")
    for d_from, d_to in it:
        payload = fetch_chunk(
            scraper,
            d_from.isoformat(),
            d_to.isoformat(),
            importance=importance,
            limit_from=0,
        )
        rows = parse_rows(payload.get("data", "") or "")
        all_rows.extend(rows)
        if verbose:
            it.set_postfix(chunk=f"{d_from}..{d_to}", rows=len(rows))
        if len(rows) >= MAX_ROWS_PER_REQUEST:
            # Safety net: if we hit the per-request cap, split this chunk.
            mid = d_from + (d_to - d_from) // 2
            if verbose:
                it.write(
                    f"chunk {d_from}..{d_to} hit cap, re-splitting into "
                    f"{d_from}..{mid} and {mid + timedelta(days=1)}..{d_to}"
                )
            for sub_from, sub_to in (
                (d_from, mid),
                (mid + timedelta(days=1), d_to),
            ):
                sub_payload = fetch_chunk(
                    scraper,
                    sub_from.isoformat(),
                    sub_to.isoformat(),
                    importance=importance,
                    limit_from=0,
                )
                all_rows.extend(parse_rows(sub_payload.get("data", "") or ""))
                time.sleep(delay_seconds)
        time.sleep(delay_seconds)

    df = pd.DataFrame([r.__dict__ for r in all_rows])
    if df.empty:
        return df

    # Dedupe on (event_id, release_datetime_utc, event): avoids any accidental
    # overlap between chunk boundaries or cap-split retries.
    df = df.drop_duplicates(
        subset=["event_id", "release_datetime_utc", "event"], keep="first"
    )
    df = df.sort_values(["release_datetime_et", "event"]).reset_index(drop=True)

    # Clip to requested window by ET date.
    mask = (df["release_datetime_et"] >= start.isoformat()) & (
        df["release_datetime_et"] < (end + timedelta(days=1)).isoformat()
    )
    return df.loc[mask].reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument(
        "--importance",
        default="1,2,3",
        help="Comma list of Investing importance levels to keep (1=low,2=med,3=high).",
    )
    p.add_argument(
        "--out",
        default=str(
            Path(__file__).parent / "investing_us_calendar.csv"
        ),
        help="Output CSV path.",
    )
    p.add_argument("--delay", type=float, default=4.0, help="Sleep between requests.")
    p.add_argument("--chunk-days", type=int, default=7, help="Days per request chunk.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    imp = tuple(int(x) for x in args.importance.split(",") if x.strip())

    df = scrape(
        start, end,
        importance=imp,
        delay_seconds=args.delay,
        chunk_days=args.chunk_days,
        verbose=not args.quiet,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"wrote {len(df)} rows to {out_path}")
    if not df.empty:
        print("date range in file:", df["release_datetime_et"].min(), "->",
              df["release_datetime_et"].max())
        print("distinct events:", df["event"].nunique())
    return 0


if __name__ == "__main__":
    sys.exit(main())
