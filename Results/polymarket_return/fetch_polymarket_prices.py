#!/usr/bin/env python3
"""
Fetch historical price data from Polymarket CLOB API for macro-economic events.

Uses:
1. Gamma API (https://gamma-api.polymarket.com) - to discover markets and token IDs
2. CLOB API (https://clob.polymarket.com) - to fetch price history per token

No authentication required for price data.

Output format matches existing manually-downloaded CSVs:
  "Date (UTC)","Timestamp (UTC)","bucket1","bucket2",...
  "MM-DD-YYYY HH:MM","<unix_ts>","0.485","0.48",...

Usage:
  python polymarket_return/fetch_polymarket_prices.py                    # fetch all events
  python polymarket_return/fetch_polymarket_prices.py --event jan_cpi    # fetch one event
  python polymarket_return/fetch_polymarket_prices.py --list-events      # list available events
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

# ---------------------------------------------------------------------------
# Event definitions
# Each event maps to a Polymarket event slug. Buckets are listed in the
# desired column order. The YES token ID for each bucket is stored so we
# can fetch per-bucket price history.
# ---------------------------------------------------------------------------
EVENTS = {
    # ── Q4 2025 GDP ──────────────────────────────────────────────────────
    "gdp_q4_2025": {
        "slug": "us-gdp-growth-in-q4-2025",
        "event_title": "US GDP growth in Q4 2025?",
        "description": "US GDP growth in Q4 2025?",
        "website_url": "https://polymarket.com/event/us-gdp-growth-in-q4-2025",
        "buckets_ordered": ["<1.0%", "1.0–1.5%", "1.5–2.0%", "2.0–2.5%", "2.5–3.0%", "3.0–3.5%", ">3.5%"],
        "expected_winning_bucket": "1.0–1.5%",
        "output_prefix": "gdp-q4-2025",
    },
    # ── Q1 2026 GDP ──────────────────────────────────────────────────────
    # Note: top bucket label is "≥3.5%" here, vs ">3.5%" for Q4 2025.
    "gdp_q1_2026": {
        "slug": "us-gdp-growth-in-q1-2026",
        "event_title": "US GDP growth in Q1 2026?",
        "description": "US GDP growth in Q1 2026?",
        "website_url": "https://polymarket.com/event/us-gdp-growth-in-q1-2026",
        "buckets_ordered": ["<1.0%", "1.0–1.5%", "1.5–2.0%", "2.0–2.5%", "2.5–3.0%", "3.0–3.5%", "≥3.5%"],
        "expected_winning_bucket": "2.0–2.5%",
        "output_prefix": "gdp-q1-2026",
    },
    # ── January 2026 CPI ─────────────────────────────────────────────────
    "jan_cpi": {
        "slug": "january-inflation-us-annual",
        "event_title": "January Inflation US - Annual",
        "description": "January Inflation US - Annual",
        "website_url": "https://polymarket.com/event/january-inflation-us-annual",
        "buckets_ordered": ["≤2.8%", "2.9%", "3.0%", "3.1%", "≥3.2%"],
        "expected_winning_bucket": "≤2.8%",
        "output_prefix": "cpi-jan",
    },
    # ── January 2026 Jobs ────────────────────────────────────────────────
    "jan_jobs": {
        "slug": "how-many-jobs-added-in-january-321",
        "event_title": "How many jobs added in January?",
        "description": "How many jobs added in January?",
        "website_url": "https://polymarket.com/event/how-many-jobs-added-in-january-321",
        "buckets_ordered": ["<0", "0-25k", "25k–50k", "50k–75k", "75k–100k", "100k–125k", ">125k"],
        "expected_winning_bucket": ">125k",
        "output_prefix": "jobs-jan",
    },
    # ── January 2026 Unemployment ────────────────────────────────────────
    "jan_unemp": {
        "slug": "january-unemployment-rate-333",
        "event_title": "January Unemployment Rate",
        "description": "January Unemployment Rate",
        "website_url": "https://polymarket.com/event/january-unemployment-rate-333",
        "buckets_ordered": ["≤4.2%", "4.3%", "4.4%", "4.5%", "4.6%", "≥4.7%"],
        "expected_winning_bucket": "4.3%",
        "output_prefix": "unemp-jan",
    },
    # ── February 2026 CPI ───────────────────────────────────────────────
    "feb_cpi": {
        "slug": "february-inflation-us-annual",
        "event_title": "February Inflation US - Annual",
        "description": "February Inflation US - Annual",
        "website_url": "https://polymarket.com/event/february-inflation-us-annual",
        "buckets_ordered": ["≤2.1%", "2.2%", "2.3%", "2.4%", "2.5%", "2.6%", "≥2.7%"],
        "expected_winning_bucket": "2.4%",
        "output_prefix": "cpi-feb",
    },
    # ── February 2026 Jobs ──────────────────────────────────────────────
    "feb_jobs": {
        "slug": "how-many-jobs-added-in-february-246",
        "event_title": "How many jobs added in February?",
        "description": "How many jobs added in February?",
        "website_url": "https://polymarket.com/event/how-many-jobs-added-in-february-246",
        "buckets_ordered": ["<25k", "25k–50k", "50k–75k", "75k–100k", "100k–125k", "125k–150k", "150k–175k", "175k+"],
        "expected_winning_bucket": "<25k",
        "output_prefix": "jobs-feb",
    },
    # ── February 2026 Unemployment ──────────────────────────────────────
    "feb_unemp": {
        "slug": "february-unemployment-rate-557",
        "event_title": "February Unemployment Rate",
        "description": "February Unemployment Rate",
        "website_url": "https://polymarket.com/event/february-unemployment-rate-557",
        "buckets_ordered": ["≤4.0%", "4.1%", "4.2%", "4.3%", "4.4%", "4.5%", "≥4.6%"],
        "expected_winning_bucket": "4.4%",
        "output_prefix": "unemp-feb",
    },
    # ── March 2026 CPI ──────────────────────────────────────────────────
    "mar_cpi": {
        "slug": "march-inflation-us-annual",
        "event_title": "March Inflation US - Annual",
        "description": "March Inflation US - Annual",
        "website_url": "https://polymarket.com/event/march-inflation-us-annual",
        "buckets_ordered": ["≤2.0%", "2.1%", "2.2%", "2.3%", "2.4%", "2.5%", "2.6%", "2.7%", "≥2.8%"],
        "expected_winning_bucket": "≥2.8%",
        "output_prefix": "cpi-mar",
    },
    # ── March 2026 Jobs ─────────────────────────────────────────────────
    "mar_jobs": {
        "slug": "how-many-jobs-added-in-march-126",
        "event_title": "How many jobs added in March?",
        "description": "How many jobs added in March?",
        "website_url": "https://polymarket.com/event/how-many-jobs-added-in-march-126",
        "buckets_ordered": ["<-150k", "-150k – -100k", "-100k – -50k", "-50k – 0", "0 – 50k", "50k – 100k", "100k+"],
        "expected_winning_bucket": "100k+",
        "output_prefix": "jobs-mar",
    },
    # ── March 2026 Unemployment ─────────────────────────────────────────
    "mar_unemp": {
        "slug": "march-unemployment-rate-561",
        "event_title": "March Unemployment Rate",
        "description": "March Unemployment Rate",
        "website_url": "https://polymarket.com/event/march-unemployment-rate-561",
        "buckets_ordered": ["≤3.9%", "4.0%", "4.1%", "4.2%", "4.3%", "4.4%", "4.5%", "4.6%", "≥4.7%"],
        "expected_winning_bucket": "4.3%",
        "output_prefix": "unemp-mar",
    },
}


def fetch_event_markets(slug: str) -> dict:
    """Fetch event metadata and all its markets from the Gamma API."""
    resp = requests.get(f"{GAMMA_API}/events/slug/{slug}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def build_bucket_token_map(event_data: dict) -> dict[str, str]:
    """
    Build mapping from bucket label (groupItemTitle) to YES token ID.

    Returns: {bucket_label: yes_token_id}
    """
    mapping = {}
    for market in event_data["markets"]:
        label = market["groupItemTitle"]
        if label in mapping:
            raise ValueError(f"Duplicate bucket label in event data: {label}")
        token_ids = json.loads(market["clobTokenIds"])
        # First token ID is always the YES token
        mapping[label] = token_ids[0]
    return mapping


def parse_iso_to_unix(timestamp_str: str | None) -> int | None:
    """Parse an ISO datetime-like string into unix seconds.

    Handles every variant the Gamma API emits: 'Z' suffix, full '+00:00',
    truncated '+00', or naive (assumed UTC). pandas' parser is strictly
    more permissive than datetime.fromisoformat in Python 3.10.
    """
    if not timestamp_str:
        return None
    return int(pd.to_datetime(timestamp_str, utc=True).timestamp())


def derive_fetch_time_range(event_data: dict) -> tuple[int, int]:
    """
    Derive start/end timestamps from event + market metadata.

    We fetch explicit `startTs`/`endTs` ranges so resolved markets do not lose
    history due the CLOB `interval=max` limitation.
    """
    start_candidates: list[int] = []
    end_candidates: list[int] = []

    for key in ("startDate", "creationDate"):
        parsed = parse_iso_to_unix(event_data.get(key))
        if parsed is not None:
            start_candidates.append(parsed)
    for key in ("endDate", "closedTime", "updatedAt"):
        parsed = parse_iso_to_unix(event_data.get(key))
        if parsed is not None:
            end_candidates.append(parsed)

    for market in event_data.get("markets", []):
        for key in ("startDate", "createdAt"):
            parsed = parse_iso_to_unix(market.get(key))
            if parsed is not None:
                start_candidates.append(parsed)
        for key in ("endDate", "closedTime", "updatedAt"):
            parsed = parse_iso_to_unix(market.get(key))
            if parsed is not None:
                end_candidates.append(parsed)

    if not start_candidates:
        raise ValueError("Unable to derive start timestamp from event/market metadata")
    if not end_candidates:
        raise ValueError("Unable to derive end timestamp from event/market metadata")

    start_ts = min(start_candidates)
    end_ts = max(end_candidates) + 30 * 86400  # extend 30 days for post-resolution prints
    now_ts = int(datetime.now(tz=timezone.utc).timestamp())
    end_ts = min(end_ts, now_ts)

    if start_ts >= end_ts:
        raise ValueError(f"Invalid fetch range derived from metadata: start={start_ts}, end={end_ts}")
    return start_ts, end_ts


def parse_yes_price(market: dict) -> float | None:
    """Extract YES price from a market's outcomePrices payload."""
    raw = market.get("outcomePrices")
    if not raw:
        return None
    try:
        prices = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not prices:
        return None
    try:
        return float(prices[0])
    except (TypeError, ValueError):
        return None


def verify_event_metadata(event_key: str, event_cfg: dict, event_data: dict) -> None:
    """
    Verify event title, bucket labels, and resolved winner against expectations.

    This check ensures exact-name consistency between our config and Polymarket
    API/website metadata before pulling price history.
    """
    expected_title = event_cfg.get("event_title")
    api_title = event_data.get("title")
    if expected_title and api_title != expected_title:
        raise ValueError(
            f"[{event_key}] Event title mismatch. expected={expected_title!r} api={api_title!r}"
        )

    api_buckets = [m.get("groupItemTitle", "") for m in event_data.get("markets", [])]
    expected_buckets = event_cfg["buckets_ordered"]
    missing = sorted(set(expected_buckets) - set(api_buckets))
    extra = sorted(set(api_buckets) - set(expected_buckets))
    if missing or extra:
        raise ValueError(
            f"[{event_key}] Bucket mismatch. missing={missing} extra={extra} "
            f"expected={expected_buckets} api={api_buckets}"
        )

    expected_winner = event_cfg.get("expected_winning_bucket")
    yes_prices = {m["groupItemTitle"]: parse_yes_price(m) for m in event_data.get("markets", [])}
    resolved_winners = sorted([label for label, yes_price in yes_prices.items() if yes_price == 1.0])

    print("  Verification:")
    print(f"    Event title OK: {api_title}")
    print(f"    Website URL: {event_cfg.get('website_url', '')}")
    print(f"    Buckets OK ({len(expected_buckets)}): {expected_buckets}")

    if expected_winner:
        if resolved_winners and expected_winner not in resolved_winners:
            raise ValueError(
                f"[{event_key}] Winner mismatch. expected={expected_winner!r} "
                f"resolved={resolved_winners!r} yes_prices={yes_prices}"
            )
        winner_status = resolved_winners if resolved_winners else "not yet resolved"
        print(f"    Winner check: expected={expected_winner} | resolved={winner_status}")


def fetch_price_history(
    token_id: str,
    start_ts: int,
    end_ts: int,
    fidelity: int = 60,
) -> list[dict]:
    """
    Fetch price history for a single token from the CLOB API.

    For resolved/closed markets, ``interval="max"`` only returns ~2 weeks of
    data.  Using explicit ``startTs``/``endTs`` in 15-day chunks returns the
    full trading history (see Polymarket/py-clob-client#216).

    Args:
        token_id: The YES token ID
        fidelity: Granularity in minutes (60 = hourly)
        start_ts: Unix timestamp for start of range
        end_ts: Unix timestamp for end of range

    Returns:
        List of {t: unix_ts, p: price} dicts
    """
    # Chunked approach: fetch in 15-day windows to avoid timeouts
    chunk_size = 15 * 86400  # 15 days
    all_history: list[dict] = []
    t = start_ts
    while t < end_ts:
        chunk_end = min(t + chunk_size, end_ts)
        resp = requests.get(
            f"{CLOB_API}/prices-history",
            params={
                "market": token_id,
                "startTs": t,
                "endTs": chunk_end,
                "fidelity": fidelity,
            },
            timeout=30,
        )
        resp.raise_for_status()
        all_history.extend(resp.json().get("history", []))
        t = chunk_end
        time.sleep(0.15)  # Rate limit between chunks

    if not all_history:
        return []

    # Keep last observation for duplicate timestamps returned at chunk boundaries.
    all_history.sort(key=lambda h: h["t"])
    deduped: list[dict] = []
    for point in all_history:
        if deduped and deduped[-1]["t"] == point["t"]:
            deduped[-1] = point
        else:
            deduped.append(point)
    return deduped


def build_combined_dataframe(
    bucket_histories: dict[str, list[dict]],
    buckets_ordered: list[str],
) -> pd.DataFrame:
    """
    Combine per-bucket price histories into a single DataFrame matching
    the existing CSV format.

    Strategy: collect all unique timestamps across all buckets, then
    for each timestamp fill in the price from each bucket (forward-fill
    gaps since not all buckets may trade at every timestamp).
    """
    # Build per-bucket Series indexed by hour-rounded timestamp.
    # The API returns timestamps at slightly different seconds within
    # the same hour for different tokens, so we round to the hour to
    # ensure one row per hour in the output.
    all_series = {}
    all_hours = set()
    for bucket, history in bucket_histories.items():
        # Round each timestamp down to its hour
        hour_ts = [h["t"] - (h["t"] % 3600) for h in history]
        price_list = [h["p"] for h in history]
        # If multiple prices in same hour, keep the last one
        s = pd.Series(price_list, index=hour_ts, name=bucket)
        s = s.groupby(s.index).last()
        all_series[bucket] = s
        all_hours.update(s.index)

    if not all_hours:
        return pd.DataFrame()

    # Create DataFrame with one row per hour
    all_hours = sorted(all_hours)
    df = pd.DataFrame(index=all_hours)
    for bucket in buckets_ordered:
        if bucket in all_series:
            df[bucket] = all_series[bucket]
        else:
            df[bucket] = float("nan")

    # Forward-fill prices (if a bucket didn't trade at a particular hour,
    # carry the last known price)
    df = df.ffill()
    # Drop rows where any bucket still has NaN (early rows before all buckets had data)
    df = df.dropna()

    # Build output columns matching existing format
    df.index.name = "unix_ts"
    df = df.reset_index()

    # Format the Date (UTC) column: "MM-DD-YYYY HH:MM"
    df["Date (UTC)"] = df["unix_ts"].apply(
        lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%m-%d-%Y %H:%M")
    )
    df["Timestamp (UTC)"] = df["unix_ts"].astype(str)

    # Reorder columns to match existing CSV format
    cols = ["Date (UTC)", "Timestamp (UTC)"] + buckets_ordered
    df = df[cols]

    return df


def fetch_and_save_event(
    event_key: str,
    output_dir: Path,
    fidelity: int = 60,
) -> Path | None:
    """
    Fetch price history for all buckets in an event and save to CSV.

    Returns: Path to saved CSV, or None if no data.
    """
    event_cfg = EVENTS[event_key]
    slug = event_cfg["slug"]
    buckets_ordered = event_cfg["buckets_ordered"]
    prefix = event_cfg["output_prefix"]

    print(f"\n{'=' * 60}")
    print(f"Fetching: {event_cfg['description']}")
    print(f"  Slug: {slug}")
    print(f"  Buckets: {buckets_ordered}")
    print(f"{'=' * 60}")

    # Step 1: Get market metadata from Gamma API
    print("  Fetching event metadata from Gamma API...")
    event_data = fetch_event_markets(slug)
    verify_event_metadata(event_key, event_cfg, event_data)
    bucket_token_map = build_bucket_token_map(event_data)

    # Determine explicit time range from metadata for chunked fetching.
    # This avoids the interval=max truncation seen on resolved markets.
    start_ts, end_ts = derive_fetch_time_range(event_data)
    start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime("%Y-%m-%d")
    end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc).strftime("%Y-%m-%d")
    print(f"  Using chunked fetch: {start_dt} to {end_dt}")

    # Verify all expected buckets exist
    missing = sorted(set(buckets_ordered) - set(bucket_token_map.keys()))
    if missing:
        raise ValueError(
            f"Missing buckets in Polymarket data: {missing} | available={list(bucket_token_map.keys())}"
        )

    # Step 2: Fetch price history for each bucket
    bucket_histories = {}
    for bucket in buckets_ordered:
        if bucket not in bucket_token_map:
            print(f"  Skipping missing bucket: {bucket}")
            continue

        token_id = bucket_token_map[bucket]
        print(f"  Fetching price history for '{bucket}' (token: {token_id[:20]}...)...")
        history = fetch_price_history(token_id, start_ts=start_ts, end_ts=end_ts, fidelity=fidelity)
        bucket_histories[bucket] = history
        print(f"    Got {len(history)} data points")

        # Rate-limit between buckets (chunked fetching has its own rate limiting)
        time.sleep(0.2)

    # Step 3: Combine into DataFrame
    df = build_combined_dataframe(bucket_histories, buckets_ordered)
    if df.empty:
        print("  No data retrieved!")
        return None

    # Deterministic filename so re-fetches overwrite the same file and
    # VARIABLE_CONFIGS in calculate_earnings.py can pin to a stable path.
    # The full date range is preserved inside the CSV's Date (UTC) column.
    filename = f"{prefix}-polymarket-price-data.csv"
    output_path = output_dir / filename

    df.to_csv(output_path, index=False)
    print(f"\n  Saved {len(df)} rows to: {output_path}")
    print(f"  Date range: {df['Date (UTC)'].iloc[0]} to {df['Date (UTC)'].iloc[-1]}")

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch Polymarket price history for macro events"
    )
    parser.add_argument(
        "--event",
        choices=list(EVENTS.keys()) + ["all"],
        default="all",
        help="Which event to fetch (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default="polymarket_return/polymarket",
        help="Output directory for CSV files (default: polymarket_return/polymarket)",
    )
    parser.add_argument(
        "--fidelity",
        type=int,
        default=60,
        help="Price data granularity in minutes (default: 60 = hourly)",
    )
    parser.add_argument(
        "--list-events",
        action="store_true",
        help="List available events and exit",
    )

    args = parser.parse_args()

    if args.list_events:
        print("Available events:")
        for key, cfg in EVENTS.items():
            print(f"  {key}: {cfg['description']} (slug: {cfg['slug']})")
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events_to_fetch = list(EVENTS.keys()) if args.event == "all" else [args.event]
    saved_files = []

    for event_key in events_to_fetch:
        try:
            path = fetch_and_save_event(event_key, output_dir, fidelity=args.fidelity)
            if path:
                saved_files.append(path)
        except requests.RequestException as e:
            print(f"  ERROR fetching {event_key}: {e}")
        except Exception as e:
            print(f"  ERROR processing {event_key}: {e}")
            raise

    print(f"\n{'=' * 60}")
    print(f"Done! Saved {len(saved_files)} files:")
    for f in saved_files:
        print(f"  {f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
