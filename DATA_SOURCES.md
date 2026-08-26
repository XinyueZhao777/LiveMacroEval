# Data sources and licensing

This file lists every external data source LiveMacroEval uses, whether it is included in this repository, and how to obtain it if it is not.

We include third-party data only where the source is public domain or where the terms clearly permit redistribution. Everything else is excluded, with the script that consumes it kept in place and documented so the input can be regenerated.

## Included in the repository

| Source | What it provides | Terms |
|---|---|---|
| Federal Reserve Economic Data (FRED), St. Louis Fed | Historical series for the auto-ARIMA baseline, including `DSPIC96` and `PCEC96` | Public domain, U.S. federal government |
| U.S. Census Bureau | New home sales workbook, `sold_cust.xlsx` | Public domain, U.S. federal government |
| Federal Reserve regional banks | Nowcast series from the Atlanta, New York, St. Louis, Cleveland, and Chicago Feds, under `Results/polymarket_return/fed/` | Public domain, U.S. federal government |
| Polymarket CLOB API | Hourly bucket prices for the CPI, unemployment, and GDP markets, under `Results/polymarket_return/polymarket/` | Public API, retrieved with `fetch_polymarket_prices.py` |
| This project | All LLM nowcast records under `LiveMacro/data_light/` and `Results/data_from_serverA_serverB/` | CC BY 4.0, see DATA_LICENSE.md |

## Excluded, and how to obtain it

### Bloomberg ECOS professional consensus

The consensus baseline for the LiveMacro Score. Distributed only through a paid Bloomberg Terminal subscription and not republished on the open web.

The scoring driver expects two files that are not in this repository:

```
Results/bloomberg_consensus/bloomberg_daily_consensus.csv
Results/bloomberg_consensus/bloomberg_release_consensus.csv
```

The first holds the cumulative daily consensus median per field and release, the second holds the final pre-release summary median. Both are built from ECOS survey exports. With a Terminal subscription you can export the survey history for each indicator and assemble these two tables. `Results/market_surprise_capture_score/step_15_4_live_scoring/build_live_scoring_bloomberg.py` documents the exact columns it reads.

### E-mini S&P 500 minute bars

The announcement-window equity return in the LiveMacro Score is computed from front-month ES futures minute bars purchased from **FirstRateData**, a commercial vendor. Their license does not permit redistribution of the bars.

We therefore exclude `Results/data_sp500futures/event_windows_1min.parquet`, which held 73,107 raw OHLCV bars.

We do include `Results/data_sp500futures/event_window_coverage.csv`. This is the derived event-study measurement rather than the underlying data feed. It holds one row per release event with the two window-endpoint closing prices, the matched contract code, and the timing offsets. This is the file the scoring pipeline actually reads, and it is what makes the LiveMacro Score reproducible. If you need to rebuild it from scratch, buy the ES minute-bar archive from FirstRateData and run:

```bash
python Results/data_sp500futures/extract_event_windows.py
```

That script documents the expected archive layout and the contract-roll calendar it follows.

### Saved vendor web pages

The original submission bundle carried locally saved HTML from Investing.com and YCharts, plus two National Association of Realtors PDF reports, under `Results/benchmark_econ/external_data_2026_04_16/raw/`. These are copyrighted pages and reports, so they are excluded here.

The numeric series extracted from them remain under `external_data_2026_04_16/processed/`, covering existing home sales, ISM manufacturing, and ISM services. These are short monthly series of published macroeconomic statistics, roughly 100 observations each. `source_manifest.csv` in that directory records the source URL, the extraction method, and the row count for every one of them.

### Investing.com economic calendar

Release timestamps, event names, and released actual values are scraped from the public Investing.com calendar. We do not redistribute the scraped calendar file itself. Regenerate it with:

```bash
python Results/data_consensus/scrape_investing_calendar.py \
    --start 2015-01-01 --end 2026-04-20 \
    --out investing_us_calendar_2015_2026.csv
python Results/ground_truth/scrape_incremental.py
python Results/ground_truth/parse_ground_truth.py
```

The parsed ground-truth output `Results/ground_truth/data/field_releases_live.csv` **is** included, since it holds only released official statistics, which are public facts published by the issuing agencies.

Note that Investing.com sits behind a Cloudflare edge that blocks many datacenter IP ranges. If the scraper returns HTTP 403 on every request, run it from a different network.

## A note on the consensus column

Investing.com publishes its own forecast column. That column is not the Bloomberg ECOS median and is not used as the consensus anywhere in the paper's headline results. `Results/ground_truth/parse_ground_truth.py` carries it through as `C` for auditing and for the lower-tier score variants, and the headline scoring driver replaces it with the Bloomberg median before any reported number is computed.
