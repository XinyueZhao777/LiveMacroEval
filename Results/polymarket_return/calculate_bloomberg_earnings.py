#!/usr/bin/env python3
"""Bet on Polymarket using the Bloomberg-economist daily-cumulative MEDIAN
consensus, in the same hourly-latest mode as the LLM models.

Drops `betting_results_bloomberg-consensus_<month>_hourly_latest.csv` files
into the existing per-variable hourly_latest folders so the existing
`plot_continuous_feb_mar.py` discovers them with no other changes.

Inputs
------
  - Results/bloomberg_consensus/bloomberg_daily_consensus.csv
        (produced by build_daily_consensus.py; supplies bloomberg_median_calendar
         per (field_id, ref_period, asof_date), already cumulative through end
         of asof_date in calendar units that match map_prediction_to_bucket.)
  - Per-variable polymarket hourly price CSVs from VARIABLE_CONFIGS.

Betting rule (mirrors `calculate_fed_nowcast_earnings(betting_mode='hourly')`
in calculate_earnings.py)
  * Each Bloomberg daily median is treated as available at end-of-day ET
    (23:59:59 ET on its asof_date), DST-correct conversion to UTC.
  * For every polymarket hour before `get_effective_cutoff_time`, take the
    latest median with asof_utc <= bet_time, map to a bucket via
    map_prediction_to_bucket, buy $1 of that bucket at the displayed share
    price, mark `is_winning_bet` against the ground-truth winning bucket.

Markets covered (matches plot_continuous_feb_mar.py's coverage)
  cpi_yoy            : 2026-02 (cpi), 2026-03 (cpi)
  unemployment_rate  : 2026-02 (unemployment_rate), 2026-03 (unemployment_rate)
  real_gdp_qoq       : 2026-Q1 (real_gdp_advance)
        NOTE: Bloomberg's first Q1 asof is 2026-04-16, after the Feb-Mar
        plot window, so the bloomberg-consensus line will be absent from the
        Feb-Mar GDP chart by design (no data inside that window).

Conda env: livemacro.

Usage:
  python polymarket_return/calculate_bloomberg_earnings.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent  # Results/

sys.path.insert(0, str(HERE))
from calculate_earnings import (  # noqa: E402  -- helpers reused as-is
    VARIABLE_CONFIGS,
    derive_winning_bucket,
    get_effective_cutoff_time,
    load_polymarket_data,
    lookup_ground_truth,
    map_prediction_to_bucket,
)

DEFAULT_BLOOMBERG_DAILY = (
    RESULTS / "bloomberg_consensus" / "bloomberg_daily_consensus.csv"
)
DEFAULT_BASE_DIR = RESULTS  # paths in VARIABLE_CONFIGS are relative to Results/
DEFAULT_OUTPUT_ROOT = HERE / "results" / "bet_hourly_latest_Apr26"

VAR_TO_FIELD: dict[str, str] = {
    "cpi_yoy":            "cpi",
    "unemployment_rate":  "unemployment_rate",
    "real_gdp_qoq":       "real_gdp_advance",
}

VAR_MARKETS: dict[str, list[str]] = {
    "cpi_yoy":           ["2026-02", "2026-03"],
    "unemployment_rate": ["2026-02", "2026-03"],
    "real_gdp_qoq":      ["2026-Q1"],
}

ET_TZ = "America/New_York"
MODEL_NAME = "bloomberg-consensus"  # hyphen, not underscore — the plot's
                                    # discovery splits filenames on '_' and
                                    # joins parts[2:month_idx] back as model.


def load_bloomberg_median(
    daily_csv: Path, field_id: str, ref_period: str
) -> pd.DataFrame:
    """Return per-asof-date Bloomberg cumulative-median consensus for one
    (field_id, ref_period), with end-of-day ET (23:59:59) DST-correctly
    converted to naive UTC.

    Columns: asof_utc, median_value (calendar units; unit-matched to the
    inputs map_prediction_to_bucket expects).
    """
    df = pd.read_csv(daily_csv, parse_dates=["asof_datetime_et"])
    sub = df[(df["field_id"] == field_id) & (df["ref_period"] == ref_period)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["asof_utc", "median_value"])
    asof_et = sub["asof_datetime_et"].dt.tz_localize(ET_TZ)
    sub["asof_utc"] = asof_et.dt.tz_convert("UTC").dt.tz_localize(None)
    sub = sub.sort_values("asof_utc").reset_index(drop=True)
    return sub[["asof_utc", "bloomberg_median_calendar"]].rename(
        columns={"bloomberg_median_calendar": "median_value"}
    )


def bet_one_market(
    variable: str,
    target_month: str,
    base_dir: Path,
    daily_csv: Path,
    out_dir: Path,
    bet_amount: float = 1.0,
) -> Path | None:
    """Run hourly-latest Bloomberg-median betting on one (variable, market).

    Returns the path of the written CSV, or None if no bets were placed.
    """
    var_cfg = VARIABLE_CONFIGS[variable]
    if target_month not in var_cfg["months"]:
        raise KeyError(f"target_month {target_month!r} missing from VARIABLE_CONFIGS[{variable!r}]")
    month_cfg = var_cfg["months"][target_month]

    buckets = month_cfg.get("buckets") or var_cfg.get("buckets")
    if not buckets:
        raise ValueError(f"No buckets configured for {variable} / {target_month}")

    poly_path = base_dir / month_cfg["polymarket_data"]
    if not poly_path.exists():
        raise FileNotFoundError(poly_path)
    poly_df = load_polymarket_data(poly_path)

    release_utc, gt_value = lookup_ground_truth(variable, target_month)
    if gt_value is None:
        print(f"  No ground-truth release yet for {variable} / {target_month}; skipping.")
        return None
    winning_bucket = derive_winning_bucket(variable, gt_value, buckets, poly_df)

    cutoff_time, _ = get_effective_cutoff_time(
        poly_df, winning_bucket, official_release_utc=release_utc,
    )

    field_id = VAR_TO_FIELD[variable]
    bb = load_bloomberg_median(daily_csv, field_id, target_month)
    if bb.empty:
        print(
            f"  No Bloomberg daily-median rows for ({field_id}, {target_month}); "
            f"skipping."
        )
        return None

    print(
        f"  {variable} / {target_month}: winning_bucket={winning_bucket!r}, "
        f"cutoff={cutoff_time}, bloomberg rows={len(bb)} "
        f"(first asof_utc={bb['asof_utc'].iloc[0]})"
    )

    trading_df = poly_df[poly_df["datetime_utc"] < cutoff_time].copy()
    rows: list[dict] = []
    for _, hour in trading_df.iterrows():
        bet_time = hour["datetime_utc"]
        eligible = bb[bb["asof_utc"] <= bet_time]
        if eligible.empty:
            continue
        latest = eligible.iloc[-1]
        median_value = float(latest["median_value"])
        predicted_bucket = map_prediction_to_bucket(median_value, variable, buckets)
        share_price = float(hour[predicted_bucket])
        if share_price <= 0 or share_price >= 1:
            continue
        rows.append({
            "datetime_utc": bet_time,
            "betting_date": bet_time.date(),
            "price_time_utc": bet_time,
            "prediction_time_local": latest["asof_utc"],
            "prediction_info": (
                f"bloomberg_median_asof_{pd.Timestamp(latest['asof_utc']).strftime('%Y-%m-%d')}"
            ),
            "prediction": median_value,
            "predicted_bucket": predicted_bucket,
            "share_price": share_price,
            "bet_amount": bet_amount,
            "shares_bought": bet_amount / share_price,
            "is_winning_bet": predicted_bucket == winning_bucket,
        })

    if not rows:
        print(
            f"  No bets placed for {variable} / {target_month} "
            f"(no Bloomberg data before market close)."
        )
        return None

    out_df = pd.DataFrame(rows)
    out_path = (
        out_dir / f"betting_results_{MODEL_NAME}_{target_month}_hourly_latest.csv"
    )
    out_df.to_csv(out_path, index=False)
    n_win = int(out_df["is_winning_bet"].sum())
    invested = float(out_df["bet_amount"].sum())
    final_value = float(out_df.loc[out_df["is_winning_bet"], "shares_bought"].sum())
    ret_pct = (final_value - invested) / invested * 100 if invested > 0 else 0.0
    print(
        f"  Wrote {out_path.name}: {len(out_df)} bets, {n_win} winning, "
        f"final return {ret_pct:.1f}%"
    )
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR,
                   help="Root for polymarket CSV paths in VARIABLE_CONFIGS "
                        "(defaults to Results/).")
    p.add_argument("--bloomberg-daily", type=Path, default=DEFAULT_BLOOMBERG_DAILY)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                   help="Per-variable hourly_latest dirs are created under this.")
    p.add_argument("--variables", nargs="+", default=list(VAR_MARKETS.keys()),
                   choices=list(VAR_MARKETS.keys()))
    args = p.parse_args(argv)

    if not args.bloomberg_daily.exists():
        raise FileNotFoundError(args.bloomberg_daily)

    print(f"Bloomberg daily consensus: {args.bloomberg_daily}")
    print(f"Output root              : {args.output_root}")

    for variable in args.variables:
        out_dir = args.output_root / variable / "hourly_latest"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {variable} ===")
        for target_month in VAR_MARKETS[variable]:
            bet_one_market(
                variable=variable,
                target_month=target_month,
                base_dir=args.base_dir,
                daily_csv=args.bloomberg_daily,
                out_dir=out_dir,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
