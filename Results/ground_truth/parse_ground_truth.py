"""Parse the combined Investing.com calendar into a canonical ground-truth
field-release table shared by every downstream tester.

Pipeline:
  1. Load the combined calendar CSV produced by `scrape_incremental.py`
     (`data/investing_us_calendar_latest.csv` by default).
  2. Tag every row with (package, variable_key, event_base) by reusing
     `data_consensus/filter_macro_events.py`.
  3. Build per-field first-release-per-reference-period rows using the same
     `MAPPING` and `keep_first_release` primitives as step 15.2, but with a
     LOOSER observed-value filter than step 15.2:
       * drops revision releases (first-release-per-ref-period),
       * keeps every row with `A` (actual) observed, regardless of whether
         `C` (consensus) is observed. This differs intentionally from
         step 15.2's `build_field_releases`, which requires both A and C.
     Rationale: downstream consumers of ground truth need the actual print
     whenever it exists (e.g., scoring forecasts that don't rely on
     Investing's consensus). Consumers that need A+C (e.g., the historical
     sigma step) can re-apply the tighter filter locally.
  4. Assign a `ref_period` column using the step-15.4 rule
     (suffix → fallback release_month − 1). This matches how the live scoring
     file currently labels ref_period.

Output schema (shared across downstream tasks):
    field_id              canonical macro field (step-15.2 MAPPING id).
    release_datetime_et   naive US/Eastern release timestamp.
    event                 raw Investing event name (for human audit).
    ref_period            canonical reference period ('YYYY-MM' or 'YYYY-Qn').
    A                     actual (calendar first-release actual). Never NaN.
    C                     consensus forecast. May be NaN when Investing did
                          not publish a consensus for this release.
    raw_surprise          A - C (float). NaN when C is NaN.

Output files:
    data/field_releases_all.csv     full history (2015 → combined max date).
    data/field_releases_live.csv    subset with release_datetime_et >= --live-start.

No downstream code is edited by this script. The caller decides when to point
step_15_4 / benchmark / plots at these files.

Conda env: livemacro.

Usage:
  python parse_ground_truth.py
  python parse_ground_truth.py --live-start 2025-11-01
  python parse_ground_truth.py --calendar-csv data/investing_us_calendar_latest.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
REPO_RESULTS = HERE.parent  # livemacro/Results

DEFAULT_CALENDAR_CSV = HERE / "data" / "investing_us_calendar_latest.csv"
DEFAULT_OUT_ALL = HERE / "data" / "field_releases_all.csv"
DEFAULT_OUT_LIVE = HERE / "data" / "field_releases_live.csv"
DEFAULT_LIVE_START = "2025-11-01"

DATA_CONSENSUS_DIR = REPO_RESULTS / "data_consensus"
STEP_15_2_DIR = (
    REPO_RESULTS
    / "market_surprise_capture_score"
    / "step_15_2_historical_preprocessing"
)

sys.path.insert(0, str(DATA_CONSENSUS_DIR))
sys.path.insert(0, str(STEP_15_2_DIR))

from filter_macro_events import filter_calendar  # noqa: E402
from preprocess_historical import (  # noqa: E402
    FIELD_IDS,
    MAPPING,
    _parse_ref_from_suffix,
    keep_first_release,
)

OUT_COLS = [
    "field_id",
    "release_datetime_et",
    "event",
    "ref_period",
    "A",
    "C",
    "raw_surprise",
]


def assign_ref_period(event: str, release_dt: pd.Timestamp) -> str:
    """Match step_15_4_live_scoring/build_live_scoring.assign_ref_period.

    Returns the step-15.2 suffix-parsed ref_period when the event name has a
    parseable period tag; otherwise falls back to (release_month - 1).
    """
    parsed = _parse_ref_from_suffix(event, release_dt)
    if parsed is not None:
        return parsed
    mo = release_dt.month - 1
    ry = release_dt.year if mo >= 1 else release_dt.year - 1
    if mo < 1:
        mo = 12
    return f"{ry}-{mo:02d}"


def _build_field_releases_actual_only(events: pd.DataFrame) -> pd.DataFrame:
    """Ground-truth variant of step_15_2.build_field_releases.

    Same per-field mapping and first-release-per-ref-period logic, but keeps
    every row with `actual` observed — `forecast` may be NaN. Produces the
    same A / C / raw_surprise columns (C and raw_surprise are NaN when the
    Investing row has no consensus).
    """
    ev = events.copy()
    ev["release_datetime_et"] = pd.to_datetime(ev["release_datetime_et"])
    parts: list[pd.DataFrame] = []
    for m in MAPPING:
        field_id = m["field_id"]
        priority = m["event_bases"]
        sub = ev[ev["event_base"].isin(priority)].copy()
        if sub.empty:
            continue
        # Collapse same-timestamp duplicates across fallback event_base names
        # (mirrors step_15_2 logic exactly).
        rank_map = {name: i for i, name in enumerate(priority)}
        sub["_rank"] = sub["event_base"].map(rank_map)
        sub.sort_values(
            ["release_datetime_et", "_rank"], kind="mergesort", inplace=True
        )
        sub = sub.drop_duplicates(
            subset=["release_datetime_et"], keep="first"
        ).drop(columns="_rank")

        kept, _dropped = keep_first_release(sub)
        # Looser filter: require A observed; C may be NaN.
        kept = kept[kept["actual"].notna()].copy()
        kept["field_id"] = field_id
        kept["A"] = kept["actual"].astype(float)
        kept["C"] = pd.to_numeric(kept["forecast"], errors="coerce")
        kept["raw_surprise"] = kept["A"] - kept["C"]
        parts.append(
            kept[
                [
                    "field_id",
                    "release_datetime_et",
                    "event",
                    "A",
                    "C",
                    "raw_surprise",
                ]
            ]
        )
    if not parts:
        raise RuntimeError("No field releases built — mapping or input empty.")
    return pd.concat(parts, ignore_index=True)


def build_canonical(calendar_csv: Path) -> pd.DataFrame:
    """Return the canonical ground-truth field-release DataFrame."""
    if not calendar_csv.exists():
        raise FileNotFoundError(
            f"Combined calendar CSV not found: {calendar_csv}. Run "
            "`scrape_incremental.py` first."
        )
    df = pd.read_csv(calendar_csv, dtype={"event_id": str})
    print(f"Loaded {len(df):,} calendar rows from {calendar_csv.name}")

    # Step 1: tag with (event_base, package, variable_key).
    tagged = filter_calendar(df)
    print(
        f"After filter_macro_events mapping: {len(tagged):,} rows "
        f"({tagged['variable_key'].nunique()} variable_keys)"
    )

    # Step 2: first-release-per-reference-period, A observed (C optional).
    kept = _build_field_releases_actual_only(tagged)
    kept["release_datetime_et"] = pd.to_datetime(kept["release_datetime_et"])
    n_with_c = int(kept["C"].notna().sum())
    print(
        f"After first-release filter: {len(kept):,} rows "
        f"(A observed; {n_with_c:,} also have C observed)"
    )

    # Step 3: canonical ref_period (step-15.4 rule).
    kept["ref_period"] = [
        assign_ref_period(e, t)
        for e, t in zip(kept["event"].tolist(), kept["release_datetime_et"].tolist())
    ]

    kept = kept.sort_values(
        ["release_datetime_et", "field_id"], kind="mergesort"
    ).reset_index(drop=True)
    return kept[OUT_COLS]


def print_summary(all_df: pd.DataFrame, live_df: pd.DataFrame, live_start: pd.Timestamp) -> None:
    print("=" * 78)
    print("Ground-truth field releases — summary")
    print("=" * 78)
    print(f"All releases:  {len(all_df):,}")
    if not all_df.empty:
        print(
            f"  date range:  {all_df['release_datetime_et'].min()} -> "
            f"{all_df['release_datetime_et'].max()}"
        )
    print(f"Live releases (>= {live_start.date()}): {len(live_df):,}")
    if not live_df.empty:
        print(
            f"  date range:  {live_df['release_datetime_et'].min()} -> "
            f"{live_df['release_datetime_et'].max()}"
        )
    print()

    if not live_df.empty:
        n_total = live_df.groupby("field_id").size().reindex(FIELD_IDS, fill_value=0)
        n_with_c = (
            live_df.assign(_has_c=live_df["C"].notna())
            .groupby("field_id")["_has_c"]
            .sum()
            .reindex(FIELD_IDS, fill_value=0)
            .astype(int)
        )
        summary = pd.DataFrame({"n_live": n_total, "n_with_C": n_with_c})
        print("-- live per-field release counts (A observed; C observed subset) --")
        print(summary.to_string())
        print(
            f"sum: n_live={int(n_total.sum())}  n_with_C={int(n_with_c.sum())}"
        )
        print(
            f"distinct live release timestamps: "
            f"{live_df['release_datetime_et'].nunique():,}"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calendar-csv", type=Path, default=DEFAULT_CALENDAR_CSV)
    p.add_argument("--out-all", type=Path, default=DEFAULT_OUT_ALL)
    p.add_argument("--out-live", type=Path, default=DEFAULT_OUT_LIVE)
    p.add_argument("--live-start", type=str, default=DEFAULT_LIVE_START)
    args = p.parse_args(argv)

    live_start = pd.Timestamp(args.live_start)
    all_df = build_canonical(args.calendar_csv)
    live_df = all_df[all_df["release_datetime_et"] >= live_start].copy()

    args.out_all.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(args.out_all, index=False)
    live_df.to_csv(args.out_live, index=False)
    print(f"Wrote {args.out_all}  ({len(all_df):,} rows)")
    print(f"Wrote {args.out_live}  ({len(live_df):,} rows)")
    print()
    print_summary(all_df, live_df, live_start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
