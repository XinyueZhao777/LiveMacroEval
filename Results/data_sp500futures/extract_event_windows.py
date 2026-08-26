"""Extract ES 1-min bars for the announcement windows required by the MSC
score (Project Outline section 8).

For each unique release timestamp T in filtered_macro_events.csv:

  window    = [T - 10 min, T + 35 min]  (widened read band)
  target    = minute containing T - 5 min  and  minute containing T + 30 min
  tolerance = up to 5 min backward slack on each endpoint

Contract selection follows FirstRateData's active-contract calendar
(ES_all_contracts_dates.txt). This implements the Outline's requirement
to "use the active front-month contract at timestamp T_g."

Outputs (all in this directory):

  event_windows_1min.parquet   long, one row per (event, bar) pair,
                               columns: event_release_timestamp_et,
                               contract_code, bar_timestamp_et, offset_sec,
                               open, high, low, close, volume
  event_window_coverage.csv    one row per unique event timestamp, with
                               per-endpoint match results and validity
  sanity_report.txt            extraction-time QC summary

Timezone: FirstRateData ES minute bars are in US/Eastern clock time. The
event file column release_datetime_et is also in ET. We work entirely in
naive ET datetimes, which is the canonical MSC join key.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
# Source of unique event release timestamps. We use the fresh ground-truth
# field-release table (one row per first-release per reference period) rather
# than the stale 2025-04 filtered_macro_events.csv snapshot so that coverage
# always tracks the latest ground truth. The downstream consumers (live
# scoring, historical regression) only score first releases, so restricting
# to first releases here is exactly correct. See
# Results/ground_truth/parse_ground_truth.py for how this file is produced.
EVENTS_CSV = (
    HERE.parent / "ground_truth" / "data" / "field_releases_all.csv"
)
ARCHIVE_ZIP = HERE / "ES_1min_archive_t6h13g.zip"
# UPDATE_ZIP / DATES_ZIP are picked up by glob at run time so each refresh
# automatically uses the newest dated zip the user dropped into this folder
# (e.g. ES_1min_update_t6h13g_0501.zip overrides the earlier
# ES_1min_update_t6h13g.zip without an edit).
UPDATE_GLOBS = ("ES_1min_update_t6h13g*.zip",)
DATES_GLOBS = ("ES_all_contracts_dates*.zip",)


def _newest_match(globs: tuple[str, ...]) -> Path:
    candidates: list[Path] = []
    for pat in globs:
        candidates.extend(HERE.glob(pat))
    if not candidates:
        raise FileNotFoundError(
            f"No file in {HERE} matches {' / '.join(globs)}; cannot locate "
            f"the update zip. Drop the latest FirstRateData ES download here."
        )
    # Newest by mtime — same data vendor, same naming convention; the most
    # recently written file is always the latest snapshot.
    return max(candidates, key=lambda p: p.stat().st_mtime)

OUT_PARQUET = HERE / "event_windows_1min.parquet"
OUT_COVERAGE = HERE / "event_window_coverage.csv"
OUT_REPORT = HERE / "sanity_report.txt"

# Window band and endpoint tolerance (Project Outline section 8).
PRE_MIN = pd.Timedelta("5min")          # T - 5m
POST_MIN = pd.Timedelta("30min")        # T + 30m
READ_PAD = pd.Timedelta("5min")         # widen read band by 5 min each side
TOL = pd.Timedelta("5min")              # endpoint slack

MONTH_LETTER = {"H": 3, "M": 6, "U": 9, "Z": 12}


def contract_index_from_zips(update_zip: Path) -> dict[str, tuple[Path, str]]:
    out: dict[str, tuple[Path, str]] = {}
    # Iterate archive first, then update — update entries override archive
    # entries when both contain the same contract code, so contracts present
    # in both zips resolve to the more recent (update) bars.
    for zp in (ARCHIVE_ZIP, update_zip):
        with zipfile.ZipFile(zp) as zf:
            for name in zf.namelist():
                stem = name.replace("ES_", "").replace("_1min.txt", "")
                if len(stem) == 3 and stem[0] in MONTH_LETTER:
                    out[stem] = (zp, name)
    return out


def read_contract(zp: Path, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(zp) as zf:
        raw = zf.read(member)
    df = pd.read_csv(
        io.BytesIO(raw),
        header=None,
        names=["ts", "open", "high", "low", "close", "volume"],
        dtype={
            "open": np.float32,
            "high": np.float32,
            "low": np.float32,
            "close": np.float32,
            "volume": np.int64,
        },
    )
    df["ts"] = pd.to_datetime(df["ts"], format="%Y-%m-%d %H:%M:%S")
    df.sort_values("ts", inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)
    return df


def load_lookup(dates_zip: Path) -> pd.Series:
    with zipfile.ZipFile(dates_zip) as zf:
        raw = zf.read("ES_all_contracts_dates.txt")
    d = pd.read_csv(io.BytesIO(raw), header=None, names=["date", "contract"],
                    dtype=str)
    d["date"] = pd.to_datetime(d["date"]).dt.date
    return d.set_index("date")["contract"]


def find_endpoint(
    df: pd.DataFrame, target_minute: pd.Timestamp, tol: pd.Timedelta
) -> tuple[int | None, pd.Timestamp | None]:
    """Return (row_index, bar_ts) for the latest bar with
    ts <= target_minute and ts > target_minute - tol. None if no bar in the
    tolerance band. df must be sorted by ts."""
    ts = df["ts"].values  # numpy datetime64[ns]
    # Latest ts <= target_minute
    idx = np.searchsorted(ts, np.datetime64(target_minute), side="right") - 1
    if idx < 0:
        return None, None
    cand = df["ts"].iat[idx]
    if cand < target_minute - tol:
        return None, None
    return int(idx), cand


def main() -> None:
    report: list[str] = []
    report.append("=" * 78)
    report.append("ES event-window extraction — sanity report")
    report.append("=" * 78)

    # 1. Load events and contract lookup.
    update_zip = _newest_match(UPDATE_GLOBS)
    dates_zip = _newest_match(DATES_GLOBS)
    report.append(f"archive zip: {ARCHIVE_ZIP.name}")
    report.append(f"update zip:  {update_zip.name}  (auto-selected: newest match)")
    report.append(f"dates zip:   {dates_zip.name}   (auto-selected: newest match)")

    ev = pd.read_csv(EVENTS_CSV, usecols=["release_datetime_et"])
    ev["release_datetime_et"] = pd.to_datetime(ev["release_datetime_et"])
    unique_ts = (
        ev["release_datetime_et"].drop_duplicates().sort_values().reset_index(drop=True)
    )
    report.append(
        f"unique event release timestamps: {len(unique_ts):,}  "
        f"({unique_ts.min()} .. {unique_ts.max()})"
    )

    lookup = load_lookup(dates_zip)
    report.append(
        f"contract-date lookup rows: {len(lookup):,}  "
        f"({lookup.index.min()} .. {lookup.index.max()})"
    )

    idx = contract_index_from_zips(update_zip)
    report.append(f"contract files available: {len(idx)}")

    # 2. Map each event to its front-month contract.
    ev_df = pd.DataFrame({"release_datetime_et": unique_ts})
    ev_df["date"] = ev_df["release_datetime_et"].dt.date
    ev_df["contract_code"] = ev_df["date"].map(lookup)
    unmapped = ev_df[ev_df["contract_code"].isna()]
    report.append(f"events with no mapped contract (date not in lookup): {len(unmapped)}")
    if len(unmapped):
        report.append(unmapped.to_string(index=False))

    # Endpoint targets.
    ev_df["t_minus_5_minute"] = (ev_df["release_datetime_et"] - PRE_MIN).dt.floor("min")
    ev_df["t_plus_30_minute"] = (ev_df["release_datetime_et"] + POST_MIN).dt.floor("min")
    ev_df["window_lo"] = ev_df["t_minus_5_minute"] - READ_PAD
    ev_df["window_hi"] = ev_df["t_plus_30_minute"] + READ_PAD

    # 3. Group by contract to minimise file reads.
    coverage_rows: list[dict] = []
    bar_frames: list[pd.DataFrame] = []
    contracts_used: list[str] = []
    for ct, grp in ev_df.dropna(subset=["contract_code"]).groupby("contract_code"):
        zp, member = idx[ct]
        df = read_contract(zp, member)
        contracts_used.append(ct)
        ts = df["ts"].values

        for _, r in grp.iterrows():
            lo = np.datetime64(r["window_lo"])
            hi = np.datetime64(r["window_hi"])
            i0 = int(np.searchsorted(ts, lo, side="left"))
            i1 = int(np.searchsorted(ts, hi, side="right"))
            sub = df.iloc[i0:i1]

            # Endpoint matching within the subslice.
            s_idx, s_ts = find_endpoint(sub, r["t_minus_5_minute"], TOL)
            e_idx, e_ts = find_endpoint(sub, r["t_plus_30_minute"], TOL)

            def close_at(ix):
                return None if ix is None else float(sub["close"].iat[ix])

            s_close = close_at(s_idx)
            e_close = close_at(e_idx)
            valid = s_ts is not None and e_ts is not None

            coverage_rows.append(
                {
                    "release_datetime_et": r["release_datetime_et"],
                    "contract_code": ct,
                    "t_minus_5_minute": r["t_minus_5_minute"],
                    "t_plus_30_minute": r["t_plus_30_minute"],
                    "start_bar_ts": s_ts,
                    "start_bar_close": s_close,
                    "start_offset_sec": (
                        None
                        if s_ts is None
                        else int((s_ts - r["t_minus_5_minute"]).total_seconds())
                    ),
                    "end_bar_ts": e_ts,
                    "end_bar_close": e_close,
                    "end_offset_sec": (
                        None
                        if e_ts is None
                        else int((e_ts - r["t_plus_30_minute"]).total_seconds())
                    ),
                    "n_bars_in_window": int(len(sub)),
                    "valid": bool(valid),
                }
            )

            if len(sub):
                out = sub.copy()
                out["event_release_timestamp_et"] = r["release_datetime_et"]
                out["contract_code"] = ct
                out["offset_sec"] = (
                    (out["ts"] - r["release_datetime_et"]).dt.total_seconds().astype(np.int64)
                )
                out.rename(columns={"ts": "bar_timestamp_et"}, inplace=True)
                bar_frames.append(
                    out[
                        [
                            "event_release_timestamp_et",
                            "contract_code",
                            "bar_timestamp_et",
                            "offset_sec",
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                        ]
                    ]
                )

    # Add un-mappable events to coverage with valid=False.
    for _, r in unmapped.iterrows():
        coverage_rows.append(
            {
                "release_datetime_et": r["release_datetime_et"],
                "contract_code": None,
                "t_minus_5_minute": None,
                "t_plus_30_minute": None,
                "start_bar_ts": None,
                "start_bar_close": None,
                "start_offset_sec": None,
                "end_bar_ts": None,
                "end_bar_close": None,
                "end_offset_sec": None,
                "n_bars_in_window": 0,
                "valid": False,
            }
        )

    coverage = pd.DataFrame(coverage_rows).sort_values("release_datetime_et")
    coverage.to_csv(OUT_COVERAGE, index=False)

    if bar_frames:
        bars = pd.concat(bar_frames, ignore_index=True)
        bars.sort_values(
            ["event_release_timestamp_et", "bar_timestamp_et"],
            kind="mergesort",
            inplace=True,
        )
        bars.reset_index(drop=True, inplace=True)
        bars.to_parquet(OUT_PARQUET, index=False, compression="zstd")
        report.append(
            f"bar rows written: {len(bars):,}  "
            f"events represented: {bars['event_release_timestamp_et'].nunique():,}"
        )
    else:
        report.append("no bar rows produced")

    # Validation summary.
    n_total = len(coverage)
    n_valid = int(coverage["valid"].sum())
    n_invalid_unmapped = int(coverage["contract_code"].isna().sum())
    n_invalid_missing_ep = n_total - n_valid - n_invalid_unmapped
    report.append(f"\nevents total / valid / invalid-unmapped / invalid-missing-endpoint:")
    report.append(
        f"  {n_total:,} / {n_valid:,} / {n_invalid_unmapped:,} / {n_invalid_missing_ep:,}"
    )

    # Endpoint offset distribution (valid events only).
    valid_cov = coverage[coverage["valid"]]
    for col in ["start_offset_sec", "end_offset_sec"]:
        q = valid_cov[col].quantile([0.0, 0.5, 0.95, 0.99, 1.0])
        report.append(
            f"\n{col} percentiles (0/50/95/99/100): "
            + ", ".join(f"{int(v)}" for v in q.values)
        )
        exact = int((valid_cov[col] == 0).sum())
        report.append(
            f"  bars landing on exact target minute: {exact:,}/{len(valid_cov):,} "
            f"({exact / max(1, len(valid_cov)):.1%})"
        )

    # Per-contract coverage.
    by_ct = (
        coverage.dropna(subset=["contract_code"])
        .groupby("contract_code")
        .agg(
            n_events=("release_datetime_et", "size"),
            n_valid=("valid", "sum"),
            first=("release_datetime_et", "min"),
            last=("release_datetime_et", "max"),
        )
        .sort_values("first")
    )
    report.append("\nper-contract event coverage:")
    report.append(by_ct.to_string())

    # Sanity: invalid events detail (first 20).
    invalid = coverage[~coverage["valid"]]
    report.append(f"\ninvalid events ({len(invalid)}), first 20:")
    report.append(invalid.head(20).to_string(index=False))

    # Log return quick view on valid events for a final plausibility check.
    if n_valid > 0:
        lr = np.log(valid_cov["end_bar_close"] / valid_cov["start_bar_close"])
        report.append(
            f"\nlog-return(T+30 close / T-5 close) over valid events: "
            f"n={len(lr):,}, mean={lr.mean():+.6f}, std={lr.std():.6f}, "
            f"min={lr.min():+.6f}, max={lr.max():+.6f}, "
            f"|r|>0.02: {(lr.abs() > 0.02).sum()}"
        )

    OUT_REPORT.write_text("\n".join(str(x) for x in report) + "\n")
    print("\n".join(str(x) for x in report))
    print(f"\nWrote:\n  {OUT_PARQUET}\n  {OUT_COVERAGE}\n  {OUT_REPORT}")


if __name__ == "__main__":
    main()
