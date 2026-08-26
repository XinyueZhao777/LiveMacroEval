"""Concatenate the two filtered-macro-events tables into a single long-history
CSV consumed by step_15_2 when --hist-start <= 2014-12-31.

Inputs (must already exist; both produced by sibling filter scripts):
  - filtered_macro_events_2010_2014.csv  (filter_macro_events_2010_2014.py)
  - filtered_macro_events.csv            (filter_macro_events.py)

Output:
  - filtered_macro_events_combined.csv

Schema is preserved from both inputs (they share columns by construction).
Re-run this any time either input is rebuilt — e.g. after a fresh scrape
or after Bloomberg consensus is merged into the upstream Investing CSV.

Conda env: livemacro.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
IN_2010_2014 = HERE / "filtered_macro_events_2010_2014.csv"
IN_2015_PLUS = HERE / "filtered_macro_events.csv"
OUT_COMBINED = HERE / "filtered_macro_events_combined.csv"


def main() -> int:
    if not IN_2010_2014.exists():
        raise FileNotFoundError(IN_2010_2014)
    if not IN_2015_PLUS.exists():
        raise FileNotFoundError(IN_2015_PLUS)

    df_old = pd.read_csv(IN_2010_2014, dtype={"event_id": str, "event_attr_id": str})
    df_new = pd.read_csv(IN_2015_PLUS, dtype={"event_id": str, "event_attr_id": str})
    print(f"  2010-2014 rows: {len(df_old):,}")
    print(f"  2015+    rows: {len(df_new):,}")

    if list(df_old.columns) != list(df_new.columns):
        only_old = [c for c in df_old.columns if c not in df_new.columns]
        only_new = [c for c in df_new.columns if c not in df_old.columns]
        raise RuntimeError(
            f"Column mismatch between inputs.\n"
            f"  Only in 2010-2014: {only_old}\n"
            f"  Only in 2015+:     {only_new}"
        )

    combined = pd.concat([df_old, df_new], ignore_index=True)
    combined = combined.sort_values(
        ["release_datetime_et", "event"], kind="mergesort"
    ).reset_index(drop=True)

    combined.to_csv(OUT_COMBINED, index=False)
    print(f"Wrote {len(combined):,} rows -> {OUT_COMBINED.name}")
    print(
        f"  date range: {combined['release_datetime_et'].min()} -> "
        f"{combined['release_datetime_et'].max()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
