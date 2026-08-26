#!/usr/bin/env python3
"""Concatenate CSV files from serverA and serverB data folders.

=== DATA PIPELINE STEP 2 of 3 ===
Run scripts in this order:
  1. python data_from_serverA_serverB/prepare_qwen_data.py
     (Convert raw Qwen JSON files into intermediate CSVs)
  2. python data_from_serverA_serverB/concat_server_data.py  [THIS SCRIPT]
     (Concatenate serverA/serverB CSVs for non-Qwen models into the intermediate folder)
  3. python data_from_serverA_serverB/concat_model_data.py
     (Combine intermediate sources across time windows into final_analysis_data)

This script does not regenerate Qwen intermediate CSVs. Run step 1 first when
you need refreshed Qwen inputs.

Usage:
    python data_from_serverA_serverB/concat_server_data.py
"""

from collections import defaultdict
from pathlib import Path
import re

import pandas as pd

from data_pipeline_utils import resolve_latest_dated_dir

# Paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "concatenated_serverA_serverB"
EXPECTED_CATEGORIES = {"core_macroeconomic_conditions", "demand_sectoral_activity"}


def get_base_filename(filename):
    """
    Extract the base filename without the server suffix.
    e.g., '2025-12_core_macroeconomic_conditions_serverA.csv' -> '2025-12_core_macroeconomic_conditions'
    """
    name = filename.replace('.csv', '')
    name = re.sub(r'_server[AB]$', '', name)
    return name


def _extract_month_and_category(base_name: str):
    """Extract (target_month, category) from '<YYYY-MM>_<category>'."""
    match = re.match(r"^(\d{4}-\d{2})_(.+)$", base_name)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def concat_csv_files():
    """Concatenate matching CSV files from serverA and serverB."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        data_servera = resolve_latest_dated_dir(BASE_DIR, "data_serverA")
        data_serverb = resolve_latest_dated_dir(BASE_DIR, "data_serverB")
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return

    print(f"Using serverA input: {data_servera.relative_to(BASE_DIR)}")
    print(f"Using serverB input: {data_serverb.relative_to(BASE_DIR)}")

    model_names = set()
    if data_servera.exists():
        model_names |= {
            p.name for p in data_servera.iterdir() if p.is_dir() and p.name.startswith("model_")
        }
    if data_serverb.exists():
        model_names |= {
            p.name for p in data_serverb.iterdir() if p.is_dir() and p.name.startswith("model_")
        }

    if not model_names:
        print(f"No model folders found in {data_servera} or {data_serverb}")

    for model_name in sorted(model_names):
        print(f"\nProcessing {model_name}...")

        output_model_dir = OUTPUT_DIR / model_name
        output_model_dir.mkdir(parents=True, exist_ok=True)
        for existing_csv in sorted(output_model_dir.glob("*.csv")):
            existing_csv.unlink()

        server_a_dir = data_servera / model_name
        server_b_dir = data_serverb / model_name

        server_a_files = {}
        if server_a_dir.exists():
            server_a_files = {
                get_base_filename(p.name): p for p in sorted(server_a_dir.glob("*.csv"))
            }

        server_b_files = {}
        if server_b_dir.exists():
            server_b_files = {
                get_base_filename(p.name): p for p in sorted(server_b_dir.glob("*.csv"))
            }

        all_base_names = sorted(set(server_a_files) | set(server_b_files))
        if not all_base_names:
            print("  No CSV files found for this model.")
            continue

        month_category_coverage = defaultdict(set)

        for base_name in all_base_names:
            server_a_file = server_a_files.get(base_name)
            server_b_file = server_b_files.get(base_name)
            print(f"  Concatenating: {base_name}")

            dfs = []
            if server_a_file is not None:
                df_a = pd.read_csv(server_a_file)
                dfs.append(df_a)
                print(f"    serverA rows: {len(df_a)}")
            else:
                print(f"    Warning: Missing serverA file for {base_name}")

            if server_b_file is not None:
                df_b = pd.read_csv(server_b_file)
                dfs.append(df_b)
                print(f"    serverB rows: {len(df_b)}")
            else:
                print(f"    Warning: Missing serverB file for {base_name}")

            if not dfs:
                print(f"    Skipping {base_name} (no input rows)")
                continue

            df_combined = pd.concat(dfs, ignore_index=True)

            timestamp_cols = [col for col in df_combined.columns if "timestamp" in col.lower()]
            if timestamp_cols:
                ts_col = timestamp_cols[0]
                sort_ts = pd.to_datetime(df_combined[ts_col], errors="coerce")
                df_combined = (
                    df_combined.assign(__sort_ts=sort_ts)
                    .sort_values(by=["__sort_ts", ts_col], kind="mergesort")
                    .drop(columns="__sort_ts")
                )

            before_dedup = len(df_combined)
            df_combined = df_combined.drop_duplicates()
            after_dedup = len(df_combined)
            if before_dedup != after_dedup:
                print(f"    Removed {before_dedup - after_dedup} duplicate rows")

            # Keep _serverA suffix for compatibility with downstream scripts.
            output_file = output_model_dir / f"{base_name}_serverA.csv"
            df_combined.to_csv(output_file, index=False)
            print(f"    Saved: {output_file.name} ({len(df_combined)} total rows)")

            month, category = _extract_month_and_category(base_name)
            if month and category:
                month_category_coverage[month].add(category)

        print("  Coverage summary by target_month:")
        if not month_category_coverage:
            print("    (no parseable YYYY-MM_<category> files)")
        for month in sorted(month_category_coverage):
            categories = month_category_coverage[month]
            cat_list = ", ".join(sorted(categories))
            missing = sorted(EXPECTED_CATEGORIES - categories)
            if missing:
                print(f"    {month}: {cat_list} | missing: {', '.join(missing)}")
            else:
                print(f"    {month}: {cat_list}")

    print(f"\n{'='*60}")
    print(f"Done! Concatenated files saved to: {OUTPUT_DIR}")
    print("Qwen intermediate CSVs are not refreshed here; run prepare_qwen_data.py first if needed.")
    print(f"\nTo generate plots from combined data, run:")
    print(f"  python remove_outlier_and_plot/generate_plots.py --data-source combined")


if __name__ == "__main__":
    concat_csv_files()
