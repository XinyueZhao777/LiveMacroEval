#!/usr/bin/env python3
"""Concatenate model data CSVs across multiple folders with timestamp cutoffs.

=== DATA PIPELINE STEP 3 of 3 ===
Run scripts in this order:
  1. python data_from_serverA_serverB/prepare_qwen_data.py
     (Convert raw Qwen JSON files into intermediate CSVs)
  2. python data_from_serverA_serverB/concat_server_data.py
     (Concatenate serverA/serverB CSVs for non-Qwen models into the intermediate folder)
  3. python data_from_serverA_serverB/concat_model_data.py  [THIS SCRIPT]
     (Combine intermediate sources across time windows into final_analysis_data)

This script handles:
1. GPT model: concatenating data from gpt-data-1111-1206, gpt-data-1206-0106,
   data/data-0105-0218/data_serverA & data_serverB (model_gpt-5-search-api),
   and every grouped CSV discovered in
   data_from_serverA_serverB/concatenated_serverA_serverB/model_gpt-5-search-api
2. Claude model: data/data-0105-0218/data_serverA & data_serverB (model_claude-sonnet-4.5-api),
   then every grouped CSV discovered in
   data_from_serverA_serverB/concatenated_serverA_serverB/model_claude-sonnet-4.5-api
3. Qwen models: dynamically discovered intermediate CSVs generated in step 1 by
   data_from_serverA_serverB/prepare_qwen_data.py
4. Any additional non-Qwen model folders found in concatenated_serverA_serverB
   are processed directly from those intermediate CSVs.

Programmatic check confirms every final file’s variable set matches backend group definitions exactly.
Created range log: variable_prediction_ranges.csv
"""

from __future__ import annotations

import argparse
import importlib.util
from collections import defaultdict
from pathlib import Path
import sys
from typing import Optional

import pandas as pd

from data_pipeline_utils import discover_grouped_csv_patterns

# Project root: go up 1 level from this script's location
# (script is in data_from_serverA_serverB/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CATEGORIES = ("core_macroeconomic_conditions", "demand_sectoral_activity")

LEGACY_VARIABLE_ALIASES = {
    "building_permits": "building_permits_level",
    "housing_starts": "housing_starts_level",
    "nonfarm_payrolls_level_thous": "nonfarm_payrolls_level",
    "nonfarm_payrolls_change_thous": "nonfarm_payrolls_change",
}

# Configuration for historical data sources. Newly added grouped CSVs in
# data_from_serverA_serverB/concatenated_serverA_serverB are discovered at runtime.
# Each source has: folder path, file patterns for each (target_month, category), and cutoff range.
GPT_LEGACY_DATA_SOURCES = [
    {
        "name": "gpt-data-1111-1206",
        "folder": "data/gpt-data-1111-1206",
        "file_patterns": {
            # (target_month, category) -> filename
            ("2025-11", "core_macroeconomic_conditions"): "2025-11.csv",
            ("2025-11", "demand_sectoral_activity"): "2025-11.csv",
            ("2025-12", "core_macroeconomic_conditions"): "2025-12.csv",
            ("2025-12", "demand_sectoral_activity"): "2025-12.csv",
        },
        "cutoff_end": "2025-12-06 00:00:00",  # Use data before this timestamp
        "ungrouped_legacy": True,
    },
    {
        "name": "gpt-data-1206-0106",
        "folder": "data/gpt-data-1206-0106",
        "file_patterns": {
            ("2025-11", "core_macroeconomic_conditions"): "2025-11_core_macroeconomic_conditions.csv",
            ("2025-11", "demand_sectoral_activity"): "2025-11_demand_sectoral_activity.csv",
            ("2025-12", "core_macroeconomic_conditions"): "2025-12_core_macroeconomic_conditions.csv",
            ("2025-12", "demand_sectoral_activity"): "2025-12_demand_sectoral_activity.csv",
        },
        "cutoff_start": "2025-12-06 00:00:00",
        "cutoff_end": "2026-01-06 00:00:00",
    },
    {
        "name": "data_0105_0218_serverA_gpt",
        "folder": "data/data-0105-0218/data_serverA/model_gpt-5-search-api",
        "file_patterns": {
            ("2025-11", "core_macroeconomic_conditions"): "2025-11_core_macroeconomic_conditions_serverA.csv",
            ("2025-11", "demand_sectoral_activity"): "2025-11_demand_sectoral_activity_serverA.csv",
            ("2025-12", "core_macroeconomic_conditions"): "2025-12_core_macroeconomic_conditions_serverA.csv",
            ("2025-12", "demand_sectoral_activity"): "2025-12_demand_sectoral_activity_serverA.csv",
            ("2026-01", "core_macroeconomic_conditions"): "2026-01_core_macroeconomic_conditions_serverA.csv",
            ("2026-01", "demand_sectoral_activity"): "2026-01_demand_sectoral_activity_serverA.csv",
        },
        "cutoff_start": "2026-01-06 00:00:00",
        "cutoff_end": "2026-02-19 00:00:00",
    },
    {
        "name": "data_0105_0218_serverB_gpt",
        "folder": "data/data-0105-0218/data_serverB/model_gpt-5-search-api",
        "file_patterns": {
            ("2025-11", "core_macroeconomic_conditions"): "2025-11_core_macroeconomic_conditions_serverB.csv",
            ("2025-11", "demand_sectoral_activity"): "2025-11_demand_sectoral_activity_serverB.csv",
            ("2025-12", "core_macroeconomic_conditions"): "2025-12_core_macroeconomic_conditions_serverB.csv",
            ("2025-12", "demand_sectoral_activity"): "2025-12_demand_sectoral_activity_serverB.csv",
            ("2026-01", "core_macroeconomic_conditions"): "2026-01_core_macroeconomic_conditions_serverB.csv",
            ("2026-01", "demand_sectoral_activity"): "2026-01_demand_sectoral_activity_serverB.csv",
        },
        "cutoff_start": "2026-01-06 00:00:00",
        "cutoff_end": "2026-02-19 00:00:00",
    },
]

CLAUDE_LEGACY_DATA_SOURCES = [
    {
        "name": "data_0105_0218_serverA_claude",
        "folder": "data/data-0105-0218/data_serverA/model_claude-sonnet-4.5-api",
        "file_patterns": {
            ("2025-12", "core_macroeconomic_conditions"): "2025-12_core_macroeconomic_conditions_serverA.csv",
            ("2025-12", "demand_sectoral_activity"): "2025-12_demand_sectoral_activity_serverA.csv",
            ("2026-01", "core_macroeconomic_conditions"): "2026-01_core_macroeconomic_conditions_serverA.csv",
            ("2026-01", "demand_sectoral_activity"): "2026-01_demand_sectoral_activity_serverA.csv",
        },
        "cutoff_end": "2026-02-19 00:00:00",
    },
    {
        "name": "data_0105_0218_serverB_claude",
        "folder": "data/data-0105-0218/data_serverB/model_claude-sonnet-4.5-api",
        "file_patterns": {
            ("2025-12", "core_macroeconomic_conditions"): "2025-12_core_macroeconomic_conditions_serverB.csv",
            ("2025-12", "demand_sectoral_activity"): "2025-12_demand_sectoral_activity_serverB.csv",
            ("2026-01", "core_macroeconomic_conditions"): "2026-01_core_macroeconomic_conditions_serverB.csv",
            ("2026-01", "demand_sectoral_activity"): "2026-01_demand_sectoral_activity_serverB.csv",
        },
        "cutoff_end": "2026-02-19 00:00:00",
    },
]

# Output configurations
OUTPUT_DIR = "data_from_serverA_serverB/final_analysis_data"
TIMESTAMP_COL = "timestamp_local"
RANGE_LOG_FILE = "variable_prediction_ranges.csv"
INTERMEDIATE_ROOT = "data_from_serverA_serverB/concatenated_serverA_serverB"
NON_QWEN_INTERMEDIATE_CUTOFFS = {
    "gpt-5-search-api": "2026-02-19 00:00:00",
    "claude-sonnet-4.5-api": "2026-02-19 00:00:00",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Concatenate model data CSVs from multiple folders with timestamp cutoffs. "
            "Supports GPT, Claude, and dynamically discovered Qwen sources. "
            "Validates variable groups and writes variable time-range logs."
        )
    )
    parser.add_argument(
        "--model",
        choices=["gpt", "claude", "qwen", "all"],
        default="all",
        help="Which model to process (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help=f"Output folder for concatenated CSVs (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--timestamp-col",
        default=TIMESTAMP_COL,
        help=f"Name of the timestamp column (default: {TIMESTAMP_COL}).",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        default=True,
        help="Sort output by timestamp column after concatenation (default: True).",
    )
    parser.add_argument(
        "--base-dir",
        default=str(PROJECT_ROOT),
        help=f"Base directory for relative paths (default: {PROJECT_ROOT}).",
    )
    parser.add_argument(
        "--range-log-file",
        default=RANGE_LOG_FILE,
        help=f"CSV filename for variable prediction ranges (default: {RANGE_LOG_FILE}).",
    )
    return parser.parse_args()


def _load_backend_group_variables(base_dir: Path) -> dict[str, set[str]]:
    """Load authoritative variable group definitions from backend/variables.py."""
    module_path = base_dir.parent / "LiveMacro_serverA" / "backend" / "variables.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Backend variable definition file not found: {module_path}")

    spec = importlib.util.spec_from_file_location("backend_variables", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load backend variables module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "VARIABLES"):
        raise AttributeError(f"{module_path} does not define VARIABLES")

    variables = getattr(module, "VARIABLES")
    group_map: dict[str, set[str]] = {}
    for group_name, defs in variables.items():
        keys = {item["key"] for item in defs if isinstance(item, dict) and "key" in item}
        group_map[group_name] = keys
    return group_map


def _load_csv(path: Path, timestamp_col: str) -> pd.DataFrame:
    """Load a CSV file and validate timestamp column exists."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if timestamp_col not in df.columns:
        raise ValueError(f"Missing column '{timestamp_col}' in {path}")
    return df


def _discover_qwen_data_sources(base_dir: Path) -> dict[str, list[dict]]:
    """Discover dynamically generated Qwen intermediate CSV files by model folder."""
    root = base_dir / INTERMEDIATE_ROOT
    if not root.exists():
        return {}

    discovered: dict[str, list[dict]] = {}

    for model_dir in sorted(root.glob("model_qwen*")):
        if not model_dir.is_dir():
            continue

        file_patterns = discover_grouped_csv_patterns(model_dir, CATEGORIES)
        if not file_patterns:
            continue

        model_name = model_dir.name.removeprefix("model_")
        discovered[model_name] = [
            {
                "name": f"qwen_intermediate_{model_name}",
                "folder": str(model_dir.relative_to(base_dir)),
                "file_patterns": file_patterns,
            }
        ]

    return discovered


def _discover_non_qwen_data_sources(base_dir: Path) -> dict[str, list[dict]]:
    """Discover grouped intermediate CSV files for non-Qwen models."""
    root = base_dir / INTERMEDIATE_ROOT
    if not root.exists():
        return {}

    discovered: dict[str, list[dict]] = {}
    legacy_sources_by_model = {
        "gpt-5-search-api": GPT_LEGACY_DATA_SOURCES,
        "claude-sonnet-4.5-api": CLAUDE_LEGACY_DATA_SOURCES,
    }

    for model_dir in sorted(root.glob("model_*")):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name.removeprefix("model_")
        if model_name.startswith("qwen"):
            continue

        file_patterns = discover_grouped_csv_patterns(model_dir, CATEGORIES)
        if not file_patterns:
            continue

        source = {
            "name": f"intermediate_{model_name}",
            "folder": str(model_dir.relative_to(base_dir)),
            "file_patterns": file_patterns,
        }

        cutoff_start = NON_QWEN_INTERMEDIATE_CUTOFFS.get(model_name)
        if cutoff_start is not None:
            source["cutoff_start"] = cutoff_start

        discovered[model_name] = [*legacy_sources_by_model.get(model_name, []), source]

    return discovered


def _filter_by_cutoff(
    df: pd.DataFrame,
    timestamp_col: str,
    cutoff_start: Optional[pd.Timestamp] = None,
    cutoff_end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Filter dataframe by timestamp range [cutoff_start, cutoff_end)."""
    if df.empty:
        return df

    ts = pd.to_datetime(df[timestamp_col], errors="coerce")
    missing = ts.isna().sum()
    if missing:
        print(
            f"Warning: {missing} rows have invalid timestamps in {timestamp_col}.",
            file=sys.stderr,
        )

    mask = ~ts.isna()
    if cutoff_start is not None:
        mask &= ts >= cutoff_start
    if cutoff_end is not None:
        mask &= ts < cutoff_end

    return df.loc[mask].copy()


def _align_columns(dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Align columns across multiple dataframes, filling missing with NaN."""
    if not dfs:
        return dfs

    # Collect all columns preserving order
    all_cols = []
    for df in dfs:
        for col in df.columns:
            if col not in all_cols:
                all_cols.append(col)

    return [df.reindex(columns=all_cols) for df in dfs]


def _collect_reference_variables(
    data_sources: list[dict],
    base_dir: Path,
    timestamp_col: str,
) -> dict[str, set[str]]:
    """Collect per-category variables from grouped (post-1206) files."""
    references: defaultdict[str, set[str]] = defaultdict(set)

    for source in data_sources:
        if source.get("ungrouped_legacy", False):
            continue
        folder = base_dir / source["folder"]
        for (_, category), filename in source["file_patterns"].items():
            filepath = folder / filename
            if not filepath.exists():
                continue
            df = _load_csv(filepath, timestamp_col)
            if df.empty or "variable" not in df.columns:
                continue
            vars_in_file = (
                df["variable"].dropna().astype(str).str.strip()
            )
            references[category].update(v for v in vars_in_file if v)

    return dict(references)


def _canonicalize_legacy_variable(variable: str, allowed_variables: set[str]) -> Optional[str]:
    """Map legacy variable names to canonical group variables if possible."""
    value = variable.strip()
    if not value:
        return None
    if value in allowed_variables:
        return value

    alias = LEGACY_VARIABLE_ALIASES.get(value)
    if alias and alias in allowed_variables:
        return alias

    # Heuristic fallback for similar legacy naming patterns.
    if value.endswith("_thous"):
        candidate = value[: -len("_thous")]
        if candidate in allowed_variables:
            return candidate

    candidate = f"{value}_level"
    if candidate in allowed_variables:
        return candidate

    return None


def _filter_to_group_variables(
    df: pd.DataFrame,
    allowed_variables: set[str],
    legacy_mode: bool,
) -> tuple[pd.DataFrame, int, set[str]]:
    """Filter rows to group-allowed variables, with legacy name normalization if needed."""
    if df.empty or "variable" not in df.columns or not allowed_variables:
        return df, 0, set()

    raw_vars = df["variable"].astype(str).str.strip()
    if legacy_mode:
        normalized = raw_vars.map(lambda x: _canonicalize_legacy_variable(x, allowed_variables))
        keep_mask = normalized.notna()
        filtered = df.loc[keep_mask].copy()
        filtered["variable"] = normalized.loc[keep_mask].astype(str).values
    else:
        keep_mask = raw_vars.isin(allowed_variables)
        filtered = df.loc[keep_mask].copy()
        filtered["variable"] = raw_vars.loc[keep_mask].values

    dropped_rows = int((~keep_mask).sum())
    dropped_vars = set(raw_vars.loc[~keep_mask].dropna().unique())
    return filtered, dropped_rows, dropped_vars


def _build_variable_range_records(
    combined: pd.DataFrame,
    model_name: str,
    target_month: str,
    category: str,
    output_file: str,
    timestamp_col: str,
) -> list[dict]:
    """Create per-variable min/max timestamp records for logging."""
    if combined.empty or "variable" not in combined.columns or timestamp_col not in combined.columns:
        return []

    with_ts = combined.copy()
    with_ts["__ts"] = pd.to_datetime(with_ts[timestamp_col], errors="coerce")
    with_ts["variable"] = with_ts["variable"].astype(str).str.strip()
    with_ts = with_ts.dropna(subset=["__ts", "variable"])
    if with_ts.empty:
        return []

    records = []
    for variable, var_df in with_ts.groupby("variable", sort=True):
        min_ts = var_df["__ts"].min()
        max_ts = var_df["__ts"].max()
        records.append(
            {
                "model": model_name,
                "target_month": target_month,
                "category": category,
                "variable": variable,
                "row_count": len(var_df),
                "prediction_start": min_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "prediction_end": max_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "output_file": output_file,
            }
        )
    return records


def _print_variable_validation(
    combined: pd.DataFrame,
    category: str,
    expected_variables_by_group: dict[str, set[str]],
) -> None:
    """Print final variable set and compare it with backend definitions."""
    if "variable" not in combined.columns:
        print("  Warning: output has no 'variable' column; skipping variable validation")
        return

    output_vars = sorted(
        v for v in combined["variable"].dropna().astype(str).str.strip().unique() if v
    )
    print(f"  Final variables ({len(output_vars)}): {', '.join(output_vars)}")

    expected = expected_variables_by_group.get(category, set())
    if not expected:
        print(f"  Warning: no backend variable definition for category '{category}'")
        return

    output_set = set(output_vars)
    missing = sorted(expected - output_set)
    unexpected = sorted(output_set - expected)

    if not missing and not unexpected:
        print(f"  Variable check vs backend: PASS ({len(expected)} expected variables)")
        return

    print("  Variable check vs backend: FAIL")
    if missing:
        print(f"    Missing variables: {', '.join(missing)}")
    if unexpected:
        print(f"    Unexpected variables: {', '.join(unexpected)}")


def process_model_data(
    model_name: str,
    data_sources: list[dict],
    output_dir: Path,
    timestamp_col: str,
    sort_output: bool,
    base_dir: Path,
    expected_variables_by_group: dict[str, set[str]],
    clean_output: bool = False,
) -> list[dict]:
    """Process and concatenate data for a single model."""
    print(f"\n{'='*60}")
    print(f"Processing {model_name.upper()} model data")
    print(f"{'='*60}")

    reference_variables = _collect_reference_variables(data_sources, base_dir, timestamp_col)
    allowed_variables_by_group: dict[str, set[str]] = {}
    for category in set(expected_variables_by_group) | set(reference_variables):
        allowed_variables_by_group[category] = (
            set(expected_variables_by_group.get(category, set()))
            | set(reference_variables.get(category, set()))
        )

    for category in CATEGORIES:
        allowed_count = len(allowed_variables_by_group.get(category, set()))
        print(f"Allowed variables for {category}: {allowed_count}")

    # Get all unique (target_month, category) combinations
    all_keys = set()
    for source in data_sources:
        all_keys.update(source["file_patterns"].keys())

    model_output_dir = output_dir / f"model_{model_name}"
    if clean_output and model_output_dir.exists():
        for existing_csv in sorted(model_output_dir.glob("*.csv")):
            existing_csv.unlink()
    model_output_dir.mkdir(parents=True, exist_ok=True)

    range_records: list[dict] = []

    for target_month, category in sorted(all_keys):
        print(f"\n--- {target_month} {category} ---")

        dfs_to_concat = []
        allowed_variables = allowed_variables_by_group.get(category, set())

        for source in data_sources:
            source_name = source["name"]
            folder = base_dir / source["folder"]
            file_patterns = source["file_patterns"]
            legacy_mode = source.get("ungrouped_legacy", False)

            # Check if this source has data for this (target_month, category)
            if (target_month, category) not in file_patterns:
                print(f"  [{source_name}] No data for this target_month/category")
                continue

            filename = file_patterns[(target_month, category)]
            filepath = folder / filename

            if not filepath.exists():
                print(f"  [{source_name}] File not found: {filepath}")
                continue

            df = _load_csv(filepath, timestamp_col)
            if df.empty:
                print(f"  [{source_name}] Empty file: {filepath}")
                continue

            # Parse cutoff timestamps
            cutoff_start = None
            cutoff_end = None
            if "cutoff_start" in source:
                cutoff_start = pd.to_datetime(source["cutoff_start"])
            if "cutoff_end" in source:
                cutoff_end = pd.to_datetime(source["cutoff_end"])

            # Filter by timestamp range
            original_len = len(df)
            df = _filter_by_cutoff(df, timestamp_col, cutoff_start, cutoff_end)
            post_cutoff_len = len(df)

            # Filter by variable group and normalize legacy variable names if needed
            dropped_vars = set()
            if allowed_variables and "variable" in df.columns:
                df, dropped_count, dropped_vars = _filter_to_group_variables(
                    df,
                    allowed_variables,
                    legacy_mode=legacy_mode,
                )
            else:
                dropped_count = 0

            cutoff_info = ""
            if cutoff_start or cutoff_end:
                start_str = str(cutoff_start)[:19] if cutoff_start is not None else "start"
                end_str = str(cutoff_end)[:19] if cutoff_end is not None else "end"
                cutoff_info = f" (cutoff: [{start_str}, {end_str}))"

            print(
                f"  [{source_name}] {len(df)}/{original_len} rows from {filename}{cutoff_info}"
            )
            if dropped_count:
                dropped_vars_str = ", ".join(sorted(dropped_vars))
                print(
                    f"    Dropped {dropped_count} rows not in '{category}'"
                    f" (after cutoff rows: {post_cutoff_len})."
                )
                if dropped_vars_str:
                    print(f"    Dropped variables: {dropped_vars_str}")

            if not df.empty:
                dfs_to_concat.append(df)

        if not dfs_to_concat:
            print(f"  No data to concatenate for {target_month} {category}")
            continue

        # Align columns and concatenate
        aligned_dfs = _align_columns(dfs_to_concat)
        combined = pd.concat(aligned_dfs, ignore_index=True)

        # Remove duplicates based on timestamp and variable
        if "variable" in combined.columns:
            before_dedup = len(combined)
            combined = combined.drop_duplicates(
                subset=[timestamp_col, "variable"],
                keep="last",
            )
            if before_dedup != len(combined):
                print(f"  Removed {before_dedup - len(combined)} duplicate rows")

            # Final strict filter to backend variable definitions.
            expected_for_category = expected_variables_by_group.get(category, set())
            if expected_for_category:
                before_strict = len(combined)
                combined, strict_dropped, strict_dropped_vars = _filter_to_group_variables(
                    combined,
                    expected_for_category,
                    legacy_mode=False,
                )
                if strict_dropped:
                    print(
                        f"  Removed {strict_dropped} rows with non-backend variables "
                        f"after concatenation."
                    )
                    print(
                        f"    Removed variables: {', '.join(sorted(strict_dropped_vars))}"
                    )
                if before_strict != len(combined):
                    print(f"  Rows after strict backend filter: {len(combined)}")

        if sort_output:
            sort_ts = pd.to_datetime(combined[timestamp_col], errors="coerce")
            combined = (
                combined.assign(__sort_ts=sort_ts)
                .sort_values(by=["__sort_ts", timestamp_col], kind="mergesort")
                .drop(columns="__sort_ts")
            )

        # Output filename
        out_filename = f"{target_month}_{category}.csv"
        out_path = model_output_dir / out_filename
        combined.to_csv(out_path, index=False)
        print(f"  -> Wrote {out_path} ({len(combined)} rows)")

        # Print variable list and verify against backend definitions
        _print_variable_validation(combined, category, expected_variables_by_group)

        # Print file-level timestamp range and append per-variable range records
        if timestamp_col in combined.columns:
            ts = pd.to_datetime(combined[timestamp_col], errors="coerce").dropna()
            if not ts.empty:
                print(
                    "  Timestamp coverage: "
                    f"{ts.min().strftime('%Y-%m-%d %H:%M:%S')} -> "
                    f"{ts.max().strftime('%Y-%m-%d %H:%M:%S')}"
                )

        range_records.extend(
            _build_variable_range_records(
                combined=combined,
                model_name=model_name,
                target_month=target_month,
                category=category,
                output_file=str(out_path.relative_to(base_dir)),
                timestamp_col=timestamp_col,
            )
        )

    return range_records


def _write_range_log(records: list[dict], output_dir: Path, filename: str) -> Optional[Path]:
    """Write variable prediction range records to CSV."""
    if not records:
        return None
    log_df = pd.DataFrame(records)
    log_df = log_df.sort_values(
        by=["model", "target_month", "category", "variable"],
        kind="mergesort",
    )
    log_path = output_dir / filename
    log_df.to_csv(log_path, index=False)
    return log_path


def main() -> int:
    args = _parse_args()
    base_dir = Path(args.base_dir)
    output_dir = base_dir / args.output_dir
    non_qwen_data_sources = _discover_non_qwen_data_sources(base_dir)
    qwen_data_sources = _discover_qwen_data_sources(base_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    expected_variables_by_group = _load_backend_group_variables(base_dir)
    for category in CATEGORIES:
        expected_count = len(expected_variables_by_group.get(category, set()))
        print(f"Backend variables for {category}: {expected_count}")

    all_range_records: list[dict] = []

    if args.model in ("gpt", "all"):
        gpt_sources = non_qwen_data_sources.get("gpt-5-search-api", GPT_LEGACY_DATA_SOURCES)
        all_range_records.extend(
            process_model_data(
                model_name="gpt-5-search-api",
                data_sources=gpt_sources,
                output_dir=output_dir,
                timestamp_col=args.timestamp_col,
                sort_output=args.sort,
                base_dir=base_dir,
                expected_variables_by_group=expected_variables_by_group,
                clean_output=True,
            )
        )

    if args.model in ("claude", "all"):
        claude_sources = non_qwen_data_sources.get(
            "claude-sonnet-4.5-api",
            CLAUDE_LEGACY_DATA_SOURCES,
        )
        all_range_records.extend(
            process_model_data(
                model_name="claude-sonnet-4.5-api",
                data_sources=claude_sources,
                output_dir=output_dir,
                timestamp_col=args.timestamp_col,
                sort_output=args.sort,
                base_dir=base_dir,
                expected_variables_by_group=expected_variables_by_group,
                clean_output=True,
            )
        )

    if args.model == "all":
        for model_name, data_sources in sorted(non_qwen_data_sources.items()):
            if model_name in {"gpt-5-search-api", "claude-sonnet-4.5-api"}:
                continue

            all_range_records.extend(
                process_model_data(
                    model_name=model_name,
                    data_sources=data_sources,
                    output_dir=output_dir,
                    timestamp_col=args.timestamp_col,
                    sort_output=args.sort,
                    base_dir=base_dir,
                    expected_variables_by_group=expected_variables_by_group,
                    clean_output=True,
                )
            )

    if args.model in ("qwen", "all"):
        if not qwen_data_sources:
            print(
                "\nNo Qwen intermediate CSV files were discovered. "
                "Run python data_from_serverA_serverB/prepare_qwen_data.py first."
            )
        for model_name, data_sources in sorted(qwen_data_sources.items()):
            all_range_records.extend(
                process_model_data(
                    model_name=model_name,
                    data_sources=data_sources,
                    output_dir=output_dir,
                    timestamp_col=args.timestamp_col,
                    sort_output=args.sort,
                    base_dir=base_dir,
                    expected_variables_by_group=expected_variables_by_group,
                    clean_output=True,
                )
            )

    log_path = _write_range_log(all_range_records, output_dir, args.range_log_file)
    if log_path is not None:
        print(f"\nWrote variable prediction range log: {log_path}")
    else:
        print("\nNo variable prediction range records were produced.")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
