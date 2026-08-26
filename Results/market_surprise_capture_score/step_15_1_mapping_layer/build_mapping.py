"""Step 15.1 Required mapping layer (Project Outline section 15.1).

This step emits the canonical mapping table that connects:

  - a scored field_id (the outline's set I, |I| = 16),
  - the Investing.com calendar event it is read from,
  - the LLM model-prediction variable it is read from in
    Results/data_from_serverA_serverB/final_analysis_data,
  - the transform that converts the model's emitted value into the exact
    scale and field shown in the calendar.

The artifact (field_mapping.csv) is the authoritative mapping source for
steps 15.2, 15.3, and 15.4. The set of field_ids here MUST match the
MAPPING hard-coded in
  step_15_2_historical_preprocessing/preprocess_historical.py::MAPPING
so that:
  * step 15.2 produces X_<field_id> surprise columns for exactly these ids,
  * step 15.3 reads step_15_2's field_mapping.csv, builds X columns for
    exactly these ids, and estimates beta_<field_id>,
  * step 15.4 will need X_<field_id> and beta_<field_id> to match.

The consistency check is enforced at test time (see test_mapping.py).

Columns emitted (outline section 15.1 "at least"):
  - field_id             canonical short id for the scored field
  - package_name         coarse release package (calendar-side grouping)
  - calendar_event_name  Investing.com `event_base` string used to read
                         the row (A, C, T) for this field
  - calendar_units       natural-language units description
  - model_output_source  `variable` column value in the model prediction
                         CSVs that this field_id reads from
  - model_transform_rule rule that maps the emitted model value to the
                         calendar scale. One of:
                           "identity"     -> calendar value = model value
                           "multiply:K"   -> calendar value = K * model value
  - active_flag          True for fields used in scoring
  - calendar_event_name_fallbacks
                         pipe-separated additional event_base strings that
                         refer to the SAME economic series under an
                         alternate Investing.com label (only non-empty for
                         the PCE price index rename in 2019-08; mirrors the
                         priority list in step 15.2).
  - notes                human-readable caveat / rationale

Transform-rule spec
-------------------
Two transforms are sufficient for the current model output set:

  identity:
    ISM Manufacturing PMI, ISM Non-Manufacturing PMI index levels
    CPI/PPI/PCE/Retail/real_pce MoM percent
    Industrial Production MoM percent
    Durable Goods Orders MoM percent
    GDP QoQ SAAR percent
    Unemployment Rate level percent

  multiply:1000:
    Nonfarm Payrolls change: model emits thousands (K); calendar stores
      raw headcount change (hundreds of thousands). Verified by
      inspecting filtered_macro_events.csv:
        2015-01-09 Nonfarm Payrolls  actual = 252000.0
        Model nonfarm_payrolls_change typical ~ +100 to +300.
    Housing Starts / Building Permits / Existing Home Sales /
    New Home Sales levels: model emits thousands (K SAAR); calendar stores
      raw count SAAR. Verified:
        2015-01-21 Housing Starts    actual = 1089000.0
        Model housing_starts_level typical ~ 1400.

No other transforms are needed for the current 16-field set.

Outputs (written alongside this script):
  - field_mapping.csv         the section 15.1 mapping artifact
  - mapping_report.txt        audit / QC summary

Usage:
    python build_mapping.py            # writes into this directory
    python build_mapping.py --out-dir /tmp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent

# Canonical mapping. One dict per scored field. The order is the authoritative
# field order used in step 15.2 and step 15.3.
MAPPING: list[dict] = [
    {
        "field_id": "real_gdp_advance",
        "package_name": "gdp",
        "calendar_event_name": "GDP (QoQ)",
        "calendar_event_name_fallbacks": [],
        "calendar_units": "% QoQ SAAR",
        "model_output_source": "real_gdp_qoq",
        "model_transform_rule": "identity",
        "active_flag": True,
        "notes": (
            "Keep only ADVANCE release (earliest of 3 same-Q# releases per "
            "reference quarter); filter applied in step 15.2."
        ),
    },
    {
        "field_id": "industrial_production",
        "package_name": "industrial_production",
        "calendar_event_name": "Industrial Production (MoM)",
        "calendar_event_name_fallbacks": [],
        "calendar_units": "% MoM",
        "model_output_source": "industrial_production_mom",
        "model_transform_rule": "identity",
        "active_flag": True,
        "notes": "",
    },
    {
        "field_id": "durable_goods",
        "package_name": "durable_goods",
        "calendar_event_name": "Durable Goods Orders (MoM)",
        "calendar_event_name_fallbacks": [],
        "calendar_units": "% MoM",
        "model_output_source": "durable_goods_mom",
        "model_transform_rule": "identity",
        "active_flag": True,
        "notes": "",
    },
    {
        "field_id": "ism_manufacturing",
        "package_name": "ism_manufacturing",
        "calendar_event_name": "ISM Manufacturing PMI",
        "calendar_event_name_fallbacks": [],
        "calendar_units": "index level",
        "model_output_source": "ism_manufacturing_index",
        "model_transform_rule": "identity",
        "active_flag": True,
        "notes": "",
    },
    {
        "field_id": "cpi",
        "package_name": "cpi",
        "calendar_event_name": "CPI (YoY)",
        "calendar_event_name_fallbacks": [],
        "calendar_units": "% YoY",
        "model_output_source": "cpi_yoy",
        "model_transform_rule": "identity",
        "active_flag": True,
        "notes": "Headline YoY (using the YoY series rather than MoM SA).",
    },
    {
        "field_id": "ppi",
        "package_name": "ppi",
        "calendar_event_name": "PPI (MoM)",
        "calendar_event_name_fallbacks": [],
        "calendar_units": "% MoM",
        "model_output_source": "ppi_mom",
        "model_transform_rule": "identity",
        "active_flag": True,
        "notes": "",
    },
    {
        "field_id": "pce_price_index",
        "package_name": "personal_income_outlays",
        "calendar_event_name": "PCE price index (MoM)",
        "calendar_event_name_fallbacks": ["PCE Deflator (MoM)"],
        "calendar_units": "% MoM",
        "model_output_source": "pce_price_index_mom",
        "model_transform_rule": "identity",
        "active_flag": True,
        "notes": (
            "Concatenate PCE Deflator (MoM) pre-2019-08 and PCE price index "
            "(MoM) post-2019-08 (same BEA series renamed by Investing.com; "
            "prefer PCE price index on any overlap). Mirrors the priority "
            "list in step 15.2 so both names collapse to this field_id."
        ),
    },
    {
        "field_id": "real_pce",
        "package_name": "personal_income_outlays",
        "calendar_event_name": "Real Personal Consumption (MoM)",
        "calendar_event_name_fallbacks": [],
        "calendar_units": "% MoM",
        "model_output_source": "real_pce_mom",
        "model_transform_rule": "identity",
        "active_flag": True,
        "notes": "",
    },
    {
        "field_id": "retail_sales",
        "package_name": "retail_sales",
        "calendar_event_name": "Retail Sales (MoM)",
        "calendar_event_name_fallbacks": [],
        "calendar_units": "% MoM",
        "model_output_source": "retail_sales_mom",
        "model_transform_rule": "identity",
        "active_flag": True,
        "notes": "",
    },
    {
        "field_id": "nonfarm_payrolls",
        "package_name": "employment_situation",
        "calendar_event_name": "Nonfarm Payrolls",
        "calendar_event_name_fallbacks": [],
        "calendar_units": "headcount level change (calendar stores raw count)",
        "model_output_source": "nonfarm_payrolls_change",
        "model_transform_rule": "multiply:1000",
        "active_flag": True,
        "notes": (
            "Model emits thousands (K). Calendar stores raw headcount change "
            "(e.g., 252000.0 for Jan-2015). Multiply model by 1000 to match."
        ),
    },
    {
        "field_id": "unemployment_rate",
        "package_name": "employment_situation",
        "calendar_event_name": "Unemployment Rate",
        "calendar_event_name_fallbacks": [],
        "calendar_units": "% level",
        "model_output_source": "unemployment_rate",
        "model_transform_rule": "identity",
        "active_flag": True,
        "notes": "",
    },
    {
        "field_id": "housing_starts",
        "package_name": "new_residential_construction",
        "calendar_event_name": "Housing Starts",
        "calendar_event_name_fallbacks": [],
        "calendar_units": "SAAR level (calendar stores raw count)",
        "model_output_source": "housing_starts_level",
        "model_transform_rule": "multiply:1000",
        "active_flag": True,
        "notes": (
            "Model emits thousands (K SAAR). Calendar stores raw count SAAR "
            "(e.g., 1089000.0 for Jan-2015). Multiply model by 1000 to match."
        ),
    },
    {
        "field_id": "building_permits",
        "package_name": "new_residential_construction",
        "calendar_event_name": "Building Permits",
        "calendar_event_name_fallbacks": [],
        "calendar_units": "SAAR level (calendar stores raw count)",
        "model_output_source": "building_permits_level",
        "model_transform_rule": "multiply:1000",
        "active_flag": True,
        "notes": (
            "All releases with valid A+C+r are kept; outline does not restrict "
            "to 08:30. From 2023 onward the revised release is a separate "
            "timestamp-group event. Model emits K SAAR; calendar stores raw "
            "count SAAR (e.g., 1032000.0 for Jan-2015). Multiply by 1000."
        ),
    },
    {
        "field_id": "existing_home_sales",
        "package_name": "existing_home_sales",
        "calendar_event_name": "Existing Home Sales",
        "calendar_event_name_fallbacks": [],
        "calendar_units": "SAAR level (calendar stores raw count)",
        "model_output_source": "existing_home_sales_level",
        "model_transform_rule": "multiply:1000",
        "active_flag": True,
        "notes": (
            "Model emits K SAAR (e.g., 4090). Calendar stores raw count SAAR "
            "(e.g., 5040000.0 for Jan-2015). Multiply model by 1000."
        ),
    },
    {
        "field_id": "new_home_sales",
        "package_name": "new_home_sales",
        "calendar_event_name": "New Home Sales",
        "calendar_event_name_fallbacks": [],
        "calendar_units": "SAAR level (calendar stores raw count)",
        "model_output_source": "new_home_sales_level",
        "model_transform_rule": "multiply:1000",
        "active_flag": True,
        "notes": (
            "Model emits K SAAR (e.g., 605). Calendar stores raw count SAAR "
            "(e.g., 481000.0 for Jan-2015). Multiply model by 1000."
        ),
    },
    {
        "field_id": "ism_services",
        "package_name": "ism_services",
        "calendar_event_name": "ISM Non-Manufacturing PMI",
        "calendar_event_name_fallbacks": [],
        "calendar_units": "index level",
        "model_output_source": "ism_services_index",
        "model_transform_rule": "identity",
        "active_flag": True,
        "notes": "Investing.com uses the legacy 'Non-Manufacturing' label for ISM Services.",
    },
]

# Allowed values for model_transform_rule. Extend only with a matching
# implementation in step 15.4's transform dispatcher.
ALLOWED_TRANSFORM_PREFIXES = ("identity", "multiply:")


def validate_mapping(mapping: list[dict]) -> None:
    """Structural validation — fails loudly if anything is off."""
    required_keys = {
        "field_id",
        "package_name",
        "calendar_event_name",
        "calendar_event_name_fallbacks",
        "calendar_units",
        "model_output_source",
        "model_transform_rule",
        "active_flag",
        "notes",
    }
    seen_ids: set[str] = set()
    seen_models: set[str] = set()
    for i, m in enumerate(mapping):
        missing = required_keys - set(m.keys())
        if missing:
            raise ValueError(f"row {i} missing keys: {sorted(missing)}")
        fid = m["field_id"]
        if fid in seen_ids:
            raise ValueError(f"duplicate field_id: {fid}")
        seen_ids.add(fid)
        mov = m["model_output_source"]
        if mov in seen_models:
            raise ValueError(
                f"duplicate model_output_source {mov} (used by {fid} and a prior row)"
            )
        seen_models.add(mov)
        rule = m["model_transform_rule"]
        if not any(rule == p or rule.startswith(p) for p in ALLOWED_TRANSFORM_PREFIXES):
            raise ValueError(
                f"field {fid}: model_transform_rule {rule!r} is not in "
                f"allowed set {ALLOWED_TRANSFORM_PREFIXES}"
            )
        if rule.startswith("multiply:"):
            k_str = rule.split(":", 1)[1]
            try:
                float(k_str)
            except ValueError:
                raise ValueError(
                    f"field {fid}: multiply factor {k_str!r} is not numeric"
                )


def to_dataframe(mapping: list[dict]) -> pd.DataFrame:
    rows = []
    for m in mapping:
        rows.append(
            {
                "field_id": m["field_id"],
                "package_name": m["package_name"],
                "calendar_event_name": m["calendar_event_name"],
                "calendar_event_name_fallbacks": " | ".join(
                    m["calendar_event_name_fallbacks"]
                ),
                "calendar_units": m["calendar_units"],
                "model_output_source": m["model_output_source"],
                "model_transform_rule": m["model_transform_rule"],
                "active_flag": bool(m["active_flag"]),
                "notes": m["notes"],
            }
        )
    return pd.DataFrame(rows)


def apply_transform(value: float, rule: str) -> float:
    """Convert a model value into calendar scale per rule.

    Exposed for reuse in step 15.4.
    """
    if pd.isna(value):
        return value
    if rule == "identity":
        return float(value)
    if rule.startswith("multiply:"):
        k = float(rule.split(":", 1)[1])
        return float(value) * k
    raise ValueError(f"Unsupported model_transform_rule: {rule!r}")


def build_report(df: pd.DataFrame) -> str:
    L: list[str] = []
    A = L.append
    A("=" * 78)
    A("Step 15.1 Required mapping layer — report")
    A("=" * 78)
    A(f"Mapping rows:     {len(df)}")
    A(f"Active rows:      {int(df['active_flag'].sum())}")
    A(
        f"Transform rules:  "
        f"{df['model_transform_rule'].value_counts().to_dict()}"
    )
    A("")

    A("-- full mapping --")
    show_cols = [
        "field_id",
        "package_name",
        "calendar_event_name",
        "calendar_units",
        "model_output_source",
        "model_transform_rule",
        "active_flag",
    ]
    A(df[show_cols].to_string(index=False))
    A("")

    fb = df[df["calendar_event_name_fallbacks"].str.len() > 0]
    A(f"-- rows with calendar_event_name fallbacks: {len(fb)} --")
    if len(fb):
        A(fb[["field_id", "calendar_event_name", "calendar_event_name_fallbacks"]]
          .to_string(index=False))
    A("")

    A("Known calendar-side concessions (not scoring fallbacks):")
    A("  * pce_price_index: PCE Deflator (MoM) and PCE price index (MoM)")
    A("    are the same BEA series; Investing.com renamed the label in")
    A("    2019-08. Both names are treated as the same series, mirroring")
    A("    step 15.2's priority list. This is a calendar-side identity")
    A("    decision, not a scoring-time fallback.")
    A("")
    A("Consistency contract:")
    A("  * field_id set and calendar_event_name set MUST match the MAPPING")
    A("    hard-coded in step_15_2_historical_preprocessing/"
      "preprocess_historical.py.")
    A("  * test_mapping.py verifies this.")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out-dir", type=Path, default=HERE)
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    validate_mapping(MAPPING)
    df = to_dataframe(MAPPING)

    csv_p = args.out_dir / "field_mapping.csv"
    report_p = args.out_dir / "mapping_report.txt"

    df.to_csv(csv_p, index=False)
    report = build_report(df)
    report_p.write_text(report)

    print(report)
    print(f"Wrote:\n  {csv_p}\n  {report_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
