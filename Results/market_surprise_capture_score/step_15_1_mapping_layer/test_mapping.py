"""Step 15.1 tests — integrity of the required mapping layer.

Verifies that the mapping artifact
  step_15_1_mapping_layer/field_mapping.csv
is internally valid AND consistent with:
  - the Investing.com calendar data
      Results/data_consensus/filtered_macro_events.csv
  - the LLM model predictions
      Results/data_from_serverA_serverB/final_analysis_data/*
  - the MAPPING hard-coded in step 15.2's preprocess_historical.py

Run from repo root:
    python Results/market_surprise_capture_score/step_15_1_mapping_layer/test_mapping.py

Exit code 0 on pass; non-zero on failure with a human-readable summary.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent
REPO_ROOT = ROOT.parent.parent

CALENDAR_CSV = ROOT.parent / "data_consensus" / "filtered_macro_events.csv"
STEP_15_2_SCRIPT = ROOT / "step_15_2_historical_preprocessing" / "preprocess_historical.py"
MODEL_PRED_DIR = REPO_ROOT / "Results" / "data_from_serverA_serverB" / "final_analysis_data"
VARIABLE_RANGES_CSV = MODEL_PRED_DIR / "variable_prediction_ranges.csv"

MAPPING_CSV = HERE / "field_mapping.csv"


# ---------------------------------------------------------------------------
# Small test harness.
# ---------------------------------------------------------------------------
class Failures:
    def __init__(self) -> None:
        self._msgs: list[str] = []

    def add(self, test: str, msg: str) -> None:
        self._msgs.append(f"[FAIL] {test}: {msg}")

    def ok(self, test: str) -> None:
        print(f"[ OK ] {test}")

    def summary_and_exit(self) -> None:
        if not self._msgs:
            print("\nAll mapping tests passed.")
            sys.exit(0)
        print("\n" + "\n".join(self._msgs))
        print(f"\n{len(self._msgs)} test(s) failed.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def load_mapping() -> pd.DataFrame:
    df = pd.read_csv(MAPPING_CSV, keep_default_na=False)
    # Bool column is written as 'True' / 'False' strings via pd.to_csv; coerce.
    df["active_flag"] = df["active_flag"].map({"True": True, "False": False}).astype(bool)
    return df


def load_step_15_2_mapping_module():
    """Load step 15.2's preprocess_historical.py purely for its MAPPING list.

    We import it dynamically so we don't need the whole 15.2 dependency tree
    (numpy, statsmodels, etc. are fine; only pandas + stdlib is strictly
    needed to read .MAPPING).
    """
    spec = importlib.util.spec_from_file_location(
        "preprocess_historical_15_2", STEP_15_2_SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def load_all_model_variables() -> set[str]:
    """Union of `variable` values across all model prediction CSVs."""
    vars_: set[str] = set()
    for p in sorted(MODEL_PRED_DIR.rglob("*.csv")):
        if p.name == "variable_prediction_ranges.csv":
            continue
        # Only read the `variable` column for speed.
        try:
            df = pd.read_csv(p, usecols=["variable"])
        except Exception as e:
            raise RuntimeError(f"failed reading {p}: {e}") from e
        vars_.update(df["variable"].dropna().astype(str).unique().tolist())
    return vars_


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------
def test_csv_exists(F: Failures) -> None:
    name = "mapping CSV exists"
    if not MAPPING_CSV.exists():
        F.add(name, f"{MAPPING_CSV} not found (run build_mapping.py first)")
        return
    F.ok(name)


def test_required_columns(F: Failures, mp: pd.DataFrame) -> None:
    name = "required columns present"
    # Outline 15.1 "at least the following columns" minus calendar_field_name
    # (intentionally dropped — see build_mapping.py).
    required = {
        "field_id",
        "package_name",
        "calendar_event_name",
        "calendar_units",
        "model_output_source",
        "model_transform_rule",
        "active_flag",
    }
    missing = required - set(mp.columns)
    if missing:
        F.add(name, f"missing columns: {sorted(missing)}")
        return
    F.ok(name)


def test_field_id_unique(F: Failures, mp: pd.DataFrame) -> None:
    name = "field_id unique"
    dup = mp["field_id"][mp["field_id"].duplicated()].tolist()
    if dup:
        F.add(name, f"duplicates: {dup}")
        return
    F.ok(name)


def test_model_output_unique(F: Failures, mp: pd.DataFrame) -> None:
    name = "model_output_source unique"
    dup = mp["model_output_source"][mp["model_output_source"].duplicated()].tolist()
    if dup:
        F.add(name, f"duplicates: {dup}")
        return
    F.ok(name)


def test_transform_rules_valid(F: Failures, mp: pd.DataFrame) -> None:
    name = "model_transform_rule values valid"
    pat = re.compile(r"^(identity|multiply:\s*-?\d+(\.\d+)?)$")
    bad = mp[~mp["model_transform_rule"].astype(str).str.match(pat)]
    if len(bad):
        F.add(name, f"invalid rules:\n{bad[['field_id','model_transform_rule']].to_string(index=False)}")
        return
    F.ok(name)


def test_consistency_with_15_2(F: Failures, mp: pd.DataFrame) -> None:
    """field_id set + calendar_event_name match step 15.2's MAPPING."""
    name = "consistency with step 15.2 MAPPING"
    try:
        mod = load_step_15_2_mapping_module()
    except Exception as e:
        F.add(name, f"could not import step 15.2 module: {e}")
        return

    m152 = mod.MAPPING

    # Same field_id set and order (both matter — step 15.2 uses FIELD_IDS
    # in a fixed order to build X columns).
    ids_15_1 = mp["field_id"].tolist()
    ids_15_2 = [r["field_id"] for r in m152]
    if ids_15_1 != ids_15_2:
        F.add(
            name,
            f"field_id order/set differs\n"
            f"  15.1: {ids_15_1}\n"
            f"  15.2: {ids_15_2}",
        )
        return

    # Per-field: primary calendar_event_name must equal event_bases[0].
    problems: list[str] = []
    for row_15_1, row_15_2 in zip(mp.to_dict("records"), m152):
        fid = row_15_1["field_id"]
        primary_15_1 = str(row_15_1["calendar_event_name"]).strip()
        primary_15_2 = row_15_2["event_bases"][0].strip()
        if primary_15_1 != primary_15_2:
            problems.append(
                f"  {fid}: primary calendar_event_name '{primary_15_1}' != "
                f"15.2 event_bases[0] '{primary_15_2}'"
            )
        # Fallbacks: pipe-separated in 15.1, list in 15.2.
        fb_15_1_s = str(row_15_1.get("calendar_event_name_fallbacks", "")).strip()
        fb_15_1 = [x.strip() for x in fb_15_1_s.split("|") if x.strip()] if fb_15_1_s else []
        fb_15_2 = list(row_15_2["event_bases"][1:])
        if fb_15_1 != fb_15_2:
            problems.append(
                f"  {fid}: fallbacks '{fb_15_1}' != 15.2 event_bases[1:] '{fb_15_2}'"
            )
    if problems:
        F.add(name, "mismatches:\n" + "\n".join(problems))
        return
    F.ok(name)


def test_calendar_events_exist(F: Failures, mp: pd.DataFrame) -> None:
    """Every primary + fallback calendar_event_name appears in the calendar
    with at least one non-null actual."""
    name = "calendar events resolve to non-null actuals"
    if not CALENDAR_CSV.exists():
        F.add(name, f"calendar CSV not found: {CALENDAR_CSV}")
        return
    cal = pd.read_csv(CALENDAR_CSV, usecols=["event_base", "actual"])
    nonnull_by_base = (
        cal.dropna(subset=["actual"]).groupby("event_base").size()
    )

    missing: list[str] = []
    for _, row in mp.iterrows():
        names = [str(row["calendar_event_name"]).strip()]
        fb = str(row.get("calendar_event_name_fallbacks", "")).strip()
        if fb:
            names.extend(x.strip() for x in fb.split("|") if x.strip())
        for nm in names:
            if nonnull_by_base.get(nm, 0) == 0:
                missing.append(f"{row['field_id']}: '{nm}' has 0 non-null actuals")
    if missing:
        F.add(name, "\n".join(missing))
        return
    F.ok(name)


def test_model_variables_exist(F: Failures, mp: pd.DataFrame) -> None:
    """Every model_output_source exists as a `variable` value in at least one
    model prediction CSV."""
    name = "model_output_source exists in predictions"
    try:
        model_vars = load_all_model_variables()
    except Exception as e:
        F.add(name, f"could not load model variables: {e}")
        return

    missing = [
        (row["field_id"], row["model_output_source"])
        for _, row in mp.iterrows()
        if str(row["model_output_source"]).strip()
        and str(row["model_output_source"]).strip() not in model_vars
    ]
    if missing:
        F.add(
            name,
            "variables not found in any prediction CSV:\n"
            + "\n".join(f"  {fid}: '{v}'" for fid, v in missing)
            + f"\n  (searched under {MODEL_PRED_DIR})",
        )
        return
    F.ok(name)


def test_unit_sanity(F: Failures, mp: pd.DataFrame) -> None:
    """Cross-check that the transform rule makes the model value land in the
    order-of-magnitude range of the calendar actual.

    For each field we sample a few calendar actuals and compute the implied
    transform-scale residual. We do NOT require closeness (the model can be
    wrong by a lot); we require that the *scale* is consistent. A factor-of-
    10+ mismatch between median |transform(model)| and median |actual| would
    indicate a unit error.

    Checks are skipped for any field where we have no overlapping
    (model variable, calendar actual) observations.
    """
    name = "transform-rule unit sanity (order-of-magnitude)"
    if not CALENDAR_CSV.exists():
        F.add(name, f"calendar CSV not found: {CALENDAR_CSV}")
        return

    # Pull median |actual| per event_base from calendar.
    cal = pd.read_csv(
        CALENDAR_CSV, usecols=["event_base", "actual"], low_memory=False
    ).dropna(subset=["actual"])
    cal["abs"] = cal["actual"].abs()
    med_actual = cal.groupby("event_base")["abs"].median()

    # Pull median |value| per variable from model CSVs.
    all_rows: list[pd.DataFrame] = []
    for p in sorted(MODEL_PRED_DIR.rglob("*.csv")):
        if p.name == "variable_prediction_ranges.csv":
            continue
        all_rows.append(pd.read_csv(p, usecols=["variable", "value"]))
    if not all_rows:
        F.add(name, f"no model CSVs found under {MODEL_PRED_DIR}")
        return
    mdf = pd.concat(all_rows, ignore_index=True).dropna(subset=["value"])
    mdf["abs"] = mdf["value"].abs()
    med_model = mdf.groupby("variable")["abs"].median()

    problems: list[str] = []
    for _, row in mp.iterrows():
        eb = str(row["calendar_event_name"]).strip()
        mv = str(row["model_output_source"]).strip()
        rule = str(row["model_transform_rule"]).strip()
        if eb not in med_actual.index or mv not in med_model.index:
            continue
        a = float(med_actual[eb])
        m = float(med_model[mv])
        if rule == "identity":
            m_scaled = m
        elif rule.startswith("multiply:"):
            m_scaled = m * float(rule.split(":", 1)[1])
        else:
            continue
        # Guard against zero medians (e.g., MoM near 0).
        if a == 0 or m_scaled == 0:
            continue
        ratio = max(a, m_scaled) / min(a, m_scaled)
        # Warn if the transformed model median is off by >10x from the
        # calendar actual median — that's almost certainly a unit bug.
        if ratio > 10.0:
            problems.append(
                f"  {row['field_id']}: median|cal|={a:,.3g}, "
                f"median|model|={m:,.3g}, rule={rule!r} -> "
                f"median|transform(model)|={m_scaled:,.3g}  (ratio={ratio:.1f}x)"
            )
    if problems:
        F.add(
            name,
            "order-of-magnitude mismatches (possible unit error):\n"
            + "\n".join(problems),
        )
        return
    F.ok(name)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main() -> int:
    F = Failures()

    test_csv_exists(F)
    if not MAPPING_CSV.exists():
        F.summary_and_exit()

    mp = load_mapping()

    test_required_columns(F, mp)
    test_field_id_unique(F, mp)
    test_model_output_unique(F, mp)
    test_transform_rules_valid(F, mp)
    test_consistency_with_15_2(F, mp)
    test_calendar_events_exist(F, mp)
    test_model_variables_exist(F, mp)
    test_unit_sanity(F, mp)

    F.summary_and_exit()
    return 0  # unreachable


if __name__ == "__main__":
    main()
