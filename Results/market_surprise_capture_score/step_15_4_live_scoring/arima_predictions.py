"""Load ARIMA benchmark forecasts and synthesize LLM-shape prediction rows
for live scoring.

Source:
  Results/benchmark_econ/results_targets_2025_11_2026_03_aic/target_month_forecasts.csv

Each ARIMA row is exactly one forecast value for one (variable, target_month)
pair, produced from one FRED-MD vintage. ARIMA forecasts don't have an intra-
window timestamp, so we synthesize one prediction row per live release whose
ref_period matches an ARIMA target. The synthetic timestamp_local is set to
`release_datetime_et - 1 second`, which guarantees the forecast is always
inside any pre-release window (6h … 14d). This matches the convention that ARIMA produces a
single forecast value per (variable, target_month) that is available
mid-month of the release month and persists until the official release.

Matching rules (consistent with build_live_scoring.aggregate_in_window):
  * Monthly fields: ARIMA target_month == live release ref_period.
  * real_gdp_advance: ARIMA target_date's quarter == ref_period ('YYYY-Qn').
    We pick the ARIMA row whose vintage_period is the latest one strictly
    before the release, guaranteeing the forecast was available pre-release.

Unit overrides (ARIMA can emit different units than the LLMs):
  * existing_home_sales uses FRED series EXHOSLUSM495S (in millions);
    LLMs emit in thousands. Override multiply:1000 -> multiply:1000000.
  * All other fields: use the default model_transform_rule from field_mapping.

The output frame follows the LLM schema consumed by load_model_predictions /
score_model, so the existing scoring loop treats ARIMA as just another model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Variables whose ARIMA-emitted unit differs from the LLM unit. See the
# module docstring.
ARIMA_TRANSFORM_OVERRIDES: dict[str, str] = {
    "existing_home_sales": "multiply:1000000",
}


def _quarter_from_date(d: pd.Timestamp) -> str:
    """pd.Timestamp -> 'YYYY-Qn'."""
    q = (d.month - 1) // 3 + 1
    return f"{d.year}-Q{q}"


def apply_arima_overrides(mapping: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the field_mapping with ARIMA-specific transform rules
    applied. Used when scoring ARIMA so multiply:1000 is replaced by the right
    ARIMA factor per field."""
    m = mapping.copy()
    for field_id, rule in ARIMA_TRANSFORM_OVERRIDES.items():
        if field_id in m["field_id"].values:
            m.loc[m["field_id"] == field_id, "model_transform_rule"] = rule
    return m


def build_arima_predictions(
    arima_csv: Path,
    mapping: pd.DataFrame,
    live_releases: pd.DataFrame,
    quarterly_field_ids: set[str],
) -> tuple[pd.DataFrame, dict]:
    """Return (preds, diagnostics).

    preds has columns:
        timestamp_local (datetime64[ns]),
        target_month (str 'YYYY-MM'),
        release_month (str 'YYYY-MM'),
        variable (str, matches model_output_source),
        value (float, raw forecast; transformation is applied by score_model),
        parsed_ok (bool, always True),
        source_csv (str, audit trail).

    diagnostics is a dict with drop stats:
        n_arima_rows_considered, n_live_releases, n_synth_rows,
        unmatched_by_field (DataFrame), vintage_distribution (dict).
    """
    arima = pd.read_csv(arima_csv)
    arima["target_date"] = pd.to_datetime(arima["target_date"])
    arima["vintage_period_ts"] = pd.to_datetime(
        arima["vintage_period"].astype(str) + "-01"
    )

    # Filter to variables we score.
    field_to_mov = dict(zip(mapping["field_id"], mapping["model_output_source"]))
    scored_vars = set(field_to_mov.values())
    arima = arima[arima["variable"].isin(scored_vars)].copy()
    arima = arima[arima["forecast"].notna()].copy()

    synth_rows: list[dict] = []
    unmatched_rows: list[dict] = []

    for r in live_releases.itertuples(index=False):
        field_id = r.field_id
        mov = field_to_mov.get(field_id)
        if mov is None:
            continue
        ref = r.ref_period
        T = r.release_datetime_et

        if field_id in quarterly_field_ids:
            # Match by target_date's quarter.
            sub = arima[arima["variable"] == mov].copy()
            sub["target_quarter"] = sub["target_date"].apply(_quarter_from_date)
            candidates = sub[sub["target_quarter"] == ref]
        else:
            # Monthly: match target_month == ref_period.
            candidates = arima[
                (arima["variable"] == mov) & (arima["target_month"] == ref)
            ]

        # Vintage must be strictly before the release timestamp so the forecast
        # could have been available pre-release.
        candidates = candidates[candidates["vintage_period_ts"] < T]

        if candidates.empty:
            unmatched_rows.append(
                {
                    "field_id": field_id,
                    "release_datetime_et": T,
                    "ref_period": ref,
                    "variable": mov,
                    "reason": "no ARIMA row with matching (variable, ref_period, vintage<release)",
                }
            )
            continue

        chosen = candidates.sort_values("vintage_period_ts").iloc[-1]
        value = float(chosen["forecast"])

        # target_month for the synthetic row must satisfy the downstream
        # matcher: for monthly it equals ref_period; for GDP it must fall in
        # the target quarter. Use target_date.month for GDP (always in-quarter).
        if field_id in quarterly_field_ids:
            target_month_synth = chosen["target_date"].strftime("%Y-%m")
        else:
            target_month_synth = ref

        synth_rows.append(
            {
                "timestamp_local": T - pd.Timedelta(seconds=1),
                "target_month": target_month_synth,
                "release_month": T.strftime("%Y-%m"),
                "variable": mov,
                "value": value,
                "parsed_ok": True,
                "source_csv": (
                    f"arima_aic/vintage={chosen['vintage_period']}/"
                    f"target_month={chosen['target_month']}/{mov}.csv"
                ),
                # Audit columns (harmless — score_model ignores unknown cols).
                "arima_vintage_period": str(chosen["vintage_period"]),
                "arima_target_month": str(chosen["target_month"]),
                "arima_target_date": chosen["target_date"].strftime("%Y-%m-%d"),
                "arima_last_observed_date": str(chosen.get("last_observed_date", "")),
                "field_id": field_id,
            }
        )

    if synth_rows:
        preds = pd.DataFrame(synth_rows)
        preds["timestamp_local"] = pd.to_datetime(preds["timestamp_local"])
        # Column order: LLM-shape first, audit cols after.
        llm_cols = [
            "timestamp_local",
            "target_month",
            "release_month",
            "variable",
            "value",
            "parsed_ok",
            "source_csv",
        ]
        audit_cols = [c for c in preds.columns if c not in llm_cols]
        preds = preds[llm_cols + audit_cols]
    else:
        preds = pd.DataFrame(
            columns=[
                "timestamp_local",
                "target_month",
                "release_month",
                "variable",
                "value",
                "parsed_ok",
                "source_csv",
            ]
        )

    unmatched_df = pd.DataFrame(unmatched_rows)
    diagnostics = {
        "n_arima_rows_considered": int(len(arima)),
        "n_live_releases": int(len(live_releases)),
        "n_synth_rows": int(len(synth_rows)),
        "unmatched_by_field": unmatched_df,
        "vintage_distribution": (
            preds["arima_vintage_period"].value_counts().to_dict()
            if len(synth_rows)
            else {}
        ),
    }
    return preds, diagnostics
