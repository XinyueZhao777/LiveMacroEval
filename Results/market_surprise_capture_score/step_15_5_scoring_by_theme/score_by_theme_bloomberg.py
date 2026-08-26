"""Step 15.5 (Bloomberg overlay) — theme-restricted scoring on the
field-level outputs from
`step_15_4_live_scoring/build_live_scoring_bloomberg.py`
(directory `bloomberg_overlay/`).

For each (model, theme, variant), computes Addendum-3 bounded scores
(BMSC, BP_RMSE, WDH) and bounded directional-return capture (BDRC,
DRC_RMSE) on the theme-restricted field rows, plus parametric-bootstrap
90/95% CIs from a joint draw of β^(b) ~ N(β̂_theme, V_β[theme, theme]).

Variants (mirroring the upstream pipeline)
------------------------------------------
  score_on_median_1d : per-event MEDIAN of (C_d, M_d) over d=1.
  score_on_median_3d : per-event MEDIAN of (C_d, M_d) over d in {1,2,3}.
  score_on_median_7d : per-event MEDIAN of (C_d, M_d) over d in {1..7}.
  final_vs_final   : Bloomberg final-summary MEDIAN as C and the
                     model's latest pre-release prediction as M.

Theme composition (4 themes; matches the paper's thematic blocks)
-----------------------------------------------------------------
  Production                          (4 fields)
  Inflation_Consumption_Services     (6 fields)
  Labor_Market                        (2 fields)
  Housing                             (4 fields)

CI methodology (mirrors the upstream `bootstrap_score_ci`)
----------------------------------------------------------
  1. Refit Huber-ridge on the historical events panel
     (`step_15_2_historical_preprocessing/timestamp_group_events.csv`)
     to recover β̂ and the HC1 sandwich V_β over the FULL field set.
     β̂ is cross-checked against
     `bloomberg_overlay/huber_beta_by_field.csv` (max-abs delta).
  2. Slice (β̂, V_β) to each theme's field indices and re-Cholesky
     V_β[theme, theme] to get L_theme.
  3. For each (model, theme, variant), filter the upstream variant CSV
     to theme fields and call `bootstrap_score_ci` with the sliced
     β̂_theme, L_theme. Same n_boot and seed as the headline pipeline
     (10,000; seed 20260427) so the layers are directly comparable.

Outputs (this folder)
---------------------
  scores_by_theme_bloomberg_aggregate.csv
      wide: one row per (model, theme), columns
      `<variant>_<metric>` (point) and `<variant>_<metric>_ci{90,95}_{lo,hi}`.
  scores_by_theme_bloomberg_BMSC_wide_<variant>.csv
      one per variant; BMSC point by model x theme.
  scores_by_theme_bloomberg_report.txt
  theme_membership_bloomberg.csv

Conda env: livemacro.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_OVERLAY_DIR = ROOT / "step_15_4_live_scoring" / "bloomberg_overlay"
DEFAULT_TS_EVENTS = (
    ROOT / "step_15_2_historical_preprocessing" / "timestamp_group_events.csv"
)
DEFAULT_MAPPING = ROOT / "step_15_1_mapping_layer" / "field_mapping.csv"

# Reuse helpers from the upstream Bloomberg-overlay pipeline.
sys.path.insert(0, str(ROOT / "step_15_4_live_scoring"))
from build_live_scoring_bloomberg import (  # noqa: E402
    bootstrap_score_ci,
    event_aggregate,
    fit_beta_for_pipeline,
    metrics_with_bdrc,
    HUBER_C,
    HUBER_LAMBDA,
    N_BOOT,
    BOOT_SEED,
    DRC_WDH_THRESHOLD_BPS,
)

# ---------------------------------------------------------------------------
# Theme composition (matches the paper's thematic blocks).
# Field IDs follow step_15_1_mapping_layer/field_mapping.csv.
# ---------------------------------------------------------------------------
THEMES: dict[str, list[str]] = {
    "Production": [
        "real_gdp_advance",          # Real GDP (BEA)
        "industrial_production",     # Industrial Production (Federal Reserve)
        "durable_goods",             # Durable Goods Orders (Census)
        "ism_manufacturing",         # ISM Manufacturing PMI (ISM)
    ],
    "Inflation_Consumption_Services": [
        "cpi",                       # CPI (BLS)
        "ppi",                       # PPI (BLS)
        "pce_price_index",           # PCE Price Index (BEA)
        "retail_sales",              # Retail Sales — Advance Monthly (Census)
        "real_pce",                  # Real PCE (BEA)
        "ism_services",              # ISM Services PMI (ISM)
    ],
    "Labor_Market": [
        "nonfarm_payrolls",          # NFP (BLS)
        "unemployment_rate",         # Unemployment Rate (BLS)
    ],
    "Housing": [
        "housing_starts",            # Housing Starts (Census)
        "building_permits",          # Building Permits (Census)
        "existing_home_sales",       # Existing Home Sales (NAR)
        "new_home_sales",            # New Home Sales (Census)
    ],
}
THEME_ORDER = list(THEMES)

VARIANT_FILES: dict[str, str] = {
    "score_on_median_1d": "bloomberg_score_on_median_field_1d.csv",
    "score_on_median_3d": "bloomberg_score_on_median_field_3d.csv",
    "score_on_median_7d": "bloomberg_score_on_median_field_7d.csv",
    "final_vs_final":   "bloomberg_final_vs_final_field.csv",
}
VARIANT_ORDER = list(VARIANT_FILES)

POINT_METRIC_KEYS = (
    "MSC", "BMSC", "BP_RMSE", "WDH",
    "BDRC", "DRC_RMSE", "n_events", "n_drc_events",
)
# Metrics for which CI columns are emitted (mirrors `bootstrap_score_ci`).
CI_METRICS = ("BMSC", "BP_RMSE", "WDH", "BDRC", "DRC_RMSE")


def _theme_point_metrics(
    field_df: pd.DataFrame,
    theme_fields: list[str],
    hf_map: dict,
) -> dict:
    """Filter `field_df` to `theme_fields`, aggregate by event, attach
    hf_return, and compute bounded scores. Returns metric dict (NaN keys
    when no matching rows)."""
    sub = field_df[field_df["field_id"].isin(theme_fields)]
    if sub.empty:
        return {k: float("nan") for k in POINT_METRIC_KEYS}
    events = event_aggregate(sub, ["release_datetime_et"])
    events["hf_return"] = events["release_datetime_et"].map(hf_map)
    return metrics_with_bdrc(events)


def _cholesky_psd(V: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.cholesky(V)
    except np.linalg.LinAlgError:
        w, U = np.linalg.eigh(V)
        return U * np.sqrt(np.maximum(w, 0.0))[None, :]


def _theme_ci(
    field_df: pd.DataFrame,
    theme_fields: list[str],
    hf_map: dict,
    beta_theme: np.ndarray,
    L_theme: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    drc_wdh_tau: float,
) -> dict:
    """Bootstrap CIs for one (model, theme, variant). Attaches hf_return
    onto the theme-restricted field rows so `bootstrap_score_ci` can
    compute BDRC."""
    sub = field_df[field_df["field_id"].isin(theme_fields)].copy()
    if sub.empty:
        return {}
    sub["hf_return"] = sub["release_datetime_et"].map(hf_map)
    return bootstrap_score_ci(
        sub, theme_fields, beta_theme, L_theme,
        n_boot=n_boot, rng=rng, drc_wdh_tau=drc_wdh_tau,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--overlay-dir", type=Path, default=DEFAULT_OVERLAY_DIR)
    p.add_argument("--ts-events-csv", type=Path, default=DEFAULT_TS_EVENTS,
                   help="historical events file used to refit β + V_β.")
    p.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING,
                   help="field_mapping.csv — defines the canonical "
                        "field_id ordering for the regression.")
    p.add_argument("--out-dir", type=Path, default=HERE)
    p.add_argument("--lambda", dest="lam", type=float, default=HUBER_LAMBDA)
    p.add_argument("--huber-c", type=float, default=HUBER_C)
    p.add_argument("--n-boot", type=int, default=N_BOOT,
                   help=f"parametric-bootstrap draws (default {N_BOOT})")
    p.add_argument("--seed", type=int, default=BOOT_SEED,
                   help=f"bootstrap RNG seed (default {BOOT_SEED})")
    p.add_argument("--drc-wdh-threshold-bps", type=float,
                   default=DRC_WDH_THRESHOLD_BPS)
    p.add_argument("--no-ci", action="store_true",
                   help="skip the parametric-bootstrap CI computation")
    p.add_argument("--point-tolerance", type=float, default=1e-6,
                   help="abs tolerance when checking refit β̂ against "
                        "bloomberg_overlay/huber_beta_by_field.csv.")
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    live_csv = args.overlay_dir / "live_field_releases_used.csv"
    needed = [*VARIANT_FILES.values(), live_csv.name]
    missing = [f for f in needed if not (args.overlay_dir / f).exists()]
    if missing:
        raise SystemExit(
            f"Missing inputs in {args.overlay_dir}: {missing}. "
            f"Run step_15_4_live_scoring/build_live_scoring_bloomberg.py first."
        )

    # ------------------------------------------------------------------
    # Stage 1 — refit β̂ + V_β on the historical events; cross-check β̂
    # against the upstream Bloomberg-overlay pipeline's β to confirm we
    # are scoring against the same coefficient layer.
    # ------------------------------------------------------------------
    mapping = pd.read_csv(args.mapping)
    field_ids = mapping["field_id"].tolist()
    fid_to_idx = {f: i for i, f in enumerate(field_ids)}

    for theme, fields in THEMES.items():
        bad = [f for f in fields if f not in fid_to_idx]
        if bad:
            raise SystemExit(f"Theme {theme!r} references unknown field(s): {bad}")

    beta_arr, L_full, fit_info = fit_beta_for_pipeline(
        args.ts_events_csv, field_ids, lam=args.lam, c=args.huber_c,
    )
    V_full = L_full @ L_full.T

    upstream_beta_csv = args.overlay_dir / "huber_beta_by_field.csv"
    if upstream_beta_csv.exists():
        upstream_beta = (
            pd.read_csv(upstream_beta_csv)
            .set_index("field_id")["beta_i_huber"]
            .reindex(field_ids)
            .to_numpy(dtype=float)
        )
        max_abs = float(np.max(np.abs(beta_arr - upstream_beta)))
        rel = max_abs / max(float(np.max(np.abs(upstream_beta))), 1e-30)
        if rel > args.point_tolerance:
            raise SystemExit(
                f"Refit β̂ disagrees with {upstream_beta_csv} "
                f"(max|Δ|={max_abs:.3e}, rel={rel:.3e}). Re-run "
                f"build_live_scoring_bloomberg.py first."
            )
        print(
            f"[refit] β̂ agrees with upstream: max|Δ|={max_abs:.3e} "
            f"(rel={rel:.3e}) ✓"
        )

    print(
        f"[refit] λ={args.lam:.6g}  c={args.huber_c}  "
        f"n_events_hist={fit_info['n_events']}  df_eff={fit_info['df_eff']:.2f}"
    )

    # ------------------------------------------------------------------
    # Stage 2 — load all variant field-level frames + hf_map.
    # ------------------------------------------------------------------
    variant_frames: dict[str, pd.DataFrame] = {}
    for variant, fname in VARIANT_FILES.items():
        df = pd.read_csv(
            args.overlay_dir / fname, parse_dates=["release_datetime_et"]
        )
        variant_frames[variant] = df

    live = pd.read_csv(live_csv, parse_dates=["release_datetime_et"])
    hf_map = (
        live[["release_datetime_et", "hf_return"]]
        .drop_duplicates("release_datetime_et")
        .set_index("release_datetime_et")["hf_return"]
        .to_dict()
    )

    # Theme-membership audit.
    membership: list[dict] = []
    for theme in THEME_ORDER:
        for f in THEMES[theme]:
            membership.append({"theme": theme, "field_id": f, "status": "active"})
    pd.DataFrame(membership).to_csv(
        args.out_dir / "theme_membership_bloomberg.csv", index=False
    )

    models = sorted(
        {m for df in variant_frames.values() for m in df["model"].unique()}
    )
    print(f"Models:   {models}")
    print(f"Themes:   {THEME_ORDER}")
    print(f"Variants: {VARIANT_ORDER}")

    # Pre-slice (β̂, L_theme) for each theme.
    theme_beta_L: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    for theme, fields in THEMES.items():
        idx = np.array([fid_to_idx[f] for f in fields], dtype=int)
        beta_theme = beta_arr[idx]
        V_theme = V_full[np.ix_(idx, idx)]
        L_theme = _cholesky_psd(V_theme)
        theme_beta_L[theme] = (beta_theme, L_theme, fields)

    drc_wdh_tau = args.drc_wdh_threshold_bps / 10000.0
    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # Stage 3 — point estimates + CIs per (model, theme, variant).
    # ------------------------------------------------------------------
    rows: list[dict] = []
    for model in models:
        for theme in THEME_ORDER:
            beta_theme, L_theme, fields = theme_beta_L[theme]
            row: dict = {
                "model": model, "theme": theme,
                "n_theme_fields": len(fields),
            }
            for variant in VARIANT_ORDER:
                df_m = variant_frames[variant]
                df_m = df_m[df_m["model"] == model]
                pt = _theme_point_metrics(df_m, fields, hf_map)
                for k in POINT_METRIC_KEYS:
                    row[f"{variant}_{k}"] = pt.get(k, float("nan"))

                if args.no_ci or df_m.empty:
                    continue
                ci = _theme_ci(
                    df_m, fields, hf_map, beta_theme, L_theme,
                    n_boot=args.n_boot, rng=rng, drc_wdh_tau=drc_wdh_tau,
                )
                if not ci:
                    continue
                for metric in CI_METRICS:
                    for tail in ("ci90_lo", "ci90_hi", "ci95_lo", "ci95_hi"):
                        key = f"{metric}_{tail}"
                        row[f"{variant}_{key}"] = ci.get(key, float("nan"))
                # Sanity: BMSC point from CI matches our earlier point.
                if not np.isnan(ci["BMSC_point"]) and not np.isnan(pt["BMSC"]):
                    delta = abs(ci["BMSC_point"] - pt["BMSC"])
                    if delta > 1e-9:
                        print(
                            f"[warn] BMSC point mismatch for "
                            f"{model}/{theme}/{variant}: "
                            f"point={pt['BMSC']:.6e} ci_point={ci['BMSC_point']:.6e}"
                        )
            rows.append(row)
    aggregate_df = pd.DataFrame(rows)
    aggregate_path = args.out_dir / "scores_by_theme_bloomberg_aggregate.csv"
    aggregate_df.to_csv(aggregate_path, index=False)

    # Convenience wide views: BMSC point by model x theme for each variant.
    for variant in VARIANT_ORDER:
        col = f"{variant}_BMSC"
        if col not in aggregate_df.columns:
            continue
        wide = aggregate_df.pivot(index="model", columns="theme", values=col)
        wide = wide.reindex(columns=THEME_ORDER)
        out_path = (
            args.out_dir / f"scores_by_theme_bloomberg_BMSC_wide_{variant}.csv"
        )
        wide.to_csv(out_path)

    # ------------------------------------------------------------------
    # Report.
    # ------------------------------------------------------------------
    L: list[str] = []
    A = L.append
    A("=" * 78)
    A("Step 15.5 (Bloomberg overlay) — theme-restricted scoring report")
    A("=" * 78)
    A(f"Overlay dir:   {args.overlay_dir}")
    A(f"Models:        {models}")
    A(f"Themes:        {THEME_ORDER}")
    A(f"Variants:      {VARIANT_ORDER}")
    A("")
    for theme in THEME_ORDER:
        A(f"  [{theme}]  fields = {THEMES[theme]}")
    A("")
    A(f"Huber refit:   λ={args.lam:.6g}  c={args.huber_c}  "
      f"n_events_hist={fit_info['n_events']}  df_eff={fit_info['df_eff']:.2f}")
    if not args.no_ci:
        A(f"Bootstrap:     n_boot={args.n_boot}, seed={args.seed}, "
          f"drc_wdh_τ={drc_wdh_tau:g}")
    else:
        A("Bootstrap:     SKIPPED (--no-ci)")
    A("")

    for variant in VARIANT_ORDER:
        col = f"{variant}_BMSC"
        if col not in aggregate_df.columns:
            continue
        A(f"-- {variant} BMSC by (model, theme) --")
        wide = aggregate_df.pivot(index="model", columns="theme", values=col)
        wide = wide.reindex(columns=THEME_ORDER)
        A(wide.round(4).to_string())
        A("")

    A("Notes:")
    A("  * Inputs come from build_live_scoring_bloomberg.py "
      "(`bloomberg_overlay/`).")
    A("  * Variants score_on_median_1d/3d/7d use per-event MEDIAN of (C_d, M_d)")
    A("    over d in {1}, {1,2,3}, {1..7}; final_vs_final uses Bloomberg's")
    A("    final-summary MEDIAN against the model's latest pre-release prediction.")
    A("  * BMSC, BP_RMSE, WDH are Addendum-3 bounded magnitude scores;")
    A("    BDRC, DRC_RMSE are bounded directional-return capture scores.")
    A("  * Theme-level BDRC is benchmarked against the realized announcement")
    A("    -window return r_g, which reflects ALL macro news at that timestamp,")
    A("    not just the theme's fields. Read BDRC as 'did the theme's")
    A("    predicted shock track the realized return?', not as a clean")
    A("    theme-isolated score.")
    A("  * CIs: parametric bootstrap of β^(b) ~ N(β̂_theme, V_β[theme, theme])")
    A("    using L_theme = chol(V_β[theme, theme]); same n_boot and seed as")
    A("    the upstream pipeline so layers are directly comparable.")
    out_text = "\n".join(L) + "\n"
    (args.out_dir / "scores_by_theme_bloomberg_report.txt").write_text(out_text)
    print(out_text)
    print(f"\nWrote outputs under: {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
