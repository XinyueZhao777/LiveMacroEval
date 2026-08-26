"""Step 15.4 (Bloomberg overlay) — live scoring (FINAL).

The headline live-scoring pipeline. Replaces the Investing.com-based
consensus C with the cumulative MEDIAN of Bloomberg economist surveys.
For each live release (field_id, ref_period, T = release_dt) covered by
Bloomberg, three score variants are computed using the same matched-
subset rule as score_pipeline_huber.py:

  score_on_median_1d : per-event (C_1, M_1) — the day-1 snapshot only.
  score_on_median_3d : per-event MEDIAN of (C_d, M_d) across d in {1..3}.
  score_on_median_7d : per-event MEDIAN of (C_d, M_d) across d in {1..7}.
  final_vs_final   : Bloomberg final-summary MEDIAN as C and the model's
                     latest pre-release prediction as M; this is the
                     apples-to-apples sibling of score_pipeline_huber.py's
                     `latest_pre_release` mode (only C differs).

For each `d` in {1..7}, M_d is the MEDIAN of the model's predictions
emitted on the calendar day `release_date - d days` (00:00-23:59 ET).
Days with zero predictions are dropped — no carry-forward. Each fresh
prediction incorporates the model's most up-to-date information, so the
day's voice is used rather than a stale carry-forward. C_d is the
cumulative-median Bloomberg consensus carried-forward to end of day d
(NaN -> drop).

The earlier `per_day_d` variant has been retired — its summary
(`mean_over_days_*`) is no longer the headline; if needed, the
intermediate `bloomberg_daily_field_releases.csv` and helpers
`per_day_scores`, `event_aggregate`, `metrics_with_bdrc` remain exported
for downstream callers (e.g., score_by_theme_bloomberg.py).

Outlier filter: every LLM is run through filter_outliers() with the
10x-per-(variable, target_month)-median rule (matches
`score_pipeline.filter_outliers` exactly). Per-model drop counts are
saved to outliers_dropped_<model>.csv in the output dir. Drop rates are
small (<3% on every model in the current run) — only true extremes are
removed.

Confidence intervals: parametric bootstrap of beta from
N(beta_hat_huber, V_huber_HC1) via score_pipeline.bootstrap_score_ci.
Computed for final_vs_final and the three score_on_median variants
(1d / 3d / 7d). Each variant feeds its own per-event (S, S_hat) field
table into the SAME bootstrap; only the (C, M) pair per event differs,
so bars across variants share the same beta-uncertainty model. Same
lambda, huber_c, seed, n_boot as huber_final/scoring_metadata.json.

See SCORE_CALCULATION.md (this folder) for the canonical reference.

Inputs
------
  - field_mapping.csv, sigma_by_field.csv, huber_final/huber_beta_by_field.csv,
    field_releases_live.csv (same as score_pipeline_huber.py).
  - Results/bloomberg_consensus/bloomberg_daily_consensus.csv (run
    build_daily_consensus.py first).
  - Results/bloomberg_consensus/bloomberg_release_consensus.csv.
  - timestamp_group_events.csv (for the Huber refit + V_huber).

Outputs (under <out_dir>, default bloomberg_overlay/)
-----------------------------------------------------------
  bloomberg_carry_forward_per_day.csv      Bloomberg C_d intermediate
  bloomberg_daily_field_releases.csv       per (model, field, event, d) intermediate
  bloomberg_score_on_median_field_1d.csv     per (model, field, event)
  bloomberg_score_on_median_field_3d.csv     per (model, field, event)
  bloomberg_score_on_median_field_7d.csv     per (model, field, event)
  bloomberg_final_vs_final_field.csv       per (model, field, event)
  bloomberg_aggregate_scores.csv           per (model) — point estimates
  bloomberg_final_vs_final_ci.csv          per (model) — CI table for final_vs_final
  bloomberg_score_on_median_1d_ci.csv      per (model) — CI table for score_on_median_1d
  bloomberg_score_on_median_3d_ci.csv      per (model) — CI table for score_on_median_3d
  bloomberg_score_on_median_7d_ci.csv      per (model) — CI table for score_on_median_7d
  bloomberg_live_scoring_report.txt
  bloomberg_live_scoring_metadata.json
  model_target_month_coverage.csv
  outliers_dropped_<model>.csv             per-model outlier rows + summary

Conda env: livemacro.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # Results/market_surprise_capture_score
REPO = ROOT.parent.parent  # livemacro repo root
REPO_RESULTS = ROOT.parent  # Results/

sys.path.insert(0, str(HERE))
from arima_predictions import (  # noqa: E402  -- only allowed helper import
    apply_arima_overrides,
    build_arima_predictions,
)

# ---------------------------------------------------------------------------
# Section A. Constants — paths, lambda, outlier filter config.
# ---------------------------------------------------------------------------
STEP_15_1 = ROOT / "step_15_1_mapping_layer"
STEP_15_2 = ROOT / "step_15_2_historical_preprocessing"

DEFAULT_MAPPING = STEP_15_1 / "field_mapping.csv"
DEFAULT_SIGMA = STEP_15_2 / "sigma_by_field.csv"
DEFAULT_TS_EVENTS = STEP_15_2 / "timestamp_group_events.csv"
DEFAULT_LIVE_GT = REPO_RESULTS / "ground_truth" / "data" / "field_releases_live.csv"
DEFAULT_COVERAGE = REPO_RESULTS / "data_sp500futures" / "event_window_coverage.csv"
DEFAULT_MODELS_ROOT = REPO / "Results" / "data_from_serverA_serverB" / "final_analysis_data"
DEFAULT_ARIMA_CSV = (
    REPO_RESULTS
    / "benchmark_econ"
    / "results_targets_2025_11_2026_03_aic"
    / "target_month_forecasts.csv"
)
DEFAULT_BLOOMBERG_DIR = REPO_RESULTS / "bloomberg_consensus"
DEFAULT_DAILY_CSV = DEFAULT_BLOOMBERG_DIR / "bloomberg_daily_consensus.csv"
DEFAULT_RELEASE_CSV = DEFAULT_BLOOMBERG_DIR / "bloomberg_release_consensus.csv"

ARIMA_MODEL_NAME = "arima_aic"

# Quarterly fields: ref_period is 'YYYY-Qn'. Match by expanding the quarter
# into its 3 constituent YYYY-MM months and requiring target_month ∈ that set.
QUARTERLY_FIELD_IDS = {"real_gdp_advance"}

# Days before release scored. d=1 means "1 day before release" with cutoff
# at release_dt - 1d + 23:59:59, etc.
DAY_OFFSETS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
MODEL_HISTORY_WINDOW = pd.Timedelta(days=14)

# Outlier filter (pre-Huber-fit predictions): drop where
# |log10(|v| / median_per_(variable, target_month))| > log10_threshold.
# Default 1.0 ⇒ 10x deviation. Conservative; only true extremes removed.
OUTLIER_LOG10_THRESHOLD = 1.0
GPT_OUTLIER_MODELS = {"gpt-5-search-api"}  # historical reference; filter is applied to ALL LLMs

# Huber-ridge regression hyperparameters (β + V_β for live scoring + CIs).
# c = 1.345 is the Huber tuning constant for 95% Gaussian efficiency
# (statsmodels.RLM default). λ is the L2 penalty on β; selected by blocked
# chronological 5-fold CV with Huber prediction loss on the historical
# `timestamp_group_events.csv` under the NO-INTERCEPT design (λ grid
# np.logspace(-4, 4, 25); λ_best is argmin CV(λ)). The CV run record is
# `cv_lambda_no_intercept.{py,csv,json}` in this directory. The constant
# is hard-coded here so the pipeline is fully self-contained (no need to
# read a precomputed JSON) and successive runs are bit-for-bit reproducible.
HUBER_C = 1.345
HUBER_LAMBDA = 21.544346900318832  # = 10**(4/3); λ_best from no-intercept CV

# Bootstrap CI defaults.
N_BOOT = 10000
BOOT_SEED = 20260427
DRC_WDH_THRESHOLD_BPS = 1.0  # for DRC_WDH_tau / active_share


# ---------------------------------------------------------------------------
# Section B. Field-mapping + IO helpers (inlined from build_live_scoring.py).
# ---------------------------------------------------------------------------
def _quarter_months(ref_period: str) -> tuple[str, ...]:
    """Expand 'YYYY-Qn' into a tuple of its three YYYY-MM months."""
    year_s, q_s = ref_period.rsplit("-Q", 1)
    q = int(q_s)
    if q not in (1, 2, 3, 4):
        raise ValueError(f"Invalid quarter in ref_period {ref_period!r}")
    year = int(year_s)
    start = 3 * q - 2
    return tuple(f"{year}-{m:02d}" for m in range(start, start + 3))


def apply_transform(value: float, rule: str) -> float:
    """Map raw model output to field calendar units. Mirrors
    step_15_1.build_mapping.apply_transform."""
    if pd.isna(value):
        return value
    if rule == "identity":
        return float(value)
    if rule.startswith("multiply:"):
        k = float(rule.split(":", 1)[1])
        return float(value) * k
    raise ValueError(f"Unsupported model_transform_rule: {rule!r}")


def load_model_predictions(
    model_dir: Path,
    date_only_policy: str = "midnight",
) -> tuple[pd.DataFrame, int]:
    """Concatenate every CSV under `model_dir` into one long frame.

    Returns (df, n_date_only) with columns: timestamp_local (naive ET),
    target_month, release_month, variable, value, source_csv,
    timestamp_is_date_only. Only parsed_ok==True rows with a numeric
    value are kept.

    `date_only_policy` controls how 'YYYY-MM-DD'-only timestamps are
    interpreted:
      'drop'     -> remove
      'eod'      -> 23:59:59 ET of that date
      'midnight' -> 00:00:00 ET of that date  (default)
    """
    files = sorted(glob.glob(str(model_dir / "*.csv")))
    if not files:
        raise RuntimeError(f"No CSVs under {model_dir}")
    frames: list[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(f)
        df["source_csv"] = Path(f).name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)

    required = {"timestamp_local", "target_month", "release_month",
                "variable", "value", "parsed_ok"}
    missing = required - set(out.columns)
    if missing:
        raise RuntimeError(
            f"Model {model_dir.name}: missing required columns {sorted(missing)}"
        )

    ts_str = out["timestamp_local"].astype(str)
    date_only_mask = ts_str.str.match(r"^\d{4}-\d{2}-\d{2}$")
    out["timestamp_is_date_only"] = date_only_mask
    n_date_only = int(date_only_mask.sum())
    out["timestamp_local"] = pd.to_datetime(out["timestamp_local"], format="mixed")

    if n_date_only > 0:
        if date_only_policy == "drop":
            out = out[~date_only_mask].copy()
        elif date_only_policy == "eod":
            bump = pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            out.loc[date_only_mask, "timestamp_local"] = (
                out.loc[date_only_mask, "timestamp_local"] + bump
            )
        elif date_only_policy == "midnight":
            pass
        else:
            raise ValueError(
                f"Unknown date_only_policy {date_only_policy!r}; "
                f"choose 'drop', 'eod', or 'midnight'."
            )

    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["target_month"] = out["target_month"].astype(str)
    out = out[out["parsed_ok"].astype(bool) & out["value"].notna()].copy()
    out.reset_index(drop=True, inplace=True)
    return out, n_date_only


def build_live_releases(
    live_gt_csv: Path,
    coverage_csv: Path,
    live_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (live_releases, coverage, drop_stats). Mirrors
    build_live_scoring.build_live_releases. Live GT must contain only
    releases at or after live_start (outline §14.1).
    """
    gt = pd.read_csv(live_gt_csv, parse_dates=["release_datetime_et"])
    coverage = pd.read_csv(coverage_csv, parse_dates=["release_datetime_et"])

    n_input = int(len(gt))
    pre_live = gt[gt["release_datetime_et"] < live_start]
    n_before_live_start = int(len(pre_live))
    if n_before_live_start:
        bad = pre_live[["field_id", "release_datetime_et", "event"]].head(10)
        raise RuntimeError(
            f"Live ground-truth file {live_gt_csv} has {n_before_live_start} "
            f"row(s) before live_start={live_start.date()}. Historical data must "
            f"not enter live scoring (outline §14.1):\n{bad.to_string(index=False)}"
        )

    # Rows where Investing.com's `C` is missing are NOT dropped: Bloomberg
    # consensus (joined later via bloomberg_release_consensus.csv) is the
    # actual scoring source. We log these rows for diagnostics; their
    # `C` and `raw_surprise` stay NaN and only `C_investing_used_in_existing_score`
    # is affected downstream.
    missing_investing_C = gt["C"].isna()
    no_investing_consensus_rows = gt[missing_investing_C][
        ["field_id", "release_datetime_et", "event", "ref_period", "A"]
    ].copy()

    cov = coverage.rename(columns={"valid": "futures_valid"})[[
        "release_datetime_et", "contract_code", "start_bar_close",
        "end_bar_close", "futures_valid",
    ]]
    gt = gt.merge(cov, on="release_datetime_et", how="left")
    has_cov = gt["futures_valid"].fillna(False).astype(bool)
    gt["hf_return"] = np.where(
        has_cov & gt["start_bar_close"].notna() & gt["end_bar_close"].notna(),
        np.log(gt["end_bar_close"].astype(float))
        - np.log(gt["start_bar_close"].astype(float)),
        np.nan,
    )
    has_hf = gt["hf_return"].notna()
    missing_futures = gt[~has_hf][[
        "field_id", "release_datetime_et", "event", "ref_period",
        "A", "C", "futures_valid",
    ]].copy()

    fr = gt.copy()
    if fr.empty:
        raise RuntimeError(
            f"No valid live releases in {live_gt_csv} (post-live_start={live_start.date()})."
        )
    fr["release_time_cohort"] = fr["release_datetime_et"].dt.strftime("%H:%M")
    fr = fr[[
        "field_id", "release_datetime_et", "event", "ref_period",
        "A", "C", "raw_surprise", "hf_return", "contract_code",
        "release_time_cohort",
    ]].sort_values(
        ["release_datetime_et", "field_id"], kind="mergesort"
    ).reset_index(drop=True)

    drop_stats = {
        "n_input": n_input,
        "n_before_live_start": n_before_live_start,
        "n_kept_no_investing_consensus": int(len(no_investing_consensus_rows)),
        "n_kept": int(len(fr)),
        "n_missing_futures": int(len(missing_futures)),
        "no_investing_consensus_rows": no_investing_consensus_rows,
        "missing_futures_rows": missing_futures,
    }
    return fr, coverage, drop_stats


# ---------------------------------------------------------------------------
# Section C. Outlier filter (10x per-(variable, target_month) median).
# ---------------------------------------------------------------------------
def filter_outliers(
    preds: pd.DataFrame, log10_threshold: float = OUTLIER_LOG10_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop predictions whose |value| deviates by >10x from the per-
    (variable, target_month) median. Returns (kept, dropped)."""
    df = preds.copy()
    df["_abs"] = df["value"].abs()
    drop_flag = pd.Series(False, index=df.index)
    for (_var, _tm), grp in df.groupby(["variable", "target_month"], sort=False):
        nonzero = grp[grp["_abs"] > 0]
        if len(nonzero) == 0:
            continue
        med = float(nonzero["_abs"].median())
        if med <= 0:
            continue
        log_ratio = np.log10(grp["_abs"].where(grp["_abs"] > 0, np.nan) / med)
        outlier_log = log_ratio.abs() > log10_threshold
        zero_mask = grp["_abs"] == 0
        outlier_zero = zero_mask & (med > 0)
        is_outlier = outlier_log.fillna(False) | outlier_zero
        drop_flag.loc[grp.index] = is_outlier.values
    kept = df[~drop_flag].drop(columns=["_abs"]).copy()
    dropped = df[drop_flag].drop(columns=["_abs"]).copy()
    return kept, dropped


# ---------------------------------------------------------------------------
# Section D. Huber-ridge fit + HC1 sandwich variance for β.
# Inlined from score_pipeline_huber.py. The historical regression of
# announcement-window returns r_g on standardized surprises X_g uses
# Huber loss ρ_δ with δ adapted from rescaled MAD.
#
# NO INTERCEPT: per Gürkaynak, Kısacıkoğlu & Wright (2018) "Missing Events
# in Event Studies", every event-study regression specification in the
# paper (eqs. 2.1, 2.3, 3.1, 3.2, 4.1, 4.2) is of the form
#     y_t = β' s_t + ε_t,
# with NO constant term. Under the null "no surprise → no expected return,"
# there is no baseline drift to absorb in the announcement window. We
# therefore fit β alone, without centering X or y.
#
# Sandwich variance: V = (n / n_eff) · A⁻¹ B A⁻¹ with
#   A = X' W X + λI                      (penalized Hessian under IRLS weights)
#   B = X' diag(ψ²(u)) X                 (score covariance)
#   df_eff = trace(weighted hat matrix)
# ---------------------------------------------------------------------------
def _huber_weights(u: np.ndarray, delta: float) -> np.ndarray:
    """ψ_δ(u)/u; 1 inside the linear region, δ/|u| outside."""
    abs_u = np.abs(u)
    out = abs_u > delta
    safe = np.where(out, abs_u, 1.0)
    return np.where(out, delta / safe, 1.0)


def _huber_psi(u: np.ndarray, delta: float) -> np.ndarray:
    """Huber influence ψ_δ(u) = clip(u, -δ, +δ)."""
    return np.clip(u, -delta, delta)


def _mad_scale(u: np.ndarray) -> float:
    """Rescaled MAD: σ̂ = 1.4826·median(|u − median(u)|)."""
    med = float(np.median(u))
    return 1.4826 * float(np.median(np.abs(u - med)))


def fit_huber_ridge(
    X: np.ndarray, y: np.ndarray, lam: float,
    c: float = HUBER_C, max_iter: int = 100, tol: float = 1e-7,
    sigma_floor: float = 1e-12,
) -> dict:
    """Closed-form Huber-ridge with NO intercept, IRLS.

    Solves   min_β   Σ ρ_δ(y_i − x_i'β)  +  λ ‖β‖²
    via IRLS:        (X' W X + λI) β = X' W y       at each iteration.

    The intercept is intentionally omitted (see module docstring above for
    the paper-based rationale). `alpha=0.0` is returned in the result dict
    purely for backward compatibility with downstream callers (the
    `_with_alpha` post-processor that lives in this directory) — it is
    NOT an estimated parameter.

    Returns {alpha, beta, sigma, delta, weights, residuals, n_iter, converged}."""
    n, p = X.shape
    if n < p + 1:
        raise ValueError(f"Need n > p, got n={n}, p={p}.")

    # Initial β via plain OLS-ridge (no intercept).
    A = X.T @ X + lam * np.eye(p)
    beta = np.linalg.solve(A, X.T @ y)

    converged, n_iter = False, 0
    for it in range(max_iter):
        u = y - X @ beta
        sigma = _mad_scale(u)
        if sigma < sigma_floor:
            converged, n_iter = True, it + 1
            break
        delta = c * sigma
        w = _huber_weights(u, delta)

        XtW = X * w[:, None]
        A = X.T @ XtW + lam * np.eye(p)
        new_beta = np.linalg.solve(A, XtW.T @ y)

        d = np.linalg.norm(new_beta - beta) / max(np.linalg.norm(beta), 1e-12)
        beta = new_beta
        n_iter = it + 1
        if d < tol:
            converged = True
            break

    u = y - X @ beta
    sigma = _mad_scale(u)
    delta = c * max(sigma, sigma_floor)
    w = _huber_weights(u, delta)
    return {
        "alpha": 0.0, "beta": beta,
        "sigma": float(sigma), "delta": float(delta),
        "weights": w, "residuals": u,
        "n_iter": n_iter, "converged": converged,
    }


def huber_ridge_sandwich(
    X: np.ndarray, lam: float, fit: dict,
) -> tuple[np.ndarray, float]:
    """HC1-equivalent sandwich variance V for the converged no-intercept
    Huber-ridge fit. Returns (V, df_eff)."""
    n, p = X.shape
    delta = fit["delta"]
    w = fit["weights"]
    u = fit["residuals"]

    XtW = X * w[:, None]
    A = X.T @ XtW + lam * np.eye(p)
    A_inv = np.linalg.inv(A)

    psi = _huber_psi(u, delta)
    XtPsi2X = (X * (psi ** 2)[:, None]).T @ X

    H = X @ A_inv @ XtW.T
    df_eff = float(np.trace(H))
    n_eff = max(int(round(n - df_eff)), 1)
    V = (n / n_eff) * (A_inv @ XtPsi2X @ A_inv)
    V = 0.5 * (V + V.T)
    return V, df_eff


def fit_beta_for_pipeline(
    ts_events_csv: Path,
    field_ids: list[str],
    lam: float = HUBER_LAMBDA,
    c: float = HUBER_C,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Refit Huber-ridge on the historical events. Returns
    (beta_hat, L_chol, info) where L_chol is the Cholesky factor of the
    HC1 sandwich V_β used by the parametric bootstrap.

    `field_ids` orders the regression columns. The historical events
    CSV must contain X_<field_id> for each entry.

    Determinism: IRLS from a fixed OLS-ridge initialization on the same
    X, y, λ produces bit-identical β across runs.
    """
    events = pd.read_csv(ts_events_csv, parse_dates=["release_timestamp"])
    events = events.sort_values("release_timestamp").reset_index(drop=True)
    surprise_cols = [f"X_{f}" for f in field_ids]
    missing = [c for c in surprise_cols if c not in events.columns]
    if missing:
        raise SystemExit(
            f"timestamp_group_events.csv missing surprise columns "
            f"for {len(missing)} field_ids (e.g., {missing[:3]}). "
            f"Re-run step_15_2_historical_preprocessing."
        )
    X = events[surprise_cols].to_numpy(dtype=float)
    y = events["hf_return"].to_numpy(dtype=float)

    fit = fit_huber_ridge(X, y, lam, c=c)
    V, df_eff = huber_ridge_sandwich(X, lam, fit)
    beta = fit["beta"]
    try:
        L = np.linalg.cholesky(V)
    except np.linalg.LinAlgError:
        wval, U = np.linalg.eigh(V)
        L = U * np.sqrt(np.maximum(wval, 0.0))[None, :]

    se = np.sqrt(np.diag(V))
    info = {
        "lambda": float(lam),
        "huber_c": float(c),
        "n_events": int(len(events)),
        "first_event": str(events["release_timestamp"].iloc[0]),
        "last_event": str(events["release_timestamp"].iloc[-1]),
        "alpha": float(fit["alpha"]),
        "sigma": float(fit["sigma"]),
        "delta": float(fit["delta"]),
        "df_eff": float(df_eff),
        "n_iter": int(fit["n_iter"]),
        "converged": bool(fit["converged"]),
        "beta_by_field": {f: float(b) for f, b in zip(field_ids, beta)},
        "marginal_se_HC1": {f: float(s) for f, s in zip(field_ids, se)},
    }
    return beta, L, info


# ---------------------------------------------------------------------------
# Section E. Parametric-bootstrap CI machinery (β̂ + L → bounded scores).
# Inlined from score_pipeline.bootstrap_score_ci. β^(b) = β̂ + L · z,
# z ~ N(0, I_p), b = 1..n_boot. Magnitude scores are bounded in [-1, 1];
# directional scores are return-weighted hit rates.
# ---------------------------------------------------------------------------
def _pivot_event_field_matrix(
    field_df: pd.DataFrame, field_ids: list[str], col: str,
) -> tuple[np.ndarray, np.ndarray]:
    events = sorted(field_df["release_datetime_et"].unique())
    fid_to_j = {f: j for j, f in enumerate(field_ids)}
    M = np.zeros((len(events), len(field_ids)))
    ev_to_i = {e: i for i, e in enumerate(events)}
    for r in field_df.itertuples(index=False):
        i = ev_to_i[r.release_datetime_et]
        j = fid_to_j[r.field_id]
        M[i, j] = float(getattr(r, col))
    return M, np.array(events)


def _per_event_returns(field_df: pd.DataFrame) -> np.ndarray:
    return (
        field_df.groupby("release_datetime_et")["hf_return"]
        .first().sort_index().to_numpy(dtype=float)
    )


def _q(a: np.ndarray, q: float) -> float:
    return float(np.quantile(a, q)) if len(a) else float("nan")


def _empty_ci(n_events: int = 0) -> dict:
    nan = float("nan")
    base = {
        "n_events": n_events, "n_drc_events": 0, "n_boot_kept": 0,
    }
    for k in ("BMSC", "BDRC"):
        base[f"{k}_point"] = nan
        base[f"{k}_median_boot"] = nan
        base[f"{k}_ci95_lo"] = nan
        base[f"{k}_ci95_hi"] = nan
        base[f"{k}_ci90_lo"] = nan
        base[f"{k}_ci90_hi"] = nan
        base[f"p_one_sided_{k}_le_0"] = nan
    for k in ("BP_RMSE", "WDH", "DRC_RMSE", "DRC_WDH", "DRC_WDH_tau",
              "active_share"):
        base[f"{k}_point"] = nan
        base[f"{k}_ci95_lo"] = nan
        base[f"{k}_ci95_hi"] = nan
        base[f"{k}_ci90_lo"] = nan
        base[f"{k}_ci90_hi"] = nan
    base["p_one_sided_DRC_WDH_le_half"] = nan
    base["p_one_sided_DRC_WDH_tau_le_half"] = nan
    return base


def bootstrap_score_ci(
    field_df: pd.DataFrame,
    field_ids: list[str],
    beta_hat: np.ndarray,
    L: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    drc_wdh_tau: float = 1e-4,
) -> dict:
    """Joint parametric bootstrap of BMSC + BDRC families under
    β^(b) ~ N(β̂, L L'). Magnitude scores follow the bounded Addendum 3
    forms (in [-1, 1]); directional scores are return-weighted hit rates.
    """
    if field_df.empty:
        return _empty_ci()
    Xreal, _ = _pivot_event_field_matrix(field_df, field_ids, "S")
    Xhat, _ = _pivot_event_field_matrix(field_df, field_ids, "S_hat")
    r = _per_event_returns(field_df)
    n_events = Xreal.shape[0]

    valid_r = ~np.isnan(r)
    n_drc = int(valid_r.sum())
    Xhat_drc = Xhat[valid_r]
    r_drc = r[valid_r]
    sum_r2 = float((r_drc ** 2).sum())
    abs_r = np.abs(r_drc)
    sign_r = np.sign(r_drc)
    sum_abs_r = float(abs_r.sum())

    # Point estimates.
    Q_pt = Xreal @ beta_hat
    Qh_pt = Xhat @ beta_hat
    Qh_drc_pt = Xhat_drc @ beta_hat
    sum_e_msc_pt = float(((Qh_pt - Q_pt) ** 2).sum())
    sum_e_drc_pt = float(((r_drc - Qh_drc_pt) ** 2).sum())
    sum_Q2_pt = float((Q_pt ** 2).sum())
    bmsc_denom_pt = sum_Q2_pt + sum_e_msc_pt
    bdrc_denom_pt = sum_r2 + sum_e_drc_pt
    bmsc_pt = (sum_Q2_pt - sum_e_msc_pt) / bmsc_denom_pt if bmsc_denom_pt > 0 else float("nan")
    bdrc_pt = (sum_r2 - sum_e_drc_pt) / bdrc_denom_pt if bdrc_denom_pt > 0 else float("nan")
    bp_msc_pt = 10000.0 * float(np.sqrt(sum_e_msc_pt / n_events)) if n_events > 0 else float("nan")
    drc_rmse_pt = 10000.0 * float(np.sqrt(sum_e_drc_pt / n_drc)) if n_drc > 0 else float("nan")

    abs_Q_pt = np.abs(Q_pt)
    sum_abs_Q_pt = float(abs_Q_pt.sum())
    if sum_abs_Q_pt > 0:
        match_msc_pt = (np.sign(Q_pt) == np.sign(Qh_pt)).astype(float)
        wdh_pt = float((abs_Q_pt * match_msc_pt).sum() / sum_abs_Q_pt)
    else:
        wdh_pt = float("nan")

    if sum_abs_r > 0:
        match_drc_pt = (np.sign(Qh_drc_pt) == sign_r).astype(float)
        drc_wdh_pt = float((abs_r * match_drc_pt).sum() / sum_abs_r)
        active_pt = (np.abs(Qh_drc_pt) > drc_wdh_tau)
        active_share_pt = float(active_pt.mean())
        if active_pt.any():
            num_tau = float((abs_r[active_pt] *
                             (np.sign(Qh_drc_pt[active_pt]) == sign_r[active_pt]).astype(float)).sum())
            den_tau = float(abs_r[active_pt].sum())
            drc_wdh_tau_pt = num_tau / den_tau if den_tau > 0 else float("nan")
        else:
            drc_wdh_tau_pt = float("nan")
    else:
        drc_wdh_pt = drc_wdh_tau_pt = active_share_pt = float("nan")

    # Vectorized bootstrap.
    p = beta_hat.shape[0]
    Z = rng.standard_normal((n_boot, p))
    beta_draws = beta_hat[None, :] + Z @ L.T
    Q_b = Xreal @ beta_draws.T
    Qh_b = Xhat @ beta_draws.T
    Qh_b_drc = Xhat_drc @ beta_draws.T

    sum_e_msc = ((Qh_b - Q_b) ** 2).sum(axis=0)
    sum_e_drc = ((r_drc[:, None] - Qh_b_drc) ** 2).sum(axis=0)
    sum_Q2 = (Q_b ** 2).sum(axis=0)

    bmsc_denom_b = sum_Q2 + sum_e_msc
    bmsc_b = np.full(n_boot, np.nan)
    valid_bmsc = bmsc_denom_b > 0
    bmsc_b[valid_bmsc] = (sum_Q2[valid_bmsc] - sum_e_msc[valid_bmsc]) / bmsc_denom_b[valid_bmsc]
    bdrc_b = np.full(n_boot, np.nan)
    bdrc_denom_b = sum_r2 + sum_e_drc
    valid_bdrc = bdrc_denom_b > 0
    bdrc_b[valid_bdrc] = (sum_r2 - sum_e_drc[valid_bdrc]) / bdrc_denom_b[valid_bdrc]

    bp_msc_b = 10000.0 * np.sqrt(sum_e_msc / n_events)
    drc_rmse_b = 10000.0 * np.sqrt(sum_e_drc / n_drc) if n_drc > 0 else np.full(n_boot, np.nan)

    abs_Q_b = np.abs(Q_b)
    match_msc_b = (np.sign(Q_b) == np.sign(Qh_b)).astype(float)
    num_wdh = (abs_Q_b * match_msc_b).sum(axis=0)
    den_wdh = abs_Q_b.sum(axis=0)
    wdh_b = np.full(n_boot, np.nan)
    np.divide(num_wdh, den_wdh, out=wdh_b, where=den_wdh > 0)

    match_drc_b = (np.sign(Qh_b_drc) == sign_r[:, None]).astype(float)
    if sum_abs_r > 0:
        drc_wdh_b = (abs_r[:, None] * match_drc_b).sum(axis=0) / sum_abs_r
    else:
        drc_wdh_b = np.full(n_boot, np.nan)

    active_b = np.abs(Qh_b_drc) > drc_wdh_tau
    abs_r_active = abs_r[:, None] * active_b
    num_tau_b = (abs_r_active * match_drc_b).sum(axis=0)
    den_tau_b = abs_r_active.sum(axis=0)
    drc_wdh_tau_b = np.full(n_boot, np.nan)
    np.divide(num_tau_b, den_tau_b, out=drc_wdh_tau_b, where=den_tau_b > 0)
    active_share_b = active_b.mean(axis=0) if n_drc > 0 else np.full(n_boot, np.nan)

    bmsc_kept = bmsc_b[~np.isnan(bmsc_b)]
    bdrc_kept = bdrc_b[~np.isnan(bdrc_b)]
    wdh_kept = wdh_b[~np.isnan(wdh_b)]
    drc_wdh_kept = drc_wdh_b[~np.isnan(drc_wdh_b)]
    drc_wdh_tau_kept = drc_wdh_tau_b[~np.isnan(drc_wdh_tau_b)]

    return {
        "n_events": int(n_events), "n_drc_events": int(n_drc),
        "BMSC_point": bmsc_pt, "BMSC_median_boot": _q(bmsc_kept, 0.5),
        "BMSC_ci95_lo": _q(bmsc_kept, 0.025), "BMSC_ci95_hi": _q(bmsc_kept, 0.975),
        "BMSC_ci90_lo": _q(bmsc_kept, 0.05),  "BMSC_ci90_hi": _q(bmsc_kept, 0.95),
        "p_one_sided_BMSC_le_0": float((bmsc_kept <= 0).mean()) if len(bmsc_kept) else float("nan"),
        "BP_RMSE_point": bp_msc_pt,
        "BP_RMSE_ci95_lo": _q(bp_msc_b, 0.025), "BP_RMSE_ci95_hi": _q(bp_msc_b, 0.975),
        "BP_RMSE_ci90_lo": _q(bp_msc_b, 0.05),  "BP_RMSE_ci90_hi": _q(bp_msc_b, 0.95),
        "WDH_point": wdh_pt,
        "WDH_ci95_lo": _q(wdh_kept, 0.025), "WDH_ci95_hi": _q(wdh_kept, 0.975),
        "WDH_ci90_lo": _q(wdh_kept, 0.05),  "WDH_ci90_hi": _q(wdh_kept, 0.95),
        "BDRC_point": bdrc_pt, "BDRC_median_boot": _q(bdrc_kept, 0.5),
        "BDRC_ci95_lo": _q(bdrc_kept, 0.025), "BDRC_ci95_hi": _q(bdrc_kept, 0.975),
        "BDRC_ci90_lo": _q(bdrc_kept, 0.05),  "BDRC_ci90_hi": _q(bdrc_kept, 0.95),
        "p_one_sided_BDRC_le_0": float((bdrc_kept <= 0).mean()) if len(bdrc_kept) else float("nan"),
        "DRC_RMSE_point": drc_rmse_pt,
        "DRC_RMSE_ci95_lo": _q(drc_rmse_b, 0.025), "DRC_RMSE_ci95_hi": _q(drc_rmse_b, 0.975),
        "DRC_RMSE_ci90_lo": _q(drc_rmse_b, 0.05),  "DRC_RMSE_ci90_hi": _q(drc_rmse_b, 0.95),
        "DRC_WDH_point": drc_wdh_pt,
        "DRC_WDH_ci95_lo": _q(drc_wdh_kept, 0.025), "DRC_WDH_ci95_hi": _q(drc_wdh_kept, 0.975),
        "DRC_WDH_ci90_lo": _q(drc_wdh_kept, 0.05),  "DRC_WDH_ci90_hi": _q(drc_wdh_kept, 0.95),
        "p_one_sided_DRC_WDH_le_half": float((drc_wdh_kept <= 0.5).mean()) if len(drc_wdh_kept) else float("nan"),
        "DRC_WDH_tau_point": drc_wdh_tau_pt,
        "DRC_WDH_tau_ci95_lo": _q(drc_wdh_tau_kept, 0.025), "DRC_WDH_tau_ci95_hi": _q(drc_wdh_tau_kept, 0.975),
        "DRC_WDH_tau_ci90_lo": _q(drc_wdh_tau_kept, 0.05),  "DRC_WDH_tau_ci90_hi": _q(drc_wdh_tau_kept, 0.95),
        "p_one_sided_DRC_WDH_tau_le_half": float((drc_wdh_tau_kept <= 0.5).mean()) if len(drc_wdh_tau_kept) else float("nan"),
        "active_share_point": active_share_pt,
        "active_share_ci95_lo": _q(active_share_b, 0.025) if len(active_share_b) else float("nan"),
        "active_share_ci95_hi": _q(active_share_b, 0.975) if len(active_share_b) else float("nan"),
        "active_share_ci90_lo": _q(active_share_b, 0.05) if len(active_share_b) else float("nan"),
        "active_share_ci90_hi": _q(active_share_b, 0.95) if len(active_share_b) else float("nan"),
        "n_boot_kept": int(min(len(bmsc_kept), len(bdrc_kept))),
    }


# ---------------------------------------------------------------------------
# Bloomberg daily consensus -> carry-forward helper.
# ---------------------------------------------------------------------------
def build_bloomberg_carry_forward(
    daily: pd.DataFrame,
    day_offsets: tuple[int, ...] = DAY_OFFSETS,
    consensus_aggregator: str = "median",
) -> pd.DataFrame:
    """For every (field_id, ref_period) and every day_offset, return the
    cumulative consensus value carried forward to end of day d.

    consensus_aggregator selects which cumulative reduction is the C_d:
      'mean'   : cumulative mean across estimates with as_of <= end_of_day_d
      'median' : cumulative median (matches investing.com's own consensus
                 for cpi/pce_price_index/ppi/real_pce/retail_sales/
                 unemployment_rate per the empirical check; recommended).

    Both columns are present in the input `daily` frame
    (bloomberg_consensus_calendar and bloomberg_median_calendar). We choose
    one and emit it as `C_d_calendar`.

    Output columns:
        field_id, ref_period, release_datetime_et, day_offset,
        end_of_day_d, C_d_calendar (NaN if no estimate yet),
        n_to_date_d, asof_date_used, consensus_aggregator.
    """
    if consensus_aggregator == "mean":
        col = "bloomberg_consensus_calendar"
    elif consensus_aggregator == "median":
        col = "bloomberg_median_calendar"
    else:
        raise ValueError(f"Unknown consensus_aggregator {consensus_aggregator!r}")

    rows: list[dict] = []
    daily = daily.copy()
    daily["asof_date"] = pd.to_datetime(daily["asof_date"]).dt.normalize()
    daily["release_datetime_et"] = pd.to_datetime(daily["release_datetime_et"])
    grouped = daily.groupby(["field_id", "ref_period"], sort=False)
    for (fid, ref), sub in grouped:
        sub = sub.sort_values("asof_date", kind="mergesort").reset_index(drop=True)
        rel = sub["release_datetime_et"].iloc[0]
        rel_date = rel.normalize()
        for d in day_offsets:
            end_d = rel_date - pd.Timedelta(days=d) + pd.Timedelta(
                hours=23, minutes=59, seconds=59
            )
            end_d_date = end_d.normalize()
            mask = sub["asof_date"] <= end_d_date
            if mask.any():
                row = sub[mask].iloc[-1]
                rows.append({
                    "field_id": fid,
                    "ref_period": ref,
                    "release_datetime_et": rel,
                    "day_offset": d,
                    "end_of_day_d": end_d,
                    "C_d_calendar": float(row[col]),
                    "n_to_date_d": int(row["n_to_date"]),
                    "asof_date_used": row["asof_date"],
                    "consensus_aggregator": consensus_aggregator,
                })
            else:
                rows.append({
                    "field_id": fid,
                    "ref_period": ref,
                    "release_datetime_et": rel,
                    "day_offset": d,
                    "end_of_day_d": end_d,
                    "C_d_calendar": float("nan"),
                    "n_to_date_d": 0,
                    "asof_date_used": pd.NaT,
                    "consensus_aggregator": consensus_aggregator,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-day model aggregator.
#
# Aggregator modes:
#
#   "median_of_day" (default) : median of model predictions whose
#              timestamp_local lies in [cutoff.normalize(), cutoff], i.e.,
#              between 00:00:00 and 23:59:59 ET of the calendar day d.
#              Returns None (drop) if zero predictions on that calendar
#              day. This is the canonical per_day_d semantics — each fresh
#              prediction incorporates the most current information, so
#              we want the day's voice rather than a stale carry-forward.
#
#   "latest" : the SINGLE prediction whose timestamp is closest to (and <=)
#              the cutoff. Matches the existing pipeline's
#              latest_pre_release / cutoff_offset modes; used for the
#              final_vs_final variant. Carry-forward semantics (no lower
#              bound on timestamp).
#
#   "mean" / "median" : aggregator over a window
#              (cutoff - max_history, cutoff]. Sensitivity-only; not the
#              canonical mode. Useful for smoothing high-frequency models.
#
# `cutoff` semantics:
#   per-day variant  : cutoff = end_of_day_d  (release_date - d days + 23:59:59)
#   final_vs_final   : cutoff = release_dt - 1s  (latest strictly before release)
# ---------------------------------------------------------------------------
def aggregate_model_per_day(
    field_id: str,
    model_output_source: str,
    ref_period: str,
    release_dt: pd.Timestamp,
    cutoff: pd.Timestamp,
    max_history: pd.Timedelta,
    preds_by_var: dict[str, pd.DataFrame],
    aggregator: str = "median_of_day",
) -> dict | None:
    """Return a summary dict for the model's M at the given cutoff.

    For aggregator='latest', max_history is ignored (matching
    latest_pre_release behaviour). For aggregator in ('mean','median'),
    max_history defines the cumulative window (cutoff - max_history, cutoff].
    For aggregator='median_of_day', the window is the calendar day of
    `cutoff` only: [cutoff.normalize(), cutoff].
    """
    sub = preds_by_var.get(model_output_source)
    if sub is None or sub.empty:
        return None
    if field_id in QUARTERLY_FIELD_IDS:
        sub = sub[sub["target_month"].isin(_quarter_months(ref_period))]
    else:
        sub = sub[sub["target_month"] == ref_period]
    if sub.empty:
        return None
    ts = sub["timestamp_local"]
    if aggregator == "median_of_day":
        start = cutoff.normalize()  # 00:00:00 ET of the calendar day
        mask = (ts >= start) & (ts <= cutoff)
        cand = sub[mask]
        if cand.empty:
            return None
        return {
            "M_d_value": float(cand["value"].median()),
            "n_forecasts_through_cutoff": int(len(cand)),
            "earliest_ts": cand["timestamp_local"].min(),
            "latest_ts": cand["timestamp_local"].max(),
            "selection_dist_to_cutoff_h": float("nan"),
            "agg": "median_of_day",
        }
    elif aggregator == "latest":
        cand = sub[ts <= cutoff]
        if cand.empty:
            return None
        idx = cand["timestamp_local"].idxmax()
        chosen = cand.loc[idx]
        return {
            "M_d_value": float(chosen["value"]),
            "n_forecasts_through_cutoff": int(len(cand)),
            "earliest_ts": cand["timestamp_local"].min(),
            "latest_ts": chosen["timestamp_local"],
            "selection_dist_to_cutoff_h": float(
                (cutoff - chosen["timestamp_local"]).total_seconds() / 3600.0
            ),
            "agg": "latest",
        }
    elif aggregator in ("mean", "median"):
        lo = cutoff - max_history
        mask = (ts > lo) & (ts <= cutoff)
        cand = sub[mask]
        if cand.empty:
            return None
        v = float(cand["value"].mean()) if aggregator == "mean" else float(cand["value"].median())
        return {
            "M_d_value": v,
            "n_forecasts_through_cutoff": int(len(cand)),
            "earliest_ts": cand["timestamp_local"].min(),
            "latest_ts": cand["timestamp_local"].max(),
            "selection_dist_to_cutoff_h": float("nan"),
            "agg": aggregator,
        }
    else:
        raise ValueError(f"Unknown aggregator {aggregator!r}")


def aggregate_model_latest_before_release(
    field_id: str,
    model_output_source: str,
    ref_period: str,
    release_dt: pd.Timestamp,
    max_history: pd.Timedelta,
    preds_by_var: dict[str, pd.DataFrame],
    aggregator: str = "latest",
) -> dict | None:
    """Picks the model's M for the final-vs-final variant.

    Default ('latest') matches the existing pipeline's latest_pre_release
    mode exactly: the SINGLE prediction with timestamp closest to (and
    strictly before) release_dt. Pass aggregator='median' to instead match
    the existing median_14d mode.
    """
    cutoff = release_dt - pd.Timedelta(seconds=1)
    return aggregate_model_per_day(
        field_id, model_output_source, ref_period,
        release_dt, cutoff, max_history, preds_by_var, aggregator=aggregator,
    )


# ---------------------------------------------------------------------------
# Per-(model, release, day) field-level scoring.
# ---------------------------------------------------------------------------
SCORE_ON_MEDIAN_1D_OFFSETS: tuple[int, ...] = (1,)
SCORE_ON_MEDIAN_3D_OFFSETS: tuple[int, ...] = (1, 2, 3)
SCORE_ON_MEDIAN_7D_OFFSETS: tuple[int, ...] = DAY_OFFSETS


def _build_score_on_median_row(
    model_name: str, field_id: str, T: pd.Timestamp, ref_period: str,
    A: float, sig: float, b: float, mov: str, rule: str,
    days: list[int], C_vals: list[float], M_vals: list[float],
) -> dict:
    """Compute median-across-days (C, M) for an event/field and the
    derived S, S_hat. Caller filters `days` to the desired offset set."""
    C_med = float(np.median(C_vals))
    M_med = float(np.median(M_vals))
    return {
        "model": model_name,
        "field_id": field_id,
        "release_datetime_et": T,
        "ref_period": ref_period,
        "n_days_used": len(days),
        "days_used": "|".join(str(d) for d in days),
        "A": A,
        "C_med": C_med,
        "M_med": M_med,
        "sigma_i": sig,
        "beta_i": b,
        "S": (A - C_med) / sig,
        "S_hat": (M_med - C_med) / sig,
        "model_output_source": mov,
        "model_transform_rule": rule,
    }


def score_one_model_per_day(
    model_name: str,
    preds_by_var: dict[str, pd.DataFrame],
    live_releases: pd.DataFrame,
    bloomberg_cf: pd.DataFrame,
    release_meta: pd.DataFrame,
    mapping: pd.DataFrame,
    sigma: pd.Series,
    beta: pd.Series,
    max_history: pd.Timedelta,
    aggregator: str = "median_of_day",
    consensus_aggregator: str = "median",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (field_per_day_df, field_som_7d_df, field_som_3d_df,
    field_som_1d_df, field_fvf_df).

    field_per_day_df rows: one per (field_id, release_dt, day_offset)
        intermediate; emitted to disk for diagnostics but no per-day
        score is reported.
    field_som_7d_df / 3d / 1d rows: one per (field_id, release_dt) using
        per-event MEDIAN of (C_d, M_d) across the day_offsets where data
        exists, restricted to d in {1..7}, {1, 2, 3}, {1} respectively.
    field_fvf_df rows: one per (field_id, release_dt) using Bloomberg's
        final summary median as C and the model's latest pre-release
        prediction as M.
    """
    mapping_by_field = mapping.set_index("field_id")
    cf_index = bloomberg_cf.set_index(
        ["field_id", "ref_period", "day_offset"]
    )["C_d_calendar"].to_dict()
    cf_meta_index = bloomberg_cf.set_index(
        ["field_id", "ref_period", "day_offset"]
    )[["end_of_day_d", "asof_date_used", "n_to_date_d"]].to_dict("index")
    release_meta_idx = release_meta.set_index(
        ["field_id", "ref_period"]
    )

    per_day_rows: list[dict] = []
    som_7d_rows: list[dict] = []
    som_3d_rows: list[dict] = []
    som_1d_rows: list[dict] = []
    final_vs_final_rows: list[dict] = []

    for r in live_releases.itertuples(index=False):
        field_id = r.field_id
        ref_period = r.ref_period
        if (field_id, ref_period) not in release_meta_idx.index:
            continue
        A = float(r.A)
        T = r.release_datetime_et
        sig = float(sigma[field_id])
        m = mapping_by_field.loc[field_id]
        mov = str(m["model_output_source"])
        rule = str(m["model_transform_rule"])
        b = float(beta[field_id])

        per_event: list[tuple[int, float, float]] = []  # (d, C_d, M_transformed)
        for d in DAY_OFFSETS:
            C_d = cf_index.get((field_id, ref_period, d), float("nan"))
            cf_extras = cf_meta_index.get((field_id, ref_period, d), {})
            end_of_day = cf_extras.get("end_of_day_d", pd.NaT)
            asof_used = cf_extras.get("asof_date_used", pd.NaT)
            n_to_date = cf_extras.get("n_to_date_d", 0)
            if pd.isna(C_d) or pd.isna(end_of_day):
                continue
            magg = aggregate_model_per_day(
                field_id, mov, ref_period, T, end_of_day,
                max_history, preds_by_var, aggregator=aggregator,
            )
            if magg is None:
                continue
            M_raw = magg["M_d_value"]
            M_transformed = apply_transform(M_raw, rule)
            S_d = (A - C_d) / sig
            S_hat_d = (M_transformed - C_d) / sig
            per_day_rows.append({
                "model": model_name,
                "field_id": field_id,
                "release_datetime_et": T,
                "ref_period": ref_period,
                "day_offset": d,
                "end_of_day_d": end_of_day,
                "asof_date_used": asof_used,
                "n_estimates_to_date": n_to_date,
                "A": A,
                "C_d": C_d,
                "M_d_raw": M_raw,
                "M_d_transformed": M_transformed,
                "n_forecasts_through_d": magg["n_forecasts_through_cutoff"],
                "earliest_ts": magg["earliest_ts"],
                "latest_ts": magg["latest_ts"],
                "agg": magg["agg"],
                "sigma_i": sig,
                "beta_i": b,
                "S": S_d,
                "S_hat": S_hat_d,
                "model_output_source": mov,
                "model_transform_rule": rule,
            })
            per_event.append((d, C_d, M_transformed))

        # Score-on-median variants: median over d-subset where both C and M exist.
        if per_event:
            for offsets, sink in (
                (SCORE_ON_MEDIAN_7D_OFFSETS, som_7d_rows),
                (SCORE_ON_MEDIAN_3D_OFFSETS, som_3d_rows),
                (SCORE_ON_MEDIAN_1D_OFFSETS, som_1d_rows),
            ):
                sub = [(d, c, mm) for (d, c, mm) in per_event if d in offsets]
                if not sub:
                    continue
                ds = [d for (d, _, _) in sub]
                Cs = [c for (_, c, _) in sub]
                Ms = [mm for (_, _, mm) in sub]
                sink.append(_build_score_on_median_row(
                    model_name, field_id, T, ref_period, A, sig, b, mov, rule,
                    ds, Cs, Ms,
                ))

        # Final-vs-final: Bloomberg final-summary median (or mean) vs
        # model latest strictly before release. Always uses 'latest' for
        # apples-to-apples with the existing latest_pre_release mode.
        rel_meta = release_meta_idx.loc[(field_id, ref_period)]
        if consensus_aggregator == "median":
            C_fin = float(rel_meta["bloomberg_median_estimate_calendar"])
        else:
            C_fin = float(rel_meta["bloomberg_avg_estimate_calendar"])
        magg_full = aggregate_model_latest_before_release(
            field_id, mov, ref_period, T, max_history, preds_by_var,
            aggregator="latest",
        )
        if magg_full is not None and not pd.isna(C_fin):
            M_raw = magg_full["M_d_value"]
            M_transformed = apply_transform(M_raw, rule)
            S = (A - C_fin) / sig
            S_hat = (M_transformed - C_fin) / sig
            final_vs_final_rows.append({
                "model": model_name,
                "field_id": field_id,
                "release_datetime_et": T,
                "ref_period": ref_period,
                "A": A,
                "C_bloomberg_final": C_fin,
                "C_investing_used_in_existing_score": float(r.C),
                "M_latest_raw": M_raw,
                "M_latest_transformed": M_transformed,
                "n_forecasts_used": magg_full["n_forecasts_through_cutoff"],
                "sigma_i": sig,
                "beta_i": b,
                "S": S,
                "S_hat": S_hat,
            })

    return (
        pd.DataFrame(per_day_rows),
        pd.DataFrame(som_7d_rows),
        pd.DataFrame(som_3d_rows),
        pd.DataFrame(som_1d_rows),
        pd.DataFrame(final_vs_final_rows),
    )


# ---------------------------------------------------------------------------
# Field-level -> event-level -> aggregate scores.
# ---------------------------------------------------------------------------
def event_aggregate(field_df: pd.DataFrame, event_keys: list[str]) -> pd.DataFrame:
    """Reduce a (field, event, [optional day]) frame to the event-level
    Q_g, Q_hat_g, e_g table. event_keys lists the columns identifying an
    event grouping (e.g., ['release_datetime_et'] or
    ['release_datetime_et', 'day_offset'])."""
    if field_df.empty:
        return pd.DataFrame(
            columns=event_keys + [
                "n_matched_fields", "matched_fields", "Q_g", "Q_hat_g", "e_g",
            ]
        )
    df = field_df.copy()
    df["contrib_Q"] = df["beta_i"] * df["S"]
    df["contrib_Q_hat"] = df["beta_i"] * df["S_hat"]
    g = df.groupby(event_keys, sort=True)
    out = g.agg(
        n_matched_fields=("field_id", "nunique"),
        matched_fields=("field_id", lambda s: "|".join(sorted(set(s)))),
        Q_g=("contrib_Q", "sum"),
        Q_hat_g=("contrib_Q_hat", "sum"),
    ).reset_index()
    out["e_g"] = (out["Q_hat_g"] - out["Q_g"]) ** 2
    return out


def metrics_from_events(events: pd.DataFrame) -> dict:
    """MSC, BMSC, BDRC*, BP-RMSE, WDH from a per-event table.

    BDRC needs hf_return per event (passed in via column 'hf_return'); we
    leave it NaN here because the per-day matrix doesn't carry hf_return.
    The aggregate-scores writer fills BDRC at the score-on-median / final
    level only (where each event has one hf_return).
    """
    if events.empty:
        return {
            "n_events": 0,
            "sum_Q_g_squared": 0.0, "sum_e_g": 0.0,
            "MSC": float("nan"), "BMSC": float("nan"),
            "BP_RMSE": float("nan"), "WDH": float("nan"),
        }
    n = len(events)
    sum_Qsq = float((events["Q_g"] ** 2).sum())
    sum_e = float(events["e_g"].sum())
    MSC = 1.0 - sum_e / sum_Qsq if sum_Qsq > 0 else float("nan")
    BMSC = (
        (sum_Qsq - sum_e) / (sum_Qsq + sum_e)
        if (sum_Qsq + sum_e) > 0 else float("nan")
    )
    BP_RMSE = 1e4 * float(np.sqrt(sum_e / n)) if n > 0 else float("nan")
    abs_Q = events["Q_g"].abs()
    sum_abs_Q = float(abs_Q.sum())
    if sum_abs_Q > 0:
        same_sign = np.sign(events["Q_hat_g"].values) == np.sign(events["Q_g"].values)
        WDH = float(np.sum(abs_Q.values * same_sign.astype(float)) / sum_abs_Q)
    else:
        WDH = float("nan")
    return {
        "n_events": n,
        "sum_Q_g_squared": sum_Qsq,
        "sum_e_g": sum_e,
        "MSC": MSC,
        "BMSC": BMSC,
        "BP_RMSE": BP_RMSE,
        "WDH": WDH,
    }


def metrics_with_bdrc(events: pd.DataFrame) -> dict:
    base = metrics_from_events(events)
    if events.empty or "hf_return" not in events.columns:
        base["BDRC"] = float("nan")
        base["DRC_RMSE"] = float("nan")
        base["n_drc_events"] = 0
        return base
    valid = events[events["hf_return"].notna()]
    if valid.empty:
        base["BDRC"] = float("nan")
        base["DRC_RMSE"] = float("nan")
        base["n_drc_events"] = 0
        return base
    r = valid["hf_return"].astype(float).to_numpy()
    Qh = valid["Q_hat_g"].astype(float).to_numpy()
    n_drc = int(len(valid))
    T_ = float(r @ r)
    F_ = float((r - Qh) @ (r - Qh))
    base["BDRC"] = (T_ - F_) / (T_ + F_) if (T_ + F_) > 0 else float("nan")
    base["DRC_RMSE"] = (
        1e4 * float(np.sqrt(F_ / n_drc)) if n_drc > 0 else float("nan")
    )
    base["n_drc_events"] = n_drc
    return base


# ---------------------------------------------------------------------------
# Huber refit + parametric-bootstrap CIs for any (model, field) score frame.
#
# Re-runs the historical Huber-ridge fit from score_pipeline_huber.py to
# recover beta_hat_huber and V_huber, then uses the SAME
# score_pipeline.bootstrap_score_ci as the legacy CI table — only the
# input field rows change (Bloomberg-final-median C for final_vs_final,
# per-event median of (C_d, M_d) for the score_on_median variants).
# beta_hat_huber from the refit must match
# huber_final/huber_beta_by_field.csv exactly (sanity check).
# ---------------------------------------------------------------------------
def bootstrap_ci_for_final_vs_final(
    field_fvf_df: pd.DataFrame,
    live_releases: pd.DataFrame,
    field_ids: list[str],
    beta_hat: np.ndarray,
    L_chol: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    drc_wdh_tau: float = 1e-4,
    mode_label: str = "final_vs_final",
) -> pd.DataFrame:
    """Per-model CI table for one score variant. Schema identical to the
    legacy score_ci.csv so cross-checks line up directly. `mode_label` is
    written to the `mode` column verbatim — pass `score_on_median_1d`,
    `score_on_median_3d`, or `score_on_median_7d` when feeding the
    corresponding per-event field table."""
    if field_fvf_df.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for model_name, sub in field_fvf_df.groupby("model", sort=False):
        sub = sub.merge(
            live_releases[["release_datetime_et", "hf_return", "event"]]
            .drop_duplicates("release_datetime_et"),
            on="release_datetime_et", how="left",
        )
        ci = bootstrap_score_ci(
            sub, field_ids, beta_hat, L_chol,
            n_boot=n_boot, rng=rng, drc_wdh_tau=drc_wdh_tau,
        )
        rows.append({"model": model_name, "mode": mode_label, **ci})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# (Per-day score variant retired; helpers below remain exported for
# downstream callers e.g. score_by_theme_bloomberg.py until that script
# is migrated to the score_on_median_* outputs.)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Per-day score table for one model.
# ---------------------------------------------------------------------------
def per_day_scores(
    field_per_day_df: pd.DataFrame,
    hf_map: dict | None = None,
) -> pd.DataFrame:
    """Return per-day metrics. When `hf_map` (release_datetime_et -> hf_return)
    is provided, BDRC and DRC_RMSE are computed per day_offset; otherwise
    those columns are NaN. The optional argument preserves backward
    compatibility for existing callers (e.g., score_by_theme_bloomberg.py)."""
    if field_per_day_df.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for d in DAY_OFFSETS:
        sub = field_per_day_df[field_per_day_df["day_offset"] == d]
        events = event_aggregate(sub, ["release_datetime_et"])
        if hf_map is not None and not events.empty:
            events = events.copy()
            events["hf_return"] = events["release_datetime_et"].map(hf_map)
            m = metrics_with_bdrc(events)
        else:
            m = metrics_from_events(events)
        m["day_offset"] = d
        m["n_field_rows"] = int(len(sub))
        rows.append(m)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sampling CI for final_vs_final (t-CI on the N latest predictions).
#
# Procedure (one per model):
#   1. For each (field i, event g), pick the N_LATEST most recent
#      predictions with `timestamp_local < release_dt` (after the upstream
#      10x-median outlier filter — predictions used here are clean).
#   2. C_{i,g} is the Bloomberg final-summary median (CONSTANT across
#      ranks). Every snapshot is scored against the same final consensus,
#      matching the final_vs_final variant.
#   3. For snapshot rank k ∈ {1..N_LATEST} (k=1 = latest), compute the
#      model-level BMSC, BDRC, BP_RMSE, WDH, DRC_RMSE using rank-k's M
#      across events that have a rank-k prediction. Result: up to
#      N_LATEST model-level scores per metric per model.
#   4. CI = mean ± t_{K-1, α/2} · SD/√K, K = number of ranks with data
#      (typically K = N_LATEST = 5 → df = 4, t_{0.975} = 2.776).
#
# Captures: stability of the model's emitted M across its most recent
#   N_LATEST predictions, conditional on β = β̂. Tracks "latest
#   performance" because all snapshots are anchored to the most-recent
#   predictions, not to fixed calendar days.
#
# Why "5 latest" instead of "5 daily snapshots":
#   - Stable, fast-emit models (gpt-5) have all 5 latest within hours of
#     release ⇒ very tight CI (the model's voice is consistent right
#     before release).
#   - Slow-emit models have 5 latest spanning several days but still uses
#     the model's actual last-emitted views — no "fill day d with median".
#   - Outlier filter already applied at model-load; ranks pull from the
#     filtered prediction set so a 10x outlier never enters a snapshot.
# ---------------------------------------------------------------------------
N_LATEST_SNAPSHOTS: int = 5

# Student's t two-sided critical values; hardcoded so the pipeline keeps
# its self-contained "no scipy / no precomputed JSON" property.
_T_CRIT_975 = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706,
               6: 2.4469, 7: 2.3646, 8: 2.3060, 9: 2.2622, 10: 2.2281}
_T_CRIT_950 = {1: 6.3138, 2: 2.9200, 3: 2.3534, 4: 2.1318, 5: 2.0150,
               6: 1.9432, 7: 1.8946, 8: 1.8595, 9: 1.8331, 10: 1.8125}


def _t_critical(df: int, two_sided_alpha: float) -> float:
    if df < 1:
        return float("nan")
    if abs(two_sided_alpha - 0.05) < 1e-9:
        table = _T_CRIT_975
    elif abs(two_sided_alpha - 0.10) < 1e-9:
        table = _T_CRIT_950
    else:
        raise ValueError(
            f"_t_critical only supports α=0.05 and α=0.10 (got {two_sided_alpha})"
        )
    return float(table.get(df, table[max(table)]))


def score_one_model_n_latest(
    model_name: str,
    preds_by_var: dict[str, pd.DataFrame],
    live_releases: pd.DataFrame,
    release_meta: pd.DataFrame,
    mapping: pd.DataFrame,
    sigma: pd.Series,
    beta: pd.Series,
    n_snapshots: int = N_LATEST_SNAPSHOTS,
    consensus_aggregator: str = "median",
) -> pd.DataFrame:
    """For each (field_id, release_dt) in `live_releases` with Bloomberg
    coverage, pick the n_snapshots most recent predictions from
    `preds_by_var` (already outlier-filtered) and emit one row per
    (field, event, snapshot_rank). C is the Bloomberg final-summary
    median (or mean per consensus_aggregator) — constant across ranks.

    snapshot_rank = 1 means the latest prediction strictly before release_dt.
    """
    mapping_by_field = mapping.set_index("field_id")
    release_meta_idx = release_meta.set_index(["field_id", "ref_period"])
    rows: list[dict] = []

    for r in live_releases.itertuples(index=False):
        field_id = r.field_id
        ref_period = r.ref_period
        if (field_id, ref_period) not in release_meta_idx.index:
            continue
        A = float(r.A)
        T = r.release_datetime_et
        sig = float(sigma[field_id])
        m = mapping_by_field.loc[field_id]
        mov = str(m["model_output_source"])
        rule = str(m["model_transform_rule"])
        b = float(beta[field_id])

        rel_meta = release_meta_idx.loc[(field_id, ref_period)]
        if consensus_aggregator == "median":
            C_fin = float(rel_meta["bloomberg_median_estimate_calendar"])
        else:
            C_fin = float(rel_meta["bloomberg_avg_estimate_calendar"])
        if pd.isna(C_fin):
            continue

        sub = preds_by_var.get(mov)
        if sub is None or sub.empty:
            continue
        if field_id in QUARTERLY_FIELD_IDS:
            sub = sub[sub["target_month"].isin(_quarter_months(ref_period))]
        else:
            sub = sub[sub["target_month"] == ref_period]
        cand = sub[sub["timestamp_local"] < T]
        if cand.empty:
            continue
        cand = cand.sort_values(
            "timestamp_local", ascending=False, kind="mergesort"
        ).head(n_snapshots).reset_index(drop=True)

        for k in range(len(cand)):
            row = cand.iloc[k]
            M_raw = float(row["value"])
            M_t = apply_transform(M_raw, rule)
            S = (A - C_fin) / sig
            S_hat = (M_t - C_fin) / sig
            rows.append({
                "model": model_name,
                "field_id": field_id,
                "release_datetime_et": T,
                "ref_period": ref_period,
                "snapshot_rank": k + 1,
                "timestamp_local": row["timestamp_local"],
                "A": A,
                "C_final": C_fin,
                "M_raw": M_raw,
                "M_transformed": M_t,
                "sigma_i": sig,
                "beta_i": b,
                "S": S,
                "S_hat": S_hat,
                "model_output_source": mov,
                "model_transform_rule": rule,
            })
    return pd.DataFrame(rows)


def sampling_ci_from_n_latest(
    field_per_snapshot_df: pd.DataFrame,
    live_releases: pd.DataFrame,
    n_snapshots: int = N_LATEST_SNAPSHOTS,
) -> pd.DataFrame:
    """Per-model t-CI table from up to n_snapshots model-level scores
    (one per snapshot_rank). Each rank is scored using the matched-subset
    rule across events that have a rank-k prediction.

    The "point" reported alongside the CI is the MEAN over the K
    rank-level scores (i.e., the centre of the t-CI). For comparison
    with the β-bootstrap point estimate (which uses rank=1 only), see
    `bloomberg_final_vs_final_ci.csv`.
    """
    if field_per_snapshot_df.empty:
        return pd.DataFrame()

    hf_map = (
        live_releases[["release_datetime_et", "hf_return"]]
        .drop_duplicates("release_datetime_et")
        .set_index("release_datetime_et")["hf_return"]
        .to_dict()
    )

    metrics = ("MSC", "BMSC", "BP_RMSE", "WDH", "BDRC", "DRC_RMSE")
    rows: list[dict] = []
    for model_name, msub in field_per_snapshot_df.groupby("model", sort=False):
        per_rank_vals: dict[int, dict] = {}
        per_rank_n_events: dict[int, int] = {}
        for k in range(1, n_snapshots + 1):
            ksub = msub[msub["snapshot_rank"] == k]
            if ksub.empty:
                continue
            events = event_aggregate(ksub, ["release_datetime_et"]).copy()
            events["hf_return"] = events["release_datetime_et"].map(hf_map)
            per_rank_vals[k] = metrics_with_bdrc(events)
            per_rank_n_events[k] = int(len(events))
        if not per_rank_vals:
            continue

        ranks_used = sorted(per_rank_vals)
        row = {
            "model": model_name,
            "n_ranks_used": len(ranks_used),
            "ranks_used": "|".join(str(k) for k in ranks_used),
        }
        for metric in metrics:
            vals = np.array(
                [per_rank_vals[k].get(metric, float("nan")) for k in ranks_used],
                dtype=float,
            )
            vals = vals[~np.isnan(vals)]
            n = int(len(vals))
            row[f"{metric}_n"] = n
            if n == 0:
                for k in ("mean", "sd", "se", "ci95_lo", "ci95_hi",
                          "ci90_lo", "ci90_hi"):
                    row[f"{metric}_{k}"] = float("nan")
                continue
            mean = float(vals.mean())
            row[f"{metric}_mean"] = mean
            if n < 2:
                row[f"{metric}_sd"] = 0.0
                row[f"{metric}_se"] = 0.0
                for k in ("ci95_lo", "ci95_hi", "ci90_lo", "ci90_hi"):
                    row[f"{metric}_{k}"] = mean
                continue
            sd = float(vals.std(ddof=1))
            se = sd / float(np.sqrt(n))
            row[f"{metric}_sd"] = sd
            row[f"{metric}_se"] = se
            for cl, alpha in (("ci95", 0.05), ("ci90", 0.10)):
                t_crit = _t_critical(n - 1, alpha)
                hw = t_crit * se
                row[f"{metric}_{cl}_lo"] = mean - hw
                row[f"{metric}_{cl}_hi"] = mean + hw
        for metric in ("BMSC", "BDRC"):
            for k in ranks_used:
                row[f"{metric}_rank{k}"] = per_rank_vals[k].get(metric, float("nan"))
            for k in ranks_used:
                row[f"n_events_rank{k}"] = per_rank_n_events[k]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Headline aggregates (no CIs in this version — see module docstring).
# ---------------------------------------------------------------------------
def aggregate_scores(
    model_name: str,
    field_som_7d_df: pd.DataFrame,
    field_som_3d_df: pd.DataFrame,
    field_som_1d_df: pd.DataFrame,
    field_final_vs_final_df: pd.DataFrame,
    live_releases_with_hf: pd.DataFrame,
) -> dict:
    """Compute headline point estimates for a single model.

    Returns a flat dict keyed by `<variant>_<metric>` for variants
    score_on_median_7d / 3d / 1d / final_vs_final and metrics MSC, BMSC,
    BP_RMSE, WDH, BDRC, DRC_RMSE, n_events, n_drc_events.
    """
    out: dict = {"model": model_name}
    hf_map = (
        live_releases_with_hf[["release_datetime_et", "hf_return"]]
        .drop_duplicates("release_datetime_et")
        .set_index("release_datetime_et")["hf_return"]
        .to_dict()
    )

    for label, frame in (
        ("score_on_median_7d", field_som_7d_df),
        ("score_on_median_3d", field_som_3d_df),
        ("score_on_median_1d", field_som_1d_df),
        ("final_vs_final", field_final_vs_final_df),
    ):
        if frame is None or frame.empty:
            continue
        events = event_aggregate(frame, ["release_datetime_et"])
        events["hf_return"] = events["release_datetime_et"].map(hf_map)
        m = metrics_with_bdrc(events)
        for k in ("MSC", "BMSC", "BP_RMSE", "WDH", "BDRC", "DRC_RMSE",
                  "n_events", "n_drc_events"):
            out[f"{label}_{k}"] = m.get(k, float("nan"))

    return out


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    p.add_argument("--sigma", type=Path, default=DEFAULT_SIGMA)
    p.add_argument("--ts-events-csv", type=Path, default=DEFAULT_TS_EVENTS,
                   help="historical events file with hf_return + X_<field_id> "
                        "columns. The pipeline refits Huber-ridge on this file "
                        "every run (no precomputed β imported).")
    p.add_argument("--lambda", dest="lam", type=float, default=HUBER_LAMBDA,
                   help=f"L2 ridge penalty (default {HUBER_LAMBDA:.6f}, "
                        f"selected by blocked chronological 5-fold CV under "
                        f"the no-intercept design — see "
                        f"cv_lambda_no_intercept.{{py,csv,json}}).")
    p.add_argument("--huber-c", type=float, default=HUBER_C,
                   help=f"Huber tuning constant (default {HUBER_C}; "
                        f"95%% Gaussian efficiency).")
    p.add_argument("--live-gt-csv", type=Path, default=DEFAULT_LIVE_GT)
    p.add_argument("--coverage-csv", type=Path, default=DEFAULT_COVERAGE)
    p.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    p.add_argument("--bloomberg-daily-csv", type=Path, default=DEFAULT_DAILY_CSV)
    p.add_argument("--bloomberg-release-csv", type=Path, default=DEFAULT_RELEASE_CSV)
    p.add_argument("--arima-csv", type=Path, default=DEFAULT_ARIMA_CSV)
    p.add_argument("--out-dir", type=Path, default=HERE / "bloomberg_overlay")
    p.add_argument("--live-start", type=str, default="2025-11-01")
    p.add_argument("--date-only-policy", choices=("drop", "eod", "midnight"),
                   default="midnight")
    p.add_argument("--max-history-days", type=int, default=14,
                   help="max history (days) for the 'mean'/'median' window "
                        "aggregators only; ignored by median_of_day and latest")
    p.add_argument("--aggregator",
                   choices=("median_of_day", "latest", "mean", "median"),
                   default="median_of_day",
                   help="per-day model aggregator: 'median_of_day' (default; "
                        "median of preds whose timestamp lies in [00:00, "
                        "23:59:59] ET of release_date - d days; drops days "
                        "with no preds); 'latest' (single most-recent "
                        "prediction by end_of_day_d, matches existing "
                        "latest_pre_release semantics); 'mean'/'median' over "
                        "the (cutoff - max_history, cutoff] window. "
                        "final_vs_final always uses 'latest' regardless.")
    p.add_argument("--consensus-aggregator", choices=("median", "mean"),
                   default="median",
                   help="Bloomberg consensus reduction across economists: "
                        "'median' (recommended; matches investing.com for "
                        "cpi/pce_price_index/ppi/real_pce/retail_sales/"
                        "unemployment_rate exactly) or 'mean'.")
    p.add_argument("--n-boot", type=int, default=N_BOOT,
                   help="parametric-bootstrap draws for final_vs_final CIs")
    p.add_argument("--seed", type=int, default=BOOT_SEED,
                   help="bootstrap RNG seed (deterministic re-runs)")
    p.add_argument("--drc-wdh-threshold-bps", type=float, default=DRC_WDH_THRESHOLD_BPS,
                   help="bps threshold for DRC_WDH_tau / active_share")
    p.add_argument("--outlier-log10-threshold", type=float, default=OUTLIER_LOG10_THRESHOLD,
                   help="log10(value/median) threshold for the per-(variable, "
                        "target_month) outlier filter (default 1.0 = 10x)")
    p.add_argument("--no-ci", action="store_true",
                   help="skip the final_vs_final bootstrap CI (β is still fitted)")
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    max_history = pd.Timedelta(days=args.max_history_days)

    # Inputs.
    mapping = pd.read_csv(args.mapping)
    sigma = pd.read_csv(args.sigma).set_index("field_id")["sigma_i"]
    field_ids = mapping["field_id"].tolist()

    # ------------------------------------------------------------------
    # Stage 1. Refit β from scratch on the historical events. No precomputed
    # β file is consumed — every run repeats the regression deterministically
    # (IRLS from a fixed OLS-ridge init on identical (X, y, λ)).
    # ------------------------------------------------------------------
    print(f"\n=== Stage 1: Huber-ridge refit on historical events ===")
    print(f"  ts_events: {args.ts_events_csv}")
    print(f"  lambda = {args.lam:.6f}, huber_c = {args.huber_c}")
    beta_arr, L_chol, fit_info = fit_beta_for_pipeline(
        ts_events_csv=args.ts_events_csv,
        field_ids=field_ids,
        lam=args.lam,
        c=args.huber_c,
    )
    print(f"  IRLS converged: {fit_info['converged']}  in {fit_info['n_iter']} iter")
    print(f"  σ̂ = {fit_info['sigma']:.4e}, δ = {fit_info['delta']:.4e}, "
          f"df_eff = {fit_info['df_eff']:.2f}")
    beta = pd.Series(beta_arr, index=field_ids, name="beta_i_huber")
    beta_df_out = pd.DataFrame({
        "field_id": field_ids,
        "beta_i_huber": beta_arr,
        "marginal_se_HC1": [fit_info["marginal_se_HC1"][f] for f in field_ids],
    })
    beta_df_out.to_csv(args.out_dir / "huber_beta_by_field.csv", index=False)

    live_start = pd.Timestamp(args.live_start)
    live_releases, _coverage, _drop_stats = build_live_releases(
        args.live_gt_csv, args.coverage_csv, live_start
    )
    live_releases.to_csv(args.out_dir / "live_field_releases_used.csv", index=False)

    if not args.bloomberg_daily_csv.exists():
        raise SystemExit(
            f"Bloomberg daily CSV not found at {args.bloomberg_daily_csv}. "
            f"Run Results/bloomberg_consensus/build_daily_consensus.py first."
        )
    if not args.bloomberg_release_csv.exists():
        raise SystemExit(
            f"Bloomberg release CSV not found at {args.bloomberg_release_csv}."
        )

    daily = pd.read_csv(
        args.bloomberg_daily_csv,
        parse_dates=["release_datetime_et", "asof_date", "asof_datetime_et"],
    )
    release_meta = pd.read_csv(
        args.bloomberg_release_csv,
        parse_dates=["release_datetime_et", "first_asof_date", "last_asof_date"],
    )
    bloomberg_cf = build_bloomberg_carry_forward(
        daily, DAY_OFFSETS, consensus_aggregator=args.consensus_aggregator,
    )
    bloomberg_cf.to_csv(
        args.out_dir / "bloomberg_carry_forward_per_day.csv", index=False
    )

    # Coverage diagnostics.
    n_live_with_coverage = len(set(zip(
        live_releases["field_id"], live_releases["ref_period"]
    )) & set(zip(release_meta["field_id"], release_meta["ref_period"])))
    print(
        f"Live releases: {len(live_releases)};  with Bloomberg coverage: "
        f"{n_live_with_coverage}"
    )

    # Discover models.
    model_dirs = sorted([
        d for d in args.models_root.iterdir()
        if d.is_dir() and d.name.startswith("model_")
    ])
    if not model_dirs:
        raise SystemExit(f"No model directories under {args.models_root}.")

    all_per_day_field: list[pd.DataFrame] = []
    all_som_7d_field: list[pd.DataFrame] = []
    all_som_3d_field: list[pd.DataFrame] = []
    all_som_1d_field: list[pd.DataFrame] = []
    all_final_field: list[pd.DataFrame] = []
    all_nlatest_field: list[pd.DataFrame] = []
    all_aggregate_rows: list[dict] = []
    outlier_report: dict[str, dict] = {}

    for md in model_dirs:
        model_name = md.name[len("model_"):]
        print(f"\n=== {model_name} ===")
        preds, n_date_only = load_model_predictions(
            md, date_only_policy=args.date_only_policy
        )
        # Apply the 10x-per-(variable, target_month)-median outlier filter
        # to every LLM. Threshold is conservative (only true extremes).
        n_pre = int(len(preds))
        kept, dropped = filter_outliers(
            preds, log10_threshold=args.outlier_log10_threshold,
        )
        preds = kept
        n_drop = int(len(dropped))
        pct = (100.0 * n_drop / n_pre) if n_pre else 0.0
        print(f"  outlier filter: kept {len(preds):,}/{n_pre:,}, "
              f"dropped {n_drop:,} ({pct:.2f}%)")
        outlier_report[model_name] = {
            "n_predictions_before": n_pre,
            "n_predictions_after": int(len(preds)),
            "n_dropped": n_drop,
            "pct_dropped": pct,
            "log10_threshold": float(args.outlier_log10_threshold),
        }
        if n_drop:
            dropped.to_csv(
                args.out_dir / f"outliers_dropped_{model_name}.csv",
                index=False,
            )

        preds_by_var = {v: g.copy() for v, g in preds.groupby("variable", sort=False)}
        f_pd, f_som7, f_som3, f_som1, f_fvf = score_one_model_per_day(
            model_name=model_name,
            preds_by_var=preds_by_var,
            live_releases=live_releases,
            bloomberg_cf=bloomberg_cf,
            release_meta=release_meta,
            mapping=mapping,
            sigma=sigma,
            beta=beta,
            max_history=max_history,
            aggregator=args.aggregator,
            consensus_aggregator=args.consensus_aggregator,
        )
        all_per_day_field.append(f_pd)
        all_som_7d_field.append(f_som7)
        all_som_3d_field.append(f_som3)
        all_som_1d_field.append(f_som1)
        all_final_field.append(f_fvf)

        # N latest snapshots per (field, event) for the sampling CI.
        f_nlatest = score_one_model_n_latest(
            model_name=model_name,
            preds_by_var=preds_by_var,
            live_releases=live_releases,
            release_meta=release_meta,
            mapping=mapping,
            sigma=sigma,
            beta=beta,
            n_snapshots=N_LATEST_SNAPSHOTS,
            consensus_aggregator=args.consensus_aggregator,
        )
        all_nlatest_field.append(f_nlatest)

        agg_row = aggregate_scores(
            model_name=model_name,
            field_som_7d_df=f_som7,
            field_som_3d_df=f_som3,
            field_som_1d_df=f_som1,
            field_final_vs_final_df=f_fvf,
            live_releases_with_hf=live_releases,
        )
        all_aggregate_rows.append(agg_row)
        print(
            f"  som_7d rows: {len(f_som7)};  som_3d rows: {len(f_som3)};  "
            f"som_1d rows: {len(f_som1)};  final_vs_final rows: {len(f_fvf)};  "
            f"n_latest rows: {len(f_nlatest)}"
        )

    # ARIMA benchmark — same machinery using synthesized predictions.
    if args.arima_csv and Path(args.arima_csv).exists():
        print(f"\n=== {ARIMA_MODEL_NAME} (benchmark) ===")
        arima_preds, _diag = build_arima_predictions(
            arima_csv=args.arima_csv,
            mapping=mapping,
            live_releases=live_releases,
            quarterly_field_ids=QUARTERLY_FIELD_IDS,
        )
        # ARIMA emits one static forecast per (variable, target_month).
        # build_arima_predictions places it at release_dt - 1s.
        # For median_of_day we need the synthetic value to land inside the
        # calendar-day window of EVERY d in {1..7}. Replicate each row
        # across those 7 days, placing the synthetic timestamp at
        # release_date - d days + 12:00 (noon ET). Replication is safe
        # because the value being scored is identical across rows.
        release_dt_per_row = arima_preds["timestamp_local"] + pd.Timedelta(seconds=1)
        release_date_per_row = release_dt_per_row.dt.normalize()
        replicated = []
        for d in DAY_OFFSETS:
            new_ts = (
                release_date_per_row - pd.Timedelta(days=d)
                + pd.Timedelta(hours=12)
            )
            rep = arima_preds.copy()
            rep["timestamp_local"] = new_ts.values
            replicated.append(rep)
        arima_preds = pd.concat(replicated, ignore_index=True)
        arima_mapping = apply_arima_overrides(mapping)
        arima_preds_by_var = {
            v: g.copy() for v, g in arima_preds.groupby("variable", sort=False)
        }
        f_pd, f_som7, f_som3, f_som1, f_fvf = score_one_model_per_day(
            model_name=ARIMA_MODEL_NAME,
            preds_by_var=arima_preds_by_var,
            live_releases=live_releases,
            bloomberg_cf=bloomberg_cf,
            release_meta=release_meta,
            mapping=arima_mapping,
            sigma=sigma,
            beta=beta,
            max_history=max_history,
            aggregator=args.aggregator,
            consensus_aggregator=args.consensus_aggregator,
        )
        all_per_day_field.append(f_pd)
        all_som_7d_field.append(f_som7)
        all_som_3d_field.append(f_som3)
        all_som_1d_field.append(f_som1)
        all_final_field.append(f_fvf)
        f_nlatest = score_one_model_n_latest(
            model_name=ARIMA_MODEL_NAME,
            preds_by_var=arima_preds_by_var,
            live_releases=live_releases,
            release_meta=release_meta,
            mapping=arima_mapping,
            sigma=sigma,
            beta=beta,
            n_snapshots=N_LATEST_SNAPSHOTS,
            consensus_aggregator=args.consensus_aggregator,
        )
        all_nlatest_field.append(f_nlatest)
        agg_row = aggregate_scores(
            model_name=ARIMA_MODEL_NAME,
            field_som_7d_df=f_som7,
            field_som_3d_df=f_som3,
            field_som_1d_df=f_som1,
            field_final_vs_final_df=f_fvf,
            live_releases_with_hf=live_releases,
        )
        all_aggregate_rows.append(agg_row)
        print(
            f"  som_7d rows: {len(f_som7)};  som_3d rows: {len(f_som3)};  "
            f"som_1d rows: {len(f_som1)};  final_vs_final rows: {len(f_fvf)};  "
            f"n_latest rows: {len(f_nlatest)}"
        )

    # Concat per-model frames.
    def _cat(frames: list[pd.DataFrame]) -> pd.DataFrame:
        kept = [df for df in frames if len(df)]
        return pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()

    field_pd = _cat(all_per_day_field)
    field_som7 = _cat(all_som_7d_field)
    field_som3 = _cat(all_som_3d_field)
    field_som1 = _cat(all_som_1d_field)
    field_fvf = _cat(all_final_field)
    field_nlatest = _cat(all_nlatest_field)
    aggregate_df = pd.DataFrame(all_aggregate_rows)

    field_pd.to_csv(args.out_dir / "bloomberg_daily_field_releases.csv", index=False)
    field_som7.to_csv(args.out_dir / "bloomberg_score_on_median_field_7d.csv", index=False)
    field_som3.to_csv(args.out_dir / "bloomberg_score_on_median_field_3d.csv", index=False)
    field_som1.to_csv(args.out_dir / "bloomberg_score_on_median_field_1d.csv", index=False)
    field_fvf.to_csv(args.out_dir / "bloomberg_final_vs_final_field.csv", index=False)
    field_nlatest.to_csv(
        args.out_dir / "bloomberg_final_vs_final_n_latest_field.csv", index=False,
    )
    aggregate_df.to_csv(args.out_dir / "bloomberg_aggregate_scores.csv", index=False)

    # ---- Outlier filter audit ------------------------------------------
    if outlier_report:
        outlier_audit_df = pd.DataFrame.from_dict(
            outlier_report, orient="index"
        ).rename_axis("model").reset_index()
        outlier_audit_df.to_csv(
            args.out_dir / "outlier_filter_audit.csv", index=False
        )

    # ---- final_vs_final + score_on_median CIs (parametric bootstrap of β
    # ----  fit in stage 1; same β draws across variants because the rng is
    # ----  shared, so each (variant, model) pair consumes its own slice).
    ci_info: dict = {"computed": False}
    score_on_median_ci_variants = (
        ("score_on_median_1d", field_som1),
        ("score_on_median_3d", field_som3),
        ("score_on_median_7d", field_som7),
    )
    if not args.no_ci:
        rng = np.random.default_rng(args.seed)
        drc_wdh_tau = args.drc_wdh_threshold_bps / 10000.0

        if not field_fvf.empty:
            print("\n=== final_vs_final bootstrap CI ===")
            ci_df = bootstrap_ci_for_final_vs_final(
                field_fvf, live_releases, field_ids,
                beta_arr, L_chol,
                n_boot=args.n_boot, rng=rng, drc_wdh_tau=drc_wdh_tau,
            )
            ci_df.to_csv(
                args.out_dir / "bloomberg_final_vs_final_ci.csv", index=False
            )
            ci_info = {
                "computed": True,
                "lambda": fit_info["lambda"],
                "huber_c": fit_info["huber_c"],
                "n_historical_events": fit_info["n_events"],
                "sigma": fit_info["sigma"],
                "delta": fit_info["delta"],
                "df_eff": fit_info["df_eff"],
                "n_boot": int(args.n_boot),
                "seed": int(args.seed),
                "drc_wdh_tau": float(drc_wdh_tau),
            }
            print(f"  bootstrap n_boot={args.n_boot}, seed={args.seed}")
        else:
            pd.DataFrame().to_csv(
                args.out_dir / "bloomberg_final_vs_final_ci.csv", index=False
            )

        for variant_label, variant_df in score_on_median_ci_variants:
            out_path = (
                args.out_dir / f"bloomberg_{variant_label}_ci.csv"
            )
            if variant_df.empty:
                pd.DataFrame().to_csv(out_path, index=False)
                continue
            print(f"\n=== {variant_label} bootstrap CI ===")
            ci_df_v = bootstrap_ci_for_final_vs_final(
                variant_df, live_releases, field_ids,
                beta_arr, L_chol,
                n_boot=args.n_boot, rng=rng, drc_wdh_tau=drc_wdh_tau,
                mode_label=variant_label,
            )
            ci_df_v.to_csv(out_path, index=False)
            print(f"  wrote {out_path.name} ({len(ci_df_v)} rows)")
    else:
        for fname in (
            "bloomberg_final_vs_final_ci.csv",
            "bloomberg_score_on_median_1d_ci.csv",
            "bloomberg_score_on_median_3d_ci.csv",
            "bloomberg_score_on_median_7d_ci.csv",
        ):
            pd.DataFrame().to_csv(args.out_dir / fname, index=False)

    # ---- final_vs_final SAMPLING CI (t-CI on the N latest predictions) -
    # For each (model, field, event), pick the N_LATEST most recent
    # predictions with timestamp < release_dt (already outlier-filtered
    # at model load). Each rank k ∈ {1..N_LATEST} produces one model-
    # level score; t-CI is computed on those K scores.
    #
    # Caveat: this CI is more sensitive to rank-selection artefacts than
    # the bootstrap CI on β. See SCORE_CALCULATION.md §B for the
    # interpretation guide. Reported alongside the bootstrap CI as a
    # complementary stability check, NOT as a replacement.
    sampling_ci_info: dict = {"computed": False}
    if not field_nlatest.empty:
        print("\n=== final_vs_final sampling CI "
              "(t-CI on N latest predictions per field-event) ===")
        sampling_ci_df = sampling_ci_from_n_latest(
            field_nlatest, live_releases, n_snapshots=N_LATEST_SNAPSHOTS,
        )
        sampling_ci_df.to_csv(
            args.out_dir / "bloomberg_final_vs_final_sampling_ci.csv", index=False
        )
        sampling_ci_info = {
            "computed": True,
            "n_latest_snapshots": int(N_LATEST_SNAPSHOTS),
            "method": "t-CI on K rank-level model scores; for each "
                      "(field, event) the N latest predictions with "
                      "timestamp < release_dt are taken (after the "
                      "10x-median outlier filter). C = Bloomberg "
                      "final-summary median (constant across ranks).",
            "captures": "stability of the model's most recent N "
                        "emissions; β is FIXED at point estimate (no β "
                        "uncertainty).",
            "caveat": "ranks are correlated for fast-emit models (5 "
                      "latest of gpt-5 are within hours of each other), "
                      "so the naive t-CI is slightly anti-conservative.",
        }
        print(f"  N = {N_LATEST_SNAPSHOTS} latest predictions; "
              f"t_{{{N_LATEST_SNAPSHOTS-1}, 0.975}} = "
              f"{_t_critical(N_LATEST_SNAPSHOTS-1, 0.05):.3f}, "
              f"t_{{{N_LATEST_SNAPSHOTS-1}, 0.95}}  = "
              f"{_t_critical(N_LATEST_SNAPSHOTS-1, 0.10):.3f}")
    else:
        pd.DataFrame().to_csv(
            args.out_dir / "bloomberg_final_vs_final_sampling_ci.csv", index=False
        )

    # ---- Coverage diagnostics: per (model, target_month) -----------------
    # Why each model has fewer scored events than the 69-event Bloomberg
    # universe: each LLM emits predictions for only some target_months.
    # We report the per-model target_month coverage so the user can see
    # the cause directly.
    coverage_rows: list[dict] = []
    bloomberg_targets = (
        release_meta.assign(target_month=release_meta["ref_period"])
        .groupby("target_month").size().rename("bloomberg_releases").reset_index()
    )
    for md in model_dirs:
        model_name = md.name[len("model_"):]
        preds, _ = load_model_predictions(md, date_only_policy=args.date_only_policy)
        tm = preds["target_month"].astype(str).value_counts().sort_index()
        for t, n in tm.items():
            coverage_rows.append({
                "model": model_name, "target_month": t,
                "n_predictions": int(n),
                "in_bloomberg_universe": (t in set(release_meta["ref_period"])),
            })
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(
        args.out_dir / "model_target_month_coverage.csv", index=False
    )

    # ---- Report ---------------------------------------------------------
    L: list[str] = []
    A = L.append
    A("=" * 78)
    A("Step 15.4 (Bloomberg overlay) — live scoring report")
    A("=" * 78)
    A(f"Bloomberg daily file:    {args.bloomberg_daily_csv}")
    A(f"Live GT file:            {args.live_gt_csv}")
    A(f"Day offsets scored:      {DAY_OFFSETS}")
    A(f"Model history window:    {max_history}")
    A(f"Within-day aggregator:   {args.aggregator}")
    A(f"Live releases (kept):    {len(live_releases)}")
    A(f"Live releases with Bloomberg coverage: {n_live_with_coverage} "
      f"(unmatched = {len(live_releases) - n_live_with_coverage}; the unmatched "
      f"are pre-Nov-2025 target months — LLMs do not predict them and Bloomberg "
      f"snapshots do not cover them.)")
    A("")

    A("-- per-model target_month coverage (drives per-model event count) --")
    A("    A model can only score (Bloomberg-covered) events whose ref_period")
    A("    equals one of its emitted target_months. Models with narrower")
    A("    target-month coverage have fewer matched events.")
    if not coverage_df.empty:
        view = coverage_df.pivot_table(
            index="model", columns="target_month", values="n_predictions",
            aggfunc="sum", fill_value=0,
        ).astype(int)
        A(view.to_string())
    A("")
    A("Bloomberg has releases for these (field, ref_period) target_months:")
    A(f"  {sorted(release_meta['ref_period'].unique())}")
    A("")

    def _fmt_score(v: float) -> str:
        return "NA" if pd.isna(v) else f"{v:+.4f}"

    def _fmt_rmse(v: float) -> str:
        return "NA" if pd.isna(v) else f"{v:.2f}"

    def _fmt_n(v) -> str:
        return "NA" if pd.isna(v) else f"{int(v)}"

    if not aggregate_df.empty:
        A("-- aggregate BMSC / BP_RMSE / WDH (one row per model) --")
        cols = [
            "model",
            "score_on_median_1d_BMSC", "score_on_median_1d_n_events",
            "score_on_median_3d_BMSC", "score_on_median_3d_n_events",
            "score_on_median_7d_BMSC", "score_on_median_7d_n_events",
            "final_vs_final_BMSC", "final_vs_final_n_events",
        ]
        cols = [c for c in cols if c in aggregate_df.columns]
        view = aggregate_df[cols].copy()
        for c in cols:
            if c == "model":
                continue
            if c.endswith("_n_events"):
                view[c] = view[c].map(_fmt_n)
            elif "BP_RMSE" in c:
                view[c] = view[c].map(_fmt_rmse)
            else:
                view[c] = view[c].map(_fmt_score)
        A(view.to_string(index=False))
        A("")
        A("-- aggregate BDRC / DRC_RMSE (one row per model) --")
        cols2 = [
            "model",
            "score_on_median_1d_BDRC", "score_on_median_1d_DRC_RMSE",
            "score_on_median_3d_BDRC", "score_on_median_3d_DRC_RMSE",
            "score_on_median_7d_BDRC", "score_on_median_7d_DRC_RMSE",
            "final_vs_final_BDRC", "final_vs_final_DRC_RMSE",
            "final_vs_final_n_drc_events",
        ]
        cols2 = [c for c in cols2 if c in aggregate_df.columns]
        if len(cols2) > 1:
            view2 = aggregate_df[cols2].copy()
            for c in cols2:
                if c == "model":
                    continue
                if c.endswith("_n_drc_events"):
                    view2[c] = view2[c].map(_fmt_n)
                elif "DRC_RMSE" in c:
                    view2[c] = view2[c].map(_fmt_rmse)
                else:
                    view2[c] = view2[c].map(_fmt_score)
            A(view2.to_string(index=False))
        A("")
    if ci_info.get("computed"):
        A("-- final_vs_final 95% CI (parametric bootstrap of beta) --")
        ci_path = args.out_dir / "bloomberg_final_vs_final_ci.csv"
        if ci_path.exists():
            ci_df = pd.read_csv(ci_path)
            ci_cols = ["model", "n_events", "BMSC_point", "BMSC_ci95_lo", "BMSC_ci95_hi",
                       "BDRC_point", "BDRC_ci95_lo", "BDRC_ci95_hi"]
            ci_cols = [c for c in ci_cols if c in ci_df.columns]
            if ci_cols and not ci_df.empty:
                v = ci_df[ci_cols].copy()
                for c in ci_cols:
                    if c == "model":
                        continue
                    if c == "n_events":
                        v[c] = v[c].map(_fmt_n)
                    else:
                        v[c] = v[c].map(_fmt_score)
                A(v.to_string(index=False))
        A("")
        A(f"  Huber refit: λ = {ci_info['lambda']:.6g}, "
          f"σ = {ci_info['sigma']:.4e}, df_eff = {ci_info['df_eff']:.2f}")
        A(f"  bootstrap n_boot = {ci_info['n_boot']}, seed = {ci_info['seed']}")
        A("")
    if sampling_ci_info.get("computed"):
        A("-- final_vs_final 95% sampling CI "
          "(t-CI on N latest predictions per (field, event)) --")
        sci_path = args.out_dir / "bloomberg_final_vs_final_sampling_ci.csv"
        if sci_path.exists():
            sci_df = pd.read_csv(sci_path)
            sci_cols = ["model", "n_ranks_used",
                        "BMSC_mean", "BMSC_sd", "BMSC_ci95_lo", "BMSC_ci95_hi",
                        "BDRC_mean", "BDRC_sd", "BDRC_ci95_lo", "BDRC_ci95_hi"]
            sci_cols = [c for c in sci_cols if c in sci_df.columns]
            if sci_cols and not sci_df.empty:
                v = sci_df[sci_cols].copy()
                for c in sci_cols:
                    if c == "model":
                        continue
                    if c == "n_ranks_used":
                        v[c] = v[c].map(_fmt_n)
                    elif c.endswith("_sd"):
                        v[c] = v[c].map(lambda x: "NA" if pd.isna(x) else f"{x:.4f}")
                    else:
                        v[c] = v[c].map(_fmt_score)
                A(v.to_string(index=False))
        A("")
        A(f"  N = {N_LATEST_SNAPSHOTS} latest predictions per (field, event); "
          f"t_{{{N_LATEST_SNAPSHOTS-1}, 0.975}} = "
          f"{_t_critical(N_LATEST_SNAPSHOTS-1, 0.05):.4f}")
        A("  Each rank k uses the k-th most recent prediction (after the "
          "10x-median outlier filter)")
        A("  against the same Bloomberg final-summary median C.")
        A("  Captures stability of the model's most recent emissions; β fixed.")
        A("")
    A("Notes:")
    A("  * C_d uses Bloomberg cumulative-MEDIAN carry-forward to end of each")
    A("    day_offset d in {1..7} (calendar days before release). Default")
    A("    --consensus-aggregator median; pass mean to switch.")
    A("  * M_d uses MEDIAN of model predictions emitted on calendar day d")
    A("    only ([start_of_day_d, end_of_day_d]); days with no preds are")
    A("    dropped (no carry-forward). Default --aggregator median_of_day.")
    A("  * score_on_median_1d / 3d / 7d use per-event MEDIAN of (C_d, M_d) over")
    A("    the day-offsets where data exists. 1d = {d=1}; 3d = {d=1,2,3};")
    A("    7d = {d=1..7}.")
    A("  * final_vs_final uses Bloomberg final-summary MEDIAN (or mean) and")
    A("    the model's latest pre-release prediction (single value).")
    A("  * BDRC / DRC_RMSE are computed on every variant.")
    A("  * Outlier filter: |log10(value / per-(variable, target_month) median)|")
    A(f"    > {args.outlier_log10_threshold:.1f} (i.e., {10**args.outlier_log10_threshold:g}x).")
    A("    Applied to every LLM; per-model drop counts in outlier_filter_audit.csv.")
    A("  * CIs (final_vs_final and score_on_median_{1d,3d,7d}): parametric")
    A("    bootstrap of β from N(β̂, V_β_HC1), where V_β is the Huber-ridge")
    A("    sandwich. Same n_boot / seed / β draws across variants — only the")
    A("    per-event (S, S_hat) field rows differ, so bars across variants")
    A("    share the same β-uncertainty model. Magnitude scores follow Addendum")
    A("    3 bounded forms; directional scores are return-weighted hits.")
    A("  * Sampling CI (final_vs_final): t-CI on K=5 model-level scores,")
    A("    where rank k uses the k-th latest prediction (per field-event)")
    A("    against the same Bloomberg final-summary median C. All ranks pull")
    A("    from the outlier-filtered prediction set. Captures stability of")
    A("    the model's most recent emissions; β fixed (no β uncertainty).")
    A("    Reported alongside the bootstrap CI as a complementary stability")
    A("    check; per-rank intermediate is bloomberg_final_vs_final_n_latest_field.csv.")
    A("  * ARIMA benchmark replicates its single static forecast across")
    A("    the 7 pre-release calendar days (timestamp at 12:00 ET each day)")
    A("    so the median_of_day aggregator picks it up on every d.")
    report_text = "\n".join(L) + "\n"
    (args.out_dir / "bloomberg_live_scoring_report.txt").write_text(report_text)
    print(report_text)

    meta = {
        "step": "15.4 Bloomberg overlay (final, self-contained)",
        "day_offsets": list(DAY_OFFSETS),
        "score_on_median_1d_offsets": list(SCORE_ON_MEDIAN_1D_OFFSETS),
        "score_on_median_3d_offsets": list(SCORE_ON_MEDIAN_3D_OFFSETS),
        "score_on_median_7d_offsets": list(SCORE_ON_MEDIAN_7D_OFFSETS),
        "aggregator": args.aggregator,
        "consensus_aggregator": args.consensus_aggregator,
        "outlier_log10_threshold": float(args.outlier_log10_threshold),
        "outlier_filter_applied_to": "all LLM models",
        "huber_lambda": float(args.lam),
        "huber_c": float(args.huber_c),
        "huber_fit": {k: v for k, v in fit_info.items()
                      if k not in ("beta_by_field", "marginal_se_HC1")},
        "n_live_releases": int(len(live_releases)),
        "n_live_with_bloomberg": int(n_live_with_coverage),
        "models": [r["model"] for r in all_aggregate_rows],
        "ci": ci_info,
        "sampling_ci": sampling_ci_info,
        "inputs": {
            "mapping": str(args.mapping),
            "sigma": str(args.sigma),
            "ts_events_csv": str(args.ts_events_csv),
            "live_gt": str(args.live_gt_csv),
            "coverage": str(args.coverage_csv),
            "models_root": str(args.models_root),
            "bloomberg_daily_csv": str(args.bloomberg_daily_csv),
            "bloomberg_release_csv": str(args.bloomberg_release_csv),
            "arima_csv": str(args.arima_csv) if args.arima_csv else None,
        },
    }
    (args.out_dir / "bloomberg_live_scoring_metadata.json").write_text(
        json.dumps(meta, indent=2, default=str)
    )
    print(f"\nWrote outputs under: {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
