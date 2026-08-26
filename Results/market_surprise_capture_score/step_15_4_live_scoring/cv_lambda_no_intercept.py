"""Blocked-chronological K-fold CV for the Huber-ridge ridge penalty λ
under the NO-INTERCEPT design.

Gürkaynak, Kısacıkoğlu & Wright (2018, "Missing Events in Event Studies")
specify the event-study regression as

    y_t = β' s_t + ε_t        (no intercept)

— see paper equations (2.1), (2.3), (3.1), (3.2), (4.1), (4.2). After
removing the intercept from `fit_huber_ridge`/`huber_ridge_sandwich` in
`build_live_scoring_bloomberg.py`, the CV-selected λ must be re-derived
because the X design changed (X is no longer mean-centered around an
unpenalized intercept).

CV protocol (matches the legacy CV in archived/score_pipeline.py):
  - K=5 contiguous chronological folds (training set = all rows outside
    fold k; validation set = fold k).
  - λ grid: np.logspace(-4, 4, 25).
  - Huber prediction loss on the validation set,
        ρ_δ(u) = 0.5·u²            if |u| ≤ δ,
                 δ·|u| − 0.5·δ²     otherwise,
    where δ = c · σ̂_train (Huber tuning const c=1.345; σ̂_train from
    rescaled MAD of the train residuals, bounded below by the σ from the
    final fit on the train fold).
  - λ_best = argmin CV(λ). λ_1se = largest λ with CV(λ) ≤ CV_best + SE.

Run:
  conda activate livemacro && \
      python step_15_4_live_scoring/cv_lambda_no_intercept.py

Inputs (defaults work in the standard layout):
  --ts-events     timestamp_group_events.csv  (step_15_2)
  --mapping       field_mapping.csv           (step_15_1)
  --huber-c       1.345
  --k             5
  --out-dir       step_15_4_live_scoring/

Outputs:
  cv_lambda_no_intercept.csv     full CV(λ) table
  cv_lambda_no_intercept.json    selected λ_best, λ_1se + metadata
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # Results/market_surprise_capture_score
STEP_15_1 = ROOT / "step_15_1_mapping_layer"
STEP_15_2 = ROOT / "step_15_2_historical_preprocessing"

DEFAULT_TS_EVENTS = STEP_15_2 / "timestamp_group_events.csv"
DEFAULT_MAPPING = STEP_15_1 / "field_mapping.csv"

# Match build_live_scoring_bloomberg.HUBER_C; explicit to keep this script
# self-contained for reproducibility.
HUBER_C = 1.345
LAMBDA_GRID = np.logspace(-4.0, 4.0, 25)


# ---------------------------------------------------------------------------
# Huber-ridge primitives (NO INTERCEPT). Kept in-file (rather than imported
# from build_live_scoring_bloomberg) so this CV stays self-contained: that
# script reads HUBER_LAMBDA at import time, and we explicitly do not want
# the CV runner to be tangled with the production constant.
# ---------------------------------------------------------------------------
def _mad_scale(u: np.ndarray) -> float:
    med = float(np.median(u))
    return 1.4826 * float(np.median(np.abs(u - med)))


def _huber_weights(u: np.ndarray, delta: float) -> np.ndarray:
    abs_u = np.abs(u)
    out = abs_u > delta
    safe = np.where(out, abs_u, 1.0)
    return np.where(out, delta / safe, 1.0)


def _huber_loss(u: np.ndarray, delta: float) -> np.ndarray:
    """Element-wise ρ_δ(u)."""
    absu = np.abs(u)
    inside = absu <= delta
    return np.where(inside, 0.5 * u * u, delta * absu - 0.5 * delta * delta)


def fit_huber_ridge_no_intercept(
    X: np.ndarray, y: np.ndarray, lam: float,
    c: float = HUBER_C, max_iter: int = 100, tol: float = 1e-7,
    sigma_floor: float = 1e-12,
) -> dict:
    """IRLS Huber-ridge with NO intercept (mirrors build_live_scoring_bloomberg)."""
    n, p = X.shape
    if n < p + 1:
        raise ValueError(f"Need n > p, got n={n}, p={p}.")

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
    return {"beta": beta, "sigma": float(sigma), "delta": float(delta),
            "n_iter": n_iter, "converged": converged}


def cv_one_lambda(X: np.ndarray, y: np.ndarray, lam: float, k: int, c: float) -> tuple[float, list[float]]:
    """Blocked chronological K-fold CV; returns (mean_loss, per_fold_losses)."""
    n = len(y)
    if k < 2:
        raise ValueError(f"Need k>=2 folds, got {k}.")
    fold_size = n // k
    losses: list[float] = []
    for i in range(k):
        v_lo = i * fold_size
        v_hi = (i + 1) * fold_size if i < k - 1 else n
        val_idx = np.arange(v_lo, v_hi)
        train_idx = np.concatenate([np.arange(0, v_lo), np.arange(v_hi, n)])
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        fit_k = fit_huber_ridge_no_intercept(X[train_idx], y[train_idx], lam, c=c)
        u_val = y[val_idx] - X[val_idx] @ fit_k["beta"]
        sigma_val = max(_mad_scale(u_val), fit_k["sigma"])
        delta_val = c * sigma_val
        losses.append(float(_huber_loss(u_val, delta_val).mean()))
    return float(np.mean(losses)), losses


# ---------------------------------------------------------------------------
# Entrypoint.
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ts-events", type=Path, default=DEFAULT_TS_EVENTS)
    p.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    p.add_argument("--huber-c", type=float, default=HUBER_C)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--out-dir", type=Path, default=HERE)
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(args.ts_events, parse_dates=["release_timestamp"])
    events = events.sort_values("release_timestamp", kind="mergesort").reset_index(drop=True)
    mapping = pd.read_csv(args.mapping)
    field_ids = mapping["field_id"].tolist()
    surprise_cols = [f"X_{f}" for f in field_ids]
    missing = [c for c in surprise_cols if c not in events.columns]
    if missing:
        raise SystemExit(f"Missing surprise columns in ts_events: {missing}")

    X = events[surprise_cols].to_numpy(dtype=float)
    y = events["hf_return"].to_numpy(dtype=float)
    if np.isnan(X).any() or np.isnan(y).any():
        raise SystemExit("X or y contains NaNs; step 15.2 should have filtered.")

    print(f"  ts_events       : {args.ts_events}")
    print(f"  n events        : {len(events):,}")
    print(f"  n fields (p)    : {len(field_ids)}")
    print(f"  Huber c         : {args.huber_c}")
    print(f"  folds (K)       : {args.k}")
    print(f"  λ grid (n={len(LAMBDA_GRID)}): {LAMBDA_GRID[0]:.4g} … {LAMBDA_GRID[-1]:.4g}")
    print()

    rows: list[dict] = []
    for lam in LAMBDA_GRID:
        mean_loss, fold_losses = cv_one_lambda(X, y, float(lam), args.k, args.huber_c)
        se = (float(np.std(fold_losses, ddof=1) / np.sqrt(len(fold_losses)))
              if len(fold_losses) > 1 else float("nan"))
        row = {"lambda": float(lam), "cv_huber_loss": mean_loss, "cv_se": se}
        for j, fm in enumerate(fold_losses):
            row[f"fold_loss_{j}"] = fm
        rows.append(row)
        print(f"  λ={float(lam):>12.6e}  loss={mean_loss:.6e}  ±{se:.2e}")

    cv_df = pd.DataFrame(rows)
    best = int(cv_df["cv_huber_loss"].idxmin())
    lam_best = float(cv_df.loc[best, "lambda"])
    loss_best = float(cv_df.loc[best, "cv_huber_loss"])
    se_best = float(cv_df.loc[best, "cv_se"])
    threshold = loss_best + (se_best if not np.isnan(se_best) else 0.0)
    eligible = cv_df[cv_df["cv_huber_loss"] <= threshold]
    lam_1se = float(eligible["lambda"].max()) if not eligible.empty else lam_best

    print()
    print(f"  λ_best          : {lam_best:.10g}   (loss={loss_best:.6e}, SE={se_best:.2e})")
    print(f"  λ_1se           : {lam_1se:.10g}")
    print()

    out_csv = args.out_dir / "cv_lambda_no_intercept.csv"
    out_json = args.out_dir / "cv_lambda_no_intercept.json"
    cv_df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps({
        "design": "y_t = β' s_t + ε_t (no intercept); Huber-ridge with L2 on β only",
        "paper_reference": "Gürkaynak, Kısacıkoğlu & Wright (2018) NBER 25016 — eqs. 2.1/2.3/3.1/3.2/4.1/4.2",
        "ts_events": str(args.ts_events),
        "mapping": str(args.mapping),
        "n_events": int(len(events)),
        "n_fields": int(len(field_ids)),
        "huber_c": float(args.huber_c),
        "k_folds": int(args.k),
        "lambda_grid": LAMBDA_GRID.tolist(),
        "lambda_best": lam_best,
        "lambda_best_loss": loss_best,
        "lambda_best_se": se_best,
        "lambda_1se": lam_1se,
    }, indent=2))

    print(f"  wrote {out_csv}")
    print(f"  wrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
