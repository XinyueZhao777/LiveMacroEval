"""Bloomberg-overlay live updater for Results/market_surprise_capture_score.

Refreshes every Bloomberg-overlay artifact that depends on the latest data
drops. The historical layer (mapping table, sigma_i) is frozen by Project
Outline §14.1 and intentionally NOT re-run here. Production betas are re-fit
inline by build_live_scoring_bloomberg.py (Huber-ridge with HC1 sandwich);
the legacy stand-alone score_pipeline*.py scripts have been moved to
step_15_4_live_scoring/archived/ and are no longer invoked.

Pipeline executed (in order):
  1. (optional) Results/ground_truth/scrape_incremental.py + parse_ground_truth.py
       Refresh field_releases_live.csv from the Investing.com calendar (the
       source of A and release_datetime_et). Skip with --skip-gt when the
       calendar is already current.
  2. Results/data_sp500futures/extract_event_windows.py
       Rebuild event_window_coverage.csv + event_windows_1min.parquet
       (futures hf_return) from the newest ES_*.zip in that folder. Skip
       with --skip-extract when coverage is already current.
  3. Results/bloomberg_consensus/build_daily_consensus.py
       Build bloomberg_daily_consensus.csv + bloomberg_release_consensus.csv
       from the per-release Bloomberg .xlsx snapshots.
  4. step_15_4_live_scoring/build_live_scoring_bloomberg.py
       Production headline. Writes step_15_4_live_scoring/bloomberg_overlay/.
  5. step_15_5_scoring_by_theme/score_by_theme_bloomberg.py
       Theme-restricted BMSC/BDRC/WDH on the bloomberg_overlay field
       CSVs.
  6. Plots: step_15_4_live_scoring/plots/plot_final_scores.py,
            step_15_4_live_scoring/plots/plot_sampling_scores.py,
            step_15_5_scoring_by_theme/plots/plot_theme_bloomberg.py.

Manual upstream data drops (NOT automated here — run the data ingest first):
  - Bloomberg .xlsx snapshots into Results/bloomberg_consensus/.
  - serverA/B raw drops into Results/data_from_serverA_serverB/data_serverA_<DATE>/
    + data_serverB_<DATE>/, then run concat_server_data.py + concat_model_data.py.
  - ES futures bars + contract dates zips into Results/data_sp500futures/.

Conda env: livemacro.

Usage:
  python update_live_scoring.py
  python update_live_scoring.py --skip-gt              # ground truth already fresh
  python update_live_scoring.py --skip-extract         # ES futures coverage already fresh
  python update_live_scoring.py --skip-plots           # only refresh CSVs
  python update_live_scoring.py --live-start 2025-11-01
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent          # Results/market_surprise_capture_score
RESULTS = HERE.parent                 # Results

# Step 1 — ground truth.
SCRAPE_INCREMENTAL = RESULTS / "ground_truth" / "scrape_incremental.py"
PARSE_GROUND_TRUTH = RESULTS / "ground_truth" / "parse_ground_truth.py"

# Step 2 — futures hf_return.
EXTRACT_EVENT_WINDOWS = RESULTS / "data_sp500futures" / "extract_event_windows.py"

# Step 3 — Bloomberg daily consensus.
BUILD_DAILY_CONSENSUS = RESULTS / "bloomberg_consensus" / "build_daily_consensus.py"

# Step 4-5 — overlay scores.
STEP_15_4 = HERE / "step_15_4_live_scoring"
BUILD_OVERLAY = STEP_15_4 / "build_live_scoring_bloomberg.py"
STEP_15_5 = HERE / "step_15_5_scoring_by_theme"
SCORE_BY_THEME = STEP_15_5 / "score_by_theme_bloomberg.py"

# Step 6 — plots.
PLOT_FINAL = STEP_15_4 / "plots" / "plot_final_scores.py"
PLOT_SAMPLING = STEP_15_4 / "plots" / "plot_sampling_scores.py"
PLOT_THEME = STEP_15_5 / "plots" / "plot_theme_bloomberg.py"

# Inputs depended on by the pipeline (preflight checks).
GT_LIVE = RESULTS / "ground_truth" / "data" / "field_releases_live.csv"
MODELS_ROOT = RESULTS / "data_from_serverA_serverB" / "final_analysis_data"
BLOOMBERG_DIR = RESULTS / "bloomberg_consensus"


def run(cmd: list[str], label: str) -> None:
    print("=" * 78)
    print(f"[update_live_scoring] {label}")
    print(f"[update_live_scoring] $ {' '.join(cmd)}")
    print("=" * 78, flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise SystemExit(
            f"[update_live_scoring] {label} failed with exit code "
            f"{result.returncode} (after {elapsed:.1f}s)."
        )
    print(f"[update_live_scoring] {label} done ({elapsed:.1f}s).\n")


def preflight(skip_gt: bool) -> None:
    """Fail fast if expected inputs are missing, with actionable messages."""
    missing: list[str] = []
    if skip_gt and not GT_LIVE.exists():
        missing.append(
            f"  ground-truth live file not found: {GT_LIVE}\n"
            f"    drop --skip-gt to refresh, or run "
            f"`python {PARSE_GROUND_TRUTH}` manually."
        )
    if not MODELS_ROOT.exists():
        missing.append(
            f"  LLM prediction root not found: {MODELS_ROOT}\n"
            f"    run the data_from_serverA_serverB pipeline first "
            f"(concat_server_data.py + concat_model_data.py)."
        )
    else:
        model_dirs = [
            d for d in MODELS_ROOT.iterdir()
            if d.is_dir() and d.name.startswith("model_")
        ]
        if not model_dirs:
            missing.append(
                f"  no model_* directories found under {MODELS_ROOT}.\n"
                f"    ensure the LLM-ingest pipeline has written model folders."
            )
    if not BLOOMBERG_DIR.exists() or not list(BLOOMBERG_DIR.glob("*.xlsx")):
        missing.append(
            f"  no Bloomberg .xlsx files in {BLOOMBERG_DIR}.\n"
            f"    drop the per-release Bloomberg Terminal snapshots first."
        )
    if missing:
        raise SystemExit(
            "[update_live_scoring] preflight failed:\n" + "\n".join(missing)
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--skip-gt",
        action="store_true",
        help="Skip Investing.com calendar refresh + parse_ground_truth.py.",
    )
    p.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip rerunning extract_event_windows.py (futures hf_return).",
    )
    p.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip the plot scripts at the end.",
    )
    p.add_argument(
        "--live-start",
        type=str,
        default="2025-11-01",
        help="inclusive live-window start (default 2025-11-01).",
    )
    args = p.parse_args(argv)

    preflight(skip_gt=args.skip_gt)
    python = sys.executable

    # Step 1: ground truth (Investing.com -> field_releases_live.csv).
    if not args.skip_gt:
        run(
            [python, str(SCRAPE_INCREMENTAL)],
            label="Step 1a: scrape_incremental.py (Investing.com calendar)",
        )
        run(
            [python, str(PARSE_GROUND_TRUTH), "--live-start", args.live_start],
            label="Step 1b: parse_ground_truth.py -> field_releases_live.csv",
        )
    else:
        print(
            "[update_live_scoring] --skip-gt: reusing existing field_releases_live.csv.\n"
        )

    # Step 2: futures hf_return.
    if not args.skip_extract:
        run(
            [python, str(EXTRACT_EVENT_WINDOWS)],
            label="Step 2: extract_event_windows.py (futures hf_return)",
        )
    else:
        print(
            "[update_live_scoring] --skip-extract: reusing existing event_window_coverage.csv.\n"
        )

    # Step 3: Bloomberg daily consensus.
    run(
        [python, str(BUILD_DAILY_CONSENSUS)],
        label="Step 3: build_daily_consensus.py",
    )

    # Step 4: Bloomberg overlay scores.
    run(
        [python, str(BUILD_OVERLAY), "--live-start", args.live_start],
        label="Step 4: build_live_scoring_bloomberg.py -> bloomberg_overlay/",
    )

    # Step 5: Theme scoring.
    run(
        [python, str(SCORE_BY_THEME)],
        label="Step 5: score_by_theme_bloomberg.py",
    )

    # Step 6: Plots.
    if not args.skip_plots:
        run(
            [python, str(PLOT_FINAL), "--all-variants"],
            label="Step 6a: plot_final_scores.py --all-variants",
        )
        run(
            [python, str(PLOT_SAMPLING), "--all-variants"],
            label="Step 6b: plot_sampling_scores.py --all-variants",
        )
        run(
            [python, str(PLOT_THEME), "--all-variants"],
            label="Step 6c: plot_theme_bloomberg.py --all-variants",
        )
    else:
        print("[update_live_scoring] --skip-plots: skipping plot scripts.\n")

    print("=" * 78)
    print("[update_live_scoring] All requested outputs refreshed.")
    print("=" * 78)
    print("  ground_truth/data/field_releases_live.csv")
    print("  data_sp500futures/event_window_coverage.csv")
    print("  bloomberg_consensus/bloomberg_daily_consensus.csv")
    print("  bloomberg_consensus/bloomberg_release_consensus.csv")
    print("  step_15_4_live_scoring/bloomberg_overlay/")
    print("  step_15_5_scoring_by_theme/scores_by_theme_bloomberg_*.csv")
    print("  step_15_4_live_scoring/plots/  step_15_5_scoring_by_theme/plots/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
