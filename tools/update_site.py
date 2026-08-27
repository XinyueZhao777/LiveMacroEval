#!/usr/bin/env python3
"""Refresh the LiveMacroEval website from the Results pipeline outputs.

Run after the biweekly data refresh (see the private Results/UPDATE_PIPELINE.md).
Rewrites docs/data/leaderboard.json and re-copies the paper figures. The site
renders every number from that JSON, so this is the only step.

IMPORTANT — this repo is public, the scoring inputs are not.
------------------------------------------------------------
The scoring overlay this reads lives under
`Results/market_surprise_capture_score/**/bloomberg_overlay/`, which the repo's
.gitignore deliberately excludes: it is derived from the Bloomberg ECOS survey
and FirstRateData ES futures bars, neither of which may be redistributed
(see DATA_SOURCES.md). So:

  * the overlay is read from a SEPARATE PRIVATE CHECKOUT, never from this repo;
  * only aggregate scores (point estimate, CI, event count) are written into
    the site — the same numbers the paper reports in Figure 2;
  * no row-level, per-release, or raw vendor value is ever copied here.

Point --results-root at your private Results/ checkout (or set
LIVEMACRO_RESULTS). Run check_release_safety.py before every push.

Usage
-----
    . /home/ruiyi/anaconda3/bin/activate && conda activate livemacro
    python tools/update_site.py --dry-run
    python tools/update_site.py --overlay investing_overlay_0906 \
        --window "Investing.com consensus proxy - target periods Apr-Aug 2026"
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent         # <repo>/tools
REPO = TOOLS.parent                             # <repo>  (public)
SITE = REPO / "docs"                            # the published site

# The private working tree that holds the pipeline outputs and paper figures.
DEFAULT_RESULTS = Path(os.environ.get("LIVEMACRO_RESULTS", "/home/ruiyi/livemacro/Results"))
DEFAULT_FIGURES = Path(os.environ.get("LIVEMACRO_FIGURES", "/home/ruiyi/livemacro/Paper/figures"))

SCORING_SUBPATH = "market_surprise_capture_score/step_15_4_live_scoring"

# code arm -> display name shown on the site
MODEL_LABELS = {
    "gpt-5-search-api": "GPT-5",
    "claude-sonnet-4.5-api": "Claude-4.5-Sonnet",
    "qwen3-235b-a22b-instruct-2507": "Qwen3-235B",
    "qwen3-next-80b-a3b-instruct": "Qwen3-80B",
    "arima_aic": "auto-ARIMA",
    "claude-code-agent": "Claude Code agent",
    "claude-code-multiagent": "Claude Code multi-agent",
    "gpt-5-search-api-reasoned": "GPT-5 (reasoned)",
}
MODEL_KIND = {"arima_aic": "econ"}          # everything else defaults to "llm"

# Figure 2 in the paper drops this arm (n=11 outlier); keep the site consistent.
DROPPED = {"claude-code-agent"}

# Only these columns are ever read out of the overlay. Aggregates only.
COLS = ("model", "n_events", "BDRC_point", "BDRC_ci90_lo", "BDRC_ci90_hi")

FIGURES = [
    "bdrc_score_no_agent_ci90.png",
    "theme_production.png",
    "theme_inflation_consumption_services.png",
    "theme_labor_market.png",
    "theme_housing.png",
    "continuous_returns_real_gdp_qoq_mar-anchor-claude-sonnet-4.5.png",
    "continuous_returns_cpi_yoy_mar-anchor-claude-sonnet-4.5.png",
    "continuous_returns_unemployment_rate_mar-anchor-claude-sonnet-4.5.png",
    "case_study_cpi_yoy_2026_03.png",
    "case_study_pce_mom_2026_03.png",
]


def read_scores(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        sys.exit(
            f"scoring CSV not found:\n  {csv_path}\n\n"
            "This file is intentionally NOT in the public repo (see DATA_SOURCES.md).\n"
            "Point --results-root at your private Results/ checkout, e.g.\n"
            "  python tools/update_site.py --results-root /home/ruiyi/livemacro/Results"
        )

    rows = []
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in COLS if c not in (reader.fieldnames or [])]
        if missing:
            sys.exit(f"{csv_path} is missing expected columns: {missing}")
        for r in reader:
            model = r["model"]
            if model in DROPPED:
                continue
            rows.append({
                "name": MODEL_LABELS.get(model, model),
                "kind": MODEL_KIND.get(model, "llm"),
                "score": round(float(r["BDRC_point"]), 3),
                "ci": [round(float(r["BDRC_ci90_lo"]), 3), round(float(r["BDRC_ci90_hi"]), 3)],
                "events": int(r["n_events"]),
                "note": "",
            })

    rows.sort(key=lambda x: -x["score"])
    if rows:
        rows[0]["note"] = "leads the panel"

    # The consensus is 0 by construction and is not a row in the CSV.
    rows.insert(0, {
        "name": "Bloomberg ECOS consensus", "kind": "human", "score": 0.0,
        "ci": None, "events": None, "note": "reference, 0 by construction",
    })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS,
                    help="private Results/ checkout holding the scoring overlays "
                         "(default: %(default)s, or $LIVEMACRO_RESULTS)")
    ap.add_argument("--figures-root", type=Path, default=DEFAULT_FIGURES,
                    help="private Paper/figures/ directory (default: %(default)s)")
    ap.add_argument("--overlay", default="bloomberg_overlay",
                    help="overlay dir under %s (default: %%(default)s)" % SCORING_SUBPATH)
    ap.add_argument("--window", default=None, help="override the headline window caption")
    ap.add_argument("--next-update", default=None,
                    help="YYYY-MM-DD of the next refresh (default: today + 14 days)")
    ap.add_argument("--skip-figures", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the JSON, write nothing")
    args = ap.parse_args()

    if args.results_root.resolve() == (REPO / "Results").resolve():
        sys.exit(
            "refusing to read the public repo's own Results/ as the pipeline source.\n"
            "The scoring overlays are withheld from this repo by design; point\n"
            "--results-root at the private checkout instead."
        )

    out = SITE / "data/leaderboard.json"
    data = json.loads(out.read_text())

    csv_path = args.results_root / SCORING_SUBPATH / args.overlay / "bloomberg_final_vs_final_ci.csv"
    data["headline"]["rows"] = read_scores(csv_path)
    # Record the overlay by name only. The absolute private path stays out of the
    # published JSON.
    data["headline"]["source"] = f"{SCORING_SUBPATH}/{args.overlay}/ (not redistributed)"
    if args.window:
        data["headline"]["window"] = args.window

    today = dt.date.today()
    data["last_updated"] = today.isoformat()
    data["next_update"] = args.next_update or (today + dt.timedelta(days=14)).isoformat()

    if args.dry_run:
        print(json.dumps(data["headline"], indent=2))
        return

    out.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote docs/data/leaderboard.json  ({len(data['headline']['rows'])} rows)")

    if not args.skip_figures:
        dest = SITE / "assets/figures"
        for name in FIGURES:
            src = args.figures_root / name
            if src.exists():
                shutil.copy2(src, dest / name)
            else:
                print(f"  ! missing figure, left as-is: {name}")
        print("refreshed docs/assets/figures/")

    print("\nrunning the release-safety check...")
    sys.stdout.flush()
    rc = subprocess.call([sys.executable, str(TOOLS / "check_release_safety.py")])
    if rc != 0:
        sys.exit("\nrefresh written, but docs/ is NOT safe to publish. Fix the above "
                 "before committing.")
    print(f"\nnext:\n  git add docs && git commit -m 'site: refresh "
          f"{data['last_updated']}' && git push")


if __name__ == "__main__":
    main()
