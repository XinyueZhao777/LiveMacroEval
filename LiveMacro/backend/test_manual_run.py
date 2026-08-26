"""
Manual end-to-end smoke test: runs forecast_once -> append_predictions -> sync_data_light
-> git_commit_and_push for every job currently in config/jobs.json.

Usage (after exporting OPENAI_API_KEY / ANTHROPIC_API_KEY / OPENROUTER_API_KEY):

    python backend/test_manual_run.py                 # all jobs, all models
    python backend/test_manual_run.py --model X       # only model X across all jobs
"""

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import LOCAL_TZ, JOBS_PATH, get_logger
from forecasting import forecast_once
from storage import append_predictions, sync_data_light
from git_utils import git_commit_and_push

logger = get_logger(__name__)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default=None,
        help="If set, run only this model across every job that lists it. "
             "Otherwise run every (job, model) pair in jobs.json.",
    )
    return p.parse_args()


def _load_jobs():
    with open(JOBS_PATH, "r") as f:
        return json.load(f)


def main():
    args = _parse_args()
    now_local = datetime.now(LOCAL_TZ)
    print(f"\n{'=' * 60}")
    print(f"Manual end-to-end smoke test — {now_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"{'=' * 60}\n")

    jobs = _load_jobs()
    print(f"Loaded {len(jobs)} jobs from {JOBS_PATH}\n")

    results = {}

    for job in jobs:
        for model in job.get("models", []):
            if args.model and model != args.model:
                continue

            tag = f"{job['id']}:{model}"
            print(f"\n{'-' * 60}\nRunning: {tag}  group={job['variable_group']}\n{'-' * 60}")

            try:
                preds, meta = forecast_once(job, model, now_local)
                append_predictions(job, model, preds, meta)

                print(
                    f"\n[ok] parsed_ok={meta['parsed_ok']} notes={meta['parsed_notes']}\n"
                    f"     target_month={meta['target_month']} release_month={meta['release_month']}\n"
                    f"     {len(preds)} variables:"
                )
                for k, v in sorted(preds.items()):
                    print(f"       {k} = {v}")
                results[tag] = True
            except Exception as e:
                print(f"\n[fail] {e}")
                results[tag] = False

    print(f"\n{'=' * 60}\nSyncing data_light ...")
    try:
        sync_data_light()
        print("[ok] sync_data_light")
    except Exception as e:
        print(f"[fail] sync_data_light: {e}")

    print("\nGit commit & push ...")
    try:
        git_commit_and_push()
        print("[ok] git_commit_and_push")
    except Exception as e:
        print(f"[fail] git_commit_and_push: {e}")

    print(f"\n{'=' * 60}\nSummary:")
    for tag, ok in results.items():
        print(f"  {'[ok]  ' if ok else '[fail]'} {tag}")
    passed = sum(results.values())
    print(f"\n{passed}/{len(results)} (job, model) pairs succeeded")
    print(f"{'=' * 60}\n")

    if passed < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
