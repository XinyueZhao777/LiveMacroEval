#!/bin/bash
# Start the live-nowcast scheduler in the background.
#
# Required env vars (export before running):
#   OPENAI_API_KEY        for gpt-5-search-api
#   ANTHROPIC_API_KEY     for claude-sonnet-4.5-api / claude-code-agent
#   OPENROUTER_API_KEY    for OpenRouter-routed models (optional)
#
# Optional:
#   ALERT_EMAIL_FROM / ALERT_EMAIL_PASSWORD / ALERT_EMAIL_TO
#     SMTP alert when a job exhausts MAX_RETRY (no-op if unset).
#   PYTHON_BIN
#     Path to the Python interpreter (defaults to `python` on PATH).
#
# Usage:
#   ./start_scheduler.sh                # run all jobs in config/jobs.json
#   ./start_scheduler.sh <runner_name>  # run only jobs whose "runner" matches

set -e

RUNNER="$1"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter '$PYTHON_BIN' not found. Set PYTHON_BIN to your env's python."
  exit 1
fi

if [ -n "$RUNNER" ]; then
  nohup "$PYTHON_BIN" scheduler.py --runner "$RUNNER" >/dev/null 2>&1 &
  echo "Started scheduler on $(hostname) with runner=$RUNNER"
else
  nohup "$PYTHON_BIN" scheduler.py >/dev/null 2>&1 &
  echo "Started scheduler on $(hostname) (all jobs)"
fi
