# LiveMacroEval — supplementary code & sample nowcast data

Supplementary materials for *Can LLMs Take the Pulse of the Economy? A Real-Time
Evaluation of LLM Nowcasts on Macroeconomic Indicators*. This bundle contains:

1. **`LiveMacro/`** — the live nowcasting service that prompts an LLM agent on
   an hourly schedule and writes one CSV per (model, target month, variable
   group).
2. **`Results/`** — the offline analytics that turn nowcast CSVs into the
   paper's tables and figures: ground-truth scraping, ES-futures event
   windowing, the LiveMacro score, the LiveBetting score, theme decomposition,
   and the auto-ARIMA econometric baseline.

A sample of the nowcast outputs (`gpt-5-search-api`) is bundled under
`LiveMacro/data/` and `Results/data_from_serverA_serverB/final_analysis_data/`.

---

## Top-level layout

```
code/
├── README.md             ← this file
├── LiveMacro/            ← (1) live nowcasting service
│   ├── backend/          ← scheduler + LLM clients + storage
│   ├── config/jobs.json  ← one job per (target_month, var_group, model)
│   ├── data/             ← raw nowcast CSVs (with raw_model_output + citations)
│   └── data_light/       ← same rows minus raw_model_output / citations
└── Results/              ← (2) offline analytics
    ├── data_from_serverA_serverB/   ← unified per-model nowcast tables
    ├── ground_truth/                ← Bloomberg calendar scraper + field_releases_live.csv
    ├── data_consensus/              ← Bloomberg calendar scrapers
    ├── data_sp500futures/           ← ES futures event-window extractor + sample
    ├── benchmark_econ/              ← auto-ARIMA univariate baseline
    ├── market_surprise_capture_score/  ← LiveMacro score (BMSC / BP-RMSE / BDRC / WDH)
    ├── polymarket_return/           ← LiveBetting score
    └── remove_outlier_and_plot/     ← outlier filtering + paper-figure renderers
```

Note on the Bloomberg calendar: Bloomberg's economic calendar is identical
in content to the public Investing.com calendar (Investing.com is the source
the shipped scrapers actually use because it is the easier of the two to
scrape).

---

## Environment

Developed against Python 3.10. Suggested setup:

```bash
conda create -n livemacro python=3.10
conda activate livemacro
pip install openai anthropic claude-agent-sdk pandas numpy scipy statsmodels \
            scikit-learn matplotlib seaborn pyarrow requests pmdarima \
            python-dateutil tzdata
```

Set the API keys whose models you actually intend to run:

```bash
export OPENAI_API_KEY=...        # for gpt-5-search-api
export ANTHROPIC_API_KEY=...     # for claude-sonnet-4.5-api / claude-code-agent
export OPENROUTER_API_KEY=...    # for OpenRouter-routed models (Qwen, etc.)
```

---

## Part 1 — `LiveMacro/` (live nowcasting service)

Each entry in `config/jobs.json` defines one (target month, variable group,
model) and instructs the scheduler to call the model on a fixed interval
inside an open prediction window. Each call prompts the model with the
forecasting task (see `backend/prompts.py`), parses the returned
`key=value` line, and appends a row to
`data/model_<model>/<target_period>_<variable_group>.csv`. The same row is
mirrored, minus the raw text columns, to `data_light/` for compact
distribution.

### Quickstart

```bash
cd LiveMacro
chmod +x backend/start_scheduler.sh
PYTHON_BIN=$(which python) backend/start_scheduler.sh           # all jobs
PYTHON_BIN=$(which python) backend/start_scheduler.sh any       # same (matches runner="any")
```

A one-shot smoke test that exercises every (job, model) in jobs.json:

```bash
cd LiveMacro
python backend/test_manual_run.py                # all models
python backend/test_manual_run.py --model gpt-5-search-api
```

Control files (touch from any shell):

```bash
mkdir -p LiveMacro/backend/control
touch LiveMacro/backend/control/STOP        # graceful shutdown
touch LiveMacro/backend/control/PAUSE       # pause but keep process
```

### `config/jobs.json`

The shipped file contains 36 jobs: 6 target months × 2 variable groups ×
3 LLM models. Every job uses `interval_secs = 3600` (hourly) and opens its
window on the 25th of the target month, matching the paper's "final week
of the reference month" convention. `stop_at` is `null` (run until
official release; in practice the scheduler is stopped manually or via the
STOP file after the release).

| Field             | Meaning                                                       |
|-------------------|---------------------------------------------------------------|
| `id`              | Unique job key (`<target>_<group>_<modeltag>`)                |
| `target_period`   | Reference period of the indicator, `YYYY-MM`                  |
| `release_period`  | Calendar month of the official release                        |
| `variable_group`  | `core_macroeconomic_conditions` or `demand_sectoral_activity` |
| `models`          | List of model keys (see `backend/llm_clients/__init__.py`)    |
| `interval_secs`   | Seconds between nowcast calls (hourly = 3600)                 |
| `start_at`        | ISO-8601 with offset; job becomes due at or after this time   |
| `stop_at`         | ISO-8601 or `null` (no stop)                                  |
| `status`          | `running` to activate; anything else to disable               |
| `last_run_local`  | Auto-updated by the scheduler; set to `null` before first run |
| `runner`          | `any` (the supplement ships a single logical runner)          |

### Adding a model

1. Implement a `generate(system_msg, user_msg) -> (text, citations)` callable in
   a new file under `backend/llm_clients/`.
2. Wire its key in `backend/llm_clients/__init__.py::get_client`.
3. Reference the key in any job's `models` list.

The shipped clients are: `gpt-5-search-api`, `claude-sonnet-4.5-api`,
`claude-code-agent`, and `claude-sonnet-4.5-openrouter` (the OpenRouter
client is a template — change `MODEL_ID` to target Qwen or any other
OpenRouter-served model).

### Output schema

`data/model_<model>/<target_period>_<variable_group>.csv`:

| Column              | Notes                                                  |
|---------------------|--------------------------------------------------------|
| `timestamp_local`   | When the call returned (America/New_York)              |
| `target_month`      | `YYYY-MM` (echoed back by the model)                   |
| `release_month`     | `YYYY-MM` of the scheduled official release            |
| `job_id`, `model`   | Provenance                                             |
| `variable`          | One of the keys in `backend/variables.py`              |
| `value`             | The model's predicted numeric value (native units)     |
| `parsed_ok`, `parsed_notes` | Output-format check                            |
| `raw_model_output`  | Full LLM response (only in `data/`, not `data_light/`) |
| `citations`         | Web-search citations (only in `data/`)                 |

---

## Part 2 — `Results/` (offline analytics)

The order below mirrors the paper.

### 2.1 Ingest LLM nowcasts → `final_analysis_data/`

```
Results/data_from_serverA_serverB/
├── concat_server_data.py
├── concat_model_data.py
├── data_pipeline_utils.py
└── final_analysis_data/      ← unified per-model nowcast tables
```

`concat_model_data.py` produces a single canonical CSV per (model,
target_period, variable_group) and writes
`final_analysis_data/variable_prediction_ranges.csv`, the coverage table
the ARIMA baseline and the LiveBetting code rely on.

### 2.2 Ground truth → `field_releases_live.csv`

```
Results/ground_truth/
├── scrape_incremental.py        ← extends the Bloomberg calendar
├── parse_ground_truth.py        ← canonicalizes to first-release-per-period
└── data/field_releases_live.csv ← live-window ground truth
```

Output schema: `field_id, release_datetime_et, event, ref_period, A, C, raw_surprise`.

### 2.3 Bloomberg calendar → `data_consensus/`

Calendar scrapers and filters. The shipped sample contains only the scripts;
the calendar itself can be re-scraped with `scrape_investing_calendar.py`
(Bloomberg's economic calendar is identical in content to the public
Investing.com calendar, and Investing.com is easier to scrape).

### 2.4 ES-futures event windows → `data_sp500futures/`

```
Results/data_sp500futures/
├── extract_event_windows.py     ← slices 1-min ES bars around each release
├── event_windows_1min.parquet   ← derived event windows (small sample)
├── event_window_coverage.csv
└── sanity_report.txt
```

### 2.5 Auto-ARIMA baseline → `benchmark_econ/`

```
Results/benchmark_econ/
├── run_benchmark.py             ← orchestrator
├── fred_loader.py / external_loader.py / variable_config.py
├── models.py / time_series_tools.py
└── external_data_2026_04_16/    ← small external snapshot for series not in FRED
```

Run:

```bash
python Results/benchmark_econ/run_benchmark.py \
    --mode target-file --ic bic \
    --output-dir Results/benchmark_econ/results_targets_2025_11_2026_03
```

### 2.6 LiveMacro Score → `market_surprise_capture_score/`

The paper's primary metric (§3.5, Figures 2 & 4). `Project Outline.md` is
the canonical methodology document.

```
Results/market_surprise_capture_score/
├── Project Outline.md
├── update_live_scoring.py                   ← orchestrator
├── step_15_1_mapping_layer/                 ← variable_key → field_id mapping
├── step_15_2_historical_preprocessing/      ← frozen historical β / σ / event sample
├── step_15_4_live_scoring/                  ← βᵢ regression + live scoring
│   ├── SCORE_CALCULATION.md
│   ├── build_live_scoring_bloomberg.py
│   ├── arima_predictions.py
│   ├── cv_lambda_no_intercept.py
│   └── plots/                               ← figure rendering scripts
└── step_15_5_scoring_by_theme/              ← per-theme decomposition
    ├── score_by_theme_bloomberg.py
    ├── theme_membership_bloomberg.csv
    └── plots/
```

### 2.7 LiveBetting Score → `polymarket_return/`

The paper's secondary metric (§3.5, Figure 3).

```
Results/polymarket_return/
├── fetch_polymarket_prices.py       ← Polymarket CLOB API scraper
├── calculate_earnings.py            ← per-model hourly bet simulator
├── calculate_bloomberg_earnings.py  ← same loop driven by Bloomberg consensus
├── plot_continuous_feb_mar.py       ← Figure 3 generator
├── polymarket/                      ← sample bucket-price CSVs
├── fed/                             ← sample Fed nowcast CSVs
└── feb_mar_continuous_20260519/     ← Figure 3 cumulative-returns curves
```

Per-month bucket schemas are not stable across Polymarket rounds. Adding a
new month requires editing the `EVENTS` dict in
`fetch_polymarket_prices.py` and the `VARIABLE_CONFIGS[var]['months']` in
`calculate_earnings.py`.

### 2.8 Paper-figure renderers → `remove_outlier_and_plot/`

```
Results/remove_outlier_and_plot/
├── render_paper_cpi_pce.py             ← Fig. 5 CPI / PCE case study
├── render_paper_all_macro_2026_03.py   ← Fig. 5-style panels for the full 2026-03 group
└── processed_final_analysis_data/      ← outlier-cleaned mirror used downstream
```

---

## Reproducing the paper's figures

| Paper figure                              | Producing script                                            |
|-------------------------------------------|-------------------------------------------------------------|
| Fig. 2 — LiveMacro Score                  | `Results/market_surprise_capture_score/step_15_4_live_scoring/plots/plot_final_scores.py` |
| Fig. 3 — LiveBetting cumulative returns   | `Results/polymarket_return/plot_continuous_feb_mar.py`      |
| Fig. 4 — LiveMacro Score by theme         | `Results/market_surprise_capture_score/step_15_5_scoring_by_theme/plots/plot_theme_bloomberg.py` |
| Fig. 5 — CPI / PCE nowcast case study     | `Results/remove_outlier_and_plot/render_paper_cpi_pce.py`   |
