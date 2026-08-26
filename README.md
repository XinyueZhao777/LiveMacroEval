# LiveMacroEval

A live, contamination-resistant benchmark for evaluating LLM agents on U.S. macroeconomic nowcasting.

This is the code and data release for *Can LLMs Take the Pulse of the Economy? A Real-Time Evaluation of LLM Nowcasts on Macroeconomic Indicators*.

LLM agents produce hourly nowcasts for sixteen U.S. headline macroeconomic indicators over a pre-release window that closes at each official release. Because every nowcast is made before the release exists, the evaluation cannot be contaminated by pretraining data. Nowcast quality is scored two ways: a **LiveMacro Score** measured against announcement-window equity returns, and a **LiveBetting Score** measured as the return from simulated Polymarket-style trading. Federal Reserve regional-bank nowcasts, the Bloomberg ECOS professional consensus, and an auto-ARIMA model serve as comparators.

## What is in this repository

```
LiveMacroEval/
├── LiveMacro/            (1) the live nowcasting service
│   ├── backend/          scheduler, LLM clients, storage, prompts
│   ├── config/jobs.json  one job per (target month, variable group, model)
│   └── data_light/       hourly nowcast records
└── Results/              (2) the offline analytics that produce the paper's figures
    ├── data_from_serverA_serverB/      unified per-model nowcast tables
    ├── ground_truth/                   release calendar scraper and parser
    ├── data_consensus/                 calendar scrapers and macro-event filters
    ├── data_sp500futures/              ES futures event-window extractor
    ├── benchmark_econ/                 auto-ARIMA univariate baseline
    ├── market_surprise_capture_score/  LiveMacro Score
    ├── polymarket_return/              LiveBetting Score
    └── remove_outlier_and_plot/        outlier filtering and figure renderers
```

`README_supplement_original.md` is the detailed walkthrough of every module, including the output schemas, the job configuration fields, and how to add a new model. Read that one for module-level detail. This file covers setup, data availability, and what you can and cannot reproduce from the public release.

## Setup

Developed and tested against Python 3.10.

```bash
conda create -n livemacro python=3.10
conda activate livemacro
pip install -r requirements.txt
```

Set the API keys for whichever models you intend to run:

```bash
export OPENAI_API_KEY=...        # gpt-5-search-api
export ANTHROPIC_API_KEY=...     # claude-sonnet-4.5-api, claude-code-agent
export OPENROUTER_API_KEY=...    # OpenRouter-routed models
```

No key is required to inspect the released nowcast data or to re-run the scoring code on it.

## Running the nowcasting service

```bash
cd LiveMacro
chmod +x backend/start_scheduler.sh
PYTHON_BIN=$(which python) backend/start_scheduler.sh
```

Each entry in `config/jobs.json` defines one target month, one variable group, and one model. The scheduler calls the model on a fixed interval inside an open prediction window, parses the returned `key=value` line, and appends a row to `data/model_<model>/<target_period>_<variable_group>.csv`. A one-shot smoke test across every configured job is available:

```bash
python backend/test_manual_run.py --model gpt-5-search-api
```

## Data availability

All nowcast records for `gpt-5-search-api` are in the repository. Nothing needs to be downloaded separately.

`LiveMacro/data/` holds the complete records, including the `raw_model_output` column with the model's full response and the `citations` column with its web-search sources. `LiveMacro/data_light/` holds the same rows with those two text columns removed, which is the more convenient file if you only need the numeric nowcasts. `Results/data_from_serverA_serverB/final_analysis_data/` holds the unified per-model tables the scoring code reads.

These directories expand to roughly 260MB on checkout. They compress extremely well, so the clone itself is only a few megabytes.

Third-party inputs that we cannot redistribute are documented in [DATA_SOURCES.md](DATA_SOURCES.md), which lists every external source, its license status, and how to obtain it. In short, the Bloomberg ECOS consensus is distributed only through a paid Bloomberg Terminal subscription, and the E-mini S&P 500 minute bars come from a commercial vendor. Both are excluded here. The scripts that consume them are included and documented, so anyone with the underlying subscriptions can regenerate the missing inputs.

## What you can reproduce from this release

Please read this section before reporting that a figure does not regenerate.

**Fully reproducible.** The nowcasting service itself, for any model you have an API key for. The auto-ARIMA baseline, which runs off FRED and the included external snapshot. The LiveBetting Score, which runs off the included Polymarket bucket prices and Federal Reserve nowcast series. All the figure-rendering scripts, given their inputs.

**Reproducible only with a Bloomberg Terminal subscription.** The headline LiveMacro Score and its theme decomposition, meaning Figures 2 and 4. The scoring driver reads `Results/bloomberg_consensus/bloomberg_daily_consensus.csv` and `bloomberg_release_consensus.csv`, which hold the ECOS survey medians. We are not permitted to redistribute those files. Everything else the score needs is in the repository, including the frozen historical betas and sigmas under `step_15_2_historical_preprocessing/` and the event-window returns under `data_sp500futures/`.

**Not yet released.** The paper evaluates four LLM agents. This initial release contains nowcast records for GPT-5 only. The Claude and Qwen records, and the corresponding job definitions, are planned for a later release. The shipped OpenRouter client is a working template, so you can point it at Qwen or any other OpenRouter-served model by changing `MODEL_ID`.

**Known gap.** The two Figure 5 renderers under `Results/remove_outlier_and_plot/` import a shared
helper module, `generate_plots_serverA_serverB.py`, which is not in this release. Both scripts will
fail on import until it is added. It supplies `POST_OUTLIER_Z_THRESHOLD`, `_robust_scale`, and
`remove_outliers`, the canonical outlier-removal logic. The outlier-cleaned data these scripts read
is already included under `processed_final_analysis_data/`, so only the rendering step is affected.

## A note on the release calendar

Release timestamps and event names are scraped from the public Investing.com economic calendar, which matches the Bloomberg Economic Calendar on both fields and is far easier to access programmatically. This applies to the calendar only.

The consensus values are a separate matter. Investing.com publishes its own forecast column, which is **not** the Bloomberg ECOS median. Wherever the paper reports a comparison against the professional consensus, the consensus is the Bloomberg ECOS survey median taken from the Terminal, joined through `Results/bloomberg_consensus/`. Several scripts carry a `_bloomberg` suffix for exactly this reason. Investing.com supplies the released actual value and the release timestamp, and nothing else.

## License

Code is released under the MIT License, see [LICENSE](LICENSE).

The nowcast data generated by this project is released under CC BY 4.0, see [DATA_LICENSE.md](DATA_LICENSE.md). Third-party data retains its original terms and is covered in [DATA_SOURCES.md](DATA_SOURCES.md).

## Citation

See [CITATION.cff](CITATION.cff).
