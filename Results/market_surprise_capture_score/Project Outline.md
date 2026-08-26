# LiveMacro Score — methodology

This folder implements the **LiveMacro Score** described in §3.5 of the paper.
The score asks how much of the equity-market-relevant macro surprise an LLM
agent anticipated, relative to the consensus baseline.

## 1. Setup

Let $\mathcal{I}$ be the set of scored indicators (the 16 in the benchmark)
and let $g \in \mathcal{G}$ index *timestamp-group events* — all indicators
that share a single release timestamp (e.g. nonfarm payrolls + unemployment
rate at 08:30 ET on the same morning).

For each indicator $i$ and release $t$:

- $A_{i,t}$ — released value
- $C_{i,t}$ — prevailing consensus
- $M_{i,t}^{-}$ — the agent's latest nowcast strictly before release time $T_{i,t}$
- $\sigma_i$ — historical scale of surprises in indicator $i$

The standardized realized and model-implied surprises are

$$S_{i,t} = \frac{A_{i,t} - C_{i,t}}{\sigma_i}, \qquad
  \hat S_{i,t} = \frac{M_{i,t}^{-} - C_{i,t}}{\sigma_i}.$$

For each event $g$,

$$X_{i,g} = \begin{cases} S_{i,t} & i \text{ released in } g \\ 0 & \text{otherwise} \end{cases},
\qquad
\hat X_{i,g} = \begin{cases} \hat S_{i,t} & i \text{ released in } g \\ 0 & \text{otherwise.} \end{cases}$$

## 2. Announcement-window return

The market reaction is the log return on E-mini S&P 500 futures over a tight
window around the release time $T_g$:

$$r_g^{HF} = \log P^{ES}(T_g + 30\text{m}) - \log P^{ES}(T_g - 5\text{m}).$$

## 3. Historical β (frozen)

Following the high-frequency event-study convention without a constant
(Gürkaynak, Kısacıkoğlu & Wright, 2018), we fit

$$r_g^{HF} = \sum_i \beta_i X_{i,g} + u_g$$

on a long historical sample by no-intercept Huber-ridge regression. The
resulting $\{\hat\beta_i\}$ encode each indicator's long-run causal weight
on the announcement-window equity return and are **frozen** before any live
event is scored. They are stored under
`step_15_2_historical_preprocessing/` and never re-estimated on a live
refresh.

## 4. Live scoring

Define the realized and model-implied β-weighted shocks at event $g$:

$$Q_g = \sum_i \hat\beta_i X_{i,g}, \qquad
  \hat Q_g = \sum_i \hat\beta_i \hat X_{i,g}.$$

The consensus baseline is by construction $\hat Q_g^{\text{cons}} \equiv 0$.

The LiveMacro Score aggregates across the live sample $g \in \mathcal{G}^{live}$,
treating $\hat Q_g$ as the model's implied announcement-window return:

$$\mathrm{SS_{model}} = \sum_g \bigl(r_g^{HF} - \hat Q_g\bigr)^2, \qquad
  \mathrm{SS_{cons}}  = \sum_g \bigl(r_g^{HF}\bigr)^2,$$

$$\boxed{\;
\mathrm{LiveMacroScore} =
  \frac{\mathrm{SS_{cons}} - \mathrm{SS_{model}}}
       {\mathrm{SS_{cons}} + \mathrm{SS_{model}}} \in [-1, 1].
\;}$$

A score of $+1$ corresponds to nowcasts that perfectly anticipate
announcement-window returns; $0$ matches the consensus; negative values are
worse than the consensus. Restricting the sum to events whose indicators
fall in a given thematic block (production / inflation–consumption /
labor / housing) yields the theme-restricted score reported in Figure 4 of
the paper.

## 5. Pipeline layout

```
market_surprise_capture_score/
├── Project Outline.md                       ← this file
├── update_live_scoring.py                   ← orchestrator
├── step_15_1_mapping_layer/                 ← indicator key → field_id table
├── step_15_2_historical_preprocessing/      ← FROZEN β, σ, historical events
├── step_15_4_live_scoring/                  ← live scoring + plot scripts
│   ├── SCORE_CALCULATION.md
│   ├── build_live_scoring_bloomberg.py      ← per-event scoring driver
│   ├── arima_predictions.py                 ← ARIMA baseline runner
│   ├── cv_lambda_no_intercept.py            ← (one-time) λ tuning for the historical fit
│   └── plots/                               ← figure rendering scripts
└── step_15_5_scoring_by_theme/              ← per-theme decomposition + figure rendering
```

## 6. Inputs and outputs

Inputs (read by `build_live_scoring_bloomberg.py`):

- `step_15_2_historical_preprocessing/timestamp_group_events.csv` — historical
  events used for the β fit.
- The consensus series (Bloomberg ECOS median) joined per release.
- `field_releases_live.csv` from `Results/ground_truth/` — released values.
- `event_windows_1min.parquet` from `Results/data_sp500futures/` — ES windows.
- Per-model nowcast tables from
  `Results/data_from_serverA_serverB/final_analysis_data/model_<model>/`.

Outputs (per model):

- Event-level scored CSVs and the aggregate LiveMacro Score table.
- Parametric-bootstrap CIs (90% / 95%) on every metric.
- Theme-restricted scores from `step_15_5_scoring_by_theme/`.

The aggregate per-model LiveMacro Score reproduces Figure 2 in the paper;
the per-theme decomposition reproduces Figure 4.
