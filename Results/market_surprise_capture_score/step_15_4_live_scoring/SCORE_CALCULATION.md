# Live scoring — calculation reference

Implementation: `build_live_scoring_bloomberg.py`. This file documents
the math implemented in that script; the surrounding methodology is in
`../Project Outline.md`.

## 1. Historical β fit (frozen)

On the historical event sample
`../step_15_2_historical_preprocessing/timestamp_group_events.csv`
(surprise columns `X_<field_id>` plus the announcement-window
`hf_return`), fit the no-intercept Huber-ridge regression

$$r_g^{HF} = \sum_i \beta_i X_{i,g} + u_g$$

by IRLS from an OLS-ridge initialization (Huber tuning constant
$c = 1.345$; ridge penalty $\lambda \approx 21.54 = 10^{4/3}$, selected by
blocked chronological 5-fold CV — see `cv_lambda_no_intercept.py`).
The HC1 sandwich $V_\beta$ is computed alongside the point estimate.

The no-intercept specification follows Gürkaynak, Kısacıkoğlu & Wright
(2018, NBER 25016): under the identifying assumption that the
announcement-window return responds only to released surprises, a zero
surprise carries no expected return, so $\alpha \equiv 0$ by construction.

## 2. Live scoring per event

For each live event $g$:

$$Q_g = \sum_i \hat\beta_i X_{i,g}, \qquad
\hat Q_g = \sum_i \hat\beta_i \hat X_{i,g}.$$

The consensus baseline reduces to $\hat Q_g^{\text{cons}} = 0$ because
$X_{i,g}$ is the standardized surprise (consensus implies zero surprise).

## 3. Aggregate metrics

Let $n$ be the number of live events with valid ES return and $n_{\text{drc}}$
the subset where futures data is available for the BDRC variant. Define

$$\mathrm{SS_{model}^{BMSC}} = \sum_g (\hat Q_g - Q_g)^2, \qquad
\mathrm{SS_{cons}^{BMSC}}  = \sum_g Q_g^2,$$

$$\mathrm{SS_{model}^{BDRC}} = \sum_g (r_g^{HF} - \hat Q_g)^2, \qquad
\mathrm{SS_{cons}^{BDRC}}  = \sum_g (r_g^{HF})^2.$$

| Metric    | Formula                                                                 |
|-----------|-------------------------------------------------------------------------|
| BMSC      | $(\mathrm{SS_{cons}^{BMSC}} - \mathrm{SS_{model}^{BMSC}}) / (\mathrm{SS_{cons}^{BMSC}} + \mathrm{SS_{model}^{BMSC}})$ |
| BDRC      | $(\mathrm{SS_{cons}^{BDRC}} - \mathrm{SS_{model}^{BDRC}}) / (\mathrm{SS_{cons}^{BDRC}} + \mathrm{SS_{model}^{BDRC}})$ |
| BP-RMSE   | $10^4 \cdot \sqrt{\mathrm{SS_{model}^{BMSC}} / n}$                       |
| DRC-RMSE  | $10^4 \cdot \sqrt{\mathrm{SS_{model}^{BDRC}} / n_{\text{drc}}}$          |
| WDH       | $\sum_g \lvert Q_g\rvert \cdot \mathbb{1}\{\mathrm{sign}(\hat Q_g) = \mathrm{sign}(Q_g)\} / \sum_g \lvert Q_g\rvert$ |
| DRC-WDH   | $\sum_g \lvert r_g^{HF}\rvert \cdot \mathbb{1}\{\mathrm{sign}(\hat Q_g) = \mathrm{sign}(r_g^{HF})\} / \sum_g \lvert r_g^{HF}\rvert$ |

BDRC corresponds to the **LiveMacro Score** in the paper (Figure 2); the
BMSC and BP-RMSE variants are reported in the appendix.

## 4. Parametric-bootstrap CIs

For each metric we draw $\beta^{(b)} \sim \mathcal{N}(\hat\beta, V_\beta)$
for $b = 1, \dots, B$ ($B = 10{,}000$, seed `20260427`), recompute the
metric with $\beta^{(b)}$, and report the 90% and 95% percentile intervals.

## 5. Run order

```
python cv_lambda_no_intercept.py       # optional, only when the design changes
python build_live_scoring_bloomberg.py # main scoring driver
python plots/plot_final_scores.py --all-variants
```

`update_live_scoring.py` in the parent folder orchestrates the live refresh
end-to-end.
