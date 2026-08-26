from __future__ import annotations

from dataclasses import dataclass
import math
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import arma_order_select_ic

from time_series_tools import apply_transform


@dataclass(frozen=True)
class ForecastResult:
    model_name: str
    order_label: str
    selected_p: int
    selected_d: int
    selected_q: int
    criterion_name: str
    criterion_value: float
    transform_name: str
    transformed_forecast: list[float]
    forecast: list[float]


def _ensure_supported_index(series: pd.Series) -> pd.Series:
    if not isinstance(series.index, pd.DatetimeIndex):
        return series
    if series.index.freq is not None:
        return series

    inferred_freq = pd.infer_freq(series.index)
    if inferred_freq is not None:
        supported_index = pd.DatetimeIndex(series.index.to_numpy(), freq=inferred_freq)
        return pd.Series(series.to_numpy(), index=supported_index, name=series.name)

    if len(series.index) < 2:
        return series

    deltas_days = np.diff(series.index.values).astype("timedelta64[D]").astype(int)
    median_step = int(np.median(deltas_days))
    if 28 <= median_step <= 31:
        freq = "MS"
    elif 89 <= median_step <= 92:
        freq = "QS"
    else:
        return series

    full_index = pd.date_range(series.index.min(), series.index.max(), freq=freq)
    return series.reindex(full_index).rename(series.name)


def _prepare_base_series(original_series: pd.Series, transform_name: str) -> tuple[pd.Series, int, bool]:
    clean = _ensure_supported_index(pd.Series(original_series).dropna().astype(float))
    if clean.empty:
        raise ValueError("Cannot fit a model on an empty series.")

    if transform_name == "level":
        return clean, 0, False
    if transform_name == "log_level":
        return pd.Series(np.log(clean), index=clean.index, name=clean.name), 0, True
    if transform_name == "diff":
        return clean, 1, False
    if transform_name == "log_diff":
        return pd.Series(np.log(clean), index=clean.index, name=clean.name), 1, True
    if transform_name == "diff2":
        return clean, 2, False
    if transform_name == "log_diff2":
        return pd.Series(np.log(clean), index=clean.index, name=clean.name), 2, True
    raise ValueError(f"Unknown transform: {transform_name}")


def _rank_candidate_orders(
    stationary_series: pd.Series,
    max_p: int,
    max_q: int,
    min_p: int,
    ic: str,
) -> list[tuple[int, int, float]]:
    clean = pd.Series(stationary_series).dropna().astype(float)
    if len(clean) < max(min_p + 10, 25):
        raise ValueError("Not enough transformed observations to select an ARIMA order.")

    criterion_name = ic.lower()
    if criterion_name not in {"aic", "bic"}:
        raise ValueError("ic must be either 'aic' or 'bic'")

    max_order_cap = max(len(clean) // 3, 0)
    effective_max_p = min(max_p, max_order_cap)
    effective_max_q = min(max_q, max_order_cap)
    if effective_max_p < min_p:
        raise ValueError("The sample is too short for the requested AR lag range.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", ValueWarning)
        selection = arma_order_select_ic(
            clean,
            max_ar=effective_max_p,
            max_ma=effective_max_q,
            ic=criterion_name,
            trend="c",
            model_kw={"enforce_stationarity": False, "enforce_invertibility": False},
        )

    criteria_table = getattr(selection, criterion_name)
    allowed = criteria_table.loc[criteria_table.index >= min_p]
    ranked = allowed.stack(future_stack=True).dropna().sort_values()
    if ranked.empty:
        raise ValueError("statsmodels could not rank any admissible ARIMA orders for this series.")

    return [(int(ar), int(ma), float(score)) for (ar, ma), score in ranked.items()]


def _fit_arima(base_series: pd.Series, order: tuple[int, int, int], trend: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", ValueWarning)
        return ARIMA(
            base_series,
            order=order,
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()


def fit_and_forecast_arima(
    original_series: pd.Series,
    transform_name: str,
    steps: int,
    max_p: int = 12,
    max_d: int = 1,
    max_q: int = 3,
    ic: str = "bic",
    min_p: int = 0,
    model_name: str = "arima",
) -> ForecastResult:
    clean = _ensure_supported_index(pd.Series(original_series).dropna().astype(float))
    stationary_series = _ensure_supported_index(apply_transform(clean, transform_name).dropna())
    base_series, d, log_output = _prepare_base_series(clean, transform_name)

    if d > max_d:
        raise ValueError(f"Transform {transform_name} requires d={d}, but --max-d is only {max_d}.")

    if d == 0:
        trend = "c"
    elif d == 1:
        trend = "t"
    else:
        trend = None
    ranked_orders = _rank_candidate_orders(
        stationary_series=stationary_series,
        max_p=max_p,
        max_q=max_q,
        min_p=min_p,
        ic=ic,
    )

    criterion_name = ic.lower()
    last_error: Exception | None = None
    for p, q, selection_score in ranked_orders:
        order = (p, d, q)
        try:
            fitted = _fit_arima(base_series, order=order, trend=trend)
            transformed_forecast = [float(value) for value in fitted.forecast(steps=steps)]
            forecast = [float(math.exp(value)) for value in transformed_forecast] if log_output else transformed_forecast
            criterion_value = float(getattr(fitted, criterion_name))
            if not np.isfinite(criterion_value):
                criterion_value = selection_score
            return ForecastResult(
                model_name=model_name,
                order_label=f"ARIMA{order}",
                selected_p=p,
                selected_d=d,
                selected_q=q,
                criterion_name=criterion_name,
                criterion_value=criterion_value,
                transform_name=transform_name,
                transformed_forecast=transformed_forecast,
                forecast=forecast,
            )
        except Exception as exc:
            last_error = exc
            continue

    if last_error is None:
        raise ValueError("Could not fit any ARIMA specification.")
    raise ValueError(f"Could not fit any ARIMA specification: {last_error}") from last_error


def fit_and_forecast_ar(
    original_series: pd.Series,
    transform_name: str,
    steps: int,
    max_p: int = 12,
    ic: str = "bic",
    min_p: int = 1,
) -> ForecastResult:
    return fit_and_forecast_arima(
        original_series=original_series,
        transform_name=transform_name,
        steps=steps,
        max_p=max_p,
        max_d=1,
        max_q=0,
        ic=ic,
        min_p=min_p,
        model_name="ar",
    )
