from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

@dataclass(frozen=True)
class ADFResult:
    test_stat: float | None
    stationary_5pct: bool
    used_lags: int
    nobs: int
    critical_values: dict[str, float]


def adf_test(series: pd.Series, max_lags: int = 12) -> ADFResult:
    clean = pd.Series(series).dropna().astype(float)
    if len(clean) < 25:
        return ADFResult(None, False, 0, len(clean), {})

    candidate_max_lag = min(max_lags, max(len(clean) // 3, 0))
    for lag in range(candidate_max_lag, -1, -1):
        try:
            test_stat, _pvalue, used_lags, nobs, critical_values, *_ = adfuller(
                clean,
                maxlag=lag,
                autolag="BIC",
                regression="c",
            )
            critical_values = {key: float(value) for key, value in critical_values.items()}
            return ADFResult(
                test_stat=float(test_stat),
                stationary_5pct=float(test_stat) < critical_values["5%"],
                used_lags=int(used_lags),
                nobs=int(nobs),
                critical_values=critical_values,
            )
        except ValueError:
            continue

    return ADFResult(None, False, 0, len(clean), {})


def apply_transform(series: pd.Series, transform_name: str) -> pd.Series:
    clean = pd.Series(series).astype(float)

    if transform_name == "level":
        result = clean
    elif transform_name == "log_level":
        if (clean.dropna() <= 0).any():
            raise ValueError("log_level requires strictly positive observations")
        result = np.log(clean)
    elif transform_name == "diff":
        result = clean.diff()
    elif transform_name == "log_diff":
        if (clean.dropna() <= 0).any():
            raise ValueError("log_diff requires strictly positive observations")
        result = np.log(clean).diff()
    elif transform_name == "diff2":
        result = clean.diff().diff()
    elif transform_name == "log_diff2":
        if (clean.dropna() <= 0).any():
            raise ValueError("log_diff2 requires strictly positive observations")
        result = np.log(clean).diff().diff()
    else:
        raise ValueError(f"Unknown transform: {transform_name}")

    result.name = series.name
    return result


def inverse_transform_forecast(
    original_history: pd.Series,
    transformed_forecast: list[float],
    transform_name: str,
) -> list[float]:
    original_history = pd.Series(original_history).dropna().astype(float)
    if original_history.empty:
        raise ValueError("Cannot invert forecasts without history")

    if transform_name == "level":
        return [float(value) for value in transformed_forecast]
    if transform_name == "log_level":
        return [float(math.exp(value)) for value in transformed_forecast]
    if transform_name == "diff":
        last = float(original_history.iloc[-1])
        forecasts: list[float] = []
        for value in transformed_forecast:
            last += float(value)
            forecasts.append(last)
        return forecasts
    if transform_name == "log_diff":
        last = float(original_history.iloc[-1])
        forecasts = []
        for value in transformed_forecast:
            last *= math.exp(float(value))
            forecasts.append(last)
        return forecasts
    if transform_name == "diff2":
        history = list(original_history.astype(float))
        if len(history) < 2:
            raise ValueError("diff2 inversion requires at least two historical observations")
        forecasts = []
        prev, curr = float(history[-2]), float(history[-1])
        for value in transformed_forecast:
            nxt = 2.0 * curr - prev + float(value)
            forecasts.append(nxt)
            prev, curr = curr, nxt
        return forecasts
    if transform_name == "log_diff2":
        history = list(original_history.astype(float))
        if len(history) < 2:
            raise ValueError("log_diff2 inversion requires at least two historical observations")
        log_prev, log_curr = math.log(float(history[-2])), math.log(float(history[-1]))
        forecasts = []
        for value in transformed_forecast:
            log_nxt = 2.0 * log_curr - log_prev + float(value)
            forecasts.append(math.exp(log_nxt))
            log_prev, log_curr = log_curr, log_nxt
        return forecasts
    raise ValueError(f"Unknown transform: {transform_name}")


def choose_auto_transform(series: pd.Series, prefers_logs: bool, max_adf_lags: int = 12) -> tuple[str, ADFResult]:
    clean = pd.Series(series).dropna().astype(float)
    positive = bool((clean > 0).all())

    if prefers_logs and positive:
        candidate_order = ["log_level", "log_diff", "level", "diff"]
    else:
        candidate_order = ["level", "diff"]

    candidates: list[tuple[str, ADFResult]] = []
    for candidate in candidate_order:
        if candidate.startswith("log") and not positive:
            continue
        transformed = apply_transform(clean, candidate).dropna()
        if len(transformed) < 25:
            continue
        adf = adf_test(transformed, max_lags=max_adf_lags)
        candidates.append((candidate, adf))
        if adf.stationary_5pct:
            return candidate, adf

    if not candidates:
        fallback = "level" if not prefers_logs or not positive else "log_level"
        transformed = apply_transform(clean, fallback).dropna()
        return fallback, adf_test(transformed, max_lags=max_adf_lags)

    return min(
        candidates,
        key=lambda item: item[1].test_stat if item[1].test_stat is not None else float("inf"),
    )


def rmse(actual: pd.Series, forecast: pd.Series) -> float:
    errors = forecast - actual
    return float(np.sqrt(np.mean(np.square(errors))))


def mae(actual: pd.Series, forecast: pd.Series) -> float:
    return float(np.mean(np.abs(forecast - actual)))


def smape(actual: pd.Series, forecast: pd.Series) -> float | None:
    denom = np.abs(actual) + np.abs(forecast)
    valid = denom > 0
    if not valid.any():
        return None
    value = np.mean(200.0 * np.abs(forecast[valid] - actual[valid]) / denom[valid])
    return float(value)


def mape(actual: pd.Series, forecast: pd.Series) -> float | None:
    valid = actual != 0
    if not valid.any():
        return None
    value = np.mean(100.0 * np.abs((forecast[valid] - actual[valid]) / actual[valid]))
    return float(value)
