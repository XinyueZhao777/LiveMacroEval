from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from external_loader import (
    DEFAULT_EXTERNAL_DIR,
    augment_monthly_vintages,
    augment_truth_frame,
    load_external_panel,
)
from fred_loader import coalesced_truth_frame, load_all_vintages
from models import fit_and_forecast_ar, fit_and_forecast_arima
from time_series_tools import mae, mape, rmse, smape
from variable_config import (
    VARIABLE_SPECS,
    build_model_series,
    build_target_from_base_series,
    build_target_series,
    supported_variables,
    variable_catalog_records,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "FRED-MD-2015-01-to-2026-03"
DEFAULT_QD_DATA_DIR = SCRIPT_DIR / "FRED-QD 2018-05 to 2026-03"
DEFAULT_TARGETS_CSV = SCRIPT_DIR.parent / "data_from_serverA_serverB" / "final_analysis_data" / "variable_prediction_ranges.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"


@dataclass(frozen=True)
class TargetContext:
    request_period: pd.Period
    target_period_label: str
    target_date: pd.Timestamp
    horizon_unit: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple ARIMA macro forecasting benchmark with a restricted AR option.")
    parser.add_argument("--mode", choices=["backtest", "target-file", "both"], default="target-file")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--qd-data-dir", type=Path, default=DEFAULT_QD_DATA_DIR)
    parser.add_argument("--external-data-dir", type=Path, default=DEFAULT_EXTERNAL_DIR)
    parser.add_argument("--targets-csv", type=Path, default=DEFAULT_TARGETS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variables", default="all", help="Comma-separated variable list, or 'all'.")
    parser.add_argument("--horizons", default="1", help="Comma-separated monthly horizons for backtests, e.g. 1,3,6,12")
    parser.add_argument(
        "--model",
        choices=["ar", "arima"],
        default="arima",
        help="Use package-selected ARIMA by default, or restricted ARIMA with q=0 for the AR baseline.",
    )
    parser.add_argument("--ic", choices=["aic", "bic"], default="bic")
    parser.add_argument("--max-p", type=int, default=12)
    parser.add_argument("--min-p", type=int, default=0)
    parser.add_argument("--max-d", type=int, default=2)
    parser.add_argument("--max-q", type=int, default=3)
    parser.add_argument(
        "--target-months",
        default="all",
        help="Comma-separated target months for target-file mode, e.g. 2026-02,2026-03, or 'all'.",
    )
    parser.add_argument("--adf-max-lags", type=int, default=12)
    parser.add_argument("--min-train-size", type=int, default=60)
    return parser.parse_args()


def resolve_variables(variable_arg: str) -> list[str]:
    if variable_arg.strip().lower() == "all":
        return list(VARIABLE_SPECS.keys())
    variables = [item.strip() for item in variable_arg.split(",") if item.strip()]
    unknown = [item for item in variables if item not in VARIABLE_SPECS]
    if unknown:
        raise ValueError(f"Unknown variables: {', '.join(unknown)}")
    return variables


def resolve_horizons(horizon_arg: str) -> list[int]:
    horizons = sorted({int(item.strip()) for item in horizon_arg.split(",") if item.strip()})
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError("All horizons must be positive integers.")
    return horizons


def resolve_target_months(target_month_arg: str) -> list[str] | None:
    if target_month_arg.strip().lower() == "all":
        return None
    months = sorted({item.strip() for item in target_month_arg.split(",") if item.strip()})
    if not months:
        raise ValueError("At least one target month must be provided, or use 'all'.")
    for month in months:
        pd.Period(month, freq="M")
    return months


def month_timestamp(period_or_string: pd.Period | str) -> pd.Timestamp:
    if isinstance(period_or_string, pd.Period):
        return period_or_string.to_timestamp(how="start")
    return pd.Period(str(period_or_string), freq="M").to_timestamp(how="start")


def month_diff(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month)


def quarter_period(period_or_string: pd.Period | str) -> pd.Period:
    request_period = period_or_string if isinstance(period_or_string, pd.Period) else pd.Period(str(period_or_string), freq="M")
    return request_period.asfreq("Q")


def quarter_timestamp(period_or_string: pd.Period | str) -> pd.Timestamp:
    return quarter_period(period_or_string).asfreq("M", how="end").to_timestamp(how="start")


def quarter_label(period_or_string: pd.Period | str) -> str:
    return str(quarter_period(period_or_string))


def quarter_diff(start: pd.Timestamp, end: pd.Timestamp) -> int:
    start_period = pd.Period(start, freq="Q")
    end_period = pd.Period(end, freq="Q")
    return (end_period.year - start_period.year) * 4 + (end_period.quarter - start_period.quarter)


def truth_frame_for_spec(
    spec,
    monthly_truth_frame: pd.DataFrame,
    quarterly_truth_frame: pd.DataFrame,
) -> pd.DataFrame:
    if spec.dataset_family == "fred_qd":
        return quarterly_truth_frame
    return monthly_truth_frame


def vintages_for_spec(spec, monthly_vintages, quarterly_vintages):
    if spec.dataset_family == "fred_qd":
        return quarterly_vintages
    return monthly_vintages


def target_context(spec, target_month: str) -> TargetContext:
    request_period = pd.Period(target_month, freq="M")
    if spec.frequency == "quarterly":
        # Quarterly GDP requests arrive as monthly request months, but the modeled
        # target is the containing quarter's quarter-end timestamp.
        return TargetContext(
            request_period=request_period,
            target_period_label=quarter_label(request_period),
            target_date=quarter_timestamp(request_period),
            horizon_unit="quarters",
        )
    return TargetContext(
        request_period=request_period,
        target_period_label=str(request_period),
        target_date=month_timestamp(request_period),
        horizon_unit="months",
    )


def run_model(
    series: pd.Series,
    model_name: str,
    transform_name: str,
    steps: int,
    args: argparse.Namespace,
):
    if model_name == "ar":
        return fit_and_forecast_ar(
            original_series=series,
            transform_name=transform_name,
            steps=steps,
            max_p=args.max_p,
            ic=args.ic,
            min_p=args.min_p,
        )
    return fit_and_forecast_arima(
        original_series=series,
        transform_name=transform_name,
        steps=steps,
        max_p=args.max_p,
        max_d=args.max_d,
        max_q=args.max_q,
        ic=args.ic,
        min_p=args.min_p,
    )


TCODE_LABELS = {
    "1": "no transformation (levels)",
    "2": "first difference",
    "3": "second difference",
    "4": "natural log",
    "5": "first difference of natural log",
    "6": "second difference of natural log",
    "7": "first difference of percent change",
}


TCODE_TO_TRANSFORM = {
    "1": "level",
    "2": "diff",
    "3": "diff2",
    "4": "log_level",
    "5": "log_diff",
    "6": "log_diff2",
}


def transform_from_tcode(tcode: str | None) -> str | None:
    if tcode is None:
        return None
    return TCODE_TO_TRANSFORM.get(tcode)


def normalize_tcode(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return str(int(float(text)))
    except ValueError:
        return text


PMI_LOG_AR1_BASE_SERIES = frozenset({"ISM_PMI", "ISM_SVC"})


def effective_model_args_for_spec(spec, args: argparse.Namespace) -> argparse.Namespace:
    effective = argparse.Namespace(**vars(args))
    if spec.base_series in PMI_LOG_AR1_BASE_SERIES:
        # Temporary PMI policy: model log(index) with a fixed AR(1).
        # If we revisit PMI later, try seasonal models on diff directly instead of
        # expanding the non-seasonal lag order in levels.
        effective.min_p = 1
        effective.max_p = 1
        effective.max_q = 0
    return effective


def latest_tcode_for_spec(spec, monthly_vintages, quarterly_vintages) -> str | None:
    if spec.base_series is None:
        return None
    vintages = vintages_for_spec(spec, monthly_vintages, quarterly_vintages)
    if not vintages:
        return None
    return normalize_tcode(vintages[-1].transform_codes.get(spec.base_series))


def build_forecast_index(last_observed: pd.Timestamp, steps: int, frequency: str) -> pd.DatetimeIndex:
    if frequency == "quarterly":
        periods = pd.period_range(start=pd.Period(last_observed, freq="Q") + 1, periods=steps, freq="Q")
        return periods.asfreq("M", how="end").to_timestamp(how="start")
    periods = pd.period_range(start=pd.Period(last_observed, freq="M") + 1, periods=steps, freq="M")
    return periods.to_timestamp(how="start")


def extend_base_series(model_series: pd.Series, forecast_values: list[float], frequency: str) -> pd.Series:
    forecast_index = build_forecast_index(model_series.index.max(), len(forecast_values), frequency)
    forecast_series = pd.Series(forecast_values, index=forecast_index, name=model_series.name, dtype=float)
    return pd.concat([model_series.astype(float), forecast_series])


def save_catalog(output_dir: Path, monthly_vintages, quarterly_vintages) -> None:
    catalog = pd.DataFrame(variable_catalog_records()).sort_values("variable")
    catalog["model_base_series"] = catalog["base_series"]
    catalog["modeling_policy"] = "fit_native_fred_base_series_only"
    catalog["fred_tcode"] = [latest_tcode_for_spec(VARIABLE_SPECS[variable], monthly_vintages, quarterly_vintages) for variable in catalog["variable"]]
    catalog["fred_tcode_description"] = catalog["fred_tcode"].map(TCODE_LABELS)
    catalog.to_csv(output_dir / "variable_catalog.csv", index=False)


def run_backtest(
    vintages,
    truth_frame: pd.DataFrame,
    variables: list[str],
    horizons: list[int],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    model_series_cache: dict[tuple[str, str, str], pd.Series] = {}
    fitted_cache: dict[tuple[tuple[str, str, str], int, str, int, int, int, str, str], tuple[object, pd.Series]] = {}
    truth_cache: dict[tuple[str, str], pd.Series] = {}

    for variable in variables:
        spec = VARIABLE_SPECS[variable]
        effective_args = effective_model_args_for_spec(spec, args)
        if not spec.supported or spec.frequency != "monthly":
            skipped.append({"mode": "backtest", "variable": variable, "reason": spec.note or "Unsupported variable"})
            continue

        truth_key = (variable, spec.dataset_family)
        if truth_key not in truth_cache:
            try:
                truth_cache[truth_key] = build_target_series(truth_frame, spec).dropna()
            except KeyError as exc:
                skipped.append({"mode": "backtest", "variable": variable, "reason": str(exc)})
                continue
        truth_series = truth_cache[truth_key]

        for vintage in vintages:
            series_key = (vintage.path.name, spec.base_series or "", spec.dataset_family)
            if series_key not in model_series_cache:
                try:
                    model_series_cache[series_key] = build_model_series(vintage.frame, spec).dropna()
                except KeyError:
                    continue
            model_series = model_series_cache[series_key]

            if len(model_series) < args.min_train_size:
                continue

            fred_tcode = normalize_tcode(vintage.transform_codes.get(spec.base_series))
            transform_name = transform_from_tcode(fred_tcode)
            if transform_name is None:
                skipped.append(
                    {
                        "mode": "backtest",
                        "variable": variable,
                        "vintage": vintage.path.name,
                        "reason": f"No transform mapping for tcode={fred_tcode!r}.",
                    }
                )
                continue

            last_observed = model_series.index.max()
            for horizon in horizons:
                target_date = last_observed + pd.DateOffset(months=horizon)
                if target_date not in truth_series.index or pd.isna(truth_series.get(target_date)):
                    continue

                model_key = (
                    series_key,
                    horizon,
                    effective_args.model,
                    effective_args.min_p,
                    effective_args.max_p,
                    effective_args.max_d,
                    effective_args.max_q,
                    effective_args.ic,
                    transform_name,
                )
                try:
                    if model_key not in fitted_cache:
                        forecast_result = run_model(model_series, effective_args.model, transform_name, horizon, effective_args)
                        extended_base = extend_base_series(model_series, forecast_result.forecast, spec.frequency)
                        fitted_cache[model_key] = (forecast_result, extended_base)
                    forecast_result, extended_base = fitted_cache[model_key]
                    forecast_value = float(build_target_from_base_series(extended_base, spec).loc[target_date])
                except Exception as exc:
                    skipped.append(
                        {
                            "mode": "backtest",
                            "variable": variable,
                            "vintage": vintage.path.name,
                            "reason": str(exc),
                        }
                    )
                    continue

                actual_value = float(truth_series.loc[target_date])
                rows.append(
                    {
                        "mode": "backtest",
                        "variable": variable,
                        "source_kind": spec.source_kind,
                        "base_series": spec.base_series,
                        "model_base_series": spec.base_series,
                        "native_fred_tcode": fred_tcode,
                        "native_fred_tcode_description": TCODE_LABELS.get(fred_tcode),
                        "vintage_period": str(vintage.period),
                        "vintage_file": vintage.path.name,
                        "last_observed_date": last_observed.date().isoformat(),
                        "target_date": target_date.date().isoformat(),
                        "horizon_months": horizon,
                        "train_size": len(model_series),
                        "transform_mode": "fred_tcode",
                        "model_transform": transform_name,
                        "adf_stat": None,
                        "adf_stationary_5pct": None,
                        "adf_used_lags": None,
                        "model": forecast_result.model_name,
                        "selected_order": forecast_result.order_label,
                        "selected_p": forecast_result.selected_p,
                        "selected_d": forecast_result.selected_d,
                        "selected_q": forecast_result.selected_q,
                        "selected_ic": forecast_result.criterion_name,
                        "selected_ic_value": forecast_result.criterion_value,
                        "ar_order_cap_hit": forecast_result.selected_p == effective_args.max_p,
                        "ma_order_cap_hit": forecast_result.selected_q == effective_args.max_q,
                        "forecast": forecast_value,
                        "actual": actual_value,
                        "error": forecast_value - actual_value,
                        "abs_error": abs(forecast_value - actual_value),
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(skipped)


def choose_vintage_for_request_period(vintages, request_period: pd.Period):
    eligible = [vintage for vintage in vintages if vintage.period <= request_period]
    if not eligible:
        return None
    return eligible[-1]


def load_target_requests(targets_csv: Path) -> pd.DataFrame:
    requests = pd.read_csv(targets_csv)
    columns = [column for column in ["category", "variable", "target_month"] if column in requests.columns]
    if "variable" not in columns or "target_month" not in columns:
        raise ValueError("Target CSV must contain variable and target_month columns.")
    return requests[columns].drop_duplicates().sort_values(columns)


def run_target_month_forecasts(
    monthly_vintages,
    quarterly_vintages,
    monthly_truth_frame: pd.DataFrame,
    quarterly_truth_frame: pd.DataFrame,
    target_requests: pd.DataFrame,
    variables: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    model_series_cache: dict[tuple[str, str, str], pd.Series] = {}
    fitted_cache: dict[tuple[tuple[str, str, str], int, str, int, int, int, str, str], tuple[object, pd.Series]] = {}
    truth_cache: dict[tuple[str, str], pd.Series] = {}

    request_subset = target_requests[target_requests["variable"].isin(variables)].copy()
    for request in request_subset.itertuples(index=False):
        variable = request.variable
        category = getattr(request, "category", "")
        target_month = str(request.target_month)
        spec = VARIABLE_SPECS[variable]
        effective_args = effective_model_args_for_spec(spec, args)

        if not spec.supported:
            skipped.append(
                {
                    "mode": "target-file",
                    "variable": variable,
                    "target_month": target_month,
                    "reason": spec.note or "Unsupported variable",
                }
            )
            continue

        target = target_context(spec, target_month)
        vintages = vintages_for_spec(spec, monthly_vintages, quarterly_vintages)
        truth_frame = truth_frame_for_spec(spec, monthly_truth_frame, quarterly_truth_frame)
        if not vintages or truth_frame.empty:
            skipped.append(
                {
                    "mode": "target-file",
                    "variable": variable,
                    "target_month": target_month,
                    "reason": f"No {spec.dataset_family} vintages are available for this variable.",
                }
            )
            continue

        vintage = choose_vintage_for_request_period(vintages, target.request_period)
        if vintage is None:
            skipped.append(
                {
                    "mode": "target-file",
                    "variable": variable,
                    "target_month": target_month,
                    "reason": "No vintage is available at or before the requested target month.",
                }
            )
            continue

        series_key = (vintage.path.name, spec.base_series or "", spec.dataset_family)
        if series_key not in model_series_cache:
            try:
                model_series_cache[series_key] = build_model_series(vintage.frame, spec).dropna()
            except KeyError as exc:
                skipped.append(
                    {
                        "mode": "target-file",
                        "variable": variable,
                        "target_month": target_month,
                        "reason": str(exc),
                    }
                )
                continue
        model_series = model_series_cache[series_key]

        truth_key = (variable, spec.dataset_family)
        if truth_key not in truth_cache:
            try:
                truth_cache[truth_key] = build_target_series(truth_frame, spec).dropna()
            except KeyError as exc:
                skipped.append(
                    {
                        "mode": "target-file",
                        "variable": variable,
                        "target_month": target_month,
                        "reason": str(exc),
                    }
                )
                continue
        truth_series = truth_cache[truth_key]

        if len(model_series) < args.min_train_size:
            skipped.append(
                {
                    "mode": "target-file",
                    "variable": variable,
                    "target_month": target_month,
                    "reason": f"Only {len(model_series)} observations are available; min_train_size is {args.min_train_size}.",
                }
            )
            continue

        last_observed = model_series.index.max()
        horizon = quarter_diff(last_observed, target.target_date) if spec.frequency == "quarterly" else month_diff(last_observed, target.target_date)
        if horizon < 1:
            skipped.append(
                {
                    "mode": "target-file",
                    "variable": variable,
                    "target_month": target_month,
                    "reason": "Target date is already observed in the selected vintage.",
                }
            )
            continue

        fred_tcode = normalize_tcode(vintage.transform_codes.get(spec.base_series))
        transform_name = transform_from_tcode(fred_tcode)
        if transform_name is None:
            skipped.append(
                {
                    "mode": "target-file",
                    "variable": variable,
                    "target_month": target_month,
                    "reason": f"No transform mapping for tcode={fred_tcode!r}.",
                }
            )
            continue

        model_key = (
            series_key,
            horizon,
            effective_args.model,
            effective_args.min_p,
            effective_args.max_p,
            effective_args.max_d,
            effective_args.max_q,
            effective_args.ic,
            transform_name,
        )
        if model_key not in fitted_cache:
            try:
                forecast_result = run_model(model_series, effective_args.model, transform_name, horizon, effective_args)
                extended_base = extend_base_series(model_series, forecast_result.forecast, spec.frequency)
                fitted_cache[model_key] = (forecast_result, extended_base)
            except Exception as exc:
                skipped.append(
                    {
                        "mode": "target-file",
                        "variable": variable,
                        "target_month": target_month,
                        "reason": str(exc),
                    }
                )
                continue
        forecast_result, extended_base = fitted_cache[model_key]

        derived_forecast_series = build_target_from_base_series(extended_base, spec).dropna()
        if target.target_date not in derived_forecast_series.index:
            skipped.append(
                {
                    "mode": "target-file",
                    "variable": variable,
                    "target_month": target_month,
                    "reason": "Derived target value is not available at the requested target date.",
                }
            )
            continue

        actual_value = truth_series.get(target.target_date)
        forecast_value = float(derived_forecast_series.loc[target.target_date])
        rows.append(
            {
                "mode": "target-file",
                "category": category,
                "variable": variable,
                "source_kind": spec.source_kind,
                "base_series": spec.base_series,
                "model_base_series": spec.base_series,
                "native_fred_tcode": fred_tcode,
                "native_fred_tcode_description": TCODE_LABELS.get(fred_tcode),
                "dataset_family": spec.dataset_family,
                "target_frequency": spec.frequency,
                "target_month": target_month,
                "target_period": target.target_period_label,
                "vintage_period": str(vintage.period),
                "vintage_file": vintage.path.name,
                "last_observed_date": last_observed.date().isoformat(),
                "target_date": target.target_date.date().isoformat(),
                "horizon_periods": horizon,
                "horizon_unit": target.horizon_unit,
                "horizon_months": horizon if spec.frequency == "monthly" else None,
                "train_size": len(model_series),
                "transform_mode": "fred_tcode",
                "model_transform": transform_name,
                "adf_stat": None,
                "adf_stationary_5pct": None,
                "adf_used_lags": None,
                "model": forecast_result.model_name,
                "selected_order": forecast_result.order_label,
                "selected_p": forecast_result.selected_p,
                "selected_d": forecast_result.selected_d,
                "selected_q": forecast_result.selected_q,
                "selected_ic": forecast_result.criterion_name,
                "selected_ic_value": forecast_result.criterion_value,
                "ar_order_cap_hit": forecast_result.selected_p == effective_args.max_p,
                "ma_order_cap_hit": forecast_result.selected_q == effective_args.max_q,
                "forecast": forecast_value,
                "actual": None if pd.isna(actual_value) else float(actual_value),
                "error": None if pd.isna(actual_value) else forecast_value - float(actual_value),
                "abs_error": None if pd.isna(actual_value) else abs(forecast_value - float(actual_value)),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(skipped)


def metric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "mode",
        "variable",
        "horizon_periods",
        "horizon_unit",
        "horizon_months",
        "model",
        "count",
        "rmse",
        "mae",
        "smape",
        "mape",
    ]
    if frame.empty:
        return pd.DataFrame(columns=metric_columns)

    metric_rows: list[dict[str, object]] = []
    with_actuals = frame.dropna(subset=["actual", "forecast"]).copy()
    if with_actuals.empty:
        return pd.DataFrame(columns=metric_columns)

    group_columns = [column for column in ["mode", "variable", "horizon_periods", "horizon_unit", "horizon_months", "model"] if column in with_actuals.columns]
    for keys, group in with_actuals.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_columns, keys, strict=False))
        actual = group["actual"].astype(float)
        forecast = group["forecast"].astype(float)
        metric_rows.append(
            {
                "mode": key_map.get("mode"),
                "variable": key_map.get("variable"),
                "horizon_periods": key_map.get("horizon_periods"),
                "horizon_unit": key_map.get("horizon_unit"),
                "horizon_months": key_map.get("horizon_months"),
                "model": key_map.get("model"),
                "count": len(group),
                "rmse": rmse(actual, forecast),
                "mae": mae(actual, forecast),
                "smape": smape(actual, forecast),
                "mape": mape(actual, forecast),
            }
        )
    return pd.DataFrame(metric_rows, columns=metric_columns).sort_values(["mode", "variable", "horizon_periods", "horizon_months"])


def plot_backtests(backtest_frame: pd.DataFrame, output_dir: Path) -> None:
    if backtest_frame.empty:
        return
    plot_dir = output_dir / "plots" / "backtest"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for (variable, horizon), group in backtest_frame.groupby(["variable", "horizon_months"]):
        ordered = group.sort_values("target_date")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(pd.to_datetime(ordered["target_date"]), ordered["actual"], label="Actual", linewidth=2)
        ax.plot(pd.to_datetime(ordered["target_date"]), ordered["forecast"], label="Forecast", linewidth=1.5)
        ax.set_title(f"{variable} | {horizon}-month horizon")
        ax.set_xlabel("Target date")
        ax.set_ylabel(variable)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{variable}_h{horizon}.png", dpi=150)
        plt.close(fig)


def plot_target_months(
    target_frame: pd.DataFrame,
    monthly_truth_frame: pd.DataFrame,
    quarterly_truth_frame: pd.DataFrame,
    output_dir: Path,
) -> None:
    if target_frame.empty:
        return
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for variable, group in target_frame.groupby("variable"):
        spec = VARIABLE_SPECS[variable]
        try:
            truth_frame = truth_frame_for_spec(spec, monthly_truth_frame, quarterly_truth_frame)
            history_window = 24 if spec.frequency == "monthly" else 12
            truth_series = build_target_series(truth_frame, spec).dropna().tail(history_window)
        except KeyError:
            continue

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(truth_series.index, truth_series.values, label="Realized history", linewidth=2)
        ax.scatter(pd.to_datetime(group["target_date"]), group["forecast"], label="Forecast", s=40)

        actual_subset = group.dropna(subset=["actual"])
        if not actual_subset.empty:
            ax.scatter(pd.to_datetime(actual_subset["target_date"]), actual_subset["actual"], label="Actual", s=40)

        ax.set_title(f"{variable} | requested {group['target_month'].iloc[0]}")
        ax.set_xlabel("Date")
        ax.set_ylabel(variable)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{variable}.png", dpi=150)
        plt.close(fig)


def write_per_target_month_outputs(
    target_frame: pd.DataFrame,
    target_skips: pd.DataFrame,
    monthly_truth_frame: pd.DataFrame,
    quarterly_truth_frame: pd.DataFrame,
    output_dir: Path,
) -> None:
    target_root = output_dir / "target_months"
    target_root.mkdir(parents=True, exist_ok=True)

    month_values: set[str] = set()
    if not target_frame.empty and "target_month" in target_frame.columns:
        month_values.update(target_frame["target_month"].astype(str))
    if not target_skips.empty and "target_month" in target_skips.columns:
        month_values.update(target_skips["target_month"].dropna().astype(str))

    for target_month in sorted(month_values):
        month_dir = target_root / target_month
        month_dir.mkdir(parents=True, exist_ok=True)

        month_frame = target_frame[target_frame["target_month"].astype(str) == target_month].copy() if not target_frame.empty else pd.DataFrame()
        month_skips = (
            target_skips[target_skips["target_month"].astype(str) == target_month].copy()
            if not target_skips.empty and "target_month" in target_skips.columns
            else pd.DataFrame()
        )

        month_frame.to_csv(month_dir / "target_month_forecasts.csv", index=False)
        metric_summary(month_frame).to_csv(month_dir / "target_month_metrics.csv", index=False)
        month_skips.to_csv(month_dir / "skipped_items.csv", index=False)
        plot_target_months(month_frame, monthly_truth_frame, quarterly_truth_frame, month_dir)


def main() -> None:
    args = parse_args()
    args.target_months_filter = resolve_target_months(args.target_months)

    variables = resolve_variables(args.variables)
    horizons = resolve_horizons(args.horizons)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    monthly_vintages = load_all_vintages(args.data_dir)
    quarterly_vintages = load_all_vintages(args.qd_data_dir)

    external_panel, external_transform_codes = load_external_panel(args.external_data_dir)
    monthly_vintages = augment_monthly_vintages(monthly_vintages, external_panel, external_transform_codes)

    save_catalog(args.output_dir, monthly_vintages, quarterly_vintages)
    monthly_truth_frame = coalesced_truth_frame(monthly_vintages)
    quarterly_truth_frame = coalesced_truth_frame(quarterly_vintages)
    monthly_truth_frame = augment_truth_frame(monthly_truth_frame, external_panel)

    all_frames: list[pd.DataFrame] = []
    all_skips: list[pd.DataFrame] = []

    if args.mode in {"backtest", "both"}:
        backtest_frame, backtest_skips = run_backtest(monthly_vintages, monthly_truth_frame, variables, horizons, args)
        backtest_frame.to_csv(args.output_dir / "backtest_forecasts.csv", index=False)
        metric_summary(backtest_frame).to_csv(args.output_dir / "backtest_metrics.csv", index=False)
        plot_backtests(backtest_frame, args.output_dir)
        all_frames.append(backtest_frame)
        all_skips.append(backtest_skips)

    if args.mode in {"target-file", "both"}:
        target_requests = load_target_requests(args.targets_csv)
        if args.target_months_filter is not None:
            target_requests = target_requests[target_requests["target_month"].astype(str).isin(args.target_months_filter)]
        target_frame, target_skips = run_target_month_forecasts(
            monthly_vintages,
            quarterly_vintages,
            monthly_truth_frame,
            quarterly_truth_frame,
            target_requests,
            variables,
            args,
        )
        target_frame.to_csv(args.output_dir / "target_month_forecasts.csv", index=False)
        metric_summary(target_frame).to_csv(args.output_dir / "target_month_metrics.csv", index=False)
        write_per_target_month_outputs(
            target_frame,
            target_skips,
            monthly_truth_frame,
            quarterly_truth_frame,
            args.output_dir,
        )
        all_frames.append(target_frame)
        all_skips.append(target_skips)

    non_empty_skips = [frame for frame in all_skips if not frame.empty]
    skipped_frame = pd.concat(non_empty_skips, ignore_index=True) if non_empty_skips else pd.DataFrame()
    skipped_frame.to_csv(args.output_dir / "skipped_items.csv", index=False)

    non_empty_frames = [frame for frame in all_frames if not frame.empty]
    combined = pd.concat(non_empty_frames, ignore_index=True) if non_empty_frames else pd.DataFrame()
    if not combined.empty:
        metric_summary(combined).to_csv(args.output_dir / "all_metrics.csv", index=False)

    supported = pd.DataFrame({"variable": supported_variables()}).sort_values("variable")
    supported.to_csv(args.output_dir / "supported_variables.csv", index=False)


if __name__ == "__main__":
    main()
