from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fred_loader import VintageData


@dataclass(frozen=True)
class ExternalSeriesSpec:
    base_series: str
    csv_filename: str
    value_column: str
    tcode: str
    description: str


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EXTERNAL_DIR = SCRIPT_DIR / "external_data_2026_04_16" / "processed"


EXTERNAL_SERIES: tuple[ExternalSeriesSpec, ...] = (
    ExternalSeriesSpec(
        base_series="EXHOSLUSM495S",
        csv_filename="existing_home_sales_investing_calendar_history.csv",
        value_column="existing_home_sales_millions_saar",
        tcode="4",
        description="Existing home sales SAAR (millions), parsed from Investing.com calendar.",
    ),
    ExternalSeriesSpec(
        base_series="HSN1F",
        csv_filename="new_home_sales_census_monthly_saar.csv",
        value_column="new_home_sales_saar_thousands",
        tcode="4",
        description="New home sales SAAR (thousands), parsed from Census workbook.",
    ),
    ExternalSeriesSpec(
        base_series="ISM_PMI",
        csv_filename="ism_manufacturing_investing_calendar_history.csv",
        value_column="ism_manufacturing_index",
        tcode="4",
        description="ISM Manufacturing PMI (index), parsed from Investing.com calendar.",
    ),
    ExternalSeriesSpec(
        base_series="ISM_SVC",
        csv_filename="ism_services_investing_calendar_history.csv",
        value_column="ism_services_index",
        tcode="4",
        description="ISM Services PMI (index), parsed from Investing.com calendar.",
    ),
    ExternalSeriesSpec(
        base_series="DSPIC96",
        csv_filename="real_dpi_fred_exact.csv",
        value_column="DSPIC96",
        tcode="5",
        description="Real Disposable Personal Income (DSPIC96), exact FRED CSV pull.",
    ),
    ExternalSeriesSpec(
        base_series="PCEC96",
        csv_filename="real_pce_fred_exact.csv",
        value_column="PCEC96",
        tcode="5",
        description="Real Personal Consumption Expenditures (PCEC96), exact FRED CSV pull.",
    ),
    ExternalSeriesSpec(
        base_series="PPIACO",
        csv_filename="PPIACO.csv",
        value_column="PPIACO",
        tcode="6",
        description="Producer Price Index for All Commodities (PPIACO), exact FRED CSV pull.",
    ),
)


def load_external_panel(external_dir: Path = DEFAULT_EXTERNAL_DIR) -> tuple[pd.DataFrame, dict[str, str]]:
    series_frames: list[pd.Series] = []
    transform_codes: dict[str, str] = {}

    for spec in EXTERNAL_SERIES:
        path = external_dir / spec.csv_filename
        raw = pd.read_csv(path)
        if "observation_date" not in raw.columns:
            raise ValueError(f"{path} is missing the observation_date column")
        if spec.value_column not in raw.columns:
            raise ValueError(f"{path} is missing the value column {spec.value_column!r}")

        dates = pd.to_datetime(raw["observation_date"], errors="coerce")
        values = pd.to_numeric(raw[spec.value_column], errors="coerce")
        series = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(dates), name=spec.base_series)
        series = series[~series.index.isna()].sort_index()
        series = series[~series.index.duplicated(keep="last")]
        series_frames.append(series)
        transform_codes[spec.base_series] = spec.tcode

    if not series_frames:
        return pd.DataFrame(), {}

    panel = pd.concat(series_frames, axis=1).sort_index()
    panel.index.name = "date"
    return panel, transform_codes


def _truncate_to_period(panel: pd.DataFrame, period: pd.Period) -> pd.DataFrame:
    cutoff = period.to_timestamp(how="end")
    return panel.loc[panel.index <= cutoff]


def augment_monthly_vintages(
    monthly_vintages: list[VintageData],
    external_panel: pd.DataFrame,
    external_transform_codes: dict[str, str],
) -> list[VintageData]:
    if external_panel.empty or not monthly_vintages:
        return monthly_vintages

    augmented: list[VintageData] = []
    external_columns = list(external_panel.columns)

    for vintage in monthly_vintages:
        truncated = _truncate_to_period(external_panel, vintage.period)
        frame = vintage.frame.copy()

        if not truncated.empty:
            reindexed = truncated.reindex(frame.index)
            extra_only = [column for column in external_columns if column not in frame.columns]
            for column in external_columns:
                series = reindexed[column]
                if column in frame.columns:
                    base = pd.to_numeric(frame[column], errors="coerce")
                    frame[column] = base.combine_first(series)
                else:
                    frame[column] = series

            for column in extra_only:
                # ensure dtype is float
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

        merged_codes = dict(vintage.transform_codes)
        for column in external_columns:
            if column not in merged_codes:
                merged_codes[column] = external_transform_codes.get(column, merged_codes.get(column, ""))

        augmented.append(
            VintageData(
                period=vintage.period,
                path=vintage.path,
                frame=frame,
                transform_codes=merged_codes,
            )
        )

    return augmented


def augment_truth_frame(
    truth_frame: pd.DataFrame,
    external_panel: pd.DataFrame,
) -> pd.DataFrame:
    if external_panel.empty:
        return truth_frame
    if truth_frame.empty:
        return external_panel.copy()

    combined_index = truth_frame.index.union(external_panel.index)
    base = truth_frame.reindex(combined_index)
    ext = external_panel.reindex(combined_index)

    for column in external_panel.columns:
        if column in base.columns:
            base[column] = base[column].combine_first(ext[column])
        else:
            base[column] = ext[column]

    return base.sort_index()
