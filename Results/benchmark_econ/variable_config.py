from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class VariableSpec:
    variable: str
    base_series: str | None
    target_kind: str
    supported: bool
    source_kind: str
    dataset_family: str = "fred_md"
    note: str = ""
    scale: float = 1.0
    frequency: str = "monthly"
    prefer_log_transform: bool = False

    @property
    def prefers_log_candidates(self) -> bool:
        return self.prefer_log_transform


VARIABLE_SPECS: dict[str, VariableSpec] = {
    "building_permits_level": VariableSpec("building_permits_level", "PERMIT", "level", True, "exact", prefer_log_transform=True),
    "building_permits_mom": VariableSpec("building_permits_mom", "PERMIT", "pct_change_1", True, "exact"),
    "building_permits_yoy": VariableSpec("building_permits_yoy", "PERMIT", "pct_change_12", True, "exact"),
    "cpi_index": VariableSpec("cpi_index", "CPIAUCSL", "level", True, "exact", prefer_log_transform=True),
    "cpi_mom": VariableSpec("cpi_mom", "CPIAUCSL", "pct_change_1", True, "exact"),
    "cpi_yoy": VariableSpec("cpi_yoy", "CPIAUCSL", "pct_change_12", True, "exact"),
    "durable_goods_level": VariableSpec(
        "durable_goods_level",
        "AMDMNOx",
        "level",
        True,
        "exact",
        scale=1.0 / 1000.0,
        prefer_log_transform=True,
    ),
    "durable_goods_mom": VariableSpec("durable_goods_mom", "AMDMNOx", "pct_change_1", True, "exact"),
    "durable_goods_yoy": VariableSpec("durable_goods_yoy", "AMDMNOx", "pct_change_12", True, "exact"),
    "existing_home_sales_level": VariableSpec(
        "existing_home_sales_level",
        "EXHOSLUSM495S",
        "level",
        True,
        "external",
        note="Existing home sales SAAR (millions), parsed from Investing.com calendar (external_data_2026_04_16).",
        prefer_log_transform=True,
    ),
    "existing_home_sales_mom": VariableSpec(
        "existing_home_sales_mom",
        "EXHOSLUSM495S",
        "pct_change_1",
        True,
        "external",
        note="Existing home sales SAAR (millions), parsed from Investing.com calendar (external_data_2026_04_16).",
    ),
    "existing_home_sales_yoy": VariableSpec(
        "existing_home_sales_yoy",
        "EXHOSLUSM495S",
        "pct_change_12",
        True,
        "external",
        note="Existing home sales SAAR (millions), parsed from Investing.com calendar (external_data_2026_04_16).",
    ),
    "housing_starts_level": VariableSpec("housing_starts_level", "HOUST", "level", True, "exact", prefer_log_transform=True),
    "housing_starts_mom": VariableSpec("housing_starts_mom", "HOUST", "pct_change_1", True, "exact"),
    "housing_starts_yoy": VariableSpec("housing_starts_yoy", "HOUST", "pct_change_12", True, "exact"),
    "industrial_production_index": VariableSpec(
        "industrial_production_index",
        "INDPRO",
        "level",
        True,
        "exact",
        prefer_log_transform=True,
    ),
    "industrial_production_mom": VariableSpec("industrial_production_mom", "INDPRO", "pct_change_1", True, "exact"),
    "industrial_production_yoy": VariableSpec("industrial_production_yoy", "INDPRO", "pct_change_12", True, "exact"),
    "ism_manufacturing_index": VariableSpec(
        "ism_manufacturing_index",
        "ISM_PMI",
        "level",
        True,
        "external",
        note="ISM Manufacturing PMI parsed from Investing.com calendar (external_data_2026_04_16).",
    ),
    "ism_manufacturing_mom": VariableSpec(
        "ism_manufacturing_mom",
        "ISM_PMI",
        "change",
        True,
        "external",
        note="ISM Manufacturing PMI parsed from Investing.com calendar (external_data_2026_04_16).",
    ),
    "ism_services_index": VariableSpec(
        "ism_services_index",
        "ISM_SVC",
        "level",
        True,
        "external",
        note="ISM Services PMI parsed from Investing.com calendar (external_data_2026_04_16).",
    ),
    "ism_services_mom": VariableSpec(
        "ism_services_mom",
        "ISM_SVC",
        "change",
        True,
        "external",
        note="ISM Services PMI parsed from Investing.com calendar (external_data_2026_04_16).",
    ),
    "new_home_sales_level": VariableSpec(
        "new_home_sales_level",
        "HSN1F",
        "level",
        True,
        "external",
        note="New home sales SAAR (thousands), parsed from Census workbook (external_data_2026_04_16).",
        prefer_log_transform=True,
    ),
    "new_home_sales_mom": VariableSpec(
        "new_home_sales_mom",
        "HSN1F",
        "pct_change_1",
        True,
        "external",
        note="New home sales SAAR (thousands), parsed from Census workbook (external_data_2026_04_16).",
    ),
    "new_home_sales_yoy": VariableSpec(
        "new_home_sales_yoy",
        "HSN1F",
        "pct_change_12",
        True,
        "external",
        note="New home sales SAAR (thousands), parsed from Census workbook (external_data_2026_04_16).",
    ),
    "nonfarm_payrolls_change": VariableSpec("nonfarm_payrolls_change", "PAYEMS", "change", True, "exact"),
    "nonfarm_payrolls_level": VariableSpec("nonfarm_payrolls_level", "PAYEMS", "level", True, "exact", prefer_log_transform=True),
    "pce_price_index_level": VariableSpec("pce_price_index_level", "PCEPI", "level", True, "exact", prefer_log_transform=True),
    "pce_price_index_mom": VariableSpec("pce_price_index_mom", "PCEPI", "pct_change_1", True, "exact"),
    "pce_price_index_yoy": VariableSpec("pce_price_index_yoy", "PCEPI", "pct_change_12", True, "exact"),
    "ppi_index": VariableSpec(
        "ppi_index",
        "PPIACO",
        "level",
        True,
        "external",
        note="Producer Price Index for All Commodities (PPIACO), exact FRED CSV pull (external_data_2026_04_16).",
        prefer_log_transform=True,
    ),
    "ppi_mom": VariableSpec(
        "ppi_mom",
        "PPIACO",
        "pct_change_1",
        True,
        "external",
        note="Producer Price Index for All Commodities (PPIACO), exact FRED CSV pull (external_data_2026_04_16).",
    ),
    "ppi_yoy": VariableSpec(
        "ppi_yoy",
        "PPIACO",
        "pct_change_12",
        True,
        "external",
        note="Producer Price Index for All Commodities (PPIACO), exact FRED CSV pull (external_data_2026_04_16).",
    ),
    "real_dpi_level": VariableSpec(
        "real_dpi_level",
        "DSPIC96",
        "level",
        True,
        "external",
        note="Real DPI (DSPIC96), exact FRED CSV pull (external_data_2026_04_16).",
        prefer_log_transform=True,
    ),
    "real_dpi_mom": VariableSpec(
        "real_dpi_mom",
        "DSPIC96",
        "pct_change_1",
        True,
        "external",
        note="Real DPI (DSPIC96), exact FRED CSV pull (external_data_2026_04_16).",
    ),
    "real_dpi_yoy": VariableSpec(
        "real_dpi_yoy",
        "DSPIC96",
        "pct_change_12",
        True,
        "external",
        note="Real DPI (DSPIC96), exact FRED CSV pull (external_data_2026_04_16).",
    ),
    "real_gdp_level": VariableSpec(
        "real_gdp_level",
        "GDPC1",
        "level",
        True,
        "exact",
        dataset_family="fred_qd",
        note="Quarterly GDP is sourced from the local FRED-QD vintages.",
        frequency="quarterly",
        prefer_log_transform=True,
    ),
    "real_gdp_qoq": VariableSpec(
        "real_gdp_qoq",
        "GDPC1",
        "pct_change_1",
        True,
        "exact",
        dataset_family="fred_qd",
        note="Quarterly GDP is sourced from the local FRED-QD vintages.",
        frequency="quarterly",
    ),
    "real_gdp_yoy": VariableSpec(
        "real_gdp_yoy",
        "GDPC1",
        "pct_change_4",
        True,
        "exact",
        dataset_family="fred_qd",
        note="Quarterly GDP is sourced from the local FRED-QD vintages.",
        frequency="quarterly",
    ),
    "real_pce_level": VariableSpec(
        "real_pce_level",
        "PCEC96",
        "level",
        True,
        "external",
        note="Real PCE (PCEC96), exact FRED CSV pull (external_data_2026_04_16).",
        prefer_log_transform=True,
    ),
    "real_pce_mom": VariableSpec(
        "real_pce_mom",
        "PCEC96",
        "pct_change_1",
        True,
        "external",
        note="Real PCE (PCEC96), exact FRED CSV pull (external_data_2026_04_16).",
    ),
    "real_pce_yoy": VariableSpec(
        "real_pce_yoy",
        "PCEC96",
        "pct_change_12",
        True,
        "external",
        note="Real PCE (PCEC96), exact FRED CSV pull (external_data_2026_04_16).",
    ),
    "retail_sales_level": VariableSpec(
        "retail_sales_level",
        "RETAILx",
        "level",
        True,
        "exact",
        scale=1.0 / 1000.0,
        prefer_log_transform=True,
    ),
    "retail_sales_mom": VariableSpec("retail_sales_mom", "RETAILx", "pct_change_1", True, "exact"),
    "retail_sales_yoy": VariableSpec("retail_sales_yoy", "RETAILx", "pct_change_12", True, "exact"),
    "unemployment_rate": VariableSpec("unemployment_rate", "UNRATE", "level", True, "exact"),
    "unemployment_rate_mom": VariableSpec("unemployment_rate_mom", "UNRATE", "change", True, "exact"),
}


def build_model_series(frame: pd.DataFrame, spec: VariableSpec) -> pd.Series:
    if not spec.supported or spec.base_series is None:
        raise KeyError(f"{spec.variable} is unsupported with the local data.")
    if spec.base_series not in frame.columns:
        raise KeyError(f"{spec.base_series} is not present in this vintage.")

    base = pd.to_numeric(frame[spec.base_series], errors="coerce")
    base.name = spec.base_series
    return base


def build_target_from_base_series(base_series: pd.Series, spec: VariableSpec) -> pd.Series:
    base = pd.to_numeric(base_series, errors="coerce")
    base.name = spec.base_series
    scaled = base * spec.scale

    if spec.target_kind == "level":
        result = scaled
    elif spec.target_kind == "change":
        result = scaled.diff()
    elif spec.target_kind == "pct_change_1":
        result = base.pct_change(periods=1, fill_method=None) * 100.0
    elif spec.target_kind == "pct_change_4":
        result = base.pct_change(periods=4, fill_method=None) * 100.0
    elif spec.target_kind == "pct_change_12":
        result = base.pct_change(periods=12, fill_method=None) * 100.0
    else:
        raise ValueError(f"Unknown target kind: {spec.target_kind}")

    result.name = spec.variable
    return result


def build_target_series(frame: pd.DataFrame, spec: VariableSpec) -> pd.Series:
    base = build_model_series(frame, spec)
    return build_target_from_base_series(base, spec)


def supported_variables() -> list[str]:
    return [name for name, spec in VARIABLE_SPECS.items() if spec.supported]


def variable_catalog_records() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variable, spec in VARIABLE_SPECS.items():
        rows.append(
            {
                "variable": variable,
                "supported": spec.supported,
                "base_series": spec.base_series,
                "target_kind": spec.target_kind,
                "frequency": spec.frequency,
                "source_kind": spec.source_kind,
                "dataset_family": spec.dataset_family,
                "scale": spec.scale,
                "note": spec.note,
            }
        )
    return rows
