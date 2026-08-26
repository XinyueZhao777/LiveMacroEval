from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd


@dataclass(frozen=True)
class VintageData:
    period: pd.Period
    path: Path
    frame: pd.DataFrame
    transform_codes: dict[str, str]


def parse_vintage_period(path: Path) -> pd.Period:
    name = path.name
    patterns = [
        re.compile(r"^(?P<year>\d{4})-(?P<month>\d{1,2})(?:-(?:MD|QD))?\.csv$"),
        re.compile(r"^FRED-(?:M|Q)D_(?P<year>\d{4})m(?P<month>\d{1,2})\.csv$"),
    ]
    for pattern in patterns:
        match = pattern.match(name)
        if match:
            year = int(match.group("year"))
            month = int(match.group("month"))
            return pd.Period(f"{year:04d}-{month:02d}", freq="M")
    raise ValueError(f"Cannot parse vintage period from {name}")


def list_vintage_files(data_dir: Path) -> list[Path]:
    csv_files = sorted(p for p in data_dir.glob("*.csv") if p.is_file())
    return sorted(csv_files, key=lambda path: (parse_vintage_period(path), path.name))


def _finalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["sasdate"] = pd.to_datetime(frame["sasdate"], format="%m/%d/%Y", errors="coerce")
    frame = frame.dropna(subset=["sasdate"]).rename(columns={"sasdate": "date"}).set_index("date").sort_index()

    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    inferred_freq = pd.infer_freq(frame.index)
    if inferred_freq is not None:
        frame = frame.asfreq(inferred_freq)

    return frame


def _stringify_transform_codes(row: dict[str, object]) -> dict[str, str]:
    return {
        str(column): str(value)
        for column, value in row.items()
        if str(column) != "sasdate" and pd.notna(value)
    }


def load_vintage(path: Path) -> VintageData:
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"{path} is empty")

    first_cell = str(raw.iloc[0, 0]).strip() if not raw.empty else ""
    if first_cell == "Transform:":
        transform_codes = _stringify_transform_codes(raw.iloc[0].to_dict())
        frame = _finalize_frame(raw.iloc[1:])
    elif first_cell == "factors":
        second_cell = str(raw.iloc[1, 0]).strip() if len(raw.index) > 1 else ""
        if second_cell != "transform":
            raise ValueError(f"{path} does not have the expected FRED-QD transform row")
        transform_codes = _stringify_transform_codes(raw.iloc[1].to_dict())
        frame = _finalize_frame(raw.iloc[2:])
    else:
        raise ValueError(f"{path} does not have a recognized FRED-MD or FRED-QD layout")

    return VintageData(
        period=parse_vintage_period(path),
        path=path,
        frame=frame,
        transform_codes=transform_codes,
    )


def load_all_vintages(data_dir: Path) -> list[VintageData]:
    return [load_vintage(path) for path in list_vintage_files(data_dir)]


def coalesced_truth_frame(vintages: list[VintageData]) -> pd.DataFrame:
    stacked_frames = []
    for order, vintage in enumerate(vintages):
        frame = vintage.frame.copy().reset_index().rename(columns={"date": "date"})
        frame["_vintage_order"] = order
        stacked_frames.append(frame)

    if not stacked_frames:
        return pd.DataFrame()

    combined = pd.concat(stacked_frames, ignore_index=True, sort=False)
    truth = combined.sort_values(["date", "_vintage_order"]).groupby("date", sort=True).last()
    truth = truth.drop(columns=["_vintage_order"], errors="ignore")
    truth.index = pd.to_datetime(truth.index)
    return truth.sort_index()
