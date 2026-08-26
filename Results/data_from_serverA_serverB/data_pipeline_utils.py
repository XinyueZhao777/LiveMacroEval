from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable


GROUPED_CSV_PATTERN = re.compile(r"^(\d{4}-\d{2})_(.+)_serverA\.csv$")


def resolve_latest_dated_dir(root: Path, base_name: str) -> Path:
    """Resolve the newest '<base_name>_<date>' directory, falling back to '<base_name>'."""
    pattern = re.compile(rf"^{re.escape(base_name)}(?:_(\d+))?$")
    candidates: list[tuple[tuple[int, int, str], Path]] = []

    for candidate in root.iterdir():
        if not candidate.is_dir():
            continue
        match = pattern.match(candidate.name)
        if not match:
            continue

        suffix = match.group(1)
        has_suffix = 1 if suffix is not None else 0
        suffix_value = int(suffix) if suffix is not None else -1
        candidates.append(((has_suffix, suffix_value, candidate.name), candidate))

    if not candidates:
        raise FileNotFoundError(
            f"No directory matching '{base_name}' or '{base_name}_<date>' was found in {root}"
        )

    return max(candidates, key=lambda item: item[0])[1]


def discover_grouped_csv_patterns(
    folder: Path,
    expected_categories: Iterable[str],
) -> dict[tuple[str, str], str]:
    """Discover '<YYYY-MM>_<category>_serverA.csv' files for supported categories."""
    if not folder.exists():
        return {}

    allowed_categories = set(expected_categories)
    file_patterns: dict[tuple[str, str], str] = {}

    for csv_file in sorted(folder.glob("*.csv")):
        match = GROUPED_CSV_PATTERN.match(csv_file.name)
        if not match:
            continue

        target_month, category = match.groups()
        if category not in allowed_categories:
            continue

        file_patterns[(target_month, category)] = csv_file.name

    return file_patterns
