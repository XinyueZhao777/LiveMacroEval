import csv
import fcntl
import io
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from config import DATA_DIR, DATA_LIGHT_DIR, get_logger

logger = get_logger(__name__)

CSV_COLS = [
    "timestamp_local",
    "target_month",
    "release_month",
    "job_id",
    "model",
    "variable",
    "value",
    "parsed_ok",
    "parsed_notes",
    "raw_model_output",
    "citations",
]

def _get_output_path(job, model_name):
    target_period = job.get("target_period")
    group = job.get("variable_group")

    base = DATA_DIR / f"model_{model_name}"
    base.mkdir(parents=True, exist_ok=True)

    filename = f"{target_period}_{group}.csv"
    return base / filename

def _ensure_csv_headers(path, columns):
    if not path.exists():
        return

    df_head = pd.read_csv(path, nrows=0)
    missing = [c for c in columns if c not in df_head.columns]
    if not missing:
        return

    df_old = pd.read_csv(path)
    for c in missing:
        df_old[c] = None
    df_old = df_old[columns]

    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as tmp_fp:
            df_old.to_csv(tmp_fp, index=False)
            tmp_fp.flush()
            os.fsync(tmp_fp.fileno())
        os.replace(tmp_name, path)
        _fsync_dir(path.parent)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise

def _fsync_dir(directory: Path):
    dir_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)

def _render_csv_bytes(rows: List[Dict[str, Any]], columns: List[str], include_header: bool) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    if include_header:
        writer.writeheader()
    for row in rows:
        writer.writerow({col: row.get(col) for col in columns})
    return buffer.getvalue().encode("utf-8")

def _append_csv_rows_locked(path: Path, columns: List[str], rows: List[Dict[str, Any]]):
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a", encoding="utf-8") as lock_fp:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)

        if path.exists():
            _ensure_csv_headers(path, columns)
            existing_bytes = path.read_bytes()
            include_header = False
            if existing_bytes and not existing_bytes.endswith(b"\n"):
                existing_bytes += b"\n"
        else:
            existing_bytes = b""
            include_header = True

        append_bytes = _render_csv_bytes(rows, columns, include_header=include_header)

        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as tmp_fp:
                if existing_bytes:
                    tmp_fp.write(existing_bytes)
                tmp_fp.write(append_bytes)
                tmp_fp.flush()
                os.fsync(tmp_fp.fileno())
            os.replace(tmp_name, path)
            _fsync_dir(path.parent)
        except Exception:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise

def append_predictions(job, model_name, preds, meta):
    out_path = _get_output_path(job, model_name)

    rows = []
    for var_key, value in preds.items():
        rows.append({
                "timestamp_local": meta["timestamp_local"],
                "target_month": meta["target_month"],
                "release_month": meta["release_month"],
                "job_id": meta["job_id"],
                "model": meta["model"],
                "variable": var_key,
                "value": value,
                "parsed_ok": meta["parsed_ok"],
                "parsed_notes": meta["parsed_notes"],
                "raw_model_output": meta["raw_model_output"],
                "citations": meta["citations"],
            }
        )
    _append_csv_rows_locked(out_path, CSV_COLS, rows)

    logger.info("Appended %d rows to %s for job=%s model=%s", len(rows),
        out_path, meta["job_id"], meta["model"],
    )

# Columns to keep in the light version (excludes raw_model_output and citations)
LIGHT_COLS = [
    "timestamp_local",
    "target_month",
    "release_month",
    "job_id",
    "model",
    "variable",
    "value",
    "parsed_ok",
    "parsed_notes",
]

def sync_data_light():
    """
    Process all CSVs in data/ and create light versions in data_light/.
    Light versions contain only LIGHT_COLS (no raw_model_output or citations).
    """
    for model_dir in DATA_DIR.iterdir():
        if not model_dir.is_dir():
            continue

        # Create corresponding directory in data_light
        light_model_dir = DATA_LIGHT_DIR / model_dir.name
        light_model_dir.mkdir(parents=True, exist_ok=True)

        for csv_file in model_dir.glob("*.csv"):
            light_path = light_model_dir / csv_file.name
            try:
                df = pd.read_csv(csv_file)
                # Keep only the light columns (that exist in the dataframe)
                cols_to_keep = [c for c in LIGHT_COLS if c in df.columns]
                df_light = df[cols_to_keep]
                df_light.to_csv(light_path, index=False)
                logger.info("Synced light CSV: %s -> %s", csv_file, light_path)
            except Exception as e:
                logger.error("Failed to sync light CSV %s: %s", csv_file, e)
