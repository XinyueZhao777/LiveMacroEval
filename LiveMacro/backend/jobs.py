import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict

from config import JOBS_PATH, LOCAL_TZ, get_logger

logger = get_logger(__name__)

def _match_runner(job_runner, runner):
    if runner is None:
        return True
    jr = job_runner or "any"
    return (jr == "any") or (jr == runner)

def load_jobs(runner):
    if not JOBS_PATH.exists():
        logger.info("jobs.json not found, creating empty list")
        return []
    with JOBS_PATH.open("r", encoding="utf-8") as f:
        try:
            jobs = json.load(f)
        except json.JSONDecodeError:
            logger.error("jobs.json is invalid JSON, treating as empty")
            return []
    if not isinstance(jobs, list):
        logger.error("jobs.json must contain a list, treating as empty")
        return []
    if runner is None:
        return jobs
    filtered = [j for j in jobs if _match_runner(j.get("runner"), runner)]
    logger.info("Loaded %d jobs (filtered by runner=%s)", len(filtered), runner)
    return filtered

def save_jobs(updated_jobs, runner):
    current = []
    if JOBS_PATH.exists():
        with JOBS_PATH.open("r", encoding="utf-8") as f:
            try:
                current = json.load(f)
            except json.JSONDecodeError:
                current = []
    if not isinstance(current, list):
        current = []

    cur_by_id = {j.get("id"): j for j in current if isinstance(j, dict) and j.get("id")}
    for uj in updated_jobs:
        jid = uj.get("id")
        if not jid:
            continue
        if jid in cur_by_id:
            for k in ("last_run_local",):  # Note: trailing comma makes it a tuple
                if k in uj:
                    cur_by_id[jid][k] = uj[k]
        else:
            cur_by_id[jid] = uj
    merged = list(cur_by_id.values())
    JOBS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with JOBS_PATH.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    logger.info("Saved jobs.json with %d jobs (runner=%s)", len(merged), runner)

def _parse_iso(dt_str):
    if dt_str is None:
        return None
    dt = datetime.fromisoformat(dt_str)
    dt = dt.replace(tzinfo=LOCAL_TZ)
    return dt

def _floor_to_interval(now, anchor, interval_secs):
    delta = (now - anchor).total_seconds()
    if delta < 0:
        return anchor
    k = int(delta // interval_secs)
    return anchor + timedelta(seconds=k * interval_secs)

def due_run_time(job, now_local):
    """
    Determine the next due run time for a job based on its schedule.
    Conditions for a run to be due:
      - status must be "running"
      - now must be within [start_at, stop_at] (if set)
      - scheduled = latest boundary <= now
      - run if last_run_local < scheduled
    """
    if job.get("status") != "running":
        return None

    interval_secs = int(job.get("interval_secs", 1800))
    start_at = _parse_iso(job.get("start_at"))
    stop_at = _parse_iso(job.get("stop_at"))
    last_run = _parse_iso(job.get("last_run_local"))

    if start_at and now_local < start_at:
        return None
    if stop_at and now_local > stop_at:
        return None

    scheduled = _floor_to_interval(now_local, start_at, interval_secs)

    if start_at and scheduled < start_at:
        scheduled = start_at
    
    if last_run is None:
        return scheduled

    if scheduled.timestamp() > last_run.timestamp():
        return scheduled

    return None

def next_run_time(job, now_local):
    """
    Compute the next scheduled run time >= now_local, anchored at start_at: start_at + k * interval_secs
    Returns None if job is not running, or if stop_at exists and next time exceeds it.
    """
    if job.get("status") != "running":
        return None

    interval_secs = int(job.get("interval_secs", 1800))
    start_at = _parse_iso(job.get("start_at"))
    stop_at = _parse_iso(job.get("stop_at"))

    if stop_at and now_local > stop_at:
        return None
    
    if now_local <= start_at:
        nxt = start_at
    else:
        cur = _floor_to_interval(now_local, start_at, interval_secs)
        nxt = cur + timedelta(seconds=interval_secs)
        if nxt < start_at:
            nxt = start_at
    if stop_at and nxt > stop_at:
        return None
    return nxt