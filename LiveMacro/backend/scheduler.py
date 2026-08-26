import time
from datetime import datetime, timezone, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Tuple, Any
import traceback
import argparse

from config import BASE_SLEEP_SECS, MAX_WORKERS, LOCAL_TZ, get_logger, REPO_ROOT, CONTROL_DIR, STOP_FILE, PAUSE_FILE
from jobs import load_jobs, due_run_time, save_jobs, next_run_time 
from forecasting import forecast_once
from storage import append_predictions, sync_data_light
from git_utils import git_commit_and_push

import os

logger = get_logger(__name__)

SCHEDULER_WHERE = os.path.join(os.path.dirname(__file__), "scheduler.where")

def _write_scheduler_where():
    """Write PID and start time to scheduler.where so other shells can find
    the running scheduler. The file is local to this checkout."""
    pid = os.getpid()
    start = datetime.now(LOCAL_TZ).isoformat()
    with open(SCHEDULER_WHERE, "w") as f:
        f.write(f"pid={pid} start={start}\n")
    logger.info("Wrote scheduler.where: pid=%s start=%s", pid, start)

def _run_single_task(job, model_name, scheduled_local_iso):
    _ = get_logger(f"worker_{model_name}")
    try:
        scheduled_local = datetime.fromisoformat(scheduled_local_iso).astimezone(LOCAL_TZ)
        preds, meta = forecast_once(job, model_name, scheduled_local)
        append_predictions(job, model_name, preds, meta)
        return job.get("id"), model_name, True, "", scheduled_local_iso
    except Exception as e:
        err = f"{e}\n{traceback.format_exc()}"
        logger.error("Task failed in worker: job=%s model=%s error=%s", job.get("id"), model_name, err)
        return job.get("id"), model_name, False, err, scheduled_local_iso

def _compute_next_wakeup(jobs, now_local):
    candidates = []
    for job in jobs:
        nxt = next_run_time(job, now_local)
        if nxt:
            candidates.append(nxt)
    if candidates:
        return min(candidates)
    return now_local + timedelta(seconds=BASE_SLEEP_SECS)

def _sleep_until(dt_local, min_sleep=0.001, check_interval=30):
    """Sleep until dt_local, but check for STOP file every check_interval seconds."""
    while True:
        if os.path.exists(STOP_FILE):
            return  # Exit sleep early, main loop will handle STOP
        now = datetime.now(LOCAL_TZ)
        delta = (dt_local - now).total_seconds()
        if delta <= 0:
            return
        time.sleep(min(check_interval, max(min_sleep, delta)))

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runner", type=str, default=None,
                   help="Run only jobs assigned to this runner (job.runner==runner or 'any'). "
                        "Example: --runner serverA")
    return p.parse_args()

def main():
    args = _parse_args()
    runner = args.runner
    print(f"Scheduler starting... runner={runner}")
    logger.info("Scheduler starting... runner=%s", runner)
    _write_scheduler_where()
    # logger.info("Scheduler starting...")
    while True:
        # STOP: graceful shutdown.
        if os.path.exists(STOP_FILE):
            logger.info("STOP detected at %s. Exiting scheduler.", STOP_FILE)
            break

        # PAUSE: Pause (process remains alive).
        if os.path.exists(PAUSE_FILE):
            logger.info("PAUSE detected at %s. Sleeping 5s...", PAUSE_FILE)
            time.sleep(5)
            continue

        loop_start = time.time()
        # now_utc = datetime.now(timezone.utc)
        now_local = datetime.now(LOCAL_TZ)
        print(f"Scheduler loop at {now_local.isoformat()}")

        jobs = load_jobs(runner=runner)
        tasks = []

        for job in jobs:
            scheduled_dt = due_run_time(job, now_local)
            if not scheduled_dt:
                continue
            models = job.get("models", [])
            if not models:
                logger.warning("Job %s has no models, skipping", job.get("id"))
                continue
            scheduled_iso = scheduled_dt.isoformat()
            for model_name in models:
                tasks.append((job, model_name, scheduled_iso))

        if tasks:
            logger.info("Tasks due: %d", len(tasks))
            
            executed = {}
            max_workers = min(len(tasks), MAX_WORKERS)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {
                    executor.submit(_run_single_task, job, model_name, scheduled_iso): (job, model_name, scheduled_iso)
                    for job, model_name, scheduled_iso in tasks
                }

                for future in as_completed(future_to_task):
                    job, model_name, scheduled_iso = future_to_task[future]
                    job_id = job.get("id")
                    try:
                        jid, mname, ok, err, sched = future.result()
                        executed[jid] = sched
                        if ok:
                            logger.info("Task OK: job=%s model=%s scheduled=%s", jid, mname, sched)
                        else:
                            logger.error("Task error: job=%s model=%s scheduled=%s err=%s", jid, mname, sched, err)
                    except Exception as e:
                        logger.error("Task crashed: job=%s model=%s scheduled=%s err=%s", job_id, model_name, scheduled_iso, e)

            for job in jobs:
                jid = job.get("id")
                if jid in executed:
                    job["last_run_local"] = executed[jid]

            save_jobs(jobs, runner)

            try:
                sync_data_light()
                git_commit_and_push()
            except Exception as e:
                logger.error("git sync failed: %s", e)
        else:
            logger.info("No tasks due at %s", now_local.isoformat())

        after_loop = datetime.now(LOCAL_TZ)
        next_wakeup = _compute_next_wakeup(load_jobs(runner), after_loop)
        logger.info("Next wakeup scheduled at %s", next_wakeup.isoformat())
        _sleep_until(next_wakeup)

if __name__ == "__main__":
    main()