import time
import re
import unicodedata
import datetime as dt
from typing import Dict, Tuple, List, Any
import smtplib
from email.mime.text import MIMEText

from config import LOCAL_TZ, get_logger, MAX_RETRY, RETRY_SLEEP_SECS, SMTP_SERVER, SMTP_PORT, ALERT_EMAIL_FROM, ALERT_EMAIL_PASSWORD, ALERT_EMAIL_TO
from variables import get_variables
from prompts import build_system_msg, build_user_prompt
from llm_clients import get_client

logger = get_logger(__name__)

# ---------- Email utils: handle error ----------
def _notify_failure(job, model_name, last_error, last_raw):
    """
    Notify via email when a job fails MAX_RETRY times in a row.
    Very simple SMTP email sender.
    """
    subject = f"[LiveMacroEval] Forecast FAILED after retries: job={job.get('id')} model={model_name}"
    body = (
        f"Job ID: {job.get('id')}\n"
        f"Model: {model_name}\n"
        f"Target: {job.get('target_period')}\n"
        f"Variable group: {job.get('variable_group')}\n\n"
        f"Error:\n{repr(last_error)}\n\n"
        f"Last raw output:\n{(last_raw or '')[:200]}\n"
    )

    if not (ALERT_EMAIL_FROM and ALERT_EMAIL_PASSWORD and ALERT_EMAIL_TO):
        logger.error(
            "Failure notification skipped (no SMTP env): job=%s model=%s",
            job.get("id"), model_name,
        )
        return

    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = ALERT_EMAIL_FROM
        msg["To"] = ", ".join(ALERT_EMAIL_TO)

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(ALERT_EMAIL_FROM, ALERT_EMAIL_PASSWORD)
        server.sendmail(ALERT_EMAIL_FROM, ALERT_EMAIL_TO, msg.as_string())
        server.quit()

        logger.error("Failure notification email SENT: job=%s model=%s", job.get("id"), model_name)
    except Exception as e:
        logger.error("Failed to send failure notification email: %s", e)


# ---------- numeric parsing utils ----------
# minus sign normalization:
def _clean_number_str(s):
    s = unicodedata.normalize("NFKC", str(s))
    s = "".join("-" if (unicodedata.category(ch) == "Pd" or ch == "\u2212") else ch for ch in s)
    # "1,234.56" -> "1234.56"
    # "0.8%" -> "0.8"
    s = s.replace("%", "").replace(",", "")
    s = re.sub(r"\s+", "", s)
    # remove any non-numeric chars except + - . e E
    s = re.sub(r"[^0-9eE+\-\.]", "", s)
    return s

def _as_float(x):
    try:
        if isinstance(x, (int, float)):
            return float(x)
        s = _clean_number_str(str(x))
        if s == "" or s in {"+", "-"}:
            return None
        return float(s)
    except Exception:
        return None

def _parse_key_value_output(raw_text):
    matches = re.findall(r"(\b[\w_]+\b)\s*=\s*([^\s]+)", raw_text)
    kv = {}
    for k, v in matches:
        if k in ("target_month", "release_month"):
            kv[k] = v
        else:
            kv[k] = _as_float(v)
    parsed_ok = len(kv) > 0
    notes = f"found={len(kv)} pairs"
    return kv, parsed_ok, notes

def _build_prompt(job, variables):
    target_period = job.get("target_period")
    year, month = map(int, target_period.split("-"))
    target_month = dt.date(year, month, 1)

    rel_str = job.get("release_period")
    r_year, r_month = map(int, rel_str.split("-"))
    release_month = dt.date(r_year, r_month, 1)

    system_msg = build_system_msg()
    user_msg = build_user_prompt(target_month, release_month, variables)

    return system_msg, user_msg

# ---------- Forecast ----------
def forecast_once(job, model_name, now_local):
    """
    Run one prediction for 1 job + 1 model.
    return preds: {var_key: value}, meta: {...}
    """
    logger.info("Forecasting: job=%s model=%s target=%s",
        job.get("id"), model_name, job.get("target_period"))
    
    var_group = job.get("variable_group")
    variables = get_variables(var_group)
    system_msg, user_msg = _build_prompt(job, variables)

    client = get_client(model_name)

    last_error: Exception | None = None
    last_raw: str | None = None

    for attempt in range(1, MAX_RETRY + 1):
        try:
            logger.info("Forecasting attempt %d/%d: job=%s model=%s target=%s",
                attempt, MAX_RETRY, job.get("id"), model_name, job.get("target_period"))
            raw, citations = client(system_msg, user_msg)
            kv, parsed_ok, notes = _parse_key_value_output(raw)

            # If parsing failed or missing vars, raise error: handle NA values.
            missing_vars = [v["key"] for v in variables if kv.get(v["key"]) is None]
            if (not parsed_ok) or missing_vars:
                raise ValueError(f"Prediction parse failed: parsed_ok={parsed_ok}, missing_vars={missing_vars}, raw='{raw[:200]}'")
            
            target_period = kv.get("target_month", job.get("target_period"))
            release_period = kv.get("release_month", job.get("target_period"))
            preds = {v["key"]: kv.get(v["key"]) for v in variables}
            meta = {"job_id": job.get("id"), "model": model_name,
                    "timestamp_local": now_local.strftime("%Y-%m-%d %H:%M:%S"),
                    "target_month": target_period,
                    "release_month": release_period,
                    "parsed_ok": parsed_ok,
                    "parsed_notes": notes,
                    "raw_model_output": raw,
                    "citations": "" if citations is None else str(citations)}
            logger.info("Forecast done: job=%s model=%s target_period=%s parsed_ok=%s notes=%s", 
                        job.get("id"), model_name, target_period, parsed_ok, notes)
            return preds, meta
        except Exception as e:
            last_error = e
            last_raw = locals().get("raw", None)
            logger.exception("Forecast attempt %d/%d FAILED: job=%s model=%s error=%s",
                attempt, MAX_RETRY, job.get("id"), model_name, e)
            if attempt < MAX_RETRY:
                time.sleep(RETRY_SLEEP_SECS)
    _notify_failure(job, model_name, last_error, last_raw)
    raise last_error