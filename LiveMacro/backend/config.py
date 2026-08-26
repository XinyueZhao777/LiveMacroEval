import os
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from zoneinfo import ZoneInfo

# ---------- Paths & basic config ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
CONFIG_DIR = REPO_ROOT / "config"
DATA_DIR = REPO_ROOT / "data"
DATA_LIGHT_DIR = REPO_ROOT / "data_light"
LOG_DIR = REPO_ROOT / "logs"

CONTROL_DIR = os.path.join(REPO_ROOT, "backend", "control")
STOP_FILE = os.path.join(CONTROL_DIR, "STOP")
PAUSE_FILE = os.path.join(CONTROL_DIR, "PAUSE")

JOBS_PATH = CONFIG_DIR / "jobs.json"

LOCAL_TZ = ZoneInfo("America/New_York")

# for scheduler
BASE_SLEEP_SECS = 10

# How many model processes can run in parallel at most?
MAX_WORKERS = 4

# LLM web search settings. Override to any U.S. region if needed.
USER_LOCATION = {
    "type": "approximate",
    "approximate": {"country": "US"},
}

DEFAULT_MODEL_NAME = "gpt-5-search-api"

def ensure_dirs():
    for p in [CONFIG_DIR, DATA_DIR, DATA_LIGHT_DIR, LOG_DIR]:
        p.mkdir(parents=True, exist_ok=True)
ensure_dirs()

# ---------- Prediction config ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MAX_RETRY = 5
RETRY_SLEEP_SECS = 5.0

# ---------- Email config ----------
# Optional: SMTP alert when a job exhausts MAX_RETRY.
# If any of ALERT_EMAIL_FROM / ALERT_EMAIL_PASSWORD / ALERT_EMAIL_TO is empty,
# _notify_failure() in forecasting.py is a no-op.
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM", "")
ALERT_EMAIL_PASSWORD = os.getenv("ALERT_EMAIL_PASSWORD", "")
ALERT_EMAIL_TO = [x.strip() for x in os.getenv("ALERT_EMAIL_TO", "").split(",") if x.strip()]

# ---------- Logging ----------
_LOGGER_INITIALIZED = False


def get_logger(name):
    """
    Return a logger with both console + rotating file handler.
    All modules use this logger, and each log entry includes the
    process name to make multi-process behavior easy to track.
    """
    global _LOGGER_INITIALIZED

    logger = logging.getLogger(name)

    if _LOGGER_INITIALIZED:
        return logger

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(processName)s] %(name)s - %(message)s"
    )

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Rotating file: daily, keep 7 days
    log_file = LOG_DIR / "backend.log"
    fh = TimedRotatingFileHandler(
        log_file, when="D", interval=1, backupCount=7, encoding="utf-8"
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    _LOGGER_INITIALIZED = True
    return logger
