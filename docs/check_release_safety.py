#!/usr/bin/env python3
"""Fail loudly if anything under docs/ should not be published.

This repo is public; several of the pipeline's inputs are not redistributable
(Bloomberg ECOS survey exports, FirstRateData ES futures bars, the scraped
Investing.com calendar -- see DATA_SOURCES.md). The website is allowed to show
AGGREGATE results derived from them, which is what the paper reports, but must
never carry the underlying rows.

Run before every push, or wire it in as a pre-commit hook:

    python docs/check_release_safety.py

Exit code 0 = safe to publish, 1 = something needs attention.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parent
REPO = SITE.parent

# Extensions that carry row-level data. None of them belong in docs/.
DATA_EXT = {".csv", ".parquet", ".xlsx", ".xls", ".pkl", ".pickle",
            ".feather", ".h5", ".hdf5", ".db", ".sqlite", ".jsonl", ".tsv"}

# Everything docs/ is allowed to contain, as suffixes or exact names.
ALLOWED_EXT = {".html", ".css", ".js", ".png", ".jpg", ".jpeg", ".svg",
               ".webp", ".ico", ".md", ".py", ".txt", ".json"}
ALLOWED_NAMES = {".nojekyll", ".gitignore", "CNAME"}

# Only this one JSON may exist, and only with these top-level keys.
ALLOWED_JSON = {"data/leaderboard.json"}
ALLOWED_JSON_KEYS = {"_comment", "last_updated", "next_update", "headline",
                     "agent_design", "indicators", "comparators"}

SECRET_PATTERNS = [
    (re.compile(r"\b(sk-[A-Za-z0-9_\-]{20,}|gh[pousr]_[A-Za-z0-9]{30,})"), "API key or GitHub token"),
    (re.compile(r"(?i)\b(api[_-]?key|secret|passwd|password|bearer)\s*[:=]\s*['\"][^'\"]{8,}"), "hardcoded credential"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key id"),
]

# Things that reveal the private infrastructure rather than the research.
LEAK_PATTERNS = [
    (re.compile(r"github\.com[:/]LiveMacro/LiveMacro(?:\.git|\b)"), "link to the PRIVATE repo"),
    (re.compile(r"\bece228-\d+\b"), "internal hostname"),
    (re.compile(r"/home/[a-z0-9_]+/"), "absolute local filesystem path"),
    (re.compile(r"(?i)\bbloomberg[_ ]?(daily|release)[_ ]?consensus\b"), "reference to a withheld Bloomberg table"),
]

# update_site.py and this file legitimately document the private paths.
PATH_EXEMPT = {"update_site.py", "check_release_safety.py", "README.md", "DEPLOY.md"}

problems: list[str] = []
notes: list[str] = []


def check_files() -> None:
    for p in sorted(SITE.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(SITE).as_posix()
        ext = p.suffix.lower()

        if ext in DATA_EXT:
            problems.append(f"{rel}: {ext} data file must not be published")
            continue
        if p.name not in ALLOWED_NAMES and ext not in ALLOWED_EXT:
            problems.append(f"{rel}: unexpected file type {ext or '(none)'}")
            continue
        if ext == ".json" and rel not in ALLOWED_JSON:
            problems.append(f"{rel}: only {sorted(ALLOWED_JSON)} may be published")
        if ext == ".pdf":
            problems.append(f"{rel}: PDF publishing was declined; remove it")


def check_leaderboard() -> None:
    f = SITE / "data/leaderboard.json"
    if not f.exists():
        problems.append("data/leaderboard.json is missing")
        return
    try:
        d = json.loads(f.read_text())
    except json.JSONDecodeError as e:
        problems.append(f"data/leaderboard.json is not valid JSON: {e}")
        return

    extra = set(d) - ALLOWED_JSON_KEYS
    if extra:
        problems.append(f"leaderboard.json has unrecognised top-level keys: {sorted(extra)} "
                        "-- confirm they carry no row-level data")

    rows = d.get("headline", {}).get("rows", [])
    if len(rows) > 12:
        problems.append(f"leaderboard.json headline has {len(rows)} rows; that looks "
                        "per-release rather than per-model")
    for r in rows:
        stray = set(r) - {"name", "kind", "score", "ci", "events", "note", "best"}
        if stray:
            problems.append(f"leaderboard row {r.get('name')!r} has unexpected fields: {sorted(stray)}")

    src = str(d.get("headline", {}).get("source", ""))
    if src.startswith("/") or "\\" in src:
        problems.append(f"leaderboard.json 'source' leaks an absolute path: {src}")

    notes.append(f"leaderboard.json: {len(rows)} aggregate rows, no row-level fields")


def check_text() -> None:
    for p in sorted(SITE.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in {".html", ".css", ".js", ".json", ".md", ".py", ".txt"}:
            continue
        rel = p.relative_to(SITE).as_posix()
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        for pat, label in SECRET_PATTERNS:
            if pat.search(text):
                problems.append(f"{rel}: possible {label}")
        for pat, label in LEAK_PATTERNS:
            if label == "absolute local filesystem path" and p.name in PATH_EXEMPT:
                continue
            m = pat.search(text)
            if m:
                problems.append(f"{rel}: {label} -- {m.group(0)!r}")


def check_gitignore() -> None:
    """Read-only: confirm both .gitignore layers still hold."""
    repo_gi = REPO / ".gitignore"
    if not repo_gi.exists():
        problems.append("repo .gitignore is missing")
    else:
        text = repo_gi.read_text()
        for needed in ("bloomberg_overlay/", "Results/bloomberg_consensus/",
                       "Results/data_sp500futures/"):
            if needed not in text:
                problems.append(f"repo .gitignore no longer excludes {needed!r} "
                                "-- a withheld input could now be committed")

    docs_gi = SITE / ".gitignore"
    if not docs_gi.exists():
        problems.append("docs/.gitignore is missing")
        return
    text = docs_gi.read_text()
    for needed in ("*.csv", "*.parquet", "*.xlsx", "*.pdf"):
        if needed not in text:
            problems.append(f"docs/.gitignore no longer excludes {needed!r}")
    notes.append("both .gitignore layers intact")


def main() -> int:
    check_files()
    check_leaderboard()
    check_text()
    check_gitignore()

    for n in notes:
        print(f"  ok   {n}")
    if problems:
        print(f"\nFAIL -- {len(problems)} issue(s) before this can be published:\n")
        for p in problems:
            print(f"  !! {p}")
        return 1
    print("\nPASS -- docs/ contains only publishable material.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
