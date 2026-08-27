#!/usr/bin/env python3
"""Gate on what docs/ is allowed to publish. Exit 0 = safe, 1 = stop.

This repo is public. Several pipeline inputs are not redistributable -- the
Bloomberg ECOS survey exports, the FirstRateData ES futures minute bars, and the
scraped Investing.com calendar (see ../DATA_SOURCES.md). The website may show the
FINAL AGGREGATE FIGURES AND TABLES that the paper reports, and nothing else. No
row-level record, no per-release value, no raw vendor number.

The check is an ALLOWLIST, not a blocklist: every file in docs/ must be named in
ALLOWED_FILES or in update_site.FIGURES, or this fails. Adding anything new to
the published site is therefore a deliberate, reviewable edit to this file. On
top of that:

  * leaderboard.json is validated against an explicit recursive schema, with
    caps on array length, numeric-literal count, and byte size, so a per-release
    or hourly dump cannot fit through even under an approved key;
  * every file's bytes are sniffed -- a CSV renamed to .txt, a table pasted into
    the HTML, or a base64 blob is caught by shape, not by extension;
  * images must really be images, and are capped in size;
  * both .gitignore layers are re-verified.

Run it: python tools/check_release_safety.py
It also runs automatically from update_site.py, from the git pre-commit hook
(tools/hooks/pre-commit), and in CI on every push (.github/workflows/).
tools/test_release_safety.py exercises it against 21 planted leaks.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
SITE = REPO / "docs"

sys.path.insert(0, str(TOOLS))
from update_site import FIGURES  # single source of truth for the figure set

# --------------------------------------------------------------------------
# 1. The allowlist. Anything under docs/ not named here fails the check.
# --------------------------------------------------------------------------
ALLOWED_FILES = {
    ".gitignore",
    ".nojekyll",
    "CNAME",                      # only once a custom domain is set up
    "index.html",
    "assets/css/style.css",
    "assets/js/main.js",
    "data/leaderboard.json",
}
ALLOWED_FILES |= {f"assets/figures/{name}" for name in FIGURES}

# Per-file byte ceilings. Generous, but a data dump blows past them.
MAX_BYTES = {
    ".json": 32 * 1024,
    ".html": 128 * 1024,
    ".css": 64 * 1024,
    ".js": 64 * 1024,
    ".md": 32 * 1024,
    ".py": 64 * 1024,
    ".png": 2 * 1024 * 1024,
}
DEFAULT_MAX_BYTES = 16 * 1024

IMAGE_MAGIC = {
    ".png": b"\x89PNG\r\n\x1a\n",
    ".jpg": b"\xff\xd8\xff",
    ".jpeg": b"\xff\xd8\xff",
    ".svg": b"<",
}

# --------------------------------------------------------------------------
# 2. leaderboard.json schema. Only the final table may live here.
# --------------------------------------------------------------------------
MAX_JSON_BYTES = 32 * 1024
MAX_ARRAY_LEN = 30        # a per-release or hourly series is far longer
MAX_NUMERIC_LITERALS = 150  # the real file holds ~25

S = str
N = (int, float)

ROW = {"name": S, "kind": S, "score": N, "ci": "ci", "events": "int_or_null",
       "note?": S}
AGENT_ROW = {"name": S, "score": N, "best?": bool, "note?": S}
SCHEMA = {
    "_comment?": S,
    "last_updated": "date",
    "next_update": "date",
    "headline": {"title": S, "window": S, "note": S, "source": S,
                 "rows": [ROW]},
    "agent_design": {"title": S, "window": S, "note": S, "rows": [AGENT_ROW]},
    "indicators": [{"theme": S, "blurb": S, "items": [S]}],
    "comparators": {"fed": [{"name": S, "target": S}]},
}

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# --------------------------------------------------------------------------
# 3. Content sniffing, applied to every text file regardless of extension.
# --------------------------------------------------------------------------
SECRET_PATTERNS = [
    (re.compile(r"\b(sk-[A-Za-z0-9_\-]{20,}|gh[pousr]_[A-Za-z0-9]{30,})"), "API key or GitHub token"),
    (re.compile(r"(?i)\b(api[_-]?key|secret|passwd|password|bearer)\s*[:=]\s*['\"][^'\"]{8,}"), "hardcoded credential"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key id"),
]
LEAK_PATTERNS = [
    (re.compile(r"github\.com[:/]LiveMacro/LiveMacro(?:\.git|\b)"), "link to the PRIVATE repo"),
    (re.compile(r"\bece228-\d+\b"), "internal hostname"),
    (re.compile(r"/home/[a-z0-9_]+/"), "absolute local filesystem path"),
    (re.compile(r"(?i)\bbloomberg[_ ]?(daily|release)[_ ]?consensus\b"), "withheld Bloomberg table name"),
    (re.compile(r"(?i)\b(firstratedata|es[_ ]futures[_ ]minute)\b"), "withheld futures-bar source"),
]
# docs/ is published content only; nothing in it may name a private path.
PROSE_EXEMPT: set[str] = set()

# A delimiter-separated row: >=4 separators and >=3 numeric fields.
CSV_ROW_RE = re.compile(r"^[^,\t]*([,\t][^,\t]*){4,}$")
NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$")
BASE64_BLOB_RE = re.compile(r"[A-Za-z0-9+/]{500,}={0,2}")
# A long run of comma-separated numbers pasted inline, e.g. an embedded series.
NUMBER_RUN_RE = re.compile(r"(-?\d+\.\d+\s*,\s*){25,}")

TEXT_EXT = {".html", ".css", ".js", ".json", ".md", ".py", ".txt", ""}

problems: list[str] = []
notes: list[str] = []


def fail(msg: str) -> None:
    problems.append(msg)


# --------------------------------------------------------------------------


def check_allowlist() -> None:
    seen = set()
    for p in sorted(SITE.rglob("*")):
        if p.is_dir() or "__pycache__" in p.parts or ".git" in p.parts:
            continue
        rel = p.relative_to(SITE).as_posix()
        seen.add(rel)
        if rel not in ALLOWED_FILES:
            fail(f"{rel}: not on the allowlist. If it belongs on the public site, "
                 "add it to ALLOWED_FILES here and say why in the commit.")
            continue
        limit = MAX_BYTES.get(p.suffix.lower(), DEFAULT_MAX_BYTES)
        size = p.stat().st_size
        if size > limit:
            fail(f"{rel}: {size:,} bytes exceeds the {limit:,} cap for {p.suffix or 'this file'} "
                 "-- unexpectedly large files are how bulk data arrives")

    required = {"index.html", "assets/css/style.css", "assets/js/main.js",
                "data/leaderboard.json", ".nojekyll", ".gitignore"}
    for r in sorted(required - seen):
        fail(f"{r} is missing -- the site will not render correctly")

    missing_figs = [f"assets/figures/{n}" for n in FIGURES
                    if f"assets/figures/{n}" not in seen]
    if missing_figs:
        notes.append(f"{len(missing_figs)} figure(s) declared but absent: "
                     f"{', '.join(missing_figs)}")
    notes.append(f"{len(seen)} file(s), all on the allowlist")


def check_images() -> None:
    for name in FIGURES:
        p = SITE / "assets/figures" / name
        if not p.exists():
            continue
        magic = IMAGE_MAGIC.get(p.suffix.lower())
        head = p.read_bytes()[:16]
        if magic and not head.startswith(magic):
            fail(f"assets/figures/{name}: not a real {p.suffix} file -- "
                 "a data file may have been renamed")


# --------------------------------------------------------------------------


def walk_schema(node, spec, path: str) -> None:
    if spec == "date":
        if not (isinstance(node, str) and DATE_RE.match(node)):
            fail(f"leaderboard.json {path}: expected YYYY-MM-DD, got {node!r}")
        return
    if spec == "int_or_null":
        if not (node is None or isinstance(node, int)):
            fail(f"leaderboard.json {path}: expected an integer or null, got {node!r}")
        return
    if spec == "ci":
        if node is None:
            return
        if not (isinstance(node, list) and len(node) == 2
                and all(isinstance(x, N) and not isinstance(x, bool) for x in node)):
            fail(f"leaderboard.json {path}: 'ci' must be null or exactly two numbers, got {node!r}")
        return

    if isinstance(spec, dict):
        if not isinstance(node, dict):
            fail(f"leaderboard.json {path}: expected an object, got {type(node).__name__}")
            return
        required = {k for k in spec if not k.endswith("?")}
        optional = {k[:-1] for k in spec if k.endswith("?")}
        for k in sorted(required - set(node)):
            fail(f"leaderboard.json {path}: missing required key {k!r}")
        for k in sorted(set(node) - required - optional):
            fail(f"leaderboard.json {path}: unknown key {k!r} -- every published field "
                 "must be declared in SCHEMA, so row-level data cannot ride along")
        for k, v in node.items():
            sub = spec.get(k, spec.get(k + "?"))
            if sub is not None:
                walk_schema(v, sub, f"{path}.{k}")
        return

    if isinstance(spec, list):
        if not isinstance(node, list):
            fail(f"leaderboard.json {path}: expected a list, got {type(node).__name__}")
            return
        if len(node) > MAX_ARRAY_LEN:
            fail(f"leaderboard.json {path}: {len(node)} elements exceeds the "
                 f"{MAX_ARRAY_LEN} cap -- this looks like per-release data, not a summary table")
        for i, item in enumerate(node):
            walk_schema(item, spec[0], f"{path}[{i}]")
        return

    if spec is bool:
        if not isinstance(node, bool):
            fail(f"leaderboard.json {path}: expected a boolean, got {node!r}")
        return
    if spec is N or spec == N:
        if isinstance(node, bool) or not isinstance(node, N):
            fail(f"leaderboard.json {path}: expected a number, got {node!r}")
        return
    if spec is S:
        if not isinstance(node, str):
            fail(f"leaderboard.json {path}: expected a string, got {type(node).__name__}")
        return


def count_numbers(o) -> int:
    if isinstance(o, bool):
        return 0
    if isinstance(o, N):
        return 1
    if isinstance(o, dict):
        return sum(count_numbers(v) for v in o.values())
    if isinstance(o, list):
        return sum(count_numbers(v) for v in o)
    return 0


def check_leaderboard() -> None:
    f = SITE / "data/leaderboard.json"
    if not f.exists():
        return  # already reported by the allowlist pass
    raw = f.read_bytes()
    if len(raw) > MAX_JSON_BYTES:
        fail(f"data/leaderboard.json: {len(raw):,} bytes exceeds the {MAX_JSON_BYTES:,} cap")
    try:
        d = json.loads(raw)
    except json.JSONDecodeError as e:
        fail(f"data/leaderboard.json is not valid JSON: {e}")
        return

    walk_schema(d, SCHEMA, "$")

    n = count_numbers(d)
    if n > MAX_NUMERIC_LITERALS:
        fail(f"data/leaderboard.json holds {n} numeric literals, over the "
             f"{MAX_NUMERIC_LITERALS} cap -- the published table should be one row "
             "per model, not one per release")

    src = str(d.get("headline", {}).get("source", ""))
    if src.startswith("/") or "\\" in src or re.search(r"/home/|C:", src):
        fail(f"data/leaderboard.json 'source' leaks a filesystem path: {src!r}")

    rows = d.get("headline", {}).get("rows", [])
    notes.append(f"leaderboard.json: {len(rows)} model rows, {n} numeric literals, "
                 f"{len(raw):,} bytes -- all within caps")


# --------------------------------------------------------------------------


def check_content() -> None:
    """Sniff every text file's bytes. Extension is not trusted."""
    for p in sorted(SITE.rglob("*")):
        if p.is_dir() or "__pycache__" in p.parts:
            continue
        rel = p.relative_to(SITE).as_posix()
        if p.suffix.lower() not in TEXT_EXT:
            continue
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue

        for pat, label in SECRET_PATTERNS:
            if pat.search(text):
                fail(f"{rel}: possible {label}")
        for pat, label in LEAK_PATTERNS:
            if rel in PROSE_EXEMPT:
                continue
            m = pat.search(text)
            if m:
                fail(f"{rel}: {label} -- {m.group(0)!r}")

        if BASE64_BLOB_RE.search(text):
            fail(f"{rel}: long base64 blob -- data may be embedded inline")
        if NUMBER_RUN_RE.search(text):
            fail(f"{rel}: a long run of comma-separated decimals -- "
                 "a numeric series appears to be embedded inline")

        # Consecutive delimiter-separated numeric rows = a pasted table.
        run = 0
        for line in text.splitlines():
            s = line.strip()
            if CSV_ROW_RE.match(s) and sum(
                    1 for c in re.split(r"[,\t]", s) if NUMERIC_RE.match(c.strip())) >= 3:
                run += 1
                if run >= 3:
                    fail(f"{rel}: {run}+ consecutive delimiter-separated numeric rows "
                         "-- tabular data appears to be embedded in this file")
                    break
            else:
                run = 0


def check_gitignore() -> None:
    repo_gi = REPO / ".gitignore"
    if not repo_gi.exists():
        fail("repo .gitignore is missing")
    else:
        text = repo_gi.read_text()
        for needed in ("bloomberg_overlay/", "Results/bloomberg_consensus/",
                       "Results/data_sp500futures/"):
            if needed not in text:
                fail(f"repo .gitignore no longer excludes {needed!r} -- "
                     "a withheld input could now be committed")

    docs_gi = SITE / ".gitignore"
    if not docs_gi.exists():
        fail("docs/.gitignore is missing")
        return
    text = docs_gi.read_text()
    for needed in ("*.csv", "*.parquet", "*.xlsx", "*.pdf"):
        if needed not in text:
            fail(f"docs/.gitignore no longer excludes {needed!r}")
    notes.append("both .gitignore layers intact")


def main() -> int:
    check_allowlist()
    check_images()
    check_leaderboard()
    check_content()
    check_gitignore()

    for n in notes:
        print(f"  ok   {n}")
    if problems:
        print(f"\nFAIL -- {len(problems)} issue(s); do not publish:\n")
        for p in problems:
            print(f"  !! {p}")
        return 1
    print("\nPASS -- docs/ holds only the final figures and tables.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
