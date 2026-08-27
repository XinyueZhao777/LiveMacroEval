# LiveMacroEval — project website

The public site for the LiveMacroEval benchmark, served by GitHub Pages from this
`docs/` folder at **https://xinyuezhao777.github.io/LiveMacroEval/**.

Plain HTML + CSS + one JS file. No build step, no dependencies.

```
docs/
├── index.html                structure and prose. Contains NO numbers.
├── data/leaderboard.json     single source of truth for every number  ← edit this
├── assets/css/style.css
├── assets/js/main.js         fetches the JSON, renders the tables
├── assets/figures/*.png      aggregate figures from the paper
├── update_site.py            regenerates the JSON from the private pipeline
├── check_release_safety.py   pre-push audit — run it every time
├── .gitignore                blocks data files from ever landing here
└── .nojekyll                 serve the files as-is, no Jekyll
```

## This repo is public. The scoring inputs are not.

Three of the pipeline's inputs cannot be redistributed (see `../DATA_SOURCES.md`):
the Bloomberg ECOS survey exports, the FirstRateData ES futures minute bars, and
the scraped Investing.com calendar. The repo's `.gitignore` already excludes them
along with `Results/market_surprise_capture_score/**/bloomberg_overlay/`, the
scoring output the website reads.

The website publishes **aggregate results only** — per-model point estimates,
confidence intervals, and event counts, exactly the numbers the paper reports in
Figure 2. No row-level, per-release, or raw vendor value goes into this folder.
Three mechanisms keep it that way:

1. `update_site.py` reads only five aggregate columns and refuses to run against
   this repo's own `Results/`; the overlay must come from your private checkout.
2. `docs/.gitignore` blocks every row-level file extension, plus PDFs.
3. `check_release_safety.py` fails on data files, stray JSON, secrets, absolute
   local paths, links to the private repo, and any weakening of either
   `.gitignore`.

## Preview locally

```bash
cd docs && python3 -m http.server 8000
# open http://localhost:8000
```

Serve it — don't open `index.html` via `file://`, or the browser blocks the
`fetch()` of `leaderboard.json` and the tables stay empty.

## The biweekly refresh

1. Run the data pipeline in your private checkout, per its `UPDATE_PIPELINE.md`.
   That produces a new dated overlay, e.g. `investing_overlay_0906/`.
2. Regenerate the site data:

   ```bash
   . /home/ruiyi/anaconda3/bin/activate && conda activate livemacro
   python docs/update_site.py --results-root /home/ruiyi/livemacro/Results
   ```

   Add `--dry-run` first to inspect. `--results-root` defaults to
   `/home/ruiyi/livemacro/Results`; set `$LIVEMACRO_RESULTS` to change it
   permanently. Paper figures come from `--figures-root`
   (default `/home/ruiyi/livemacro/Paper/figures`).
3. Audit, then push:

   ```bash
   python docs/check_release_safety.py && \
   git add docs && git commit -m "site: refresh $(date +%F)" && git push
   ```

Pages redeploys in about a minute.

### Which overlay

The default is `bloomberg_overlay`, the **frozen paper window** (Nov 2025 – Mar
2026). The Bloomberg-based score is frozen at the 2026-05-05 cutoff; later months
are scored against the Investing.com consensus proxy and are comparable only to
each other. If you point the site at one of those, pass a matching caption so the
leaderboard says which window it is:

```bash
python docs/update_site.py --overlay investing_overlay_0906 \
  --window "Investing.com consensus proxy · target periods Apr – Aug 2026"
```

### Adding a model arm

Add its code name to `MODEL_LABELS` in `update_site.py`. Arms listed in `DROPPED`
are excluded to stay consistent with Figure 2 of the paper.

## Editing content

Prose lives in `index.html`; numbers live in `data/leaderboard.json`. Keeping that
split is what makes the refresh a one-command operation.
