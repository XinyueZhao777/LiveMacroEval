# Enabling GitHub Pages

The site lives in `docs/` on `main`. One-time setup:

```bash
gh api -X POST repos/LiveMacroEval/LiveMacroEval.github.io/pages \
  -f 'source[branch]=main' -f 'source[path]=/docs'
```

Or in the UI: **Settings → Pages → Source: Deploy from a branch →
`main` / `/docs` → Save**.

Live at **https://livemacroeval.github.io/** about a minute later.
`.nojekyll` must stay present, otherwise Jekyll ignores any path starting with an
underscore. Every path in the site is relative, so it works unchanged from a repo
subpath or from a bare domain.

## Custom domain (later)

```bash
echo "livemacroeval.org" > docs/CNAME && git add docs/CNAME && git commit -m "custom domain" && git push
```

Then point DNS at GitHub — a `CNAME` record for `www` → `livemacroeval.github.io`,
or four `A` records at the apex IPs `185.199.108-111.153` — set the domain under
Settings → Pages, and tick **Enforce HTTPS**.

## Paper PDF

Not hosted; `docs/.gitignore` blocks `*.pdf` and the safety check flags any that
appear. The "Paper (arXiv)" button in `index.html` is disabled — swap its `href`
and drop the `disabled` class once the preprint is up.

## Ongoing updates

See `README.md`. The refresh is `python tools/update_site.py`, which runs the
release-safety audit itself; then commit and push.
`python tools/check_release_safety.py`, then commit and push.
