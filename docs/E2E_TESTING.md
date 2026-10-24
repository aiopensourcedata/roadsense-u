# Browser end-to-end verification

The end-to-end runner uses real local services and Chrome/Chromium. It is separate from the ordinary pytest suite and uses generated synthetic scenes only.

## Run it

From the project source directory, with Python 3.11–3.13, Docker Compose and a virtual environment:

```bash
pip install -e '.[dev,dashboard,e2e]'
# Create local credentials once; keep an existing .env.
python scripts/init_local_env.py
docker compose up -d --build --wait

# Uses an installed Google Chrome/Chromium if available.
# If neither is installed, provision the Playwright browser explicitly:
python -m playwright install chromium

python scripts/run_e2e.py --screenshots docs/assets
pytest -q
```

Skip the initializer if `.env` already exists. The runner reads the API key and PostgreSQL URL from environment settings; credentials never appear in the browser or screenshots. Use `--browser-executable /path/to/chrome` to select a browser, or `--dashboard-url http://127.0.0.1:8502` for a separately running dashboard. Its API URL/database still come from `ROADSENSE_` settings. On Linux, a downloaded Chromium may require OS libraries; `python -m playwright install --with-deps chromium` explicitly provisions those dependencies where permitted.

Run without `--screenshots` to exercise the browser without changing documentation images. The screenshot option deliberately overwrites the five reviewed gallery images and their provenance manifest. Do not use it on real traffic footage.

## What the runner actually does

1. Checks the API health, unauthenticated rejection, malformed-event rejection and request-size limit.
2. Generates a 100-frame, five-second crossing fixture with two known agents. Its dedicated color-component detector and gated tracker are explicitly selected; pretrained YOLO/ByteTrack perception is not evaluated by this synthetic browser suite.
3. Processes four episodes for two fictional cameras at a unique synthetic location. Every trajectory and rule score is computed from the generated frames. A 0.75-second alert cooldown deliberately produces multiple emissions per interaction; these are not independent labeled conflicts.
4. Submits the events over authenticated HTTP and independently queries PostgreSQL for exactly the computed event IDs. It verifies retrieval, UUID-idempotent replay and synthetic/real query isolation.
5. Supplies synthetic exposure windows with two known generated agents per episode, then checks camera totals and exposure normalization. These are explicitly synthetic truth counts, not inferred traffic volume or validated real-world annotations.
6. Runs the separate 480-row synthetic tabular baseline benchmark. That table is not extracted from the crossing video and is not real-world model validation.
7. Opens the actual dashboard in a headless browser, verifies the UI count against PostgreSQL, changes the provenance filter, and inspects overview, evidence/metadata, analytics, processed artifacts and evaluation views.
8. Opens the actual local MJPEG viewer while the pipeline processes the synthetic clip. It checks that frames arrive and that no annotated recording was created.
9. Checks a mobile viewport for page overflow and records uncaught JavaScript errors.
10. Writes a JSON result with explicit checks and, when requested, browser screenshots with SHA256 hashes.

## Evidence and retention

- `outputs/e2e-<run-id>/results.json`: local check results, versions and generated location ID.
- `outputs/e2e-latest.json`: most recent successful run.
- `docs/assets/provenance.json`: successful screenshot run, image descriptions, synthetic flags and hashes.
- `docs/assets/*.png`: five actual browser captures, all referenced by README.

Generated input, trajectories, features, model artifacts and reports stay under ignored `outputs/` directories. Synthetic API records remain available for inspection under their unique location ID and follow the configured database retention. The runner does not delete or modify pre-existing observations. It creates no remote services and uploads no frames.

A screenshot of a working view is visual evidence, not a quantitative accuracy result. The checks do not establish physical-camera operation, geographic generalization, crash anticipation, correct real-world speed, privacy certification or deployment security. Browser checks require a running local stack; the normal pytest suite remains independent of Docker, browsers and camera hardware.

## Continuous integration

`.github/workflows/e2e.yml` provides an opt-in manual workflow. It starts Compose with generated local credentials, installs Chromium, runs the browser checks and retains only the synthetic screenshots and JSON reports as short-lived artifacts. It has no publication step. A local execution does not imply hosted CI has executed.
