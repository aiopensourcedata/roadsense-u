# Contributing

RoadSense-U is an early research prototype. Contributions should preserve the distinction between measured features, model estimates, labeled conflicts and observed collisions. Synthetic tests are not real-world validation.

## Development

Use Python 3.11–3.13. From a fresh source checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade 'pip>=26.2,<27' 'setuptools>=84.1.0,<85'
pip install -e '.[dev,dashboard,audit]'
ruff format --check src scripts tests
ruff check src scripts tests
pytest -q
bandit -r src scripts -ll
python scripts/check_release.py
python -m build --outdir outputs/release
python -m twine check outputs/release/*
python scripts/check_release.py --artifacts outputs/release/*
```

Browser E2E instructions and opt-in screenshot capture are in [docs/E2E_TESTING.md](docs/E2E_TESTING.md). Public gallery changes must update the provenance manifest and README references; do not add real footage to the gallery.

Tests generate their own small synthetic fixtures; they do not require YOLO weights, cameras, Docker or paid services. Preview HTTP smoke check: `python scripts/check_preview.py`. PostgreSQL integration: create private `.env` with `scripts/init_local_env.py`, start Compose, then run `scripts/check_postgres.py`. Do not post `.env` or unredacted connection strings in issues.

Install CPU PyTorch and `.[vision]` only when working on real detection; follow README and the upstream model/dependency licenses. Any downloaded dataset, image or model needs independently documented permission. Keep footage, generated data, weights, model artifacts and local reports outside commits.

## Changes and review

Explain the concrete problem and resulting behavior, relevant tests, and any research limitations in the pull request. Add focused tests for changed calculations, validation or failure behavior. Avoid tests that merely repeat implementation. Use type hints and focused functions. Preserve causal feature construction and event/clip/location isolation in evaluation. Do not add runtime downloads, telemetry, raw-video uploads or recordings as defaults.

Use deterministic seeds for experiments. Report the complete protocol and failed experiments; never describe synthetic performance as real-world evidence. Security reports belong in the private channel described in SECURITY.md. Maintainers review contributions before merging; no CLA or additional copyright assignment is requested by this repository.

## Release review

Follow docs/OPEN_SOURCE_READINESS.md. Review rights to contributed material, package contents, and checks from a clean checkout. Scan existing Git history before publishing a fork that contains additional history. CI files are provided, but local checks do not prove a hosted workflow has executed.
