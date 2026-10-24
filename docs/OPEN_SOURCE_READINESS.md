# Release checklist

Version 0.1.0 — research alpha. Last local review: 2026-10-24.

## Included checks

- Tests, formatting, static security checks, and dependency advisory review.
- Wheel/source builds, Twine validation, and archive-content inspection.
- PostgreSQL/API/dashboard/live-preview integration using synthetic fixtures.
- Source allowlist and credential-signature checks with `scripts/check_release.py`.
- Five reviewed synthetic browser captures with hashes and README references.

See [project status](PROGRESS.md) for results and [E2E testing](E2E_TESTING.md) for reproduction. A passing check is not proof of security, real-world accuracy, or license ownership.

## Before each publication

1. Confirm rights to all contributed material and compatibility with AGPL-3.0-only and third-party terms.
2. Run `python scripts/check_release.py --require-git` from the actual publishing checkout. Review staged files and archives. For existing history, use a dedicated history secret scanner and rotate exposed credentials.
3. Run tests and check hosted CI. Review dependency advisories again when dependencies change.
4. Keep `.env`, datasets, recordings, model weights, local environments, and generated reports private. Only the five allowlisted synthetic documentation PNGs belong in the gallery.
5. Enable private vulnerability reporting and review repository permissions. Never publish production credentials in issues or workflow logs.

The initial publication uses a fresh source-only Git repository because the development workspace has no usable original Git history. `scripts/prepare_public_snapshot.py` copies only allowlisted source files; it does not certify or migrate any previous history.

## Deployment and research limits

The local stack uses a shared API key, prototype schema creation, and loopback-only viewers. Remote hosting needs HTTPS, migrations, authorization boundaries, rate/read limits, scheduled retention, backups, and monitoring. No independent container OS audit or production certification has been performed.

Rule scores are uncalibrated heuristics. The classical model is offline only. Synthetic checks do not validate crash prediction, cross-location generalization, physical speed accuracy, causal hotspots, or novelty. See the [experiment plan](EXPERIMENT_PLAN.md).
