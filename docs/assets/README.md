# README screenshot gallery

All five PNGs are actual Chrome/Chromium browser captures produced by `scripts/run_e2e.py`. They are not mockups, generated images or illustrations. The viewed data and live video are deliberately synthetic. No participant footage, identifying images, API keys or downloaded detector example images are included.

| Screenshot | What the browser displayed |
|---|---|
| `overview.png` | Computed alerts retrieved from PostgreSQL, scores and heuristic track quality for two fictional cameras. |
| `event-evidence.png` | One stored alert's measured CPA, time to CPA, motion, residual indicator and rule explanations. |
| `location-analytics.png` | Alert counts and normalization using known synthetic agent counts in matching exposure windows. |
| `evaluation.png` | Results of the executed, separate synthetic tabular baseline experiment and its disjoint partitions. |
| `live-preview.png` | An annotated frame while the generated crossing clip was actively being processed and served via local MJPEG. |

`provenance.json` records the successful capture run, explicit check outcomes, browser/Python versions, synthetic provenance and SHA256 hashes. Every PNG is referenced in the project README; the documentation test checks that the set and hashes agree. Hashes ensure gallery/manifest consistency, not pixel-identical reproduction across runs. Timestamps, generated IDs and rendering can change.

To refresh the gallery from the running local stack:

```bash
python scripts/run_e2e.py --screenshots docs/assets
pytest -q
```

This is an explicit opt-in to storing synthetic browser screenshots. It is separate from pipeline footage recording, which remains disabled. The source manifest and release checker allow these five reviewed PNGs explicitly; raw recordings, datasets, weights and local outputs remain excluded.
