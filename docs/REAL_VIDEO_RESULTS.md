# Bellevue video evaluation

Date: 2026-10-27. **The software path works; reliable conflict detection is not established.** A high-scoring duplicate-track alert remains a reproducible failure.

The [City of Bellevue dataset repository](https://github.com/City-of-Bellevue/TrafficVideoDataset) links the supplied Drive folder and permits research use with an as-is disclaimer. Its 21 MP4s total 6.67 GB and show one previously used intersection. No conflict annotations were listed. Videos and extracted frames remain in ignored local directories; unrestricted redistribution rights were not verified.

Four new recordings were selected by size/date before inference, totaling 146,047,991 verified bytes: the three smallest new files and the smallest September 10 file. This convenience sample is neither representative nor location-held-out. Configuration/model hashes were frozen; no thresholds were tuned.

| Recording suffix | Inference frames | CPU FPS | Scoreable pair observations | Alert emissions |
|---|---:|---:|---:|---:|
| `2017-09-10_22-08-50` | 400 | 5.62 | 0 / 0 | 0 |
| `2017-09-11_12-08-33` | 400 | 4.48 | 1,455 / 33,836 | 8 |
| `2017-09-11_15-08-36` | 400 | 4.28 | 1,084 / 67,384 | 4 |
| `2017-09-11_16-08-37` | 400 | 3.99 | 553 / 58,657 | 2 |

Actual YOLO11n/ByteTrack processed 1,600 frames, about 54 seconds of source timeline. Full files were separately decoded for timing checks: 19,863 frames returned. Full-file gaps above 0.5 seconds numbered 7/0/6/5 respectively; maximum 11.734 seconds. The 15:08 file produced a recoverable H.264 warning with unmeasured artifact impact. Full-file decode is not full-file inference.

Only **3,092/159,877 (1.93%)** correlated pair observations were scoreable. Most abstentions concern track quality. Thirteen emissions have complete processed three-second horizons; one is censored at the noon prefix end. These are not independent conflicts, exposure-normalized rates, recall or precision. Zero nighttime pairs cannot establish nighttime detection accuracy.

The noon maximum, score **0.767731** at PTS **11.566933 seconds**, pairs IDs **275/280** on the same vehicle at the lower-left boundary. The identity guard had only two overlapping observations (IoU 0.531314), below confirmation prerequisites, and emitted an alert. A single assistant visual screen supports this identity diagnosis; no independent conflict adjudication is claimed. This is a new failure beyond the earlier corrected duplicate example.

All 14 metadata records passed authenticated local API submission, exact HTTP retrieval and independent PostgreSQL payload comparison. UUID replay was idempotent; statistics retrieval passed. Request bodies totaled 15,670 bytes including one replay, excluding HTTP overhead. Chromium verified the dashboard's matching count and evidence view with no uncaught browser errors. No frames were uploaded. These checks establish integration even for incorrect predictions.

Independent verification checked all 1,600 timeline rows, 14 event schemas/coverage sidecars and recomputed all 3,092 accepted scores. All four completion reports were present; the outer batch shell subsequently returned 143, so a clean outer-shell exit is not claimed. The separate artifact verifier exited successfully. The automated suite passed 61 tests. Browser inspection also exposed a timezone display error; the chart now explicitly uses UTC and preserves the original ISO timestamp in tooltips. After rebuilding/restarting the local dashboard, Chromium produced identical chart captures in UTC, Europe/Budapest and America/Los_Angeles browser timezones. The 61-test suite and Ruff passed after this change.

Local evidence: `outputs/drive-evaluation-01/` contains frozen selection/hashes, source audits, per-run exports, backend/browser verification, private screenshots and `REPORT.md` with reproduction commands. `independent-review.csv` has 14 candidates plus four fixed-frame controls with blank adjudication fields. Sampled images were screened; continuous playback and independent review were not claimed. API timestamps are ingestion-relative, not verified 2017 capture times.

Next: address early duplicate/edge-track reliability and low coverage on development data, independently label actors/outcomes, validate geometry, and test untouched physical sites against frozen acceptance criteria. No real-world accuracy, crash-prediction, generalization or novelty claim follows from this run.
