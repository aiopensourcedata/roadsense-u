# Local interaction annotation protocol

This is a pilot protocol drafted during development, not a preregistered validation study. It supports development screening and preparation of a small pilot. It does not establish traffic-safety ground truth from model scores. Bellevue is already a development camera; keep future external test locations untouched.

## Freeze selection before model comparison

Choose a prediction timestamp t, a two-second pre-context and a three-second future horizon. Use complete source media timestamps, not average FPS. Select additional disjoint windows with a fixed seed and without scores/outcome labels. Include known alert and regression windows separately and disclose this selection bias. Hash the selection manifest before annotation. The combined sample is not representative traffic exposure or independent event prevalence.

The supplied tool creates a score-hidden local HTML page, blank reviewer CSV, separate model-context JSON, and optional timestamped contact sheets. The assistant's screening notes are a separate file and are not loaded by the review page. Prior knowledge of development alerts means that assistant screening is not formally blinded. A second reviewer should use the blank page/sheet without opening model context or first-pass annotations, and disclose any prior exposure to development alerts.

## Record observations before judgments

For each window, record distinct physical users, visibility/occlusion, possible duplicates or identity changes, observed contact, and any apparent abrupt avoidance. Overlapping boxes do not prove contact, two physical users or a conflict. A detection ID is not a physical identity. Absence of visible contact does not rule out a conflict.

Screening categories are `apparently_ordinary_in_samples`, `requires_full_rate_review`, `invalid_pair_diagnostic`, and `censored`. A two-Hz contact sheet supports screening, not reliable detection of rapid braking or a validated negative label. Use full-rate source playback and inspect relevant actors at full resolution before final annotation.

A provisional research conflict candidate requires two distinct users on apparently converging paths plus a clear abrupt braking, swerving or stopping response to the other user's path within (t,t+3s]. An apparent response is a visual annotation judgment, not proof of driver intent or causation. This is a qualitative research definition needing trained-reviewer agreement, not a physical severity threshold or certified near-miss definition. If the response, actor distinctness, timing or visibility is unclear, leave the label unresolved. An apparent negative requires adequate visibility throughout the full future horizon and no qualifying response/contact; sampled-frame absence alone is insufficient for adjudication. Physical speed or deceleration cannot be established from uncalibrated pixels.

Missing footage, actor exit/occlusion, uncertain contact and ambiguous identity must be recorded. Timestamp continuity under a configurable tolerance does not prove visibility. Under the current conservative full-horizon protocol, known gaps censor both labels. Do not turn unresolved/censored observations into zeroes.

## Independent review and adjudication

1. Each reviewer records a stable reviewer ID, method, notes, provisional label and event onset/actors when a candidate is present. Do not copy the first reviewer's labels or infer labels from alerts.
2. Compare agreement, identify disagreements and document adjudication. Preserve original sheets. Never fabricate a second reviewer or agreement statistic.
3. Keep `adjudicated_label` blank until that process is completed. A first-pass provisional zero is not training-ready ground truth.
4. Window screening is not a pair-level training table. Identify canonical actors and onset times before creating baseline CSVs; never broadcast one window label to every detected pair.

Report review coverage, missingness, selection method, sample size, and counts of unresolved/censored windows. Do not claim precision, recall, missed-event rate or warning accuracy from unadjudicated screening. If no confirmed positives exist, recall is unavailable. Frames and overlapping alert windows are correlated; a camera/location-independent evaluation needs a larger separate dataset.

## Prepare a local review set

Install `.[review]` only when exporting contact sheets. Supply ffprobe JSON from the **same source**, plus the cached comparison report:

```bash
python scripts/prepare_video_review.py /path/to/video.mp4 \
  --timestamps outputs/source/ffprobe-timestamps.json \
  --comparison outputs/comparison/comparison.json \
  --output outputs/review-01 --count 12 --seed 42 \
  --write-review-frames
```

The explicit frame-export flag retains local sampled stills. No video clips are copied or uploaded. Open the generated `review.html` locally; it references the original file. If the browser blocks local media, use a local video player and the listed source timestamps. Labels saved by the page stay in browser local storage; export CSV to retain them independently. Censored windows cannot receive binary labels through the page. Review-file labels remain provisional; adjudication is separate.

Source videos, stills, filled annotation sheets, absolute file paths and model reports remain under ignored outputs. Do not publish footage or labels derived from restricted sources without verifying rights. Hashes freeze source/selection identity, not correctness of annotation or permissions.
