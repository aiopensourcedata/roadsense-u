# Actor-level development review

Scene collision timestamps do not identify the visible actors, detector misses, or incorrect track assignments. This workflow prepares actor-box annotations separately from the scene-risk experiment. It uses inspected training sources only. No actor accuracy result exists until the review is complete.

## Executed preparation — 2026-10-27

The local bundle is `outputs/actor-review-01/index.html`. It contains **29 unique sampled frames** covering ten eligible snapshots and causal context approximately 0.5 and 1 second earlier. Repeated source frames are deduplicated. The snapshots are the same four training clips used in the resolution diagnostic, not a new evaluation cohort.

Independent detector-only YOLO instances ran on all 29 sampled frames at 640 and 960: **58 single-frame inferences**. Existing ByteTrack outputs were reused from the completed comparison. Raw detection uses the same local weights, confidence, classes and device, without tracker callbacks. No risk thresholds, quality gates or tracker settings were changed.

| Model-output count at ten scheduled snapshots | 640 | 960 |
|---|---:|---:|
| At least two raw detector boxes | 6 | 5 |
| At least two exported tracked boxes | 1 | 1 |
| Raw pair available, tracked pair absent | 5 | 4 |

These are counts of model boxes, **not distinct physical actors or missed conflicts**. Provisional assistant visual screening found examples with one prominent car, blurred visibility, apparent duplicate boxes and apparent roadside-sign detections. These notes are neither exhaustive annotations nor independent ground truth. Low-confidence proposals cannot be assumed correct; simply lowering tracker thresholds is not supported by this evidence.

The page/template deliberately contain no detector boxes, risk scores, published outcomes or pre-filled actor labels. Prediction records are in a separate file. This is procedural blinding, not an access-control boundary: reviewers must not inspect `predictions.json`, diagnostic reports or prior screening notes while annotating.

Chromium loaded all 29 real images at their expected original dimensions with zero HTTP(S) page requests or JavaScript errors. Synthetic browser tests verified drawing normalized boxes, explicit completion, JSON export and resume. Attempting to evaluate the actual blank review template failed without producing metrics.

## Prepare a bundle

From the project root, using the existing private experiment artifacts:

```bash
python scripts/prepare_actor_review.py \
  --development outputs/perception-development-01 \
  --downloads outputs/labelled-pilot-03/downloads.json \
  --audits outputs/labelled-pilot-03 \
  --output outputs/actor-review-new \
  --write-review-frames --raw-detections
```

Choose a new output directory for each preparation. The command refuses non-training partitions, changed source/weight bytes, incomplete comparisons and mismatched detector configuration or source timing. `diagnostic.json` has `status: complete` only after preparation succeeds. An incomplete folder is not a completed study artifact.

Both optional flags are explicit. Without `--write-review-frames`, no JPEGs are retained and the annotation page explains that images are unavailable. Without `--raw-detections`, raw-model counts are unavailable rather than zero; cached track outputs can still be reviewed. Missing dependencies/models fail visibly.

## Annotate independently

Open `outputs/actor-review-01/index.html` in a browser; it works directly from a local file and needs no server. Give separate copies of the blank bundle to at least two reviewers. Use temporary reviewer identifiers, without personal information.

1. Enter a reviewer ID. Draw boxes around the **visible extent** of supported road users. Choose car, motorcycle, bus, truck, bicycle, person or unknown.
2. Assign temporary actor IDs such as `car-A`, consistent across frames of the same clip. IDs have no meaning across clips and do not identify people.
3. Mark physical objects, depictions/reflections, and uncertain objects separately. Choose clear, partial or uncertain visibility. Tiny or ambiguous objects should not be forced into a confident class.
4. Inspect the entire frame before marking it reviewed. An empty actor list is valid only after explicit review. Drawing/removing a box resets that completion flag.
5. Export JSON before closing the page. There is no automatic persistence, upload or browser local storage. Exported drafts can be resumed with the file input.

The page's exports are always `first_pass` / `human_visual`. An assistant visual screen must instead declare `assistant_visual_screen`; synthetic fixtures must declare `synthetic_test`. Neither qualifies as independently adjudicated labels.

Compare independent reviews and resolve disagreements with the original footage. Create a final JSON using the same manifest/frame keys, reviewed boxes and stable local actor IDs. Only after actual adjudication, set `review_stage` to `adjudicated`, provide at least two distinct `reviewer_ids`, and provide `adjudicator_id`. The software validates these declarations; it cannot establish reviewer independence or verify identity. It does not automatically average boxes or fabricate consensus.

## Evaluate completed annotations

```bash
python scripts/evaluate_actor_review.py \
  --bundle outputs/actor-review-01 \
  --review /path/to/adjudicated-actor-review.json \
  --output outputs/actor-evaluation-01/actor-metrics.json
```

Exploratory first-pass scoring requires `--allow-first-pass`, and the output retains that provenance. Blank/partial reviews, duplicate actor/frame IDs, invalid boxes, missing prediction frames and mismatched manifest hashes are rejected. No real actor metrics were computed during preparation.

Matching is class-aware at IoU ≥ 0.5, maximizing valid match count before summed IoU. Boxes are normalized against original image dimensions. Unmatched predictions overlapping an uncertain region at IoU ≥ 0.5 are neutralized. Depictions are negatives; they are not physical actors or ordinary ignore regions. Partially visible physical objects with a resolved class remain eligible; uncertain visibility/class is ignored explicitly. Document any changes to this policy before scoring.

The report compares detector-only and tracked-box precision/recall on the **same sampled frames** and retains per-frame matches/track IDs for investigation. These are correlated development observations, not COCO AP, continuous MOT accuracy, ID-switch counts, collision prediction accuracy or camera-held-out performance. The preceding context supports manual continuity review but does not provide dense tracking truth. Actual colliding-participant labels require separate temporal/outcome review; they cannot be inferred from these boxes alone.

## Privacy and evidence

The executed bundle explicitly retained 29 local JPEGs (approximately 9 MB together with metadata), plus a private browser screenshot. They remain ignored by Git. Do not redistribute source-derived images without checking dataset terms, rights and privacy. Delete sampled images and review screenshots once their justified review purpose ends; keep only necessary de-identified measurements/annotations. Ordinary pipeline recording/upload defaults remain disabled. No API event submissions, raw-video uploads or public screenshot changes were made in this step.

Local evidence: `manifest.json`, `provenance.json`, `predictions.json`, `diagnostic.json`, `review-template.json` and `verification.json` under `outputs/actor-review-01/`. Completed scene-risk results and the earlier resolution comparison remain unchanged.
