# Perception and participant-association development

The published-label scene experiment exposed missing usable measurements and alerts involving unverified actors. This development step investigates those failures. It does not establish detection accuracy, correct-actor warnings, camera generalization or a publishable improvement.

## Implemented behavior

- `PipelineConfig.inference_size`: default 640, multiples of 32 from 320 to 1920. `run_video.py` and `run_webcam.py` accept `--inference-size`. The value is passed to YOLO/ByteTrack as `imgsz`; exported boxes and motion coordinates remain in the original image coordinate system. The byte content of the frozen default YAML is unchanged.
- Pair records now contain `projection`: current anchors, velocities, projected anchors at clipped closest approach, separation, midpoint, units, track IDs, scoring eligibility and reliability reasons. Projection uses past/current motion only. Pixel anchors are bottom centers of detector boxes, not measured physical contact points. Coordinates are approximate under perspective; metric output still requires valid ground-plane calibration.
- `event_geometry.jsonl` links those measurements to emitted event UUIDs. This is a local metadata sidecar; the authenticated cloud event schema and submission payload are unchanged. No extra footage is recorded or uploaded.
- The dashboard discovers nested completed pipeline runs and plots measured versus projected anchors. Every pair is explicitly unverified. A midpoint does not identify a collision location. Actor correctness still needs manual annotation and adjudication.
- `evaluation.perception.perception_summary` counts tracked observations, confidence/quality gate coverage, class observations, track-history flags and pair abstention reasons. Counts are correlated; track IDs do not measure physical traffic volume, detector recall or ID-switch accuracy.

## Reproduce the paired development comparison

With the locally acquired ACCIDENT sources and frozen manifests from the completed pilot:

```bash
python scripts/compare_detector_resolution.py \
  --plan outputs/labelled-pilot-03/frozen-plan.json \
  --downloads outputs/labelled-pilot-03/downloads.json \
  --audits outputs/labelled-pilot-03 \
  --indices 0 4 8 12 --sizes 640 960 \
  --output outputs/perception-development-01
```

These ignored artifacts/media are not distributed with the code. Dataset acquisition remains subject to its CC BY-NC-SA 4.0 terms; see [datasets](DATASETS.md). No additional download is required for the existing local experiment. A different installation can run the ordinary video command on its own permitted input:

```bash
python scripts/run_video.py /path/to/traffic.mp4 \
  --inference-size 960 --preview --output outputs/resolution-960-01
```

The comparison refuses non-training partitions and changed source/configuration/weight hashes. It writes a development plan before inference, alternates resolution order between clips, runs one worker and processes original media PTS through 7.25 seconds. Independent trackers are initialized for each clip/size. The scheduled snapshots are 1, 4 and 7 seconds, using existing source-binding and future-observation checks. Published onset times only define eligible diagnostic snapshots; impact regions never enter detection, pair selection or risk scoring. Risk thresholds and quality gates stay fixed.

`results.json` records each completed run, its scheduled snapshots, descriptive perception counts and runtime. `summary.json` is written only when the selected cohort completes and identical scheduled timestamps are verified across sizes. Interrupted runs remain incomplete; do not infer full-cohort results from a partial file. Re-running identical arguments resumes completed reports. Runtime includes initialization/decoding/export and can be affected by other local processes; this small sequential experiment is not a controlled deployment latency benchmark.

## Observed development results — 2026-10-27

Four previously inspected training clips, one per published collision type; 519 original frames per resolution, 1,038 actual YOLO/ByteTrack frames total. No newly held-out clips were used. All scheduled prediction timestamps match exactly between sizes.

| Descriptive measurement | 640 | 960 |
|---|---:|---:|
| Eligible scheduled snapshots | 10 | 10 |
| Scoreable snapshots | 1 | 1 |
| Snapshots with no tracked pair | 9 | 9 |
| Tracked object observations | 897 | 947 |
| Observations passing quality gate | 486 | 533 |
| Scoreable pair observations | 279 / 1,476 | 337 / 1,410 |
| Emitted heuristic events | 1 | 1 |
| Total measured run time | 189.03 s | 255.01 s |

Higher resolution changes tracker output but does not improve scheduled snapshot coverage here. The nine other eligible snapshots lack any tracked pair, rather than having existing pairs rejected by the quality gate. This does not prove nine missed conflicts: negative snapshots, actor visibility, scene-level labels and the lack of actor truth prevent that interpretation. Lowering the risk quality gate cannot create a missing tracked pair. Between scheduled snapshots, insufficient tracking quality accounts for 1,197 / 1,073 pair abstentions at 640 / 960 respectively.

The 640 runs reproduce **all 1,476 existing pair records and risk decisions exactly** over matching prefixes of the immutable prior caches, after removing the new projection field. Thus the projection export and explicit default inference size did not alter the recorded baseline decisions on these clips. All emitted UUIDs match their geometry sidecars; projected separation matches exported CPA distance. The cloud client sent zero bytes, and no annotated videos were recorded. Runtime was affected by concurrent test/build activity and is descriptive, not an isolated speed comparison.

Detailed local evidence: `outputs/perception-development-01/development-plan.json`, `results.json`, `summary.json`, `detailed-summary.json`, `verification.json`, `implementation-provenance.json`, and `dashboard-verification.json`. Chromium verified the actual Docker dashboard's geometry view; the private screenshot contains metadata/geometry, not raw footage.

## Interpretation and next experiment

The subsequent [actor-review workflow](ACTOR_REVIEW.md) now separates detector-only proposals from cached track outputs and prepares explicit manual annotation without inventing actor ground truth.

More scoreable observations could include more wrong detections or wrong participant pairs. Fewer alerts could hide genuine interactions. Neither change proves improvement without actor-level truth. All 28 previously evaluated source videos are now considered inspected for subsequent development; a future performance comparison must use fresh frozen sources, ideally verified separate cameras, with actor and outcome adjudication. The 960 setting stays opt-in regardless of this development result.
