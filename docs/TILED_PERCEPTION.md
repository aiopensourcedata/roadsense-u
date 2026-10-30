# Experimental tiled perception

Tiled inference is an opt-in candidate for investigating missing/distant road-user measurements. It does not change the risk formula, quality threshold, default detector behavior or physical-unit assumptions. No improved actor accuracy or warning reliability is established by adding it.

## Behavior

```bash
python scripts/run_video.py /path/to/traffic.mp4 \
  --tiled --inference-size 640 --output outputs/tiled-run-01
```

The default is still full-frame YOLO/ByteTrack. `PipelineConfig.tiling.enabled` defaults to false. With tiling enabled, one full-image view and four overlapping crops use the same local weights, detector confidence and inference size. Crop coordinates are translated into original-image pixels before merging and tracking. No ground-truth impact box, manual ROI or future frame selects a crop. Recording remains opt-in.

The configurable two-by-two grid uses 20% overlap by default. Class-agnostic non-maximum suppression removes competing boxes above IoU 0.6; thresholds are configurable through YAML. One upstream ByteTrack instance receives all merged proposals once per source frame, retaining its existing default thresholds. Raw detections are never assigned fabricated track IDs. `perception_diagnostics` in `report.json` counts actual model calls, proposals and suppression.

`tiling.grid: 1` is a full-frame tracker-interface control: it preserves detector ordering and the detector's original NMS. It exists to check whether the explicit tracker interface changes the baseline before attributing a result to crops.

Costs/limitations include extra model calls (usually five per frame), crop-border truncation, competing class predictions, remaining seam duplicates, and suppression of genuine overlapping actors. Confidence changes across crops are not calibrated accuracy gains. Perspective and pixel-scale limitations remain. The synthetic detector cannot be combined with this option.

## Development comparison

```bash
python scripts/run_tiled_development.py \
  --development outputs/perception-development-01 \
  --downloads outputs/labelled-pilot-03/downloads.json \
  --audits outputs/labelled-pilot-03 \
  --output outputs/tiled-development-new
```

The fixed protocol uses the four already-inspected training sources and source PTS through 7.25 seconds. It records a local frozen plan and code/input hashes before inference. It first checks the full-frame tracker bridge against immutable pair records on clip 12; a mismatch stops the candidate comparison. It then measures all four candidate prefixes with the same scheduled timestamps, risk gates and future-observation checks as the baseline. It does not search thresholds on the completed test cohort.

`results.json` may be partial during execution. `summary.json` is written only after completion. Inspect per-source coverage and both positive/negative availability; neither additional raw boxes nor additional emissions are accuracy or false-alarm improvements. Cached baseline runtime is not a controlled timing comparator. Actor truth is still required before recommending a production change.

Emission counts cover the whole processed prefix, which may include frames after the published onset. They are not counts of correct pre-event warnings. Scheduled scene snapshots exclude post-onset times; a separate timestamp audit is needed to distinguish pre/post-onset emissions.

## Compare on exactly the same actor annotations

After candidate processing completes, an additional prediction file can reuse the existing 29-frame review manifest without copying raw images, changing its labels, or revealing model outputs in the blind editor:

```bash
python scripts/extend_actor_predictions.py \
  --bundle outputs/actor-review-01 \
  --development outputs/tiled-development-new \
  --downloads outputs/labelled-pilot-03/downloads.json \
  --output outputs/tiled-development-new/actor-predictions.json

python scripts/evaluate_actor_review.py \
  --bundle outputs/actor-review-01 \
  --review /path/to/adjudicated-actor-review.json \
  --predictions outputs/tiled-development-new/actor-predictions.json \
  --output outputs/actor-tiled-comparison/actor-metrics.json
```

The extension retains all four original stages and adds raw/tracked tiled outputs. Each stage must cover exactly the same source-bound frame keys. Detector-only diagnostics decode the original videos, rather than re-detecting lossy review JPEGs. Blank/partial reviews still fail; no automatic labels or performance numbers are supplied.

## Prior work

Sliced inference is established work: [Akyon, Altinuc and Temizel, 2022](https://arxiv.org/abs/2202.06934); [official Ultralytics integration guide](https://docs.ultralytics.com/guides/sahi-tiled-inference/). This implementation is a small full-frame-plus-crop/NMS adapter, not a reproduction of all SAHI merging/fine-tuning features and not a novelty claim. Any paper experiments and figures belong in the private `.research/` workspace.

## Executed development result — 2026-10-27

The full-frame bridge reproduced all 1,475 prior pair records/decisions on the control clip. The candidate processed 519 source frames with 2,595 detector requests; the additional bridge processed 218 frames. All complete frame timelines match the cached baseline.

| Profile | Usable / 10 eligible | Usable positive / 4 | Usable negative / 6 | Prefix emissions |
|---|---:|---:|---:|---:|
| YOLO11n, 640 | 1 | 1 | 0 | 1 |
| YOLO11n, 960 | 1 | 1 | 0 | 1 |
| YOLO11n, tiled 640 | 1 | 1 | 0 | 7 |
| YOLO11s, 640 | 2 | 1 | 1 | 1 |

Tiling did not improve scheduled coverage. Of its seven emissions, five precede the published onset and two occur at/after it; these are not seven correct early warnings. The larger detector added one usable **negative** horizon. Positive availability remains 1/4 across every profile. No independent actor or warning-accuracy gain is established. Both candidates remain opt-in; default settings are unchanged.

Local evidence: `outputs/tiled-development-01/` and `outputs/capacity-development-01/`. The combined `actor-comparison-predictions.json` has eight stages over the original 29 review frames. No additional raw images were retained. Original labels and completed scene-test results remain unchanged.

## Reproduce the larger-checkpoint comparator

```bash
python scripts/download_weights.py --model yolo11s
python scripts/compare_detector_resolution.py \
  --plan outputs/labelled-pilot-03/frozen-plan.json \
  --downloads outputs/labelled-pilot-03/downloads.json \
  --audits outputs/labelled-pilot-03 --sizes 640 \
  --weights data/yolo11s.pt \
  --weights-sha256 85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5 \
  --output outputs/capacity-development-new
```

The official checkpoint is 19,313,732 bytes. The downloader checks disk space, limits reads and pins the initially reviewed HTTPS download hash; it is not an independent publisher signature. Existing checkpoints are not overwritten. Ordinary video use can select `--config configs/detector_capacity.yaml`; this does not change `configs/default.yaml`.

For an identical-frame actor comparison, first prepare a metadata-only capacity bundle using `prepare_actor_review.py --development outputs/capacity-development-new` with the same downloads/audits and `--raw-detections` (without `--write-review-frames`). Pass that output folder as `--capacity-bundle` to `extend_actor_predictions.py`. The binding permits different image presentation paths only when all source identities, dimensions and timestamps match exactly. It never transfers or generates labels. Separate detector confidence values are not automatically calibrated to equal accuracy across checkpoints.
