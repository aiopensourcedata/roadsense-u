# Pair identity and observation coverage

Two separate checks address failures found in the local Bellevue pilot. Risk thresholds remain unchanged. These are research safeguards, not validated duplicate classification or safety guarantees.

## Causal pair identity check

`detection/pair_identity.py` retains detections and IDs. For same-class pairs it measures sustained box overlap, normalized center separation, and agreement of relative box position over a trailing window. It abstains with `ambiguous_pair_identity` when the evidence cannot reliably establish two distinct objects. Other geometric features remain recorded.

Default `pair_identity` settings:

| Setting | Default | Meaning |
|---|---:|---|
| `enabled` | `true` | Enable pair-specific ambiguity abstention. |
| `min_iou` | 0.5 | Current and historical overlap threshold. |
| `max_center_separation` | 0.2 | Separation divided by mean box diagonal. |
| `window_s` | 0.5 | Trailing evidence window; at most 128 samples per candidate pair. |
| `min_span_s` / `min_samples` | 0.2 / 4 | Minimum temporal evidence. |
| `min_overlap_fraction` | 0.8 | Fraction of samples meeting the overlap threshold. |
| `max_offset_drift_per_s` | 0.25 | Maximum slope of normalized relative center position. |
| `max_offset_residual` | 0.05 | Maximum RMS residual around that relative-motion fit. |
| `max_gap_s` | 0.2 | Expire evidence after an observation gap. |

These are initial engineering heuristics, informed by a development failure, not optimized or validated parameters. The drift and residual checks avoid treating overlap alone as sufficient evidence. History resets after reconnection and expires when pairs disappear. No future frames are used. Geometry thresholds (`proximity_px`, horizon, quality threshold and alert threshold) are independent and unchanged.

Limitations:

- Distinct, persistently occluded objects with similar apparent motion may also be ambiguous. Coverage can fall; the guard does not prove duplication.
- There is a confirmation delay. Short duplicate tracks or early alerts may escape this check.
- Different-class duplicates are not handled, preserving common person/bicycle overlap but leaving class-confusion failures unresolved.
- The guard does not merge identities, repair tracks, or deduplicate every alert involving a suspected track and third objects.
- Perspective, camera movement, distorted lenses and genuinely incorrect motion estimates remain separate problems.

`features.jsonl` includes `pair_identity` measurements and abstention reasons. Reports count `identity_ambiguity_abstentions`; events produced with the enabled guard use `rules-0.1+identity-0.1`. Preview displays an unavailable maximum when no pair was scoreable, rather than presenting zero as a measured risk score.

## Retrospective future-horizon coverage

Every completed run now writes:

- `frame_timestamps.csv`: every processed frame, including frames without detections, plus timestamp source and discontinuity marker.
- `event_evaluation.jsonl`: event UUID, prediction time, horizon, `future_observation_complete`, and reason.
- `report.json`: segment count and complete/censored event-horizon counts.

`evaluation/coverage.py` forms observed segments split by reconnection or a timestamp gap exceeding `observation_max_gap_s` (default 0.5 seconds). A horizon is eligible only if it lies inside one segment. Gaps and truncated endings censor both positive and negative evaluation episodes under this conservative protocol. A prediction outside the observed timeline is ineligible too.

This is a continuity policy, not proof of visibility, complete capture, correct timestamps or correct labels. Choose the gap tolerance for the frame cadence and annotation task, freeze it before evaluation, and audit short missing intervals as well. A detected event before a later gap is still excluded under the present full-horizon protocol. Offline censoring never retrospectively changes a live score or a previously submitted event.

Recompute a sidecar from a complete frame timeline:

```bash
python scripts/assess_event_coverage.py \
  outputs/run/frame_timestamps.csv outputs/run/events.jsonl \
  --output outputs/coverage-review.jsonl
```

Do not substitute detection-only `trajectories.csv`: a missing detection does not mean the frame was missing. For an older run, recover a complete timeline from the source video and document its provenance. Do not invent timestamps from average FPS for variable-rate files.

### Training and held-out CSVs

Real-data observations now require `future_observation_complete` with explicit true/false values. Join coverage to the relevant prediction timestamp/clip; alert-only sidecars do not cover all non-alert training episodes. Optional `future_observation_reason` records why a row is censored (`reason` in the event sidecar can be renamed when joining).

Rows flagged false are excluded **before** target validation, including rows whose outcome is unknown. Reports retain exclusion counts and reasons. If all rows are excluded, fitting/evaluation fails. Coverage flags are a caller-supplied data contract and still require auditing; true does not replace conflict or visibility annotation. Synthetic legacy fixtures remain supported with coverage explicitly reported as not provided.

Do not fill censored labels with zero. Keep event/clip/location splits disjoint after filtering, and report how exclusions affect each camera and label population.

## Reproduce a cached comparison

For a reference run made before the guard or with `pair_identity.enabled: false`:

```bash
python scripts/compare_pair_identity.py outputs/reference \
  --timestamps outputs/reference/frame_timestamps.csv \
  --output outputs/identity-comparison
```

The tool checks that the old alert timestamps, pairs and scores are reproduced, applies the new identity check to the same boxes, retains the old geometric/quality scores, and writes `comparison.json` plus `ambiguities.jsonl`. It does not rerun YOLO or measure end-to-end inference FPS. Removing an alert can alter later cooldown emissions, so compare event timelines as well as counts. Outputs are not submitted to the API.

The numeric Bellevue fixture under `tests/fixtures/` contains measured boxes for a development duplicate case and a separate same-camera distinct-object control. It contains no video. Synthetic crossing, occlusion, fluctuating motion, gap and pipeline tests cover additional behavior. Neither these controls nor a smaller alert count establish real-world precision, recall or unseen-location generalization.
