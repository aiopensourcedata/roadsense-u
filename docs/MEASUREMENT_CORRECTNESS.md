# Measurement correctness update — 28 October 2026

This update fixes three correctness defects before additional perception tuning. It does not establish better real-world detection, participant attribution or warning accuracy.

## Causal motion discontinuity rejection

Incoming points are checked before entering the trailing motion fit. The innovation is the distance from the previous fitted position extrapolated using its velocity and the actual timestamp increment. The threshold uses only the previous state:

`max(unit_floor, 10 × prior_residual, 5 × prior_speed × dt)`.

The configurable constructor floors are **5 pixels** and **0.5 metres**. These are engineering tolerances, not validated uncertainty bounds or physically equivalent calibrations. A flagged observation clears the old motion history, seeds a new segment, has zero quality and cannot produce a feature-dependent alert. Fresh samples and at least 0.1 seconds of history are required before motion becomes available again. Gap and coordinate changes also reset history.

Tests cover a 100-pixel jump and its metre-scaled equivalent at 10/30/60 Hz, immediate abstention, recovery and smooth acceleration. This guard cannot establish identity correctness, detect every jump during initialization, or guarantee that legitimate abrupt motion is retained. Its real-data coverage cost still requires measurement. Do not loosen quality gates to offset the new abstentions.

## Shared runtime and general-benchmark eligibility

The general CSV benchmark calls the runtime `estimate` function rather than maintaining another geometric formula. The pipeline passes identity ambiguity to that same function. Quality, uncertainty, identity and nonpositive CPA-time behavior therefore share the implementation.

CSV fields `scoreable` and `identity_ambiguous`, when provided, must be explicit booleans; `measurement_uncertainty` must be finite and nonnegative for available measurements. A missing perception row must declare `scoreable=false`; feature values can then be missing. Older CSVs without uncertainty/identity fields explicitly assume zero residual uncertainty and unambiguous identity. That compatibility assumption is recorded in the report and does not establish measurement reliability.

Both feature-dependent baselines fit/predict only on runtime-eligible measurements. Unavailable test episodes remain in operating recall and negative-episode alert denominators. AP, Brier and reliability bins apply only to accepted predictions; reports label that scope and show positive/negative abstentions and coverage. Thresholds use accepted validation predictions; insufficient accepted classes or negative validation episodes raise an error rather than inventing a model/threshold. The held-out evaluation CLI uses the same prediction mask and operating metrics. Negative-episode fractions remain distinct from camera-hour false-alert rates.

## Fail-closed scene artifacts

The separate scene-pilot evaluator rejects unavailable rows with any rule score, nonboolean availability, invalid IDs, inconsistent clip/source/partition mapping and mismatched cohort declarations. It also masks rule predictions by availability, as already done for feature-dependent learned predictions. No-feature priors intentionally remain separate references.

The saved scene-pilot data contained no contradictory scores. Cached evaluation is replayed into a new output directory and its full report, including the bootstrap, matches the original exactly. Original source artifacts retain their SHA-256 hashes. This verifies evaluator compatibility only: historical motion features were not regenerated using the corrected jump guard. New inference must receive a new evidence version and cannot silently replace published/local historical results.

## Remaining priorities

Temporal clock-reset handling, interrupted-run finalization, nonblocking live delivery, calibration/run/event provenance and pinned vision integration remain open engineering work. Independent participant/outcome annotation, motion-reference validation, verified recording/site identities and normal-traffic exposure remain research blockers. Detector size and tiling are not priorities on the current evidence: neither improved positive development availability.
