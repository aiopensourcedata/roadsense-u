# Project status

Updated: 2026-10-24. Version 0.1.0, research alpha.

## Implemented

- File, device, HTTP/RTSP input; optional YOLO11n/ByteTrack; synthetic fixtures.
- Trajectories, motion, manual ground-plane calibration, geometric rules, and quality abstention.
- Authenticated event API, PostgreSQL, exposure statistics, retention/deletion, and retry client.
- Local browser preview, event dashboard, and Docker Compose development stack.
- Offline calibrated logistic baseline and grouped evaluation.

## Verification

Python 3.12 local checks: 41 automated tests; Ruff formatting/lint; Bandit medium/high gate; source/wheel build, Twine, and release-content checks passed. Dependency advisory checks passed at audit time; they are time-sensitive.

The browser suite passed 16 checks with 12 computed synthetic alert emissions. It verified PostgreSQL IDs independently, API boundaries, replay, exposure normalization, dashboard views, mobile layout, and live preview. Public capture evidence: [manifest](assets/provenance.json). Generated execution reports remain under ignored `outputs/`.

YOLO/ByteTrack also ran on a repeated static-image smoke fixture. This verifies execution, not moving-traffic accuracy. Hosted CI results are reported by GitHub, separately from these local checks.

## Outstanding work

- Validate moving traffic, physical webcam/phone input, calibration error, and track stability.
- Obtain permitted labeled data and run held-out-location experiments before performance claims.
- Integrate the classical model online only after offline validation.
- Add migrations, stronger authorization, rate/read limits, and operational controls for remote deployments.

Physical footprints, automatic observed-zone PET, reliable ID-switch diagnosis, temporal networks, federated learning, messaging, and object storage are not implemented. The next research step is a manually reviewed, permitted fixed-camera traffic clip and labeled conflict targets.

## Local video pilot — 2026-10-26

Reviewed the existing `traffic.mp4` run: outputs end at 53.28 s with no final report; all 29 emitted event UUIDs independently matched local PostgreSQL. Sampled alert footage needs manual conflict annotation. Ran the first 900 frames (30 s) of `Weast.mp4`: 183 s elapsed, 4.92 FPS, 13 local diagnostic emissions; 1,165/18,609 pair observations passed quality checks. Eight artifact-consistency checks passed. Camera/background motion and unsupported three-wheelers limit this clip's risk interpretation. No new diagnostic events were submitted, no annotated video was recorded, and no footage was published. Detailed local evidence is under ignored `outputs/traffic-review/`. These are execution checks, not labeled accuracy results.

## Bellevue pilot — 2026-10-26

The complete downloaded Bellevue file decoded (7,418 frames); independent timestamp inspection found 36 gaps above 0.5 s, with a maximum of 60.535 s. CPU detection/tracking/rules ran on the first 1,303 frames through 55.801 s: 341 s elapsed, 3.82 FPS, five heuristic emissions, and 3,619/137,269 pair observations accepted (2.6%). Eight artifact checks passed. A sequentially extracted alert frame indicates duplicate tracks on one car despite accepted quality; three alerts have future-horizon timestamp gaps. Distortion remains uncorrected and ground truth unavailable. Outputs and visual evidence remain local under `outputs/bellevue-review-01/`; no events or footage were uploaded. Next: duplicate-pair diagnostics/regression cases and gap-aware annotation/evaluation before additional risk claims.

## Identity and coverage correction — 2026-10-26

Implemented a causal, pair-specific ambiguity guard without deleting tracks or altering geometric thresholds. Added complete frame timelines, retrospective event-horizon sidecars and explicit coverage filtering for real baseline CSVs. Cached ablation reproduced all five original Bellevue alerts, then removed the 0.943 duplicate-car emission: 5→4 alerts, 3,619→3,618 scoreable observations; 111 pair observations flagged ambiguous, mostly already rejected by quality. This is a development regression, not measured accuracy improvement.

Actual YOLO11n/ByteTrack also ran on 200 later source frames with the final guard: 44.4 s elapsed, 4.51 FPS, 127 accepted observations, 46 identity abstentions and three emissions, all retrospectively censored by timestamp gaps. A manually reviewed distinct-car pair retained nine scoreable observations and no identity abstentions. No previously scoreable observations were removed in this audit window. No footage/events were uploaded.

Final suite: 53 tests passed, including measured duplicate/distinct-object controls, opposing/fluctuating motion, genuine occlusion ambiguity, gap boundaries, censored labels, cached replay integrity and pipeline behavior. Ruff and Bandit medium/high checks passed. Detailed local evidence: `outputs/bellevue-identity-ablation-02/` and `outputs/bellevue-independent-review-02/`. Documentation: [identity and coverage](IDENTITY_AND_COVERAGE.md). Remaining limits include delayed confirmation, different-class duplicates, genuine occlusion abstention, no camera-motion/lens correction and no independent conflict labels.

## Annotation pilot — 2026-10-27

Prepared a frozen 17-window Bellevue review set (four retained alerts, one withheld regression and 12 additional disjoint continuous windows selected without scores/outcome labels). Visually screened all nominal 2 Hz contact sheets: 11 apparently ordinary, two requiring full-rate review, three censored and one invalid-pair diagnostic. First-pass provisional labels remain unadjudicated; no independent reviewer or validated conflict ground truth was invented.

Fresh YOLO11n/ByteTrack processed 1,824 selected frames in 370.8 s, resetting tracks per sampled window. At the 12 fixed timestamps, six windows had no scoreable pair and only 20/1,324 reported pairs were scoreable. Other windows had partial coverage; their low scores cannot establish whole-scene negatives. Ten emissions occurred throughout the sampled contexts; all have source-timestamp coverage, while nine cross unprocessed intervals in the sparse pipeline timeline. None has independently adjudicated outcomes. No accuracy/missed-event metric is claimed.

Added a deterministic selection helper, explicit frame-export opt-in, local score-hidden video review page, blank independent CSV, provenance hashes and [annotation protocol](ANNOTATION_PROTOCOL.md). Chrome verified local playback, all 17 windows, censored-label blocking, CSV export/formula escaping and no uncaught JS errors. Local evidence: `outputs/bellevue-annotation-01/`. Filled sheets, video references and real-media sheets remain ignored/private. Tests: 57 passing; Ruff and Bandit medium/high checks passed. Independent full-rate review and actor-specific adjudication remain the next research task.
