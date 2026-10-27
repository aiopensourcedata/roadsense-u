# Project status

Updated: 2026-10-27. Version 0.1.0, research alpha.

## Latest development — actor review and detector/tracker separation

Prepared a score-blind actor-box editor, provenance-validated annotation schemas and a fail-closed sampled-box evaluator. Matching is class-aware at IoU ≥ 0.5 with maximum valid match count; uncertainty/depictions are handled explicitly. First-pass results require explicit opt-in, and declared adjudication requires two human reviewers and an adjudicator ID. These declarations do not establish reviewer independence.

Prepared 29 unique original-image review frames for ten eligible snapshots plus causal context from the four training clips. Ran 58 independent detector-only inferences at 640/960, reusing immutable track caches. At 640, six snapshots have ≥2 raw boxes versus one with ≥2 tracked boxes (960: five versus one). Raw counts include apparent false/duplicate proposals in provisional assistant screening and are not actor accuracy. Formal annotations remain blank; no actor precision/recall or participant-warning improvement is claimed.

Chromium verified all 29 real images, no HTTP(S) page requests, blank labels and evaluation rejection of the empty template. Synthetic browser tests verified box drawing/export/resume. Full suite: 96 tests passed, with the existing upstream Starlette warning; Ruff and Bandit checks passed. The final source-content guard passed for 150 files. Explicitly retained local sampled JPEGs/screenshot are ignored/private; no raw-video upload, new API events, commit or push. [Review protocol and commands](ACTOR_REVIEW.md). Next: two independent reviews and adjudication, then separate detector misses from tracker assignment losses before changing production settings.

## Latest development — perception diagnostics

Added configurable YOLO inference size (default 640), original-coordinate pair projections, event-linked local geometry sidecars, nested run discovery and a dashboard inspection chart with unverified-participant labels. The cloud payload/schema, frozen default YAML, risk formula and quality gates are unchanged. Implemented a training-only paired resolution runner and descriptive perception/abstention counts.

Executed 1,038 real inference frames on four inspected training clips at 640/960. Both sizes score only one of ten eligible snapshots; nine have no tracked pair. Higher resolution does not resolve this sample's snapshot coverage failure. All 1,476 baseline-640 pair records/decisions exactly reproduce prior caches. No raw recordings, uploads or new cloud events. Chromium verified actual Docker geometry inspection. Detailed evidence is in `outputs/perception-development-01/`; [protocol and results](PERCEPTION_DEVELOPMENT.md).

87 automated tests passed, including the Streamlit geometry-panel test; Ruff lint/format, Bandit medium/high and the 144-file source-content guard passed. Corrected existing formatting failures required by CI. Actor accuracy, near-miss performance and cross-camera generalization remain unvalidated. No commit/push or paper draft. Next: independently annotate visible actors, detections and track continuity on development clips before selecting a perception or association change; reserve fresh sources for evaluation.

## Implemented

- File, device, HTTP/RTSP input; optional YOLO11n/ByteTrack; synthetic fixtures.
- Trajectories, motion, manual ground-plane calibration, geometric rules, and quality abstention.
- Authenticated event API, PostgreSQL, exposure statistics, retention/deletion, and retry client.
- Local browser preview, event dashboard, and Docker Compose development stack.
- Offline calibrated logistic baseline and grouped evaluation.

## Verification

Python 3.12 current local checks: 61 automated tests; Ruff formatting/lint; Bandit medium/high gate; source/wheel build, Twine, and release-content checks passed. Dependency advisory checks passed at audit time; they are time-sensitive.

The browser suite passed 16 checks with 12 computed synthetic alert emissions. It verified PostgreSQL IDs independently, API boundaries, replay, exposure normalization, dashboard views, mobile layout, and live preview. Public capture evidence: [manifest](assets/provenance.json). Generated execution reports remain under ignored `outputs/`.

YOLO/ByteTrack ran on a static-image fixture, a physical laptop camera and moving-traffic videos. These verify execution; labeled perception/conflict accuracy remains unmeasured. Hosted CI results are reported by GitHub, separately from these local checks.

## Outstanding work

- Quantify moving-traffic/laptop-camera accuracy, calibration error and track stability; test USB/phone inputs.
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

## Research-readiness audit — 2026-10-27

Added a fail-closed research-input audit, an unchanged-formula synthetic motion-quality diagnostic and a draft study contract. The current evidence audit finds zero admissible adjudicated labels and no pair-level observation table, so research-performance claims remain unsupported. Matched-noise controls (100 independent tracks/scenario) show 0.826 px/s error at both stationary/moving speeds but 1.88% vs 100% scoring coverage. The real frozen-point audit attributes 1,088/1,324 pairs to low detector confidence, 145 to insufficient history, 66 to fit penalties, five to identity ambiguity and 20 to scoreable observations. These are diagnostics, not accuracy estimates.

Expanded primary-source review with ACCIDENT and USDOT metadata. Inspected only a bounded 9.7 MB documentation archive: USDOT validation has 41 runs at one site and run-level TTC-based labels; the sample cannot automatically validate our temporal/cross-site target. No traffic archive was downloaded or access form submitted. See [validation status](VALIDATION_STATUS.md).

61 tests, Ruff and Bandit medium/high checks passed. Fresh synthetic E2E passed all 16 checks (`outputs/e2e-aaaf6b53/results.json`) after correcting an eager dashboard metric read to wait for the verified count. No published screenshots changed. Local detailed audit: `outputs/research-readiness/REPORT.md`. New diagnostics are not yet pushed. Independent labels, external sites, operating budgets and matched real-data comparisons are still required.

## Four additional Bellevue recordings — 2026-10-27

Verified the supplied folder against the City of Bellevue's official research dataset. Inventoried 21 files (6.67 GB); downloaded four new files (146 MB), selected by size/date before inference. All share the existing development intersection. Full-file timing scans returned 19,863 frames, with source gaps and one recovered H.264 warning recorded. Frozen YOLO11n/ByteTrack/rules processed 400 frames per file: 1,600 frames, 14 heuristic emissions, 3.99–5.62 CPU FPS. Only 3,092/159,877 correlated pair observations were scoreable (1.93%). Thirteen emissions have complete processed horizons; one is prefix-end censored.

Found a remaining failure: noon IDs 275/280 cover the same edge-of-image vehicle yet emit score 0.768 before the identity guard accumulates sufficient evidence. No threshold was changed. No independent labels or external-location validation were obtained. Blank review sheet and private visual evidence were prepared.

All 14 actual computed metadata records matched HTTP retrieval and independent PostgreSQL queries; replay was idempotent. Chromium verified the 14-record dashboard/evidence view. No frames were uploaded or published. Browser inspection found a UTC-axis/local-time mismatch, corrected in the chart configuration. Results and execution limits: [real-video evaluation](REAL_VIDEO_RESULTS.md), local `outputs/drive-evaluation-01/REPORT.md`. Next: independently annotate actors/outcomes, address early duplicate/quality failures on development data, and preserve other physical sites for evaluation.

UTC display correction verified after rebuilding/restarting the dashboard: identical chart captures across UTC, Budapest and Los Angeles browser timezones. Final checks: 61 tests and Ruff passed; release content guard passed for 123 source files. No commit or push was made for this evaluation.

## Paired development experiments — 2026-10-27

Ran 2,400 additional real YOLO11n/ByteTrack frames (four disjoint windows, source frames 600–1199). Compared five fixed causal reliability policies across nine cached runs: 4,200 frames / 406,352 correlated pair observations, all reference scores/decisions/emissions reproduced. Pending-overlap abstention removes the known development duplicate emission but leaves all four later-window emissions unchanged; the latter include two source-gap-censored horizons. Three provisional distinct-car controls retain 85/85 scoreable observations with pending abstention versus 62/85 with a half-second history requirement. A deterministic synthetic distinct-crossing control loses two of 21 steps under pending abstention. Production risk policy was not changed.

Counterfactual coordinate scaling changes 206/81 threshold decisions at 0.5x/2x with fixed pixel proximity; proportional normalization preserves decisions algebraically. This is measurement sensitivity, not resized-video validation or improved accuracy. Four source-video review pages / 12 windows were prepared with blank labels; Chromium verified playback, score hiding, CSV export and censoring safeguards. Other physical locations remain untouched. No independent labels or real supervised benchmark exist yet.

Added reproducible reliability/scale CLI experiments and sequential `--start-frame` reading with preserved variable media PTS. Final checks: 70 tests, Ruff, Bandit medium/high and 128-file source-content guard passed. Local run log, frozen manifests, numeric results, plots and review index: `outputs/experiments-20261027-01/`. No manuscript, commit or push. Next required input: independent actor/outcome review and adjudication before accuracy/generalization experiments.

## Published-label collision experiments — 2026-10-27

Verified public ACCIDENT metadata and CC BY-NC-SA 4.0 terms; fetched original annotations and 16 bounded clips (47.4 MB) without authentication or gate bypass. Bound labels to full decoded frame PTS, recording the consistent count/last-index discrepancy and a pre-inference protocol amendment. First source-separated pilot processed 3,047 frames: four test collisions / nine eligible snapshots, 22.2% measurement coverage, one scene-time hit and one false alert. Its only temporal hit selected the opposite carriageway in a post-evaluation case check. Training lacked scoreable negatives; no classifier result was invented.

Retired the inspected cohort into development roles and froze 12 new source IDs (31.9 MB): four validation and eight test. Reused immutable feature caches for 12 training / four calibration sources; actual logistic fitting used ten train observations (six positive/four negative) and four calibration observations (two/two). Parameters exactly match the fingerprint saved before fresh test evaluation. Added 1,730 real inference frames; 4,777 unique inferred/cached frames across 28 labeled sources.

Fresh test: 18 eligible snapshots, six scoreable. Fixed rules and fixed-0.5 logistic each yield one temporal scene hit out of eight collisions; logistic also yields one false alert. Validation-selected logistic issues no test warnings. Accepted-only AP/Brier differences use only six observations and establish no learned warning improvement. Training-range abstention changes no predictions. Both temporal hits project outside the published impact regions; correct colliding actors are not validated. These are source-video splits and scene-collision labels, not independent camera locations or pair near-miss truth.

Added source binding/snapshot extraction, leakage/abstention metrics, frozen source evaluation CLI, parameter provenance/model cards, and dashboard support for nested scene reports. Final automated suite: 80 passing tests; Ruff, Bandit medium/high and source-content checks passed. Source media, model files and detailed numeric evidence remain ignored/private under `outputs/labelled-pilot-02/` and `outputs/labelled-pilot-03/`. [Experimental summary](LABELLED_COLLISION_PILOT.md). No manuscript, commit or push. Next: validate perception and actor association on development data before a fresh camera-held-out study.
