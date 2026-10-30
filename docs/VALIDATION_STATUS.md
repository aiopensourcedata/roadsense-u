# Validation status — 2026-10-27

RoadSense-U is usable research software. The central hypothesis and novelty remain unestablished. This report records executed checks and missing evidence; it does not certify safety or infer labels from model output.

## Current evidence

The latest actor-review workflow has also prepared 29 score-blind review frames and separate raw-detector/track diagnostics; annotations remain blank, so actor accuracy is unmeasured. The latest full automated suite passed 105 tests. See [actor review](ACTOR_REVIEW.md).

The controlled tiled/larger-checkpoint comparison still has only 1/4 usable positive development horizons for every profile. YOLO11s raises total availability from 1/10 to 2/10 by adding a negative horizon. Tiling leaves availability unchanged and produces extra emissions, including two post-onset records. All eight raw/tracked candidate stages are bound to the same 29 review frames; no improved actor or warning accuracy is established. [Protocol and results](TILED_PERCEPTION.md).

Latest development diagnostic: 1,038 actual YOLO/ByteTrack frames across four inspected training clips at 640/960 inference size. Both resolutions score one of ten eligible snapshots; nine lack a tracked pair. Explicit 640 reproduces all 1,476 cached pair records and decisions. New local pair-geometry evidence is inspectable in the dashboard, but participant correctness is still unverified. See [perception development results](PERCEPTION_DEVELOPMENT.md). No risk-policy improvement or new held-out performance is claimed.

| Question | Evidence | Assessment |
|---|---|---|
| Does the software workflow execute? | Automated tests, local browser/API/PostgreSQL checks and actual webcam/video runs. | Supported for the tested environment/cases; production safety is not assessed. |
| Does identity abstention address the Bellevue regression? | Cached ablation reproduced five original alerts and withheld the reviewed duplicate-car emission. Other 3,618 scoreable observations retained. | Development regression addressed; no general false-alarm improvement established. |
| Are there independent pair-level conflict outcomes? | The 17-window pilot contains provisional screening only; the independent sheet has no completed labels. | Not ready for real accuracy/recall claims. |
| Is uncertainty/quality scoring adequate? | Matched-noise synthetic controls show speed-dependent coverage. Real-view audit finds low detector confidence dominates rejection. | Needs labeled perception/motion validation; do not relax gates merely to increase coverage. |
| Has the calibrated classical model been evaluated on real labeled data? | Expanded published-collision pilot fitted on 10 accepted train / 4 calibration observations. On eight fresh test sources, fixed rules/logistic each yield one temporal scene hit; budget-selected logistic yields none. | Real labeled scene pilot executed; low coverage and no learned warning improvement established. Not pair-level near-miss validation. |
| Is unseen-location generalization demonstrated? | No independently labeled, geographically held-out experiment has run. | Not established. |
| Are calibration/temporal/uncertainty ablations complete? | Interfaces/features exist; the current classical model uses four pixel-space features. No matched real-data joint study. | Incomplete. Metric and pixel features must not be pooled; temporal feature comparisons still need implementation/data. |
| Are resource and false-alarm budgets met? | Latency/throughput and metadata bytes are measured, but acceptable deployment budgets are undeclared. | No budget-pass claim. |
| Is the research gap novel? | Exploratory review expanded with ACCIDENT and USDOT benchmark work. Competing methods not reproduced. | Unverified. |

## Executed coverage diagnosis

An additional [externally labeled scene-collision pilot](LABELLED_COLLISION_PILOT.md) now exists. Its first four-source test produced 22.2% snapshot coverage and one temporal scene hit, with a participant-association failure in that case. After explicitly retiring those inspected sources into development roles, the expanded comparison fitted the classical model and evaluated eight fresh sources. Coverage was 6/18 snapshots; fixed models each yielded one scene hit, and budget-selected logistic yielded none. Both fixed-model hits project outside the published impact regions in a post-evaluation check. These scene labels do not resolve the missing pair-level near-miss outcomes or establish the research hypothesis.

At the twelve frozen Bellevue prediction timestamps, 20 of 1,324 reported pair observations were scoreable. Six timestamps had no scoreable pair. Exclusive diagnostics for rejected observations:

| Recorded cause / necessary-condition bin | Observations |
|---|---:|
| Detection confidence below the 0.5 quality threshold | 1,088 |
| Insufficient motion history | 145 |
| Fit-quality penalty despite adequate detection confidence | 66 |
| Identity ambiguity | 5 |

These are correlated pair observations, not distinct failures or road-user counts. A fit penalty can reflect noise, occlusion, real maneuvers or incorrect tracks. The bins do not identify the true cause without independently reviewed tracks.

## Controlled synthetic motion experiment

The production formula is unchanged: quality decreases with residual error relative to estimated displacement. We simulated 100 independent tracks per scenario, matched noise across speeds, excluded initial warmup, and bootstrapped whole trajectories. One-pixel noise at 30 Hz produced:

| Synthetic trajectory | Velocity RMSE (px/s) | Mean per-track fraction passing quality ≥0.5 |
|---|---:|---:|
| Stationary | 0.826 | 1.88% |
| Constant 5 px/s motion | 0.826 | 100% |
| Constant 20 px/s motion | 0.826 | 100% |

Equal absolute velocity error can therefore yield different quality acceptance. This is a signal-to-residual policy, not calibrated absolute accuracy. The result diagnoses a trade-off; it does not prove that raising stationary-track quality would improve conflict prediction. Such a change could admit noise-induced alerts. Additional controls cover larger noise, a measurement jump and missing observations. Placeholder zero velocities without sufficient history are excluded from velocity RMSE. No production thresholds were tuned.

Reproduce:

```bash
python scripts/diagnose_motion_quality.py \
  --repeats 100 --seed 42 --output outputs/motion-diagnostic-new.json
```

## Evidence-input audit

```bash
python scripts/assess_research_readiness.py \
  --review outputs/bellevue-annotation-01/assistant-first-pass.csv \
  --review outputs/bellevue-annotation-01/independent-review.csv \
  --output outputs/evidence-audit-new.json
```

The current audit exits **2** and explains that no independently labeled pair-observation CSV and no admissible adjudicated labels were supplied. It does not turn provisional zeroes or censored outcomes into ground truth. `--observations` checks the existing causal/coverage/unit/location-split contract when a real table is available. Structural readiness cannot verify reviewer independence, truth of provenance flags or scientific improvement; hypothesis/novelty remain unassessed by the audit.

The draft study contract in [validation_study.yaml](../configs/validation_study.yaml) leaves deployment budgets and minimum meaningful effect unspecified. Choose and freeze them before external test results. It is a study document, not a `PipelineConfig` runtime YAML.

## Dataset compatibility findings

- **ACCIDENT**, submitted April 10, 2026, studies surveillance accident localization/type and geographic distribution shift with heuristic and learned baselines. Full-clip collision detection is different from causal three-second conflict anticipation; competitor reproduction requires a matched target/protocol. Maintainers document authenticated Kaggle acquisition. The subsequent public-API check verified declared CC BY-NC-SA 4.0 terms and obtained 28 bounded clips totaling 79.3 MB, without downloading the full archive; see [dataset acquisition update](DATASETS.md) and [executed scene pilot](LABELLED_COLLISION_PILOT.md). [Paper](https://arxiv.org/abs/2604.09819), [official setup](https://github.com/accidentbench/ACCIDENT/blob/main/dataset/README.md).
- **USDOT Intersection Safety Challenge:** inspected official catalog metadata and its 9.7 MB supporting-document archive, not video archives. Training is unlabeled. Validation has 41 runs at one physical intersection; its public sample is listed as about 3 GB. Documentation section 4.3.2 describes 3D road-user boxes plus run-level TTC/path-intersection conflict labels, not pair-level onset times for our target. Eight visual cameras at one site do not make eight independent locations. Catalog rights are marked public but the license field is empty; reuse/redistribution terms need clarification before a dataset adapter/release. [Training catalog](https://data.transportation.gov/Roadways-and-Bridges/Intersection-Safety-Challenge-Stage-1B-Training-Da/vq7s-mv3v), [validation catalog](https://data.transportation.gov/Roadways-and-Bridges/Intersection-Safety-Challenge-Stage-1B-Validation-/j58q-ymi4/about_data), [official challenge](https://www.its.dot.gov/research-areas/intersection-safety/challenge/).
- **TU-DAT:** the official repository says its boxes were based on YOLOv8. Machine-generated boxes must not silently become independent perception/conflict ground truth. [Official source](https://github.com/pavana27/TU-DAT).

No access gate was bypassed, no request form submitted, no external party contacted and no multi-gigabyte traffic archive downloaded. Public metadata availability does not establish license compatibility or label-task compatibility.

## Next evidence required

The subsequent [four-recording Bellevue evaluation](REAL_VIDEO_RESULTS.md) verified actual video-to-API-to-PostgreSQL-to-dashboard execution, but found another high-scoring same-vehicle duplicate and only 1.93% scoreable pair observations. All recordings share the earlier development site. These negative findings do not change the absence of admissible independent labels or establish cross-location performance.

1. Independently review/adjudicate actors, visibility and conflict onset under the agreed target. Window screening cannot supply one label to every pair.
2. Curate permitted positive and normal episodes from multiple verified sites, with canonical clip/event/actor IDs and source hashes. Keep an external site set untouched. Use data-volume/power analysis rather than claiming that the splitter's four-group minimum is sufficient evidence.
3. Validate tracks, units, calibration and usable coverage against independent references. Address perception/quality failure modes using development data only.
4. Freeze operating budgets/splits; run geometric, calibrated classical and predeclared quality/identity/temporal/calibration ablations on identical eligible episodes. Report misses and abstention alongside precision/recall, calibration, warning time, latency and communication measurements, with uncertainty clustered by independent locations.
5. Reproduce the closest applicable competing method before a novelty/superiority claim. Negative results and engineering contributions remain valid outcomes.

Local detailed results live under ignored `outputs/research-readiness/`. Synthetic model scores demonstrate runner functionality only. No real-world accuracy, reliable crash prediction, cost saving or certified warning capability has been established.


## 28 October — correctness and lifecycle remediation

See [measurement correctness](MEASUREMENT_CORRECTNESS.md) and [lifecycle/provenance](RELIABILITY_AND_PROVENANCE.md). Historical scene tables remain cached results from their original implementation. The corrected motion/timing paths require a new inference evidence version before comparison; cached report reproduction does not rerun perception. Independent actor/site/motion references and normal camera-hours remain missing. Larger detector/tiling candidates are not promoted.
