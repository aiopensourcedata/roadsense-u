# Published-label collision pilot

Executed 2026-10-27 using [ACCIDENT](https://www.kaggle.com/datasets/picekl/accident), [Picek et al., 2026](https://arxiv.org/abs/2604.09819). The paper describes consensus impact-frame, impact-region and collision-type annotations. These are external dataset labels, not RoadSense-U predictions; they were not independently re-adjudicated here. The retrieved public dataset metadata declares CC BY-NC-SA 4.0. Videos stay local and are not redistributed.

## First frozen pilot

Selected 16 clips, four per multi-vehicle collision type, from a bounded public catalog. Total download: 47,405,635 bytes. Four original video sources per train/calibration/validation/test partition; this is a custom source-disjoint convenience pilot, not the official benchmark or a verified camera-disjoint split.

The target is the **published scene collision onset within three media-time seconds**, not a labeled pair-level near miss. Fixed snapshots use the first observed frame at/after 1/4/7 seconds with bounded lateness. Features use past/current measurements only. Post-onset and future-incomplete horizons are excluded; missing features remain abstentions. The maximum current-policy scoreable pair supplies features without using the annotation's impact box or future trajectories.

Full decode returned 7,889 frames. Actual CPU YOLO11n/ByteTrack processed 3,047 frames. All 16 source bindings passed after a documented pre-inference count/index convention amendment: files decoded to `no_frames+1`, with collision-frame PTS matching the published time. Original failed checks were preserved. Later detector processing stops after the last scheduled snapshot; full source audits independently provide future label coverage. No inference threshold was lowered.

| Partition | Positive / negative eligible snapshots | Scoreable positive / negative |
|---|---:|---:|
| Train | 4 / 6 | 1 / 0 |
| Calibration | 4 / 4 | 2 / 2 |
| Validation | 4 / 5 | 4 / 3 |
| Test | 4 / 5 | 1 / 1 |

At the fixed 0.45 threshold, the test has one scene-time hit, one false alert, and seven abstentions. Scene-proxy precision is 50%, operating recall including abstained positives is 25%, and snapshot coverage is 22.2%. The validation-selected threshold gives identical test decisions. The exploratory four-source bootstrap recall interval is [0, 0.75]; it is unstable. This is not a deployed false-alarm rate. Accepted-only AP uses only two observations from one source and cannot establish improvement over a full-coverage prior.

The classical model could not fit: accepted training data lacked negatives. No replacement examples or relaxed quality gate were introduced into this completed pilot.

**The sole temporal hit does not establish a correct collision warning.** A post-evaluation case screen places its selected pair on the opposite carriageway from the published impact region; its projected CPA midpoint is outside that region. This is a single-case diagnostic, not independently adjudicated actor matching. Scene coincidence must not be presented as correct participant identification.

Local evidence, exact selections, hashes, amendments, JSON metrics, plots and reproduction commands: `outputs/labelled-pilot-02/REPORT.md`. Code: `scripts/run_collision_pilot.py`, `scripts/evaluate_collision_pilot.py`. No source footage was uploaded; no manuscript was drafted.

## Expanded comparison

The first pilot's inspected sources are now development data. Twelve become training and four remain separate calibration sources. Twelve fresh source IDs were selected before inference: four validation and eight test sources, with one/two per collision type respectively. Additional download: 31,942,109 bytes. Results belong to a separate experiment under `outputs/labelled-pilot-03/`; the first pilot remains unchanged. A previous test is not relabeled as untouched.

The expanded comparison includes fixed and validation-selected rules, a training-prevalence reference with full and matched perception coverage, calibrated logistic models, and training-range abstention. Fixed 0.5 probability decisions are separate from validation-budget decisions; an unavailable validation threshold is never replaced silently. The scene model is explicitly labeled as offline collision-onset research and is not integrated as a pair-level safety model.

### Executed expanded results

The classifier fitted on ten usable training snapshots (six positive/four negative) and four separate calibration snapshots (two/two). Its scaler, logistic and sigmoid parameters exactly match the fingerprint saved before fresh test evaluation. The twelve new clips contributed 1,730 actual YOLO/ByteTrack inference frames; the complete 28-source cohort contains 4,777 unique inferred/cached frames.

Eight fresh test collision sources yield 18 eligible snapshots, of which only six are scoreable. Five positive and seven negative snapshots abstain.

| Method | Timely scene hits / 8 collisions | False alerts in 10 negative snapshots | Scoreable coverage |
|---|---:|---:|---:|
| Rules, fixed 0.45 | 1 / 8 | 0 | 6 / 18 |
| Logistic, fixed 0.5 | 1 / 8 | 1 | 6 / 18 |
| Logistic, validation-selected threshold | 0 / 8 | 0 | 6 / 18 |

The rule's 100% alert precision comes from just one alert and does not establish reliable precision. The two fixed models' hits occur on different clips. Accepted-only AP is 0.6667 for rules and 0.7000 for logistic on the same six snapshots; matched-prior AP is 0.5000. Logistic Brier is 0.24965 versus 0.25510 for the matched prior. These tiny-sample differences do not establish improvement. Source-cluster recall intervals remain wide; no camera-independent inference is supported. The training-range policy changes no accepted predictions.

Both fixed models' single temporal hits project their selected pair's CPA midpoint outside the corresponding published impact region in a post-evaluation diagnostic. This does not independently adjudicate actors, but prevents treating temporal scene coincidence as correct participant identification. No learned warning improvement is demonstrated.

Detailed numeric outputs, source/partition provenance, model card, plots and reproduction commands are in `outputs/labelled-pilot-03/REPORT.md`. All new test sources have now been evaluated; future tuning requires a fresh test cohort. Automated checks passed 80 tests, Ruff, Bandit medium/high checks and source-content checks. No manuscript, commit or push was made.

No near-miss, actor-level correctness, natural-traffic false-alarm, physical-speed or unseen-camera claim follows from either protocol. Future algorithm changes need a fresh evaluation cohort after development decisions.
