# Experiment plan

## Hypothesis and target

Test whether trajectory geometry plus causal temporal behavior and uncertainty handling improve conflict anticipation on unseen fixed-camera locations at a controlled false-alert budget, latency and communication cost. This is unverified.

The first supervised target is: at prediction time t, a manually labeled conflict involving the observed pair begins in (t,t+3s]. Annotators must define conflict severity, start/end, road users, visibility and ambiguity before inspecting scores. A labeled near miss requires an agreed annotation protocol; a confirmed collision requires observed contact/authoritative labeling. Interaction detection, predicted conflict, labeled near miss and actual collision are distinct outcomes.

All feature windows end at t. Ground-truth event times are target/metric columns only and never model inputs. Exclude censored negatives lacking a full future horizon; exclude post-event windows. Blind dual annotation and agreement analysis are needed. Separate real and simulated provenance and audit license/consent. Build canonical source/clip IDs across edited/reuploaded clips, hash exact copies and manually check approximate duplicates.

## Implemented baseline pipeline

`scripts/generate_synthetic_data.py` makes a 5-second software fixture and a separate 480-row generated benchmark (8 fictional locations). The tabular benchmark is **not extracted from real video**, has a simplified noisy target and no real conflict labels. `scripts/train_baseline.py` runs a geometric rule and standardized logistic regression with disjoint sigmoid calibration. Independent per-location logistic models run with within-location event splits; they are not unseen-location comparisons. The temporal network and federated extensions are not implemented.

CSV: `event_id, clip_id, location_id, prediction_time_s, event_time_s, label, cpa_distance, time_to_cpa_s, closing_speed, quality, synthetic`. The present baseline assumes pixel/second features and fixed 60px/3s rule scales. Inputs explicitly mixing synthetic/real provenance or declaring non-pixel units are rejected. Legacy files without a `units` column are assumed to be pixel-space; data provenance remains the caller's responsibility. Do not mix metric and pixel units or infer physical comparability across camera scales. The generator labels simulated CPA within 25px under noisy geometry and future arrival within 3s as positive. It supports software testing, not a proposed ground-truth road-safety definition.

Train/calibration/threshold-validation/test groups are disjoint. Location mode assigns entire locations. Event mode joins connected clip/event identities so adjacent windows cannot cross partitions. Canonical duplicate IDs remain a curation prerequisite. Four groups are the minimum; much more data is required for reliable estimates. Positive labels are checked to lie in the 3s horizon. The probability baseline enforces one prespecified prediction timestamp per event and rejects duplicate event IDs. Group helpers support repeated frames for split testing, but the current probability experiment does not pool them. For real sliding windows whose horizon targets change, define fixed horizon evaluation episodes or extend the evaluation protocol before use.

Model fitting uses only training rows, sigmoid fitting only calibration rows, and threshold selection only validation episodes. The selected threshold controls the empirical **negative-episode false-positive fraction** at 5% on validation, then stays fixed for the test set. It does not guarantee the same test rate and is not false alarms per camera-hour. Rules have no Brier score because their scores are not probabilities. Model predictions are calibrated per observation; event max aggregation can destroy calibration when episodes contain multiple windows. Use one prespecified timestamp per event for probability calibration metrics, or add independent episode-level recalibration before claiming calibrated event probabilities. Generated rows use one timestamp per episode.

## Metrics and uncertainty

Implemented: precision/recall, average precision as PR-AUC summary, recall at the validation-selected false-positive budget, observed test false-positive fraction, Brier and reliability bins for probability outputs, early-warning time among detected positive events plus explicit misses. Early warning is observed event time minus first qualifying pre-event evaluated timestamp. Its interpretation depends on the sampling schedule; event-only positive selection is biased and must not conceal misses.

Location-cluster bootstrap resamples entire held-out locations for AP 95% percentile intervals; few locations produce unstable intervals, so report camera/event counts. Add two-stage location/event resampling when sufficient data exist; never treat frames as independent replicates. Range-based abstention reports coverage and accepted-only performance; it is not calibrated epistemic uncertainty or conformal coverage.

Run reports include p50/p95 processing latency over the latest up to 10,000 frames (with explicit sample count), throughput, CPU seconds, process RSS and attempted JSON-body bytes including retries (not packet-level network usage). Synchronous cloud submission affects latency/FPS. Optional cost helper uses explicit user-supplied CPU-hour and egress tariffs, excludes idle capacity, database/storage/request costs and does not claim a cloud bill. Measure actual deployment consumption for cost conclusions. Camera-level accuracy requires labeled data and is not the dashboard's mean-score comparison.

## Required experiments

1. Freeze source-level and location-level splits before tuning; retain a never-touched external test location set. Report label prevalence and traffic exposure by location.
2. Compare rules, per-location logistic models and centralized logistic models on compatible in-location tests; evaluate centralized models separately on unseen locations. Bootstrap paired differences on the same test observations.
3. Add causal temporal features to centralized models, then a small temporal network if justified by data volume. Match detection, horizon and split protocols.
4. Ablate calibration, perspective normalization, trajectory smoothing, uncertainty abstention, sigmoid calibration, braking/turn features and gap resets. Sweep coverage against precision/recall and missed-warning time.
5. Evaluate occlusions, dense scenes, night/rain, vulnerable road users, ID switches, camera vibration and missing/late frames. Compare measured speed against independent references.
6. Compare metadata policy/cooldown settings on network traffic, event recall and latency. Measure false alerts per camera-hour on complete normal video with deployment cooldown, not only sampled negative clips.
7. Only after centralized training works on real data, compare FedAvg across location clients under matched initialization, training budget, privacy threat model and communication accounting. FL alone does not ensure privacy; gradients can leak information. No FL superiority assumed.

## Failure criteria and competing explanations

Reject an improvement claim if uncertainty intervals include no meaningful gain, false alarms exceed the preregistered budget, gains vanish after duplicate/camera control, warning arrives after the event, recall improves solely by suppressing coverage, or computation/network cost violates the deployment budget. Camera scale, traffic mix, prevalence, exposure, label ambiguity, detector domain shift, changed annotation horizon and benign aggressive driving can explain apparent gains. Report negative results and all exclusions. Federated/temporal/uncertainty-specific novelty is unverified until reproducible matched experiments and a broader review support it.
