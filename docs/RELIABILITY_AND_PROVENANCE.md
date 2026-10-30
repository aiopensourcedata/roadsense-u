# Run lifecycle, delivery and provenance

## Timing and partial runs

Repeated, backward, negative or unavailable prerecorded PTS trigger a discontinuity. Exported media time remains increasing for indexing, but fallback values are explicitly labeled and the raw PTS is retained. Recovery starts a new timing segment; tracker, motion, identity history and cooldown reset. Horizons crossing a clock failure are censored. A backend with no usable clock cannot silently produce continuous motion from repaired timestamps. These repairs do not reconstruct physical capture time.

Live sources use monotonic decode-arrival time. The report measures decode, inference, scoring/event-metadata work and arrival-to-processed latency separately. Events include arrival-to-alert time when available. **Capture-to-alert age remains unavailable**: buffered frames may be old when OpenCV returns them. Neither throughput nor arrival latency establishes real-time capture freshness. Hardware/source timestamp support and a buffering study are still required for that claim.

Ctrl+C and processing exceptions finalize `report.json` atomically with `status=interrupted` or `failed`, plus retrospective event coverage, then propagate the original exception. Initialization failures receive a zero-frame failed report. The dashboard discovers partial runs and labels them incomplete. Reports with no status are explicitly legacy/unspecified. Power loss/SIGKILL and disk-write failure cannot guarantee finalization.

## Bounded asynchronous delivery

Supplying a cloud client enables a local SQLite outbox. The processing thread writes metadata durably; a separate sender performs bounded HTTP retries. `outbox_capacity` defaults to 1000 pending records. A full queue rejects the enqueue and increments a counter; the original event remains in `events.jsonl`. Transient failures now retain retry state and retry automatically with bounded backoff; permanent HTTP 4xx failures remain blocked for explicit recovery. The old stop-after-failure policy is retained only as an experimental baseline. See [local cloud controls](CLOUD_ENTERPRISE.md).

`delivery_shutdown_s` bounds flushing at shutdown (default one second). An in-flight request may finish after the report snapshot, so delivery counters and attempted bytes are labeled snapshots. No raw imagery enters this queue. Retained metadata is sensitive operational data: remove local outbox/JSONL files when no longer required; API retention does not delete local files.

After stopping the original producer/sender and fixing connectivity/authentication:

```bash
python scripts/replay_outbox.py outputs/my-run/delivery.sqlite
```

Successful replay removes pending records. JSONL replay remains available for queue-rejected events; UUID idempotency prevents duplicate storage. Replaying historical events may fail the API's configured retention window; do not silently rewrite their timestamps.

## Immutable run binding

`run_manifest.json` records a SHA-256 digest over canonical JSON containing session/camera/location IDs, native decoded geometry, camera pose version, sanitized configuration, source-file hash when available, detector-weight/calibration hashes, resolved ByteTrack settings, source-code file hashes, installed dependency versions and timestamp semantics. Live URLs and credentials are omitted. A live stream has no immutable content hash. Hash binding detects inconsistency; it is not a signature or evidence that a declared physical camera is correct.

Events carry this bounded manifest and its digest. The API verifies the digest and camera/session/timing agreement, persists the manifest with the first event, rejects conflicting manifests for one run and exposes authenticated `GET /runs/{session_id}`. Event payloads distinguish media time/raw PTS, timestamp origin and **server-assigned ingestion time**. Legacy payloads remain readable/replayable with `legacy_unknown` timing. Local manifests also exist for runs with no alerts; remote manifests are created only when an event is delivered. Repeating a manifest increases metadata bytes; reports count actual attempted JSON bodies.

Upgrade the API alongside the producer: an older strict schema rejects new fields. The added `runs` table uses prototype schema creation; it does not require changing existing event columns. Production migration/versioning remains separate deployment work.

## Calibration geometry

Runtime metric projection now requires a calibration `binding` matching the configured camera, native width/height, `image_transform=decoded_native` and an explicit `camera_pose_version`. Changing camera pose, crop/rotation or resolution requires new calibration and a new run. The current acquisition path applies no crop/rotation transform. Legacy unbound artifacts fail closed for runtime metric use; standalone projection mathematics remains usable for unit tests.

Example binding (use your actual values, not these illustrative dimensions):

```json
{"camera_id":"junction-01","width":1920,"height":1080,"image_transform":"decoded_native","camera_pose_version":"survey-2026-10-28"}
```

The runtime YAML must declare the same camera/pose. Outside-hull detections are excluded by default and counted. Explicit `calibration_fallback: image_space` enables pixel fallback; mixed-unit pair exclusions are counted. No metric/pixel comparison is performed. Binding cannot detect undeclared camera movement, and calibration still needs independent physical reference validation.

## Dependency boundary and cached reproduction

`configs/vision_environment.json` records 13 exact versions tested locally. A dedicated Python 3.12 CI job installs those constraints and the explicitly downloaded, checksum-checked YOLO11n checkpoint. Opt-in tests exercise actual YOLO blank-frame inference, real tracker creation/empty input/persistence/reset and the direct tiled adapter. These generated-array tests establish compatibility, not detection accuracy. Hosted CI execution is separate from local checks.

```bash
python scripts/check_vision_environment.py
ROADSENSE_VISION_INTEGRATION=1 pytest tests/test_vision_integration.py -q
```

`scripts/diagnose_calibration.py` describes raw/calibrated probabilities and constant references from a trusted saved scene model, without tuning or overwriting its experiment. `scripts/package_experiment.py` makes a local source-plus-cached-result archive, excluding media, weights, credentials and private research. Its manifest hashes every file; `scripts/reproduce_scene.py` refits and compares the complete cached report. Review dataset-derived metadata rights before redistribution. This lightweight package does not reconstruct historical acquisition/inference or complete the independent scientific evidence.


For a local PostgreSQL software integration check, set the existing environment configuration and run `python scripts/verify_delivery_postgres.py --output outputs/delivery-check-01`. It creates and removes only its own randomly named temporary schema, starts an isolated localhost API, processes generated synthetic video and verifies actual HTTP delivery, direct SQL counts, manifest persistence, authentication and replay. It requires local schema-creation permissions and binds only loopback. It does not rebuild the existing Compose deployment or establish traffic accuracy.
