# Usage and configuration

Detailed commands and measurement assumptions for the RoadSense-U research prototype. Start with the [quick start](../README.md#quick-start).

## Real video, webcam, USB or phone

Install CPU PyTorch first to avoid downloading unnecessary CUDA libraries, then the detector extra:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[vision]'
python scripts/download_weights.py
python scripts/run_video.py /path/to/traffic.mp4 --submit --output outputs/traffic
python scripts/run_webcam.py 0 --max-frames 300 --output outputs/webcam
python scripts/run_video.py 'http://PHONE_IP:PORT/video' --max-frames 300 --output outputs/phone
```

The downloader fetches one approximately 6 MB pretrained YOLO11n model from the official Ultralytics release after checking disk space, and verifies a pinned SHA256 from the initially reviewed upstream download. The pin is not an independent author signature. Inference requires local weights and never falls back to synthetic detections. The vision extra installs the official `ultralytics-opencv-headless` variant, keeping one OpenCV wheel variant in the environment. Existing environments with standard `ultralytics` and GUI `opencv-python` should use a fresh virtual environment. The YOLO integration uses Ultralytics' AGPL-3.0/enterprise licensing; review [their terms](https://www.ultralytics.com/license) before redistribution or deployment. A GPU is optional: set `device: '0'` in a copied configuration and install compatible PyTorch. USB/laptop cameras use device indices; phone streaming requires an app providing an HTTP MJPEG or RTSP feed accessible on the same network. Connection timeout behavior depends on OpenCV's backend. MP4/AVI use file timestamps, with recorded FPS fallback when necessary. Live motion uses arrival timestamps and is affected by buffering/network jitter.

Use `--config path.yaml` to change camera/location IDs, rule scales, thresholds, or calibration. The default 0.1 detection floor preserves lower-score boxes for ByteTrack association; increasing it above the tracker's low-score range disables that recovery stage. A source video is user-provided; no restricted dataset footage is bundled. The helper `scripts/make_detector_smoke_video.py` explicitly downloads the official Ultralytics bus example and makes a **static repeated-image smoke fixture**, not a real motion-validation clip. Process it with `--config configs/detector_smoke.yaml` so its synthetic temporal provenance remains labeled.

## Watch video live

Run processing with `--preview`, then open **http://127.0.0.1:8765** in your browser:

```bash
source .venv/bin/activate
python scripts/run_webcam.py 0 --preview --output outputs/live-01
# Or view a video file while it is processed:
python scripts/run_video.py /path/to/traffic.mp4 --preview --output outputs/video-preview-01
# No camera needed for a short, labeled synthetic demonstration:
python scripts/run_video.py data/synthetic/crossing.avi --synthetic --preview --output outputs/synthetic-preview-01
```

The preview shows annotated detections, track IDs, trajectories and measured motion/rule scores. It uses an MJPEG stream with only the latest JPEG kept in memory; it does not record or upload footage. Add `--save-video` separately to retain footage or `--submit` to send event metadata. File playback is paced at source speed when possible; slow inference limits viewing FPS. Live camera input shows frames as they are processed and may have capture/network buffering delay. Open the URL while the command is running; the preview stops at end of file or on Ctrl+C. Use a new output directory for each run. Change the port with `--preview-port 8766` if needed. This local viewer binds only to loopback and is separate from the dashboard; do not expose it publicly.

## Run without Docker

For an explicit development-only SQLite service:

```bash
source .venv/bin/activate
set -a; source .env; set +a
export ROADSENSE_DATABASE_URL=sqlite:///outputs/roadsense.db
uvicorn roadsense.cloud.api:create_app --factory --host 127.0.0.1 --port 8000
```

For local PostgreSQL set `ROADSENSE_DATABASE_URL=postgresql+psycopg://USER:PASSWORD@localhost:5432/roadsense`. SQLite tests do not establish PostgreSQL compatibility; the integration smoke script checks actual PostgreSQL when a service is available:

```bash
python scripts/check_postgres.py
pytest -q
ruff check src scripts tests
```

## Calibration and interpretation

Without calibration, positions, distances, speeds and accelerations are pixels, px/s and px/s². They are not physical speed or severity. Manual calibration JSON contains `image_points`, `world_points` in meters (at least four noncollinear pairs), positive `uncertainty_m`, and **independent** `validation_image_points`/`validation_world_points` (at least two). Optional `max_error_m` defaults to 1. Only positions within the calibration hull yield metric measurements. Set `calibration: path.json` in YAML. See the synthetic calibration for file shape, not for real-camera calibration.

Bottom-center boxes approximate road contact. A fixed camera, approximately planar road, reliable timestamps and known metric correspondences are assumptions. Lens distortion, hills, camera shake, occlusion and box jitter can invalidate estimates. Fit residuals and declared calibration error are reliability indicators, not statistical confidence bounds. Trailing least squares smooths motion causally. Track gaps and coordinate changes reset history; jump flags only heuristically identify possible ID switches. IDs are scoped by session and `tracking_epoch` (incremented on reconnection), are not person/vehicle identities, and may switch or fragment.

CPA is the minimum separation in the configured future horizon under constant velocity. Shared path arrivals require forward, nonparallel path intersection. Conditional disk TTC is implemented and tested but omitted by the video pipeline because physical footprints are unknown. Observed occupancy-interval PET is implemented separately; automatic zone occupancy annotation is not implemented. Acceleration/speed changes and direction changes are measured features; braking attribution requires physical calibration and validation. Rules report a heuristic score and measured explanations. Low quality/high residual inputs abstain; no crash probabilities are claimed.

## Reproducible baseline experiment

```bash
python scripts/train_baseline.py data/synthetic/benchmark.csv --output outputs/benchmark
# A fitted model can be evaluated on a separate held-out dataset with a fixed validation-selected threshold:
python scripts/evaluate_model.py outputs/benchmark/baseline.joblib /path/to/heldout.csv --threshold 0.5
```

The input validator rejects mixed synthetic/real rows and non-pixel unit pools. Legacy CSV files lacking `units` are treated as pixels; declare units explicitly for new data. The train command fits a logistic baseline, calibrates on separate groups, selects thresholds on a third partition and tests on held-out locations. It compares a geometric rule and independent within-location models. Only load trusted local joblib artifacts. The classical model is an **offline research baseline**, not yet an online alert backend. Features use pixel coordinates; do not pool metric features or compare pixel scales as equivalent physical distances. CSV requirements, target, limitations, ablations and split protocol: [experiment plan](EXPERIMENT_PLAN.md). Synthetic metrics must not be cited as real-world performance or evidence for novelty.
