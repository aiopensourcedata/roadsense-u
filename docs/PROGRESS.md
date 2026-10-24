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
