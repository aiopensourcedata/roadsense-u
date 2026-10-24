# Security policy

RoadSense-U 0.1.x is an experimental research prototype. It has no production security certification or response-time guarantee. Supported fixes target the latest development version; there is no long-term support branch yet.

## Reporting

Use [GitHub private vulnerability reporting](https://github.com/aiopensourcedata/roadsense-u/security/advisories/new) when enabled. Do not include credentials, personal footage or exploit details in public issues. If private reporting is unavailable, request a private reporting channel without disclosing the vulnerability. Maintainer: [Farid Eyvazov (@frdayvz85)](https://github.com/frdayvz85).

## Deployment boundaries

Keep the preview, dashboard and development database on loopback. The local preview and dashboard are not multi-user authenticated services. The API uses a shared environment API key; it is not tenant isolation. Remote deployments need HTTPS, authorization boundaries, connection/read timeouts, rate limiting, backups, key rotation and scheduled retention purge. Do not expose development Compose ports or the browser preview publicly.

Only load reviewed local model weights and joblib files. They may execute deserialization code. Video and stream inputs must come from trusted/permitted sources; malformed codecs, buffering and camera motion are not robustly defended against by this prototype. No face/plate recognition is implemented, but locally retained footage may still identify people.

## Checks and their limits

CI runs tests, formatting, static checks and a dependency advisory scan. Dependency scans depend on advisory coverage and availability and cannot prove software is vulnerability-free. The release checker excludes known private/artifact paths and checks common secret signatures and local development secret values without displaying them; it is not a comprehensive secret detector. Before making an existing repository public, scan its **full Git history** with a dedicated secret scanner and rotate any exposed credential. `.gitignore` never removes already tracked files or history.
