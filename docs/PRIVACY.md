# Privacy, security and safety

This prototype is a research and analysis tool, not a certified real-time collision-avoidance system. A geometric predicted conflict is neither a confirmed near miss nor a collision. No identity recognition, face recognition or license plate recognition is implemented.

## Defaults

Video is decoded locally; raw frames are not stored by the pipeline or transmitted. Annotation video requires explicit `--save-video` opt-in. Optional `--preview` serves annotated images only on loopback to a local browser, retains the latest JPEG in memory, sends no-cache headers and stops with the processor. Preview is separate from the authenticated metadata API and must not be exposed publicly. User input files remain the user's responsibility. No clip upload implementation is enabled. Events contain local track IDs, class labels, timestamps, camera/location IDs and measured feature summaries. Such metadata can still be sensitive and should be access controlled. Synthetic fixture exports are explicitly labeled. Session IDs and reconnection epochs scope track numbers, which are temporary tracking identifiers, not real-world identities.

The synthetic/detector smoke helpers deliberately write generated fixtures and the explicitly fetched official example image. Dashboard selection reads local generated artifacts; it does not automatically upload footage. Processing user-uploaded videos currently uses the CLI/file path; an in-browser upload-processing flow is not implemented. Annotated files may retain identifiable people/plates; this prototype does not implement reliable anonymization, so do not share such footage without a separate privacy process.

## Access and transport

Data, statistics, exposure, delete and maintenance endpoints require an environment-configured API key of at least 24 characters. `.env.example` contains no secret; the local initializer creates private random values without displaying them. Development Compose services bind to loopback. Key rotation, per-camera permissions, multi-user roles, TLS termination and rate limits must be configured before remote use. The Python client rejects remote plain HTTP; an internal Compose service-name exception allows container-to-container traffic. Client connections are direct (`trust_env=False`) to avoid routing metadata via ambient HTTP proxy settings. Deployments needing a proxy must configure a reviewed transport explicitly.

The API validates bounded metadata types, timestamps and finite values, rejects unknown fields, and limits raw request bytes including chunked bodies before JSON parsing. Reverse proxy request timeouts and connection/concurrency limits are still required. The public health endpoint checks database connectivity without returning credentials. Structured request logs exclude API keys, body content and source URLs. Exception messages returned to clients should not include database connection strings. Dependency/package behavior must be reviewed when upgrading.

## Retention and deletion

`ROADSENSE_RETENTION_DAYS` defaults to 30. Startup purge and authenticated `POST /maintenance/purge` delete old database events and exposures; externally schedule that endpoint for continuous enforcement. Authenticated `DELETE /events/{event_id}` removes a selected event. Exposure IDs/windows remain only for aggregate denominators and expire by retention. Prototype metadata uses event timestamps, so callers providing old capture dates may need a configured research retention policy; rejected events remain in local JSONL for review.

Local CSV/JSONL/video outputs and database volumes/backups do **not** expire automatically. Set a project-specific schedule and restrict directory permissions. Schedule local artifact deletion and backup expiry separately. Removing a Compose container does not remove its database volume. Never use destructive volume deletion as routine cleanup when research data must be retained.

## Threats and limits

A shared API key is a basic prototype control, not tenant isolation. No certified anonymity, differential privacy or gradient privacy guarantees are claimed. Fitted joblib/model weight files are trusted local executable deserialization artifacts: never load untrusted uploads. Use signed/reviewed release artifacts in production. Camera URLs may contain credentials; avoid shell history and use dedicated protected configuration for actual deployments. No camera movement or adversarial-input robustness guarantee is provided. Safety decisions must not rely on these unvalidated estimates.

## Opt-in documentation screenshots

The browser E2E command can explicitly capture the generated synthetic scene and its dashboard outputs with `--screenshots docs/assets`. These reviewed synthetic-only PNGs and their provenance manifest are intentionally retained in the source documentation. They contain no identifying footage, source credentials or pretrained assets. The option is separate from raw-video recording and does not enable frame uploads. E2E-generated metadata remains flagged synthetic and follows database retention.
