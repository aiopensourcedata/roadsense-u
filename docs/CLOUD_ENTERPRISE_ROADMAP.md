# Cloud and enterprise extension — proposed scope

Status: initial local slice implemented and exercised on 30 October 2026. See [setup and actual limits](CLOUD_ENTERPRISE.md). This document preserves the design and later gates; not every future extension is implemented. No remote resources have been provisioned or enterprise readiness established.

## Starting point

Implemented: edge video processing, authenticated metadata API, PostgreSQL, multi-camera/location IDs, idempotent ingestion, durable bounded delivery outbox, hash-bound run provenance, retention endpoints, Docker Compose and dashboard. Local actual-HTTP/PostgreSQL integration has been verified previously. This is not a geographically distributed cloud deployment or a production security/availability certification.

The original development mode uses one broad shared key. Scoped mode now enforces camera/role permissions, without organization/user isolation. Persistent transient retry, blocked permanent failures and scheduled central retention complement explicit recovery/purge. A camera ID alone is never an authorization boundary.

## First implementation slice

Keep edge inference separate from a provider-neutral remote metadata service, initially one organization with multiple camera-scoped service credentials. Begin with one API deployment, PostgreSQL, HTTPS termination and operational metrics; Kubernetes, federation and a messaging cluster are unnecessary for this slice.

1. **Delivery reliability:** persist retry state, bounded reconnect backoff and queue age; distinguish permanent rejection from transient outages; keep UUID idempotency, explicit overflow and replay. Test restarts, lost acknowledgements and unavailable servers. Bound expired/rejected records explicitly rather than silently deleting them.
2. **Observability:** expose authenticated queue depth/oldest pending age, accepted/rejected/retried records, server ingestion latency and producer status. Label shutdown snapshots. Device capture age remains unknown unless a trustworthy acquisition clock is added.
3. **Access boundaries:** separate ingestion, read and administrative permissions; derive permitted cameras from authenticated credentials and enforce them on every event/run/exposure/query/deletion route. Bind run lookups to the same policy. Keep the existing key only as an explicit local-development mode. Add denial tests before any organization-isolation claim.
4. **Database lifecycle:** versioned migrations, upgrade/rollback tests against a disposable PostgreSQL instance, scheduled retention and a demonstrated backup/restore procedure. Treat local edge queues, JSONL, backups and central records as separate retention surfaces.
5. **Deployment:** documented single-server HTTPS configuration, secret injection, health/readiness checks, resource limits and recovery instructions. Remote deployment requires an actual target and credentials; paid services are not required for local experiments.

If separate organizations are needed, add tenant identity, membership, database constraints, query isolation and cross-tenant denial tests through migrations before accepting their data. A tenant_id field by itself provides no isolation. SSO/OIDC integration, managed object storage and MQTT are later needs, not evidence of current capability. Raw footage remains local by default.

## Experiment interface

Use source-bound frozen computed event streams to separate transport behavior from perception. Generate explicit synthetic camera clients for repeatable load tests; do not describe them as independent real cameras or use their counts to validate detector throughput.

Compare synchronous HTTP, the current bounded outbox and the proposed recoverable outbox at identical payloads/capacity/retry budgets. A single-process local baseline provides a no-network reference. Controlled profiles should include healthy service, added delay, bandwidth restriction, server outage, rate limiting, lost responses, producer restart and queue exhaustion. Change one factor at a time before combining failures.

Measure creation/queue/acknowledgement age with monotonic clocks on one host; across hosts report clock synchronization error and distinguish server ingestion from capture. Report p50/p95/p99 delay, full delay distributions, duplicates/conflicts, persisted versus generated UUIDs, lost/overflow/expired records, drain time, local processing throughput, attempted body bytes, CPU and memory. Report warm-up, repetitions, confidence intervals over independent runs and all failures. No real-time warning claim follows from metadata arrival measurements.

Increasing simulated clients tests service/delivery load. Concurrent detector instances on specified hardware are a separate inference-capacity experiment. A failed or unreachable producer must appear as missing coverage, not a safe road or a zero-risk location.

## Gates

Choose numerical operating budgets and target scale before inspecting final experimental results. No performance claim until the frozen comparison runs; no cost saving claim from an assumed tariff alone; no uptime/SLA guarantee from a short benchmark. Security, licensing, operational backup recovery and independent traffic accuracy require their own evidence. A cloud-ready architecture does not establish enterprise readiness or improve traffic prediction by itself.

Detailed paper planning remains in the private research workspace. Existing traffic-label/site/exposure limitations still apply.
