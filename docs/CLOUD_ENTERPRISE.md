# Local cloud and enterprise lab

This is a **single-organization research/development deployment**, not a certified enterprise service. It adds scoped service credentials, audit records, schema revisions, scheduled retention, recoverable metadata delivery and a loopback HTTPS gateway. No remote infrastructure or paid account is required.

## Start the isolated lab

Install the project using the README. Reuse your private `.env`, or run `python scripts/init_local_env.py` for a new checkout; the initializer refuses to overwrite an existing file.

```bash
python scripts/create_service_credentials.py --output outputs/cloud-enterprise/credentials
docker compose -p roadsense-enterprise -f docker-compose.enterprise.yml up -d --build --wait
mkdir -p outputs/cloud-enterprise/tls
docker compose -p roadsense-enterprise -f docker-compose.enterprise.yml exec -T gateway \
  cat /data/caddy/pki/authorities/local/root.crt > outputs/cloud-enterprise/tls/root.crt
```

The generated environment files are private (mode 0600); never commit or log their values. The generator refuses an existing directory. To rotate, generate a new directory, set `ROADSENSE_CREDENTIAL_DIR`, and recreate the API/dashboard. Disabled credentials are rejected; in-flight requests may finish during rotation.

- Dashboard: http://localhost:18501 — enable the synthetic filter for test events.
- Verified local TLS API: https://localhost:18443.
- PostgreSQL: loopback port 55432, database `roadsense_enterprise`.
- Direct API: loopback port 18000 for local diagnostics.

The project and volumes are separate from the original Compose stack. `down` preserves data; `down -v` deletes this lab's volumes. Host outputs are not mounted into the scoped dashboard: credentials, backups, research artifacts and other cameras must not bypass API authorization through a filesystem view.

Caddy uses an internal CA for this localhost lab. Trust the exported public CA explicitly; do not disable certificate verification. Browsers warn until the CA is trusted. Never export the CA private key. `configs/cloud/Caddyfile.remote` is a separate public-domain template requiring a real domain and reachable validation ports; these commands do not deploy remotely. See the official [TLS](https://caddyserver.com/docs/caddyfile/directives/tls) and [reverse proxy](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy) documentation.

## Connect a producer

Generated read/ingest credentials cover `camera-local`; `--camera CAMERA_ID` changes that scope. Use the same ID in pipeline YAML. Each detector/tracker instance belongs to one camera. Source only the single-token ingest file, not the JSON-valued server file:

```bash
set -a
. outputs/cloud-enterprise/credentials/ingest.env
set +a
export ROADSENSE_API_URL=https://localhost:18443
export ROADSENSE_API_CA_FILE=outputs/cloud-enterprise/tls/root.crt
python scripts/generate_synthetic_data.py
python scripts/run_video.py data/synthetic/crossing.avi \
  --synthetic --submit --output outputs/cloud-demo-01
```

Use `--preview` for browser video. Actual footage requires vision dependencies/weights and omits `--synthetic`. Recording remains opt-in; these checks do not validate traffic accuracy.

## Permissions and operations

`ROADSENSE_AUTH_MODE=scoped` disables shared-key bypass. `ROADSENSE_SERVICE_CREDENTIALS` is a JSON list containing credential ID, SHA-256 token hash, role, camera list and enabled flag. Use independently generated high-entropy tokens. Only administrators may use wildcard camera scope.

| Role | Access |
|---|---|
| ingest | Submit events/exposures within camera scope |
| read | Retrieve events, manifests and aggregates within scope |
| admin | Read/write/delete within scope; wildcard scope required for global metrics/purge |

Queries enforce scope even without an explicit camera filter, including exposure denominators and run lookup. `GET /audit` provides successful mutation records to administrators; structured HTTP logs record principal/status and denials without headers/tokens/bodies. This is not tenant isolation, interactive login, SSO or a complete security certification. Keep the dashboard loopback-only or behind a separately authenticated access gateway.

Wildcard administrators can query `GET /operations`: process-local response counts, recent p95 latency, schema revision, stored counts and the last retention result. These are not distributed SLA metrics. Producer reports separately expose persistent queue depth/age, retries, blocked/rejected records and recent submission times; the server does not know an unreported edge queue.

Transient transport failures, 429 and 5xx retry with persistent bounded exponential backoff. Permanent 4xx remain blocked for explicit recovery. Queue overflow rejects enqueues, retaining original JSONL. Delivery is at least once with UUID-idempotent storage. One producer/sender should own each queue file. Shutdown flushing is bounded, so in-flight requests may finish after the report snapshot. Pre-upgrade queue timestamps start at migration time and therefore bound original age only from below. After correcting permanent errors, use `scripts/replay_outbox.py`; pending records are removed only after acknowledgement. Capture freshness remains unknown with current OpenCV input.

## Schema, retention and backup

Revision 1 adopts/creates the original compatible metadata tables; revision 2 adds mutation auditing. Migration application is serialized with a PostgreSQL advisory lock (SQLite uses an immediate transaction). Missing required legacy columns fail closed. The enterprise entrypoint migrates explicitly, then starts with `auto_migrate=false`.

```bash
python scripts/manage_database.py revision
python scripts/manage_database.py upgrade
python scripts/manage_database.py backup --path outputs/private-backups/snapshot-01.json
```

Commands target `ROADSENSE_DATABASE_URL`, which may still refer to the original development database. Select the intended database through the environment; never paste its credentials into public commands/logs. Logical backups use a repeatable PostgreSQL transaction, checksums and mode 0600, bounded to 100,000 rows/100 MiB. `restore --path PATH` refuses a nonempty target and restores transactionally. Only restore trusted snapshots: a checksum is not authentication. Backups are unencrypted sensitive metadata; protect/encrypt storage and expire copies separately. Larger deployments need secured native backups/WAL/PITR and measured recovery objectives.

Downgrade requires `--target` and `--allow-data-loss`. Stop the API and use only a disposable/restorable target: 2→1 deletes audits; 1→0 deletes core records. No automatic production rollback is claimed.

Central retention runs at startup, through the purge endpoint, and at `ROADSENSE_RETENTION_INTERVAL_S` (default 3600; lab 60). Zero disables scheduling. Failed passes are visible and retried next interval. It covers events, exposures, orphan manifests and old mutation audits, not edge queues, recordings or backups.

## Reproduce verification and experiments

```bash
python scripts/verify_enterprise_local.py --output outputs/enterprise-check-01
python scripts/benchmark_cloud_delivery.py --output outputs/cloud-benchmark-01 \
  --cameras 1 4 --repeats 3 --events 6
```

The verifier checks actual TLS trust, scoped denials, computed synthetic ingestion, concurrent manifest creation, replay, PostgreSQL backup/restore counts and downgrade/upgrade in a temporary schema. It preserves the source database.

The benchmark freezes a protocol before comparisons, rotates policy order, and sends repeated computed synthetic metadata over actual loopback HTTP. It records persistence, acknowledgements, producer lag, queue rejection, bytes and local resources. Simulated senders do not establish simultaneous real-camera inference capacity. Queue exhaustion demonstrates a loss/delay trade-off. Pending records may already be persisted after a lost acknowledgement: pending and persisted are not disjoint classes. Results are development measurements, not WAN, cloud-price, traffic-accuracy, availability or novelty proof.
