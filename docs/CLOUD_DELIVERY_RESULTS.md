# Local delivery development experiment — 30 October 2026

These are executed **synthetic software/transport measurements**, not traffic-warning accuracy or an enterprise SLA. The workload repeats features from one computed synthetic crossing event, assigns deterministic test UUIDs and simulated sender IDs, and uses loopback HTTP plus SQLite.

Protocol: 1 and 4 simulated senders; 6 records per sender; 3 repetitions; 3 delivery policies; 6 profiles = **108 trials, 1,620 generated test records**. The script and core hashes and all generated/persisted UUID partitions were verified. Policy execution order rotates across repetitions. A first 90-trial smoke run is separate from this expanded development run.

| Simulated senders | Profile | Synchronous persisted | Stop-on-failure persisted | Recoverable persisted |
|---|---|---:|---:|---:|
| 1 | healthy | 100.0% | 100.0% | 100.0% |
| 1 | delay | 100.0% | 100.0% | 100.0% |
| 1 | outage | 33.3% | 0.0% | 100.0% |
| 1 | rate_limit | 22.2% | 0.0% | 100.0% |
| 1 | lost_ack | 100.0% | 16.7% | 100.0% |
| 1 | queue_full | 100.0% | 22.2% | 22.2% |
| 4 | healthy | 100.0% | 100.0% | 100.0% |
| 4 | delay | 100.0% | 100.0% | 100.0% |
| 4 | outage | 34.7% | 0.0% | 100.0% |
| 4 | rate_limit | 45.8% | 0.0% | 100.0% |
| 4 | lost_ack | 100.0% | 16.7% | 100.0% |
| 4 | queue_full | 100.0% | 22.2% | 19.4% |

Values are mean per-trial eventual persistence fractions within the configured experiment window. They do not measure physical incidents, independent sites or capture freshness. A lost acknowledgement may leave a persisted record unacknowledged/pending at the producer; these classes overlap.

The recoverable policy persists every generated record in the tested healthy, delayed, 120 ms outage, initial 429 and lost-ack profiles. With capacity forced to one during a burst, it persists only about 19–22%; across those six trials, 72 enqueues are rejected. Rejected originals remain available through the normal pipeline JSONL policy; the isolated benchmark records their generated/unpersisted UUIDs. Retry recovery does not overcome finite admission capacity. Synchronous submission persists the burst workload while delaying the producer. These are different delay/loss/resource trade-offs, not universal superiority.

For example, with four senders and injected delay, median trial p95 producer scheduling lag is 536 ms synchronously versus 148 ms with recovery. This is metadata-producer lag, **not inference FPS or capture-to-alert latency**.

The runner records full acknowledgement ages, illustrative 0.5-second acknowledgement deadlines, attempted JSON-body bytes, process CPU and RSS. It does not measure TLS/header bytes, billable provider costs or WAN behavior. The client and server share a process/host; fixed services and ordinary system activity remain present. Three trials give weak and sometimes degenerate bootstrap intervals. Budgets are exploratory development settings, not preregistered safety requirements. Do not reuse these inspected workloads as independent test evidence.

Local deployment verification separately exercised trusted HTTPS, camera/role denials, idempotent replay, concurrent shared-manifest ingestion, PostgreSQL backup/restore and migration recovery. That does not turn simulated load into real-camera capacity or prove multi-tenant security.

Reproduce using the commands in [the lab guide](CLOUD_ENTERPRISE.md). Detailed generated plans/results remain under ignored outputs; the script emits its own fresh frozen plan. Paper interpretation and drafts remain private.
