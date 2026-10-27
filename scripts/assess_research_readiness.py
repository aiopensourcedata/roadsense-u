# SPDX-License-Identifier: AGPL-3.0-only
"""Audit research evidence; exit 2 when real benchmark inputs are not ready."""

import argparse
import json
from pathlib import Path

from roadsense.evaluation.readiness import audit_research_inputs

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review", type=Path, action="append", default=[])
    p.add_argument("--observations", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.output.exists():
        p.error("Output exists; choose a new report path")
    report = audit_research_inputs(a.review, a.observations, a.seed)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                "data_contract_ready": report["ready_to_execute_real_benchmark_under_data_contract"],
                "blockers": report["blocking_evidence"],
            },
            indent=2,
        )
    )
    raise SystemExit(
        0
        if report["ready_to_execute_real_benchmark_under_data_contract"] and not report["blocking_evidence"]
        else 2
    )
