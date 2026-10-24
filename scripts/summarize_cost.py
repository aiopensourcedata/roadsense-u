# SPDX-License-Identifier: AGPL-3.0-only
"""Calculate an approximate scenario cost using explicit user-provided tariffs."""

import argparse
import json
from pathlib import Path

from roadsense.evaluation.metrics import approximate_cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("--cpu-hour-price", type=float, required=True)
    parser.add_argument("--egress-gb-price", type=float, required=True)
    args = parser.parse_args()
    if min(args.cpu_hour_price, args.egress_gb_price) < 0:
        parser.error("Tariffs must be nonnegative")
    report = json.loads(Path(args.report).read_text())
    print(
        json.dumps(
            approximate_cost(
                report["cpu_time_s"],
                report["network_body_bytes_attempted"],
                args.cpu_hour_price,
                args.egress_gb_price,
            ),
            indent=2,
        )
    )
