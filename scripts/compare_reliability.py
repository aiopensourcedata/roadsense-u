# SPDX-License-Identifier: AGPL-3.0-only
"""Run five fixed reliability policies on the same cached measurements; no accuracy claims."""

import argparse
import json
from pathlib import Path

from roadsense.evaluation.reliability import compare_reliability

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_reliability(args.reference, args.output)
    print(
        json.dumps(
            {
                "synthetic": result["synthetic"],
                "reference_reproduced": result["reference_decisions_scores_alerts_reproduced"],
                "reported_pair_observations": result["reported_pair_observations"],
                "policies": {
                    k: {"accepted": v["accepted"], "emissions": v["emissions"]}
                    for k, v in result["policies"].items()
                },
            },
            indent=2,
        )
    )
