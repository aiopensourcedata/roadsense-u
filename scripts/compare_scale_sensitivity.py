# SPDX-License-Identifier: AGPL-3.0-only
"""Measure counterfactual pixel-scale sensitivity on cached measurements."""

import argparse
import json
from pathlib import Path

from roadsense.evaluation.scale_sensitivity import compare_scale_sensitivity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_scale_sensitivity(args.reference, args.output)
    print(
        json.dumps(
            {
                "experiment": result["experiment"],
                "observations": result["observations"],
                "original_decisions_reproduced": result["original_decisions_reproduced"],
            },
            indent=2,
        )
    )
