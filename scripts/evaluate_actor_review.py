# SPDX-License-Identifier: AGPL-3.0-only
"""Evaluate complete actor box annotations, with explicit review provenance."""

import argparse
import json
from pathlib import Path

from roadsense.evaluation.actor_review import evaluate_actor_review


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--predictions", type=Path, help="Additional comparison stages bound to the same manifest"
    )
    parser.add_argument(
        "--allow-first-pass", action="store_true", help="Exploratory only; not adjudicated truth"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output file")
    result = evaluate_actor_review(args.bundle, args.review, args.allow_first_pass, args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "stages"}, indent=2))


if __name__ == "__main__":
    main()
