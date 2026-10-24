# SPDX-License-Identifier: AGPL-3.0-only
import argparse

from roadsense.evaluation.benchmark import benchmark

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fit, calibrate and evaluate with disjoint groups")
    p.add_argument("csv")
    p.add_argument("--output", default="outputs/benchmark")
    p.add_argument("--split", choices=["event", "location"], default="location")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    r = benchmark(a.csv, a.output, a.split, a.seed)
    print(
        {
            k: {
                m: v
                for m, v in values.items()
                if m in ["precision", "recall", "brier_score", "pr_auc_average_precision"]
            }
            for k, values in r["baselines"].items()
        }
    )
