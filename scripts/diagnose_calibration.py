# SPDX-License-Identifier: AGPL-3.0-only
"""Describe a trusted saved scene model; never tune or overwrite its experiment."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from roadsense.evaluation.scene_benchmark import validate_scene_observations
from roadsense.provenance import file_sha256


def diagnose(root: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Choose a fresh diagnostic output")
    frame = pd.read_csv(root / "eligible-snapshots.csv")
    validate_scene_observations(frame)
    model_path = root / "scene-baseline.joblib"
    model = joblib.load(model_path)  # Trusted local artifacts only.
    for phase, attribute in [("train", "training_source_ids"), ("calibration", "calibration_source_ids")]:
        if sorted(frame.loc[frame.partition == phase, "source_id"].unique()) != getattr(model, attribute):
            raise ValueError("Model source provenance disagrees with observations")
    prior = float(frame.loc[frame.partition == "train", "label"].mean())
    result = {
        "scope": "Post-hoc descriptive diagnosis of inspected sources; no fresh held-out claim or threshold tuning",
        "model_sha256": file_sha256(model_path),
        "observations_sha256": file_sha256(root / "eligible-snapshots.csv"),
        "partitions": {},
    }
    for phase, all_rows in frame.groupby("partition"):
        rows = all_rows.loc[all_rows.scoreable]
        summary = {
            "eligible": len(all_rows),
            "accepted": len(rows),
            "positive_accepted": int(rows.label.sum()),
            "baselines": {},
        }
        if len(rows):
            raw = model.estimator.predict_proba(rows[model.estimator.feature_names_in_])[:, 1]
            for name, values in [
                ("uncalibrated_logistic", raw),
                ("calibrated_logistic", model.predict(rows)),
                ("constant_half", np.full(len(rows), 0.5)),
                ("training_prior", np.full(len(rows), prior)),
            ]:
                summary["baselines"][name] = {
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "brier": float(brier_score_loss(rows.label, values)),
                    "ap": float(average_precision_score(rows.label, values))
                    if rows.label.nunique() == 2
                    else None,
                    "scope": "accepted-only; fitting/calibration rows are not independent evaluation",
                }
        result["partitions"][phase] = summary
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.evaluation, args.output), indent=2))
