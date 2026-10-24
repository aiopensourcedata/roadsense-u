# SPDX-License-Identifier: AGPL-3.0-only
"""Evaluate a trusted fitted model on a new held-out CSV; threshold is provided, never fit here."""

import argparse
import json
from pathlib import Path

import joblib

from roadsense.evaluation.benchmark import event_table
from roadsense.evaluation.metrics import early_warning, metrics
from roadsense.evaluation.validation import load_observations, validate_holdout

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model")
    p.add_argument("csv")
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--output", default="outputs/heldout-evaluation.json")
    a = p.parse_args()
    model = joblib.load(a.model)  # Never load untrusted pickle/joblib artifacts.
    frame = load_observations(a.csv)
    validate_holdout(model, frame)
    predictions = model.predict(frame)
    events = event_table(frame, predictions)
    result = metrics(events.label, events.score, a.threshold)
    result["early_warning"] = early_warning(frame, predictions, a.threshold)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))
