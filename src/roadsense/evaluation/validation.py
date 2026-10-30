# SPDX-License-Identifier: AGPL-3.0-only
"""Validate the causal, single-observation, pixel-space baseline data contract."""

import numpy as np
import pandas as pd

from roadsense.risk.classical_model import FEATURES


def load_observations(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"event_id": str, "clip_id": str, "location_id": str})
    required = [
        "event_id",
        "clip_id",
        "location_id",
        "prediction_time_s",
        "event_time_s",
        "label",
        *FEATURES,
        "synthetic",
    ]
    if any(key not in frame for key in required) or frame.empty:
        raise ValueError(f"A nonempty table with required columns is needed: {required}")
    provenance = (
        frame.synthetic.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    )
    if provenance.isna().any():
        raise ValueError("Synthetic provenance must be explicit true/false")
    if provenance.nunique() != 1:
        raise ValueError("Do not pool synthetic and real observations in one benchmark")
    frame["synthetic"] = provenance
    coverage = {
        "provided": "future_observation_complete" in frame,
        "input_rows": len(frame),
        "excluded_rows": 0,
        "excluded_rows_by_location": {},
        "excluded_reasons": {},
    }
    if coverage["provided"]:
        complete = (
            frame.future_observation_complete.astype(str)
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )
        if complete.isna().any():
            raise ValueError("Future observation coverage must be explicit true/false")
        coverage["excluded_rows"] = int((~complete).sum())
        coverage["excluded_rows_by_location"] = (
            frame.loc[~complete, "location_id"].fillna("missing").value_counts().to_dict()
        )
        if "future_observation_reason" in frame:
            coverage["excluded_reasons"] = (
                frame.loc[~complete, "future_observation_reason"]
                .fillna("unspecified")
                .value_counts()
                .to_dict()
            )
        frame = frame.loc[complete].copy()
        if frame.empty:
            raise ValueError("No observations have a complete future horizon")
    elif not bool(provenance.iloc[0]):
        raise ValueError("Real observations require explicit future_observation_complete coverage")
    coverage["retained_rows"] = len(frame)
    frame.attrs["observation_coverage"] = coverage
    for key in ("event_id", "clip_id", "location_id"):
        if frame[key].isna().any() or frame[key].str.strip().eq("").any():
            raise ValueError(f"Missing or empty {key}")
    if frame.event_id.duplicated().any():
        raise ValueError("Probability baseline requires one prespecified prediction timestamp per event")
    if not frame.label.isin([0, 1]).all():
        raise ValueError("Target labels must be binary")
    for column in ("scoreable", "identity_ambiguous"):
        if column in frame:
            parsed = (
                frame[column]
                .astype(str)
                .str.lower()
                .map({"true": True, "false": False, "1": True, "0": False})
            )
            if parsed.isna().any():
                raise ValueError(f"{column} must be explicit true/false")
            frame[column] = parsed
    available = frame.get("scoreable", pd.Series(True, index=frame.index))
    numeric = frame.loc[available, FEATURES].to_numpy(dtype=float)
    if not np.isfinite(frame.prediction_time_s.to_numpy(dtype=float)).all() or not np.isfinite(numeric).all():
        raise ValueError("Prediction timestamps and available features must be finite")
    if "measurement_uncertainty" in frame:
        uncertainty = frame.loc[available, "measurement_uncertainty"].to_numpy(dtype=float)
        if not np.isfinite(uncertainty).all() or (uncertainty < 0).any():
            raise ValueError("Available uncertainty must be finite and nonnegative")
    if (
        (frame.prediction_time_s < 0).any()
        or (frame.cpa_distance < 0).any()
        or (frame.time_to_cpa_s < 0).any()
    ):
        raise ValueError("Prediction times, CPA distances and times must be nonnegative")
    if not frame.loc[available, "quality"].between(0, 1).all():
        raise ValueError("Track quality must lie in [0,1]")
    lead = frame.event_time_s - frame.prediction_time_s
    positive = frame.label == 1
    if not np.isfinite(lead[positive]).all() or not lead[positive].between(0, 3, inclusive="right").all():
        raise ValueError("Positive target must occur in (prediction_time, prediction_time+3 seconds]")
    if lead[~positive].between(0, 3, inclusive="right").any():
        raise ValueError("Negative target contradicts an event inside the prediction horizon")
    if "units" in frame and not frame.units.eq("px").all():
        raise ValueError("The current baseline requires pixel-space features; metric pooling is unsupported")
    return frame


def validate_holdout(model, frame: pd.DataFrame) -> None:
    """Fail closed if a trusted model lacks split provenance or known IDs overlap."""
    for column, attribute in [
        ("event_id", "used_event_ids"),
        ("clip_id", "used_clip_ids"),
        ("location_id", "used_location_ids"),
    ]:
        if not hasattr(model, attribute):
            raise ValueError("Model lacks split provenance; retrain before held-out evaluation")
        if set(frame[column].astype(str)) & getattr(model, attribute):
            raise ValueError(f"Held-out evaluation rejected: previously used {column}")
