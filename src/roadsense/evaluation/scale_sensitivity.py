# SPDX-License-Identifier: AGPL-3.0-only
"""Coordinate perturbations of cached pixel measurements, not resized-video validation."""

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

from roadsense.cloud.schemas import FeatureSummary
from roadsense.config import PipelineConfig
from roadsense.motion.features import PairFeatures
from roadsense.risk.rules import estimate


def rescale_features(features: PairFeatures, factor: float) -> PairFeatures:
    """Change pixel coordinate scale; hold time, angles and dimensionless quality fixed."""
    if features.units != "px" or not math.isfinite(factor) or factor <= 0:
        raise ValueError("Coordinate sensitivity requires pixel features and a positive finite factor")
    spatial = (
        "distance",
        "relative_speed",
        "closing_speed",
        "cpa_distance",
        "measurement_uncertainty",
        "braking_a",
        "braking_b",
    )
    return replace(features, **{name: getattr(features, name) * factor for name in spatial})


def compare_scale_sensitivity(reference: Path, output: Path) -> dict:
    """Compare 0.5x/1x/2x coordinates with fixed and proportionally scaled proximity.

    Scores/acceptance are checked against the original current-policy cache at
    1x. Threshold changes are per observation before cooldown, not alert counts.
    Identity flags remain fixed: uniform coordinate rescaling preserves their
    dimensionless geometry by construction. No detector/quality model is rerun.
    """
    if output.exists():
        raise ValueError("Output exists; choose a fresh directory")
    report = json.loads((reference / "report.json").read_text())
    hashes = {
        name: hashlib.sha256((reference / name).read_bytes()).hexdigest()
        for name in ("report.json", "features.jsonl")
    }
    cfg = PipelineConfig(**report["config"])
    modes = {
        f"{scale:g}x_{kind}": {
            "coordinate_factor": scale,
            "proximity_factor": scale if kind == "normalized" else 1,
            "accepted": 0,
            "acceptance_changes": 0,
            "jointly_accepted": 0,
            "absolute_score_difference_sum": 0.0,
            "maximum_absolute_score_difference": 0.0,
            "threshold_decision_changes_among_jointly_accepted": 0,
        }
        for scale in (0.5, 1.0, 2.0)
        for kind in ("fixed", "normalized")
    }
    count = 0
    with (reference / "features.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["synthetic"] != report["synthetic"]:
                raise ValueError("Feature provenance mismatch")
            feature = PairFeatures(**FeatureSummary.model_validate(row["features"]).model_dump())
            identity_abstention = row["reasons"] == ["ambiguous_pair_identity"]
            if identity_abstention and not cfg.pair_identity.enabled:
                raise ValueError("Identity abstention in unguarded reference")
            base = estimate(feature, cfg.horizon_s, cfg.proximity_px, cfg.min_quality)
            accepted = not base.abstained and not identity_abstention
            if (
                row["abstained"] != (not accepted)
                or (not accepted and row["risk_score"] is not None)
                or (
                    accepted
                    and (
                        row["risk_score"] is None
                        or not math.isclose(base.score, row["risk_score"], abs_tol=1e-10, rel_tol=1e-10)
                    )
                )
            ):
                raise ValueError("Original current-policy decisions could not be reproduced")
            count += 1
            for mode in modes.values():
                risk = estimate(
                    rescale_features(feature, mode["coordinate_factor"]),
                    cfg.horizon_s,
                    cfg.proximity_px * mode["proximity_factor"],
                    cfg.min_quality,
                )
                chosen = not risk.abstained and not identity_abstention
                mode["accepted"] += chosen
                mode["acceptance_changes"] += chosen != accepted
                if accepted and chosen:
                    difference = abs(risk.score - base.score)
                    mode["jointly_accepted"] += 1
                    mode["absolute_score_difference_sum"] += difference
                    mode["maximum_absolute_score_difference"] = max(
                        mode["maximum_absolute_score_difference"],
                        difference,
                    )
                    mode["threshold_decision_changes_among_jointly_accepted"] += (
                        risk.score >= cfg.alert_threshold
                    ) != (base.score >= cfg.alert_threshold)
    for mode in modes.values():
        difference_sum = mode.pop("absolute_score_difference_sum")
        mode["mean_absolute_score_difference_jointly_accepted"] = (
            difference_sum / mode["jointly_accepted"] if mode["jointly_accepted"] else None
        )
    if modes["1x_fixed"]["accepted"] != count - report["abstained_pair_observations"]:
        raise ValueError("Reference observation counts could not be reproduced")
    if any(
        hashlib.sha256((reference / name).read_bytes()).hexdigest() != digest
        for name, digest in hashes.items()
    ):
        raise ValueError("Reference artifacts changed during scale comparison")
    result = {
        "experiment": "counterfactual_uniform_coordinate_scale",
        "source_run": str(reference),
        "synthetic": report["synthetic"],
        "detector_injected_for_test": report.get("detector_injected_for_test", False),
        "source_sha256": hashes,
        "config": cfg.model_dump(),
        "observations": count,
        "original_decisions_reproduced": True,
        "conditions": modes,
        "limitations": [
            "Synthetic coordinate perturbation of measurements; not actual image resizing or a new camera.",
            "Quality and identity held fixed; perception error/perspective distortion are not simulated.",
            "Normalized invariance is algebraic by construction, not a learned improvement or novelty.",
            "No outcome labels or independent camera units: no accuracy or inferential interval.",
            "Threshold-decision changes are correlated observations before deployment cooldown.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
