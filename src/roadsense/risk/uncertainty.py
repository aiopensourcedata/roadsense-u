# SPDX-License-Identifier: AGPL-3.0-only
"""Measurement quality abstention; quality is not a calibrated confidence probability."""

from roadsense.motion.features import PairFeatures


def abstention_reason(features: PairFeatures, min_quality: float, proximity: float) -> str | None:
    if features.quality < min_quality:
        return "insufficient_track_quality"
    if features.measurement_uncertainty > proximity:
        return "measurement_uncertainty_exceeds_proximity_scale"
    return None
