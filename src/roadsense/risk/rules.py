# SPDX-License-Identifier: AGPL-3.0-only
"""Interpretable geometric score. This is NOT a probability of collision."""

import math
from dataclasses import dataclass

from roadsense.motion.features import PairFeatures
from roadsense.risk.uncertainty import abstention_reason


@dataclass
class RiskEstimate:
    score: float | None
    abstained: bool
    reasons: list[str]
    model_version: str = "rules-0.1"
    score_kind: str = "heuristic"


def estimate(
    features: PairFeatures,
    horizon_s: float = 3,
    proximity: float = 60,
    min_quality: float = 0.5,
    identity_ambiguous: bool = False,
) -> RiskEstimate:
    if not math.isfinite(horizon_s) or not math.isfinite(proximity) or min(horizon_s, proximity) <= 0:
        raise ValueError("Risk horizon and proximity must be positive and finite")
    if not 0 <= min_quality <= 1:
        raise ValueError("Minimum quality must lie in [0,1]")
    if identity_ambiguous:
        return RiskEstimate(None, True, ["ambiguous_pair_identity"])
    reason = abstention_reason(features, min_quality, proximity)
    if reason:
        return RiskEstimate(None, True, [reason])
    if features.closing_speed <= 0 or features.time_to_cpa_s <= 0:
        return RiskEstimate(0.0, False, ["no_future_closing_motion"])
    spatial = math.exp(-0.5 * (features.cpa_distance / proximity) ** 2)
    temporal = max(0.0, 1 - features.time_to_cpa_s / horizon_s)
    score = spatial * temporal
    reasons = [
        f"CPA={features.cpa_distance:.2f} {features.units}",
        f"time_to_CPA={features.time_to_cpa_s:.2f} s",
        f"closing_speed={features.closing_speed:.2f} {features.units}/s",
    ]
    if features.ttc_s is not None:
        reasons.append(f"disk_TTC={features.ttc_s:.2f} s")
    return RiskEstimate(float(score), False, reasons)
