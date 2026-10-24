# SPDX-License-Identifier: AGPL-3.0-only
from test_features import state

from roadsense.motion.features import pair_features
from roadsense.risk.rules import estimate


def test_closing_and_abstention():
    a, b = state(1, [-5, 0], [5, 0]), state(2, [0, -5], [0, 5])
    f = pair_features(a, b)
    risk = estimate(f)
    assert risk.score > 0.5 and not risk.abstained
    b.quality = 0.1
    assert estimate(pair_features(a, b)).abstained
    b.quality = 1
    b.velocity = -b.velocity
    a.velocity = -a.velocity
    assert estimate(pair_features(a, b)).score == 0
