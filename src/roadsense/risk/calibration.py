# SPDX-License-Identifier: AGPL-3.0-only
"""Disjoint-set sigmoid probability calibration."""

import numpy as np
from sklearn.linear_model import LogisticRegression


class SigmoidCalibrator:
    def fit(self, scores, labels):
        if len(np.unique(labels)) != 2:
            raise ValueError("Calibration requires both classes")
        self.model = LogisticRegression(random_state=42).fit(np.asarray(scores).reshape(-1, 1), labels)
        return self

    def predict(self, scores):
        return self.model.predict_proba(np.asarray(scores).reshape(-1, 1))[:, 1]
