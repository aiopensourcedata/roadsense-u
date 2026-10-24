# SPDX-License-Identifier: AGPL-3.0-only
"""Interpretable logistic baseline; artifact loading is only safe for trusted local files."""

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from roadsense.risk.calibration import SigmoidCalibrator

FEATURES = ["cpa_distance", "time_to_cpa_s", "closing_speed", "quality"]


class ClassicalModel:
    def __init__(self):
        self.estimator = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=42),
        )
        self.calibrator = SigmoidCalibrator()

    def fit(self, train, calibration):
        if train.label.nunique() != 2:
            raise ValueError("Training requires both target classes")
        self.estimator.fit(train[FEATURES], train.label)
        self.calibrator.fit(self.estimator.decision_function(calibration[FEATURES]), calibration.label)
        self.feature_min = train[FEATURES].min().to_numpy()
        self.feature_max = train[FEATURES].max().to_numpy()
        return self

    def predict(self, frame):
        return self.calibrator.predict(self.estimator.decision_function(frame[FEATURES]))

    def out_of_range(self, frame):
        """Simple range abstention heuristic, not a complete OOD detector."""
        values = frame[FEATURES].to_numpy()
        return np.any(
            (values < self.feature_min) | (values > self.feature_max) | ~np.isfinite(values), axis=1
        )

    def save(self, path: str):
        joblib.dump(self, path)
