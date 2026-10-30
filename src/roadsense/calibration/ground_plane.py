# SPDX-License-Identifier: AGPL-3.0-only
"""Manual planar homography. Metric results are restricted to the calibrated image hull."""

import json
from pathlib import Path

import cv2
import numpy as np

from roadsense.calibration.validation import validate_points


class GroundPlane:
    def __init__(
        self,
        image_points,
        world_points,
        uncertainty_m: float,
        validation_image_points,
        validation_world_points,
        max_error_m: float = 1.0,
        binding: dict | None = None,
    ):
        if not np.isfinite(max_error_m) or max_error_m <= 0:
            raise ValueError("Calibration validation tolerance must be positive and finite")
        self.binding = binding
        image, world = validate_points(image_points, world_points)
        if not np.isfinite(uncertainty_m) or uncertainty_m <= 0:
            raise ValueError("A positive declared calibration uncertainty in meters is required")
        self.h, _ = cv2.findHomography(image, world, method=0)
        if self.h is None or not np.isfinite(self.h).all() or np.linalg.matrix_rank(self.h) != 3:
            raise ValueError("Invalid homography")
        self.hull = cv2.convexHull(image.astype("float32"))
        vi, vw = np.asarray(validation_image_points, float), np.asarray(validation_world_points, float)
        if vi.shape != vw.shape or vi.ndim != 2 or vi.shape[1] != 2 or len(vi) < 2:
            raise ValueError("Two or more independent validation points are required")
        if not np.isfinite(vi).all() or not np.isfinite(vw).all():
            raise ValueError("Validation points must be finite")
        if len(np.unique(vi, axis=0)) != len(vi) or len(np.unique(vw, axis=0)) != len(vw):
            raise ValueError("Validation points must be distinct")
        if any(np.linalg.norm(image - p, axis=1).min() < 1e-6 for p in vi):
            raise ValueError("Validation points must be independent of fitting points")
        projected = cv2.perspectiveTransform(vi.reshape(-1, 1, 2), self.h).reshape(-1, 2)
        self.validation_rmse_m = float(np.sqrt(np.mean(np.sum((projected - vw) ** 2, axis=1))))
        if not np.isfinite(self.validation_rmse_m) or self.validation_rmse_m > max_error_m:
            raise ValueError("Calibration fails independent validation tolerance")
        self.uncertainty_m = max(uncertainty_m, self.validation_rmse_m)

    def validate_binding(self, camera_id: str, width: int, height: int, pose_version: str):
        """Fail closed before metric measurements on mismatched acquisition geometry."""
        expected = {
            "camera_id": camera_id,
            "width": width,
            "height": height,
            "image_transform": "decoded_native",
            "camera_pose_version": pose_version,
        }
        if pose_version == "unspecified" or self.binding != expected:
            raise ValueError(
                "Calibration requires matching camera, native resolution, transform and explicit pose version"
            )

    def project(self, point: np.ndarray) -> np.ndarray | None:
        point = np.asarray(point, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("Projection requires one finite 2D image point")
        if cv2.pointPolygonTest(self.hull, tuple(map(float, point)), False) < 0:
            return None
        q = self.h @ np.array([*point, 1.0])
        if abs(q[2]) < 1e-9:
            return None
        result = q[:2] / q[2]
        return result if np.isfinite(result).all() else None

    @classmethod
    def load(cls, path: str):
        return cls(**json.loads(Path(path).read_text()))
