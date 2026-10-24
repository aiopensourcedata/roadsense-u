# SPDX-License-Identifier: AGPL-3.0-only
"""Ground-plane correspondence validation."""

import cv2
import numpy as np


def validate_points(image_points, world_points):
    image, world = np.asarray(image_points, dtype=float), np.asarray(world_points, dtype=float)
    if image.shape != world.shape or image.ndim != 2 or image.shape[1] != 2 or len(image) < 4:
        raise ValueError("At least four paired 2D correspondences are required")
    for points in (image, world):
        if not np.isfinite(points).all() or len(np.unique(points, axis=0)) != len(points):
            raise ValueError("Correspondences must be finite and distinct")
        if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
            raise ValueError("Collinear calibration points")
        if cv2.contourArea(cv2.convexHull(points.astype("float32"))) <= 1e-6:
            raise ValueError("Degenerate ground plane")
    return image, world
