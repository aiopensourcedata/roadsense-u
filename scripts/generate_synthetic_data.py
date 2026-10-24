# SPDX-License-Identifier: AGPL-3.0-only
"""Generate a small, explicitly synthetic crossing video plus causal benchmark rows."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def generate(output: str, seed: int = 42):
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(directory / "crossing.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 20, (640, 480))
    if not writer.isOpened():
        raise RuntimeError("Synthetic video encoder unavailable")
    truth = []
    try:
        for i in range(100):
            t = i / 20
            image = np.full((480, 640, 3), 35, np.uint8)
            cv2.line(image, (0, 240), (640, 240), (90, 90, 90), 80)
            cv2.line(image, (320, 0), (320, 480), (90, 90, 90), 80)
            # Ground-contact points cross at t=3.0 seconds.
            a, b = (int(80 + 80 * t), 240), (320, int(30 + 70 * t))
            for point, color in [(a, (0, 0, 240)), (b, (0, 240, 0))]:
                x, y = point
                cv2.rectangle(image, (x - 14, y - 20), (x + 14, y), color, -1)
            cv2.putText(image, "SYNTHETIC SOFTWARE FIXTURE", (10, 35), 0, 0.7, (255, 255, 255), 2)
            writer.write(image)
            truth.append({"timestamp_s": t, "car": a, "bicycle": b, "synthetic": True})
    finally:
        writer.release()
    (directory / "truth.json").write_text(json.dumps(truth))
    (directory / "calibration.json").write_text(
        json.dumps(
            {
                "image_points": [[0, 0], [639, 0], [639, 479], [0, 479]],
                "world_points": [[0, 0], [63.9, 0], [63.9, 47.9], [0, 47.9]],
                "uncertainty_m": 0.1,
                "validation_image_points": [[100, 100], [500, 350]],
                "validation_world_points": [[10, 10], [50, 35]],
            }
        )
    )
    rng = np.random.default_rng(seed)
    fields = [
        "event_id",
        "clip_id",
        "location_id",
        "prediction_time_s",
        "event_time_s",
        "label",
        "cpa_distance",
        "time_to_cpa_s",
        "closing_speed",
        "quality",
        "synthetic",
    ]
    with (directory / "benchmark.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for location in range(8):
            for event in range(60):
                # Fixed-horizon target defined from a future simulated minimum separation with noise.
                cpa = float(rng.uniform(0, 120))
                tcpa = float(rng.uniform(0.1, 5))
                closing = float(rng.uniform(-10, 100))
                y = int(cpa + rng.normal(0, 8) < 25 and tcpa < 3 and closing > 0)
                w.writerow(
                    dict(
                        zip(
                            fields,
                            [
                                f"L{location}E{event}",
                                f"L{location}C{event}",
                                f"L{location}",
                                0,
                                tcpa if y else "",
                                y,
                                cpa,
                                tcpa,
                                closing,
                                float(rng.uniform(0.5, 1)),
                                True,
                            ],
                        )
                    )
                )
    print(f"Wrote synthetic video, geometry and benchmark fixtures to {directory}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="data/synthetic")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    generate(a.output, a.seed)
