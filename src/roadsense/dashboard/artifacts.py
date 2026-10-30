# SPDX-License-Identifier: AGPL-3.0-only
"""Local run discovery and measured/projection chart rows, without retaining imagery."""

import json
from pathlib import Path


def processed_runs(root: Path) -> list[Path]:
    """Find nested finalized pipeline reports, including explicit partial runs; ignore other research report formats."""
    runs = []
    for path in root.rglob("report.json"):
        try:
            report = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(report, dict) and {"config", "frames", "throughput_fps", "events"} <= report.keys():
            runs.append(path.parent)
    return sorted(runs, key=lambda p: (p / "report.json").stat().st_mtime, reverse=True)


def projection_rows(projection: dict) -> list[dict]:
    """Two endpoints per hypothesized participant; keep current and future distinct."""
    return [
        {"Track": str(tid), "Stage": stage, "Order": order, "x": point[0], "y": point[1]}
        for tid, current, future in zip(
            projection["track_ids"],
            projection["positions_now"],
            projection["projected_positions"],
            strict=True,
        )
        for order, (stage, point) in enumerate([("Measured anchor", current), ("Projected anchor", future)])
    ]
