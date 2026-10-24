# SPDX-License-Identifier: AGPL-3.0-only
"""Descriptive aggregates; normalized rates require matching validated exposure windows."""

from collections import Counter, defaultdict
from datetime import datetime, timezone


def aggregate(events: list[dict], exposures: list[dict]) -> dict:
    by_camera, by_location, by_day, patterns = Counter(), Counter(), Counter(), Counter()
    scores = defaultdict(list)
    for e in events:
        by_camera[e["camera_id"]] += 1
        by_location[e["location_id"]] += 1
        by_day[datetime.fromisoformat(e["timestamp"]).astimezone(timezone.utc).date().isoformat()] += 1
        patterns[" + ".join(sorted(e["classes"]))] += 1
        scores[e["camera_id"]].append(e["risk_score"])
    normalized = {}
    cameras = set(by_camera) | {x["camera_id"] for x in exposures}
    for camera in cameras:
        windows = [x for x in exposures if x["camera_id"] == camera]
        volume = sum(x["road_users"] for x in windows)
        covered = sum(
            any(
                w["location_id"] == e["location_id"]
                and datetime.fromisoformat(w["start"])
                <= datetime.fromisoformat(e["timestamp"])
                < datetime.fromisoformat(w["end"])
                for w in windows
            )
            for e in events
            if e["camera_id"] == camera
        )
        normalized[camera] = {
            "covered_events": covered,
            "road_users": volume,
            "events_per_1000_road_users": 1000 * covered / volume if volume else None,
        }
    days = sorted(by_day)
    return {
        "event_count": len(events),
        "by_camera": dict(by_camera),
        "by_location": dict(by_location),
        "daily_counts": dict(by_day),
        "interaction_patterns": dict(patterns),
        "last_observed_day_count_change": by_day[days[-1]] - by_day[days[-2]] if len(days) > 1 else None,
        "mean_score_by_camera": {k: sum(v) / len(v) for k, v in scores.items()},
        "exposure_normalized": normalized,
        "interpretation": "Alert counts, not labeled collisions. Counts depend on traffic, detection and cooldown.",
    }
