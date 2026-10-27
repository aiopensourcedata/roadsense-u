# SPDX-License-Identifier: AGPL-3.0-only
import json
from pathlib import Path

import pytest

from roadsense.dashboard.artifacts import processed_runs, projection_rows


def test_nested_run_discovery_ignores_metrics_and_partial_reports(tmp_path):
    run = tmp_path / "development" / "clip-00" / "960"
    run.mkdir(parents=True)
    (run / "report.json").write_text(
        json.dumps(
            {
                "config": {},
                "frames": 3,
                "events": 1,
                "throughput_fps": 2,
            }
        )
    )
    (tmp_path / "report.json").write_text('{"precision": 0.5}')
    broken = tmp_path / "partial"
    broken.mkdir()
    (broken / "report.json").write_text('{"frames":')
    other = tmp_path / "other"
    other.mkdir()
    (other / "report.json").write_text("[]")
    assert processed_runs(tmp_path) == [run]


def test_geometry_rows_separate_measured_and_projected_anchors():
    rows = projection_rows(
        {
            "track_ids": [2, 3],
            "positions_now": [[1, 2], [3, 4]],
            "projected_positions": [[5, 6], [7, 8]],
        }
    )
    assert len(rows) == 4
    assert [r["Stage"] for r in rows[:2]] == ["Measured anchor", "Projected anchor"]
    assert rows[0]["Track"] == rows[1]["Track"] == "2"
    assert rows[-1]["x"] == 7


def test_projection_panel_renders_actual_nested_artifacts(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "src/roadsense/dashboard/app.py"
    run = tmp_path / "outputs" / "nested" / "run"
    run.mkdir(parents=True)
    (run / "report.json").write_text(
        json.dumps(
            {
                "config": {},
                "frames": 3,
                "events": 1,
                "throughput_fps": 2,
                "synthetic": True,
            }
        )
    )
    projection = {
        "timestamp_s": 1,
        "offset_s": 2,
        "units": "px",
        "track_ids": [2, 3],
        "positions_now": [[1, 2], [3, 4]],
        "projected_positions": [[5, 6], [7, 8]],
    }
    (run / "event_geometry.jsonl").write_text(
        json.dumps({"event_id": "test-event", "projection": projection})
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ROADSENSE_API_URL", "http://127.0.0.1:1")
    app = AppTest.from_file(str(app_path)).run(timeout=30)
    assert not app.exception
    assert any("Unverified track pair" in warning.value for warning in app.warning)
    assert any("test-event" in value.value for value in app.json)
    charts = app.get("vega_lite_chart")
    assert len(charts) == 1
    spec = json.loads(charts[0].proto.spec)
    assert spec["layer"][0]["encoding"]["y"]["scale"]["reverse"]
    assert spec["layer"][1]["encoding"]["shape"]["field"] == "Stage"
