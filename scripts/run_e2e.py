# SPDX-License-Identifier: AGPL-3.0-only
"""Exercise video -> authenticated API -> PostgreSQL -> real browser, with optional README screenshots.

Requires the local Compose stack, the e2e/dashboard extras and Chrome or Playwright Chromium.
Only generated synthetic scenes are captured. Credentials are never embedded in pages or screenshots.
"""

import argparse
import hashlib
import json
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import httpx
from generate_synthetic_data import generate
from playwright.sync_api import expect, sync_playwright
from sqlalchemy import select

from roadsense import __version__
from roadsense.camera.preview import LivePreview
from roadsense.cloud.client import EventClient
from roadsense.cloud.database import EventRow, connect
from roadsense.cloud.schemas import Exposure, TrafficEvent
from roadsense.config import PipelineConfig, Settings
from roadsense.evaluation.benchmark import benchmark
from roadsense.pipeline import run

SCREENSHOTS = {
    "overview.png": "Computed synthetic alerts retrieved from PostgreSQL and shown in the dashboard.",
    "event-evidence.png": "Measured features and the heuristic explanation for a stored synthetic alert.",
    "location-analytics.png": "Synthetic camera counts and rates using explicitly supplied synthetic exposure.",
    "evaluation.png": "An actually executed offline synthetic benchmark, not real-world model validation.",
    "live-preview.png": "Browser MJPEG preview of a currently processed, annotated synthetic video frame.",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard-url", default="http://127.0.0.1:8501")
    parser.add_argument(
        "--screenshots", type=Path, help="Explicitly write synthetic browser screenshots to this directory"
    )
    parser.add_argument("--browser-executable", help="Optional Chrome/Chromium executable path")
    args = parser.parse_args()
    cfg = Settings()
    if not cfg.database_url.startswith("postgresql"):
        raise SystemExit("End-to-end verification requires actual PostgreSQL, not SQLite")
    run_id = uuid4().hex[:8]
    directory = Path("outputs") / f"e2e-{run_id}"
    directory.mkdir(parents=True)
    input_directory = directory / "input"
    generate(str(input_directory))
    location = f"synthetic-e2e-{run_id}"
    metadata = EventClient(cfg.api_url, cfg.api_key)
    engine, factory = connect(cfg.database_url)
    checks: dict[str, bool] = {}
    screenshots: list[dict] = []
    stored: list[TrafficEvent] = []
    base = datetime.now(timezone.utc) - timedelta(hours=3)
    headers = {"X-API-Key": cfg.api_key}
    try:
        with httpx.Client(trust_env=False, base_url=cfg.api_url, timeout=10) as http:
            http.get("/health").raise_for_status()
            if http.get("/events").status_code != 401:
                raise AssertionError("Protected API accepted an unauthenticated request")
            if http.post("/events", json={}, headers=headers).status_code != 422:
                raise AssertionError("Malformed event was accepted")
            if http.post("/events", content=b"x" * 70000, headers=headers).status_code != 413:
                raise AssertionError("Request-size limit did not reject an oversized body")
            checks["authentication_validation_size_limit"] = True
            for camera in ("north", "east"):
                for episode in range(2):
                    start = base + timedelta(minutes=30 * episode + (5 if camera == "east" else 0))
                    output = Path("outputs") / f"e2e-{run_id}-{camera}-{episode}"
                    report = run(
                        str(input_directory / "crossing.avi"),
                        str(output),
                        PipelineConfig(
                            detector="synthetic",
                            synthetic=True,
                            camera_id=f"demo-{camera}-{run_id}",
                            location_id=location,
                            cooldown_s=0.75,
                        ),
                        metadata,
                        video_start=start,
                    )
                    if report["frames"] != 100 or report["events"] < 1 or report["submission_failures"]:
                        raise AssertionError("Video did not produce and submit computed alerts")
                    stored.extend(
                        TrafficEvent.model_validate_json(line)
                        for line in (output / "events.jsonl").read_text().splitlines()
                    )
                    # Two known generated agents per five-second episode, not a count of tracker IDs.
                    exposure = Exposure(
                        camera_id=f"demo-{camera}-{run_id}",
                        location_id=location,
                        start=start,
                        end=start + timedelta(seconds=5),
                        road_users=2,
                        method="manual_count",
                        synthetic=True,
                    )
                    http.post(
                        "/exposures", json=exposure.model_dump(mode="json"), headers=headers
                    ).raise_for_status()
            checks["video_motion_rules_http_submission"] = True
            ids = {str(e.event_id) for e in stored}
            with factory() as session:
                rows = list(session.scalars(select(EventRow).where(EventRow.location_id == location)))
                if {r.event_id for r in rows} != ids:
                    raise AssertionError("Independent PostgreSQL query differs from computed events")
            checks["independent_postgresql_persistence"] = True
            params = {"synthetic": True, "location_id": location, "limit": 1000}
            response = http.get("/events", params=params, headers=headers)
            response.raise_for_status()
            if {e["event_id"] for e in response.json()} != ids:
                raise AssertionError("HTTP retrieval differs from persisted event IDs")
            replay = http.post("/events", json=stored[0].model_dump(mode="json"), headers=headers)
            replay.raise_for_status()
            if replay.json()["created"]:
                raise AssertionError("Idempotent replay duplicated a stored event")
            checks["retrieval_and_idempotent_replay"] = True
            real = http.get("/events", params={**params, "synthetic": False}, headers=headers)
            real.raise_for_status()
            if real.json():
                raise AssertionError("Synthetic events leaked into real-data queries")
            checks["synthetic_real_isolation"] = True
            response = http.get(
                "/stats",
                params={
                    "synthetic": True,
                    "location_id": location,
                    "start": (base - timedelta(seconds=1)).isoformat(),
                    "end": datetime.now(timezone.utc).isoformat(),
                },
                headers=headers,
            )
            response.raise_for_status()
            stats = response.json()
            if stats["event_count"] != len(stored) or len(stats["by_camera"]) != 2:
                raise AssertionError("Aggregate counts differ from stored computed alerts")
            for rate in stats["exposure_normalized"].values():
                if rate["road_users"] != 4 or rate["events_per_1000_road_users"] is None:
                    raise AssertionError("Synthetic exposure normalization is missing")
            checks["camera_statistics_and_exposure"] = True

        evaluation_output = Path("outputs") / f"e2e-{run_id}-evaluation"
        experiment = benchmark(str(input_directory / "benchmark.csv"), str(evaluation_output))
        checks["offline_grouped_baseline_experiment"] = bool(experiment["synthetic"])
        with sync_playwright() as playwright:
            executable = args.browser_executable or shutil.which("google-chrome") or shutil.which("chromium")
            launch = {"headless": True, "args": ["--disable-dev-shm-usage"]}
            if executable:
                launch["executable_path"] = executable
            browser = playwright.chromium.launch(**launch)
            context = browser.new_context(
                viewport={"width": 1500, "height": 1320},
                device_scale_factor=1,
                color_scheme="dark",
                timezone_id="UTC",
                reduced_motion="reduce",
            )
            page = context.new_page()
            page_errors = []
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            response = page.goto(
                f"{args.dashboard_url}/?synthetic=true&location={location}", wait_until="domcontentloaded"
            )
            if response is None or response.status != 200:
                raise AssertionError("Dashboard page did not load")
            expect(
                page.get_by_role("heading", name="Traffic interaction intelligence", exact=True)
            ).to_be_visible(timeout=30000)
            expect(page.get_by_text("API and database connected", exact=True)).to_be_visible(timeout=30000)
            expect(page.get_by_role("heading", name="Interaction overview", exact=True)).to_be_visible()
            expect(
                page.get_by_text(
                    "SYNTHETIC DATA · Generated trajectories and measurements for software validation only.",
                    exact=True,
                )
            ).to_be_visible()
            metrics = page.locator('[data-testid="stMetric"]')
            if (metrics.nth(0).inner_text().splitlines()[-1]).replace(",", "") != str(len(stored)):
                raise AssertionError("Dashboard count does not match independently verified PostgreSQL rows")
            checks["browser_dashboard_matches_postgresql"] = True

            def capture(name: str) -> None:
                if args.screenshots is None:
                    return
                args.screenshots.mkdir(parents=True, exist_ok=True)
                page.evaluate("window.scrollTo(0, 0)")
                # Wait for chart canvases and web fonts, then capture actual browser pixels.
                page.evaluate("document.fonts.ready")
                page.screenshot(path=str(args.screenshots / name), animations="disabled", full_page=True)
                payload = (args.screenshots / name).read_bytes()
                screenshots.append(
                    {
                        "file": name,
                        "description": SCREENSHOTS[name],
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "synthetic": True,
                    }
                )

            expect(page.locator('[data-testid="stVegaLiteChart"]').first).to_be_visible(timeout=20000)
            capture("overview.png")
            toggle = page.get_by_role("checkbox", name="Include synthetic fixtures (only)", exact=True)
            toggle.press("Space")
            expect(toggle).not_to_be_checked()
            expect(
                page.get_by_text(
                    "No matching events stored. Process a permitted video or enable the synthetic fixture filter.",
                    exact=True,
                )
            ).to_be_visible(timeout=20000)
            toggle.press("Space")
            expect(toggle).to_be_checked()
            expect(page.get_by_role("heading", name="Interaction overview", exact=True)).to_be_visible()
            expect(page.locator('[data-testid="stMetric"]').nth(0)).to_contain_text(str(len(stored)))
            checks["browser_provenance_filter"] = True
            page.get_by_role("tab", name="Event evidence", exact=True).click()
            expect(page.get_by_role("heading", name="Evidence behind an alert", exact=True)).to_be_visible()
            expect(page.get_by_text("Measured explanation", exact=True)).to_be_visible()
            capture("event-evidence.png")
            page.get_by_text("Event metadata and complete feature summary", exact=True).click()
            expect(page.get_by_text("tracking_epoch", exact=False).first).to_be_visible()
            checks["browser_event_evidence_and_metadata"] = True
            page.get_by_role("tab", name="Location analytics", exact=True).click()
            expect(page.get_by_text("Exposure-normalized alert rates", exact=True)).to_be_visible()
            capture("location-analytics.png")
            checks["browser_analytics"] = True
            page.get_by_role("tab", name="Processed runs", exact=True).click()
            expect(
                page.get_by_role("heading", name="Processed video and trajectories", exact=True)
            ).to_be_visible()
            expect(page.get_by_text("Processed frames", exact=True)).to_be_visible()
            checks["browser_processed_artifacts"] = True
            page.get_by_role("tab", name="Evaluation", exact=True).click()
            expect(page.get_by_role("heading", name="Offline model evaluation", exact=True)).to_be_visible()
            expect(
                page.get_by_text(
                    "SYNTHETIC BENCHMARK · These metrics do not establish real-world traffic performance or model superiority.",
                    exact=True,
                )
            ).to_be_visible()
            capture("evaluation.png")
            checks["browser_evaluation"] = True

            with LivePreview(port=0) as preview, ThreadPoolExecutor(max_workers=1) as pool:
                live_page = context.new_page()
                live_page.set_viewport_size({"width": 1100, "height": 900})
                live_page.goto(preview.url, wait_until="domcontentloaded")
                ready = threading.Event()

                class Observer:
                    count = 0

                    def update(self, image):
                        preview.update(image)
                        self.count += 1
                        if self.count >= 45:
                            ready.set()

                future = pool.submit(
                    run,
                    str(input_directory / "crossing.avi"),
                    str(directory / "preview-run"),
                    PipelineConfig(detector="synthetic", synthetic=True),
                    preview=Observer(),
                    realtime=True,
                )
                if not ready.wait(timeout=20):
                    raise AssertionError("Live processing did not publish expected frames")
                expect(live_page.get_by_role("heading", name="RoadSense-U live preview")).to_be_visible()
                live_page.wait_for_function("document.querySelector('img').naturalWidth > 0", timeout=10000)
                if args.screenshots is not None:
                    live_page.screenshot(
                        path=str(args.screenshots / "live-preview.png"), full_page=True, animations="disabled"
                    )
                    screenshots.append(
                        {
                            "file": "live-preview.png",
                            "description": SCREENSHOTS["live-preview.png"],
                            "sha256": hashlib.sha256(
                                (args.screenshots / "live-preview.png").read_bytes()
                            ).hexdigest(),
                            "synthetic": True,
                        }
                    )
                preview_report = future.result(timeout=20)
                if preview_report["frames"] != 100 or (directory / "preview-run/annotated.mp4").exists():
                    raise AssertionError("Preview did not process all frames without recording")
                live_page.close()
            checks["browser_live_mjpeg_without_recording"] = True
            if page_errors:
                raise AssertionError(f"Browser JavaScript errors: {page_errors}")
            checks["no_browser_javascript_errors"] = True
            mobile_context = browser.new_context(
                viewport={"width": 390, "height": 844},
                is_mobile=True,
                device_scale_factor=1,
                color_scheme="dark",
                timezone_id="UTC",
            )
            mobile = mobile_context.new_page()
            mobile.on("pageerror", lambda error: page_errors.append(str(error)))
            mobile.goto(
                f"{args.dashboard_url}/?synthetic=true&location={location}", wait_until="domcontentloaded"
            )
            expect(mobile.get_by_role("heading", name="Interaction overview", exact=True)).to_be_visible(
                timeout=30000
            )
            if not mobile.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 2"):
                raise AssertionError("Dashboard overflows the mobile viewport")
            if page_errors:
                raise AssertionError(f"Browser JavaScript errors: {page_errors}")
            checks["browser_mobile_layout"] = True
            mobile_context.close()
            browser_version = browser.version
            context.close()
            browser.close()
        result = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "project_version": __version__,
            "python": sys.version.split()[0],
            "browser": browser_version,
            "location_id": location,
            "synthetic": True,
            "computed_alerts": len(stored),
            "known_generated_agents_per_episode": 2,
            "episodes": 4,
            "cooldown_s": 0.75,
            "checks": checks,
            "screenshots": screenshots,
            "limitations": "Synthetic scenarios and tabular benchmark only; no real-world safety validation. Browser screenshots do not include credentials or real footage.",
        }
        (directory / "results.json").write_text(json.dumps(result, indent=2))
        Path("outputs/e2e-latest.json").write_text(json.dumps(result, indent=2))
        if args.screenshots is not None:
            (args.screenshots / "provenance.json").write_text(json.dumps(result, indent=2))
        print(f"Dashboard demo: {args.dashboard_url}/?synthetic=true&location={location}")
        print(
            json.dumps(
                {
                    "passed_checks": len(checks),
                    "computed_alerts": len(stored),
                    "screenshots": len(screenshots),
                    "report": str(directory / "results.json"),
                },
                indent=2,
            )
        )
    finally:
        metadata.close()
        engine.dispose()


if __name__ == "__main__":
    main()
