# SPDX-License-Identifier: AGPL-3.0-only
"""Explore actual API events and local run artifacts, with explicit research provenance."""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import altair as alt
import httpx
import pandas as pd
import streamlit as st

from roadsense.dashboard.artifacts import processed_runs, projection_rows
from roadsense.dashboard.evaluation import evaluation_rows, is_scene_pilot

st.set_page_config(page_title="RoadSense-U · Research workspace", page_icon="◈", layout="wide")
st.markdown(
    """<style>
.block-container {padding-top:2.2rem; padding-bottom:2rem; max-width:1500px}
h1 {letter-spacing:-.04em} [data-testid="stMetric"] {border:1px solid #29364c;
 border-radius:12px; padding:16px; background:rgba(127,149,178,.05)}
[data-testid="stMetricValue"] {font-size:2rem}
</style>""",
    unsafe_allow_html=True,
)

url = os.getenv("ROADSENSE_API_URL", "http://127.0.0.1:8000")
key = os.getenv("ROADSENSE_API_KEY", "")
st.sidebar.title("◈ RoadSense-U")
st.sidebar.caption("TRAJECTORIES → EVIDENCE → EVENTS")
st.sidebar.divider()
st.sidebar.subheader("Data selection")
synthetic = st.sidebar.checkbox(
    "Include synthetic fixtures (only)", value=st.query_params.get("synthetic") == "true"
)
camera = st.sidebar.text_input("Camera filter (optional)", value=st.query_params.get("camera", ""))
location = st.sidebar.text_input("Location filter (optional)", value=st.query_params.get("location", ""))
days = st.sidebar.slider("Days", 1, 30, 7)
st.sidebar.caption(
    "Synthetic and real observations are queried separately. Filters apply to stored event metadata."
)
st.sidebar.divider()
st.sidebar.markdown("**Research alpha · v0.1.0**")
st.sidebar.caption("Local processing. Compact metadata. Explicit uncertainty.")

st.caption("ROADSENSE-U / RESEARCH WORKSPACE")
st.title("Traffic interaction intelligence")
st.write("Follow the trajectories. Inspect the evidence. Evaluate the estimates.")
st.warning("Research estimates only. Predicted conflicts are not confirmed near misses or collisions.")
if synthetic:
    st.info("SYNTHETIC DATA · Generated trajectories and measurements for software validation only.")

params = {"synthetic": synthetic, "limit": 1000}
if camera:
    params["camera_id"] = camera
if location:
    params["location_id"] = location
now = datetime.now(timezone.utc)
params.update(start=(now - timedelta(days=days)).isoformat(), end=now.isoformat())
events, stats = [], None
try:
    with httpx.Client(trust_env=False, base_url=url, headers={"X-API-Key": key}, timeout=5) as client:
        health = client.get("/health")
        health.raise_for_status()
        response = client.get("/events", params=params)
        response.raise_for_status()
        events = response.json()
        response = client.get("/stats", params={k: v for k, v in params.items() if k != "limit"})
        response.raise_for_status()
        stats = response.json()
    st.success("API and database connected")
except (httpx.HTTPError, ValueError) as exc:
    st.error(f"API connection failed: {type(exc).__name__}. Check service status and API key.")

frame = pd.json_normalize(events) if events else pd.DataFrame()
overview, evidence, analytics, artifacts, evaluation_tab = st.tabs(
    ["Overview", "Event evidence", "Location analytics", "Processed runs", "Evaluation"]
)

with overview:
    st.subheader("Interaction overview")
    if stats is not None:
        columns = st.columns(4)
        columns[0].metric("Stored alerts", stats["event_count"])
        columns[1].metric("Cameras in this window", len(stats["by_camera"]))
        columns[2].metric("Mean reported score", f"{frame.risk_score.mean():.3f}" if events else "—")
        columns[3].metric("Mean track quality", f"{frame['features.quality'].mean():.2f}" if events else "—")
        st.caption(
            "Alert emissions depend on thresholds and cooldowns. They are not counts of independent labeled conflicts. Track quality is a heuristic reliability indicator."
        )
    if len(events) == 1000:
        st.info("Showing the first 1,000 events. Narrow the filters for complete detail.")
    if events:
        st.markdown("**Reported scores over time**")
        st.caption(
            "Each point is an alert emission. Gaps are unobserved; rule scores are heuristic, not crash probabilities."
        )
        timeline = frame[["timestamp", "camera_id", "risk_score", "score_kind"]].copy()
        chart = (
            alt.Chart(timeline)
            .mark_circle(size=85)
            .encode(
                x=alt.X(
                    "timestamp:T",
                    title="Event timestamp (UTC)",
                    scale=alt.Scale(type="utc"),
                    axis=alt.Axis(format="%H:%M"),
                ),
                y=alt.Y(
                    "risk_score:Q",
                    title="Reported score",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                color=alt.Color(
                    "camera_id:N",
                    title="Camera",
                    scale=alt.Scale(range=["#50dfbd", "#7cb6ff", "#ffca7a", "#c4a0ff"]),
                ),
                tooltip=[
                    alt.Tooltip("timestamp:N", title="Timestamp (UTC)"),
                    "camera_id:N",
                    "risk_score:Q",
                    "score_kind:N",
                ],
            )
            .properties(height=250)
        )
        st.altair_chart(chart, width="stretch")
        event_rows = frame[
            ["timestamp", "camera_id", "event_type", "risk_score", "score_kind", "features.units"]
        ].copy()
        event_rows["event_type"] = event_rows["event_type"].str.replace("_", " ").str.capitalize()
        event_rows = event_rows.rename(
            columns={
                "timestamp": "Timestamp (UTC)",
                "camera_id": "Camera",
                "event_type": "Estimate type",
                "risk_score": "Score",
                "score_kind": "Score semantics",
                "features.units": "Units",
            }
        )
        st.dataframe(event_rows, hide_index=True, width="stretch")
    else:
        st.info(
            "No matching events stored. Process a permitted video or enable the synthetic fixture filter."
        )

with evidence:
    st.subheader("Evidence behind an alert")
    if events:
        index = st.selectbox(
            "Event supporting measurements",
            range(len(events)),
            index=max(range(len(events)), key=lambda i: events[i]["risk_score"]),
            format_func=lambda i: (
                f"{events[i]['camera_id']} · video t={events[i]['video_time_s']:.2f}s · score={events[i]['risk_score']:.3f}"
            ),
        )
        event = events[index]
        st.caption(
            f"Timestamp origin: {event.get('timestamp_origin', 'legacy_unknown')} · server ingestion: {event.get('ingested_at') or 'unavailable'}"
        )
        if event.get("timestamp_origin") != "provided_capture_time":
            st.info(
                "Original capture time is unknown. The timeline uses ingestion-origin media time or a legacy timestamp."
            )
        features = event["features"]
        unit = features["units"]
        columns = st.columns(4)
        columns[0].metric("Reported score", f"{event['risk_score']:.3f}", help=event["score_kind"])
        columns[1].metric("Predicted CPA distance", f"{features['cpa_distance']:.2f} {unit}")
        columns[2].metric("Time to CPA", f"{features['time_to_cpa_s']:.2f} s")
        columns[3].metric("Track quality", f"{features['quality']:.2f}")
        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Measured explanation**")
            for reason in event["explanations"]:
                st.write(f"• {reason}")
            st.caption(
                f"{' + '.join(event['classes'])} · model {event['model_version']} · {event['score_kind']}"
            )
            st.markdown("**Interpretation**")
            st.write(
                "CPA projects constant-velocity motion over the configured horizon. Pixel measurements are not physical speed or distance."
            )
            if features["ttc_s"] is None:
                st.caption(
                    "TTC is unavailable: validated metric geometry and physical footprints are required. Predicted arrival differences are not observed PET."
                )
        with right:
            st.markdown("**Supporting measurements**")
            values = {
                "Separation": f"{features['distance']:.2f} {unit}",
                "Closing speed": f"{features['closing_speed']:.2f} {unit}/s",
                "Relative speed": f"{features['relative_speed']:.2f} {unit}/s",
                "Measurement uncertainty indicator": f"{features['measurement_uncertainty']:.3f} {unit}",
                "Source video time": f"{event['video_time_s']:.2f} s",
            }
            st.dataframe(
                pd.DataFrame(values.items(), columns=["Measurement", "Value"]),
                hide_index=True,
                width="stretch",
            )
            st.caption("Uncertainty is a residual-based indicator, not a statistical confidence interval.")
        with st.expander("Event metadata and complete feature summary"):
            st.json(event)
    else:
        st.info("Select a window with stored events to inspect supporting measurements.")

with analytics:
    st.subheader("Descriptive location analytics")
    if stats is not None and stats["event_count"]:
        left, right = st.columns(2)
        with left:
            st.markdown("**Alerts by camera**")
            counts = pd.DataFrame(stats["by_camera"].items(), columns=["camera_id", "alerts"])
            st.altair_chart(
                alt.Chart(counts)
                .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
                .encode(
                    x=alt.X("camera_id:N", title="Camera", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("alerts:Q", title="Stored alert emissions"),
                    color=alt.Color("camera_id:N", legend=None),
                )
                .properties(height=230),
                width="stretch",
            )
        with right:
            st.markdown("**Observed interaction patterns**")
            patterns = pd.DataFrame(stats["interaction_patterns"].items(), columns=["Class pair", "Alerts"])
            st.dataframe(patterns, hide_index=True, width="stretch")
            st.caption(
                "Class pairs describe reported alerts. These observations do not establish crash causation or model accuracy."
            )
        st.markdown("**Exposure-normalized alert rates**")
        rates = pd.DataFrame([{"Camera": k, **v} for k, v in stats["exposure_normalized"].items()])
        rates = rates.rename(
            columns={
                "covered_events": "Alerts in exposure windows",
                "road_users": "Observed road users",
                "events_per_1000_road_users": "Alerts / 1,000 road users",
            }
        )
        st.dataframe(rates, hide_index=True, width="stretch")
        st.caption(
            "Only matching exposure windows contribute. Missing or zero traffic volume leaves rates undefined. Unique track IDs are not validated traffic volume."
        )
        with st.expander("Complete API aggregate response"):
            st.json(stats)
    else:
        st.info("No matching aggregate observations.")

root = Path("outputs")
with artifacts:
    st.subheader("Processed video and trajectories")
    runs = processed_runs(root)
    if runs:
        selected = st.selectbox("Processed run", runs, format_func=str)
        report = json.loads((selected / "report.json").read_text())
        st.caption(f"Run status: {report.get('status', 'legacy_unspecified')}")
        if report.get("status") in {"interrupted", "failed"}:
            st.warning(
                "Partial run: future horizons may be censored; this is not completed experimental evidence."
            )
        if report.get("synthetic"):
            st.info("SYNTHETIC RUN · Generated input, not real-world traffic validation.")
        columns = st.columns(4)
        columns[0].metric("Processed frames", report["frames"])
        columns[1].metric("Emitted alerts", report["events"])
        columns[2].metric("Measured throughput", f"{report['throughput_fps']:.1f} FPS")
        latency = report.get("processing_latency_p95_ms")
        columns[3].metric("Processing latency · p95", f"{latency:.1f} ms" if latency is not None else "—")
        video = selected / "annotated.mp4"
        if video.exists():
            st.video(str(video))
            st.caption("Locally retained annotated footage; recording was explicitly enabled.")
        else:
            st.caption(
                "Video recording was disabled for this run. Trajectories and metadata are available below."
            )
        if (selected / "trajectories.csv").exists():
            st.dataframe(
                pd.read_csv(selected / "trajectories.csv", nrows=1000), hide_index=True, width="stretch"
            )
        with st.expander("Run configuration and measured performance"):
            st.json(report)
        geometry_path = selected / "event_geometry.jsonl"
        if geometry_path.exists():
            geometries = [json.loads(line) for line in geometry_path.read_text().splitlines() if line.strip()]
            if geometries:
                st.subheader("Local pair geometry")
                chosen = st.selectbox(
                    "Geometry event",
                    geometries,
                    format_func=lambda g: f"{g['projection']['timestamp_s']:.3f}s · {g['event_id']}",
                )
                projection = chosen["projection"]
                unit = projection["units"]
                st.warning(
                    "Unverified track pair. Projected anchors and their midpoint do not establish "
                    "the colliding participants or a collision location."
                )
                st.caption(
                    f"Constant-velocity projection {projection['offset_s']:.2f}s ahead; coordinates in {unit}."
                )
                points = alt.Chart(pd.DataFrame(projection_rows(projection))).encode(
                    x=alt.X("x:Q", title=f"x ({unit})", scale=alt.Scale(zero=False)),
                    y=alt.Y("y:Q", title=f"y ({unit})", scale=alt.Scale(reverse=unit == "px", zero=False)),
                    color="Track:N",
                    tooltip=["Track", "Stage", "x", "y"],
                )
                st.altair_chart(
                    points.mark_line().encode(order="Order:Q")
                    + points.mark_point(size=100).encode(shape="Stage:N"),
                    width="stretch",
                )
                with st.expander("Complete local projection evidence"):
                    st.json(chosen)
    else:
        st.info("Process a video to create local artifacts.")

with evaluation_tab:
    st.subheader("Offline model evaluation")
    st.caption(
        "Offline artifacts have their own dataset provenance; the sidebar filter applies to stored API events."
    )
    reports = sorted(root.rglob("evaluation.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if reports:
        choice = st.selectbox("Evaluation artifact", reports, format_func=str)
        evaluation = json.loads(choice.read_text())
        if evaluation.get("synthetic"):
            st.warning(
                "SYNTHETIC BENCHMARK · These metrics do not establish real-world traffic performance or model superiority."
            )
        if is_scene_pilot(evaluation):
            st.warning(
                "PUBLISHED SCENE-COLLISION PILOT · A timely scene alert does not verify the colliding actors. "
                "Source-video splits do not establish independent camera locations."
            )
            st.caption(evaluation.get("target", ""))
            st.caption(
                "Accepted-only AP/Brier require coverage context. Missing predictions are abstentions, "
                "and these labels do not identify near misses."
            )
        rows = evaluation_rows(evaluation)
        for name, reason in evaluation.get("unavailable_baselines", {}).items():
            st.info(f"Unavailable baseline: {name}. {reason}")
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        partitions = evaluation.get("partitions", {})
        columns = st.columns(4)
        for column, name in zip(columns, ["train", "calibration", "validation", "test"]):
            part = partitions.get(name, {})
            column.metric(f"{name.capitalize()} observations", part.get("snapshots", part.get("rows", 0)))
        st.caption(
            "Fitting, probability calibration, threshold selection and final testing use disjoint groups. Rules have no Brier score because their scores are not probabilities."
        )
        calibrated = evaluation.get("baselines", {}).get("centralized_logistic", {})
        bins = calibrated.get("reliability_bins", [])
        if bins:
            reliability = (
                alt.Chart(pd.DataFrame(bins))
                .mark_line(point=True, color="#50dfbd")
                .encode(
                    x=alt.X(
                        "mean_score:Q", title="Mean predicted probability", scale=alt.Scale(domain=[0, 1])
                    ),
                    y=alt.Y(
                        "observed_fraction:Q",
                        title="Observed positive fraction",
                        scale=alt.Scale(domain=[0, 1]),
                    ),
                    tooltip=["count:Q", "mean_score:Q", "observed_fraction:Q"],
                )
            )
            reference = (
                alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
                .mark_line(strokeDash=[5, 5], color="#76879f")
                .encode(x="x:Q", y="y:Q")
            )
            st.altair_chart((reference + reliability).properties(height=230), width="stretch")
        with st.expander("Complete evaluation protocol, results and limitations"):
            st.json(evaluation)
    else:
        st.info("Run scripts/train_baseline.py to create a reproducible evaluation artifact.")
