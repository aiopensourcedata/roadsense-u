# SPDX-License-Identifier: AGPL-3.0-only
"""Prepare local, score-blind review windows and optional sampled contact sheets.

The source video is read in place; no clips are copied or uploaded. Frame export
requires an explicit flag. No labels are inferred from model predictions.
"""

import argparse
import csv
import hashlib
import html
import importlib.util
import json
import os
from bisect import bisect_left
from pathlib import Path

import cv2
import numpy as np

from roadsense.evaluation.coverage import ObservationCoverage
from roadsense.evaluation.review import sample_review_windows

FIELDS = [
    "window_id",
    "prediction_time_s",
    "future_observation_complete",
    "object_distinctness",
    "visibility",
    "observed_contact",
    "possible_evasion",
    "screening_assessment",
    "provisional_conflict_label",
    "adjudicated_label",
    "reviewer_id",
    "review_method",
    "notes",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", type=Path)
    p.add_argument("--timestamps", type=Path, required=True, help="ffprobe frame JSON from the same source")
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--write-review-frames", action="store_true")
    p.add_argument("--synthetic", action="store_true", help="Explicitly label generated test media")
    args = p.parse_args()
    if args.output.exists():
        p.error("Output exists; choose a fresh directory")
    if not args.video.is_file():
        p.error("Video does not exist")
    if args.write_review_frames and importlib.util.find_spec("matplotlib") is None:
        p.error("Frame export requires pip install -e '.[review]'")
    times = [
        float(r["best_effort_timestamp_time"]) for r in json.loads(args.timestamps.read_text())["frames"]
    ]
    coverage = ObservationCoverage()
    for t in times:
        coverage.observe(t)
    comparison = json.loads(args.comparison.read_text())
    if "synthetic" in comparison and bool(comparison["synthetic"]) != args.synthetic:
        p.error("Video and comparison synthetic provenance must match")
    alerts = comparison["alerts"]["guarded"]
    withheld = [
        x
        for x in comparison["alerts"]["baseline"]
        if not any(
            abs(x["timestamp_s"] - y["timestamp_s"]) < 1e-8
            and sorted(x["track_ids"]) == sorted(y["track_ids"])
            for y in alerts
        )
    ]
    known = alerts + withheld
    excluded = [(max(0, x["timestamp_s"] - 2), x["timestamp_s"] + 3.1) for x in known]
    extra = sample_review_windows(times, excluded, args.count, args.seed)
    windows = []
    model_context = {}
    for x in known:
        t = x["timestamp_s"]
        windows.append(
            {
                "start_s": max(0, t - 2),
                "prediction_time_s": t,
                "end_s": t + 3.1,
                "prediction_frame": bisect_left(times, t),
                "future_observation_complete": coverage.assess(t)["future_observation_complete"],
                "_model": x,
            }
        )
    windows.extend(extra)
    windows.sort(key=lambda x: x["prediction_time_s"])
    args.output.mkdir(parents=True)
    for i, w in enumerate(windows, 1):
        w["window_id"] = f"W{i:02d}"
        context = w.pop("_model", None)
        if context:
            model_context[w["window_id"]] = context
        # Use actual observed frames only; never interpolate through a source gap.
        indices = sorted(
            {
                min(bisect_left(times, t), len(times) - 1)
                for t in np.arange(w["start_s"], w["end_s"] + 0.001, 0.5)
            }
        )
        w["sample_frames"] = [
            {"frame": k, "timestamp_s": times[k]} for k in indices if times[k] <= w["end_s"]
        ]
    digest = hashlib.sha256(args.video.read_bytes()).hexdigest()
    manifest = {
        "source_file": args.video.name,
        "synthetic": args.synthetic,
        "source_sha256": digest,
        "seed": args.seed,
        "max_gap_s": 0.5,
        "additional_windows": args.count,
        "windows": windows,
        "selection": "Additional windows chosen without scores/labels; known alert/regression windows included separately.",
        "scope": "Development screening sample; dependent same-camera observations, not a held-out-location test.",
        "labels_validated": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output / "manifest.sha256").write_text(
        hashlib.sha256((args.output / "manifest.json").read_bytes()).hexdigest() + "\n"
    )
    (args.output / "model_context.json").write_text(json.dumps(model_context, indent=2))
    with (args.output / "independent-review.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for w in windows:
            writer.writerow({k: w[k] for k in FIELDS if k in w})
    if args.write_review_frames:
        os.environ.setdefault("MPLCONFIGDIR", str((args.output / "matplotlib-cache").resolve()))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        targets = {sample["frame"] for w in windows for sample in w["sample_frames"]}
        frames = {}
        cap = cv2.VideoCapture(str(args.video))
        if not cap.isOpened():
            raise RuntimeError("Cannot open review video")
        try:
            for i in range(max(targets) + 1):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Source decoding failed at frame {i}")
                if i in targets:
                    if abs(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 - times[i]) > 0.001:
                        raise ValueError("Timestamp cache does not match the decoded source")
                    scale = min(1.0, 640 / frame.shape[1])
                    thumbnail = cv2.resize(
                        frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale))
                    )
                    frames[i] = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()
        for w in windows:
            samples = w["sample_frames"]
            fig, axes = plt.subplots(3, 4, figsize=(20, 9.3), dpi=120)
            for ax in axes.flat:
                ax.axis("off")
            for ax, s in zip(axes.flat, samples):
                ax.imshow(frames[s["frame"]])
                ax.set_title(f"{s['timestamp_s']:.3f}s | frame {s['frame']}", fontsize=10)
            fig.suptitle(
                f"{'SYNTHETIC | ' if args.synthetic else ''}{w['window_id']} | t={w['prediction_time_s']:.3f}s | sampled frames, not continuous playback",
                fontsize=14,
            )
            fig.tight_layout()
            fig.savefig(args.output / f"{w['window_id']}.jpg")
            plt.close(fig)
    source = html.escape(args.video.resolve().as_uri(), quote=True)
    items = "".join(
        f'<option value="{i}">{w["window_id"]} — {w["prediction_time_s"]:.3f}s</option>'
        for i, w in enumerate(windows)
    )
    # The page intentionally contains no scores, model IDs or assistant annotations.
    data = json.dumps(
        [
            {
                k: w[k]
                for k in ("window_id", "start_s", "prediction_time_s", "end_s", "future_observation_complete")
            }
            for w in windows
        ]
    )
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><title>RoadSense-U local review</title>
<style>body{max-width:1100px;margin:30px auto;font:16px system-ui;background:#101820;color:#edf4f5;padding:16px}video,img{width:100%}button,select,input{padding:10px;margin:5px}textarea{width:95%;height:70px}label{display:inline-block}small{color:#b7c8cf}</style>
<h1>Local interaction review</h1><p>Review the source video before assigning labels. Missing footage is unevaluable. Scores and first-pass annotations are hidden. This page uploads nothing.</p>
<select id="windows">OPTIONS</select><button id="replay">Play window</button><button id="export">Export my CSV</button><p id="status"></p>
<video id="video" controls preload="metadata" src="SOURCE"></video><p><small>Playback can stall/jump at encoded gaps. The sampled sheet is context only. If local playback is blocked, open the original video in a local player and use the listed source times.</small></p>
<label>My reviewer ID <input id="reviewer" placeholder="Choose a stable ID"></label>
<label>Conflict label <select id="label"><option value="">Unresolved</option><option value="0">Provisional no conflict</option><option value="1">Provisional conflict candidate</option></select></label>
<label>Visibility <select id="visibility"><option value="">Unreviewed</option><option>adequate</option><option>partial</option><option>unassessable</option></select></label>
<textarea id="notes" placeholder="Describe distinct users, maneuvers, occlusion, contact, timing and uncertainty. Do not infer physical speed from pixels."></textarea><button id="save">Save this window locally</button>
<p id="saved"></p><details><summary>Sampled frames (2 Hz)</summary><img id="sheet" alt="Sampled source frames"></details>
<script>const windows=DATA;const selector=document.getElementById('windows'),video=document.getElementById('video');const key='roadsense-review-SHA';let answers=JSON.parse(localStorage.getItem(key)||'{}');
function current(){return windows[Number(selector.value)]}function load(){let w=current(),a=answers[w.window_id]||{};document.getElementById('status').textContent=`${w.window_id}: ${w.start_s.toFixed(3)}–${w.end_s.toFixed(3)}s; prediction t=${w.prediction_time_s.toFixed(3)}s; future footage ${w.future_observation_complete?'meets timestamp continuity policy':'CENSORED — do not label negative'}`;for(let f of ['label','visibility','notes'])document.getElementById(f).value=a[f]||'';document.getElementById('sheet').src=w.window_id+'.jpg';document.getElementById('saved').textContent='';}
selector.onchange=load;document.getElementById('replay').onclick=()=>{video.currentTime=current().start_s;video.play().catch(()=>{})};video.ontimeupdate=()=>{if(video.currentTime>=current().end_s)video.pause()};
document.getElementById('save').onclick=()=>{let w=current(),label=document.getElementById('label').value;if(!w.future_observation_complete&&label!==''){alert('This horizon is censored. Leave its conflict label unresolved.');return}answers[w.window_id]={label,visibility:document.getElementById('visibility').value,notes:document.getElementById('notes').value,reviewer_id:document.getElementById('reviewer').value};localStorage.setItem(key,JSON.stringify(answers));document.getElementById('saved').textContent='Saved in this browser only.'};
function quote(x){let s=String(x??'');if(/^[=+@-]/.test(s))s="'"+s;return '"'+s.replaceAll('"','""')+'"'}document.getElementById('export').onclick=()=>{let rows=[['window_id','prediction_time_s','future_observation_complete','provisional_conflict_label','visibility','notes','reviewer_id','adjudicated_label']];for(let w of windows){let a=answers[w.window_id]||{};rows.push([w.window_id,w.prediction_time_s,w.future_observation_complete,a.label||'',a.visibility||'',a.notes||'',a.reviewer_id||'',''])}let link=document.createElement('a');link.href=URL.createObjectURL(new Blob([rows.map(r=>r.map(quote).join(',')).join('\\n')],{type:'text/csv'}));link.download='my-review.csv';link.click();URL.revokeObjectURL(link.href)};load();</script></html>"""
    page = (
        page.replace("OPTIONS", items).replace("SOURCE", source).replace("DATA", data).replace("SHA", digest)
    )
    if args.synthetic:
        page = page.replace(
            "<h1>Local interaction review</h1>",
            "<h1>Local interaction review</h1><p>SYNTHETIC fixture — not real traffic evidence.</p>",
        )
    (args.output / "review.html").write_text(page)
    print(
        json.dumps(
            {
                "windows": len(windows),
                "additional_windows": len(extra),
                "future_censored": sum(not w["future_observation_complete"] for w in windows),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
