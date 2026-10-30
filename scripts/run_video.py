# SPDX-License-Identifier: AGPL-3.0-only
"""Run prerecorded video, device index or phone HTTP/RTSP stream."""

import argparse
import json
import logging
from contextlib import nullcontext
from datetime import datetime

from roadsense.camera.preview import LivePreview
from roadsense.cloud.client import EventClient
from roadsense.config import PipelineConfig, Settings
from roadsense.pipeline import run


def main(default_source=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", default=default_source)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="outputs/run")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--save-video", action="store_true", help="Opt in to retaining annotated footage")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument(
        "--preview", action="store_true", help="Watch annotated video locally in your browser; no recording"
    )
    parser.add_argument("--preview-port", type=int, default=8765)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--inference-size", type=int, help="YOLO size, multiple of 32 (default 640)")
    parser.add_argument("--tiled", action="store_true", help="Experimental full-frame plus overlapping crops")
    parser.add_argument(
        "--start-frame", type=int, help="Sequentially skip source frames; preserve original media timestamps"
    )
    parser.add_argument(
        "--video-start", type=datetime.fromisoformat, help="Capture start ISO8601 with timezone"
    )
    args = parser.parse_args()
    if args.source is None:
        parser.error("Provide a video path, device index or stream URL")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = PipelineConfig.load(args.config).model_dump()
    cfg["save_video"] = args.save_video or cfg["save_video"]
    if args.synthetic:
        cfg.update(synthetic=True, detector="synthetic")
    if args.max_frames is not None:
        cfg["max_frames"] = args.max_frames
    if args.inference_size is not None:
        cfg["inference_size"] = args.inference_size
    if args.tiled:
        cfg["tiling"]["enabled"] = True
    if args.start_frame is not None:
        cfg["start_frame"] = args.start_frame
    client = None
    if args.submit:
        settings = Settings()
        client = EventClient(settings.api_url, settings.api_key, ca_file=settings.api_ca_file)
    try:
        with LivePreview(args.preview_port) if args.preview else nullcontext() as preview:
            if preview is not None:
                logging.info("Live preview: %s (open in your browser; Ctrl+C to stop)", preview.url)
            print(
                json.dumps(
                    run(
                        args.source,
                        args.output,
                        PipelineConfig(**cfg),
                        client,
                        args.video_start,
                        preview=preview,
                        realtime=args.preview,
                    ),
                    indent=2,
                )
            )
    except KeyboardInterrupt:
        logging.info("Video processing stopped.")
    finally:
        if client:
            client.close()


if __name__ == "__main__":
    main()
