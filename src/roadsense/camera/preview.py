# SPDX-License-Identifier: AGPL-3.0-only
"""Opt-in loopback browser preview; keep only the latest annotated JPEG in memory."""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

PAGE = b"""<!doctype html><html><head><meta charset="utf-8">
<title>RoadSense-U live preview</title>
<style>body{background:#0c1423;color:#ecf2fa;font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 24px;line-height:1.6}
h1{letter-spacing:-.035em}img{display:block;max-width:100%;max-height:75vh;margin:24px auto;border:1px solid #29364c;border-radius:12px}p{color:#adbed5}</style></head><body>
<h1>RoadSense-U live preview</h1>
<p>Research estimates, not confirmed collisions. Live images stay on this computer.
Preview does not record video. Stop processing with Ctrl+C in the terminal.</p>
<img src="/stream.mjpg" alt="Waiting for processed frames...">
<p>Shows the latest processed frame; processing speed depends on your hardware.
The preview ends when processing stops.</p></body></html>"""


class LivePreview:
    """Serve annotated frames locally without recording or uploading them.

    This is a local development viewer, limited to four stream viewers. It is deliberately
    bound to loopback and must not be exposed via a public reverse proxy.
    """

    def __init__(self, port: int = 8765):
        if not 0 <= port <= 65535:
            raise ValueError("Preview port must be between 0 and 65535")
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._sequence = 0
        self._closed = False
        self._viewers = threading.BoundedSemaphore(4)
        preview = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                expected = {f"127.0.0.1:{preview.port}", f"localhost:{preview.port}"}
                if self.headers.get("Host") not in expected:
                    self.send_error(403, "Loopback host required")
                    return
                path = self.path.split("?", 1)[0]
                if path == "/":
                    self.respond(200, "text/html; charset=utf-8", PAGE)
                elif path == "/frame.jpg":
                    with preview._condition:
                        jpeg = preview._jpeg
                    if jpeg is None:
                        self.respond(503, "text/plain", b"Waiting for the first processed frame")
                    else:
                        self.respond(200, "image/jpeg", jpeg)
                elif path == "/stream.mjpg":
                    self.stream()
                else:
                    self.send_error(404)

            def respond(self, status: int, content_type: str, body: bytes):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cross-Origin-Resource-Policy", "same-origin")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    pass

            def stream(self):
                if not preview._viewers.acquire(blocking=False):
                    self.respond(503, "text/plain", b"Maximum local viewers reached")
                    return
                try:
                    self.connection.settimeout(3)
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cross-Origin-Resource-Policy", "same-origin")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    seen = -1
                    while True:
                        with preview._condition:
                            preview._condition.wait_for(
                                lambda: preview._closed or preview._sequence != seen, timeout=1
                            )
                            if preview._closed:
                                break
                            jpeg, sequence = preview._jpeg, preview._sequence
                        if jpeg is None or sequence == seen:
                            seen = sequence
                            continue
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                            + str(len(jpeg)).encode()
                            + b"\r\n\r\n"
                            + jpeg
                            + b"\r\n"
                        )
                        self.wfile.flush()
                        seen = sequence
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    pass
                finally:
                    preview._viewers.release()

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_port
        self.url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def update(self, image: np.ndarray) -> None:
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise RuntimeError("Cannot encode live preview frame")
        with self._condition:
            self._jpeg = encoded.tobytes()
            self._sequence += 1
            self._condition.notify_all()

    def __exit__(self, *args):
        with self._condition:
            self._closed = True
            self._jpeg = None
            self._condition.notify_all()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=3)
