#!/usr/bin/env python3
"""
Jarvis web gateway.

Serves the frontend on :3000 and proxies OpenAI-compatible SSE
requests to the existing Jarvis proxy on localhost:9000.

No third-party Python packages are required.
"""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HOST = "0.0.0.0"
PORT = 3000
PROXY_URL = "http://127.0.0.1:9000/v1/chat/completions"
STATIC_DIR = Path(__file__).resolve().parent / "static"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print(f"[web] {self.address_string()} - {fmt % args}")

    def _send(self, status, content_type, body, extra_headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _json_error(self, status, message):
        body = json.dumps({"error": message}).encode()
        self._send(status, "application/json; charset=utf-8", body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]

        if path in ("/", "/index.html"):
            file_path = STATIC_DIR / "index.html"
        elif path == "/health":
            self._send(200, "application/json; charset=utf-8", b'{"status":"ok"}')
            return
        elif path.startswith("/static/"):
            relative = path.removeprefix("/static/")
            file_path = STATIC_DIR / relative
        else:
            self._json_error(404, "Not found")
            return

        try:
            file_path = file_path.resolve()
            if not file_path.is_relative_to(STATIC_DIR.resolve()):
                raise FileNotFoundError
            data = file_path.read_bytes()
        except (FileNotFoundError, OSError):
            self._json_error(404, "Not found")
            return

        content_type = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".svg": "image/svg+xml",
        }.get(file_path.suffix, "application/octet-stream")

        self._send(200, content_type, data)

    def do_POST(self):
        if self.path != "/api/chat":
            self._json_error(404, "Not found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                raise ValueError("Empty request body")

            payload = json.loads(self.rfile.read(length))

            pipeline = payload.get("pipeline", "normal")
            if pipeline not in ("normal", "fast"):
                pipeline = "normal"

            upstream_payload = {
                "messages": payload.get("messages", []),
                "stream": True,
                "pipeline": pipeline,
            }

            # Preserve common OpenAI-compatible generation parameters.
            for key in (
                "model",
                "temperature",
                "top_p",
                "min_p",
                "max_tokens",
                "max_completion_tokens",
                "stop",
            ):
                if key in payload:
                    upstream_payload[key] = payload[key]

            body = json.dumps(upstream_payload).encode("utf-8")

            request = Request(
                PROXY_URL,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                    "Cache-Control": "no-cache",
                },
                method="POST",
            )

            upstream = urlopen(request, timeout=3600)

            self.send_response(upstream.status)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            self.wfile.flush()

            # IMPORTANT:
            # HTTPResponse.read(4096) can wait for the buffer to fill and makes
            # token streaming feel very laggy. readline() forwards each SSE
            # event as soon as it arrives. We also stop at [DONE] instead of
            # waiting for the upstream HTTP connection to close.
            while True:
                line = upstream.readline()
                if not line:
                    break

                self.wfile.write(line)
                self.wfile.flush()

                if line.strip() == b"data: [DONE]":
                    break

            upstream.close()

        except HTTPError as exc:
            try:
                error_body = exc.read()
            except Exception:
                error_body = json.dumps({"error": str(exc)}).encode()
            self._send(
                exc.code,
                "application/json; charset=utf-8",
                error_body,
            )
        except (URLError, TimeoutError, ConnectionError, BrokenPipeError) as exc:
            # BrokenPipe is expected if the browser cancels a generation.
            if isinstance(exc, BrokenPipeError):
                return
            self._json_error(502, str(exc))
        except (ValueError, json.JSONDecodeError) as exc:
            self._json_error(400, str(exc))
        except Exception as exc:
            self._json_error(500, str(exc))


if __name__ == "__main__":
    print(f"Jarvis Web UI : http://0.0.0.0:{PORT}")
    print(f"Jarvis Proxy   : {PROXY_URL}")
    print("Press Ctrl+C to stop.")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
