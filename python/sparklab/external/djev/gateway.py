"""Thin lifecycle/error handling around the vendored upstream decision API."""
from __future__ import annotations

import argparse
import json
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
import urllib.error
import urllib.request

from . import upstream


class Server(ThreadingHTTPServer):
    request_queue_size = 128
    daemon_threads = True


class Handler(upstream.Handler):
    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def do_GET(self):
        if self.path == "/v1/models":
            return self._json(200, {"object": "list", "data": [
                {"id": upstream.ARGS.model, "object": "model", "owned_by": "sparklab"}
            ]})
        if self.path not in ("/health", "/metrics"):
            return self._json(404, {"error": {"message": "unknown route"}})
        try:
            with urllib.request.urlopen(upstream.ARGS.upstream + self.path, timeout=3) as response:
                raw = response.read()
            if self.path == "/health":
                return self._json(200, {"status": "ok", "model": upstream.ARGS.model, "backend": "djev"})
            self.send_response(200)
            self.send_header("content-type", "text/plain; version=0.0.4")
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except (OSError, urllib.error.URLError):
            return self._json(503, {"status": "unavailable", "error": "vLLM is unavailable"})

    def _read_request(self):
        req, images = super()._read_request()
        if not isinstance(req, dict):
            raise ValueError("body must be a JSON object")
        if "seed" in req and (type(req["seed"]) is not int or not 0 <= req["seed"] < 2**63):
            raise ValueError("seed must be an integer in [0, 2**63)")
        if req.get("model", upstream.ARGS.model) != upstream.ARGS.model:
            raise ValueError("model must match the served model")
        return req, images

    def do_POST(self):
        if self.path not in ("/v1/systemone", "/v1/chat/completions"):
            return self._json(404, {"error": {"message": "unknown route"}})
        try:
            length = int(self.headers.get("content-length", "0"))
            if not 0 < length <= 16 * 1024 * 1024:
                return self._json(413, {"error": {"message": "body must be 1 byte to 16 MiB"}})
            super().do_POST()
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            self._json(422, {"error": {"message": str(exc), "type": "validation_error"}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--canvas", type=int, required=True)
    args = parser.parse_args()
    upstream.ARGS = SimpleNamespace(upstream=args.upstream, model=args.model)
    upstream.CANVAS_LEN = args.canvas
    upstream.CANVAS_STEP = 16
    upstream.init_tokenizer(upstream.AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True))
    Server(("0.0.0.0", 8011), Handler).serve_forever()


if __name__ == "__main__":
    main()
