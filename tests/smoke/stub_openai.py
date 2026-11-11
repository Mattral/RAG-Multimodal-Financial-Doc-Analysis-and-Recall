"""Minimal OpenAI-compatible chat server for the image smoke test (stdlib only).

Answers every POST /v1/chat/completions with the same fixed completion, so the
smoke test exercises the real OpenAI client path without a key or a model.

Usage: python tests/smoke/stub_openai.py [PORT]   (default 9000)
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ANSWER = (
    "Operating cash flow was 3,308 million dollars in the third quarter of 2023 "
    "[Source: smoke_filing.pdf, Page 2]."
)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # health probe for the workflow
        self._send(200, {"status": "ok"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if not self.path.endswith("/chat/completions"):
            self._send(404, {"error": f"unexpected path {self.path}"})
            return
        self._send(
            200,
            {
                "id": "stub-1",
                "object": "chat.completion",
                "model": body.get("model", "stub"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": ANSWER},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            },
        )

    def _send(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("stub_openai: " + fmt % args + "\n")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
