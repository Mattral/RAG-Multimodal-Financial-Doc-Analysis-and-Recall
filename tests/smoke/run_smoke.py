"""End-to-end smoke test against a running API (stdlib only).

1. Ingest tests/fixtures/smoke_filing.pdf and require that the hi_res parser handled
   it (not the pypdf fallback).
2. POST /api/v1/retrieve: the top chunk must be the page-2 operating-cash-flow text,
   cited to smoke_filing.pdf, page 2.
3. POST /api/v1/query: a non-empty answer (from the stub LLM) with sources.

Usage: python tests/smoke/run_smoke.py BASE_URL API_KEY
Exits non-zero with a message on the first failed check.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "smoke_filing.pdf"
EXPECTED_TEXT = "Operating cash flow was 3,308 million dollars"
QUESTION = "What was the operating cash flow in the third quarter of 2023?"


def fail(message: str, detail: object = None) -> None:
    print(f"SMOKE FAIL: {message}")
    if detail is not None:
        print(json.dumps(detail, indent=2, default=str)[:4000])
    sys.exit(1)


def request(base: str, key: str, method: str, path: str, body=None, headers=None, timeout=600):
    req = urllib.request.Request(base + path, data=body, method=method)
    req.add_header("X-API-Key", key)
    for name, value in (headers or {}).items():
        req.add_header(name, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"null")


def post_json(base: str, key: str, path: str, payload: dict):
    return request(
        base, key, "POST", path, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )


def wait_ready(base: str, key: str, seconds: int = 300) -> None:
    deadline = time.time() + seconds
    last = None
    while time.time() < deadline:
        try:
            status, body = request(base, key, "GET", "/readyz", timeout=10)
            if status == 200:
                return
            last = (status, body)
        except OSError as exc:
            last = str(exc)
        time.sleep(3)
    fail("API did not become ready", last)


def ingest(base: str, key: str) -> dict:
    boundary = uuid.uuid4().hex
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="process_vision"\r\n\r\nfalse\r\n',
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        f'filename="{FIXTURE.name}"\r\nContent-Type: application/pdf\r\n\r\n',
    ]
    body = "".join(parts).encode() + FIXTURE.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    status, result = request(
        base,
        key,
        "POST",
        "/api/v1/ingest",
        body,
        {"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    if status != 200 or result.get("status") != "success":
        fail(f"ingest returned {status}", result)
    return result


def main() -> None:
    base, key = sys.argv[1].rstrip("/"), sys.argv[2]
    wait_ready(base, key)

    result = ingest(base, key)
    parser = (result.get("parsers") or {}).get(FIXTURE.name)
    if parser != "unstructured/hi_res":
        fail(f"expected the hi_res parser for {FIXTURE.name}, got {parser!r}", result)
    if not result.get("num_chunks"):
        fail("ingest produced no chunks", result)
    print(f"ingest ok: parser={parser} chunks={result['num_chunks']}")

    status, retrieved = post_json(base, key, "/api/v1/retrieve", {"query": QUESTION, "top_k": 3})
    if status != 200 or not retrieved.get("chunks"):
        fail(f"retrieve returned {status} with no chunks", retrieved)
    top = retrieved["chunks"][0]
    if EXPECTED_TEXT not in top["text"]:
        fail("top retrieved chunk is not the operating cash flow passage", retrieved)
    if top["document"] != FIXTURE.name or top["page"] != 2:
        fail(f"wrong citation: document={top['document']!r} page={top['page']!r}", top)
    print(f"retrieve ok: {top['document']} p.{top['page']} score={top['score']:.3f}")

    status, answer = post_json(base, key, "/api/v1/query", {"query": QUESTION, "top_k": 3})
    if status != 200 or not (answer.get("answer") or "").strip():
        fail(f"query returned {status} without an answer", answer)
    sources = answer.get("sources") or []
    if not any(s.get("document") == FIXTURE.name and s.get("page") == 2 for s in sources):
        fail("query sources do not cite smoke_filing.pdf page 2", answer)
    print(f"query ok: {answer['answer'][:80]!r} ({len(sources)} sources)")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
