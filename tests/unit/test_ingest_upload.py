"""POST /api/v1/ingest keeps the uploaded file name as the citation source.

Uploads used to be written to NamedTemporaryFile(suffix=".pdf"), so the parser cited
API-ingested documents as e.g. "tmpk3j2x9ab.pdf", and the file was never deleted.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from filing_qa.api.routers.ingest import _safe_filename


def _client_recording_paths(seen):
    from filing_qa.api.app import create_app

    async def ingest(file_paths, tenant_id, process_vision):
        for p in file_paths:
            seen.append((p, Path(p).read_bytes(), os.path.exists(p)))
        return {"status": "success", "num_files": len(file_paths)}

    pipeline = MagicMock()
    pipeline.ingest = ingest
    app = create_app()
    app.state.pipeline = pipeline
    return TestClient(app)


def test_upload_is_parsed_under_its_own_name_and_removed_after():
    seen = []
    with _client_recording_paths(seen) as client:
        resp = client.post(
            "/api/v1/ingest",
            files={"file": ("tesla_10q_q3_2023.pdf", b"%PDF-1.4 test", "application/pdf")},
        )
    assert resp.status_code == 200
    path, content, existed = seen[0]
    assert Path(path).name == "tesla_10q_q3_2023.pdf"
    assert content == b"%PDF-1.4 test"
    assert existed
    assert not os.path.exists(path)  # cleaned up after ingest
    assert resp.json()["original_filename"] == "tesla_10q_q3_2023.pdf"


@pytest.mark.parametrize(
    "uploaded, expected",
    [
        ("10-K 2023.pdf", "10-K 2023.pdf"),
        ("../../etc/passwd", "passwd.pdf"),
        ("..\\..\\windows\\win.ini", "win.ini.pdf"),
        ("/abs/path/report.PDF", "report.PDF"),
        ("..", "document.pdf"),
        ("", "document.pdf"),
        (None, "document.pdf"),
        ("q3;rm -rf $HOME`x`.pdf", "q3_rm -rf _HOME_x_.pdf"),
    ],
)
def test_safe_filename(uploaded, expected):
    assert _safe_filename(uploaded) == expected


def test_traversal_name_stays_inside_the_temp_dir():
    seen = []
    with _client_recording_paths(seen) as client:
        client.post(
            "/api/v1/ingest",
            files={"file": ("../../../tmp/evil.pdf", b"%PDF-1.4", "application/pdf")},
        )
    path = Path(seen[0][0])
    assert path.name == "evil.pdf"
    assert path.parent.name.startswith("filing-qa-ingest-")
