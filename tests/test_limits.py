from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from sempress.api import app


@pytest.mark.parametrize("limit_bytes", [1024])
def test_upload_rejects_payloads_over_limit(monkeypatch, tmp_path, limit_bytes):
    storage_root = tmp_path / "jobs"
    storage_root.mkdir(parents=True, exist_ok=True)
    db_path = storage_root / "jobs.sqlite3"

    monkeypatch.setattr(app, "STORAGE_ROOT", storage_root)
    monkeypatch.setattr(app, "DB_PATH", db_path)
    monkeypatch.setattr(app, "MAX_UPLOAD_BYTES", limit_bytes)

    app._JOB_STORE.clear()
    app._init_db()

    with TestClient(app.create_app()) as client:
        create_payload = {
            "source": {"mode": "upload", "filename": "big.csv"},
            "config": {"lock_cols": [], "residual_cols": [], "k": 8, "uncertainty_threshold": 0.2},
        }

        response = client.post("/v1/jobs", json=create_payload, headers={"X-API-Key": "demo-key"})
        assert response.status_code == 201
        data = response.json()
        job_id = data["job_id"]
        upload_url = data["upload_url"]

        parsed = urlparse(upload_url)
        token = parse_qs(parsed.query)["token"][0]

        too_large = b"x" * (limit_bytes + 1)
        upload_resp = client.put(
            f"/v1/jobs/{job_id}/upload?token={token}",
            data=too_large,
            headers={"X-API-Key": "demo-key", "Content-Type": "application/octet-stream"},
        )
        assert upload_resp.status_code == 413
        detail = upload_resp.json()["detail"]
        assert "exceeds configured limit" in detail

    app._JOB_STORE.clear()


def test_purge_jobs_clears_storage(monkeypatch, tmp_path):
    storage_root = tmp_path / "jobs"
    storage_root.mkdir(parents=True, exist_ok=True)
    db_path = storage_root / "jobs.sqlite3"

    monkeypatch.setattr(app, "STORAGE_ROOT", storage_root)
    monkeypatch.setattr(app, "DB_PATH", db_path)
    monkeypatch.setattr(app, "MAX_UPLOAD_BYTES", None)

    app._JOB_STORE.clear()
    app._init_db()

    with TestClient(app.create_app()) as client:
        create_payload = {
            "source": {"mode": "upload", "filename": "sample.csv"},
            "config": {"lock_cols": [], "residual_cols": [], "k": 8, "uncertainty_threshold": 0.2},
        }

        response = client.post("/v1/jobs", json=create_payload, headers={"X-API-Key": "demo-key"})
        assert response.status_code == 201
        job_id = response.json()["job_id"]
        job_dir = storage_root / job_id
        assert job_dir.exists()

        purge_resp = client.delete("/v1/jobs", headers={"X-API-Key": "demo-key"})
        assert purge_resp.status_code == 200
        assert purge_resp.json()["purged"] == 1
        assert not job_dir.exists()

        with app._db_connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            assert count == 0

    app._JOB_STORE.clear()
