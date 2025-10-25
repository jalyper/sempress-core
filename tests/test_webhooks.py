from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from sempress.api import app
from sempress.api.models import JobConfig, JobCreateRequest, JobStatus, JobSummary


def test_dispatch_webhook_sends_payload(tmp_path):
    job_id = "job_test_hook"
    now = datetime.now(timezone.utc)

    request = JobCreateRequest(
        source={"mode": "upload", "filename": "input.csv"},
        config=JobConfig(),
        webhook_url="https://example.com/hooks",
        metadata={"pipeline": "demo"},
    )
    summary = JobSummary(
        job_id=job_id,
        status=JobStatus.succeeded,
        created_at=now,
        updated_at=now,
        compression_ratio=2.5,
    )
    record = app.JobRecord(
        request=request,
        summary=summary,
        upload_token=None,
        tenant_id="tenant-123",
        artifact_tokens={},
        storage_path=tmp_path,
        input_path=None,
    )

    app._JOB_STORE[job_id] = record

    captured = {}

    def fake_sender(url, payload):
        captured["url"] = url
        captured["payload"] = payload

    original_sender = app._WEBHOOK_SENDER
    app._WEBHOOK_SENDER = fake_sender

    try:
        asyncio.run(app._dispatch_webhook(job_id))
    finally:
        app._WEBHOOK_SENDER = original_sender
        app._JOB_STORE.pop(job_id, None)

    assert str(captured["url"]) == "https://example.com/hooks"
    payload = captured["payload"]
    assert payload["job_id"] == job_id
    assert payload["tenant_id"] == "tenant-123"
    assert payload["status"] == JobStatus.succeeded.value
    assert payload["metadata"] == {"pipeline": "demo"}
    assert payload["config"] == JobConfig().model_dump()
    assert payload["job"]["job_id"] == job_id
    assert payload["job"]["status"] == JobStatus.succeeded.value


def test_dispatch_webhook_skips_missing_url(tmp_path):
    job_id = "job_no_hook"
    now = datetime.now(timezone.utc)

    request = JobCreateRequest(
        source={"mode": "upload", "filename": "input.csv"},
        config=JobConfig(),
    )
    summary = JobSummary(
        job_id=job_id,
        status=JobStatus.processing,
        created_at=now,
        updated_at=now,
    )
    record = app.JobRecord(
        request=request,
        summary=summary,
        upload_token=None,
        tenant_id="tenant-xyz",
        artifact_tokens={},
        storage_path=tmp_path,
        input_path=None,
    )

    app._JOB_STORE[job_id] = record

    calls = []

    def fake_sender(url, payload):
        calls.append((url, payload))

    original_sender = app._WEBHOOK_SENDER
    app._WEBHOOK_SENDER = fake_sender

    try:
        asyncio.run(app._dispatch_webhook(job_id))
    finally:
        app._WEBHOOK_SENDER = original_sender
        app._JOB_STORE.pop(job_id, None)

    assert calls == []
