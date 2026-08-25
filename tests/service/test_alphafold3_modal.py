"""AlphaFold3 service submission coordination contracts."""

# ruff: noqa: D101,D102,D103

from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from biomodals.service.alphafold3 import modal as af3_modal
from biomodals.service.alphafold3.modal import AlphaFold3ToolAdapter
from biomodals.service.store import JobState, ServiceStore


def _store(tmp_path: Path) -> tuple[ServiceStore, UUID]:
    store = ServiceStore(tmp_path / "service.sqlite3")
    store.initialize()
    user = store.create_user(
        email="user@example.com",
        display_name="User",
        token_digest=b"setup",
        token_expires_at=100,
        now=1,
        is_admin=True,
        active_job_limit=10,
    )
    store.set_password_from_token(
        b"setup",
        password_hash="test",  # noqa: S106
        session_token_digest=b"session",
        csrf_digest=b"csrf",
        now=2,
        absolute_expires_at=1000,
    )
    return store, user.user_id


def _admit(
    store: ServiceStore,
    owner: UUID,
    *,
    job_id: UUID,
    ordinal: int,
    request_digest: str = "a" * 64,
    publication_scope_digest: str = "b" * 64,
):
    return store.admit_job(
        owner_user_id=owner,
        tool="alphafold3",
        display_name="prediction",
        idempotency_key=f"request-{ordinal}",
        request_digest=request_digest,
        publication_scope_digest=publication_scope_digest,
        modal_environment="main",
        modal_app_name="AlphaFold3",
        modal_app_version=1,
        tool_active_job_limit=10,
        global_active_job_limit=10,
        max_active_provider_calls=4,
        max_active_gpu_provider_calls=1,
        pending_validation_id=uuid4(),
        now=10 + ordinal,
        new_job_id=job_id,
    ).job


@pytest.mark.anyio
async def test_identical_job_waits_behind_active_predecessor(tmp_path: Path) -> None:
    store, owner = _store(tmp_path)
    _admit(
        store,
        owner,
        job_id=uuid4(),
        ordinal=1,
        request_digest="a" * 64,
    )
    current = _admit(
        store,
        owner,
        job_id=uuid4(),
        ordinal=2,
        request_digest="c" * 64,
    )
    adapter = AlphaFold3ToolAdapter(SimpleNamespace(), store)

    waiting = await adapter.stage(current)

    assert waiting is not None
    assert waiting.reason == "waiting_for_shared_publication"


@pytest.mark.anyio
async def test_terminal_predecessor_authorizes_exact_claim_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, owner = _store(tmp_path)
    predecessor = _admit(store, owner, job_id=uuid4(), ordinal=1)
    predecessor = store.request_cancel(predecessor.job_id, now=12)
    assert predecessor.state == JobState.CANCELLED
    current = _admit(store, owner, job_id=uuid4(), ordinal=2)
    captured: dict[str, object] = {}

    class Validated:
        def request(self, **kwargs):
            captured.update(kwargs)
            return object()

    class Validations:
        def get_claimed(self, *_args, **_kwargs):
            return Validated()

    async def hydrate() -> None:
        return None

    async def to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    volume = SimpleNamespace(hydrate=SimpleNamespace(aio=hydrate))
    adapter = AlphaFold3ToolAdapter(Validations(), store)
    monkeypatch.setattr(adapter, "_volume", lambda _job: volume)
    monkeypatch.setattr(af3_modal.asyncio, "to_thread", to_thread)
    monkeypatch.setattr(af3_modal, "stage_execution_request", lambda *_args: None)
    monkeypatch.setattr(af3_modal, "stage_execution_launch", lambda *_args: None)

    assert await adapter.stage(current) is None
    assert captured["repair_execution_run_ids"] == (predecessor.job_id,)
