"""Administrator and per-User policy persistence contracts."""

# ruff: noqa: D103

from pathlib import Path

import pytest

from biomodals.service.store import LastActiveAdminError, ServiceStore


def _user(
    store: ServiceStore,
    email: str,
    *,
    is_admin: bool = False,
    active_job_limit: int = 2,
):
    return store.create_user(
        email=email,
        display_name=email.partition("@")[0],
        token_digest=email.encode(),
        token_expires_at=100,
        now=1,
        is_admin=is_admin,
        active_job_limit=active_job_limit,
    )


def test_user_admin_status_and_active_job_limit_are_configurable(
    tmp_path: Path,
) -> None:
    store = ServiceStore(tmp_path / "state.sqlite3")
    store.initialize()
    administrator = _user(store, "admin@example.com", is_admin=True)
    user = _user(store, "alice@example.com", active_job_limit=3)

    updated = store.update_user(
        user.user_id,
        is_admin=True,
        active_job_limit=7,
        now=2,
    )

    assert updated.is_admin is True
    assert updated.active_job_limit == 7
    assert store.list_users() == [administrator, updated]


def test_first_user_must_be_an_administrator(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "state.sqlite3")
    store.initialize()

    with pytest.raises(ValueError, match="first User.*administrator"):
        _user(store, "ordinary@example.com")

    administrator = _user(store, "admin@example.com", is_admin=True)
    assert administrator.is_admin is True


def test_last_active_admin_cannot_be_disabled_or_demoted(tmp_path: Path) -> None:
    store = ServiceStore(tmp_path / "state.sqlite3")
    store.initialize()
    first = _user(store, "alice@example.com", is_admin=True)

    with pytest.raises(LastActiveAdminError):
        store.update_user(first.user_id, active=False, now=2)
    with pytest.raises(LastActiveAdminError):
        store.update_user(first.user_id, is_admin=False, now=2)

    second = _user(store, "bob@example.com", is_admin=True)
    demoted = store.update_user(first.user_id, is_admin=False, now=3)

    assert demoted.is_admin is False
    stored_second = store.get_user(second.user_id)
    assert stored_second is not None
    assert stored_second.is_admin is True
