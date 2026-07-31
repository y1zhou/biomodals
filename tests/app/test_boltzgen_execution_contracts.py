"""Tests for BoltzGen publication claims and completion validation."""

# ruff: noqa: D101,D102,D103,D107

from pathlib import Path
from types import SimpleNamespace

import pytest

from biomodals.app.design import boltzgen_app
from biomodals.app.design.boltzgen.execution_contracts import (
    acquire_output_claim,
    boltzgen_output_claim_key,
    is_boltzgen_run_complete,
    write_boltzgen_task_publication,
)


class FakeVolume:
    def __init__(self) -> None:
        self.commits = 0
        self.reloads = 0

    def commit(self) -> None:
        self.commits += 1

    def reload(self) -> None:
        self.reloads += 1


class FakeClaimStore:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def get(self, key: str, default=None):
        return self.values.get(key, default)

    def put(self, key: str, value: str, *, skip_if_exists: bool = False) -> bool:
        if skip_if_exists and key in self.values:
            return False
        self.values[key] = value
        return True


def test_claim_requires_same_owner_or_explicit_terminal_replacement(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    store = FakeClaimStore()
    claim_key = boltzgen_output_claim_key(run_dir, output_root=tmp_path)

    acquire_output_claim(store, claim_key=claim_key, owner="call-1")
    acquire_output_claim(store, claim_key=claim_key, owner="call-1")
    with pytest.raises(RuntimeError, match="another Provider Call"):
        acquire_output_claim(store, claim_key=claim_key, owner="call-2")

    acquire_output_claim(
        store,
        claim_key=claim_key,
        owner="call-2",
        replace_owner="call-1",
    )
    with pytest.raises(RuntimeError, match="another Provider Call"):
        acquire_output_claim(
            store,
            claim_key=claim_key,
            owner="call-3",
            replace_owner="call-1",
        )
    assert not (run_dir / ".lock").exists()


def test_worker_preserves_claim_on_failure_and_redelivery_finishes_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    volume = FakeVolume()
    claim_store = FakeClaimStore()
    out_dir = tmp_path / "run"
    monkeypatch.setattr(
        boltzgen_app,
        "CONF",
        SimpleNamespace(
            output_volume=volume,
            output_volume_mountpoint=str(tmp_path),
            output_volume_name="outputs",
        ),
    )
    monkeypatch.setattr(boltzgen_app, "BOLTZGEN_OUTPUT_CLAIMS", claim_store)

    def fail(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("preempted")

    monkeypatch.setattr(boltzgen_app, "run_command", fail)
    worker = boltzgen_app.run_boltzgen_task.get_raw_f()
    task_fingerprint = "a" * 64
    with pytest.raises(RuntimeError, match="preempted"):
        worker(
            out_dir=str(out_dir),
            input_yaml_path=str(tmp_path / "input.yaml"),
            claim_owner="call-1",
            task_fingerprint=task_fingerprint,
        )

    assert "call-1" in claim_store.values.values()
    assert volume.commits == 0

    def finish(*args, **kwargs):
        del args, kwargs
        final = out_dir / "final_ranked_designs"
        final.mkdir(parents=True)
        (final / "results_overview.pdf").write_bytes(b"pdf")

    monkeypatch.setattr(boltzgen_app, "run_command", finish)
    assert worker(
        out_dir=str(out_dir),
        input_yaml_path=str(tmp_path / "input.yaml"),
        claim_owner="call-1",
        task_fingerprint=task_fingerprint,
    ) == str(out_dir)
    assert is_boltzgen_run_complete(
        out_dir,
        task_fingerprint=task_fingerprint,
    )
    assert not is_boltzgen_run_complete(
        out_dir,
        task_fingerprint="b" * 64,
    )
    assert "call-1" in claim_store.values.values()
    assert volume.commits == 2


def test_collection_commits_artifacts_before_its_publication_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication_path = Path("example/results/fingerprint.json")
    marker = tmp_path / publication_path
    commit_states: list[bool] = []
    volume = SimpleNamespace(
        reload=lambda: None,
        commit=lambda: commit_states.append(marker.is_file()),
    )
    monkeypatch.setattr(
        boltzgen_app,
        "CONF",
        SimpleNamespace(
            output_volume=volume,
            output_volume_mountpoint=str(tmp_path),
            output_volume_name="outputs",
        ),
    )
    run_dir = tmp_path / "example" / "outputs" / "run-a"
    final = run_dir / "final_ranked_designs"
    final.mkdir(parents=True)
    (final / "results_overview.pdf").write_bytes(b"pdf")
    task_fingerprint = "a" * 64
    write_boltzgen_task_publication(
        run_dir,
        task_fingerprint=task_fingerprint,
    )

    result = boltzgen_app.collect_boltzgen_data.get_raw_f()(
        run_name="example",
        run_ids=["run-a"],
        task_fingerprints={"run-a": task_fingerprint},
        filter_results=False,
        publication_path=publication_path.as_posix(),
    )

    assert result["status"] == "complete"
    assert commit_states == [False, True]


def test_worker_rejects_output_outside_the_mounted_volume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    volume = FakeVolume()
    monkeypatch.setattr(
        boltzgen_app,
        "CONF",
        SimpleNamespace(
            output_volume=volume,
            output_volume_mountpoint=str(tmp_path),
            output_volume_name="outputs",
        ),
    )
    worker = boltzgen_app.run_boltzgen_task.get_raw_f()

    with pytest.raises(ValueError, match="inside the output Volume"):
        worker(
            out_dir=str(tmp_path.parent / "escape"),
            input_yaml_path=str(tmp_path / "input.yaml"),
            claim_owner="call-1",
            task_fingerprint="a" * 64,
        )

    assert not (tmp_path.parent / "escape").exists()
    assert volume.reloads == 0


def test_task_publication_rejects_corrupt_final_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    final_pdf = run_dir / "final_ranked_designs" / "results_overview.pdf"
    final_pdf.parent.mkdir(parents=True)
    final_pdf.write_bytes(b"pdf")
    fingerprint = "a" * 64

    write_boltzgen_task_publication(
        run_dir,
        task_fingerprint=fingerprint,
    )

    assert is_boltzgen_run_complete(
        run_dir,
        task_fingerprint=fingerprint,
    )
    final_pdf.write_bytes(b"corrupt")
    assert not is_boltzgen_run_complete(
        run_dir,
        task_fingerprint=fingerprint,
    )
