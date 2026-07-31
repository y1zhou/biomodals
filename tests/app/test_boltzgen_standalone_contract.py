"""Tests for BoltzGen app run layout behavior."""

# ruff: noqa: D101,D102,D103,D107

from pathlib import Path
from types import SimpleNamespace

from biomodals.app.design import boltzgen_app


class FakeOutputVolume:
    def __init__(self) -> None:
        self.commit_count = 0
        self.reload_count = 0

    def commit(self) -> None:
        self.commit_count += 1

    def reload(self) -> None:
        self.reload_count += 1


def test_prepare_boltzgen_run_uses_app_run_layout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_volume = FakeOutputVolume()
    monkeypatch.setattr(
        boltzgen_app,
        "CONF",
        SimpleNamespace(
            output_volume_mountpoint=str(tmp_path), output_volume=output_volume
        ),
    )

    boltzgen_app.prepare_boltzgen_run.get_raw_f()(
        yaml_content=b"name: demo\n",
        run_name="demo",
        additional_files={"templates/input.cif": b"data"},
    )

    run_root = tmp_path / "demo"
    assert (
        run_root / "inputs" / "config" / "demo.yaml"
    ).read_bytes() == b"name: demo\n"
    assert (
        run_root / "inputs" / "config" / "templates" / "input.cif"
    ).read_bytes() == b"data"
    assert (run_root / "outputs").is_dir()
    assert output_volume.commit_count == 1


def test_get_run_ids_salvage_mode_reads_outputs_from_app_run_layout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_volume = FakeOutputVolume()
    run_root = tmp_path / "demo"
    complete = run_root / "outputs" / "complete"
    complete.joinpath("final_ranked_designs").mkdir(parents=True)
    complete.joinpath("final_ranked_designs", "results_overview.pdf").write_bytes(
        b"pdf"
    )
    incomplete = run_root / "outputs" / "incomplete"
    incomplete.mkdir()

    monkeypatch.setattr(
        boltzgen_app,
        "CONF",
        SimpleNamespace(
            output_volume_mountpoint=str(tmp_path), output_volume=output_volume
        ),
    )

    assert boltzgen_app.get_run_ids.get_raw_f()(
        run_name="demo",
        num_parallel_runs=2,
        salvage_mode=True,
    ) == ["complete", "incomplete"]
    assert boltzgen_app.get_run_ids.get_raw_f()(
        run_name="demo",
        num_parallel_runs=2,
        salvage_mode=True,
        skip_finished=True,
    ) == ["incomplete"]
    assert output_volume.reload_count == 2
