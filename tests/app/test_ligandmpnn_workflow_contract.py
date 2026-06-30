"""Tests for LigandMPNN workflow-compatible result contracts."""

# ruff: noqa: D103

import tempfile
from pathlib import Path

from biomodals.app.design import ligandmpnn_app
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    InlineBytes,
)
from biomodals.schema.storage import ZSTD_MEDIA_TYPE


class UnexpectedRemoteFunction:
    """Sentinel remote object for paths a test must not call."""

    def remote(self, *args: object, **kwargs: object) -> object:
        """Fail if the sentinel is invoked."""
        raise AssertionError(
            f"Unexpected live remote call: args={args}, kwargs={kwargs}"
        )


def test_build_ligandmpnn_cli_args_constructs_run_mode_args() -> None:
    assert ligandmpnn_app.build_ligandmpnn_cli_args(
        script_mode="run",
        model_type="protein_mpnn",
        batch_size=4,
        number_of_batches=3,
        parse_atoms_with_zero_occupancy=True,
        pack_side_chains=True,
        number_of_packs_per_design=5,
        sc_num_samples=7,
        repack_everything=True,
        redesigned_residues="A1 A2",
    ) == {
        "--model_type": "protein_mpnn",
        "--batch_size": "4",
        "--number_of_batches": "3",
        "--parse_atoms_with_zero_occupancy": True,
        "--temperature": "0.1",
        "--save_stats": "1",
        "--pack_side_chains": True,
        "--number_of_packs_per_design": "5",
        "--repack_everything": True,
        "--pack_with_ligand_context": True,
        "--sc_num_denoising_steps": "3",
        "--sc_num_samples": "7",
        "--redesigned_residues": "A1 A2",
    }


def test_build_base_command_uses_app_run_layout_for_scratch_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    cli_args: dict[str, str | int | float | bool] = {"--model_type": "protein_mpnn"}

    cmd, workdir = ligandmpnn_app.build_base_command(
        run_name="mpnn-run",
        script_mode="run",
        struct_bytes=b"ATOM\n",
        cli_args=cli_args,
        bias_aa_per_residue_bytes=b"{}",
        omit_aa_per_residue_bytes=b"{}",
    )

    assert workdir == tmp_path / "mpnn-run"
    assert (workdir / "inputs" / "mpnn-run.pdb").read_bytes() == b"ATOM\n"
    assert (workdir / "inputs" / "bias_AA_per_residue.json").read_bytes() == b"{}"
    assert (workdir / "inputs" / "omit_AA_per_residue.json").read_bytes() == b"{}"
    assert (workdir / "outputs").is_dir()
    assert (workdir / "logs").is_dir()
    assert cmd[1] == str(ligandmpnn_app.CONF.git_clone_dir / "run.py")
    assert cli_args["--pdb_path"] == str(workdir / "inputs" / "mpnn-run.pdb")
    assert cli_args["--bias_AA_per_residue"] == str(
        workdir / "inputs" / "bias_AA_per_residue.json"
    )
    assert cli_args["--omit_AA_per_residue"] == str(
        workdir / "inputs" / "omit_AA_per_residue.json"
    )


def test_ligandmpnn_workflow_result_returns_inline_zstd_archive(
    monkeypatch,
) -> None:
    seen_kwargs = {}

    def fake_ligandmpnn_run(**kwargs):
        seen_kwargs.update(kwargs)
        return b"tarball"

    monkeypatch.setattr(ligandmpnn_app, "_ligandmpnn_run", fake_ligandmpnn_run)

    result = ligandmpnn_app.ligandmpnn_run.get_raw_f()(
        run_name="../mpnn-run",
        script_mode="run",
        struct_bytes=b"ATOM\n",
        seeds=[1, 2],
        cli_args={
            "--model_type": "protein_mpnn",
            "--batch_size": "4",
            "--number_of_batches": "3",
        },
    )

    assert seen_kwargs == {
        "run_name": "mpnn-run",
        "script_mode": "run",
        "struct_bytes": b"ATOM\n",
        "seeds": [1, 2],
        "cli_args": {
            "--model_type": "protein_mpnn",
            "--batch_size": "4",
            "--number_of_batches": "3",
        },
        "bias_aa_per_residue_bytes": None,
        "omit_aa_per_residue_bytes": None,
    }
    assert result.status == AppRunStatus.SUCCEEDED
    output = result.outputs[0]
    assert output.name == "LigandMPNN_outputs"
    assert output.kind == ArtifactKind.ARCHIVE
    assert output.storage == InlineBytes(
        data=b"tarball",
        filename="mpnn-run_LigandMPNN.tar.zst",
        media_type=ZSTD_MEDIA_TYPE,
    )
    assert output.metadata == {
        "archive_format": "tar.zst",
        "run_name": "mpnn-run",
    }


def test_ligandmpnn_local_entrypoint_writes_tarball_from_inline_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text("ATOM\n", encoding="utf-8")
    calls = {}

    class FakeDownloadWeights:
        def remote(self, *, force: bool) -> None:
            calls["download_force"] = force

    class FakeWorkflowFunction:
        def remote(self, **kwargs):
            calls["workflow"] = kwargs
            return AppRunResult(
                status=AppRunStatus.SUCCEEDED,
                outputs=[
                    AppOutput(
                        name="LigandMPNN_outputs",
                        kind=ArtifactKind.ARCHIVE,
                        storage=InlineBytes(
                            data=b"tarball",
                            filename="mpnn-run_LigandMPNN.tar.zst",
                            media_type=ZSTD_MEDIA_TYPE,
                        ),
                    )
                ],
            )

    monkeypatch.setattr(ligandmpnn_app, "download_weights", FakeDownloadWeights())
    monkeypatch.setattr(
        ligandmpnn_app,
        "ligandmpnn_run",
        FakeWorkflowFunction(),
    )

    raw_f = ligandmpnn_app.submit_ligandmpnn_task.info.raw_f
    assert raw_f is not None
    raw_f(
        input_pdb=str(input_pdb),
        script_mode="run",
        out_dir=str(tmp_path),
        run_name="../mpnn-run",
        force_download_models=True,
        model_type="protein_mpnn",
        seeds="1,2",
        batch_size=4,
        number_of_batches=3,
        parse_atoms_with_zero_occupancy=True,
        pack_side_chains=True,
        number_of_packs_per_design=5,
        sc_num_samples=7,
        repack_everything=True,
    )

    assert calls["download_force"] is True
    assert calls["workflow"]["run_name"] == "mpnn-run"
    assert calls["workflow"]["script_mode"] == "run"
    assert calls["workflow"]["struct_bytes"] == b"ATOM\n"
    assert calls["workflow"]["seeds"] == [1, 2]
    assert calls["workflow"]["cli_args"] == {
        "--model_type": "protein_mpnn",
        "--batch_size": "4",
        "--number_of_batches": "3",
        "--parse_atoms_with_zero_occupancy": True,
        "--temperature": "0.1",
        "--save_stats": "1",
        "--pack_side_chains": True,
        "--number_of_packs_per_design": "5",
        "--repack_everything": True,
        "--pack_with_ligand_context": True,
        "--sc_num_denoising_steps": "3",
        "--sc_num_samples": "7",
    }
    assert (tmp_path / "mpnn-run_LigandMPNN.tar.zst").read_bytes() == b"tarball"
