"""Tests for RFdiffusion workflow-compatible result contracts."""

# ruff: noqa: D103

from pathlib import Path
from types import SimpleNamespace

from biomodals.app.design import rfdiffusion_app
from biomodals.helper.app_run import AppRunLayout
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    VolumePath,
)


def test_build_rfdiffusion_hydra_overrides_constructs_structured_args() -> None:
    assert rfdiffusion_app.build_rfdiffusion_hydra_overrides(
        contigs="100-150/0 E333-526",
        num_designs=2,
        hotspot_res="E405 E408",
        noise_scale_ca=0.5,
        noise_scale_frame=0.25,
        rfd_args="diffuser.T=20",
    ) == (
        "inference.num_designs=2 "
        "denoiser.noise_scale_ca=0.5 "
        "denoiser.noise_scale_frame=0.25 "
        "'contigmap.contigs=[100-150/0 E333-526]' "
        "'ppi.hotspot_res=[E405,E408]' "
        "diffuser.T=20"
    )


def test_rfdiffusion_workflow_result_references_cached_output_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    seen_kwargs = {}
    run_dir = tmp_path / "rfd-run"

    def fake_run_rfdiffusion_infer(**kwargs):
        seen_kwargs.update(kwargs)
        layout = AppRunLayout.from_run_root(run_dir)
        layout.outputs_dir.mkdir(parents=True)
        layout.logs_dir.mkdir(parents=True)
        return {
            "run_dir": str(layout.run_root),
            "outputs_dir": str(layout.outputs_dir / "rfd-scaffolds"),
            "log_path": str(layout.logs_dir / "rfd-run-RFdiffusion.log"),
        }

    monkeypatch.setattr(
        rfdiffusion_app,
        "_rfdiffusion_infer",
        fake_run_rfdiffusion_infer,
    )
    monkeypatch.setattr(
        rfdiffusion_app,
        "CONF",
        SimpleNamespace(
            name=rfdiffusion_app.CONF.name,
            output_volume_mountpoint=str(tmp_path),
            output_volume_name=rfdiffusion_app.CONF.output_volume_name,
        ),
    )

    result = rfdiffusion_app.rfdiffusion_infer.get_raw_f()(
        input_pdb_bytes=b"ATOM\n",
        input_pdb_name="input.pdb",
        run_name="../rfd-run",
        hydra_overrides="inference.num_designs=2",
    )

    assert seen_kwargs == {
        "input_pdb_bytes": b"ATOM\n",
        "input_pdb_name": "input.pdb",
        "run_name": "rfd-run",
        "hydra_overrides": "inference.num_designs=2",
    }
    assert result.status == AppRunStatus.SUCCEEDED
    output = result.outputs[0]
    assert output.name == "RFdiffusion_outputs"
    assert output.kind == ArtifactKind.DIRECTORY
    assert output.storage == VolumePath(
        volume_name=rfdiffusion_app.CONF.output_volume_name,
        path="rfd-run/outputs/rfd-scaffolds",
    )
    assert output.metadata == {"run_name": "rfd-run"}
    log = result.logs[0]
    assert log.name == "RFdiffusion_log"
    assert log.kind == ArtifactKind.LOGS
    assert log.storage == VolumePath(
        volume_name=rfdiffusion_app.CONF.output_volume_name,
        path="rfd-run/logs/rfd-run-RFdiffusion.log",
    )


def test_rfdiffusion_local_entrypoint_writes_tarball_from_workflow_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_pdb = tmp_path / "input.pdb"
    input_pdb.write_text("ATOM\n", encoding="utf-8")
    calls = {}

    class FakeWorkflowFunction:
        def remote(self, **kwargs):
            calls["workflow"] = kwargs
            return AppRunResult(
                status=AppRunStatus.SUCCEEDED,
                outputs=[
                    AppOutput(
                        name="RFdiffusion_outputs",
                        kind=ArtifactKind.DIRECTORY,
                        storage=VolumePath(
                            volume_name=rfdiffusion_app.CONF.output_volume_name,
                            path="rfd-run/outputs/rfd-scaffolds",
                        ),
                        metadata={"run_name": "rfd-run"},
                    )
                ],
            )

    class FakeBundleFunction:
        def remote(self, *, run_name: str) -> bytes:
            calls["bundle"] = run_name
            return b"tarball"

    monkeypatch.setattr(
        rfdiffusion_app,
        "rfdiffusion_infer",
        FakeWorkflowFunction(),
    )
    monkeypatch.setattr(
        rfdiffusion_app,
        "bundle_rfdiffusion_outputs",
        FakeBundleFunction(),
    )

    raw_f = rfdiffusion_app.submit_rfdiffusion_task.info.raw_f
    assert raw_f is not None
    raw_f(
        run_name="../rfd-run",
        input_pdb=str(input_pdb),
        contigs="100-150/0 E333-526",
        num_designs=2,
        hotspot_res="E405,E408",
        out_dir=str(tmp_path),
    )

    assert calls["workflow"]["run_name"] == "rfd-run"
    assert calls["workflow"]["input_pdb_bytes"] == b"ATOM\n"
    assert "inference.num_designs=2" in calls["workflow"]["hydra_overrides"]
    assert (
        "contigmap.contigs=[100-150/0 E333-526]" in calls["workflow"]["hydra_overrides"]
    )
    assert "ppi.hotspot_res=[E405,E408]" in calls["workflow"]["hydra_overrides"]
    assert calls["bundle"] == "rfd-run"
    assert (tmp_path / "rfd-run_RFdiffusion.tar.zst").read_bytes() == (b"tarball")
