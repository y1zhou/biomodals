"""Tests for FlowPacker workflow-compatible result contracts."""

# ruff: noqa: D103

from pathlib import Path
from types import SimpleNamespace

import yaml

from biomodals.app.fold import flowpacker_app
from biomodals.schema import AppRunStatus, ArtifactKind, VolumePath


def test_flowpacker_workflow_result_stores_archive_in_volume(
    tmp_path,
    monkeypatch,
) -> None:
    seen_kwargs = {}

    class FakeRunFlowPacker:
        def get_raw_f(self):
            def fake_run_flowpacker(**kwargs):
                seen_kwargs.update(kwargs)
                return b"tarball"

            return fake_run_flowpacker

    class FakeOutputVolume:
        def __init__(self):
            self.commit_count = 0

        def commit(self):
            self.commit_count += 1

    output_volume = FakeOutputVolume()
    output_volume_name = flowpacker_app.CONF.output_volume_name
    monkeypatch.setattr(flowpacker_app, "run_flowpacker", FakeRunFlowPacker())
    monkeypatch.setattr(
        flowpacker_app,
        "CONF",
        SimpleNamespace(
            output_volume=output_volume,
            output_volume_mountpoint=str(tmp_path),
            output_volume_name=output_volume_name,
        ),
    )

    result = flowpacker_app.run_flowpacker_workflow.get_raw_f()(
        input_files=[("input.pdb", b"ATOM\n")],
        run_name="../packed",
    )

    assert seen_kwargs["run_name"] == "packed"
    assert result.status == AppRunStatus.SUCCEEDED
    assert len(result.outputs) == 1
    output = result.outputs[0]
    assert output.name == "flowpacker_outputs"
    assert output.kind == ArtifactKind.ARCHIVE
    assert output.storage == VolumePath(
        volume_name=output_volume_name,
        path="workflow/packed/packed.tar.zst",
        media_type="application/zstd",
    )
    assert output.metadata == {
        "archive_format": "tar.zst",
        "filename": "packed.tar.zst",
    }
    assert (
        Path(tmp_path) / "workflow" / "packed" / "packed.tar.zst"
    ).read_bytes() == b"tarball"
    assert output_volume.commit_count == 1


def test_flowpacker_config_uses_volume_checkpoint_paths(tmp_path) -> None:
    config_path = tmp_path / "biomodals.yaml"
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()

    flowpacker_app._write_flowpacker_config(
        config_path,
        input_dir=input_dir,
        model_name="cluster",
        use_confidence=True,
        n_samples=1,
        num_steps=10,
        sample_coeff=5.0,
    )

    config = yaml.safe_load(config_path.read_text())
    assert Path(config["ckpt"]) == flowpacker_app._checkpoint_path("cluster")
    assert Path(config["ckpt"]).is_absolute()
    assert (
        Path(config["ckpt"]).parent == flowpacker_app.CONF.git_clone_dir / "checkpoints"
    )
    assert Path(config["conf_ckpt"]) == flowpacker_app._checkpoint_path("confidence")


def test_flowpacker_checkpoint_download_copies_git_lfs_files_to_volume(
    tmp_path,
    monkeypatch,
) -> None:
    class FakeModelVolume:
        def __init__(self):
            self.commit_count = 0

        def commit(self):
            self.commit_count += 1

    git_clone_dir = tmp_path / "FlowPacker"
    checkpoint_dir = git_clone_dir / "checkpoints"
    cache_dir = tmp_path / "model-cache"
    checkpoint_dir.mkdir(parents=True)
    cache_dir.mkdir()

    fake_conf = SimpleNamespace(
        git_clone_dir=git_clone_dir,
        model_volume_mountpoint=str(cache_dir),
    )
    fake_model_volume = FakeModelVolume()

    def fake_run_command(cmd, *, cwd=None, env=None):
        if cmd[:3] == ["git", "lfs", "pull"]:
            assert cwd == git_clone_dir
            assert env == {"GIT_LFS_SKIP_SMUDGE": "0"}
            for checkpoint_name in flowpacker_app.APP_INFO.checkpoint_names:
                (checkpoint_dir / f"{checkpoint_name}.pth").write_bytes(
                    f"{checkpoint_name}-weights".encode()
                )

    monkeypatch.setattr(flowpacker_app, "CONF", fake_conf)
    monkeypatch.setattr(flowpacker_app, "MODEL_VOLUME", fake_model_volume)
    monkeypatch.setattr("biomodals.helper.shell.run_command", fake_run_command)

    flowpacker_app.download_flowpacker_checkpoints.get_raw_f()(force=False)

    for checkpoint_name in flowpacker_app.APP_INFO.checkpoint_names:
        assert (cache_dir / f"{checkpoint_name}.pth").read_bytes() == (
            f"{checkpoint_name}-weights".encode()
        )
    assert fake_model_volume.commit_count == 1
