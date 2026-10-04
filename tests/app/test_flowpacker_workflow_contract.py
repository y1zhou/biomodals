"""Tests for FlowPacker workflow-compatible result contracts."""

# ruff: noqa: D103

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from biomodals.app.fold import flowpacker_app
from biomodals.schema import AppRunStatus


@pytest.mark.parametrize(
    "failure", [None, "empty", "partial", "corrupt", "confidence", "exit"]
)
def test_flowpacker_publishes_only_complete_structures(tmp_path, monkeypatch, failure):
    """Exercise parsing, packaging, publication and cleanup; replace only the model."""
    import io
    import subprocess
    import tarfile

    import orjson
    import zstandard

    from biomodals.helper.shell import run_command

    repo = tmp_path / "upstream"
    repo.mkdir()
    checkpoints = tmp_path / "models"
    checkpoints.mkdir()
    (repo / "checkpoints").symlink_to(checkpoints, target_is_directory=True)
    for name in ("cluster", "confidence"):
        (checkpoints / f"{name}.pth").write_bytes(b"fixture weights")
    monkeypatch.setattr(
        flowpacker_app,
        "CONF",
        SimpleNamespace(
            git_clone_dir=repo,
            model_volume_mountpoint=str(checkpoints),
            output_volume_mountpoint=str(tmp_path / "volume"),
            output_volume_name="test",
            repo_commit_hash="pinned",
            output_volume=SimpleNamespace(commit=lambda: None),
        ),
    )
    staged = []
    pdb = b"ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C\nEND\n"

    def model(cmd, **kwargs):
        import sys

        config = yaml.safe_load(
            (repo / "config/inference" / f"{cmd[2]}.yaml").read_text()
        )
        staged.append(Path(config["data"]["test_path"]))
        run_command([sys.executable, "-c", "print('model diagnostic')"], **kwargs)
        if failure == "exit":
            raise subprocess.CalledProcessError(1, cmd)
        output = repo / "samples" / cmd[3]
        output.mkdir(parents=True)
        (output / "output_dict.pth").write_bytes(b"fixture metrics")
        for folder in ("run_1", "run_2", "best_run"):
            if failure == "empty" or (failure == "confidence" and folder == "best_run"):
                continue
            (output / folder).mkdir(parents=True)
            for stem in ("alpha", "beta"):
                if failure == "partial" and stem == "beta":
                    continue
                (output / folder / f"{stem}.pdb").write_bytes(
                    b"invalid" if failure == "corrupt" else pdb
                )

    monkeypatch.setattr(flowpacker_app, "run_command", model)
    kwargs = dict(
        input_files=[("alpha.pdb", pdb), ("beta.pdb", pdb)],
        run_name="packed",
        n_samples=2,
        use_confidence=True,
    )
    archive = tmp_path / "volume/workflow/packed/outputs/packed.tar.zst"
    if failure:
        with pytest.raises((
            ValueError,
            FileNotFoundError,
            subprocess.CalledProcessError,
        )):
            flowpacker_app.run_flowpacker_workflow.local(**kwargs)
        assert not archive.exists()
    else:
        result = flowpacker_app.run_flowpacker_workflow.local(**kwargs)
        assert result.status == AppRunStatus.SUCCEEDED
        with zstandard.ZstdDecompressor().stream_reader(
            io.BytesIO(archive.read_bytes())
        ) as stream:
            with tarfile.open(fileobj=stream, mode="r|") as tar:
                files = {
                    member.name: tar.extractfile(member).read()
                    for member in tar
                    if member.isfile()
                }
        manifest = orjson.loads(
            next(
                data
                for name, data in files.items()
                if name.endswith("/validation.json")
            )
        )
        assert len(manifest["structures"]) == 6
        assert len([name for name in files if name.endswith(".pdb")]) == 6
    assert staged and all(not path.exists() for path in staged)
    assert not list((repo / "samples").glob("*"))
    assert (
        "model diagnostic"
        in (tmp_path / "volume/workflow/packed/logs/flowpacker.log").read_text()
    )


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

    flowpacker_app.download_flowpacker_checkpoints.local(force=False)

    for checkpoint_name in flowpacker_app.APP_INFO.checkpoint_names:
        assert (cache_dir / f"{checkpoint_name}.pth").read_bytes() == (
            f"{checkpoint_name}-weights".encode()
        )
    assert fake_model_volume.commit_count == 1
