"""Pinned HuDiff source, runtime, and staged checkpoint identities."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import re
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.request import urlopen
from uuid import uuid4

import orjson

SOURCE_COMMIT = "bb7636f182699f98c37855dad05a5c6c61b576bd"
MODEL_REVISION = "3455856e5d97aa72dea98653b44a1aec800fc998"
ARCHIVE_URL = (
    "https://huggingface.co/cloud77/HuDiff/resolve/"
    f"{MODEL_REVISION}/release_data_dir.tar.gz"
)
ARCHIVE_SIZE_BYTES = 2_070_382_005
ARCHIVE_SHA256 = "95d4e9091463939e3032996ae10bee8c8df72d11326d4498344eb1abe0bcb949"
MODEL_MANIFEST = "manifest.json"
ANTIBODY_CHECKPOINT = "checkpoints/antibody/hudiffab.pt"


@dataclass(frozen=True, slots=True)
class CheckpointSpec:
    """One retained file from the immutable published model archive."""

    path: str
    size_bytes: int
    sha256: str


CHECKPOINTS = (
    CheckpointSpec(
        "checkpoints/abnativ/vh_model.ckpt",
        194_187_065,
        "de1a3fdaaa9ae178a77478602dbea7f04f23bd13acea41bdbe0eaa6cfa8dbdbb",
    ),
    CheckpointSpec(
        "checkpoints/abnativ/vhh_model.ckpt",
        194_187_397,
        "eb22d50d30729eaa61b9cc6083b0fbc8a0a3dcef181ee08663593ab7173801d8",
    ),
    CheckpointSpec(
        "checkpoints/abnativ/vkappa_model.ckpt",
        194_188_393,
        "937f5ef0e8a1a3594f1f84157c22421f8ca247f83d2a8f2edf7ca347c75f1281",
    ),
    CheckpointSpec(
        "checkpoints/abnativ/vlambda_model.ckpt",
        194_188_725,
        "4ee34a8514d53f4dffbbcc53d84ab62de2081cf85a96a142b86ccd524be5dbc3",
    ),
    CheckpointSpec(
        ANTIBODY_CHECKPOINT,
        479_136_082,
        "204e1c69aff239555efe76c10ba316d5348394c222d3d89f4089e6103ce21227",
    ),
    CheckpointSpec(
        "checkpoints/nanobody/hudiffnb.pt",
        418_768_789,
        "2c103342c6ded156a3bad7b1273afba911bac42c9a18ba11e3498ca1f3c6b4e2",
    ),
)
ANTIBODY_CHECKPOINT_SHA256 = next(
    item.sha256 for item in CHECKPOINTS if item.path == ANTIBODY_CHECKPOINT
)
RUNTIME_ENVIRONMENT_SHA256 = (
    "6a9d9cce1584eb6cde0a838e1523e317c05dcd0cbe31d7cfb1830ff484edb67c"
)
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
CUDA_DETERMINISM_POLICY = "torch-strict+cudnn-deterministic+cublas-4096x8"
RUNTIME_IDENTITY = "|".join((
    f"hudiff={SOURCE_COMMIT}",
    f"models={MODEL_REVISION}",
    "python=3.10",
    "torch=1.13.0+cu116",
    "cuda=11.6",
    "numpy=1.23.5",
    "sequence-models=1.8.0",
    "abnumber=0.3.2",
    "anarci=2020.04.23",
    "hmmer=3.3.2",
    "biopython=1.79",
    "pandas=1.5.3",
    "scipy=1.9.3",
    "easydict=1.13",
    "einops=0.6.1",
    "pyyaml=6.0.3",
    "tqdm=4.70.0",
    f"resolved-environment={RUNTIME_ENVIRONMENT_SHA256}",
    f"cuda-determinism={CUDA_DETERMINISM_POLICY}",
    "wrapper-protocol=3",
    "patch-protocol=1",
))


def runtime_environment_sha256() -> str:
    """Fingerprint every resolved Conda build and Python distribution."""
    conda = sorted(
        (
            item["name"],
            item["version"],
            item["build"],
            item["channel"],
        )
        for path in Path("/opt/conda/conda-meta").glob("*.json")
        if isinstance((item := orjson.loads(path.read_bytes())), dict)
    )
    if not conda:
        raise RuntimeError("Conda returned an empty package inventory")
    python = sorted(
        (
            re.sub(r"[-_.]+", "-", name).lower(),
            distribution.version,
        )
        for distribution in importlib.metadata.distributions()
        if (name := distribution.metadata["Name"])
    )
    payload = {"conda": conda, "python": python}
    return hashlib.sha256(
        orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    ).hexdigest()


def assert_runtime_environment() -> str:
    """Fail the image build if dependency resolution drifts from its identity."""
    observed = runtime_environment_sha256()
    if observed != RUNTIME_ENVIRONMENT_SHA256:
        raise RuntimeError(
            "HuDiff-Ab resolved runtime changed: "
            f"expected {RUNTIME_ENVIRONMENT_SHA256}, observed {observed}"
        )
    return observed


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_revision": MODEL_REVISION,
        "archive_size_bytes": ARCHIVE_SIZE_BYTES,
        "archive_sha256": ARCHIVE_SHA256,
        "checkpoints": [
            {
                "path": item.path,
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
            }
            for item in CHECKPOINTS
        ],
    }


def assert_hudiff_assets(root: Path) -> dict[str, object]:
    """Validate the manifest and every retained inference checkpoint."""
    try:
        manifest = orjson.loads((root / MODEL_MANIFEST).read_bytes())
    except (OSError, orjson.JSONDecodeError) as exc:
        raise RuntimeError("HuDiff model manifest is missing or invalid") from exc
    if manifest != _expected_manifest():
        raise RuntimeError("HuDiff model manifest has an unexpected identity")
    for item in CHECKPOINTS:
        path = root.joinpath(*PurePosixPath(item.path).parts)
        if (
            not path.is_file()
            or path.stat().st_size != item.size_bytes
            or _digest(path) != item.sha256
        ):
            raise RuntimeError(f"HuDiff checkpoint is missing or corrupt: {item.path}")
    return manifest


def _checkpoint_member(name: str) -> PurePosixPath | None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe HuDiff archive member: {name}")
    try:
        index = path.parts.index("checkpoints")
    except ValueError:
        return None
    relative = PurePosixPath(*path.parts[index:])
    return (
        relative if relative.as_posix() in {item.path for item in CHECKPOINTS} else None
    )


def _publish_checkpoints(root: Path, staging: Path) -> None:
    """Atomically replace identical checkpoint files, then publish the manifest."""
    for item in CHECKPOINTS:
        relative = PurePosixPath(item.path)
        source = staging.joinpath(*relative.parts)
        destination = root.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
    temporary = root / f".{MODEL_MANIFEST}.{uuid4().hex}.part"
    temporary.write_bytes(
        orjson.dumps(_expected_manifest(), option=orjson.OPT_SORT_KEYS)
    )
    os.replace(temporary, root / MODEL_MANIFEST)


def stage_hudiff_assets(root: Path) -> dict[str, object]:
    """Download, verify, and atomically publish the checkpoint-only subtree."""
    root.mkdir(parents=True, exist_ok=True)
    try:
        return assert_hudiff_assets(root)
    except RuntimeError:
        pass

    archive = root / f".release-data.{uuid4().hex}.part"
    staging = root / f".checkpoints.{uuid4().hex}.part"
    digest = hashlib.sha256()
    size = 0
    try:
        with (
            urlopen(ARCHIVE_URL, timeout=300) as response,  # noqa: S310
            archive.open("wb") as output,
        ):  # noqa: S310
            while chunk := response.read(8 * 1024 * 1024):
                size += len(chunk)
                if size > ARCHIVE_SIZE_BYTES:
                    raise ValueError("HuDiff model archive exceeds its expected size")
                digest.update(chunk)
                output.write(chunk)
        if size != ARCHIVE_SIZE_BYTES or digest.hexdigest() != ARCHIVE_SHA256:
            raise ValueError("HuDiff model archive failed SHA-256 validation")

        staging.mkdir()
        seen: set[str] = set()
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle:
                relative = _checkpoint_member(member.name)
                if relative is None:
                    continue
                if not member.isfile() or member.issym() or member.islnk():
                    raise ValueError(
                        f"Unsupported HuDiff checkpoint member: {member.name}"
                    )
                key = relative.as_posix()
                if key in seen:
                    raise ValueError(f"Duplicate HuDiff checkpoint member: {key}")
                seen.add(key)
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError(f"Unreadable HuDiff checkpoint member: {key}")
                target = staging.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
        if seen != {item.path for item in CHECKPOINTS}:
            raise ValueError("HuDiff model archive lacks required checkpoints")
        for item in CHECKPOINTS:
            path = staging.joinpath(*PurePosixPath(item.path).parts)
            if path.stat().st_size != item.size_bytes or _digest(path) != item.sha256:
                raise ValueError(f"HuDiff checkpoint failed validation: {item.path}")

        _publish_checkpoints(root, staging)
        return assert_hudiff_assets(root)
    finally:
        archive.unlink(missing_ok=True)
        shutil.rmtree(staging, ignore_errors=True)
