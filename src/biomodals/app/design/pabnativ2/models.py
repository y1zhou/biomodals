"""Pinned software and staged model identities for p-AbNatiV2."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import md5, sha256
from pathlib import Path
from tarfile import open as open_tar
from urllib.request import urlopen
from uuid import uuid4

import orjson

MODEL_MANIFEST = "manifest.json"
PAIRED_CHECKPOINT = "vpaired2_model.ckpt"
STRUCTURE_CHECKPOINT = "antibodybuilder3.ckpt"
STRUCTURE_ARCHIVE_MEMBER = "plddt-loss/best_second_stage.ckpt"


@dataclass(frozen=True, slots=True)
class PAbNatiV2Identity:
    """Result-affecting software and model identities."""

    abnativ_version: str = "2.0.8"
    abnativ_commit: str = "eb517f1f0b947084cb7e44a54ef34103e9692f5e"
    abodybuilder3_commit: str = "18e4058015a39c5405c08a0d5629cf302627b253"
    torch_version: str = "2.6.0"
    torch_cuda_version: str = "cu124"
    numpy_version: str = "1.26.4"
    pandas_version: str = "2.2.3"
    scipy_version: str = "1.14.1"
    biopython_version: str = "1.84"
    pytorch_lightning_version: str = "2.5.5"
    lightning_version: str = "2.5.5"
    matplotlib_version: str = "3.9.2"
    seaborn_version: str = "0.13.2"
    pdbfixer_version: str = "1.12.0"
    freesasa_version: str = "2.2.1"
    protein_topmodel_version: str = "1.0.1"
    anarci_version: str = "2020.04.23"
    hmmer_version: str = "3.4"
    rasa_structure_count: int = 4
    pssm_frequency_cutoff: float = 0.01
    nativeness_weight: float = 10.0
    pairing_weight: float = 1.0


@dataclass(frozen=True, slots=True)
class DownloadSpec:
    """One immutable published artifact downloaded during explicit staging."""

    filename: str
    url: str
    size_bytes: int
    md5_hex: str


IDENTITY = PAbNatiV2Identity()
ABNATIV_WHEEL_SHA256 = (
    "1345b2d27d5f5edd13d80e1944f11bee8c53aef4030f190d9fc4ee5de3b743b1"
)
ABNATIV_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/ff/41/"
    "6d9941a360eaba2bcc3e806308fd388f30cee5373c85b29322a0556b515c/"
    f"abnativ-{IDENTITY.abnativ_version}-py3-none-any.whl"
    f"#sha256={ABNATIV_WHEEL_SHA256}"
)
PAIRED_MODEL = DownloadSpec(
    filename=PAIRED_CHECKPOINT,
    url=("https://zenodo.org/record/17295347/files/vpaired2_model.ckpt?download=1"),
    size_bytes=1_223_267_160,
    md5_hex="d9cd90dc5b1a71873720c77c3bfe0906",
)
STRUCTURE_MODEL_ARCHIVE = DownloadSpec(
    filename="output.tar.gz",
    url="https://zenodo.org/records/11354577/files/output.tar.gz",
    size_bytes=440_793_236,
    md5_hex="2c1d734ed74013865bd95fc2f95cff1e",
)

RUNTIME_IDENTITY = "|".join((
    f"abnativ={IDENTITY.abnativ_version}@{IDENTITY.abnativ_commit}",
    f"abnativ-wheel-sha256={ABNATIV_WHEEL_SHA256}",
    f"abodybuilder3={IDENTITY.abodybuilder3_commit}",
    f"torch={IDENTITY.torch_version}",
    f"torch-cuda={IDENTITY.torch_cuda_version}",
    f"numpy={IDENTITY.numpy_version}",
    f"pandas={IDENTITY.pandas_version}",
    f"scipy={IDENTITY.scipy_version}",
    f"biopython={IDENTITY.biopython_version}",
    f"pdbfixer={IDENTITY.pdbfixer_version}",
    f"freesasa={IDENTITY.freesasa_version}",
    f"pytorch-lightning={IDENTITY.pytorch_lightning_version}",
    f"lightning={IDENTITY.lightning_version}",
    f"matplotlib={IDENTITY.matplotlib_version}",
    f"seaborn={IDENTITY.seaborn_version}",
    f"protein-topmodel={IDENTITY.protein_topmodel_version}",
    f"anarci={IDENTITY.anarci_version}",
    f"hmmer={IDENTITY.hmmer_version}",
))


def _digest_file(path: Path, algorithm: str) -> str:
    digest = md5(usedforsecurity=False) if algorithm == "md5" else sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _download(root: Path, spec: DownloadSpec) -> Path:
    target = root / spec.filename
    temporary = root / f".{spec.filename}.{uuid4().hex}.part"
    digest = md5(usedforsecurity=False)
    size = 0
    try:
        with (
            urlopen(  # noqa: S310 - both immutable specifications use HTTPS
                spec.url, timeout=180
            ) as response,
            temporary.open("wb") as out,
        ):
            while chunk := response.read(8 * 1024 * 1024):
                size += len(chunk)
                if size > spec.size_bytes:
                    raise ValueError(
                        f"Downloaded {spec.filename} exceeds expected size"
                    )
                digest.update(chunk)
                out.write(chunk)
        if size != spec.size_bytes or digest.hexdigest() != spec.md5_hex:
            raise ValueError(f"Downloaded {spec.filename} failed checksum validation")
        temporary.replace(target)
        return target
    finally:
        temporary.unlink(missing_ok=True)


def _extract_structure_checkpoint(root: Path, archive: Path) -> dict[str, str | int]:
    target = root / STRUCTURE_CHECKPOINT
    temporary = root / f".{STRUCTURE_CHECKPOINT}.{uuid4().hex}.part"
    digest = sha256()
    size = 0
    try:
        with open_tar(archive, "r:gz") as bundle:
            member = bundle.getmember(STRUCTURE_ARCHIVE_MEMBER)
            if not member.isfile() or member.size <= 0 or member.size > 2 * 1024**3:
                raise ValueError("ABodyBuilder3 checkpoint archive member is invalid")
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError(
                    "ABodyBuilder3 checkpoint archive member is unreadable"
                )
            with source, temporary.open("wb") as out:
                while chunk := source.read(8 * 1024 * 1024):
                    size += len(chunk)
                    if size > member.size:
                        raise ValueError(
                            "ABodyBuilder3 checkpoint exceeds declared size"
                        )
                    digest.update(chunk)
                    out.write(chunk)
            if size != member.size:
                raise ValueError("ABodyBuilder3 checkpoint extraction is incomplete")
        temporary.replace(target)
        return {"size_bytes": size, "sha256": digest.hexdigest()}
    finally:
        temporary.unlink(missing_ok=True)


def _load_manifest(root: Path) -> dict[str, object] | None:
    path = root / MODEL_MANIFEST
    try:
        value = orjson.loads(path.read_bytes())
    except (OSError, orjson.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def assert_pabnativ2_paired_checkpoint(root: Path) -> None:
    """Validate the sequence scorer's only checkpoint, without structure assets."""
    paired = root / PAIRED_CHECKPOINT
    if (
        not paired.is_file()
        or paired.stat().st_size != PAIRED_MODEL.size_bytes
        or _digest_file(paired, "md5") != PAIRED_MODEL.md5_hex
    ):
        raise RuntimeError("p-AbNatiV2 paired checkpoint is missing or corrupt")


def assert_pabnativ2_assets(root: Path) -> dict[str, object]:
    """Validate the complete staged publication and return its manifest."""
    manifest = _load_manifest(root)
    if manifest is None or manifest.get("schema_version") != 1:
        raise RuntimeError("p-AbNatiV2 model manifest is missing or unsupported")
    if manifest.get("abnativ_commit") != IDENTITY.abnativ_commit:
        raise RuntimeError("p-AbNatiV2 model manifest has the wrong source identity")
    if manifest.get("paired_model_md5") != PAIRED_MODEL.md5_hex:
        raise RuntimeError("p-AbNatiV2 paired model identity is invalid")
    if manifest.get("structure_archive_md5") != STRUCTURE_MODEL_ARCHIVE.md5_hex:
        raise RuntimeError("p-AbNatiV2 structure model identity is invalid")

    assert_pabnativ2_paired_checkpoint(root)

    structure = manifest.get("structure_checkpoint")
    if not isinstance(structure, dict):
        raise RuntimeError("p-AbNatiV2 structure checkpoint manifest is invalid")
    structure_path = root / STRUCTURE_CHECKPOINT
    if (
        not structure_path.is_file()
        or structure_path.stat().st_size != structure.get("size_bytes")
        or _digest_file(structure_path, "sha256") != structure.get("sha256")
    ):
        raise RuntimeError("p-AbNatiV2 structure checkpoint is missing or corrupt")
    return manifest


def stage_pabnativ2_assets(root: Path) -> dict[str, object]:
    """Download, verify, and atomically publish the two inference checkpoints."""
    root.mkdir(parents=True, exist_ok=True)
    try:
        return assert_pabnativ2_assets(root)
    except RuntimeError:
        pass

    paired = _download(root, PAIRED_MODEL)
    archive = _download(root, STRUCTURE_MODEL_ARCHIVE)
    structure = _extract_structure_checkpoint(root, archive)
    archive.unlink()
    manifest: dict[str, object] = {
        "schema_version": 1,
        "abnativ_version": IDENTITY.abnativ_version,
        "abnativ_commit": IDENTITY.abnativ_commit,
        "abodybuilder3_commit": IDENTITY.abodybuilder3_commit,
        "paired_model_md5": PAIRED_MODEL.md5_hex,
        "paired_model_size_bytes": paired.stat().st_size,
        "structure_archive_md5": STRUCTURE_MODEL_ARCHIVE.md5_hex,
        "structure_archive_size_bytes": STRUCTURE_MODEL_ARCHIVE.size_bytes,
        "structure_archive_member": STRUCTURE_ARCHIVE_MEMBER,
        "structure_checkpoint": structure,
    }
    temporary = root / f".{MODEL_MANIFEST}.{uuid4().hex}.part"
    temporary.write_bytes(orjson.dumps(manifest, option=orjson.OPT_SORT_KEYS))
    temporary.replace(root / MODEL_MANIFEST)
    return assert_pabnativ2_assets(root)
