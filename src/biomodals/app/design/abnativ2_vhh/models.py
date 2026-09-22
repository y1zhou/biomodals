"""Independent VHH runtime and weights; never upgrade the paired environment."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import orjson

from biomodals.helper.artifacts import file_size_sha256
from biomodals.helper.model_download import DownloadSpec, download_model_asset

SOURCE_COMMIT = "413ebd3995f9383bcb225638810d7fa0b5a3dbe4"
NBFORGE_COMMIT = "f5c90aa6a81968759890ae269d4ba137d0a6a61b"
NBFORGE_SIZE = 98_674_455
NBFORGE_SHA256 = "1762018a09b1e6f12ef68de66db67df5ceca269f3fa53bcbf0f27998d0ad3195"
MODEL_ROOT = Path("/biomodals-store/abnativ2-vhh")
MODELS = (
    DownloadSpec(
        "vh2_model.ckpt",
        "https://zenodo.org/record/17295347/files/vh2_model.ckpt?download=1",
        1_094_871_186,
        "9996070cb403da33539f5caa5f2ebb0d",
    ),
    DownloadSpec(
        "vhh2_model.ckpt",
        "https://zenodo.org/record/17295347/files/vhh2_model.ckpt?download=1",
        1_094_871_570,
        "f41823c3c987a0f29615a53e597ec0ea",
    ),
)
RUNTIME_IDENTITY = "|".join((
    f"abnativ=2.0.9@{SOURCE_COMMIT}",
    f"nbforge=0.1.1@{NBFORGE_COMMIT}",
    "python=3.12|torch=2.6.0+cu124|numpy=2.2.6|pandas=2.2.3|scipy=1.15.3",
    "biopython=1.86|lightning=2.5.6|pytorch-lightning=2.5.6",
    "anarci=2020.04.23|hmmer=3.4|freesasa=2.2.1|pdbfixer=1.12.0|openmm=8.3.1",
    "matplotlib=3.10.8|seaborn=0.13.2|protein-topmodel=1.0.1",
    "VH2+VHH2|enhanced|weights=2,1|forbidden=C,M|colormap-patch=1|wrapper=1",
))


def valid_model(path: Path, spec: DownloadSpec) -> bool:
    """Compare published bytes, not a cached filename or a provider marker."""
    if not path.is_file() or path.stat().st_size != spec.size_bytes:
        return False
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, lambda: hashlib.md5(usedforsecurity=False))
    return digest.hexdigest() == spec.md5_hex


def expected_manifest() -> dict[str, object]:
    """Bind exact native software and all three checkpoint identities."""
    return {
        "schema_version": 1,
        "source_commit": SOURCE_COMMIT,
        "nbforge_commit": NBFORGE_COMMIT,
        "models": [asdict(spec) for spec in MODELS],
        "nbforge": {"size_bytes": NBFORGE_SIZE, "sha256": NBFORGE_SHA256},
    }


def assert_assets(root: Path) -> dict[str, object]:
    """Fail inference fast rather than downloading implicitly on a GPU."""
    try:
        manifest = orjson.loads((root / "manifest.json").read_bytes())
        if manifest != expected_manifest():
            raise ValueError("Unexpected VHH model manifest")
        if any(not valid_model(root / spec.filename, spec) for spec in MODELS):
            raise ValueError("Missing or corrupt VH2/VHH2 checkpoint")
        if file_size_sha256(root / "nbforge.ckpt") != (NBFORGE_SIZE, NBFORGE_SHA256):
            raise ValueError("Missing or corrupt NbForge checkpoint")
    except (OSError, ValueError) as error:
        raise RuntimeError("AbNatiV2-VHH model preparation is incomplete") from error
    return manifest


def stage_assets(root: Path, bundled_nbforge: Path) -> dict[str, object]:
    """CPU-only explicit preparation; publish manifest after verified assets."""
    root.mkdir(parents=True, exist_ok=True)
    try:
        return assert_assets(root)
    except RuntimeError:
        pass
    for spec in MODELS:
        if not valid_model(root / spec.filename, spec):
            download_model_asset(root, spec)
    if file_size_sha256(bundled_nbforge) != (NBFORGE_SIZE, NBFORGE_SHA256):
        raise ValueError("Pinned NbForge package checkpoint changed")
    # Supply an explicit non-default checkpoint path to native NbForge. It then
    # loads the full verified weights without writing an inference cache at run time.
    temporary = root / f".nbforge-{uuid4().hex}.part"
    try:
        shutil.copyfile(bundled_nbforge, temporary)
        temporary.replace(root / "nbforge.ckpt")
    finally:
        temporary.unlink(missing_ok=True)
    manifest = expected_manifest()
    temporary = root / f".manifest-{uuid4().hex}.part"
    temporary.write_bytes(orjson.dumps(manifest, option=orjson.OPT_SORT_KEYS))
    temporary.replace(root / "manifest.json")
    return assert_assets(root)
