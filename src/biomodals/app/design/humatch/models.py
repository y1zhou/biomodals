"""Immutable runtime and inference-asset identities for Humatch."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import md5  # noqa: S324 - Zenodo publishes MD5 asset checksums.
from importlib.util import find_spec
from pathlib import Path
from urllib.request import urlopen

ZENODO_RECORD_ID = "13764771"


@dataclass(frozen=True, slots=True)
class HumatchIdentity:
    """Pinned software versions that define Humatch's scientific behavior."""

    package_version: str = "1.0.1"
    source_commit: str = "06205ad50f64b84468fea14eea573a3fce699550"
    asset_record_id: str = ZENODO_RECORD_ID
    asset_doi: str = "10.5281/zenodo.13764771"
    tensorflow_version: str = "2.17.0"
    keras_version: str = "3.4.1"
    numpy_version: str = "1.26.4"
    pandas_version: str = "2.2.3"
    anarci_version: str = "2020.04.23"
    hmmer_version: str = "3.4"
    biopython_version: str = "1.84"
    pyyaml_version: str = "6.0.2"
    matplotlib_version: str = "3.9.2"
    seaborn_version: str = "0.13.2"
    requests_version: str = "2.32.3"


@dataclass(frozen=True, slots=True)
class AssetSpec:
    """One immutable file from the versioned Zenodo inference record."""

    filename: str
    subdirectory: str
    size_bytes: int
    md5_hex: str

    @property
    def url(self) -> str:
        """Return this file's version-record download URL."""
        return (
            f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/"
            f"{self.filename}/content"
        )


IDENTITY = HumatchIdentity()

_WEIGHTS = (
    AssetSpec(
        "heavy.weights.h5",
        "trained_models",
        28_760_968,
        "9a881232a25bf8b6f8399df4ec19acd5",
    ),
    AssetSpec(
        "light.weights.h5",
        "trained_models",
        28_796_968,
        "f9ef8c5e75ee157c7d2b0efadb5b2a32",
    ),
    AssetSpec(
        "paired.weights.h5",
        "trained_models",
        58_979_368,
        "2941dd7079ab3437110cdfdb76b3fb01",
    ),
)
_LOOKUP_MD5 = {
    "hv1": "9711fdc8f530d2271dec68d1c4be9e23",
    "hv2": "c4a9004924618c9e180924054057d8b0",
    "hv3": "700ae1924edfd91315dd4f0260c55e0e",
    "hv4": "7f30ed73ce572e5066efec787b17cf8f",
    "hv5": "cae05e7b784a4f97649a0ee12dd9c06f",
    "hv6": "e547a389b196b34a1b47a32256ff5a3c",
    "hv7": "8120c43c8e434be94fa96ea4557d2ec6",
    "lv1": "ad9a0984e14f6fe1f99e5ff8c0ea5ad3",
    "lv2": "0626db755d2da579ba2318843bceb936",
    "lv3": "ea32c7580f2d37af8afc89384f70f5cb",
    "lv4": "d2fa2c4bff009eb74c40e834ce731388",
    "lv5": "57b6a4ee43be602be511bfc5e5f3fb32",
    "lv6": "77fbc95dfec6cdd54f27a927196ce321",
    "lv7": "694fef081c295c146ebd172fddbde4f4",
    "lv8": "f0e0164f59edb1015a74597c00dcd92b",
    "lv9": "3e71dc72b385881cbe13a8a713ad4738",
    "lv10": "f3ce2a2e8ea25ed3f966422369032c53",
    "kv1": "392ea94e341e4be485a63d7fb75cc989",
    "kv2": "28a432b82b6d2992fa7cc1e0087f56b0",
    "kv3": "cb2f2ee6e8a635d8a4ba5547840b71b9",
    "kv4": "65aac2dd8ceccf7f5a5690c52b679109",
    "kv5": "ff0090f14602d1cfb834775e955098d4",
    "kv6": "2950bd2e4d832ebb08bfd6c5ba01eef8",
    "kv7": "fb7e245eb74db6804cec71d7ea5748db",
}
ASSETS = _WEIGHTS + tuple(
    AssetSpec(f"{family}.npy", "germline_likeness_lookup_arrays", 32_128, checksum)
    for family, checksum in _LOOKUP_MD5.items()
)

RUNTIME_IDENTITY = "|".join((
    f"humatch={IDENTITY.package_version}@{IDENTITY.source_commit}",
    f"assets={IDENTITY.asset_doi}",
    f"tensorflow={IDENTITY.tensorflow_version}",
    f"keras={IDENTITY.keras_version}",
    f"numpy={IDENTITY.numpy_version}",
    f"pandas={IDENTITY.pandas_version}",
    f"anarci={IDENTITY.anarci_version}",
    f"hmmer={IDENTITY.hmmer_version}",
    f"biopython={IDENTITY.biopython_version}",
))


def _humatch_package_root() -> Path:
    spec = find_spec("Humatch")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Pinned Humatch package is not installed")
    return Path(next(iter(spec.submodule_search_locations)))


def _verified(path: Path, asset: AssetSpec) -> bool:
    if not path.is_file() or path.stat().st_size != asset.size_bytes:
        return False
    digest = md5(path.read_bytes(), usedforsecurity=False).hexdigest()
    return digest == asset.md5_hex


def download_humatch_assets() -> None:
    """Download and verify only the immutable assets needed for inference."""
    package_root = _humatch_package_root()
    for asset in ASSETS:
        destination = package_root / asset.subdirectory / asset.filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if _verified(destination, asset):
            continue
        temporary = destination.with_suffix(destination.suffix + ".part")
        digest = md5(usedforsecurity=False)
        size = 0
        with (
            urlopen(asset.url, timeout=120) as response,  # noqa: S310
            temporary.open("wb") as output,
        ):
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > asset.size_bytes:
                    raise ValueError(f"Asset exceeds expected size: {asset.filename}")
                digest.update(chunk)
                output.write(chunk)
        if size != asset.size_bytes or digest.hexdigest() != asset.md5_hex:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Asset checksum mismatch: {asset.filename}")
        temporary.replace(destination)


def assert_humatch_assets() -> None:
    """Fail before inference if any baked asset is absent or corrupt."""
    package_root = _humatch_package_root()
    invalid = [
        asset.filename
        for asset in ASSETS
        if not _verified(package_root / asset.subdirectory / asset.filename, asset)
    ]
    if invalid:
        raise RuntimeError(
            f"Humatch inference assets are invalid: {', '.join(invalid)}"
        )
