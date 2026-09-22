"""Size-bounded downloads for published model assets with upstream MD5 digests."""

from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from urllib.request import urlopen
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class DownloadSpec:
    """One immutable published artifact downloaded during explicit staging."""

    filename: str
    url: str
    size_bytes: int
    md5_hex: str


def download_model_asset(root: Path, spec: DownloadSpec) -> Path:
    """Publish only a complete size/checksum-verified download, without buffering."""
    target = root / spec.filename
    temporary = root / f".{spec.filename}.{uuid4().hex}.part"
    digest = md5(usedforsecurity=False)
    size = 0
    try:
        with (
            urlopen(spec.url, timeout=180) as response,  # noqa: S310 - app-owned HTTPS specifications
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
