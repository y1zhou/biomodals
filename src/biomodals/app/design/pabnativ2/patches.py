"""Guarded runtime compatibility patches for pinned upstream sources."""

from __future__ import annotations

import hashlib
from importlib.util import find_spec
from pathlib import Path
from urllib.request import urlopen

ABNATIV_COMMIT = "eb517f1f0b947084cb7e44a54ef34103e9692f5e"
PSSM_SHA256 = {
    "VH2_log2_pssm.npy": (
        "e72e2a61cdf6d16c16f1153c1cbc2a02bbb618c4977382595a01da6063c0f6cd"
    ),
    "VH2_pssm.npy": (
        "6ea288d4a8fea68ee2acf36e257b3a2ecf02f809bdd8145f6381c7b27957b92e"
    ),
    "VKappa2_log2_pssm.npy": (
        "bdcf93bfce0e60728b13ff16ad41d088564faf269017035be5aedf00358e7116"
    ),
    "VKappa2_pssm.npy": (
        "52a4f1a99abed826570f5befdceb783673db399035e797ed7d74d0caee6695f5"
    ),
    "VLambda2_log2_pssm.npy": (
        "b0ec30222affa6492a5cbc65a294f535772e30424ca48331c36f5c8b253ffc26"
    ),
    "VLambda2_pssm.npy": (
        "bae477cd81e038ab3045c04083902d818ecd01eb38f9f96087b5466cef7c8b5a"
    ),
}
_PSSM_SOURCE_ROOT = (
    "https://gitlab.doc.ic.ac.uk/sormanni-lab/abnativ/-/raw/"
    f"{ABNATIV_COMMIT}/abnativ/humanisation/pssms"
)


def _package_root(package: str) -> Path:
    spec = find_spec(package)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"Unable to locate installed package {package!r}")
    return Path(spec.origin).parent


def _replace_once(path: Path, old: str, new: str) -> None:
    content = path.read_text(encoding="utf-8")
    if new in content and old not in content:
        return
    if content.count(old) != 1:
        raise RuntimeError(f"Compatibility patch precondition failed for {path}")
    path.write_text(content.replace(old, new), encoding="utf-8")


def apply_upstream_compatibility_patches() -> None:
    """Apply the two source-identity-preserving Python 3.12 compatibility fixes."""
    _replace_once(
        _package_root("abnativ") / "model/alignment/plotter.py",
        '(0.25, "DimGray")',
        '(0.250001, "DimGray")',
    )
    _replace_once(
        _package_root("abodybuilder3") / "openfold/utils/tensor_utils.py",
        "return data[ranges]",
        "return data[tuple(ranges)]",
    )


def install_missing_abnativ_pssms() -> None:
    """Restore PSSMs omitted from the AbNatiV 2.0.8 distribution wheel."""
    target_root = _package_root("abnativ") / "humanisation/pssms"
    target_root.mkdir(parents=True, exist_ok=True)
    for filename, expected_sha256 in PSSM_SHA256.items():
        target = target_root / filename
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == (
            expected_sha256
        ):
            continue
        temporary = target.with_suffix(f"{target.suffix}.part")
        with urlopen(  # noqa: S310 - fixed HTTPS origin and commit
            f"{_PSSM_SOURCE_ROOT}/{filename}", timeout=60
        ) as response:
            content = response.read()
        actual_sha256 = hashlib.sha256(content).hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Checksum mismatch for {filename}: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )
        temporary.write_bytes(content)
        temporary.replace(target)
