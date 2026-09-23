"""Guarded plotting compatibility fix; no scoring or search changes."""

from importlib import import_module
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path


def apply_colormap_patch(root: Path | None = None) -> None:
    """Separate a duplicate color stop rejected by current matplotlib releases."""
    if root is None:
        spec = find_spec("abnativ")
        if spec is None or spec.origin is None:
            raise RuntimeError("Pinned AbNatiV package is missing")
        root = Path(spec.origin).parent
    path = root / "model/alignment/plotter.py"
    old, new = '(0.25, "DimGray")', '(0.250001, "DimGray")'
    text = path.read_text()
    if new in text and old not in text:
        return
    if text.count(old) != 1 or new in text:
        raise RuntimeError("AbNatiV colormap compatibility patch precondition failed")
    path.write_text(text.replace(old, new))


def verify_runtime() -> None:
    """Fail image construction on dependency drift or incomplete native imports."""
    expected = {
        "abnativ": "2.0.9",
        "nbforge": "0.1.1",
        "torch": "2.6.0+cu124",
        "numpy": "2.2.6",
        "pandas": "2.2.3",
        "scipy": "1.15.3",
        "biopython": "1.86",
        "lightning": "2.5.6",
        "pytorch-lightning": "2.5.6",
        "matplotlib": "3.10.8",
        "seaborn": "0.13.2",
        "protein-topmodel": "1.0.1",
    }
    for package, pinned in expected.items():
        if version(package) != pinned:
            raise RuntimeError(f"VHH runtime dependency differs: {package} != {pinned}")
    for name in (
        "abnativ.humanisation.vhh_humanisation_functions",
        "NbForge.cli",
        "NbForge.lightning_module",
        "pdbfixer",
        "openmm",
        "freesasa",
    ):
        import_module(name)
