"""Helper utility scripts."""

from modal import Image


def patch_image_for_helper(image: Image) -> Image:
    """Patch a Modal Image to include helper dependencies."""
    # This is a bit hacky, but because Modal's .add_local_python_source()
    # does not install the package, the metadata.requires call would not work
    # in the runtime, so we make sure dependencies are installed here.
    from importlib import metadata

    try:
        helper_deps = metadata.requires("biomodals") or []
    except metadata.PackageNotFoundError:
        helper_deps = []

    return (
        image.apt_install("zstd", "fd-find")
        .uv_pip_install(helper_deps)
        .add_local_python_source("biomodals", copy=True)
    )


def hash_string(s: str) -> str:
    """Hash a string using a simple algorithm."""
    import hashlib

    return hashlib.sha256(s.encode()).hexdigest()
