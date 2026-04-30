"""Helper utility scripts."""

from modal import Image


def patch_image_for_helper(image: Image, copy_patch_files: bool = False) -> Image:
    """Patch a Modal Image to include helper dependencies.

    Args:
        image: The Modal Image to patch.
        copy_patch_files: Whether to copy patch files into the image. By default,
            the files are added to containers on startup and are not built into
            the actual Image, which speeds up deployment.
            Set to `True` to copy the files into an Image layer at build time instead.
            This can slow down iteration since it requires a rebuild of the Image
            and any subsequent build steps whenever the included files change,
            but it is required if you want to run additional build steps after this one.
    """
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
        .add_local_python_source(
            "biomodals.helper",
            "biomodals.app.constant",
            "biomodals.app.config",
            copy=copy_patch_files,
        )
    )


def hash_string(s: str) -> str:
    """Hash a string using a simple algorithm."""
    import hashlib

    return hashlib.sha256(s.encode()).hexdigest()
