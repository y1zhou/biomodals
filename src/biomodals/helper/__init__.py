"""Helper utility scripts."""

from collections.abc import Iterable

from modal import Image


def patch_image_for_helper(
    image: Image,
    *,
    copy_patch_files: bool = False,
    include_workflow_modules: bool = False,
    skip_deps: Iterable[str] | None = None,
) -> Image:
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
        include_workflow_modules: Whether to include workflow modules in the patch.
            By default, only helper dependencies are included.
        skip_deps: A list of package names to skip when installing
            `biomodals` dependencies. By default, all dependencies are included.
            This is to help with older project apps on Python <3.12.
    """
    # This is a bit hacky, but because Modal's .add_local_python_source()
    # does not install the package, the metadata.requires call would not work
    # in the runtime, so we make sure dependencies are installed here.
    from importlib import metadata

    try:
        helper_deps = metadata.requires("biomodals") or []
    except metadata.PackageNotFoundError:
        helper_deps = []

    mods = ["biomodals.helper", "biomodals.app.config", "biomodals.schema"]
    if include_workflow_modules:
        mods.append("biomodals.workflow")

    new_image = image.apt_install("zstd", "fd-find")
    if skip_deps is not None:
        import re

        skip_deps_set = set(skip_deps)
        package_name_pattern = re.compile(r"^[\w_\-.]+")
        helper_deps = [
            dep
            for dep in helper_deps
            if next(package_name_pattern.finditer(dep)).group(0) not in skip_deps_set
        ]
    if helper_deps:
        new_image = new_image.uv_pip_install(helper_deps)

    return new_image.add_local_python_source(*mods, copy=copy_patch_files)


def hash_string(s: str) -> str:
    """Hash a string using a simple algorithm."""
    import hashlib

    return hashlib.sha256(s.encode()).hexdigest()
