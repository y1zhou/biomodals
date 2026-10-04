"""Helper utility scripts."""

from collections.abc import Iterable

from modal import Image

from biomodals.helper.artifacts import sha256_bytes


def patch_image_for_helper(
    image: Image,
    *,
    copy_patch_files: bool = False,
    include_workflow_modules: bool = False,
    skip_deps: Iterable[str] | None = None,
    ignore_dep_versions: bool = False,
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
        include_workflow_modules: Whether to include workflow modules in addition
            to the shared helper and execution modules.
        skip_deps: A list of package names to skip when installing
            `biomodals` dependencies. By default, all dependencies are included.
            This is to help with older project apps on Python <3.12.
        ignore_dep_versions: Whether to install `biomodals` dependencies without
            their version specifiers. This is useful for older app images where
            the pinned/latest dependency versions require newer Python versions.
            The Modal SDK requirement is always retained from package metadata.
    """
    mods = [
        "biomodals.helper",
        "biomodals.app.config",
        "biomodals.schema",
        "biomodals.execution",
    ]
    if include_workflow_modules:
        # Workflow composition roots import app metadata and execution classes.
        # The shared coordinator unpickles those roots in its own image, not in
        # the included apps' images, so it needs their local source as well.
        mods.extend(("biomodals.workflow", "biomodals.app"))

    new_image = image.apt_install("zstd", "fd-find")
    helper_deps = helper_dependencies(
        skip_deps=skip_deps, ignore_dep_versions=ignore_dep_versions
    )
    if helper_deps:
        new_image = new_image.uv_pip_install(helper_deps)

    return new_image.add_local_python_source(*mods, copy=copy_patch_files)


def helper_dependencies(
    *, skip_deps: Iterable[str] | None = None, ignore_dep_versions: bool = False
) -> list[str]:
    """Read image requirements from project metadata, also used by runtime CI.

    Preserve environment markers and the Modal SDK version when relaxing
    requirements for older interpreters. No image or package install is performed.
    """
    # Resolve metadata locally: source injection does not install the package's
    # distribution metadata in remote images.
    from importlib import metadata

    try:
        helper_deps = metadata.requires("biomodals") or []
    except metadata.PackageNotFoundError:
        helper_deps = []

    if ignore_dep_versions:
        import re

        requirement_name_pattern = re.compile(r"^\s*([\w_\-.]+(?:\[[^\]]+\])?)")
        stripped_deps = []
        for dep in helper_deps:
            requirement, separator, marker = dep.partition(";")
            match = requirement_name_pattern.match(requirement)
            if (
                match is None
                or " @ " in requirement
                or match.group(1).split("[", 1)[0].lower() == "modal"
            ):
                stripped_deps.append(dep)
                continue
            stripped = match.group(1)
            if separator:
                stripped = f"{stripped} ; {marker.strip()}"
            stripped_deps.append(stripped)
        helper_deps = stripped_deps

    if skip_deps is not None:
        import re

        skip_deps_set = set(skip_deps)
        package_name_pattern = re.compile(r"^[\w_\-.]+")
        helper_deps = [
            dep
            for dep in helper_deps
            if next(package_name_pattern.finditer(dep)).group(0) not in skip_deps_set
        ]
    return helper_deps


def hash_string(s: str) -> str:
    """Hash a string using a simple algorithm."""
    return sha256_bytes(s.encode())
