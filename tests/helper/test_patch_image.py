"""Tests for helper image patching."""

# ruff: noqa: D101,D102,D103,D107

from importlib import metadata
from typing import Any, cast

from biomodals import helper as helper_module


class FakeImage:
    def __init__(self) -> None:
        self.apt_packages = ()
        self.installed_deps = []
        self.mods = ()
        self.copy = None

    def apt_install(self, *packages):
        self.apt_packages = packages
        return self

    def uv_pip_install(self, deps):
        self.installed_deps = deps
        return self

    def add_local_python_source(self, *mods, copy=False):
        self.mods = mods
        self.copy = copy
        return self


def test_patch_image_can_ignore_dependency_versions(monkeypatch) -> None:
    deps = [
        "modal>=1.5.1",
        "polars[calamine,numpy]>=1.39.3",
        "backports-strenum>=1.3.1,<2.0 ; python_full_version < '3.11'",
        "custom @ https://example.invalid/custom.whl",
    ]
    monkeypatch.setattr(metadata, "requires", lambda package: deps)

    image = FakeImage()
    helper_module.patch_image_for_helper(
        cast(Any, image),
        ignore_dep_versions=True,
    )

    assert image.installed_deps == [
        "modal",
        "polars[calamine,numpy]",
        "backports-strenum ; python_full_version < '3.11'",
        "custom @ https://example.invalid/custom.whl",
    ]
    assert "biomodals.execution" in image.mods


def test_patch_image_includes_execution_with_workflow_modules(monkeypatch) -> None:
    monkeypatch.setattr(metadata, "requires", lambda package: [])
    image = FakeImage()

    helper_module.patch_image_for_helper(
        cast(Any, image),
        include_workflow_modules=True,
    )

    assert image.mods == (
        "biomodals.helper",
        "biomodals.app.config",
        "biomodals.schema",
        "biomodals.execution",
        "biomodals.workflow",
    )
