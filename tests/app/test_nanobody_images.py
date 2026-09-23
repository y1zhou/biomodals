"""Local Modal build-graph and compatibility checks, without remote images."""

# ruff: noqa: D103

import ast
import inspect
from pathlib import Path

import pytest
from modal._image import _Image

from biomodals.app.design.abnativ2_vhh import app as ab_app
from biomodals.app.design.abnativ2_vhh import patches
from biomodals.app.design.hudiff_nb import app as hu_app


def _image(image):
    return next(
        value
        for key, value in image.__dict__.items()
        if key.startswith("_sync_original")
    )


@pytest.mark.parametrize(
    "image", [ab_app.runtime_image, hu_app.runtime_image, hu_app.staging_image]
)
def test_no_build_step_follows_deferred_source_mounts(image):
    seen = set()

    def visit(layer):
        if id(layer) in seen:
            return
        seen.add(id(layer))
        for base in layer.deps():
            if isinstance(base, _Image):
                visit(base)
                if layer._rep != "Image(local files)":
                    assert base._rep != "Image(local files)"

    visit(_image(image))


def test_image_source_closures_and_interpreter_versions():
    for app, package, minimum in (
        (ab_app, "abnativ2_vhh", (3, 12)),
        (hu_app, "hudiff_nb", (3, 10)),
    ):
        assert {f"biomodals.app.design.{package}", "biomodals.app.design.vhh"} <= set(
            _image(app.runtime_image)._added_python_source_set
        )
        root = Path(app.__file__).parent
        for path in (*root.glob("*.py"), root.parent / "vhh.py"):
            ast.parse(path.read_text(), feature_version=minimum)
    assert {
        "biomodals.app.design.hudiff_nb.worker",
        "biomodals.app.design.hudiff_nb.patches",
        "biomodals.app.design.vhh",
        "biomodals.app.design.hudiff_ab.models",
    } <= set(_image(hu_app.staging_image)._added_python_source_set)
    assert set(ab_app.app.registered_functions) == {
        "abnativ2_vhh_humanize",
        "abnativ2_vhh_score",
        "stage_abnativ2_vhh_models",
    }
    assert set(hu_app.app.registered_functions) == {
        "hudiff_nb_humanize",
        "stage_hudiff_nb_models",
    }


def test_scipy_pin_uses_the_available_python_wheel():
    # The exact 1.15.3 release exists on PyPI, not in conda-forge. Inspect
    # actual Modal-generated installation commands without building an image.
    commands, seen = [], set()

    def visit(layer):
        if id(layer) in seen:
            return
        seen.add(id(layer))
        if "uv_pip_install" in layer._rep or "micromamba_install" in layer._rep:
            build = inspect.getclosurevars(layer._load).nonlocals["dockerfile_function"]
            commands.extend(build("2025.06").commands)
        for base in layer.deps():
            if isinstance(base, _Image):
                visit(base)

    visit(_image(ab_app.runtime_image))
    scipy_commands = [command for command in commands if "scipy==1.15.3" in command]
    assert len(scipy_commands) == 1
    assert "uv pip install" in scipy_commands[0]


def test_colormap_patch_guards_exact_source_and_is_idempotent(tmp_path):
    target = tmp_path / "model/alignment/plotter.py"
    target.parent.mkdir(parents=True)
    # Exact color stops in the pinned AbNatiV 2.0.9 plotter.py:867-876.
    target.write_text(
        'stops = [(0, "magenta"), (0.25, "purple"), '
        '(0.25, "DimGray"), (0.95, "DimGray"), (1, "ForestGreen")]'
    )
    patches.apply_colormap_patch(tmp_path)
    once = target.read_bytes()
    patches.apply_colormap_patch(tmp_path)
    assert target.read_bytes() == once
    tree = ast.parse(once)
    stops = ast.literal_eval(tree.body[0].value)
    assert [position for position, _ in stops] == [0, 0.25, 0.250001, 0.95, 1]
    target.write_text("changed upstream source")
    with pytest.raises(RuntimeError, match="precondition"):
        patches.apply_colormap_patch(tmp_path)
