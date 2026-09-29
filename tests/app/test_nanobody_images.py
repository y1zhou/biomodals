"""Guarded upstream colormap patch regression."""

# ruff: noqa: D103

import ast

import pytest

from biomodals.app.design.abnativ2_vhh import patches


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
