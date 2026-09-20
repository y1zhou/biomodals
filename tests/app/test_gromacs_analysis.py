"""Streaming statistics against pinned Biotite batch analysis; no remote calls."""

# ruff: noqa: D103

from importlib.metadata import version

import biotite.structure as struc
import biotite.structure.io as strucio
import matplotlib
import numpy as np
import orjson
import polars as pl
import pytest
from biotite.structure.io.xtc import XTCFile
from numpy.testing import assert_allclose

from biomodals.app.bioinfo.gromacs import analysis
from biomodals.app.bioinfo.gromacs.execution import ANALYSIS_PACKAGES

matplotlib.use("Agg")


def _trajectory(frames=37):
    template = struc.AtomArray(17)
    template.atom_name = np.array(["N", "CA", "C", "O"] * 4 + ["O"])
    template.element = np.array(["N", "C", "C", "O"] * 4 + ["O"])
    template.res_name = np.array(["ALA"] * 16 + ["HOH"])
    template.res_id = np.repeat(np.arange(1, 6), [4, 4, 4, 4, 1])
    template.chain_id[:] = "A"
    template.hetero[-1] = True
    rng = np.random.default_rng(93)
    reference = rng.normal(size=(17, 3)).astype(np.float32) * 5
    coordinates = []
    for i in range(frames):
        # Internal motion plus large rigid displacement makes a chunk-local
        # reference incorrect. The water must never contribute to the statistics.
        deformed = reference + np.sin(i / 9) * rng.normal(size=(17, 3))
        coordinates.append(struc.rotate(deformed, [i / 20, i / 30, 0]) + [i, 0, i / 2])
    coordinates = np.array(coordinates, dtype=np.float32)
    template.coord = coordinates[0]
    return template, coordinates


def _files(tmp_path, frames=37):
    template, coordinates = _trajectory(frames)
    path = tmp_path / "production_parent_nopbc.xtc"
    pdb = tmp_path / "production_parent_nopbc_centered.pdb"
    strucio.save_structure(pdb, template)
    trajectory = XTCFile()
    trajectory.set_coord(coordinates)
    trajectory.set_time(np.arange(frames, dtype=np.float32) * 10)
    trajectory.set_box(
        np.repeat(np.eye(3, dtype=np.float32)[None] * 100, frames, axis=0)
    )
    trajectory.write(path)
    # External oracle: the original pinned Biotite full-stack calculation,
    # including XTC quantization, PDB annotations, alignment and atomic masses.
    loaded = strucio.load_structure(pdb)
    mask = struc.filter_amino_acids(loaded)
    native = XTCFile.read(path, atom_i=np.flatnonzero(mask))
    stack = native.get_structure(loaded[mask])
    aligned, _ = struc.superimpose(stack[0], stack)
    ca = aligned[:, aligned.atom_name == "CA"]
    expected = {
        "rmsd": struc.rmsd(aligned[0], aligned),
        "rg": struc.gyration_radius(aligned),
        "rmsf": struc.rmsf(struc.average(ca), ca),
    }
    return path, pdb, expected, aligned


def test_analysis_environment_matches_the_runtime_pins():
    for package in ANALYSIS_PACKAGES:
        name, expected = package.split("==")
        assert version(name) == expected


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 128])
def test_statistics_match_batch_across_uneven_chunks(chunk_size):
    template, coordinates = _trajectory()
    template = template[:-1]
    coordinates = coordinates[:, :-1]
    batch, _ = struc.superimpose(
        coordinates[0], struc.from_template(template, coordinates)
    )
    stats = analysis.TrajectoryStatistics(template)
    rmsd, rg = [], []
    for start in range(0, len(coordinates), chunk_size):
        d, g = stats.update(coordinates[start : start + chunk_size])
        rmsd.append(d)
        rg.append(g)
    assert_allclose(np.concatenate(rmsd), struc.rmsd(batch[0], batch), atol=2e-5)
    assert_allclose(np.concatenate(rg), struc.gyration_radius(batch), atol=2e-5)
    ca = batch[:, batch.atom_name == "CA"]
    assert_allclose(stats.rmsf(), struc.rmsf(struc.average(ca), ca), atol=2e-5)
    assert_allclose(stats.last_frame.coord, batch[-1].coord, atol=2e-5)
    assert stats.count == len(coordinates)


@pytest.mark.parametrize("chunk_size", [1, 7, 128])
def test_real_xtc_streaming_preserves_all_rows_units_titles_and_last_frame(
    tmp_path, monkeypatch, chunk_size
):
    from matplotlib.axes import Axes

    path, pdb, expected, aligned = _files(tmp_path)
    titles = []
    original_title = Axes.set_title

    def title(ax, value, **kwargs):
        titles.append(value)
        return original_title(ax, value, **kwargs)

    monkeypatch.setattr(Axes, "set_title", title)
    monkeypatch.setattr(
        XTCFile,
        "read",
        lambda *args, **kwargs: pytest.fail(
            "Must stream, not load the entire trajectory"
        ),
    )
    analysis.analyze_trajectory(
        path,
        pdb,
        prefix="production_parent",
        run_name="child",
        make_figures=True,
        chunk_frames=chunk_size,
    )
    assert titles == ["child"] * 3
    for metric, values in expected.items():
        csv = tmp_path / f"{metric}_production_parent.csv"
        table = pl.read_csv(csv)
        assert table.height == len(values)
        assert_allclose(table[metric].to_numpy(), values, atol=2e-5)
        assert all(
            len(value.rsplit(".", 1)[-1]) == 5
            for value in csv.read_text().splitlines()[1].split(",")
        )
        assert (
            (tmp_path / f"{metric}_production_parent.png")
            .read_bytes()
            .startswith(b"\x89PNG")
        )
        if metric != "rmsf":
            assert_allclose(table["time_ns"].to_numpy(), np.arange(37) / 100, atol=1e-6)
    final = strucio.load_structure(tmp_path / "production_parent_last_frame.pdb")
    assert_allclose(final.coord, aligned[-1].coord, atol=0.001)


def test_cache_repair_and_interruption_do_not_reuse_partial_csvs(tmp_path, monkeypatch):
    path, pdb, expected, _ = _files(tmp_path)
    kwargs = dict(
        prefix="production_parent", run_name="child", make_figures=False, chunk_frames=7
    )
    analysis.analyze_trajectory(path, pdb, **kwargs)
    csv = tmp_path / "rmsd_production_parent.csv"
    complete = csv.read_bytes()
    marker = tmp_path / ".biomodals/gromacs/analysis-production_parent.json"
    iterator = XTCFile.read_iter
    monkeypatch.setattr(
        XTCFile,
        "read_iter",
        lambda *args, **kwargs: pytest.fail(
            "Validated cache must not decode XTC again"
        ),
    )
    analysis.analyze_trajectory(path, pdb, **kwargs)
    csv.write_bytes(complete.replace(b"0.00000", b"9.99999", 1))
    closed = []

    def interrupted(*args, **kwargs):
        with analysis.closing(iterator(*args, **kwargs)) as chunks:
            try:
                yield next(chunks)
                raise KeyboardInterrupt("Worker interrupted after one chunk")
            finally:
                closed.append(True)

    monkeypatch.setattr(XTCFile, "read_iter", interrupted)
    with pytest.raises(KeyboardInterrupt):
        analysis.analyze_trajectory(path, pdb, **kwargs)
    assert closed == [True]
    assert not marker.exists()
    assert pl.read_csv(csv.with_suffix(".csv.part")).height == 7
    monkeypatch.setattr(XTCFile, "read_iter", iterator)
    analysis.analyze_trajectory(path, pdb, **kwargs)
    assert csv.read_bytes() == complete
    assert pl.read_csv(csv).height == len(expected["rmsd"])
    assert not list(tmp_path.glob("*.part"))
    evidence = orjson.loads(marker.read_bytes())
    assert evidence["identity"]["analysis_policy"] == "streaming-v1"
    # Input identity changes must invalidate otherwise valid outputs.
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    monkeypatch.setattr(
        XTCFile,
        "read_iter",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("changed input")),
    )
    with pytest.raises(ValueError, match="changed input"):
        analysis.analyze_trajectory(path, pdb, **kwargs)


@pytest.mark.parametrize("count", [1, 5000, 5001, 9999, 20001])
def test_only_plotting_is_bounded_full_csv_is_unchanged(tmp_path, count):
    path = tmp_path / "data.csv"
    pl.DataFrame({
        "time_ns": np.arange(count),
        "rmsd": np.arange(count) / 10,
    }).write_csv(path)
    sampled = analysis.plot_samples(path, count=count)
    assert sampled.height <= 5000
    assert sampled["time_ns"][0] == 0
    assert sampled["time_ns"][-1] == count - 1
    assert pl.read_csv(path).height == count


def test_empty_trajectory_is_not_published(tmp_path, monkeypatch):
    path, pdb, _, _ = _files(tmp_path)
    monkeypatch.setattr(
        XTCFile, "read_iter", lambda *args, **kwargs: (item for item in ())
    )
    with pytest.raises(ValueError, match="no frames"):
        analysis.analyze_trajectory(
            path, pdb, prefix="production_parent", run_name="child", make_figures=False
        )
    assert not (
        tmp_path / ".biomodals/gromacs/analysis-production_parent.json"
    ).exists()


def test_large_trajectory_plots_are_labelled_overviews_not_sampled_statistics(
    tmp_path, monkeypatch
):
    from matplotlib.axes import Axes

    path, pdb, expected, _ = _files(tmp_path, frames=5001)
    plotted = []
    labels = []
    original_plot = Axes.plot
    original_label = Axes.set_xlabel

    def plot(ax, x, y, **kwargs):
        plotted.append(len(x))
        return original_plot(ax, x, y, **kwargs)

    def label(ax, value, **kwargs):
        labels.append(value)
        return original_label(ax, value, **kwargs)

    monkeypatch.setattr(Axes, "plot", plot)
    monkeypatch.setattr(Axes, "set_xlabel", label)
    analysis.analyze_trajectory(
        path, pdb, prefix="production_parent", run_name="child", make_figures=True
    )
    assert plotted[:2] == [2501, 2501]
    assert plotted[2] == 4
    assert labels == ["Time (ns)\nSampled overview; full resolution in CSV"] * 2 + [
        "Residue Index"
    ]
    for metric in ("rmsd", "rg"):
        table = pl.read_csv(tmp_path / f"{metric}_production_parent.csv")
        assert table.height == 5001
        assert_allclose(table[metric].to_numpy(), expected[metric], atol=0.001)
