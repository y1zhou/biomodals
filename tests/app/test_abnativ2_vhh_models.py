"""Offline asset lifecycle: never publish incomplete or changed model bytes."""

# ruff: noqa: D103

from hashlib import md5, sha256
from io import BytesIO
from pathlib import Path

import pytest

from biomodals.app.design.abnativ2_vhh import models
from biomodals.helper import model_download
from biomodals.helper.model_download import DownloadSpec, download_model_asset


def _spec(name: str, content: bytes) -> DownloadSpec:
    return DownloadSpec(
        name,
        f"https://example.org/{name}",
        len(content),
        md5(content, usedforsecurity=False).hexdigest(),
    )


@pytest.mark.parametrize("content", [b"short", b"corrupt", b"oversized bytes"])
def test_download_rejects_incomplete_or_changed_bytes_without_replacing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: bytes
) -> None:
    spec = _spec("model.ckpt", b"correct")
    target = tmp_path / spec.filename
    target.write_bytes(b"previous")
    monkeypatch.setattr(model_download, "urlopen", lambda *a, **kw: BytesIO(content))
    with pytest.raises(ValueError, match="expected size|checksum"):
        download_model_asset(tmp_path, spec)
    assert target.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [target]


def test_staging_reuses_verified_assets_and_repairs_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = {"vh2.ckpt": b"vh2", "vhh2.ckpt": b"vhh2"}
    specs = tuple(_spec(name, content) for name, content in assets.items())
    bundled = tmp_path / "bundled.ckpt"
    bundled.write_bytes(b"nbforge")
    monkeypatch.setattr(models, "MODELS", specs)
    monkeypatch.setattr(models, "NBFORGE_SIZE", 7)
    monkeypatch.setattr(models, "NBFORGE_SHA256", sha256(b"nbforge").hexdigest())
    downloads: list[str] = []

    def fetch(url: str, **kwargs: object) -> BytesIO:
        downloads.append(url.rsplit("/", 1)[1])
        return BytesIO(assets[downloads[-1]])

    monkeypatch.setattr(model_download, "urlopen", fetch)
    root = tmp_path / "staged"
    manifest = models.stage_assets(root, bundled)
    assert manifest == models.expected_manifest()
    assert downloads == ["vh2.ckpt", "vhh2.ckpt"]
    assert models.stage_assets(root, bundled) == manifest
    assert len(downloads) == 2
    (root / "vhh2.ckpt").write_bytes(b"xxxx")
    with pytest.raises(RuntimeError, match="preparation is incomplete"):
        models.assert_assets(root)
    assert models.stage_assets(root, bundled) == manifest
    assert downloads == ["vh2.ckpt", "vhh2.ckpt", "vhh2.ckpt"]
    (root / "nbforge.ckpt").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="preparation is incomplete"):
        models.assert_assets(root)
    assert models.stage_assets(root, bundled) == manifest
    assert (root / "nbforge.ckpt").read_bytes() == b"nbforge"


def test_bad_bundled_predictor_cannot_publish_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(models, "MODELS", ())
    bundled = tmp_path / "bundled.ckpt"
    bundled.write_bytes(b"wrong")
    root = tmp_path / "staged"
    with pytest.raises(ValueError, match="checkpoint changed"):
        models.stage_assets(root, bundled)
    with pytest.raises(RuntimeError, match="preparation is incomplete"):
        models.assert_assets(root)
