"""Tiny native PNGs in a synthetic GROMACS result for API/browser tests."""

from __future__ import annotations

import hashlib
import io
import struct
import zipfile
import zlib

import orjson


def _chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data))
    )


def _png(rgb: bytes) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(b"\0" + rgb))
        + _chunk(b"IEND", b"")
    )


PLOTS = {
    "rmsd": _png(b"\x11\x22\x33"),
    "rg": _png(b"\x44\x55\x66"),
    "rmsf": _png(b"\x77\x88\x99"),
}


def trajectory_archive() -> bytes:
    """Build the preview members without simulating molecular dynamics."""
    buffer = io.BytesIO()
    records = []
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("input.pdb", "ATOM\nEND\n")
        archive.writestr("outputs/trajectory_nopbc.xtc", b"trajectory")
        for metric, content in PLOTS.items():
            path = f"outputs/{metric}_production_fixture.png"
            role = "radius_of_gyration" if metric == "rg" else metric
            archive.writestr(path, content)
            records.append({
                "path": path,
                "role": f"production_{role}_plot",
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            })
        archive.writestr(
            "metadata/manifest.json",
            orjson.dumps({
                "archive_schema_version": 5,
                "run_name": "fixture",
                "files": records,
            }),
        )
    return buffer.getvalue()
