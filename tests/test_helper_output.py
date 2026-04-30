# ruff: noqa: D100, D101, D102

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from biomodals.helper.output import (
    build_local_output_path,
    resolve_local_output_dir,
    write_local_tarball,
)


class TestHelperOutput(unittest.TestCase):
    def test_resolve_local_output_dir_uses_explicit_directory(self) -> None:
        with TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "nested"

            resolved = resolve_local_output_dir(str(out_dir))

            self.assertEqual(resolved, out_dir.resolve())
            self.assertFalse(resolved.exists())

    def test_resolve_local_output_dir_defaults_to_cwd(self) -> None:
        with TemporaryDirectory() as tmpdir:
            original_cwd = Path.cwd()
            try:
                os.chdir(tmpdir)

                resolved = resolve_local_output_dir(None)

                self.assertEqual(resolved, Path(tmpdir).resolve())
            finally:
                os.chdir(original_cwd)

    def test_write_local_tarball_raises_when_file_exists_by_default(self) -> None:
        with TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "run.tar.zst"
            out_file.write_bytes(b"existing")

            with self.assertRaisesRegex(
                FileExistsError, f"Output file already exists: {out_file}"
            ):
                write_local_tarball(out_file, b"new")

            self.assertEqual(out_file.read_bytes(), b"existing")

    def test_write_local_tarball_writes_bytes_and_returns_path(self) -> None:
        with TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "nested" / "run.tar.zst"

            written = write_local_tarball(out_file, b"tarball bytes")

            self.assertEqual(written, out_file)
            self.assertEqual(out_file.read_bytes(), b"tarball bytes")

    def test_build_local_output_path_supports_suffix_before_extension(self) -> None:
        with TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)

            out_file = build_local_output_path(
                out_dir,
                run_name="sample",
                suffix="_Protenix",
                extension=".tar.zst",
            )

            self.assertEqual(out_file, out_dir / "sample_Protenix.tar.zst")


if __name__ == "__main__":
    unittest.main()
