# ruff: noqa: D100, D101, D102

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from biomodals.app.bioinfo.rosetta_app import (
    _build_rosetta_command,
    _normalize_bytes_rosetta_job,
    _normalize_volume_rosetta_job,
)


class TestRosettaApp(unittest.TestCase):
    def test_build_rosetta_command_uses_optional_script_and_flags(self) -> None:
        command = _build_rosetta_command(
            binary="rosetta_scripts",
            pdb_path=Path("/work/input/model.pdb"),
            out_dir=Path("/work/outputs/1"),
            rosetta_script_path=Path("/work/input/protocol.xml"),
            flags_path=Path("/work/input/options.flags"),
        )

        self.assertEqual(
            command,
            [
                "rosetta_scripts",
                "-parser:protocol",
                "/work/input/protocol.xml",
                "@/work/input/options.flags",
                "-s",
                "/work/input/model.pdb",
                "-out:path:all",
                "/work/outputs/1",
            ],
        )

    def test_normalize_volume_job_uses_mounted_paths_and_task_output_dir(self) -> None:
        job = _normalize_volume_rosetta_job(
            {
                "index": 7,
                "binary": "relax",
                "pdb": "run/7/input.pdb",
                "rosetta_script": None,
                "flags_file": "run/_flags/abc.flags",
            },
            mount_dir=Path("/mnt/out"),
            workdir=Path("/mnt/out/run"),
        )

        self.assertEqual(job.index, "7")
        self.assertEqual(job.binary, "relax")
        self.assertEqual(job.pdb_path, Path("/mnt/out/run/7/input.pdb"))
        self.assertIsNone(job.rosetta_script_path)
        self.assertEqual(job.flags_path, Path("/mnt/out/run/_flags/abc.flags"))
        self.assertEqual(job.out_dir, Path("/mnt/out/run/7"))

    def test_normalize_bytes_job_stages_inputs_and_defaults_binary(self) -> None:
        with TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            job = _normalize_bytes_rosetta_job(
                {
                    "index": "job/1",
                    "pdb_bytes": b"ATOM\n",
                    "pdb_name": "../model.pdb",
                    "rosetta_script_text": "<ROSETTASCRIPTS />\n",
                    "rosetta_script_name": "protocol.xml",
                    "flags_text": "-mute all\n",
                    "flags_name": "opts.flags",
                },
                inputs_dir=workdir / "inputs",
                outputs_dir=workdir / "outputs",
            )

            self.assertEqual(job.index, "job_1")
            self.assertEqual(job.binary, "relax")
            self.assertEqual(job.pdb_path.read_bytes(), b"ATOM\n")
            self.assertEqual(job.pdb_path.name, "model.pdb")
            self.assertIsNotNone(job.rosetta_script_path)
            self.assertEqual(
                job.rosetta_script_path.read_text(), "<ROSETTASCRIPTS />\n"
            )
            self.assertIsNotNone(job.flags_path)
            self.assertEqual(job.flags_path.read_text(), "-mute all\n")
            self.assertEqual(job.out_dir, workdir / "outputs" / "job_1")

    def test_normalize_bytes_job_requires_pdb_bytes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(TypeError, "missing pdb_bytes"):
                _normalize_bytes_rosetta_job(
                    {"index": "bad", "pdb_bytes": "not bytes"},
                    inputs_dir=Path(tmpdir) / "inputs",
                    outputs_dir=Path(tmpdir) / "outputs",
                )


if __name__ == "__main__":
    unittest.main()
