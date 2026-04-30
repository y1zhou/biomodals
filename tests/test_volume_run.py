# ruff: noqa: D100, D101, D102

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from biomodals.helper.volume_run import (
    build_volume_run_paths,
    has_completed_output_files,
)


class TestVolumeRun(unittest.TestCase):
    def test_build_volume_run_paths_documents_af3score_layout_policy(self) -> None:
        paths = build_volume_run_paths(
            mount_root=Path("/mnt/out"),
            run_name="sample-run",
            metrics_filename="af3score_metrics.csv",
        )

        self.assertEqual(
            paths,
            {
                "mount_root": Path("/mnt/out"),
                "run_root": Path("/mnt/out/sample-run"),
                "inputs_dir": Path("/mnt/out/sample-run/inputs"),
                "prep_dir": Path("/mnt/out/sample-run/prepare"),
                "output_dir": Path("/mnt/out/sample-run/outputs"),
                "failed_dir": Path("/mnt/out/sample-run/outputs/failed_records"),
                "metrics_csv": Path("/mnt/out/sample-run/af3score_metrics.csv"),
            },
        )

    def test_has_completed_output_files_requires_all_expected_files(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            sample_dir = output_dir / "input_a" / "seed-10_sample-0"
            sample_dir.mkdir(parents=True)
            (sample_dir / "summary_confidences.json").write_text("{}")

            self.assertFalse(
                has_completed_output_files(
                    output_dir,
                    "input_a",
                    sample_subdir="seed-10_sample-0",
                    required_files=("summary_confidences.json", "confidences.json"),
                )
            )

            (sample_dir / "confidences.json").write_text("{}")

            self.assertTrue(
                has_completed_output_files(
                    output_dir,
                    "input_a",
                    sample_subdir="seed-10_sample-0",
                    required_files=("summary_confidences.json", "confidences.json"),
                )
            )


if __name__ == "__main__":
    unittest.main()
