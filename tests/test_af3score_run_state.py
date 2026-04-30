# ruff: noqa: D100, D101, D102

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from biomodals.app.score.af3score_app import _has_completed_outputs, _run_paths


class TestAF3ScoreRunState(unittest.TestCase):
    def test_run_paths_preserves_existing_keys_and_layout(self) -> None:
        paths = _run_paths("run-a")

        self.assertEqual(
            set(paths),
            {
                "mount_root",
                "run_root",
                "inputs_dir",
                "prep_dir",
                "output_dir",
                "failed_dir",
                "metrics_csv",
            },
        )
        self.assertEqual(paths["mount_root"], Path("/biomodals-outputs"))
        self.assertEqual(paths["run_root"], Path("/biomodals-outputs/run-a"))
        self.assertEqual(paths["inputs_dir"], Path("/biomodals-outputs/run-a/inputs"))
        self.assertEqual(paths["prep_dir"], Path("/biomodals-outputs/run-a/prepare"))
        self.assertEqual(paths["output_dir"], Path("/biomodals-outputs/run-a/outputs"))
        self.assertEqual(
            paths["failed_dir"],
            Path("/biomodals-outputs/run-a/outputs/failed_records"),
        )
        self.assertEqual(
            paths["metrics_csv"],
            Path("/biomodals-outputs/run-a/af3score_metrics.csv"),
        )

    def test_has_completed_outputs_preserves_required_af3score_files(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            sample_dir = output_dir / "input_a" / "seed-10_sample-0"
            sample_dir.mkdir(parents=True)
            (sample_dir / "summary_confidences.json").write_text("{}")

            self.assertFalse(_has_completed_outputs(output_dir, "input_a"))

            (sample_dir / "confidences.json").write_text("{}")

            self.assertTrue(_has_completed_outputs(output_dir, "input_a"))


if __name__ == "__main__":
    unittest.main()
