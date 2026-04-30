# ruff: noqa: D100, D101, D102

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from biomodals.app.catalog import (
    AppNotFoundError,
    docstring_to_markdown_table,
    get_all_apps,
    parse_app_reference,
    resolve_app_path,
)


class TestAppCatalog(unittest.TestCase):
    def test_app_name_lookup_uses_fake_app_home(self) -> None:
        with TemporaryDirectory() as tmpdir:
            app_home = Path(tmpdir)
            app_file = app_home / "fold" / "fakefold_app.py"
            app_file.parent.mkdir()
            app_file.write_text('"""Fake app."""\n')

            resolved = resolve_app_path("fakefold", app_home=app_home)

            self.assertEqual(resolved, app_file.resolve())

    def test_path_lookup_uses_fake_app_file(self) -> None:
        with TemporaryDirectory() as tmpdir:
            app_file = Path(tmpdir) / "custom_app.py"
            app_file.write_text('"""Custom app."""\n')

            resolved = resolve_app_path(str(app_file), app_home=Path(tmpdir) / "apps")

            self.assertEqual(resolved, app_file)

    def test_parse_app_reference_splits_entrypoint_once(self) -> None:
        app_ref = parse_app_reference("alphafold3::submit_alphafold3_task")

        self.assertEqual(app_ref.app, "alphafold3")
        self.assertEqual(app_ref.entrypoint, "submit_alphafold3_task")

    def test_missing_app_raises_with_requested_name(self) -> None:
        with TemporaryDirectory() as tmpdir:
            with self.assertRaises(AppNotFoundError) as raised:
                resolve_app_path("missing", app_home=Path(tmpdir))

            self.assertEqual(raised.exception.app_name, "missing")
            self.assertEqual(str(raised.exception), "Application 'missing' not found.")

    def test_get_all_apps_keeps_existing_discovery_shape(self) -> None:
        with TemporaryDirectory() as tmpdir:
            app_home = Path(tmpdir) / "src" / "biomodals" / "app"
            app_file = app_home / "fold" / "fakefold_app.py"
            ignored_file = app_home / "workflow" / "fake_workflow.py"
            app_file.parent.mkdir(parents=True)
            ignored_file.parent.mkdir()
            app_file.write_text('"""Fake app."""\n')
            ignored_file.write_text('"""Ignored."""\n')
            apps = get_all_apps(app_home=app_home, cwd=Path(tmpdir))

            self.assertEqual(
                apps, {"fakefold": Path("src/biomodals/app/fold/fakefold_app.py")}
            )

    def test_docstring_to_markdown_table_keeps_continuation_lines(self) -> None:
        def local_entrypoint(
            input_json: str,
            use_cache: bool = True,
        ) -> None:
            """Run a fake app.

            Args:
                input_json: Input JSON path.
                    Continuation line with more detail: keep the colon.
                use_cache: Reuse cached model weights.
                    Disable for clean reruns.
            """

        rows = docstring_to_markdown_table(local_entrypoint)

        self.assertIn(
            "| `--input-json` | **Required** | Input JSON path. "
            "Continuation line with more detail: keep the colon. |",
            rows,
        )
        self.assertIn(
            "| `--use-cache`/`--no-use-cache` | True | "
            "Reuse cached model weights. Disable for clean reruns. |",
            rows,
        )


if __name__ == "__main__":
    unittest.main()
