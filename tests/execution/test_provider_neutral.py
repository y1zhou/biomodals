"""Provider-neutral kernel source-contract tests."""

import ast
from pathlib import Path


def test_root_execution_modules_do_not_import_modal() -> None:
    """Keep provider SDK and Modal-host imports below ``execution.modal``."""
    execution_root = Path("src/biomodals/execution")
    violations: list[str] = []
    for path in execution_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported = (node.module or "",)
            else:
                continue
            if any(
                name == "modal"
                or name.startswith("modal.")
                or name.startswith("biomodals.execution.modal")
                for name in imported
            ):
                violations.append(f"{path}:{node.lineno}")

    assert violations == []
