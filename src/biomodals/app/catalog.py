"""Application discovery and help-table helpers for the Biomodals CLI."""

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

APP_HOME = Path(__file__).parent.resolve()


class AppNotFoundError(ValueError):
    """Raised when an app reference cannot be resolved."""

    def __init__(self, app_name: str) -> None:
        """Initialize the error with the unresolved app name."""
        self.app_name = app_name
        super().__init__(f"Application '{app_name}' not found.")


@dataclass(frozen=True)
class AppReference:
    """Parsed app reference with optional Modal entrypoint."""

    app: str
    entrypoint: str | None = None


def get_all_apps(
    use_absolute_paths: bool = False,
    *,
    app_home: Path = APP_HOME,
    cwd: Path | None = None,
) -> dict[str, Path]:
    """Retrieve all available biomodals applications."""
    available_apps: dict[str, Path] = {}
    base_cwd = Path.cwd() if cwd is None else cwd
    for app_file in app_home.glob("*/*_app.py"):
        app_path = (
            app_file.resolve()
            if use_absolute_paths
            else app_file.relative_to(base_cwd, walk_up=True)
        )
        app_name = app_file.stem.replace("_app", "")
        available_apps[app_name] = app_path
    return available_apps


def parse_app_reference(app_name_or_path: str) -> AppReference:
    """Split an app reference into app name/path and optional entrypoint."""
    app_name, separator, entrypoint_name = app_name_or_path.partition("::")
    return AppReference(app=app_name, entrypoint=entrypoint_name if separator else None)


def resolve_app_path(
    app_name_or_path: str,
    *,
    app_home: Path = APP_HOME,
) -> Path:
    """Resolve an app name or filesystem path to an app file path."""
    all_apps = get_all_apps(use_absolute_paths=True, app_home=app_home)
    if app_name_or_path in all_apps:
        return all_apps[app_name_or_path]

    app_path = Path(app_name_or_path).expanduser()
    if not app_path.exists():
        raise AppNotFoundError(app_name_or_path)
    return app_path


def app_path_to_module_path(app_path: Path, *, app_home: Path = APP_HOME) -> str:
    """Convert an app path to a module path."""
    module_path = (
        str(app_path.resolve().relative_to(app_home))
        .replace("/", ".")
        .replace("\\", ".")
        .replace(".py", "")
        .replace("-", "_")
    )
    return f"biomodals.app.{module_path}"


def _arg_descriptions_from_google_docstring(doc: str) -> dict[str, str]:
    args_start = doc.find("Args:\n")
    if args_start == -1:
        return {}

    args_doc = doc[args_start:]
    doc_lines = args_doc.split("\n")
    if len(doc_lines) < 2:
        return {}

    first_arg_line = doc_lines[1]
    indent_level = len(first_arg_line) - len(first_arg_line.lstrip())
    continuation_indent = indent_level * 2
    arg_descriptions: dict[str, str] = {}

    for i, line in enumerate(doc_lines):
        if line.strip() == "Args:":
            continue
        line_indent = len(line) - len(line.lstrip())
        if line_indent == indent_level and ":" in line:
            arg_name, description = line.strip().split(":", maxsplit=1)
            description = description.strip()

            next_line_index = i + 1
            while next_line_index < len(doc_lines) and doc_lines[
                next_line_index
            ].startswith(" " * continuation_indent):
                description += " " + doc_lines[next_line_index].strip()
                next_line_index += 1

            arg_descriptions[arg_name] = description

    return arg_descriptions


def docstring_to_markdown_table(f: Callable) -> list[str]:
    """Convert a function docstring with Args into Markdown table rows."""
    sig = inspect.signature(f)
    arg_descriptions = _arg_descriptions_from_google_docstring(inspect.getdoc(f) or "")
    if not arg_descriptions:
        return []

    table_rows = [
        "| Flag | Default | Description |",
        "|-----:|:--------|:------------|",
    ]
    for name, p in sig.parameters.items():
        flag_base = name.replace("_", "-")
        default = (
            f"{p.default}"
            if p.default is not inspect.Parameter.empty
            else "**Required**"
        )
        flag = (
            f"`--{flag_base}`"
            if type(p.default) is not bool
            else f"`--{flag_base}`/`--no-{flag_base}`"
        )
        description = arg_descriptions.get(name, "")
        table_rows.append(f"| {flag} | {default} | {description} |")

    return table_rows
