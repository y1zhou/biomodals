"""Application discovery and help-table helpers for the Biomodals CLI."""

import importlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import modal

APP_HOME = Path(__file__).parent.resolve()
BIOMODALS_HOME = APP_HOME.parent
WORKFLOW_HOME = BIOMODALS_HOME / "workflow"


class AppNotFoundError(ValueError):
    """Raised when an app reference cannot be resolved."""

    def __init__(self, app_name: str) -> None:
        """Initialize the error with the unresolved app name."""
        self.app_name = app_name
        super().__init__(f"Application '{app_name}' not found.")


def get_all_apps(
    use_absolute_paths: bool = False,
    *,
    app_home: Path = APP_HOME,
    cwd: Path | None = None,
    suffix: str = "app",
) -> dict[str, Path]:
    """Retrieve all available biomodals applications."""
    available_apps: dict[str, Path] = {}
    base_cwd = Path.cwd() if cwd is None else cwd
    glob_pattern = f"*/*_{suffix}.py" if app_home == APP_HOME else f"*_{suffix}.py"
    for app_file in app_home.glob(glob_pattern):
        app_path = (
            app_file.resolve()
            if use_absolute_paths
            else app_file.relative_to(base_cwd, walk_up=True)
        )
        app_name = app_file.stem.removesuffix(f"_{suffix}")
        if suffix == "workflow":
            app_name = f"workflow-{app_name}"
        available_apps[app_name] = app_path
    return available_apps


@dataclass(frozen=True)
class AppFunction:
    """Information about a Modal or local entrypoint function."""

    name: str
    func_type: Literal["modal", "local_entrypoint"]
    docstring: str | None
    args_table: list[str]


class BiomodalsApp:
    """Metadata container for an app.

    Attributes:
        name (str): The name of the app.
        category (str): The category of the app.
        path (Path): The path to the app file.
        module (str): The module path of the app.
        module_doc (str | None): The docstring of the module.
        functions (list[AppFunction]): A list of local entrypoint or Modal functions defined in the app.

        _entrypoint (str | None): The name of the entrypoint function of interest.
        _func_idx (dict[str, int]): A mapping of function names to their index in the functions list.
        _local_entrypoint_idx (list[int]): A list of indices of local entrypoint functions in the functions list.
        _remote_modal_func_idx (list[int]): A list of indices of remote Modal functions in the functions list.
    """

    def __init__(
        self, app_name_or_path: str, all_apps: dict[str, Path] | None = None
    ) -> None:
        """Initialize the app with a given name or path."""
        # Extract entrypoint name if specified
        name_or_path, separator, entrypoint_name = app_name_or_path.partition("::")
        self._entrypoint = None
        if entrypoint_name:
            self._entrypoint = entrypoint_name

        # Normalize app name & path
        self._all_apps = all_apps or get_all_apps(
            use_absolute_paths=True, app_home=APP_HOME
        )
        self.name, self.path = self.resolve_app_path(name_or_path)
        self.category = self.path.parent.name
        self.module = self.app_path_to_module_path(self.path)

        # Load functions and build index for quick lookup
        self.functions: list[AppFunction] = []
        self.populate_functions()

        self._func_idx: dict[str, int] = {}
        self._local_entrypoint_idx: list[int] = []
        self._remote_modal_func_idx: list[int] = []
        for idx, func in enumerate(self.functions):
            self._func_idx[func.name] = idx
            if func.func_type == "local_entrypoint":
                self._local_entrypoint_idx.append(idx)
            elif func.func_type == "modal":
                self._remote_modal_func_idx.append(idx)
            else:
                raise ValueError(f"Unknown function type: {func.func_type}")

        if self._entrypoint and self._entrypoint not in self._func_idx:
            raise AppNotFoundError(
                f"Entrypoint '{self._entrypoint}' not found in app '{self.name}'."
            )

    def __getitem__(self, name: str | int) -> AppFunction:
        """Get a function by its name or index."""
        if isinstance(name, str):
            return self.functions[self._func_idx[name]]
        return self.functions[name]

    def resolve_app_path(self, app_name_or_path: str) -> tuple[str, Path]:
        """Resolve an app name or filesystem path to its name and absolute path."""
        if app_name_or_path in self._all_apps:
            return app_name_or_path, self._all_apps[app_name_or_path]

        app_path = Path(app_name_or_path).expanduser()
        if not app_path.exists():
            raise AppNotFoundError(app_name_or_path)
        return app_path.stem.removesuffix("_app"), app_path

    @staticmethod
    def app_path_to_module_path(app_path: Path) -> str:
        """Convert an app path to a module path."""
        resolved_path = app_path.resolve()
        if resolved_path.is_relative_to(APP_HOME):
            module_path = (
                str(resolved_path.relative_to(APP_HOME))
                .replace("/", ".")
                .replace("\\", ".")
                .replace(".py", "")
                .replace("-", "_")
            )
            return f"biomodals.app.{module_path}"
        module_path = (
            str(resolved_path.relative_to(BIOMODALS_HOME))
            .replace("/", ".")
            .replace("\\", ".")
            .replace(".py", "")
            .replace("-", "_")
        )
        return f"biomodals.{module_path}"

    def populate_functions(self):
        """Collect all functions within the app."""
        module = importlib.import_module(self.module)
        self.module_doc = module.__doc__

        for obj in dir(module):
            f = getattr(module, obj)
            if isinstance(f, modal.Function):
                raw_f = f.get_raw_f()
                func_type = "modal"
            elif isinstance(f, modal.app.LocalEntrypoint):
                raw_f = f.info.raw_f
                func_type = "local_entrypoint"
            else:
                continue
            self.functions.append(
                AppFunction(
                    name=obj,
                    func_type=func_type,
                    docstring=raw_f.__doc__ or None,
                    args_table=_docstring_to_markdown_table(raw_f),
                )
            )


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


def _docstring_to_markdown_table(f: Callable) -> list[str]:
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
