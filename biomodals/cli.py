"""Helper script for constructing actual modal run commands."""

import importlib
import inspect
from pathlib import Path
from typing import Annotated, get_args, get_origin

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

# ruff: noqa: S603
APP_HOME = Path(__file__).parent.resolve() / "app"


app = typer.Typer()
console = Console()


@app.callback(invoke_without_command=True, no_args_is_help=True)
def callback():
    """Biomodals CLI - List and get help for biomodals applications.

    This CLI helps users discover available biomodals applications and view their help documentation.
    """


##########################################
# Helper functions
##########################################
def get_all_apps(use_absolute_paths: bool = False) -> dict[str, Path]:
    """Retrieve all available biomodals applications."""
    available_apps: dict[str, Path] = {}
    cwd = Path.cwd()
    for app_file in APP_HOME.glob("*/*_app.py"):
        app_path = (
            app_file.resolve()
            if use_absolute_paths
            else app_file.relative_to(cwd, walk_up=True)
        )
        app_name = app_file.stem.replace("_app", "")
        available_apps[app_name] = app_path
    return available_apps


def app_path_to_module_path(app_path: Path) -> str:
    """Convert an app path to a module path."""
    module_path = (
        str(app_path.resolve().relative_to(APP_HOME))
        .replace("/", ".")
        .replace("\\", ".")
        .replace(".py", "")
        .replace("-", "_")
    )
    return f"biomodals.app.{module_path}"


def _run_command(cmd: list[str], **kwargs) -> None:
    """Run a shell command and stream output to stdout."""
    import subprocess as sp

    print(f"Running command: {' '.join(cmd)}")
    # Set default kwargs for sp.Popen
    kwargs.setdefault("stdout", sp.PIPE)
    kwargs.setdefault("stderr", sp.STDOUT)
    kwargs.setdefault("bufsize", 1)
    kwargs.setdefault("encoding", "utf-8")

    with sp.Popen(cmd, **kwargs) as p:
        if p.stdout is None:
            raise RuntimeError("Failed to capture stdout from the command.")

        buffered_output = None
        while (buffered_output := p.stdout.readline()) != "" or p.poll() is None:
            print(buffered_output, end="", flush=True)

        if p.returncode != 0:
            raise sp.CalledProcessError(p.returncode, cmd, buffered_output)


def parse_flags_to_kwargs(flags: list[str], sig: inspect.Signature) -> dict:
    """Parse command-line flags into a dictionary of keyword arguments.

    Args:
        flags: List of command-line flags (e.g., ["--input-yaml", "file.yaml", "--num-designs", "5"])
        sig: Function signature to match parameter types against

    Returns:
        Dictionary of parsed keyword arguments ready to pass to the function

    """
    kwargs = {}
    i = 0

    while i < len(flags):
        flag = flags[i]

        if not flag.startswith("--"):
            raise ValueError(f"Expected flag starting with '--', got: {flag}")

        # Handle --no-flag pattern (boolean False)
        if flag.startswith("--no-"):
            param_name = flag[5:].replace("-", "_")
            kwargs[param_name] = False
            i += 1
            continue

        # Handle regular --flag pattern
        param_name = flag[2:].replace("-", "_")

        # Find the parameter in the signature
        if param_name not in sig.parameters:
            raise ValueError(f"Unknown parameter: {param_name}")

        param = sig.parameters[param_name]
        param_type = param.annotation

        # Handle union types (e.g., str | None)
        origin = get_origin(param_type)
        if origin is not None:
            type_args = get_args(param_type)
            # Filter out None from union types
            non_none_types = [t for t in type_args if t is not type(None)]
            if non_none_types:
                param_type = non_none_types[0]

        # Determine if this is a boolean flag or requires a value
        if param_type is bool or param_type == "bool":
            # Boolean flag without value means True
            kwargs[param_name] = True
            i += 1
        else:
            # Get the next item as the value
            if i + 1 >= len(flags):
                raise ValueError(f"Flag {flag} requires a value")

            value_str = flags[i + 1]

            # Convert the value to the appropriate type
            try:
                if param_type is int or param_type == "int":
                    kwargs[param_name] = int(value_str)
                elif param_type is float or param_type == "float":
                    kwargs[param_name] = float(value_str)
                elif param_type is str or param_type == "str":
                    kwargs[param_name] = value_str
                else:
                    # Default to string for unknown types
                    kwargs[param_name] = value_str
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Failed to convert value '{value_str}' for parameter '{param_name}' to type {param_type}"
                ) from e

            i += 2

    return kwargs


##########################################
# CLI Commands
##########################################
@app.command(name="list")
def list_available_apps(
    use_absolute_paths: Annotated[
        bool,
        typer.Option("--absolute", "-a", help="Use absolute paths for app locations."),
    ] = False,
) -> dict[str, Path]:
    """Show a list of all available biomodals applications."""
    table = Table("App name", "App path", "Category")

    available_apps = get_all_apps(use_absolute_paths)
    for app_name, app_path in available_apps.items():
        app_category = app_path.parent.name
        table.add_row(f"[green]{app_name}[/green]", str(app_path), app_category)

    console.print(
        "\n:dna: To see help for an application, use:\n"
        "     [bold]biomodals help <[green]app-name[/green]>[/bold]"
        " or [bold]biomodals help <[green]app-path[/green]>[/bold]"
    )
    console.print(
        "\n:dna: To run an application on [link=https://modal.com]modal.com[/link], use:\n"
        r"     [bold]modal run <[green]app-path[/green]>[/bold] [gray]\[OPTIONS][/gray]"
    )
    console.print("\n:dna: [bold]Available biomodals applications:[/bold]")
    console.print(table)
    return available_apps


@app.command(name="help")
def show_app_help(
    app_name: Annotated[
        str, typer.Argument(help="Name or path of the app to show help for.")
    ],
):
    """Show help for a specific biomodals application."""
    all_apps = get_all_apps(use_absolute_paths=True)
    if app_name in all_apps:
        app_path = all_apps[app_name]
    else:
        app_path = Path(app_name).expanduser()
        if not app_path.exists():
            console.print(
                f"[bold red]Error:[/bold red] Application '{app_name}' not found."
            )
            raise typer.Exit(code=1)

    module_path = app_path_to_module_path(app_path)
    try:
        module = importlib.import_module(module_path)

        console.print(
            f"[bold]Help for application '[green]{app_path}[/green]':[/bold]\n"
        )
        if docstring := module.__doc__:
            rendered_doc = Markdown(docstring)
            console.print(rendered_doc)
        else:
            console.print("No documentation available.")
    except ImportError as e:
        console.print(f"[bold red]Error:[/bold red] Failed to import '{module_path}'")
        raise typer.Exit(code=1) from e


@app.command(name="run")
def run_command(
    app_name_or_path: Annotated[
        str, typer.Argument(help="Name or path of the app to generate run command for.")
    ],
    flags: Annotated[
        list[str] | None,
        typer.Argument(help="Additional flags to pass to the modal run command."),
    ] = None,
):
    """Generate the modal run command for a specific biomodals application.

    Use with: `biomodals run <app-name> -- [OPTIONS]`, where `[OPTIONS]` are
    additional flags to pass to the `modal run <app-name>` command.
    """
    import modal

    all_apps = get_all_apps(use_absolute_paths=True)
    if app_name_or_path in all_apps:
        app_path = all_apps[app_name_or_path]
    else:
        app_path = Path(app_name_or_path).expanduser()
        if not app_path.exists():
            console.print(
                f"[bold red]Error:[/bold red] Application '{app_name_or_path}' not found."
            )
            raise typer.Exit(code=1)

    if flags:
        module_path = app_path_to_module_path(app_path)
        mod = importlib.import_module(module_path)
        # find the local entrypoint function
        entrypoint_func = None
        for obj in dir(mod):
            f = getattr(mod, obj)
            if callable(f) and isinstance(f, modal.app.LocalEntrypoint):
                entrypoint_func = f
                break

        if entrypoint_func is None:
            console.print(
                f"[bold red]Error:[/bold red] No local entrypoint found in '{app_path}'"
            )
            raise typer.Exit(code=1)

        # Get the function signature from the wrapped function
        sig = inspect.signature(entrypoint_func.info.raw_f)

        # Parse the flags into a dict that can be passed to the function
        try:
            kwargs = parse_flags_to_kwargs(flags, sig)
            console.print(
                f"[bold green]Running {app_name_or_path} with arguments:[/bold green]"
            )
            for key, value in kwargs.items():
                console.print(f"  {key}: {value}")

            # Call the function with parsed arguments
            entrypoint_func(**kwargs)
        except ValueError as e:
            console.print(f"[bold red]Error parsing flags:[/bold red] {e}")
            raise typer.Exit(code=1) from e
    else:
        _run_command(["biomodals", "help", str(app_path)])


if __name__ == "__main__":
    app()
