"""Helper script for constructing actual modal run commands."""

import importlib
from pathlib import Path
from typing import Annotated

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
        for obj in dir(mod):
            f = getattr(mod, obj)
            if callable(f) and isinstance(f, modal.app.LocalEntrypoint):
                # Parse the flags into a dict that can be passed to the function

                break
    else:
        _run_command(["biomodals", "help", str(app_path)])


if __name__ == "__main__":
    app()
