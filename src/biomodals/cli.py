"""Helper script for constructing actual modal run commands."""

import importlib
import shlex
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from biomodals.app.helper.shell import run_command

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
# Helper Functions
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


##########################################
# CLI Commands
##########################################
@app.command(name="list")
@app.command(name="ls")
@app.command(name="l")
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
@app.command(name="h")
def show_app_help(
    app_name: Annotated[
        str, typer.Argument(help="Name or path of the app to show help for.")
    ],
):
    """Show help for a specific biomodals application."""
    import modal

    all_apps = get_all_apps(use_absolute_paths=True)
    app_name, entrypoint_name = (
        app_name.split("::") if "::" in app_name else (app_name, None)
    )
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

        remote_modal_functions: list[str] = []
        local_entrypoint_docstring: str = ""
        for obj in dir(module):
            f = getattr(module, obj)
            if isinstance(f, modal.Function):
                # When an entrypoint name is specified, show only its docstring
                if entrypoint_name is not None and obj == entrypoint_name:
                    console.print(
                        f"[bold]Docstring for entrypoint function '[green]{entrypoint_name}[/green]':[/bold]\n"
                    )
                    console.print(
                        f.get_raw_f().__doc__ or "No documentation available."
                    )
                    return

                remote_modal_functions.append(obj)

            if isinstance(f, modal.app.LocalEntrypoint):
                local_entrypoint_docstring = f.info.raw_f.__doc__ or ""

        console.print(
            f"[bold]Help for application '[green]{app_path}[/green]':[/bold]\n"
        )
        console.print(
            "\n\n[bold underline2]Module documentation[/bold underline2]\n",
            justify="center",
            highlight=True,
        )
        if remote_modal_functions:
            console.print(
                f"[bold]Modal functions in this app:[/bold] [green]{', '.join(remote_modal_functions)}[/green]\n"
            )
        if docstring := module.__doc__:
            rendered_doc = Markdown(docstring)
            console.print(rendered_doc)
        if local_entrypoint_docstring:
            console.print(
                "\n\n[bold underline2]Local entrypoint documentation[/bold underline2]\n",
                justify="center",
                highlight=True,
            )
            console.print(local_entrypoint_docstring)
        if not (docstring or local_entrypoint_docstring):
            console.print("No documentation available.")
    except ImportError as e:
        console.print(f"[bold red]Error:[/bold red] Failed to import '{module_path}'")
        raise typer.Exit(code=1) from e


@app.command(name="run")
@app.command(name="r")
def run_modal_app(
    app_name_or_path: Annotated[
        str, typer.Argument(help="Name or path of the app to generate run command for.")
    ],
    modal_mode: Annotated[
        str,
        typer.Option("--mode", "-m", help="Modal command to use ('run' or 'shell')."),
    ] = "run",
    detach: Annotated[
        bool,
        typer.Option("--detach", "-d", help="Run the modal command in detached mode."),
    ] = False,
    gpu: Annotated[
        str | None,
        typer.Option("--gpu", help="GPU type to use for the modal run (e.g. 'L40S'). "),
    ] = None,
    timeout: Annotated[
        int | None,
        typer.Option(
            "--timeout",
            help="Timeout in seconds for the modal run. If not specified, use the app default.",
        ),
    ] = None,
    flags: Annotated[
        list[str] | None,
        typer.Argument(help="Additional flags to pass to the modal run command."),
    ] = None,
):
    """Generate the modal run command for a specific biomodals application.

    Use with: `biomodals run <app-name> [OPTIONS] -- [app-options]`, where `[app-options]` are
    additional flags to pass to the `modal run <app-name>` command.
    """
    all_apps = get_all_apps(use_absolute_paths=True)
    app_name_or_path, entrypoint_name = (
        app_name_or_path.split("::")
        if "::" in app_name_or_path
        else (app_name_or_path, None)
    )
    if app_name_or_path in all_apps:
        app_path = all_apps[app_name_or_path]
    else:
        app_path = Path(app_name_or_path).expanduser()
        if not app_path.exists():
            console.print(
                f"[bold red]Error:[/bold red] Application '{app_name_or_path}' not found."
            )
            raise typer.Exit(code=1)

    full_app = (
        str(app_path) if entrypoint_name is None else f"{app_path}::{entrypoint_name}"
    )
    cmd = ["modal", modal_mode]
    if detach:
        cmd.append("-d")
    cmd.append(str(full_app))

    if modal_mode == "shell":
        console.print(
            "To start an interactive shell for the app, run:\n"
            f"[bold green]uv run {shlex.join(cmd)}[/bold green]"
        )
    elif flags:
        # TODO: figure out a way to tag run names into the app.
        # Previously we used the MODAL_APP environment variable for ephemeral
        # apps run with the --run-name flag, but with the new AppConfig API
        # this is no longer read.
        import os

        env = os.environ.copy()
        if gpu is not None:
            env["GPU"] = gpu
        if timeout is not None:
            env["TIMEOUT"] = str(timeout)
        run_command([*cmd, *flags], env=env)
    elif entrypoint_name is not None:
        run_command(["biomodals", "help", str(full_app)], try_rich_print=True)
    else:
        run_command(["biomodals", "help", str(app_path)], try_rich_print=True)


if __name__ == "__main__":
    app()
