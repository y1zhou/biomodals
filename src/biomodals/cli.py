"""Helper script for constructing actual modal run commands."""

import importlib
import shlex
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from biomodals.app.catalog import (
    AppNotFoundError,
    app_path_to_module_path,
    docstring_to_markdown_table,
    get_all_apps,
    parse_app_reference,
    resolve_app_path,
)
from biomodals.helper.shell import run_command

# ruff: noqa: S603

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
def _git_last_modified(app_path: Path) -> float:
    """Get the last commit timestamp for a file using git log.

    Returns the Unix timestamp of the last commit that touched the file,
    or None if the file is not tracked by git or git is not available.
    """
    try:
        output = run_command(
            ["git", "log", "-1", "--format=%ct", "--", str(app_path)],
            verbose=False,
            cwd=Path.cwd(),
        )
        if output and output[0].strip():
            return float(output[0].strip())
    except Exception:  # noqa: S110
        pass

    # fallback to file mod time if git info is not available
    return float(app_path.stat().st_mtime)


##########################################
# CLI Commands
##########################################
@app.command(
    name="list",
    help="Show a list of all available biomodals applications (aliases: ls, l).",
)
@app.command(name="ls", hidden=True)
@app.command(name="l", hidden=True)
def list_available_apps(
    use_absolute_paths: Annotated[
        bool,
        typer.Option("--absolute", "-a", help="Use absolute paths for app locations."),
    ] = False,
    sort_by: Annotated[
        Literal["name", "category", "group", "time", "date", "updated"],
        typer.Option(
            "--sort-by",
            "-s",
            help="Key to sort the applications by in the table display.",
            case_sensitive=False,
        ),
    ] = "time",
    reverse: Annotated[
        bool,
        typer.Option(
            "--reverse", "-r", help="Reverse the sorting order in the table display."
        ),
    ] = False,
    use_git_time: Annotated[
        bool,
        typer.Option(
            "--git-time",
            help="Use the last git commit time instead of file modification time for sorting.",
        ),
    ] = False,
    short: Annotated[
        bool,
        typer.Option(
            "--short",
            help="Only show app names without paths or additional info.",
            is_flag=True,
        ),
    ] = False,
) -> dict[str, Path]:
    """Show a list of all available biomodals applications."""
    from datetime import datetime

    table_headers = ["App name", "App path", "Category", "Updated at"]

    available_apps = get_all_apps(use_absolute_paths)
    table_rows: list[tuple[str, str, str, str]] = []
    for app_name, app_path in available_apps.items():
        app_category = app_path.parent.name
        updated_date = (
            _git_last_modified(app_path)
            if use_git_time
            else float(app_path.stat().st_mtime)
        )
        table_rows.append((
            f"[green]{app_name}[/green]",
            str(app_path),
            app_category,
            datetime.fromtimestamp(updated_date).strftime("%Y-%m-%d %H:%M:%S"),
        ))
    match sort_by:
        case "name":
            sort_by_idx = table_headers.index("App name")
        case "category" | "group":
            sort_by_idx = table_headers.index("Category")
        case "time" | "date" | "updated":
            sort_by_idx = table_headers.index("Updated at")
        case _:
            raise ValueError(f"Invalid sort key: {sort_by}")
    table_rows.sort(key=lambda x: x[sort_by_idx], reverse=reverse)
    if short:
        for r in table_rows:
            console.print(r[0])
        return available_apps

    table = Table(*table_headers)
    for r in table_rows:
        table.add_row(*r)

    console.print(
        "\n:dna: To see help for an application, use:\n"
        "     [bold]biomodals help <[green]app-name-or-path[/green]>[/bold]"
    )
    console.print(
        "\n:dna: To run an application on [link=https://modal.com]modal.com[/link], use:\n"
        r"     [bold]biomodals run <[green]app-name-or-path[/green]>[/bold] -- [gray]\[OPTIONS][/gray]"
    )
    console.print("\n:dna: [bold]Available biomodals applications:[/bold]")
    console.print(table)
    return available_apps


@app.command(
    name="help",
    no_args_is_help=True,
    help="Show help for a specific biomodals application (alias: h).",
)
@app.command(name="h", no_args_is_help=True, hidden=True)
def show_app_help(
    app_name: Annotated[
        str, typer.Argument(help="Name or path of the app to show help for.")
    ],
):
    """Show help for a specific biomodals application."""
    import modal

    app_reference = parse_app_reference(app_name)
    try:
        app_path = resolve_app_path(app_reference.app)
    except AppNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    module_path = app_path_to_module_path(app_path)
    try:
        module = importlib.import_module(module_path)

        remote_modal_functions: list[str] = []
        # local_entrypoint_docstring: str = ""
        args_table: list[str] = []
        for obj in dir(module):
            f = getattr(module, obj)
            if isinstance(f, modal.Function):
                # When an entrypoint name is specified, show only its docstring
                if (
                    app_reference.entrypoint is not None
                    and obj == app_reference.entrypoint
                ):
                    console.print(
                        "[bold]Docstring for entrypoint function "
                        f"'[green]{app_reference.entrypoint}[/green]':[/bold]\n"
                    )
                    console.print(
                        f.get_raw_f().__doc__ or "No documentation available."
                    )
                    return

                remote_modal_functions.append(obj)

            if isinstance(f, modal.app.LocalEntrypoint):
                # local_entrypoint_docstring = f.info.raw_f.__doc__ or ""
                args_table = docstring_to_markdown_table(f.info.raw_f)

        console.print(f"[bold]Help for application '[green]{app_path}[/green]':[/bold]")
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
            console.print(Markdown(docstring))
        if args_table:
            console.print(
                "\n\n[bold underline2]Entrypoint CLI flags[/bold underline2]",
                justify="center",
                highlight=True,
            )
            console.print(Markdown("\n".join(args_table)))
            # console.print(local_entrypoint_docstring)
        if not (docstring or args_table):
            console.print("No documentation available.")
    except ImportError as e:
        console.print(f"[bold red]Error:[/bold red] Failed to import '{module_path}'")
        raise typer.Exit(code=1) from e


@app.command(
    name="run",
    no_args_is_help=True,
    help="Run a biomodals application on Modal (alias: r).",
)
@app.command(name="r", no_args_is_help=True, hidden=True)
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
    """Run a biomodals application on Modal.

    Use with: `biomodals run <app-name> [OPTIONS] -- [app-options]`, where `[app-options]` are
    additional flags to pass to the `modal run <app-name>` command.
    """
    app_reference = parse_app_reference(app_name_or_path)
    try:
        app_path = resolve_app_path(app_reference.app)
    except AppNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1) from e

    full_app = (
        str(app_path)
        if app_reference.entrypoint is None
        else f"{app_path}::{app_reference.entrypoint}"
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
    elif app_reference.entrypoint is not None:
        run_command(["biomodals", "help", str(full_app)], try_rich_print=True)
    else:
        run_command(["biomodals", "help", str(app_path)], try_rich_print=True)


@app.command(
    name="deploy",
    no_args_is_help=True,
    help="Deploy a biomodals application to Modal (alias: d).",
)
@app.command(name="d", no_args_is_help=True, hidden=True)
def deploy_app(
    app_name_or_path: Annotated[
        str, typer.Argument(help="Name or path of the app to generate run command for.")
    ],
    name: Annotated[
        str | None, typer.Option("--name", "-n", help="Name of the deployment.")
    ] = None,
    tag: Annotated[
        str | None,
        typer.Option("--tag", "-t", help="Tag the deployment with a version."),
    ] = None,
):
    """Deploy a biomodals application to Modal."""
    app_reference = parse_app_reference(app_name_or_path)
    try:
        app_path = resolve_app_path(app_reference.app)
    except AppNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1) from e
    cmd = ["modal", "deploy"]
    if name:
        cmd.extend(["--name", name])
    if tag:
        cmd.extend(["--tag", tag])
    cmd.append(str(app_path))
    run_command(cmd)


if __name__ == "__main__":
    app()
