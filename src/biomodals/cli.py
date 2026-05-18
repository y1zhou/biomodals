"""Helper script for constructing actual modal run commands."""

import shlex
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from biomodals.app.catalog import AppNotFoundError, BiomodalsApp, get_all_apps
from biomodals.helper.shell import run_command

# ruff: noqa: S603

app = typer.Typer()
console = Console()


@app.callback(invoke_without_command=True, no_args_is_help=True)
def callback():
    """Biomodals CLI - List and get help for biomodals applications.

    This CLI helps users discover available biomodals applications and view their help documentation.
    """
    ...


##########################################
# CLI Commands
##########################################
def _load_app(name: str) -> BiomodalsApp:
    """Load a biomodals app by name or path."""
    # TODO(workflows): add a workflow-aware loader that can receive a workflow
    # catalog from get_all_apps(app_home=WORKFLOW_HOME, suffix="workflow").
    try:
        return BiomodalsApp(name)
    except AppNotFoundError as e:
        console.print(f"[bold red]Error[/bold red] failed to find app '{name}': {e}")
        raise typer.Exit(code=1) from e
    except ImportError as e:
        console.print(f"[bold red]Error[/bold red] Failed to import '{name}': {e}")
        raise typer.Exit(code=1) from e


def _print_title(title: str) -> None:
    """Styling for titles."""
    console.print(
        f"\n\n[bold underline2]{title}[/bold underline2]\n",
        justify="center",
        highlight=True,
    )


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
        Literal["name", "category", "group", "path"],
        typer.Option(
            "--sort-by",
            "-s",
            help="Key to sort the applications by in the table display.",
            case_sensitive=False,
        ),
    ] = "path",
    reverse: Annotated[
        bool,
        typer.Option(
            "--reverse", "-r", help="Reverse the sorting order in the table display."
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
    # TODO(workflows): add a workflow listing mode or sibling command that calls
    # get_all_apps(app_home=WORKFLOW_HOME, suffix="workflow") and labels rows as
    # workflows instead of applications.
    table_headers = ["App name", "Category", "App path"]
    available_apps = get_all_apps(use_absolute_paths)
    table_rows: list[tuple[str, str, str]] = []
    for app_name, app_path in available_apps.items():
        app_category = app_path.parent.name
        table_rows.append((
            f"[green]{app_name}[/green]",
            app_category,
            str(app_path),
        ))
    match sort_by:
        case "name":
            sort_by_idx = table_headers.index("App name")
        case "category" | "group":
            sort_by_idx = table_headers.index("Category")
        case "path":
            sort_by_idx = table_headers.index("App path")
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
    console.print(
        "\n:dna: If an app contains multiple local entrypoints, use it as:\n"
        "     [bold]<[green]app-name-or-path[/green]>::<[green]function-name[/green]>[/bold]\n"
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
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed help for all functions."),
    ] = False,
):
    """Show help for a specific biomodals application.

    If unsure which app to use, run `biomodals list` to see available apps.
    If you would like to see help for a local entrypoint or Modal function,
    add `::<function-name>` to the app name to show help for that specific function.
    """
    # TODO(workflows): route workflow names through a workflow catalog once the
    # workflow CLI surface is stable.
    app = _load_app(app_name)
    if app._entrypoint is not None:
        # When an entrypoint name is specified, show only its docstring
        f = app[app._entrypoint]
        console.print(
            f"[bold]Help for {f.func_type} function"
            f"'[green]{f.name}[/green]'"
            f" in app '[green]{app.name}[/green]' ({app.category}):[/bold]\n"
        )
        console.print(f.docstring or "No documentation available.")
        if table_rows := f.args_table:
            _print_title("Entrypoint CLI flags")
            console.print(Markdown("\n".join(table_rows)))
        return

    # When no entrypoint is specified, show the app help
    console.print(
        "[bold]Help for application"
        f" '[green]{app.name}[/green]' ({app.category}):[/bold]"
    )
    if app.module_doc:
        _print_title("Module documentation")
        console.print(Markdown(app.module_doc))
    if app._remote_modal_func_idx:
        remote_modal_functions = [app[x] for x in app._remote_modal_func_idx]

        _print_title("Remote Modal functions in this app")
        remote_func_names = ", ".join([x.name for x in remote_modal_functions])
        console.print(f"[green]{remote_func_names}[/green]\n")
        if verbose:
            for f in remote_modal_functions:
                if f.docstring:
                    console.print(f"\n[bold green]{f.name}[/bold green]")
                    console.print(Markdown(f.docstring))

    if f_indices := app._local_entrypoint_idx:
        _print_title("Local entrypoint(s) in this app")
        for f_idx in f_indices:
            f = app[f_idx]

            if f.args_table:
                console.print(f"[bold green]{f.name}[/bold green] CLI flags:\n")
                console.print(Markdown("\n".join(f.args_table)))
            elif f.docstring:
                console.print(f"[bold green]{f.name}[/bold green] documentation:\n")
                console.print(Markdown(f.docstring))


@app.command(
    name="run",
    no_args_is_help=True,
    help="Run a biomodals application on Modal (alias: r).",
)
@app.command(name="r", no_args_is_help=True, hidden=True)
def run_modal_app(
    app_name_or_path: Annotated[
        str, typer.Argument(help="Name or path of the app to run.")
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
    # TODO(workflows): add workflow run semantics separately from Modal app runs
    # so workflow-* names can stage workflow inputs before invoking orchestrators.
    import sys

    app = _load_app(app_name_or_path)

    full_app = (
        str(app.path) if app._entrypoint is None else f"{app.path}::{app._entrypoint}"
    )
    cmd = [sys.executable, "-m", "modal", modal_mode]
    if detach:
        cmd.append("-d")
    cmd.append(str(full_app))

    if modal_mode == "shell":
        console.print(
            "To start an interactive shell for the app, run:\n"
            f"[bold green]{shlex.join(cmd)}[/bold green]"
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
    elif app._entrypoint is not None:
        run_command(["biomodals", "help", str(full_app)], try_rich_print=True)
    else:
        run_command(["biomodals", "help", str(app.path)], try_rich_print=True)


@app.command(
    name="deploy",
    no_args_is_help=True,
    help="Deploy a biomodals application to Modal (alias: d).",
)
@app.command(name="d", no_args_is_help=True, hidden=True)
def deploy_app(
    app_name_or_path: Annotated[
        str, typer.Argument(help="Name or path of the app to deploy.")
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
    app = _load_app(app_name_or_path)
    cmd = ["modal", "deploy"]
    if name:
        cmd.extend(["--name", name])
    if tag:
        cmd.extend(["--tag", tag])
    cmd.append(str(app.path))
    run_command(cmd)


if __name__ == "__main__":
    app()
