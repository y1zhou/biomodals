"""Helper script for constructing actual modal run commands."""

import shlex
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from biomodals.helper.catalog import (
    WORKFLOW_HOME,
    AppNotFoundError,
    BiomodalsApp,
    CatalogType,
    get_catalog,
)
from biomodals.helper.shell import run_command

# ruff: noqa: S603

app = typer.Typer()
app_commands = typer.Typer(no_args_is_help=True)
workflow_commands = typer.Typer(no_args_is_help=True)
console = Console()


@app.callback(invoke_without_command=True, no_args_is_help=True)
def callback():
    """Biomodals CLI - List and get help for biomodals applications.

    This CLI helps users discover available biomodals applications and view their help documentation.
    """
    ...


app.add_typer(app_commands, name="app", help="Discover and run Biomodals apps.")
app.add_typer(
    workflow_commands, name="workflow", help="Discover Biomodals workflow entrypoints."
)


##########################################
# Helper functions
##########################################
def _load_entry(entry_type: CatalogType, name: str) -> BiomodalsApp:
    """Load a biomodals app or workflow by name or path."""
    all_entries = get_catalog(entry_type, use_absolute_paths=True)
    name_or_path = name.partition("::")[0]
    if entry_type == "workflow" and name_or_path not in all_entries:
        workflow_path = Path(name_or_path).expanduser()
        if workflow_path.exists() and not workflow_path.resolve().is_relative_to(
            WORKFLOW_HOME
        ):
            console.print(
                "[bold red]Error[/bold red] Workflow paths must be under "
                f"'[green]{WORKFLOW_HOME}[/green]' so they import through "
                "'[green]biomodals.workflow[/green]'."
            )
            raise typer.Exit(code=1)

    try:
        return BiomodalsApp(name, all_apps=all_entries)
    except AppNotFoundError as e:
        console.print(
            f"[bold red]Error[/bold red] failed to find {entry_type} '{name}': {e}"
        )
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
def _list_available_entries(
    list_type: CatalogType,
    *,
    use_absolute_paths: bool,
    sort_by: Literal["name", "category", "group", "path"],
    reverse: bool,
    short: bool,
) -> dict[str, Path]:
    """Show a list of available biomodals apps or workflows."""
    title = list_type.capitalize()
    table_headers = [f"{title} name", "Category", f"{title} path"]
    available_apps = get_catalog(list_type, use_absolute_paths=use_absolute_paths)
    table_rows: list[tuple[str, str, str]] = []
    for app_name, app_path in available_apps.items():
        app_category = app_path.parent.name
        table_rows.append((f"[green]{app_name}[/green]", app_category, str(app_path)))
    match sort_by:
        case "name":
            sort_by_idx = 0
        case "category" | "group":
            sort_by_idx = 1
        case "path":
            sort_by_idx = 2
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

    if list_type == "app":
        console.print(
            "\n:dna: To see help for an application, use:\n"
            "     [bold]biomodals app help <[green]app-name-or-path[/green]>[/bold]"
        )
        console.print(
            "\n:dna: To run an application on [link=https://modal.com]modal.com[/link], use:\n"
            r"     [bold]biomodals app run <[green]app-name-or-path[/green]>[/bold] -- [gray]\[OPTIONS][/gray]"
        )
        console.print(
            "\n:dna: If an app contains multiple local entrypoints, use it as:\n"
            "     [bold]<[green]app-name-or-path[/green]>::<[green]function-name[/green]>[/bold]\n"
        )
    else:
        console.print(
            "\n:dna: To see help for a workflow, use:\n"
            "     [bold]biomodals workflow help <[green]workflow-name-or-path[/green]>[/bold]"
        )
        console.print(
            "\n:dna: To run a workflow on [link=https://modal.com]modal.com[/link], use:\n"
            r"     [bold]biomodals workflow run <[green]workflow-name-or-path[/green]>[/bold] -- [gray]\[OPTIONS][/gray]"
        )
        console.print(
            "\n:dna: If a workflow contains multiple local entrypoints, use it as:\n"
            "     [bold]<[green]workflow-name-or-path[/green]>::<[green]function-name[/green]>[/bold]\n"
        )
    console.print(f"\n:dna: [bold]Available biomodals {list_type}s:[/bold]")
    console.print(table)
    return available_apps


@app_commands.command(
    name="list",
    help="Show a list of all available biomodals applications (aliases: ls, l).",
)
@app_commands.command(name="ls", hidden=True)
@app_commands.command(name="l", hidden=True)
@app.command(
    name="list", help="Deprecated alias for 'biomodals app list'.", deprecated=True
)
@app.command(name="ls", hidden=True, deprecated=True)
@app.command(name="l", hidden=True, deprecated=True)
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
            "--short", help="Only show app names without paths or additional info."
        ),
    ] = False,
) -> dict[str, Path]:
    """Show a list of all available biomodals applications."""
    return _list_available_entries(
        "app",
        use_absolute_paths=use_absolute_paths,
        sort_by=sort_by,
        reverse=reverse,
        short=short,
    )


@workflow_commands.command(
    name="list",
    help="Show a list of all available biomodals workflows (aliases: ls, l).",
)
@workflow_commands.command(name="ls", hidden=True)
@workflow_commands.command(name="l", hidden=True)
def list_available_workflows(
    use_absolute_paths: Annotated[
        bool,
        typer.Option(
            "--absolute", "-a", help="Use absolute paths for workflow locations."
        ),
    ] = False,
    sort_by: Annotated[
        Literal["name", "category", "group", "path"],
        typer.Option(
            "--sort-by",
            "-s",
            help="Key to sort the workflows by in the table display.",
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
            "--short", help="Only show workflow names without paths or additional info."
        ),
    ] = False,
) -> dict[str, Path]:
    """Show a list of all available biomodals workflows."""
    return _list_available_entries(
        "workflow",
        use_absolute_paths=use_absolute_paths,
        sort_by=sort_by,
        reverse=reverse,
        short=short,
    )


def _show_entry_help(list_type: CatalogType, entry_name: str, *, verbose: bool) -> None:
    """Show help for a specific biomodals app or workflow."""
    catalog_entry = _load_entry(list_type, entry_name)
    if catalog_entry._entrypoint is not None:
        # When an entrypoint name is specified, show only its docstring
        f = catalog_entry[catalog_entry._entrypoint]
        console.print(
            f"[bold]Help for {f.func_type} function"
            f"'[green]{f.name}[/green]'"
            f" in {list_type} '[green]{catalog_entry.name}[/green]'"
            f" ({catalog_entry.category}):[/bold]\n"
        )
        console.print(f.docstring or "No documentation available.")
        if table_rows := f.args_table:
            _print_title("Entrypoint CLI flags")
            console.print(Markdown("\n".join(table_rows)))
        return

    # When no entrypoint is specified, show the app help
    console.print(
        f"[bold]Help for {list_type}"
        f" '[green]{catalog_entry.name}[/green]'"
        f" ({catalog_entry.category}):[/bold]"
    )
    if catalog_entry.module_doc:
        _print_title("Module documentation")
        console.print(Markdown(catalog_entry.module_doc))
    if catalog_entry._remote_modal_func_idx:
        remote_modal_functions = [
            catalog_entry[x] for x in catalog_entry._remote_modal_func_idx
        ]

        _print_title(f"Remote Modal functions in this {list_type}")
        remote_func_names = ", ".join([x.name for x in remote_modal_functions])
        console.print(f"[green]{remote_func_names}[/green]\n")
        if verbose:
            for f in remote_modal_functions:
                if f.docstring:
                    console.print(f"\n[bold green]{f.name}[/bold green]")
                    console.print(Markdown(f.docstring))

    if f_indices := catalog_entry._local_entrypoint_idx:
        _print_title(f"Local entrypoint(s) in this {list_type}")
        for f_idx in f_indices:
            f = catalog_entry[f_idx]

            if f.args_table:
                console.print(f"[bold green]{f.name}[/bold green] CLI flags:\n")
                console.print(Markdown("\n".join(f.args_table)))
            elif f.docstring:
                console.print(f"[bold green]{f.name}[/bold green] documentation:\n")
                console.print(Markdown(f.docstring))


@app_commands.command(
    name="help",
    no_args_is_help=True,
    help="Show help for a specific biomodals application (alias: h).",
)
@app_commands.command(name="h", no_args_is_help=True, hidden=True)
@app.command(
    name="help",
    no_args_is_help=True,
    help="Deprecated alias for 'biomodals app help'.",
    deprecated=True,
)
@app.command(name="h", no_args_is_help=True, hidden=True, deprecated=True)
def show_app_help(
    app_name: Annotated[
        str, typer.Argument(help="Name or path of the app to show help for.")
    ],
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed help for all functions."),
    ] = False,
) -> None:
    """Show help for a specific biomodals application.

    If unsure which app to use, run `biomodals app list` to see available apps.
    If you would like to see help for a local entrypoint or Modal function,
    add `::<function-name>` to the app name to show help for that specific function.
    """
    _show_entry_help("app", app_name, verbose=verbose)


@workflow_commands.command(
    name="help",
    no_args_is_help=True,
    help="Show help for a specific biomodals workflow (alias: h).",
)
@workflow_commands.command(name="h", no_args_is_help=True, hidden=True)
def show_workflow_help(
    workflow_name: Annotated[
        str, typer.Argument(help="Name or path of the workflow to show help for.")
    ],
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed help for all functions."),
    ] = False,
) -> None:
    """Show help for a specific biomodals workflow."""
    _show_entry_help("workflow", workflow_name, verbose=verbose)


@app_commands.command(
    name="run",
    no_args_is_help=True,
    help="Run a biomodals application on Modal (alias: r).",
)
@app_commands.command(name="r", no_args_is_help=True, hidden=True)
@app.command(
    name="run",
    no_args_is_help=True,
    help="Deprecated alias for 'biomodals app run'.",
    deprecated=True,
)
@app.command(name="r", no_args_is_help=True, hidden=True, deprecated=True)
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
    import os
    import sys

    app = _load_entry("app", app_name_or_path)

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
        return

    # TODO: figure out a way to tag run names into the app.
    # Previously we used the MODAL_APP environment variable for ephemeral
    # apps run with the --run-name flag, but with the new AppConfig API
    # this is no longer read.
    env = os.environ.copy()
    if gpu is not None:
        env["GPU"] = gpu
    if timeout is not None:
        env["TIMEOUT"] = str(timeout)

    if flags:
        run_command([*cmd, *flags], env=env, output_mode="inherit")
    elif app._entrypoint is not None:
        run_command(cmd, env=env, output_mode="inherit")
    else:
        _show_entry_help("app", str(app.path), verbose=False)


def _resolve_workflow_entrypoint(workflow: BiomodalsApp) -> str:
    """Return the explicit or only local workflow entrypoint."""
    if workflow._entrypoint is not None:
        return workflow._entrypoint

    local_entrypoints = [
        workflow[entrypoint_idx] for entrypoint_idx in workflow._local_entrypoint_idx
    ]
    if len(local_entrypoints) == 1:
        return local_entrypoints[0].name

    if len(local_entrypoints) > 1:
        entrypoint_names = ", ".join(
            f"[green]{workflow.name}::{entrypoint.name}[/green]"
            for entrypoint in local_entrypoints
        )
        console.print(
            "[bold red]Error[/bold red] Workflow "
            f"'[green]{workflow.name}[/green]' contains multiple local entrypoints; "
            f"choose one explicitly: {entrypoint_names}"
        )
        raise typer.Exit(code=1)

    console.print(
        "[bold red]Error[/bold red] Workflow "
        f"'[green]{workflow.name}[/green]' does not define a local entrypoint."
    )
    raise typer.Exit(code=1)


@workflow_commands.command(
    name="run",
    no_args_is_help=True,
    help="Run a biomodals workflow on Modal (alias: r).",
)
@workflow_commands.command(name="r", no_args_is_help=True, hidden=True)
def run_workflow(
    workflow_name_or_path: Annotated[
        str, typer.Argument(help="Name or path of the workflow to run.")
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
            help="Timeout in seconds for the modal run. If not specified, use the workflow default.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Build the workflow and print its DAG graph without submitting it.",
        ),
    ] = False,
    flags: Annotated[
        list[str] | None,
        typer.Argument(help="Additional flags to pass to the workflow entrypoint."),
    ] = None,
):
    """Run a biomodals workflow on Modal.

    Use with: `biomodals workflow run <workflow-name> [OPTIONS] -- [workflow-options]`,
    where `[workflow-options]` are passed to the workflow local entrypoint.
    """
    import os
    import sys

    workflow = _load_entry("workflow", workflow_name_or_path)
    entrypoint = _resolve_workflow_entrypoint(workflow)
    full_workflow = f"{workflow.module}::{entrypoint}"

    cmd = [sys.executable, "-m", "modal", modal_mode]
    if detach:
        cmd.append("-d")
    cmd.extend(["-m", full_workflow])

    if modal_mode == "shell":
        console.print(
            "To start an interactive shell for the workflow, run:\n"
            f"[bold green]{shlex.join(cmd)}[/bold green]"
        )
        return

    env = os.environ.copy()
    if gpu is not None:
        env["GPU"] = gpu
    if timeout is not None:
        env["TIMEOUT"] = str(timeout)

    entrypoint_flags = list(flags or [])
    if dry_run and "--dry-run" not in entrypoint_flags:
        entrypoint_flags.insert(0, "--dry-run")

    run_command([*cmd, *entrypoint_flags], env=env, output_mode="inherit")


@app_commands.command(
    name="deploy",
    no_args_is_help=True,
    help="Deploy a biomodals application to Modal (alias: d).",
)
@app_commands.command(name="d", no_args_is_help=True, hidden=True)
@app.command(
    name="deploy",
    no_args_is_help=True,
    help="Deprecated alias for 'biomodals app deploy'.",
    deprecated=True,
)
@app.command(name="d", no_args_is_help=True, hidden=True, deprecated=True)
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
    app = _load_entry("app", app_name_or_path)
    cmd = ["modal", "deploy"]
    if name:
        cmd.extend(["--name", name])
    if tag:
        cmd.extend(["--tag", tag])
    cmd.append(str(app.path))
    run_command(cmd, output_mode="inherit")


if __name__ == "__main__":
    app()
