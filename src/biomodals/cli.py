"""Helper script for constructing actual modal run commands."""

import importlib
import shlex
import subprocess
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID, uuid4

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from biomodals.execution import DeploymentIdentity, ExecutionSnapshot
from biomodals.execution.modal import deployed_execution_coordinator
from biomodals.helper.catalog import (
    WORKFLOW_HOME,
    AppNotFoundError,
    BiomodalsApp,
    CatalogType,
    get_catalog,
)
from biomodals.helper.cli_command import (
    build_app_run_command,
    build_modal_app_history_command,
    build_modal_deploy_command,
    build_workflow_run_command,
    modal_env_overrides,
    resolve_workflow_entrypoint,
    select_modal_deployment_version,
)
from biomodals.helper.cli_entrypoint import invoke_local_entrypoint
from biomodals.helper.shell import run_command
from biomodals.service.admin import app as admin_commands
from biomodals.service.config import AdminSettings
from biomodals.service.store import ServiceStore

# ruff: noqa: S603

app = typer.Typer()
app_commands = typer.Typer(no_args_is_help=True)
workflow_commands = typer.Typer(no_args_is_help=True)
run_commands = typer.Typer(no_args_is_help=True)
api_commands = typer.Typer(no_args_is_help=True)
console = Console()


@app.callback(invoke_without_command=True, no_args_is_help=True)
def callback():
    """Discover, run, and administer BioModals apps and workflows.

    Use the command groups to run Modal compute, control durable runs, or
    operate the optional API service.
    """
    ...


app.add_typer(app_commands, name="app", help="Discover and run BioModals apps.")
app.add_typer(
    workflow_commands, name="workflow", help="Discover and run BioModals workflows."
)
app.add_typer(
    run_commands,
    name="run",
    help="Inspect and control a durable BioModals Execution Run.",
)
app.add_typer(api_commands, name="api", help="Run and administer the BioModals API.")
api_commands.add_typer(
    admin_commands,
    name="admin",
    help="Manually manage Biomodals API users.",
)


@api_commands.command(name="serve", help="Start the local Biomodals API server.")
def serve_api(
    host: Annotated[
        str,
        typer.Option(help="Address on which to listen."),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option(min=1, max=65535, help="TCP port on which to listen."),
    ] = 4144,
) -> None:
    """Run the deployed-app FastAPI factory with one Uvicorn worker."""
    try:
        import uvicorn
    except ImportError as exc:
        console.print(
            "[bold red]Error[/bold red] API dependencies are not installed. "
            "Run '[green]uv sync --extra api[/green]'."
        )
        raise typer.Exit(code=1) from exc

    uvicorn.run(
        "biomodals.service.api:create_deployed_app",
        factory=True,
        host=host,
        port=port,
        workers=1,
    )


@api_commands.command(
    name="transition-execution-state",
    help="Replace pre-release Job execution state with the current schema.",
)
def transition_execution_state(
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            help="Confirm deletion of legacy Job history and local execution state.",
        ),
    ] = False,
) -> None:
    """Preserve accounts and settings while discarding legacy Job history."""
    if not yes:
        console.print(
            "[bold red]Error[/bold red] This discards legacy Job history. "
            "Re-run with '[green]--yes[/green]' after stopping the API service."
        )
        raise typer.Exit(code=1)
    store = ServiceStore(AdminSettings.from_environment().database_path)
    try:
        discarded_jobs = store.transition_execution_state()
    except (OSError, RuntimeError, ValueError) as exc:
        console.print(f"[bold red]Error[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    typer.echo(
        "Transitioned service execution state; "
        f"discarded {discarded_jobs} legacy Job(s)."
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
        typer.Option(
            "--detach",
            "-d",
            help="Detach the source-backed development Modal command.",
        ),
    ] = False,
    gpu: Annotated[
        str | None,
        typer.Option(
            "--gpu",
            help="GPU type for a source-backed development run (e.g. 'L40S').",
        ),
    ] = None,
    timeout: Annotated[
        int | None,
        typer.Option(
            "--timeout",
            help="Timeout in seconds for a source-backed development run.",
        ),
    ] = None,
    development: Annotated[
        bool,
        typer.Option(
            "--development",
            help=("Run an app from current source without cross-command recovery."),
        ),
    ] = False,
    environment: Annotated[
        str,
        typer.Option(
            "--environment",
            "-e",
            help="Modal Environment containing a deployed app coordinator.",
        ),
    ] = "main",
    deployment_name: Annotated[
        str | None,
        typer.Option(
            "--deployment-name",
            help="Modal app name. Defaults to the app's declared name.",
        ),
    ] = None,
    version: Annotated[
        int | None,
        typer.Option(
            "--version",
            min=1,
            help="Exact Modal deployment version. Defaults to the latest deployment.",
        ),
    ] = None,
    restart_from: Annotated[
        UUID | None,
        typer.Option(
            "--restart-from",
            help="Create a Successor Run from this Execution Run UUID.",
        ),
    ] = None,
    flags: Annotated[
        list[str] | None,
        typer.Argument(help="Additional flags to pass to the modal run command."),
    ] = None,
):
    """Run a biomodals application on Modal.

    Use with: `biomodals app run <app-name> [OPTIONS] -- [app-options]`.
    """
    import os

    app = _load_entry("app", app_name_or_path)
    coordinated_entrypoint = _coordinated_app_entrypoint(app)
    if (
        modal_mode != "shell"
        and app._entrypoint is not None
        and coordinated_entrypoint is None
        and not development
    ):
        console.print(
            "[bold red]Error[/bold red] This app entrypoint does not expose a "
            "deployment coordinator. Rerun with --development for explicit "
            "source-backed execution."
        )
        raise typer.Exit(code=1)
    if development and (version is not None or deployment_name is not None):
        console.print(
            "[bold red]Error[/bold red] --version and --deployment-name are "
            "unavailable in source-backed development mode"
        )
        raise typer.Exit(code=1)
    if (
        modal_mode != "shell"
        and not development
        and (detach or gpu is not None or timeout is not None)
    ):
        console.print(
            "[bold red]Error[/bold red] --detach, --gpu, and --timeout are "
            "available only in source-backed development mode"
        )
        raise typer.Exit(code=1)
    if modal_mode != "shell" and coordinated_entrypoint is not None and not development:
        try:
            resolved_deployment_name = deployment_name or _deployment_name(app)
            resolved_version = _resolve_deployment_version(
                deployment_name=resolved_deployment_name,
                environment=environment,
                requested_version=version,
            )
        except (ImportError, OSError, subprocess.CalledProcessError, ValueError) as exc:
            console.print(
                "[bold red]Error[/bold red] Could not resolve exact app "
                f"deployment: {exc}"
            )
            raise typer.Exit(code=1) from exc
    elif restart_from is not None:
        if development:
            message = "--restart-from is unavailable in source-backed development mode"
        else:
            message = "--restart-from requires a coordinator-aware app entrypoint"
        console.print(f"[bold red]Error[/bold red] {message}")
        raise typer.Exit(code=1)
    elif coordinated_entrypoint is None and (
        version is not None or deployment_name is not None
    ):
        console.print(
            "[bold red]Error[/bold red] Deployment coordinator options require "
            "a coordinator-aware app entrypoint"
        )
        raise typer.Exit(code=1)

    if modal_mode == "shell":
        cmd = build_app_run_command(
            app_path=app.path,
            entrypoint=app._entrypoint,
            modal_mode=modal_mode,
            detach=detach,
            flags=flags,
        )
        console.print(
            "To start an interactive shell for the app, run:\n"
            f"[bold green]{shlex.join(cmd)}[/bold green]"
        )
        return

    if coordinated_entrypoint is not None and not development:
        invoke_local_entrypoint(
            module_name=app.module,
            entrypoint_name=coordinated_entrypoint,
            flags=flags or (),
            overrides={
                "use_deployed_coordinator": True,
                "deployment_environment": environment,
                "deployment_name": resolved_deployment_name,
                "deployment_version": resolved_version,
                "restart_from": None if restart_from is None else str(restart_from),
            },
            program_name=(f"biomodals app run {app.name}::{coordinated_entrypoint} --"),
            environment_name=environment,
        )
        return

    cmd = build_app_run_command(
        app_path=app.path,
        entrypoint=app._entrypoint,
        modal_mode=modal_mode,
        detach=detach,
        flags=flags,
    )

    # TODO: figure out a way to tag run names into the app.
    # Previously we used the MODAL_APP environment variable for ephemeral
    # apps run with the --run-name flag, but with the new AppConfig API
    # this is no longer read.
    env = os.environ.copy()
    env.update(modal_env_overrides(gpu=gpu, timeout=timeout))

    if flags:
        run_command(list(cmd), env=env, output_mode="inherit")
    elif app._entrypoint is not None:
        run_command(list(cmd), env=env, output_mode="inherit")
    else:
        _show_entry_help("app", str(app.path), verbose=False)


def _coordinated_app_entrypoint(app: BiomodalsApp) -> str | None:
    """Return the selected entrypoint only when its module opts into the kernel."""
    entrypoint = app._entrypoint
    module_name = getattr(app, "module", None)
    if entrypoint is None or not isinstance(module_name, str):
        return None
    module = importlib.import_module(module_name)
    declared = getattr(module, "EXECUTION_COORDINATOR_ENTRYPOINTS", ())
    if not isinstance(declared, frozenset | set | tuple | list) or not all(
        isinstance(value, str) and value for value in declared
    ):
        raise ValueError(
            f"App '{app.name}' has an invalid coordinator entrypoint declaration"
        )
    return entrypoint if entrypoint in declared else None


def _deployment_name(entry: BiomodalsApp) -> str:
    """Return one app or workflow module's Modal deployment name."""
    module = importlib.import_module(entry.module)
    config = getattr(module, "CONF", None)
    deployment_name = getattr(config, "name", None)
    if not isinstance(deployment_name, str) or not deployment_name:
        raise ValueError(f"'{entry.name}' does not declare a deployment name")
    return deployment_name


def _run_coordinator(
    *,
    environment: str,
    deployment_name: str,
    deployment_version: int,
    execution_run_id: UUID,
):
    """Resolve the standard coordinator for one explicit run location."""
    deployment = DeploymentIdentity(
        environment=environment,
        deployment_name=deployment_name,
        deployment_version=deployment_version,
    )
    return deployed_execution_coordinator(
        execution_run_id=execution_run_id,
        deployment=deployment,
    )


def _print_execution_snapshot(snapshot: ExecutionSnapshot) -> None:
    """Render the common durable execution view."""
    run = snapshot.run
    console.print(f"Execution Run ID: [green]{run.execution_run_id}[/green]")
    console.print(
        "Deployment Identity: "
        f"[green]{run.deployment.environment}/"
        f"{run.deployment.deployment_name}/"
        f"v{run.deployment.deployment_version}[/green]"
    )
    console.print(f"Status: [green]{run.status.value}[/green]")
    if run.status_reason is not None:
        console.print(f"Reason: {run.status_reason.value}")
    if run.status_message:
        console.print(f"Message: {run.status_message}")
    console.print(
        "Active Provider Calls: "
        f"{snapshot.active_provider_calls.total} total, "
        f"{snapshot.active_provider_calls.gpu} GPU"
    )


def _print_spawned_execution_run(
    *,
    execution_run_id: UUID,
    environment: str,
    deployment_name: str,
    deployment_version: int,
    call: object,
) -> None:
    """Print the explicit location of one detached coordinator call."""
    console.print(f"Execution Run ID: [green]{execution_run_id}[/green]")
    console.print(
        "Deployment Identity: "
        f"[green]{environment}/{deployment_name}/v{deployment_version}[/green]"
    )
    console.print(
        "Coordinator FunctionCall ID: "
        f"[green]{getattr(call, 'object_id', call)}[/green]"
    )


@run_commands.command(name="status")
def status_execution_run(
    environment: Annotated[
        str,
        typer.Option("--environment", help="Modal Environment containing the run."),
    ],
    deployment_name: Annotated[
        str,
        typer.Option("--deployment-name", help="Modal app deployment name."),
    ],
    deployment_version: Annotated[
        int,
        typer.Option(
            "--deployment-version",
            min=1,
            help="Exact numeric Modal deployment version.",
        ),
    ],
    execution_run_id: Annotated[
        UUID,
        typer.Option("--execution-run-id", help="Opaque Execution Run UUID."),
    ],
) -> None:
    """Read one persisted Execution Run without advancing it."""
    try:
        snapshot = _run_coordinator(
            environment=environment,
            deployment_name=deployment_name,
            deployment_version=deployment_version,
            execution_run_id=execution_run_id,
        ).status.remote()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]Error[/bold red] Could not read run status: {exc}")
        raise typer.Exit(code=1) from exc
    _print_execution_snapshot(snapshot)


@run_commands.command(name="cancel")
def cancel_execution_run(
    environment: Annotated[
        str,
        typer.Option("--environment", help="Modal Environment containing the run."),
    ],
    deployment_name: Annotated[
        str,
        typer.Option("--deployment-name", help="Modal app deployment name."),
    ],
    deployment_version: Annotated[
        int,
        typer.Option(
            "--deployment-version",
            min=1,
            help="Exact numeric Modal deployment version.",
        ),
    ],
    execution_run_id: Annotated[
        UUID,
        typer.Option("--execution-run-id", help="Opaque Execution Run UUID."),
    ],
) -> None:
    """Request idempotent cancellation of one Execution Run."""
    try:
        snapshot = _run_coordinator(
            environment=environment,
            deployment_name=deployment_name,
            deployment_version=deployment_version,
            execution_run_id=execution_run_id,
        ).cancel.remote()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]Error[/bold red] Could not cancel run: {exc}")
        raise typer.Exit(code=1) from exc
    _print_execution_snapshot(snapshot)


@run_commands.command(name="resume")
def resume_execution_run_command(
    environment: Annotated[
        str,
        typer.Option("--environment", help="Modal Environment containing the run."),
    ],
    deployment_name: Annotated[
        str,
        typer.Option("--deployment-name", help="Modal app deployment name."),
    ],
    deployment_version: Annotated[
        int,
        typer.Option(
            "--deployment-version",
            min=1,
            help="Exact numeric Modal deployment version.",
        ),
    ],
    execution_run_id: Annotated[
        UUID,
        typer.Option("--execution-run-id", help="Opaque Execution Run UUID."),
    ],
) -> None:
    """Resume a suspended or state-unknown Run without retrying failed Tasks."""
    try:
        call = _run_coordinator(
            environment=environment,
            deployment_name=deployment_name,
            deployment_version=deployment_version,
            execution_run_id=execution_run_id,
        ).resume.spawn()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]Error[/bold red] Could not resume run: {exc}")
        raise typer.Exit(code=1) from exc
    _print_spawned_execution_run(
        execution_run_id=execution_run_id,
        environment=environment,
        deployment_name=deployment_name,
        deployment_version=deployment_version,
        call=call,
    )


@run_commands.command(name="restart")
def restart_execution_run(
    environment: Annotated[
        str,
        typer.Option(
            "--environment",
            help="Modal Environment containing the predecessor Run.",
        ),
    ],
    deployment_name: Annotated[
        str,
        typer.Option(
            "--deployment-name",
            help="Predecessor Modal app deployment name.",
        ),
    ],
    deployment_version: Annotated[
        int,
        typer.Option(
            "--deployment-version",
            min=1,
            help="Exact predecessor Modal deployment version.",
        ),
    ],
    execution_run_id: Annotated[
        UUID,
        typer.Option(
            "--execution-run-id",
            help="Opaque predecessor Execution Run UUID.",
        ),
    ],
    target_environment: Annotated[
        str,
        typer.Option(
            "--target-environment",
            help="Modal Environment for the Successor Run.",
        ),
    ],
    target_deployment_name: Annotated[
        str,
        typer.Option(
            "--target-deployment-name",
            help="Modal app deployment name for the Successor Run.",
        ),
    ],
    target_deployment_version: Annotated[
        int,
        typer.Option(
            "--target-deployment-version",
            min=1,
            help="Exact Modal deployment version for the Successor Run.",
        ),
    ],
    max_active_provider_calls: Annotated[
        int | None,
        typer.Option(
            "--max-active-provider-calls",
            min=1,
            help="Override the predecessor's total active-call limit.",
        ),
    ] = None,
    max_active_gpu_provider_calls: Annotated[
        int | None,
        typer.Option(
            "--max-active-gpu-provider-calls",
            min=0,
            help="Override the predecessor's active GPU-call limit.",
        ),
    ] = None,
) -> None:
    """Create a new Successor Run without mutating the predecessor."""
    successor_execution_run_id = uuid4()
    try:
        coordinator = _run_coordinator(
            environment=target_environment,
            deployment_name=target_deployment_name,
            deployment_version=target_deployment_version,
            execution_run_id=successor_execution_run_id,
        )
        coordinator.prepare_restart.remote(
            predecessor_execution_run_id=str(execution_run_id),
            predecessor_deployment_environment=environment,
            predecessor_deployment_name=deployment_name,
            predecessor_deployment_version=deployment_version,
            max_active_provider_calls=max_active_provider_calls,
            max_active_gpu_provider_calls=max_active_gpu_provider_calls,
        )
        call = coordinator.drive_prepared.spawn()
    except KeyboardInterrupt:
        console.print(f"Successor Execution Run ID: {successor_execution_run_id}")
        console.print(
            "Restart submission outcome is unknown; inspect this Run before retrying."
        )
        raise
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]Error[/bold red] Could not restart run: {exc}")
        console.print(f"Successor Execution Run ID: {successor_execution_run_id}")
        console.print(
            "Restart submission outcome is unknown; inspect this Run before retrying."
        )
        raise typer.Exit(code=1) from exc
    _print_spawned_execution_run(
        execution_run_id=successor_execution_run_id,
        environment=target_environment,
        deployment_name=target_deployment_name,
        deployment_version=target_deployment_version,
        call=call,
    )


def _resolve_workflow_entrypoint(workflow: BiomodalsApp) -> str:
    """Return the explicit or only local workflow entrypoint."""
    local_entrypoints = [
        workflow[entrypoint_idx].name
        for entrypoint_idx in workflow._local_entrypoint_idx
    ]
    try:
        return resolve_workflow_entrypoint(
            workflow_name=workflow.name,
            explicit_entrypoint=workflow._entrypoint,
            local_entrypoints=local_entrypoints,
        )
    except ValueError as exc:
        console.print(f"[bold red]Error[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


def _resolve_deployment_version(
    *,
    deployment_name: str,
    environment: str,
    requested_version: int | None,
) -> int:
    """Preflight one exact deployment through Modal history."""
    command = build_modal_app_history_command(
        deployment_name=deployment_name,
        environment=environment,
    )
    lines = run_command(
        list(command),
        output_mode="capture",
        show_command=False,
    )
    return select_modal_deployment_version(
        "\n".join(lines),
        requested_version=requested_version,
    )


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
        typer.Option(
            "--detach",
            "-d",
            help="Detach the source-backed development Modal command.",
        ),
    ] = False,
    gpu: Annotated[
        str | None,
        typer.Option(
            "--gpu",
            help="GPU type for a source-backed development run (e.g. 'L40S').",
        ),
    ] = None,
    timeout: Annotated[
        int | None,
        typer.Option(
            "--timeout",
            help="Timeout in seconds for a source-backed development run.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Build the workflow and print its DAG graph without submitting it.",
        ),
    ] = False,
    development: Annotated[
        bool,
        typer.Option(
            "--development",
            help=(
                "Run against current source without durable cross-command "
                "coordinator lookup."
            ),
        ),
    ] = False,
    environment: Annotated[
        str,
        typer.Option(
            "--environment",
            "-e",
            help="Modal Environment containing the deployed workflow.",
        ),
    ] = "main",
    deployment_name: Annotated[
        str | None,
        typer.Option(
            "--deployment-name",
            help="Modal app name. Defaults to the workflow's declared name.",
        ),
    ] = None,
    version: Annotated[
        int | None,
        typer.Option(
            "--version",
            min=1,
            help="Exact Modal deployment version. Defaults to the latest deployment.",
        ),
    ] = None,
    restart_from: Annotated[
        UUID | None,
        typer.Option(
            "--restart-from",
            help="Create a Successor Run from this Execution Run UUID.",
        ),
    ] = None,
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

    workflow = _load_entry("workflow", workflow_name_or_path)
    entrypoint = _resolve_workflow_entrypoint(workflow)
    if (
        modal_mode != "shell"
        and not development
        and (detach or gpu is not None or timeout is not None)
    ):
        console.print(
            "[bold red]Error[/bold red] --detach, --gpu, and --timeout are "
            "available only in source-backed development mode"
        )
        raise typer.Exit(code=1)
    if restart_from is not None:
        if development:
            message = "--restart-from is unavailable in source-backed development mode"
        elif dry_run:
            message = "--restart-from is unavailable for a local dry run"
        elif modal_mode == "shell":
            message = "--restart-from is unavailable for an interactive shell"
        else:
            message = None
        if message is not None:
            console.print(f"[bold red]Error[/bold red] {message}")
            raise typer.Exit(code=1)
    if modal_mode != "shell" and not dry_run and not development:
        try:
            resolved_deployment_name = deployment_name or _deployment_name(workflow)
            resolved_version = _resolve_deployment_version(
                deployment_name=resolved_deployment_name,
                environment=environment,
                requested_version=version,
            )
        except (ImportError, OSError, subprocess.CalledProcessError, ValueError) as exc:
            console.print(
                "[bold red]Error[/bold red] Could not resolve exact workflow "
                f"deployment: {exc}"
            )
            raise typer.Exit(code=1) from exc

    if modal_mode == "shell":
        cmd = build_workflow_run_command(
            workflow_module=workflow.module,
            entrypoint=entrypoint,
            modal_mode=modal_mode,
            detach=detach,
            dry_run=dry_run,
            flags=flags,
        )
        console.print(
            "To start an interactive shell for the workflow, run:\n"
            f"[bold green]{shlex.join(cmd)}[/bold green]"
        )
        return

    if not development:
        deployed = not dry_run
        invoke_local_entrypoint(
            module_name=workflow.module,
            entrypoint_name=entrypoint,
            flags=flags or (),
            overrides={
                "dry_run": dry_run,
                "use_deployed_coordinator": deployed,
                "deployment_environment": environment if deployed else "development",
                "deployment_name": resolved_deployment_name if deployed else None,
                "deployment_version": resolved_version if deployed else 1,
                "restart_from": None if restart_from is None else str(restart_from),
            },
            program_name=f"biomodals workflow run {workflow.name}::{entrypoint} --",
            environment_name=environment if deployed else None,
        )
        return

    cmd = build_workflow_run_command(
        workflow_module=workflow.module,
        entrypoint=entrypoint,
        modal_mode=modal_mode,
        detach=detach,
        dry_run=dry_run,
        flags=flags,
    )

    env = os.environ.copy()
    env.update(modal_env_overrides(gpu=gpu, timeout=timeout))

    run_command(list(cmd), env=env, output_mode="inherit")


@app_commands.command(
    name="deploy",
    no_args_is_help=True,
    help="Deploy a biomodals application to Modal (alias: d).",
)
@app_commands.command(name="d", no_args_is_help=True, hidden=True)
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
    env: Annotated[
        str | None,
        typer.Option("--env", "-e", help="Modal Environment to deploy into."),
    ] = None,
    strategy: Annotated[
        Literal["rolling", "recreate"] | None,
        typer.Option("--strategy", help="Deployment strategy."),
    ] = None,
):
    """Deploy a biomodals application to Modal."""
    app = _load_entry("app", app_name_or_path)
    cmd = build_modal_deploy_command(
        app_path=app.path,
        name=name,
        tag=tag,
        env=env,
        strategy=strategy,
    )
    run_command(list(cmd), output_mode="inherit")


if __name__ == "__main__":
    app()
