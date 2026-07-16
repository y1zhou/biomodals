"""Administrator commands for the department-hosted API service."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, NoReturn

import typer

from biomodals.service.config import ServiceSettings
from biomodals.service.store import ServiceStore

if TYPE_CHECKING:
    from biomodals.service.auth import AuthService

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Manually manage Biomodals API users.",
)


def _auth_service() -> AuthService:
    from biomodals.service.auth import AuthService

    settings = ServiceSettings.from_environment()
    store = ServiceStore(settings.database_path)
    store.initialize()
    return AuthService(store, frontend_url=settings.frontend_url)


def _fail(exc: Exception) -> NoReturn:
    typer.echo(f"Error: {exc}", err=True)
    raise typer.Exit(code=1) from exc


@app.command("create-user")
def create_user(
    email: Annotated[str, typer.Argument(help="Company email address.")],
    display_name: Annotated[
        str,
        typer.Option("--display-name", help="Name shown in the web application."),
    ],
) -> None:
    """Create a user and print their one-time password setup link."""
    try:
        link = _auth_service().create_user(email, display_name=display_name)
    except (LookupError, ValueError) as exc:
        _fail(exc)
    typer.echo(link)


@app.command("reset-password")
def reset_password(
    email: Annotated[str, typer.Argument(help="Company email address.")],
) -> None:
    """Print a new one-time password reset link for an active user."""
    try:
        link = _auth_service().create_password_reset(email)
    except (LookupError, ValueError) as exc:
        _fail(exc)
    typer.echo(link)


@app.command("disable-user")
def disable_user(
    email: Annotated[str, typer.Argument(help="Company email address.")],
) -> None:
    """Disable a user and revoke their sessions and password links."""
    try:
        principal = _auth_service().disable_user(email)
    except (LookupError, ValueError) as exc:
        _fail(exc)
    typer.echo(f"Disabled {principal.email}")
