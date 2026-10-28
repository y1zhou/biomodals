"""Public capabilities, liveness and readiness for the Biomodals API process."""

from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from biomodals.service.artifacts import ArtifactCache
from biomodals.service.http_contract import ErrorResponse
from biomodals.service.store import ServiceStore


class HealthView(BaseModel):
    """Minimal local service probe."""

    status: Literal["ok"]


class CapabilitiesView(BaseModel):
    """Public startup capabilities; no provider credentials or Tool metadata."""

    remote_execution_enabled: bool


def create_operations_router(
    *,
    store: ServiceStore,
    cache: ArtifactCache | None,
) -> APIRouter:
    """Create public service routes around explicit readiness dependencies."""
    router = APIRouter(prefix="/api/v1", tags=["operations"])

    @router.get("/capabilities", response_model=CapabilitiesView)
    async def capabilities(request: Request, response: Response) -> CapabilitiesView:
        response.headers["Cache-Control"] = "no-store"
        return CapabilitiesView(
            remote_execution_enabled=(
                request.app.state.configuration.settings.remote_execution_enabled
            )
        )

    @router.get("/health", response_model=HealthView)
    async def health() -> HealthView:
        return HealthView(status="ok")

    @router.get(
        "/ready",
        response_model=HealthView,
        responses={503: {"model": ErrorResponse}},
    )
    async def ready(request: Request) -> HealthView:
        task: asyncio.Task[None] | None = request.app.state.reconciler_task
        try:
            if not request.app.state.ready or (task is not None and task.done()):
                raise RuntimeError("Background service is unavailable")
            store.check_ready()
            if cache is not None:
                await cache.check_ready_async()
        except Exception as exc:
            raise HTTPException(503, "Service is not ready") from exc
        return HealthView(status="ok")

    return router
