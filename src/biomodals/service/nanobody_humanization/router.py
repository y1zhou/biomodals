"""Authenticated local preparation before explicit scientific Job submission."""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from biomodals.service.auth import AuthenticatedSession
from biomodals.service.http_contract import (
    CodedAPIError,
    PrivateResultRoute,
    require_session,
    require_unsafe_session,
)
from biomodals.service.nanobody_humanization.contracts import (
    NanobodyOptions,
    NanobodyPreparation,
    NanobodyPreparationRequest,
)
from biomodals.workflow.nanobody_humanization.preparation import prepare_batch


def create_router(*, max_parents: int = 100) -> APIRouter:
    """Reuse the bounded analysis executor without another local job lifecycle."""
    if not 1 <= max_parents <= 200:
        raise ValueError("Nanobody max_parents must be between 1 and 200")
    router = APIRouter(
        prefix="/api/v1/nanobody-humanization",
        tags=["nanobody-humanization"],
        route_class=PrivateResultRoute,
    )

    @router.get("/options", response_model=NanobodyOptions)
    async def options(
        session: Annotated[AuthenticatedSession, Depends(require_session)],
    ) -> NanobodyOptions:
        return NanobodyOptions(max_parents=max_parents)

    @router.post("/prepare", response_model=NanobodyPreparation)
    async def prepare(
        body: NanobodyPreparationRequest,
        request: Request,
        session: Annotated[AuthenticatedSession, Depends(require_unsafe_session)],
    ) -> NanobodyPreparation:
        if len(body.parents) > max_parents:
            raise CodedAPIError(
                422,
                "batch_too_large",
                f"At most {max_parents} parents are allowed per job",
            )
        parents, issues = await asyncio.get_running_loop().run_in_executor(
            request.app.state.antibody_analysis.pool, prepare_batch, body.parents
        )
        return NanobodyPreparation.from_prepared(body.parents, parents, issues)

    return router
