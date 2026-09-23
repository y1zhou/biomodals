"""Authenticated ephemeral antibody analysis; no admission or persistence."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from biomodals.helper.antibody import SequenceDetail, sequence_detail
from biomodals.service.antibody_sequence_analysis.analysis import AnalysisService
from biomodals.service.antibody_sequence_analysis.contracts import (
    AnalysisOptions,
    AnalysisRequest,
    AnalysisResponse,
    SequenceRequest,
)
from biomodals.service.auth import AuthenticatedSession
from biomodals.service.http_contract import (
    CodedAPIError,
    CodedErrorResponse,
    PrivateResultRoute,
    require_session,
    require_unsafe_session,
)


def create_router(service: AnalysisService) -> APIRouter:
    """Attach local operations without a ToolRegistration or remote preflight."""
    router = APIRouter(
        prefix="/api/v1/antibody-sequence-analysis",
        tags=["antibody-sequence-analysis"],
        route_class=PrivateResultRoute,
    )

    @router.get("/options", response_model=AnalysisOptions)
    async def options(
        session: Annotated[AuthenticatedSession, Depends(require_session)],
    ) -> AnalysisOptions:
        return AnalysisOptions()

    @router.post(
        "/analyze",
        response_model=AnalysisResponse,
        responses={503: {"model": CodedErrorResponse}},
    )
    async def analyze(
        body: AnalysisRequest,
        session: Annotated[AuthenticatedSession, Depends(require_unsafe_session)],
    ) -> AnalysisResponse:
        if len({group.id for group in body.groups}) != len(body.groups):
            raise CodedAPIError(
                422, "duplicate_group_id", "Use distinct identifiers for the two groups"
            )
        return await service.analyze(body)

    @router.post(
        "/sequence",
        response_model=SequenceDetail,
        responses={503: {"model": CodedErrorResponse}},
    )
    async def sequence(
        body: SequenceRequest,
        session: Annotated[AuthenticatedSession, Depends(require_unsafe_session)],
    ) -> SequenceDetail:
        try:
            return await service.run(
                sequence_detail,
                body.sequence,
                body.scheme,
                body.parental_sequence,
            )
        except ValueError as error:
            raise CodedAPIError(422, "invalid_sequence", str(error)) from error

    return router
